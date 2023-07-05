import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
MAX_SEQ_LEN = 256


class MovingAverage:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.avg = None

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if self.avg is None:
            self.avg = value
        else:
            self.avg = self.beta * self.avg + (1 - self.beta) * value

    def get(self):
        return self.avg


class TextProcessor:

    def __init__(self):
        import nltk
        self.sent_tokenize = nltk.sent_tokenize
        nltk.download('punkt')

        self.CLEAN_HTML = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        self.CLEAN_PUNKT = re.compile('[' + re.escape('!"#$%&()*+,.:;<=>?@[\\]^_`{|}~') + ']')
        self.CLEAN_WHITE = re.compile(r'\s+')

    def clean_text(self, text):
        text = re.sub(self.CLEAN_HTML, ' ', text)
        text = re.sub(self.CLEAN_PUNKT, ' ', text.lower())
        text = re.sub(self.CLEAN_WHITE, ' ', text)
        return text.strip()

    def __call__(self, text):
        return [self.clean_text(sent) for sent in self.sent_tokenize(text)]


class IMDBDataset(torch.utils.data.Dataset):

    def __init__(self, train=True):
        super().__init__()
        split = "unsupervised" if train else "test"
        raw_dataset = load_dataset("imdb", split=split)
        self.tokens = tokenize_dataset(raw_dataset, f"imdb_{split}", "text")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]


def load_imdb_dataset(train=True):
    return IMDBDataset(train)


def add_special_tokens(tokens_a, token_b=None):
    tokens = torch.cat([
        torch.tensor([TOKENIZER.cls_token_id]),
        tokens_a,
        torch.tensor([TOKENIZER.sep_token_id])
    ])
    normal_mask = torch.tensor([False] + [True] * len(tokens_a) + [False])
    segment_id = torch.zeros_like(tokens)
    if token_b is not None:
        tokens = torch.cat([
            tokens,
            token_b,
            torch.tensor([TOKENIZER.sep_token_id])
        ])
        normal_mask = torch.cat([
            normal_mask,
            torch.tensor([True] * len(token_b) + [False])
        ])
        segment_id = torch.cat([
            segment_id,
            torch.ones(len(token_b) + 1, dtype=torch.int16)
        ])
    return dict(
        input_ids=tokens.long(),
        normal_mask=normal_mask,
        segment_ids=segment_id.long())


class SST2Dataset(torch.utils.data.Dataset):

    def __init__(self, train=True):
        super().__init__()
        split = "train" if train else "validation"
        self.raw_dataset = load_dataset("glue", "sst2", split=split)
        self.tokens = tokenize_dataset(self.raw_dataset, f"sst2_{split}", "sentence")

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        label = self.raw_dataset[index]['label']
        tokens = self.tokens[index][0]
        out = add_special_tokens(tokens)
        out.update(labels=label)
        return out


def load_sst2_dataset(train=True):
    return SST2Dataset(train)


def tokenize_dataset(dataset, name, key):
    path = Path('.cache')
    path.mkdir(exist_ok=True)
    path = path / f"{name}_tokens.pt"
    if path.exists():
        return torch.load(path)
    print(f"Tokenizing {name} dataset...")
    text_processor = TextProcessor()
    out = []
    for i in tqdm(range(len(dataset))):
        text = dataset[i][key]
        # If the item is a tuple, it is a labeled dataset
        if isinstance(text, tuple):
            text = text
        text = text_processor(text)
        ids = TOKENIZER(text,
                        max_length=MAX_SEQ_LEN,
                        add_special_tokens=False,
                        truncation=True)['input_ids']
        ids = [torch.tensor(ids, dtype=torch.int16, device='cpu') for ids in ids]
        out.append(ids)
    torch.save(out, path)
    return out

############################################################################################################
# This is copied from the PyTorch source code and only modified to allow padding

import collections
import contextlib
import re
import torch

from typing import Callable, Dict, Optional, Tuple, Type, Union

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def default_convert(data):
    r"""
        Function that converts each NumPy array element into a :class:`torch.Tensor`. If the input is a `Sequence`,
        `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
        If the input is not an NumPy array, it is left unchanged.
        This is used as the default function for collation when both `batch_sampler` and
        `batch_size` are NOT defined in :class:`~torch.utils.data.DataLoader`.
        The general input type to output type mapping is similar to that
        of :func:`~torch.utils.data.default_collate`. See the description there for more details.
        Args:
            data: a single data point to be converted
        Examples:
            >>> # xdoctest: +SKIP
            >>> # Example with `int`
            >>> default_convert(0)
            0
            >>> # Example with NumPy array
            >>> default_convert(np.array([0, 1]))
            tensor([0, 1])
            >>> # Example with NamedTuple
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_convert(Point(0, 0))
            Point(x=0, y=0)
            >>> default_convert(Point(np.array(0), np.array(0)))
            Point(x=tensor(0), y=tensor(0))
            >>> # Example with List
            >>> default_convert([np.array([0, 1]), np.array([2, 3])])
            [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        try:
            return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, str):
        try:
            return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    r"""
        General collate function that handles collection type of element within each batch
        and opens function registry to deal with specific element types. `default_collate_fn_map`
        provides default collate functions for tensors, numpy arrays, numbers and strings.
        Args:
            batch: a single batch to be collated
            collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
              If the element type isn't present in this dictionary,
              this function will go through each key of the dictionary in the insertion order to
              invoke the corresponding collate function if the element type is a subclass of the key.
        Examples:
            >>> # Extend this function to handle batch of tensors
            >>> def collate_tensor_fn(batch, *, collate_fn_map):
            ...     return torch.stack(batch, 0)
            >>> def custom_collate(batch):
            ...     collate_map = {torch.Tensor: collate_tensor_fn}
            ...     return collate(batch, collate_fn_map=collate_map)
            >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
            >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})
        Note:
            Each collate function requires a positional argument for batch and a keyword argument
            for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


# This function is new
def padded_stack(tensors, pad_length, dim=0, *, out=None):
    padded_tensors = []
    for tensor in tensors:
        padding = torch.zeros(pad_length - tensor.size(0), *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, padding], dim=0)
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors, dim=dim, out=out)


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    max_length = max(t.size(0) for t in batch)
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = elem[0].numel() * max_length * len(batch)
        storage = elem.storage()._new_shared(numel, device=elem.device)
        shape = [len(batch), max_length] + list(elem.shape[1:])
        out = elem.new(storage).resize_(shape)
    return padded_stack(batch, pad_length=max_length, dim=0, out=out)


def collate_numpy_array_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.as_tensor(batch)


def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch, dtype=torch.float64)


def collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return torch.tensor(batch)


def collate_str_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}
with contextlib.suppress(ImportError):
    import numpy as np
    # For both ndarray and memmap (subclass of ndarray)
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn


def padded_collate(batch):
    """Like torch.utils.data.dataloader.default_collate, but pads the data to the maximum length.
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)

class SST2Model(torch.nn.Module):

    def __init__(self, bert_encoder, train_encoder=True):
        """
        Args:
            bert_encoder: An instance of a BERTEncoder
            train_encoder: wheter the encoder should be trained or not.
        """
        super().__init__()

        self.bert_encoder = bert_encoder
        for param in self.bert_encoder.parameters():
            param.requires_grad = train_encoder
        self.classifier = torch.nn.Linear(bert_encoder.d_model, 1, bias=False)

    def forward(self, input_ids):
        """
        Predicts the sentiment of a sentence (positive or negative)
        Args:
            input_ids: tensor of shape (batch_size, seq_len) containing the token ids of the sentences
        Returns:
            tensor of shape (batch_size) containing the predicted sentiment
        """
        h = self.bert_encoder(input_ids)
        return self.classifier(h[:, 0]).view(-1)

def train_sst2(bert_encoder, train_encoder=False, epochs=3, batch_size=256, lr=1e-3, device='cuda'):
    sst2_dataset = load_sst2_dataset(train=True)
    loader = DataLoader(sst2_dataset, batch_size=batch_size, shuffle=True, collate_fn=padded_collate, num_workers=4)
    sst2_model = SST2Model(bert_encoder, train_encoder=train_encoder).to(device).train()
    opt = torch.optim.AdamW(sst2_model.classifier.parameters(), lr=lr)
    loss_avg = MovingAverage()
    acc_avg = MovingAverage()
    for ep in range(epochs):
        with tqdm(loader, desc=f'Epoch {ep}') as pbar:
            for batch in pbar:
                opt.zero_grad(set_to_none=True)
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].float().to(device)
                logits = sst2_model(input_ids)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                opt.step()
                loss_avg.update(loss)
                acc_avg.update(((logits > 0) == labels).float().mean())
                pbar.set_postfix(
                    loss=loss_avg.get(),
                    acc=acc_avg.get()
                )
    return sst2_model

@torch.no_grad()
def validate_sst2(model, device):
    model.eval()
    dataset = load_sst2_dataset(train=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=padded_collate)
    accs = []
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids)
        pred = logits > 0
        accs.append((pred == labels).float())
    return torch.cat(accs).mean().item() * 100