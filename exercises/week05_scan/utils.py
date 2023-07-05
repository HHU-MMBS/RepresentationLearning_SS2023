
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import random
import matplotlib.pyplot as plt
from torchvision import transforms as T
from tqdm import tqdm
from sklearn.manifold import TSNE

def load_data( batch_size=128, train_split="unlabeled", test_split="test", transf = T.ToTensor(), num_workers=2, shuffle=False):
    train_ds = torchvision.datasets.STL10(root='../data', split=train_split, transform=transf, download=True)
    val_ds = torchvision.datasets.STL10(root='../data', split=test_split, transform=transf, download=True)
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=shuffle)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=shuffle)
    return train_dl, val_dl

def imshow(img, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
    """
    shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
    """
    # undoes the normalization
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def prevalidate(model, val_loader,criterion, device):
    ### START CODE HERE ### (≈ 12 lines of code)
    model.eval()
    correct, total = 0, 0
    loss_step = []
    with torch.no_grad():
        for data in val_loader:
            inp_data,labels = data
            inp_data, labels = inp_data.to(device), labels.to(device)
            outputs = model(inp_data)
            val_loss = criterion(outputs, labels)
            loss_step.append(val_loss.item())
        # dont forget to take the means here
        val_loss_epoch = np.mean(loss_step)
    ### END CODE HERE ###
        return val_loss_epoch


def save_model(model, path, epoch, optimizer, val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }, path)


def validate(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    loss_step = []
    with torch.no_grad():
        for data in val_loader:
            inp_data,labels = data
            inp_data = inp_data.to(device)
            labels = labels.to(device)
            outputs = model(inp_data)
            val_loss = criterion(outputs, labels)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_step.append(val_loss.item())
        # dont forget to take the means here
        val_acc = (100 * correct / total).cpu().numpy() 
        val_loss_epoch = torch.tensor(loss_step).mean().numpy()
        return val_acc , val_loss_epoch

# Provided
@torch.no_grad()
def get_features(model, dataloader, device):
    model = model.to(device)
    feats, labs = [], []
    for i in dataloader:
        inp_data,labels = i
        inp_data = inp_data.to(device)
        features = model(inp_data)
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        features = features.cpu().detach()
        labels = labels.cpu().detach()
        feats.append(features)
        labs.append(labels)
    f = torch.cat(feats, dim=0)
    l = torch.cat(labs, dim=0)
    return f,l


def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_step = []
    correct, total = 0, 0
    for data in train_loader:
        # Move the data to the GPU
        inp_data,labels = data
        inp_data = inp_data.to(device)
        labels = labels.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_step.append(loss.item())
    # dont forget the means here
    loss_curr_epoch = np.mean(loss_step)
    train_acc = (100 * correct / total).cpu()
    return loss_curr_epoch, train_acc


def linear_eval(model, optimizer, num_epochs, train_loader, val_loader, device):
    best_val_loss = 1e8
    best_val_acc = 0
    model = model.to(device)
    dict_log = {"train_acc_epoch":[], "val_acc_epoch":[], "loss_epoch":[], "val_loss":[]}
    train_acc, _ = validate(model, train_loader, device)
    val_acc, _ = validate(model, val_loader, device)
    print(f'Init Accuracy of the model: Train:{train_acc:.3f} \t Val:{val_acc:3f}')
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss_curr_epoch, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        val_acc, val_loss = validate(model, val_loader, device)

        # Print epoch results to screen 
        msg = (f'Ep {epoch}/{num_epochs}: Accuracy : Train:{train_acc:.2f} \t Val:{val_acc:.2f} || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
        pbar.set_description(msg)
        # Track stats
        dict_log["train_acc_epoch"].append(train_acc)
        dict_log["val_acc_epoch"].append(val_acc)
        dict_log["loss_epoch"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  }, f'best_model_min_val_loss.pth')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  }, f'best_model_max_val_acc.pth')
    return dict_log


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model


def default(val, def_val):
    return def_val if val is None else val

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


def tsne_plot_embeddings(features, labels, class_names, title="T-SNE plot"):
    plt.figure(figsize=(12, 12))
    latent_space_tsne = TSNE(2, verbose = True, n_iter = 2000, metric="cosine", perplexity=50, learning_rate=500)
    xa_tsne = latent_space_tsne.fit_transform(features.cpu().numpy()[:, :])
    colors = plt.rcParams["axes.prop_cycle"]()  
    for class_idx in range(len(class_names)):
        c = next(colors)["color"]
        plt.scatter(xa_tsne[:,0][labels==class_idx], xa_tsne[:,1][labels==class_idx], color=c, label=class_names[class_idx])

    plt.legend(class_names, fontsize=18, loc='center left', bbox_to_anchor=(1.05, 0.5))
    if title is not None:
        plt.title(title, fontsize=18)

    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.savefig("tsne_plot_embeddings_solution.png")
    plt.show()

# based on https://github.com/elad-amrani/self-classifier/blob/e5e3fb98d71bd6961031bbd308826017fd9753ec/src/cls_eval.py
def compute_clustering_metrics(targets, preds, min_samples_per_class, verbose=False):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from sklearn.metrics import adjusted_mutual_info_score as adjusted_nmi
    from sklearn.metrics import adjusted_rand_score as adjusted_rand_index
    from scipy.optimize import linear_sum_assignment
    val_nmi = nmi(targets, preds)
    val_adjusted_nmi = adjusted_nmi(targets, preds)
    val_adjusted_rand_index = adjusted_rand_index(targets, preds)

    # compute accuracy
    num_classes = max(targets.max(), preds.max()) + 1
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for ii in range(preds.shape[0]):
        count_matrix[preds[ii], targets[ii]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]

    if len(np.unique(preds)) > len(np.unique(targets)):  # if using over-clustering, append remaining clusters to best option
        for cls_idx in np.unique(preds):
            if reassignment[cls_idx, 1] not in targets:
                reassignment[cls_idx, 1] = count_matrix[cls_idx].argmax()

    acc = count_matrix[reassignment[:, 0], reassignment[:, 1]].sum().astype(np.float32) / preds.shape[0]
    
    # extract max accuracy classes
    num_samples_per_class = count_matrix[reassignment[:, 0], :].sum(axis=1)
    acc_per_class = np.where(num_samples_per_class >= min_samples_per_class,
                             count_matrix[reassignment[:, 0], reassignment[:, 1]] / num_samples_per_class, 0)
    max_acc_classes = np.argsort(acc_per_class)[::-1]
    acc_per_class = acc_per_class[max_acc_classes]
    num_samples_per_class = num_samples_per_class[max_acc_classes]
    if verbose:
        print('=> number of samples: {}'.format(len(targets)))
        print('=> number of unique assignments: {}'.format(len(set(preds))))
        print('=> NMI: {:.3f}%'.format(val_nmi * 100.0))
        print('=> Adjusted NMI: {:.3f}%'.format(val_adjusted_nmi * 100.0))
        print('=> Adjusted Rand-Index: {:.3f}%'.format(val_adjusted_rand_index * 100.0))
        print('=> Accuracy: {:.3f}%'.format(acc * 100.0))

    return acc * 100.0, val_nmi * 100.0, val_adjusted_nmi * 100.0, val_adjusted_rand_index * 100.0, acc_per_class


class SST2Model(nn.Module):
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
        self.classifier = nn.Linear(bert_encoder.d_model, 1, bias=False)

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
