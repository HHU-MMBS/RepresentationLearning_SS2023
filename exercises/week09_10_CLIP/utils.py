import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

@torch.no_grad()
def get_features(model, dataloader, device):
    model.eval()
    model = model.to(device)
    feats, labs = [], []
    for i in dataloader:
        inp_data,labels = i
        inp_data = inp_data.to(device)
        features = model(inp_data)
        features = features.cpu().detach().flatten(start_dim=1)
        labels = labels.cpu().detach()
        feats.append(features)
        labs.append(labels)
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

def auroc_score(score_in, score_out):
    score_in = score_in.cpu()
    score_out = score_out.cpu()
    labels = torch.cat((torch.ones_like(score_in), torch.zeros_like(score_out)))
    return roc_auc_score(labels.numpy(), torch.cat((score_in, score_out)).numpy()) * 100

@torch.no_grad()
def OOD_classifier_knn(train_features, test_features, k=1):
    # k = -1 for whole trainset
    if k < 0:
        k = len(train_features)

    num_chunks = 128  # num of test images in loop
    num_test_images = test_features.shape[0]
    imgs_per_chunk = num_test_images // num_chunks
    cos_sim = torch.zeros(num_test_images).cuda()

    train_features = nn.functional.normalize(train_features, dim=-1, p=2)
    test_features = nn.functional.normalize(test_features, dim=-1, p=2)
        
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        idx_next_chunk = min((idx + imgs_per_chunk), num_test_images)
        features = test_features[idx : idx_next_chunk, :]
        # calculate the metric and compute ood scores
        similarity =  features @ train_features.T
        top_sim, _ = similarity.topk(k, largest=True, sorted=True, dim=-1)
        cos_sim[idx: idx_next_chunk] = top_sim.mean(dim=1)
    return cos_sim

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


def linear_eval(model, optimizer, num_epochs, train_loader, val_loader, device, prefix="CLIP"):
    best_acc = 0
    model = model.to(device)
    dict_log = {"train_acc_epoch":[], "val_acc_epoch":[], "loss_epoch":[], "val_loss":[]}
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
        
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'best_acc': best_acc,
                  }, f'{prefix}_best_max_train_acc.pth')
    return dict_log

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']}")
    return model
