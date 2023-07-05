
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

def unnormalize_image(image, mean=0, std=1.0, permute=True):
    """
    Transform the given image by un-normalizing it with the given mean and standard deviation.
    :param image: image to unnormalize as numpy array.
    :param mean: mean to use for unnormalization.
    :param std: standard deviation to use for unnormalization.
    :param permute: flag to indicate if image should be flipped and its dimensions rearranged.
    :return: unnormalized image as numpy array.
    """
    if type(mean) != torch.Tensor:
        mean = torch.tensor(mean, dtype=torch.float32)

    if type(std) != torch.Tensor:
        std = torch.tensor(std, dtype=torch.float32)

    unnormalize = T.Normalize(
        (-mean / std).tolist(), 
        (1.0 / std).tolist()
    )

    unnormalized_image = unnormalize(image).numpy()
    if permute:
        return np.transpose(unnormalized_image, (1, 2, 0))
    return unnormalized_image

def imshow(img, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
    """
    shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
    """
    plt.imshow(unnormalize_image(img, mean=mean, std=std, permute=True))


def prevalidate(model, val_loader,criterion, device):
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
        return val_loss_epoch

def pretrain_one_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    loss_step = []
    for data in train_loader:
        # Move the data to the GPU
        inp_data, labels = data
        inp_data, labels = inp_data.to(device), labels.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_step.append(loss.item())
    # dont forget the means here
    loss_curr_epoch = np.mean(loss_step)
    return loss_curr_epoch

def save_model(model, path, epoch, optimizer, val_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }, path)

def pretrain(model, optimizer, num_epochs, train_loader, val_loader, criterion, device):
    dict_log = {"train_loss":[], "val_loss":[]}
    best_val_loss = 1e8
    model = model.to(device)
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss_curr_epoch = pretrain_one_epoch(model, optimizer, train_loader, criterion, device)
        val_loss = prevalidate(model, val_loader, criterion, device)

        # Print epoch results to screen 
        msg = (f'Ep {epoch+1}/{num_epochs}: || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
        pbar.set_description(msg)

        dict_log["train_loss"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)
        
        # Use this code to save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, f'best_model_min_val_loss.pth', epoch, optimizer, val_loss)   
    return dict_log



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
def get_features(model, dataloader, device):
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
        msg = (f'Ep {epoch+1}/{num_epochs}: Accuracy : Train:{train_acc:.2f} \t Val:{val_acc:.2f} || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}')
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
