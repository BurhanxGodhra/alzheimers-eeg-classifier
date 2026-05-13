import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def augment_eeg(x, noise_std=0.05, channel_drop=2, time_shift_max=20):
    
    batch_size, channels, time_len = x.shape
    
    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std
        x = x + noise
    
    if channel_drop > 0 and channels > channel_drop:
        drop_chs = random.sample(range(channels), channel_drop)
        x[:, drop_chs, :] = 0
        
    if time_shift_max > 0:
        shift = random.randint(-time_shift_max, time_shift_max)
        x = torch.roll(x, shifts=shift, dims=-1)
    return x

def train_epoch(model, train_loader, criterion, optimizer, device, augment=True):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if augment:
            batch_x = augment_eeg(batch_x)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            _, pred = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    return total_loss / len(loader.dataset), 100 * correct / total, all_preds, all_labels
