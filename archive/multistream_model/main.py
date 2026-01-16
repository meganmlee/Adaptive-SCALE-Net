import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from config import *
from dataset import (
    load_split_dataset, 
    MI_load_split_dataset, 
    P300_load_split_dataset, 
    ImaginedSpeech_load_split_dataset
)
from model import MultiExpertNet

def get_loaders():
    if task == "SSVEP":
        return load_split_dataset(data_dir, num_seen, seed)
    elif task == "MI":
        return MI_load_split_dataset(data_dir, num_seen, seed)
    elif task == "P300":
        return P300_load_split_dataset(data_dir, num_seen, seed, num_workers=num_workers)
    elif task == "Imagined_speech":
        return ImaginedSpeech_load_split_dataset(data_dir, num_seen, seed, num_workers=num_workers)
    else:
        raise ValueError(f"Unknown task: {task}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        inputs, labels = batch[0], batch[1]
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item()})

    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs, labels = batch[0], batch[1]
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100 * correct / total

def main():
    print(f"--- Running MultiExpertNet for {task} ---")
    print(f"Device: {device}")
    
    loaders = get_loaders()
    
    model = MultiExpertNet(
        n_channels=channels, 
        n_classes=num_classes, 
        n_fft=n_fft
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders['val'], criterion, device, desc="Validation")
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_dualstream.pth"))
    
    print("\n--- Final Testing ---")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_dualstream.pth")))
    
    if 'test1' in loaders:
        test1_loss, test1_acc = evaluate(model, loaders['test1'], criterion, device, desc="Test Seen")
        print(f"Test 1 (Seen Subjects): Acc: {test1_acc:.2f}%")
        
    if 'test2' in loaders:
        test2_loss, test2_acc = evaluate(model, loaders['test2'], criterion, device, desc="Test Unseen")
        print(f"Test 2 (Unseen Subjects): Acc: {test2_acc:.2f}%")

if __name__ == "__main__":
    main()