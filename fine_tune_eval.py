import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import os
import random
from torchvision import datasets, transforms
from models.base_encoder import BaseEncoder  
from models.base_encoder_2x import BaseEncoderResNet2x
from models.base_encoder_4x import BaseEncoderResNet4x
from collections import defaultdict
from collections import defaultdict
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USER_DIR = "/mnt/data/swolf"
IMAGENET_TRAIN_DIR = os.path.join(USER_DIR, "imagenet_10_percent/train") 
IMAGENET_ONE_TRAIN_DIR = os.path.join(USER_DIR, "imagenet_1_percent/train") 
IMAGENET_VAL_DIR= os.path.join(USER_DIR, "imagenet_validation_set") 
MODEL_DIR = os.path.join(USER_DIR, "results/models")
LOG_DIR = os.path.join(USER_DIR, "results/logs")
LOG_FILE = os.path.join(LOG_DIR, "finetune_metrics4x_onepercent_simclr_v3.json")
BASE_ENCODER = os.path.join(MODEL_DIR, "simclr_final_base_encoder4x_imagenet_v1.pth")

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def init_imagenet_loaders(batch_size=256):
    train_dataset = datasets.ImageFolder(root=IMAGENET_ONE_TRAIN_DIR, transform=get_train_transform())
    val_dataset = datasets.ImageFolder(root=IMAGENET_VAL_DIR, transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, 1000  

def load_trained_encoder(encoder: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"Loaded encoder from {checkpoint_path}")
    return encoder

def save_metrics(metrics, file_path):
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")

def train_finetune(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 1000,
    epochs: int = 60,
    base_lr: float = 0.05,
    batch_size: int = 64
):
    print("start training")
    for param in encoder.parameters():
        param.requires_grad = True

    classifier = nn.Linear(2048, num_classes).to(device)
    model = nn.Sequential(encoder, classifier).to(device)

    scaled_lr = base_lr * (4096 / 256)
    optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    metrics = {"train_loss": [], "train_acc": [], "val_acc": [], "val_top5_acc": []}
    acc_steps = 64
    for epoch in range(epochs):
        model.train()
        accumulated_loss = 0.0
        accumulated_corrects = 0
        total_samples = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            #Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / acc_steps
            loss.backward()

            #Backprop
            if (step + 1) % acc_steps == 0 or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            batch_size = images.size(0)
            accumulated_loss += loss.item() * batch_size * acc_steps
            _, preds = torch.max(logits, dim=1)
            accumulated_corrects += (preds == labels).sum().item()
            total_samples += batch_size

        epoch_loss = accumulated_loss / total_samples
        epoch_acc = accumulated_corrects / total_samples

        print(f"Fine-Tuning Epoch [{epoch+1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        metrics["train_loss"].append(epoch_loss)
        metrics["train_acc"].append(epoch_acc)
        
        if val_loader is not None:
            val_acc, _ = evaluate_classifier(model, val_loader)
            print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.4f}")
            metrics["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_fine_tuned_model4x_onepercent_epoch_{epoch+1}.pth"))
                print(f"Best model saved with Accuracy: {best_val_acc:.4f}")

    save_metrics(metrics, LOG_FILE)

    return model


def evaluate_classifier(model, data_loader):
    model.eval()
    correct_top1, correct_top5, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            
            # Top-1
            _, pred_top1 = torch.max(logits, dim=1)
            correct_top1 += (pred_top1.long() == labels.long()).sum().item()

            # Top-5
            _, top5_inds = torch.topk(logits, k=5, dim=1)
            for i in range(labels.size(0)):
                if labels[i].item() in top5_inds[i].tolist():
                    correct_top5 += 1

            total_samples += labels.size(0)

    top1_acc = correct_top1 / total_samples
    top5_acc = correct_top5 / total_samples
    return top1_acc, top5_acc

def main():
    epochs = 60
    
    train_loader, val_loader, num_classes = init_imagenet_loaders(batch_size=256)
    encoder = BaseEncoderResNet4x()
    encoder = load_trained_encoder(encoder, BASE_ENCODER)
    model = train_finetune(encoder, train_loader, val_loader, num_classes, epochs=epochs, base_lr=0.05, batch_size=64)

    top1_acc, top5_acc = evaluate_classifier(model, val_loader)
    print(f"ImageNet Final linear eval: Top-1={top1_acc*100:.2f}%  Top-5={top5_acc*100:.2f}%")

if __name__ == "__main__":
    main()
