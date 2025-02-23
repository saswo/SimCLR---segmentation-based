import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchvision import datasets, transforms
from models.base_encoder import BaseEncoder  
from models.base_encoder_2x import BaseEncoderResNet2x
from models.base_encoder_4x import BaseEncoderResNet4x
from models.linear_classifier import LinearClassifier
from torchvision.datasets import CIFAR10

USER_DIR = "/mnt/data/swolf"
IMAGNET_TRAIN_DIR = os.path.join(USER_DIR, "imagenet_10_percent/train") 
IMAGENET_VAL_DIR= os.path.join(USER_DIR, "imagenet_validation_set") 
MODEL_DIR = os.path.join(USER_DIR, "results/models")
LOG_DIR = os.path.join(USER_DIR, "results/logs")
LOG_FILE = os.path.join(LOG_DIR, "rcifar_128_v2.json")
BASE_ENCODER = os.path.join(MODEL_DIR, "simclr_final_base_encoder_real_cifar_128_final.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_train_transform_for_cifar():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)) 
    ])

def get_val_transform_for_cifar():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)) 
    ])

def get_train_transform_for_imageNet():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_val_transform_for_imageNet():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def init_cifar_loaders(batch_size=64):
    train_dataset = CIFAR10(root='./data', train=True, transform=get_train_transform_for_cifar(), download=True)
    val_dataset = CIFAR10(root='./data', train=False, transform=get_val_transform_for_cifar(), download=True)
    num_classes = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes

def init_imagenet_loaders(batch_size=256):
    train_dataset = datasets.ImageFolder(root=IMAGNET_TRAIN_DIR, transform=get_train_transform_for_imageNet())
    val_dataset   = datasets.ImageFolder(root=IMAGENET_VAL_DIR, transform=get_val_transform_for_imageNet())
    num_classes = 1000

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, num_classes

def save_metrics(metrics, file_path):
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")

def load_trained_encoder(encoder: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"Loaded encoder from checkpoint: {checkpoint_path}")
    return encoder

def freeze_encoder(encoder: nn.Module):
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

def train_linear_classifier(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_classes: int = 1000,
    epochs: int = 90,
    base_lr: float = 0.1,
    batch_size: int = 64
):
    
    freeze_encoder(encoder)
    encoder.to(device)

    classifier = LinearClassifier(in_features=2048, num_classes=num_classes).to(device)

    scaled_lr = base_lr * (batch_size/256)
    print(f"Using scaled LR = {scaled_lr:.4f}")

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=scaled_lr,
        momentum=0.9,
        weight_decay=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    metrics = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        classifier.train()
        accumulated_loss = 0.0
        accumulated_corrects = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = encoder(images).detach()

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            accumulated_loss += loss.item() * batch_size
            _, preds = torch.max(logits, dim=1)
            accumulated_corrects += (preds == labels).sum().item()
            total_samples += batch_size

        epoch_loss = accumulated_loss / total_samples
        epoch_acc  = accumulated_corrects / total_samples

        print(f"Linear Eval Training Epoch [{epoch+1}/{epochs}] "
              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        metrics["train_loss"].append(epoch_loss)
        metrics["train_acc"].append(epoch_acc)
        
        if val_loader is not None:
            val_acc, _ = evaluate_classifier(encoder, classifier, val_loader)
            print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.4f}")
            metrics["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(classifier.state_dict(), os.path.join(MODEL_DIR, f"best_linear_classifier_epoch_{epoch+1}_cifar_v5.pth"))
                print(f"Best model saved with Accuracy: {best_val_acc:.4f}")

    save_metrics(metrics, LOG_FILE)

    return classifier

def evaluate_classifier(encoder: nn.Module, classifier: nn.Module, data_loader: DataLoader):
    encoder.eval()
    classifier.eval()

    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            features = encoder(images)
            logits   = classifier(features)

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
    #train_loader, val_loader, num_classes = init_imagenet_loaders(batch_size=256)
    train_loader, val_loader, num_classes = init_cifar_loaders(batch_size=64)

    encoder = BaseEncoder()
    load_trained_encoder(encoder, BASE_ENCODER)

    classifier = train_linear_classifier(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=90,
        base_lr=0.1,
        batch_size=64
    )

    top1_acc, top5_acc = evaluate_classifier(encoder, classifier, val_loader)
    print(f"ImageNet Final linear eval: Top-1={top1_acc*100:.2f}%  Top-5={top5_acc*100:.2f}%")

if __name__ == "__main__":
    main()