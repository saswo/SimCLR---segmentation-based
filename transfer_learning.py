import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from torchvision.datasets import Caltech101
from models.base_encoder import BaseEncoder  
from models.linear_classifier import LinearClassifier

USER_DIR = "/mnt/data/swolf"
MODEL_DIR = os.path.join(USER_DIR, "results/models")
LOG_DIR = os.path.join(USER_DIR, "results/logs")

BASE_ENCODER_STANDARD = os.path.join(MODEL_DIR, "simclr_final_base_encoder_imagenet_v1.pth")
BASE_ENCODER_SEGMENTATION = os.path.join(MODEL_DIR, "simclr_final_base_encoder_seg_v2.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_rgb(img):
    """Ensures images are in RGB format without losing color information."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def get_transform_for_transfer_learning():
    return transforms.Compose([
        transforms.Lambda(ensure_rgb), 
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def init_caltech101_loaders(batch_size=64):
    train_dataset = Caltech101(root="./data", transform=get_transform_for_transfer_learning(), target_type="category", download=True)
    
    num_classes = 101
    train_size = int(0.8 * len(train_dataset))  
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader, num_classes

def load_trained_encoder(encoder: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()
    print(f"Loaded encoder from checkpoint: {checkpoint_path}")
    return encoder

def train_linear_classifier(
   encoder: nn.Module,
   train_loader: DataLoader,
   test_loader: DataLoader,
   num_classes: int = 101,
   epochs: int = 90,
   base_lr: float = 0.1,
   batch_size: int = 64
):
   encoder.eval()  # Freeze encoder weights
   classifier = LinearClassifier(in_features=2048, num_classes=num_classes).to(device)

   scaled_lr = base_lr * (batch_size / 256)
   optimizer = torch.optim.SGD(classifier.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=1e-6)
   criterion = nn.CrossEntropyLoss()

   best_test_acc = 0.0
   train_acc_progression = []

   for epoch in range(epochs):
      classifier.train()
      total_loss = 0.0
      correct_preds = 0
      total_samples = 0

      for images, labels in train_loader:
         images, labels = images.to(device), labels.to(device)

         with torch.no_grad():
               features = encoder(images)

         logits = classifier(features)
         loss = criterion(logits, labels)

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         batch_size = images.size(0)
         total_loss += loss.item() * batch_size
         _, preds = torch.max(logits, dim=1)
         correct_preds += (preds == labels).sum().item()
         total_samples += batch_size

      epoch_loss = total_loss / total_samples
      epoch_acc = correct_preds / total_samples
      train_acc_progression.append({"epoch": epoch + 1, "train_acc": epoch_acc})

      print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

      test_acc, _ = evaluate_classifier(encoder, classifier, test_loader)
      print(f"Test Accuracy: {test_acc:.4f}")

      if test_acc > best_test_acc:
         best_test_acc = test_acc
         torch.save(classifier.state_dict(), os.path.join(MODEL_DIR, f"best_linear_classifier_caltech101.pth"))
         print(f"New best model saved with Accuracy: {best_test_acc:.4f}")

   return classifier, train_acc_progression, best_test_acc

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
            logits = classifier(features)

            # Top-1 Accuracy
            _, pred_top1 = torch.max(logits, dim=1)
            correct_top1 += (pred_top1 == labels).sum().item()

            # Top-5 Accuracy
            _, top5_inds = torch.topk(logits, k=5, dim=1)
            for i in range(labels.size(0)):
                if labels[i].item() in top5_inds[i].tolist():
                    correct_top5 += 1

            total_samples += labels.size(0)

    top1_acc = correct_top1 / total_samples
    top5_acc = correct_top5 / total_samples
    return top1_acc, top5_acc

def save_metrics(results, filename):
    file_path = os.path.join(LOG_DIR, filename)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {file_path}")

def main():
    train_loader, test_loader, num_classes = init_caltech101_loaders(batch_size=64)

    # Standard Base Encoder
    print("\nEvaluating Standard Base Encoder")
    encoder_standard = BaseEncoder()
    load_trained_encoder(encoder_standard, BASE_ENCODER_STANDARD)

    classifier_standard, train_acc_standard, best_test_standard = train_linear_classifier(
        encoder=encoder_standard,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes
    )

    top1_standard, top5_standard = evaluate_classifier(encoder_standard, classifier_standard, test_loader)
    results_standard = {
        "model": "Standard Base Encoder",
        "best_test_acc": best_test_standard,
        "top1_acc": top1_standard,
        "top5_acc": top5_standard,
        "train_acc_progression": train_acc_standard
    }
    save_metrics(results_standard, "2caltech101_standard.json")

    # Segmentation-Based Base Encoder
    print("\nEvaluating Segmentation-Based Base Encoder")
    encoder_segmentation = BaseEncoder()
    load_trained_encoder(encoder_segmentation, BASE_ENCODER_SEGMENTATION)

    classifier_segmentation, train_acc_segmentation, best_test_segmentation = train_linear_classifier(
        encoder=encoder_segmentation,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes
    )

    top1_segmentation, top5_segmentation = evaluate_classifier(encoder_segmentation, classifier_segmentation, test_loader)
    results_segmentation = {
        "model": "Segmentation-Based Base Encoder",
        "best_test_acc": best_test_segmentation,
        "top1_acc": top1_segmentation,
        "top5_acc": top5_segmentation,
        "train_acc_progression": train_acc_segmentation
    }
    save_metrics(results_segmentation, "2caltech101_segmentation.json")


if __name__ == "__main__":
    main()
