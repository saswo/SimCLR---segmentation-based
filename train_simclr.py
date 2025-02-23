import torch
from torch.utils.data import DataLoader
from models.base_encoder import BaseEncoder
from models.projection_head import ProjectionHead
from models.loss_function import NTXentLoss
from torch.optim.lr_scheduler import LambdaLR
from flash.core.optimizers import LARS
import os
import json
from models.base_encoder_2x import BaseEncoderResNet2x
from models.base_encoder_4x import BaseEncoderResNet4x
from torchvision.datasets import ImageFolder
from augmentations import Augmentation
import multiprocessing

USER_DIR = "/mnt/data/swolf"
IMAGNET_TRAIN_DIR = os.path.join(USER_DIR, "imagenet_10_percent/train") 
IMAGENET_VAL_DIR= os.path.join(USER_DIR, "imagenet_validation_set") 
CHECKPOINT_DIR = os.path.join(USER_DIR, "results/checkpoints")
LOG_DIR = os.path.join(USER_DIR, "results/logs")
MODEL_DIR = os.path.join(USER_DIR, "results/models")

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    def lr_lambda(current_step):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps) 
        else:
            progress = float(current_step - warmup_steps) / float(total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))) 

    return LambdaLR(optimizer, lr_lambda)

def save_checkpoint(encoder, projection_head, optimizer, epoch):
    save_path = f"{CHECKPOINT_DIR}/simclr_epoch_{epoch}_base_encoder_seg_v2.pth"
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'projection_head_state_dict': projection_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch} -> {save_path}")

def save_loss_history(loss_history):
    loss_path = f"{LOG_DIR}/loss_base_encoder_seg_v2.json"
    with open(loss_path, "w") as f:
        json.dump(loss_history, f, indent=4)
    print(f"Loss saved: {loss_path}")

def save_final_model(encoder, projection_head):
    final_model_path = f"{MODEL_DIR}/simclr_final_base_encoder_seg_v2.pth"
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'projection_head_state_dict': projection_head.state_dict(),
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")

def train_simclr(encoder, projection_head, train_loader, epochs, temperature, lr, accumulate_steps=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, projection_head = encoder.to(device), projection_head.to(device)

    optimizer = LARS(
        params=list(encoder.parameters()) + list(projection_head.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-6
    )

    steps_per_epoch = len(train_loader) // accumulate_steps
    scheduler = cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_epochs=10,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    loss_fn = NTXentLoss(temperature=temperature)
    loss_history = []
    
    for epoch in range(epochs):
        encoder.train()
        projection_head.train()
        total_loss = 0

        for batch_idx, ((view_1, view_2), _) in enumerate(train_loader):
            view_1, view_2 = view_1.to(device), view_2.to(device)

            h_1 = encoder(view_1)
            h_2 = encoder(view_2)
            z_1 = projection_head(h_1)
            z_2 = projection_head(h_2)
            loss = loss_fn(z_1, z_2) / accumulate_steps
            loss.backward()
            total_loss += loss.item()

            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
        avg_loss = total_loss / len(train_loader)
        loss_history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        save_loss_history(loss_history)
        if (epoch + 1) % 5 == 0: 
            save_checkpoint(encoder, projection_head, optimizer, epoch+1)
    save_final_model(encoder, projection_head)

def main_training():
    torch.cuda.empty_cache() 
    torch.backends.cudnn.benchmark = True  

    dataset = ImageFolder(root=IMAGNET_TRAIN_DIR, transform=Augmentation())
    train_loader = DataLoader (
        dataset,   
        batch_size=64,
        shuffle=True,
        num_workers=18,
        pin_memory=True,  
        drop_last=True,
        prefetch_factor=4
    )

    encoder = BaseEncoder()
    projection_head = ProjectionHead(input_dim=2048, hidden_dim=2048, output_dim=128)
    
    print("Training SimCLR...")
    train_simclr(encoder, projection_head, train_loader, epochs=100, temperature=0.1, lr=4.8, accumulate_steps=64)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True) 
    main_training()

  
