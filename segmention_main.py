import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from segmentation import load_deeplabv3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_deeplabv3()
model.to(device)

def segment_image(image):
    """Runs DeepLabV3 to extract foreground mask and checks segmentation quality."""
    orig_size = image.size  

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]  

    mask = output.argmax(0).byte().cpu().numpy()  

    mask_pil = Image.fromarray(mask * 255)  
    mask_resized = mask_pil.resize(orig_size, Image.NEAREST)  
    mask = np.array(mask_resized) > 0  

    foreground_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])  

    if foreground_ratio < 0.05:
        return None  

    if foreground_ratio > 0.90:
        return None  

    return mask

def apply_foreground_augmentation(foreground):
    """Apply transformations to the foreground object."""
    augmentations = transforms.Compose([
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
    ])
    return augmentations(foreground)

def apply_background_augmentation(background):
    """Apply transformations to the background."""
    augmentations = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))], p=0.85),
        transforms.RandomGrayscale(p=0.3) 
    ])
    return augmentations(background)

def crop_and_resize_segmented_based_augmentations(image):
    augmentations = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0), ratio=(3/4, 4/3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return augmentations(image)

def normal_simclr_augmentations(image):
    """Applies standard SimCLR augmentations."""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return transform(image)

def segmentation_based_augmentation(image):
    """Applies different augmentations to the foreground and background.
    Falls back to normal SimCLR augmentations if segmentation fails.
    """
    mask = segment_image(image)

    if mask is None:
        return normal_simclr_augmentations(image)

    image_np = np.array(image)

    # Separate foreground & background
    mask_fg = mask[:, :, None]  
    mask_bg = ~mask_fg  

    foreground = Image.fromarray((image_np * mask_fg).astype(np.uint8))
    background = Image.fromarray((image_np * mask_bg).astype(np.uint8))

    # Apply different augmentations
    foreground_aug = apply_foreground_augmentation(foreground)
    background_aug = apply_background_augmentation(background)

    foreground_np = np.array(foreground_aug)
    background_np = np.array(background_aug)

    # Merge foreground and background back into one image
    combined = np.where(mask_fg, foreground_np, background_np)

    combined_pil = Image.fromarray(combined.astype(np.uint8))
    crop = crop_and_resize_segmented_based_augmentations(combined_pil)

    return crop
