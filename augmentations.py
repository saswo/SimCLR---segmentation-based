import torchvision.transforms as transforms

class Augmentation:
    def __init__(self):
        self.apply_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=21)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, x):
        view_1 = self.apply_augmentation(x)
        view_2 = self.apply_augmentation(x)

        return view_1, view_2