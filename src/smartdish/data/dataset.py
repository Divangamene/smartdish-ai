"""Carregamento de imagens organizadas em pastas por classe:

data/processed/train/<classe>/*.jpg
data/processed/val/<classe>/*.jpg
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloader(root: str, image_size: int = 224, batch_size: int = 32,
                   train: bool = True, num_workers: int = 2) -> DataLoader:
    ds = datasets.ImageFolder(root, transform=get_transforms(image_size, train))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
