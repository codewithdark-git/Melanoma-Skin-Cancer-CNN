import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def transform_data(train_path, test_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust normalization for grayscale
    ])

    
    # Load the datasets with grayscale transformation
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader



