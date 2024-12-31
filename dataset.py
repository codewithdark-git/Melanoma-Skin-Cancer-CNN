import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image  # Required for transforms to work on NumPy arrays
import numpy as np
from config import logging, device

class NumpyDataset(Dataset):
    """
    Custom Dataset to load data directly from NumPy arrays with transformations.
    """
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy.ndarray): Feature data (e.g., images) as a NumPy array.
            labels (numpy.ndarray): Labels as a NumPy array.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the feature and label
        feature = self.data[idx]
        label = self.labels[idx]

        # Convert the feature (image) to a PIL image
        feature = Image.fromarray((feature.squeeze() * 255).astype(np.uint8))  # Squeeze if single channel

        # If transforms are provided, apply them to the feature
        if self.transform:
            feature = self.transform(feature)

        # Convert the label to a PyTorch tensor
        
        label = torch.tensor(label, dtype=torch.float, device=device)
        

        return feature, label

def transform_data(train_images, train_labels):
    """
    Transforms the data and loads it into DataLoader.
    """
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust normalization for grayscale
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat grayscale channel 3 times
    ])

    # Create Dataset and DataLoader
    dataset = NumpyDataset(train_images, train_labels, transform=transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    logging.info(f"The dataset create successfully")
    logging.info(f"Train dataset loaded. Size: {len(dataset)}")
    
    return data_loader
