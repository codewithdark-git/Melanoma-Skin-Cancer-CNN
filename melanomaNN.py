import torch
import torch.nn as nn
import torch.nn.functional as F

class MelanomaCNN(nn.Module):
    """
    Convolutional Neural Network for melanoma classification.

    Architecture:
        - 3 Convolutional layers with max pooling
        - 2 Fully connected layers
        - Output layer with softmax activation

    Input shape: (batch_size, 1, 224, 224)
    Output shape: (batch_size, 2) - [benign_prob, melanoma_prob]
    """

    def __init__(self):
        """Initialize the network layers."""
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5
        )

        # Fully connected layers
        # After three convolutional layers with max pooling, the input to the fully connected layers will have the following dimensions:
        # input_size = (224, 224)
        # After conv1 + pooling -> (32, 220, 220)
        # After conv2 + pooling -> (64, 108, 108)
        # After conv3 + pooling -> (128, 52, 52)
        self.fc1 = nn.Linear(128 * 24 * 24, 512)  # Flattened size from conv3
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224)

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, 2)
        """
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        # Third convolutional block
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 24 * 24)  # Flattened size after conv3

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Apply softmax along the class dimension

        return x

    def get_feature_dims(self, input_size=(1, 224, 224)):
        """
        Calculate feature dimensions at each layer.
        Useful for debugging and architecture visualization.

        Args:
            input_size (tuple): Input dimensions (channels, height, width)

        Returns:
            dict: Dictionary containing feature dimensions at each layer
        """
        dims = {}
        x = torch.zeros(1, *input_size)  # Create dummy input

        # Conv1
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        dims['conv1'] = x.shape

        # Conv2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        dims['conv2'] = x.shape

        # Conv3
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        dims['conv3'] = x.shape

        # Flattened
        dims['flatten'] = x.view(-1, 128 * 24 * 24).shape

        return dims
