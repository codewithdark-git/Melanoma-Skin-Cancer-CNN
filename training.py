import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np
import tqdm
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("melanoma_processing.log"),
        logging.StreamHandler(),
    ],
)


class MelanomaClassifier:
    def __init__(self, model, img_size=50, batch_size=100, learning_rate=0.001, epochs=2, model_path="Models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path

        # Initialize model, optimizer, and loss function
        if model is None:
            raise ValueError("Please provide a valid model instance.")
        self.net = model.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def transform_data(self, train_path, test_path):
        """Transforms the data and loads it into DataLoader."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Train or test path does not exist.")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust normalization for grayscale
        ])

        # Load datasets with transformations
        train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        logging.info(f"Train and test datasets loaded. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
        return train_loader, test_loader

    def train_model(self, train_loader):
        """Train the model with progress tracking using tqdm."""
        self.net.train()  # Set model to training mode

        for epoch in range(self.epochs):
            progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(progress_bar):
                # Move data to the same device as the model
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Update progress bar description
                progress_bar.set_description(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

            # Log epoch loss
            epoch_loss = running_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}/{self.epochs} Loss: {epoch_loss:.4f}")

    def evaluate_accuracy(self, data_loader, dataset_name="Validation"):
        """Evaluates the accuracy on the given dataset."""
        self.net.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = (correct / total) * 100
        logging.info(f"{dataset_name} Accuracy: {accuracy:.2f}%")
        print(f"{dataset_name} Accuracy: {accuracy:.2f}%")

    def summary_model(self):
        """Displays a summary of the model architecture."""
        return summary(self.net, input_size=(1, 224, 224), device=str(self.device))

    def save_model(self, model_name="pretrained"):
        """Saves the trained model to a file."""
        os.makedirs(self.model_path, exist_ok=True)
        file_path = os.path.join(self.model_path, f"{model_name}_model.pth")
        torch.save(self.net.state_dict(), file_path)
        logging.info(f"Model saved to {file_path}")
        print(f"Model saved to {file_path}")

    def load_model(self, model_name="pretrained"):
        """Loads the model from a file."""
        file_path = os.path.join(self.model_path, f"{model_name}_model.pth")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")
        self.net.load_state_dict(torch.load(file_path, map_location=self.device))
        self.net.eval()
        logging.info(f"Model loaded from {file_path}")
        print(f"Model loaded from {file_path}")


