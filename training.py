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
from config import logging, device



class MelanomaClassifier:
    def __init__(self, model, model_name, img_size=50, batch_size=100, learning_rate=0.001, epochs=2, model_path="Models"):
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path
        self.model_name = model_name

        # Initialize model, optimizer, and loss function
        if model is None:
            raise ValueError("Please provide a valid model instance.")
        self.net = model.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()


    def train_model(self, train_loader):
        """Train the model with progress tracking using tqdm."""
        print(f"""
              
                ****************************************
                ||=    Training MelanomaClassifier
                ||=    -------------------------------
                ||=    Batch size: {self.batch_size}
                ||=    Learning rate: {self.learning_rate}
                ||=    Epochs: {self.epochs}
                ||=    Model path: {self.model_path}
                ||=    Device: {self.device}
                ||=    Model : {self.model_name}
                ****************************************
        
            """)
        self.net.to(self.device)
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

    def save_model(self):
        """Saves the trained model to a file."""
        os.makedirs(self.model_path, exist_ok=True)
        file_path = os.path.join(self.model_path, f"{self.model_name}_model.pth")
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


