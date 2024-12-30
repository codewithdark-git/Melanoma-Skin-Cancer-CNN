import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from melanomaNN import MelanomaCNN

class MelanomaClassifier:
    def __init__(self, img_size=50, batch_size=100, learning_rate=0.001, epochs=2, model_path="saved_model.pth"):
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_path = model_path

        # Initialize model, optimizer, and loss function
        self.net = MelanomaCNN()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def load_data(self, file_path, dataset_type="train"):
        """Loads and preprocesses data from a saved .npz file."""
        data = np.load(file_path, allow_pickle=True)
        if dataset_type == "train":
            X = torch.Tensor(data['train_images']) / 255.0  # Normalize pixel values
            y = torch.Tensor(data['train_labels'])  # One-hot encoded labels
        elif dataset_type == "test":
            X = torch.Tensor(data['X_test']) / 255.0
            y = torch.Tensor(data['Y_test'])
        else:
            raise ValueError("Invalid dataset_type. Choose 'train' or 'test'.")
        return X, y

    def train(self, train_X, train_y):
        """Trains the model."""
        self.train_X = train_X
        self.train_y = train_y
        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch + 1}/{self.epochs}")
            for i in range(0, len(self.train_X), self.batch_size):

                # Prepare batches
                batch_X = self.train_X[i:i + self.batch_size].view(-1, 1, self.img_size, self.img_size)
                batch_y = self.train_y[i:i + self.batch_size]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.net(batch_X)

                # Compute loss
                loss = self.loss_function(outputs, batch_y)

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                # Print progress
                progress = (i + self.batch_size) / len(self.train_X) * 100
                print(f"Epoch {epoch + 1}/{self.epochs}, Progress: {progress:.2f}% - Loss: {loss.item():.4f}")

    def evaluate_accuracy(self, X, y, dataset_name="Validation"):
        """Evaluates the accuracy on the given dataset."""
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size].view(-1, 1, self.img_size, self.img_size)
                batch_y = y[i:i + self.batch_size]

                outputs = self.net(batch_X)
                predicted = torch.argmax(outputs, dim=1)
                labels = torch.argmax(batch_y, dim=1)

                correct += (predicted == labels).sum().item()
                total += batch_y.size(0)

        accuracy = (correct / total) * 100
        print(f"{dataset_name} Accuracy: {accuracy:.2f}%")

    def save_model(self):
        """Saves the trained model to a file."""
        torch.save(self.net.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the model from a file."""
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()
        print(f"Model loaded from {self.model_path}")

if __name__ == "__main__":
    # Hyperparameters and file paths
    IMG_SIZE = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 2
    MODEL_PATH = "Models/saved_model.pth"
    DATA_PATH = "melanoma_dataset.npz"

    # Create classifier instance
    classifier = MelanomaClassifier(img_size=IMG_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, model_path=MODEL_PATH)

    # Load training data and train the model
    train_X, train_y = classifier.load_data(DATA_PATH, dataset_type="train")
    classifier.train(train_X, train_y)
    classifier.evaluate_accuracy(train_X, train_y, dataset_name="Training")
    classifier.save_model()

    # # Load testing data and evaluate the model
    # test_X, test_y = classifier.load_data(DATA_PATH, dataset_type="train")
    # classifier.load_model()
    # classifier.evaluate_accuracy(test_X, test_y, dataset_name="Testing")
