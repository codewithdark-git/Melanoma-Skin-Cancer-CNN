from training import MelanomaClassifier
from melanomaNN import MelanomaCNN
from torchvision import datasets, transforms, models
from config import (
            device,logging,
            BATCH_SIZE,
            IMG_SIZE,
            LEARNING_RATE,
            EPOCHS, data_config
            )
import torch.nn as nn
import numpy as np
from dataset import transform_data
from preprocessing import MelanomaDataProcessor
from melanomaNN import MelanomaCNN


data = np.load('melanoma_dataset.npz')
X_train = data['train_images']
Y_train = data['train_labels']
X_test = data['test_images']
Y_test = data['test_labels']


train_loader = transform_data(X_train, Y_train)


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the first convolutional layer to accept 1 channel (grayscale)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Optionally, adjust the final fully connected layer if needed (for binary classification)
model.fc = nn.Linear(model.fc.in_features, 2)  # For binary classification (adjust based on your use case)

# the custom cnn model
# model = MelanomaCNN()

classifier = MelanomaClassifier(model=model, img_size=224, batch_size=32, model_name="Pre-trained model")

# Train and evaluate
classifier.train_model(train_loader)
classifier.evaluate_accuracy(train_loader, dataset_name="Train")
classifier.save_model()
