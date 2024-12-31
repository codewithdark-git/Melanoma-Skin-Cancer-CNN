
from training import MelanomaClassifier
from melanomaNN import MelanomaCNN
from torchvision import datasets, transforms, models
import torch.nn as nn


# Define constants
BATCH_SIZE = 32
IMG_SIZE = 224
LEARNING_RATE = 0.001
EPOCHS = 2



# Load dataset
train_dir = 'melanoma_cancer_dataset/train'
test_dir = 'melanoma_cancer_dataset/test'
print(f'The pipeline running')


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the first convolutional layer to accept 1 channel (grayscale)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Optionally, adjust the final fully connected layer if needed (for binary classification)
model.fc = nn.Linear(model.fc.in_features, 2)  # For binary classification (adjust based on your use case)


classifier = MelanomaClassifier(model=model, img_size=224, batch_size=32)


# Load and transform data
train_loader, test_loader = classifier.transform_data(train_dir, test_dir)

# Train and evaluate
classifier.train_model(train_loader)
classifier.evaluate_accuracy(test_loader, dataset_name="Test")
classifier.save_model(model_name="pretrained")
