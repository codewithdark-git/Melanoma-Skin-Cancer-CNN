"""
create the full model train pipeline from the data to evaluate the model
"""


from preprocessing import MelanomaDataProcessor
from melanomaNN import MelanomaCNN
from training import MelanomaClassifier
import numpy as np
import torch
import torch.optim


# Define parameters

DATA_PATH = "melanoma_dataset.npz"
MODEL_PATH = "Models/saved_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 2

def main():
    # Create data processor
    data_config = {
            'ben_training_folder': "melanoma_cancer_dataset/train/benign",
            'mal_training_folder': "melanoma_cancer_dataset/train/malignant",
            'ben_testing_folder': "melanoma_cancer_dataset/test/benign",
            'mal_testing_folder': "melanoma_cancer_dataset/test/malignant"
        }
        
        # Initialize and run processor
    processor = MelanomaDataProcessor(img_size=224)
    (train_images, train_labels), (test_images, test_labels) = processor.process_dataset(data_config)

    # Save processed data
    np.savez_compressed(
        'melanoma_dataset.npz',
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels
    )


    # Initialize and train the classifier
    classifier = MelanomaClassifier(img_size=IMG_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, model_path=MODEL_PATH)

    # Load training data and train the model
    train_X, train_y = classifier.load_data(DATA_PATH, dataset_type="train")
    classifier.train(train_X, train_y)
    classifier.evaluate_accuracy(train_X, train_y, dataset_name="Training")
    classifier.save_model()


    # Load testing data and evaluate the model

    test_X, test_y = classifier.load_data(DATA_PATH, dataset_type="test")
    classifier.load_model()
    classifier.evaluate_accuracy(test_X, test_y, dataset_name="Testing")


if __name__ == '__main__':
    main()