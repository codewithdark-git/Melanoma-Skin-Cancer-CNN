import torch
import numpy as np
import logging
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE = 32
IMG_SIZE = 224
LEARNING_RATE = 0.001
EPOCHS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("melanoma_processing.log"),
        logging.StreamHandler(),
    ],
)


data_config = {
        'ben_training_folder': "melanoma_cancer_dataset/train/benign",
        'mal_training_folder': "melanoma_cancer_dataset/train/malignant",
        'ben_testing_folder': "melanoma_cancer_dataset/test/benign",
        'mal_testing_folder': "melanoma_cancer_dataset/test/malignant",
    }