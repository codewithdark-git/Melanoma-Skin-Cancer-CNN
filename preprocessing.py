import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from config import data_config
import logging

class MelanomaDataProcessor:
    def __init__(self, img_size: int = 50):
        self.img_size = img_size
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('melanoma_processing.log'),
                logging.StreamHandler()
            ]
        )

    def load_and_process_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return cv2.resize(img, (self.img_size, self.img_size))

    def process_directory(self, directory: str, label: np.ndarray) -> Tuple[List, List]:
        """
        Process all images in a directory.

        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        directory_path = Path(directory)

        if not directory_path.exists():
            logging.error(f"Directory not found: {directory}")
            return images, labels

        files = list(directory_path.glob('*.jpg')) + list(directory_path.glob('*.png'))

        for file_path in tqdm(files, desc=f"Processing {directory_path.name}"):
            try:
                img_array = self.load_and_process_image(str(file_path))
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                logging.warning(f"Error processing {file_path}: {str(e)}")

        return images, labels

    def process_dataset(self, data_config: Dict) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                         Tuple[np.ndarray, np.ndarray]]:
        """
        Process the entire dataset.

        Returns:
            Tuple of ((train_images, train_labels), (test_images, test_labels))
        """
        # Process training data
        ben_train_images, ben_train_labels = self.process_directory(
            data_config['ben_training_folder'],
            np.array([1, 0])
        )
        mal_train_images, mal_train_labels = self.process_directory(
            data_config['mal_training_folder'],
            np.array([0, 1])
        )

        # Balance benign training data
        ben_train_images = ben_train_images[:len(mal_train_images)]
        ben_train_labels = ben_train_labels[:len(mal_train_labels)]

        # Process testing data
        ben_test_images, ben_test_labels = self.process_directory(
            data_config['ben_testing_folder'],
            np.array([1, 0])
        )
        mal_test_images, mal_test_labels = self.process_directory(
            data_config['mal_testing_folder'],
            np.array([0, 1])
        )

        # Combine images and labels
        train_images = np.array(ben_train_images + mal_train_images)
        train_labels = np.array(ben_train_labels + mal_train_labels)
        test_images = np.array(ben_test_images + mal_test_images)
        test_labels = np.array(ben_test_labels + mal_test_labels)

        # Create shuffling index
        train_shuffle_idx = np.random.permutation(len(train_images))
        test_shuffle_idx = np.random.permutation(len(test_images))

        # Shuffle both images and labels using the same index
        train_images = train_images[train_shuffle_idx]
        train_labels = train_labels[train_shuffle_idx]
        test_images = test_images[test_shuffle_idx]
        test_labels = test_labels[test_shuffle_idx]

        # Log dataset statistics
        self.log_dataset_stats(
            len(ben_train_images),
            len(mal_train_images),
            len(ben_test_images),
            len(mal_test_images)
        )

        return (train_images, train_labels), (test_images, test_labels)

    def log_dataset_stats(self, ben_train: int, mal_train: int,
                         ben_test: int, mal_test: int):
        logging.info("\nDataset Statistics:")
        logging.info(f"Benign training samples: {ben_train}")
        logging.info(f"Malignant training samples: {mal_train}")
        logging.info(f"Benign testing samples: {ben_test}")
        logging.info(f"Malignant testing samples: {mal_test}")
        logging.info(f"Total training samples: {ben_train + mal_train}")
        logging.info(f"Total testing samples: {ben_test + mal_test}")

if __name__ == '__main__':

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


