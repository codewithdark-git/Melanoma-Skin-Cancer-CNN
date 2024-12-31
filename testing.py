from training import MelanomaClassifier
from dataset import transform_data
from config import test_images, test_labels


test_loader = transform_data(test_images, test_labels)
classifier = MelanomaClassifier()
classifier.load_model()
classifier.evaluate_accuracy(test_loader, dataset_name="Testing")
