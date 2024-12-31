from training import MelanomaClassifier


BATCH_SIZE = 32
DATA_PATH = "melanoma_dataset.npz"

# Initialize the classifier instance

classifier = MelanomaClassifier()

test_X, test_y = classifier.load_data(DATA_PATH, dataset_type="testing")
classifier.load_model()
classifier.evaluate_accuracy(test_X, test_y, dataset_name="Testing")
