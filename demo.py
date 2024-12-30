import cv2
import numpy as np
import torch
from melanomaNN import MelanomaCNN
import warnings

warnings.filterwarnings("ignore")

class MelanomaDemo:
    def __init__(self, model_path="Models/saved_model.pth", img_size=50):
        self.model_path = model_path
        self.img_size = img_size
        self.net = self._load_model()

    def _load_model(self):
        """Loads the trained model."""
        net = MelanomaCNN()
        net.load_state_dict(torch.load(self.model_path))
        net.eval()
        print(f"Model loaded from {self.model_path}")
        return net

    def preprocess_image(self, image_path):
        """Reads and preprocesses the input image."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_tensor = torch.Tensor(img_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        return img_tensor

    def predict(self, image_tensor):
        """Makes a prediction on the preprocessed image tensor."""
        with torch.no_grad():
            output = self.net(image_tensor)[0]
        prediction = "BENIGN" if output[0] >= output[1] else "MELANOMA"
        confidence = round(float(output.max()), 3)
        return prediction, confidence

    def demo(self, image_path):
        """Runs the melanoma prediction demo."""
        print("=" * 60)
        print("WARNING! DISCLAIMER!")
        print("THIS IS NOT REAL MEDICAL ADVICE!")
        print("THIS IS JUST A DEMONSTRATION.")
        print("CONSULT A REAL DOCTOR FOR MEDICAL CONCERNS!")
        print("=" * 60)

        try:
            image_tensor = self.preprocess_image(image_path)
            prediction, confidence = self.predict(image_tensor)

            print(f'Prediction: {prediction}')
            print(f'Confidence: {confidence}')
            print("=" * 60)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    image_path = "melanoma_cancer_dataset/test/benign/melanoma_9605.jpg"  # Replace with your image path
    demo = MelanomaDemo()
    demo.demo(image_path)
