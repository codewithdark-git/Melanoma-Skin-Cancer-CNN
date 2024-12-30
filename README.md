# Melanoma Detection Demo

This repository provides a demonstration of a deep learning-based system for detecting melanoma from grayscale images. The model predicts whether an input image is classified as "BENIGN" or "MELANOMA," along with a confidence score. **Please note that this application is for educational purposes only and should not be used for real medical diagnosis. Always consult a medical professional for concerns regarding melanoma or other health issues.**

---

## Features
- Pretrained neural network model for binary classification of melanoma.
- Preprocessing pipeline for resizing and normalizing grayscale images.
- Confidence scoring for predictions.
- Easy-to-use class-based implementation.

---

## Disclaimer
> **This application is not a substitute for medical advice or diagnosis. This is a demonstration project intended for programming tutorials and educational purposes. Please consult a qualified healthcare provider for medical concerns.**

---

## Installation

### Prerequisites
- Python 3.7+
- Required libraries:
  - `torch`
  - `torchvision`
  - `numpy`
  - `opencv-python`

Install the dependencies using pip:

```bash
pip install torch torchvision numpy opencv-python
```

---

## How to Run

### Step 1: Prepare the Model
Ensure that the trained model file `saved_model.pth` is present in the project directory. If you do not have the model file, you can train it separately and save it using PyTorch.

### Step 2: Prepare an Input Image
Save a grayscale image (e.g., `sample_image.jpg`) that you want to test in the project directory. The image will be resized to 50x50 pixels during preprocessing.

### Step 3: Run the Demo
Use the following command to run the demonstration:

```bash
python demo.py
```

Replace `demo.py` with the name of the Python file containing the class-based implementation.

---

## Code Overview

### Class: `MelanomaDemo`
The `MelanomaDemo` class provides methods for loading the model, preprocessing input images, and making predictions.

#### Key Methods:
1. **`__init__`**: Initializes the demo by loading the model.
2. **`_load_model`**: Loads the trained PyTorch model from a `.pth` file.
3. **`preprocess_image`**: Resizes, normalizes, and converts an image to a PyTorch tensor.
4. **`predict`**: Performs inference on the input tensor and returns the prediction and confidence score.
5. **`demo`**: Handles the overall demonstration flow, including displaying warnings and prediction results.

### Example Output:
When you run the demo, the output will look like this:

```
============================================================
WARNING! DISCLAIMER!
THIS IS NOT REAL MEDICAL ADVICE!
THIS IS JUST A DEMONSTRATION.
CONSULT A REAL DOCTOR FOR MEDICAL CONCERNS!
============================================================

Prediction: MELANOMA
Confidence: 0.879
============================================================
```

---

## Directory Structure
```
.
Melamoma-Skin-cancer
├── melamoma_cancer_dataset
│   ├── test
│   │   ├── melanoma
│   │   └── benign
│   └── train
|       ├── melanoma
|       └── benign
├── melanoma_dataset.npz
├── melanoma_processing.log
├── Models
│   └── saved_model.pth
├── demo.py                  # Main demo script
├── melanomaNN.py            # Neural network model definition
├── preprocessing.py         # processing data from dataset
├── training.py              # Training script
└── README.md
```

---

## Training the Model
The model used in this demo can be trained using a separate script. Ensure that the dataset is correctly prepared and saved in `.npz` format, with separate arrays for training and testing data.

---

## Contributions
Contributions are welcome! If you have ideas for improvements or additional features, feel free to open a pull request or create an issue.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- PyTorch: https://pytorch.org
- OpenCV: https://opencv.org

---

**Stay safe, and remember: Always consult a medical professional for health concerns!**

