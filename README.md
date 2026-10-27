
# ShadowFox Image Classification

## 1. Project Overview
This project uses Machine Learning and a Convolutional Neural Network (CNN)
to classify images into 10 categories using the CIFAR-10 dataset.

## 2. Objectives
- Build an image classification model using TensorFlow and Keras.
- Preprocess images for model training.
- Train and evaluate a CNN model.
- Predict the category of a user-uploaded image.
- Provide a simple web interface using Streamlit.

## 3. Technologies Used
- Python 3.13
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- Streamlit

## 4. Dataset
The CIFAR-10 dataset contains 60,000 color images of size 32x32 pixels
divided into 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

It contains 50,000 training images and 10,000 test images.

## 5. Project Structure
```text
ShadowFox_Image_Classification/
├── models/
│   └── image_classifier.keras
├── outputs/
│   └── training_curves.png
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## 6. Installation
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use a Python version compatible with the installed TensorFlow release.

## 7. Train the Model
```powershell
python train.py
```

The training script saves the model in the `models` directory
and the training graphs in the `outputs` directory.

## 8. Predict an Image
```powershell
python predict.py "C:\path\to\your\image.jpg"
```

Replace the example path with the actual image path.

## 9. Run the Web Application
```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:
http://localhost:8501

Upload an image and click **Classify Image** to see the prediction.

## 10. Example Result
A test image named `dog.jpg` was classified as:

- Predicted class: Dog
- Model confidence: 69.81%

This is an example prediction, not the overall test accuracy.

## 11. Limitations
The model was trained on 32x32 CIFAR-10 images. Predictions on
high-resolution real-world photographs may be less reliable.

## 12. Future Improvements
- Improve model accuracy through data augmentation.
- Experiment with deeper CNN architectures.
- Evaluate performance on additional test images.
- Improve the user interface and prediction visualization.
