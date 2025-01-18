# Handwritten Digit Recognition

A deep learning-powered handwritten digit recognition system that allows users to draw digits on a canvas and a trained CNN model will try to recognize it.

## Features
✅ Users can draw digits on an interactive canvas  
✅ Preprocessing done with OpenCV  
✅ Prediction using a trained Convolutional Neural Network (CNN)  
✅ Model built with TensorFlow  

## Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/LNikolov01/handwritten_digit_recognition.git
cd Handwritten_Digit_Recognition
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Application
```bash
python3 app.py
```
- Draw a digit (0-9) in the window that appears.
- Press 's' to classify the digit. (Or 'q' to quit)
- The program will print the recognized digit and confidence score.

## Project Structure
```
Handwritten_Digit_Recognition/
│── app.py               # Main application script
│── train_model.py       # Model training script
│── draw_digit.py        # Allows users to draw digits
│── process_image.py     # Image preprocessing for the model
│── requirements.txt     # Dependencies
│── cnn_model.h5         # Trained CNN model
│── README.md            # Project documentation
```

## Example Output
```
Predicted Digit: 3
Confidence: 98.75%
```

## Model Details
- **Architecture:** Convolutional Neural Network (CNN)
- **Dataset:** Trained on the MNIST dataset (28x28 grayscale images)

## Planned Future Improvements
- ✅ Web-based interface with Flask
- ✅ Deployment as a web app
- ✅ Model training script for custom datasets