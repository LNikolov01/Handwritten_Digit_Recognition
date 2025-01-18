import subprocess
import cv2
import numpy as np
import tensorflow as tf
from preprocessing.process_image import preprocess_image

MODEL_PATH = "models/cnn_model.h5"

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_digit(image_path, model):
    # Preprocess the image
    processed_image = preprocess_image(image_path)  # Should return a NumPy array

    # Get prediction probabilities
    predictions = model.predict(processed_image)  # Returns an array of probabilities for each digit (0-9)

    predicted_digit = np.argmax(predictions)  # Get the digit with the highest probability
    confidence = np.max(predictions)  # Get the confidence for the prediction

    return predicted_digit, confidence

if __name__ == "__main__":
    print("Launching digit drawing interface...")
    
    # Run draw_digit.py as a subprocess
    subprocess.run(["python3", "drawing_interface/draw_digit.py"])
    
    IMAGE_PATH = "test_images/drawn_digit.png"  # Ensure the image is saved as "drawn_digit.png"
    
    # Load model
    model = load_model()

    # Predict the digit
    predicted_digit, confidence = predict_digit(IMAGE_PATH, model)

    # Display results
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence * 100:.2f}%")  # Converted to percentage format