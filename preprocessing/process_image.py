import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess an image for digit recognition.
    - Converts to grayscale
    - Resizes to 28x28
    - Inverts colors (white text on black)
    - Normalizes pixel values
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)  # Invert colors if background is white
    img = img / 255.0  # Normalize pixel values (0-1)
    img = img.reshape(1, 28, 28, 1)  # Reshape for CNN model

    return img