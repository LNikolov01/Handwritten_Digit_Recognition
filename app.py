import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.process_image import preprocess_image

# Load the trained model
model = load_model("models/cnn_model.h5")

# Predict function with confidence filtering
def predict_digit(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0]  # Get prediction array
    top_3_indices = np.argsort(predictions)[-3:][::-1]  # Get top 3 predictions
    top_3_confidences = predictions[top_3_indices] * 100  # Convert to percentages
    
    # Keep the main prediction and any additional predictions with confidence > 5%
    filtered_predictions = [(top_3_indices[0], top_3_confidences[0])]
    for i in range(1, len(top_3_indices)):
        if top_3_confidences[i] > 5:
            filtered_predictions.append((top_3_indices[i], top_3_confidences[i]))
    
    return filtered_predictions

image_path = "drawing_interface/test_images/drawn_digit.png"
predictions = predict_digit(image_path)

print("Predictions:")
for rank, (digit, confidence) in enumerate(predictions, 1):
    print(f"{rank}. {digit} ({confidence:.2f}%)")