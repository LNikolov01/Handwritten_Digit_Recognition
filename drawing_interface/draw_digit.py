import cv2
import numpy as np
import os

drawing = False
last_x, last_y = None, None
canvas = np.ones((400, 400), dtype="uint8") * 255  # White canvas
save_path = "test_images/drawn_digit.png"

def draw(event, x, y, flags, param):
    global drawing, last_x, last_y, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and last_x is not None and last_y is not None:
            cv2.line(canvas, (last_x, last_y), (x, y), (0, 0, 0), thickness=15)  # Draw smoother lines
            last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_x, last_y = None, None

cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", draw)

while True:
    cv2.imshow("Draw a Digit", canvas)
    key = cv2.waitKey(1)
    
    if key == ord("s"):  # Press 's' to save
        if not os.path.exists("test_images"):
            os.makedirs("test_images")
        cv2.imwrite(save_path, canvas)
        print(f"✅ Image saved to {save_path}")
        break
    elif key == ord("q"):  # Press 'q' to quit
        break

cv2.destroyAllWindows()