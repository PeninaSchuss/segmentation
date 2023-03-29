import cv2
import numpy as np

# Load the image
img = cv2.imread('hello.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert the image to binary
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a list to store individual characters
chars = []

# Loop through the contours
for contour in contours:
    # Get the bounding rectangle of each contour
    x, y, w, h = cv2.boundingRect(contour)

    # Ignore small contours
    if w < 10 or h < 10:
        continue

    # Extract the character by cropping the image
    char = thresh[y:y + h, x:x + w]

    # Add the character to the list
    chars.append((x, char))

# Sort the characters by their x-coordinate in descending order
chars = sorted(chars, key=lambda x: x[0], reverse=True)

# Display the segmented characters
for i, (x, char) in enumerate(chars):
    # Flip the character horizontally to correct the orientation
   # char = cv2.flip(char, 1)

    # Display the character
    cv2.imshow(f'Char {i + 1}', char)
    cv2.waitKey(0)

cv2.destroyAllWindows()
