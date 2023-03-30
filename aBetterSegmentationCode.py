import cv2
import numpy as np

# Load the image
img = cv2.imread('call.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert the image to binary
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform connected component analysis to separate the characters
output = cv2.connectedComponentsWithStats(thresh, connectivity=8)

# Get the number of characters (excluding the background)
num_chars = output[0] - 1

# Define a list to store individual characters
chars = []

# Define a list to store the coordinates of each character
coords = []

# Loop through each character
for i in range(num_chars):
    # Extract the character by cropping the image
    char = output[1] == i + 1
    char = char.astype(np.uint8) * 255

    # Store the character and its coordinates
    chars.append(char)
    coords.append(output[2][i+1][:2])

# Sort the characters based on their x-coordinates (from right to left)
chars = [char for _, char in sorted(zip(coords, chars), key=lambda x: x[0][0], reverse=True)]
# Display the segmented characters
for i, char in enumerate(chars):
    # Display the character
    cv2.imshow(f'Char {i + 1}', char)
    cv2.waitKey(0)

cv2.destroyAllWindows()
