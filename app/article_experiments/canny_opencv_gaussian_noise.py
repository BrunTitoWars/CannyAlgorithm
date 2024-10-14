import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Gaussian noise parameters
mean = 0    
std_dev = 10   


def salt_and_pepper_noise(image, salt_prob=0.005,pepper_prob=0.005):
        noisy_image = np.copy(image)
        random_matrix = np.random.rand(*image.shape)
        noisy_image[random_matrix < salt_prob] = 255
        noisy_image[random_matrix > 1 - pepper_prob] = 0
        return noisy_image
# Function to add Gaussian noise to an image

def add_gaussian_noise(image, mean, std_dev):
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Load the image
img = cv.imread('../../assets/girl_in_beach.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check the file path."

# Add Gaussian noise to the image
noisy_img = salt_and_pepper_noise(img)

# Define threshold pairs for Canny edge detection
thresholds = [(50, 60), (50, 100), (50, 150), (50, 255)]
edges_images = []

# Apply Canny edge detection with different thresholds
for low, high in thresholds:
    edges = cv.Canny(noisy_img, low, high)
    edges_images.append(edges)

# Plot the edge-detected images for different thresholds
plt.figure(figsize=(10, 7))

for i, (low_threshold, high_threshold) in enumerate(thresholds):
    plt.subplot(2, 2, i + 1)
    plt.imshow(edges_images[i], cmap='gray')
    plt.title(f'Canny: {low_threshold}-{high_threshold}')
    plt.xticks([]), plt.yticks([])

# Save the figure as open_cv_canny.png
plt.tight_layout()
plt.savefig('../../assets/open_cv_canny.png')
plt.show()
