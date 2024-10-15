import numpy as np
import cv2
from matplotlib import pyplot as plt

class ImprovedCanny:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    def detect_edges(self):
        noisy_img = self.salt_and_pepper_noise(self.image)
        median_blur = self.median_filter(noisy_img)
        gradient_magnitude, gradient_direction = self.calculate_gradients(median_blur)
        nms = self.non_maximum_suppression(gradient_magnitude, gradient_direction)

        thresholds = [(50, 60), (50, 100), (50, 150), (50, 255)]
        edge_images = []

        for low_threshold, high_threshold in thresholds:
            thresholded = self.double_threshold(nms, low_threshold, high_threshold)
            edges = self.edge_tracking(thresholded)
            edges = np.clip(edges, 0, 255)
            edges_uint8 = np.uint8(edges)
            edge_images.append(edges_uint8)

        plt.figure(figsize=(10, 7))


        for i, (low_threshold, high_threshold) in enumerate(thresholds):
            plt.subplot(2, 2, i + 1)  
            plt.imshow(edge_images[i], cmap='gray')
            plt.title(f'Thresholds: {low_threshold}-{high_threshold}')
            plt.xticks([]), plt.yticks([])

        
        plt.tight_layout()
        plt.savefig('../../assets/thresholds_improved_canny.png')
        plt.show()

    def gaussian_filter(self,image, kernel_size=5, sigma=1):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def median_filter(self,image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)


    def calculate_gradients(self,image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        return magnitude, direction


    def non_maximum_suppression(self,gradient_magnitude, gradient_direction):
        angle = np.rad2deg(gradient_direction)
        angle[angle < 0] += 180


        nms = np.zeros_like(gradient_magnitude)

        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                try:
                    q = 255
                    r = 255

                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = gradient_magnitude[i, j + 1]
                        r = gradient_magnitude[i, j - 1]
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = gradient_magnitude[i + 1, j - 1]
                        r = gradient_magnitude[i - 1, j + 1]
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = gradient_magnitude[i + 1, j]
                        r = gradient_magnitude[i - 1, j]
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = gradient_magnitude[i - 1, j - 1]
                        r = gradient_magnitude[i + 1, j + 1]

                    if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                        nms[i, j] = gradient_magnitude[i, j]
                    else:
                        nms[i, j] = 0

                except IndexError as e:
                    pass

        return nms

    def double_threshold(self,image, low_threshold, high_threshold):
        res = np.zeros_like(image)
        strong = 255
        weak = 50

        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def salt_and_pepper_noise(self,image, salt_prob=0.005,pepper_prob=0.005):
        noisy_image = np.copy(image)
        random_matrix = np.random.rand(*image.shape)
        noisy_image[random_matrix < salt_prob] = 255
        noisy_image[random_matrix > 1 - pepper_prob] = 0
        return noisy_image


    def edge_tracking(self,image, weak=50, strong=255):
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                if image[i, j] == weak:
                    if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                            or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
        return image


    def gaussian_noise(image, mean=0, sigma=10):
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255)  
        return noisy_image.astype(np.uint8)


img = cv2.imread('assets/girl_in_beach.jpg', cv2.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

canny_operator = ImprovedCanny(image_path='../../assets/girl_in_beach.jpg')
canny_operator.detect_edges()
# Mostrar resultados
#cv2.imshow('Imagem Original', img)
#cv2.imshow('Imagem com Ruído', noisy_img)
#cv2.imshow('Imagem Mediana', np.uint8(median_blur))
#cv2.imshow('Non-Maximum Suppression', np.uint8(nms / nms.max() * 255))
#cv2.imshow('Double Threshold', np.uint8(thresholded))
#cv2.imshow('Canny - Implementação Manual', edges_uint8)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

