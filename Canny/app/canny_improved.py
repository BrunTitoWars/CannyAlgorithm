import numpy as np
import cv2

# 1. Função para filtro gaussiano
def gaussian_filter(image, kernel_size=5, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# 2. Função para calcular o valor do gradiente e direção
def calculate_gradients(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction

# 3. Função para Supressão Não Máxima (NMS)
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    # Convertendo a direção do gradiente para graus
    angle = np.rad2deg(gradient_direction)
    angle[angle < 0] += 180

    # Inicializando a imagem de supressão com zeros
    nms = np.zeros_like(gradient_magnitude)

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            try:
                q = 255
                r = 255

                # Verificar direção do gradiente
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

                # Supressão não máxima
                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    nms[i, j] = gradient_magnitude[i, j]
                else:
                    nms[i, j] = 0

            except IndexError as e:
                pass

    return nms

# 4. Função de Double Threshold
def double_threshold(image, low_threshold, high_threshold):
    res = np.zeros_like(image)
    strong = 255
    weak = 50

    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res

# 5. Função de rastreamento de bordas
def edge_tracking(image, weak=50, strong=255):
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

# 6. Função de aplicação de ruído Salt and Pepper
def salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Cria uma cópia da imagem original
    noisy_image = np.copy(image)

    # Gera um array aleatório do mesmo tamanho da imagem
    random_matrix = np.random.rand(*image.shape)

    # Define os pixels como 255 (sal) com base na probabilidade
    noisy_image[random_matrix < salt_prob] = 255

    # Define os pixels como 0 (pimenta) com base na probabilidade
    noisy_image[random_matrix > 1 - pepper_prob] = 0

    return noisy_image

# Carregar imagem
img = cv2.imread('../assets/fruits.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

# 1. Aplicar filtro gaussiano
gaussian_blur = gaussian_filter(img)

# 2. Calcular gradiente e direção
gradient_magnitude, gradient_direction = calculate_gradients(gaussian_blur)

# 3. Aplicar supressão não máxima
nms = non_maximum_suppression(gradient_magnitude, gradient_direction)

# 4. Aplicar double threshold
thresholded = double_threshold(nms, 50, 100)

# 5. Rastreamento de bordas
edges = edge_tracking(thresholded)

# Normalizar os valores antes de converter para uint8
edges = np.clip(edges, 0, 255)
edges_uint8 = np.uint8(edges)

# Mostrar resultados de cada etapa
#cv2.imshow('Imagem Original', img)
#cv2.imshow('Imagem Gaussian Blur', np.uint8(gaussian_blur))
#cv2.imshow('Gradiente Magnitude', np.uint8(gradient_magnitude / gradient_magnitude.max() * 255))
#cv2.imshow('Non-Maximum Suppression', np.uint8(nms / nms.max() * 255))
#cv2.imshow('Double Threshold', np.uint8(thresholded))
#cv2.imshow('Canny - Implementação Manual', edges_uint8)

# Aplicar ruído Salt and Pepper
salt_prob = 0.02  # Probabilidade do ruído "sal" (branco)
pepper_prob = 0.02  # Probabilidade do ruído "pimenta" (preto)
noisy_img = salt_and_pepper_noise(edges_uint8, salt_prob, pepper_prob)

# Mostrar imagem original e com ruído
#cv2.imshow('Imagem com Salt and Pepper', noisy_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Substitua os cv2.imshow() por cv2.imwrite()
cv2.imwrite('../output/original_image.png', img)
cv2.imwrite('../output/gaussian_blur.png', np.uint8(gaussian_blur))
cv2.imwrite('../output/gradient_magnitude.png', np.uint8(gradient_magnitude / gradient_magnitude.max() * 255))
cv2.imwrite('../output/nms.png', np.uint8(nms / nms.max() * 255))
cv2.imwrite('../output/thresholded.png', np.uint8(thresholded))
cv2.imwrite('../output/edges.png', edges_uint8)
cv2.imwrite('../output/noisy_image.png', noisy_img)
