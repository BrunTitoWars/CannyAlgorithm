from __future__ import print_function
from __future__ import division
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Carregar a imagem em escala de cinza
img = cv.imread('../assets/fruits.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Aplicar detecção de bordas Canny
edges = cv.Canny(img, 100, 200)

# Configurar o subplot para as duas imagens
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# Mostrar a imagem
#plt.show()

# Salvar a imagem ao invés de mostrar
plt.tight_layout()
plt.savefig('../output/canny_opencv/fruits_edges.png')  # Salvar na pasta 'output'
plt.close()  # Fechar a figura
