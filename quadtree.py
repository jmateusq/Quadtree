import numpy as np
from PIL import Image
import time
from pympler import asizeof

class QuadtreeNode:
    def _init_(self, x, y, x_size, y_size, color=None):
        self.x = x
        self.y = y
        self.x_size = x_size
        self.y_size = y_size
        self.color = color
        self.children = (None, None, None, None)

def mean_area(image, x, y, x_size, y_size):
    sub_image = image[y:y + y_size, x:x + x_size]
    return np.mean(sub_image)

def build_quadtree(image, x, y, x_size, y_size):
    # Se alguma dimensão é 0, logo o nó é nulo.
    if x_size < 1 or y_size < 1:
        return None

    mean = mean_area(image, x, y, x_size, y_size)
    color = 255 if mean > 127 else 0

    # Define o tamanho mínimo que os nós podem ter
    if x_size <= 1 and y_size <= 1:
        return QuadtreeNode(x, y, x_size, y_size, color)

    # Define se o nó é homogêneo
    if mean == 255 or mean == 0:
        return QuadtreeNode(x, y, x_size, y_size, color)

    x_half_size = x_size // 2
    y_half_size = y_size // 2

    node = QuadtreeNode(x, y, x_size, y_size)
    
    node.children = (
        build_quadtree(image, x, y, x_half_size, y_half_size),
        build_quadtree(image, x + x_half_size, y, x_size - x_half_size, y_half_size),
        build_quadtree(image, x, y + y_half_size, x_half_size, y_size - y_half_size),
        build_quadtree(image, x + x_half_size, y + y_half_size, x_size - x_half_size, y_size - y_half_size)
    )
    return node

def display_inverted_image(QuadtreeNode, image):
    if QuadtreeNode is not None:
        if QuadtreeNode.color is not None:
            if QuadtreeNode.color == 255:
                image[QuadtreeNode.y:QuadtreeNode.y+QuadtreeNode.y_size, QuadtreeNode.x:QuadtreeNode.x+QuadtreeNode.x_size] = 0
            else:
                image[QuadtreeNode.y:QuadtreeNode.y+QuadtreeNode.y_size, QuadtreeNode.x:QuadtreeNode.x+QuadtreeNode.x_size] = 255
        else:
            for child in QuadtreeNode.children:
                display_inverted_image(child, image)

def display_edge(QuadtreeNode, image):
    if QuadtreeNode is not None:
        if QuadtreeNode.color is not None:
            if QuadtreeNode.y_size < 30 and QuadtreeNode.x_size < 30:
                image[QuadtreeNode.y:QuadtreeNode.y+QuadtreeNode.y_size, QuadtreeNode.x:QuadtreeNode.x+QuadtreeNode.x_size] = 0
            else:
                image[QuadtreeNode.y:QuadtreeNode.y+QuadtreeNode.y_size, QuadtreeNode.x:QuadtreeNode.x+QuadtreeNode.x_size] = 255
        else:
            for child in QuadtreeNode.children:
                display_edge(child, image)

# Carregar a imagem
image_path = '/home/allan/Desktop/QuadTree/New Project (2) (1).jpg'
image = Image.open(image_path).convert('L')
image_np = np.array(image)

# Binariza a imagem (apenas valores 0 e 255)
binary_image_np = np.where(image_np > 128, 255, 0).astype(np.uint8)

# Constroi a quadtree
quadtree = build_quadtree(binary_image_np, 0, 0, binary_image_np.shape[1], binary_image_np.shape[0])

# Marca o tempo antes de fazer o processamento
initial_time = time.time()

# Cria uma imagem de saída para visualizar a quadtree
output_image_np = np.zeros_like(binary_image_np)

# Executa o processamento
display_inverted_image(quadtree, output_image_np)

# Converte de matriz para imagem da PIL
output_image = Image.fromarray(output_image_np.astype(np.uint8))

# Marca o tempo depois de fazer o processamento
final_time = time.time()

# Imprime o tempo todal de execução
print(f"Tempo de processamento: {final_time - initial_time :.3f} s\n")

# Imprime a comparação do uso de memória
print("=-"*15 + " Imagem " + "=-"*15)
print(f"Dimensão da imagem: {output_image_np.shape}")
print(f"Tamanho em bytes: {asizeof.asizeof(output_image_np)}")
print(f"Tamanho em megabytes: {asizeof.asizeof(output_image_np) / 2**20:.1f} MB\n")

print("=-"*15 + " QuadTree " + "=-"*15)
quadtree_size = asizeof.asizeof(quadtree)
print(f"Tamanho em bytes: {quadtree_size}")
print(f"Tamanho em gigabytes: {quadtree_size / 2**20:.1f} MB\n")

# Mostrar imagem
output_image.show()
