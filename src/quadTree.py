import numpy as np
from PIL import Image
import time
import cv2
from pympler import asizeof
import warnings
import os
import platform
import math

def clear_terminal():
    # Verifica o sistema operacional e executa o comando correspondente
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')


# Inibe todos os warnings
warnings.filterwarnings('ignore')

class QuadtreeNode:
    def __init__(self, x, y, x_size, y_size, color=None):
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

def display_inverted_image_no_quadtree(image):
    """Inverte os valores de cada pixel da imagem binarizada."""
    inverted_image = np.where(image == 0, 255, 0)
    return inverted_image


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



def check_collision(node, x, y):
    """Verifica se um ponto (x, y) colide com um objeto dentro da Quadtree,
       aproveitando áreas homogêneas."""
    
    # Verifica se o ponto está dentro da área do nó
    if not (node.x <= x < node.x + node.x_size and node.y <= y < node.y + node.y_size):
        return False
    # Se o nó for homogêneo, retornamos com base na cor
    if node.color == 0:  # Objeto presente (preto)
        return True
    elif node.color == 255:  # Espaço vazio (branco)
        return False

    # Caso não seja homogêneo, percorremos os filhos
    for child in node.children:
        if child is not None:
            if check_collision(child, x, y):
                return True
    return False

def check_collision_direct(image, x, y):
    """Verifica diretamente se o pixel (x, y) colide com um objeto."""
    return image[int(y), int(x)] == 0  # 0 significa colisão (obstáculo)


def shoot(quadtree, start_x, start_y, angle, image_size):   
    """Simula um tiro em uma direção e verifica se colide com um objeto na Quadtree."""
    x, y = start_x, start_y
    delta_x = 1 * math.cos(angle)
    delta_y = 1 * math.sin(angle)
    height = image_size[0]
    width =  image_size[1]
    
    while 0 <= x < height and 0 <= y < width:
        # Encontrar o nó quadtree onde o tiro está
        node = find_quadtree_node(quadtree, int(x), int(y))

        if node is None:
            print("Nenhuma colisão detectada.")
            return None  # Nenhuma colisão
        
        # Se o nó for homogêneo
        if node.color == 0:  # Objeto presente (preto)
            print(f"Colisão detectada em ({int(x)}, {int(y)})")
            return (int(x), int(y))  # Ponto da colisão

        # Calcular a próxima borda para pular áreas homogêneas
        x_next, y_next = calculate_next_boundary(node, x, y, delta_x, delta_y)

        # Atualiza a posição do tiro
        x, y = x_next, y_next

    print("Nenhuma colisão detectada.")
    return None  # Nenhuma colisão


def find_quadtree_node(node, x, y):
    """Encontra o nó da quadtree que contém o ponto (x, y)."""
    # Verifica se o ponto está dentro da área do nó
    if not (node.x <= x < node.x + node.x_size and node.y <= y < node.y + node.y_size):
        return None

    # Se o nó é homogêneo, retornamos esse nó
    if node.color is not None:
        return node

    # Caso contrário, percorremos os filhos
    for child in node.children:
        if child is not None:
            result = find_quadtree_node(child, x, y)
            if result:
                return result
    return None


def calculate_next_boundary(node, x, y, delta_x, delta_y):
    """Calcula o próximo ponto de saída ao longo do ângulo delta_x, delta_y."""
    # Calcula a distância até as bordas do nó
    t_x = (node.x + node.x_size - x) / delta_x if delta_x > 0 else (node.x - x) / delta_x
    t_y = (node.y + node.y_size - y) / delta_y if delta_y > 0 else (node.y - y) / delta_y

    # Define o menor tempo para alcançar a borda
    t = min(t_x, t_y)

    # Calcula a próxima posição
    x_next = x + delta_x * t
    y_next = y + delta_y * t

    return x_next, y_next



def shoot_direct(image, start_x, start_y, angle):
    """Simula um tiro em uma direção fixa e verifica colisão sem usar Quadtree."""
    x, y = start_x, start_y
    delta_x = math.cos(angle) * 1
    delta_y = math.sin(angle) * 1
    height, width = image.shape

    while 0 <= x < width and 0 <= y < height:
        # Verifica se houve colisão
        if check_collision_direct(image, int(x), int(y)):
            print(f"Colisão detectada em ({int(x)}, {int(y)})")
            return (int(x), int(y))  # Retorna o ponto de colisão
        # Atualiza a posição do tiro
        x += delta_x
        y += delta_y

    print("Nenhuma colisão detectada.")
    return None  # Nenhuma colisão






option = -1
option_process = -1
show_image = 0

while(option != 0):
    option_process=-1
    # Carregar a imagem
    print("=-"*10 + "Qual imagem você deseja processar" + "=-"*10)
    print("1 - Homogênea")
    print("2 - Heterogênea")
    print("3 - Intermediaria")
    print("=-"*36)
    while(option<0 or option>3):
        option = int(input("Selecione uma opção pelo número \n"))
    if option == 1:
        image_path = './images/imagem_homogênea.jpg'
    elif option == 2:
        image_path = './images/imagem_heterogenea.png'
    else:
        image_path = './images/imagem_intermediaria.jpg'
    
    image = Image.open(image_path).convert('L')
    input("Pressione enter para construir a quadtree")
    image_np = np.array(image)
    # Binariza a imagem (apenas valores 0 e 255)
    binary_image_np = np.where(image_np > 128, 255, 0).astype(np.uint8)

    # Constroi a quadtree
    initial_time = time.time()
    quadtree = build_quadtree(binary_image_np, 0, 0, binary_image_np.shape[1], binary_image_np.shape[0])
    final_time = time.time()
    print(f"Tempo de construção da quadtree: {final_time - initial_time :.3f} s")
    input("Pressione enter para ver as opções de processamento")


    
    # Cria uma imagem de saída para visualizar a quadtree
    output_image_np = np.zeros_like(binary_image_np)

    while(option_process!=5):
        # Executa o processamento
        clear_terminal()
        option_process = -1
        print("Qual processamento deseja fazer?:\n")
        print("1 - Inverter imagem")
        print("2 - Detectar borda")
        print("3 - Simular tiro")
        print("4 - Mostrar imagem atual")
        print("5 - Mudar imagem")
        while(option_process<1 or option_process>5):
            option_process = int(input("Escolha uma opção pelo número\n"))
        
        if(option_process == 1):
            show_image = 1
            initial_time = time.time()
            display_inverted_image(quadtree, output_image_np)
            final_time = time.time()
            print(f"Tempo de processamento com quadtree: {final_time - initial_time :.10f} s\n")
            initial_time = time.time()
            display_inverted_image_no_quadtree(output_image_np)
            final_time = time.time()
            print(f"Tempo de processamento direto: {final_time - initial_time :.10f} s\n")
        elif(option_process == 2):
            show_image = 1
            initial_time = time.time()
            display_edge(quadtree,output_image_np)
            final_time = time.time()
            print(f"Tempo de processamento com quadtree: {final_time - initial_time :.10f} s\n")
            initial_time = time.time()
            # Usa o algoritmo Canny para detectar bordas
            edges = cv2.Canny(binary_image_np, 100, 200)
            # Cria um kernel para a operação de dilatação
            kernel = np.ones((10, 10), np.uint8) 
            # Aplica a dilatação para aumentar a espessura das bordas
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            dilated_edges_image = Image.fromarray(dilated_edges)
            final_time = time.time()
            print(f"Tempo de processamento direto: {final_time - initial_time :.10f} s\n")
        elif option_process == 3:
            # Simulação do tiro
            show_image = 0
            start_x = int(input("Informe a coordenada X inicial do tiro: "))
            start_y = int(input("Informe a coordenada Y inicial do tiro: "))
            angle = float(input("Informe o ângulo de disparo (em graus): "))
            angle = math.radians(angle)  # Converter para radianos
            size = binary_image_np.shape
            initial_time = time.time()
            # Simula o tiro e verifica a colisão
            collision_point = shoot(quadtree, start_x, start_y, angle, size)
            final_time = time.time()

            print(f"Tempo de processamento quadtree: {final_time - initial_time :.10f} s\n")

            initial_time = time.time()
            collision_point = shoot_direct(binary_image_np, start_x, start_y, angle)
            final_time = time.time()
            print(f"Tempo de processamento matriz: {final_time - initial_time :.10f} s\n")
        elif option_process == 4:
            image.show()
        # Converte de matriz para imagem da PIL
        output_image = Image.fromarray(output_image_np.astype(np.uint8))
        # Imprime o tempo todal de execução

        if (option_process<4):
            
            # Imprime a comparação do uso de memória
            print("=-"*15 + " Imagem " + "=-"*15)
            print(f"Dimensão da imagem: {output_image_np.shape}")
            print(f"Tamanho em bytes: {asizeof.asizeof(output_image_np)}")
            print(f"Tamanho em megabytes: {asizeof.asizeof(output_image_np) / 2**20:.1f} MB\n")

            print("=-"*15 + " QuadTree " + "=-"*15)
            quadtree_size = asizeof.asizeof(quadtree)
            print(f"Tamanho em bytes: {quadtree_size}")
            print(f"Tamanho em megabytes: {quadtree_size / 2**20:.1f} MB\n")

        if (show_image):
            show_image = 0
            output_image.show()
        # Mostrar imagem
        input("Pressione enter para prosseguir")
        clear_terminal()
    option = -1