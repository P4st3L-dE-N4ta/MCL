import numpy as np
import matplotlib.pyplot as plt
import yaml

def carregar_pgm(filename):
    """Lê um arquivo PGM e retorna uma matriz numpy com os valores"""
    with open(filename, 'rb') as f:
        header = f.readline()
        if header.strip() != b'P5':
            raise ValueError("Arquivo não é PGM binário P5")

        # pula comentários
        while True:
            line = f.readline()
            if line.startswith(b'#'):
                continue
            else:
                break

        # lê dimensões
        width, height = [int(i) for i in line.split()]
        maxval = int(f.readline())

        # lê dados da imagem
        data = np.fromfile(f, dtype=np.uint8 if maxval < 256 else np.uint16, count=width*height)
        data = data.reshape((height, width))

    return data

def carregar_yaml(filename):
    """Lê o arquivo YAML e retorna as infos do mapa"""
    with open(filename, 'r') as f:
        map_info = yaml.safe_load(f)
    return map_info

def mostrar_mapa_e_area_util(pgm_file, yaml_file, threshold=200):
    # Carrega o mapa
    map_data = carregar_pgm(pgm_file)
    map_info = carregar_yaml(yaml_file)

    print("Informações do mapa carregado:")
    for k, v in map_info.items():
        print(f"{k}: {v}")

    # Áreas consideradas livres: pixels com valor maior que threshold
    area_util = map_data > threshold

    # Mostrar mapa completo
    plt.figure(figsize=(10,10))
    plt.imshow(map_data, cmap='gray')
    plt.title("Mapa Completo (tons de cinza)")
    plt.colorbar(label='Valor do pixel')
    plt.show()

    # Mostrar área útil
    plt.figure(figsize=(10,10))
    plt.imshow(area_util, cmap='Greens')
    plt.title(f"Área útil (pixels > {threshold})")
    plt.show()

if __name__ == "__main__":
    # Exemplos - altera para os teus ficheiros
    pgm_file = "lab.pgm"
    yaml_file = "lab.yaml"
    mostrar_mapa_e_area_util(pgm_file, yaml_file, threshold=250)
