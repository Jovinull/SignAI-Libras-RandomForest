import os
import pickle

import mediapipe as mp
import cv2

# Inicialização do MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

# Diretório onde os dados estão armazenados
DATA_DIR = './data'

# Listas para armazenar os dados e as labels
data = []
labels = []

# A função os.listdir() retorna uma lista com os nomes de todos os arquivos e diretórios presentes
# no diretório especificado. É útil quando precisamos iterar sobre todos os arquivos em um
# determinado diretório, como no caso deste código, onde precisamos processar todas as imagens
# em um diretório específico para extrair características das mãos.
for dir_ in os.listdir(DATA_DIR):
    # Itera sobre cada imagem dentro do diretório da classe atual (dir_)
    # dentro do diretório principal de dados (DATA_DIR), utilizando a função
    # os.listdir() para listar todos os arquivos e diretórios dentro do caminho
    # formado por os.path.join(DATA_DIR, dir_).
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Lista temporária para armazenar as coordenadas normalizadas das landmarks
        x_ = []        # Lista temporária para armazenar as coordenadas x das landmarks
        y_ = []        # Lista temporária para armazenar as coordenadas y das landmarks

        # Converte a imagem da representação BGR (Azul, Verde, Vermelho) para RGB (Vermelho, Verde, Azul).
        # Isso é feito usando a função cv2.cvtColor() do OpenCV, onde cv2.COLOR_BGR2RGB especifica a conversão
        # de BGR para RGB. Essa conversão é útil quando precisamos exibir ou processar a imagem em bibliotecas
        # que esperam o formato de cor RGB, como Matplotlib.
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processa a imagem com o MediaPipe Hands
        results = hands.process(img_rgb)
        
        # Verifica se há landmarks das mãos detectadas
        if results.multi_hand_landmarks:
            # Verifica se há apenas uma mão detectada
            if len(results.multi_hand_landmarks) >= 1:
                # Acessa as landmarks da primeira mão detectada na imagem.
                # A variável results.multi_hand_landmarks é uma lista onde cada elemento representa as landmarks
                # detectadas para uma mão na imagem. Ao acessar o elemento zero dessa lista (results.multi_hand_landmarks[0]),
                # estamos selecionando as landmarks da primeira mão detectada na imagem. Se houver mais de uma mão na imagem
                # e você estiver interessado em todas elas, pode iterar sobre results.multi_hand_landmarks para acessar as
                # landmarks de cada mão individualmente.
                hand_landmarks = results.multi_hand_landmarks[0]

                # Verifica se todas as 21 landmarks estão presentes
                if len(hand_landmarks.landmark) == 21:
                    # Extrai as coordenadas x e y das landmarks
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    # Normaliza as coordenadas em relação ao canto superior esquerdo do retângulo delimitador
                    min_x = min(x_)
                    min_y = min(y_)
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min_x)
                        data_aux.append(landmark.y - min_y)

                    # Adiciona os dados normalizados e a label à lista correspondente
                    data.append(data_aux)
                    labels.append(dir_)
                    
# Salva os dados e as labels em um arquivo pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

