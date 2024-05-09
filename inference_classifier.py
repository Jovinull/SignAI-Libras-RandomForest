import pickle
import cv2
import mediapipe as mp
import numpy as np

# Carrega o modelo treinado
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicialização do Hands:
# - static_image_mode=False para detecção em tempo real de mãos em vídeo.
# - min_detection_confidence=0.7 para garantir que as detecções tenham alta confiança.
# - min_tracking_confidence=0.7 para garantir que o rastreamento das mãos tenha alta confiança.
# - max_num_hands=2 para permitir a detecção de até duas mãos simultaneamente.
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

# Dicionário de rótulos
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'Ç', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K',
               12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V',
               23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

# Função para pré-processar os dados da mão
def preprocess_hand_data(hand_landmarks):
    """
    Função para pré-processar os dados das landmarks da mão.
    Normaliza as coordenadas das landmarks em relação ao canto superior esquerdo do retângulo delimitador.
    """
    data_aux = []
    x_ = []
    y_ = []

    # Extrai as coordenadas x e y das landmarks
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)

    # Calcula as coordenadas mínimas em x e y
    min_x = min(x_)
    min_y = min(y_)

    # Normaliza as coordenadas das landmarks em relação ao canto superior esquerdo do retângulo delimitador
    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min_x)
        data_aux.append(y - min_y)

    return data_aux

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Inverte o frame horizontalmente para corresponder ao espelho da câmera
    frame = cv2.flip(frame, 1)

    # Processa a imagem com o MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Pré-processa os dados da mão
            data_aux = preprocess_hand_data(hand_landmarks)

            # Realiza a previsão com o modelo
            prediction = model.predict([np.asarray(data_aux)])

            # Obtém o rótulo previsto
            predicted_character = labels_dict[int(prediction[0])]

            # Desenha o quadrado delimitador em volta da mão
            brect = cv2.boundingRect(np.array([[(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in hand_landmarks.landmark]]))
            cv2.rectangle(frame, (brect[0], brect[1]), (brect[0]+brect[2], brect[1]+brect[3]), (0, 255, 0), 2)

            # Desenha as landmarks da mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Exibe o rótulo previsto
            cv2.putText(frame, predicted_character, (brect[0], brect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibe o frame
    cv2.imshow('Frame', frame)
    
    # Condição de saída do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
