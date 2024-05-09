import os
import cv2

# Diretório onde os dados serão armazenados
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Número de classes (categorias) e tamanho do conjunto de dados para cada classe
number_of_classes = 27
dataset_size = 200

# Inicialização da captura de vídeo da webcam
# Certifique-se de que essa seja a sua webcam (0 -> Webcam Nativa)
# Caso tenha mais de uma vá testando até encontrar a que deseja
cap = cv2.VideoCapture(0)

# Loop para cada classe
for j in range(number_of_classes):
    # Criar um diretório para armazenar os dados da classe, se não existir
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Loop para coletar dados para a classe atual
    done = False
    while True:
        # Capturar um quadro da webcam
        ret, frame = cap.read()
        
        # Adicionar texto de instrução ao quadro
        cv2.putText(frame, 'Pronto? Pressione "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Mostrar o quadro com a instrução na tela
        cv2.imshow('frame', frame)
        
        # Aguardar pressionamento da tecla 'Q' para continuar
        if cv2.waitKey(25) == ord('q'):
            break

    # Contador para o número de imagens capturadas
    counter = 0
    while counter < dataset_size:
        # Capturar um quadro da webcam
        ret, frame = cap.read()
        
        # Mostrar o quadro na tela
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Salvar o quadro como uma imagem no diretório correspondente à classe
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        # Incrementar o contador de imagens capturadas
        counter += 1

# Liberar os recursos da câmera e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()
