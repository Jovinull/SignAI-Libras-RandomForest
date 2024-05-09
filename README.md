# Projeto de Reconhecimento de Sinais de Libras com Random Forest Classifier

Este projeto tinha como objetivo desenvolver um sistema de reconhecimento de sinais de Libras em tempo real usando técnicas de aprendizado de máquina e visão computacional. A ideia era utilizar a biblioteca MediaPipe Hands para detectar e rastrear as mãos em imagens de vídeo da webcam, extrair características das mãos detectadas e alimentar essas características em um modelo de classificação, como Random Forest, para prever o sinal de Libras correspondente.

### Descontinuação do Projeto

Este projeto foi descontinuado devido a algumas limitações encontradas durante o desenvolvimento. A abordagem inicial de usar Random Forest para classificação mostrou-se inadequada para o problema em questão. As RandomForestClassifier não foram capazes de capturar adequadamente a complexidade e as nuances dos sinais de Libras, resultando em baixa precisão e desempenho insatisfatório.

### Recomeço

Decidi recomeçar o projeto, explorando outras técnicas e modelos de aprendizado de máquina mais adequados para o reconhecimento de sinais de Libras. Estou em busca de modelos mais sofisticados e métodos de pré-processamento de dados que possam capturar melhor as características dos sinais de Libras e melhorar a precisão do sistema de reconhecimento.

Estou comprometido em continuar trabalhando neste projeto e encontrar uma solução eficaz para o reconhecimento de sinais de Libras em tempo real. Estou aberto a explorar novas abordagens e colaborações para alcançar esse objetivo.

# Configuração do Ambiente

Para executar o projeto de interpretação de sinais de libras com o MediaPipe, é necessário configurar um ambiente Python e instalar as dependências listadas no arquivo `requirements.txt`. Também é importante ressaltar que a versão Python do projeto é a [3.10.11](https://www.python.org/downloads/release/python-31011/).

## Passo 1: Configurar uma Máquina Virtual (opcional, mas recomendado)

1. **Verificar a Versão do Python**:

   Para garantir que você está usando a versão correta do Python, você pode verificar qual versão está instalada em seu sistema. No terminal, digite:
   ```
   python --version
   ```
   Certifique-se de que a versão exibida corresponda à versão especificada no projeto. Se não corresponder, você pode precisar instalar a versão correta do Python.

2. **Criar uma Máquina Virtual (Virtual Environment)**:
   ```
   python -m venv env
   ```

3. **Ativar a Máquina Virtual**:
   - No Windows:
     ```
     venv\Scripts\activate
     ```
   - No macOS e Linux:
     ```
     source venv/bin/activate
     ```

## Passo 2: Instalar Dependências

1. **Clonar o Repositório (se ainda não estiver clonado)**:
   ```
   git clone <link>
   cd seu-repositorio
   ```

2. **Instalar as Dependências**:
   ```
   pip install -r requirements.txt
   ```

Após seguir esses passos, o ambiente estará configurado e pronto para executar o projeto.

# Funcionamento do Projeto

## Coleta de Dados para Treinamento do Modelo de Interpretação de Sinais de Libras

Este script Python foi projetado para coletar dados de entrada para treinar um modelo de interpretação de sinais de Libras. Ele captura imagens da webcam e as organiza em um conjunto de dados rotulado por classe.

### Utilização

1. **Executar o Script**: Execute o script Python em um ambiente adequado (por exemplo, utilizando o Python Virtual Environment).

2. **Coleta de Dados**: O script irá abrir a webcam e exibir a imagem ao vivo. Ele solicitará que você pressione a tecla "Q" quando estiver pronto para começar a coleta de dados para cada classe.

3. **Coleta de Dados por Classe**: Para cada classe especificada, o script captura várias imagens da webcam e as salva em um diretório correspondente.

4. **Finalização**: Após a coleta de dados para todas as classes, o script encerra a captura de vídeo e fecha todas as janelas.

### Parâmetros

- `DATA_DIR`: Diretório onde os dados serão armazenados.
- `number_of_classes`: Número total de classes (categorias).
- `dataset_size`: Número de imagens a serem coletadas para cada classe.
- `cap`: Objeto de captura de vídeo da webcam.

### Estrutura de Diretórios

O conjunto de dados será organizado no seguinte formato:
```
/data
    /0
        image1.jpg
        image2.jpg
        ...
    /1
        image1.jpg
        image2.jpg
        ...
    /2
        image1.jpg
        image2.jpg
        ...
```

Cada subdiretório dentro de `/data` corresponde a uma classe específica, e as imagens capturadas são armazenadas dentro desses subdiretórios.

### Observações

- Certifique-se de possuir permissões de acesso para salvar arquivos no diretório especificado por `DATA_DIR`.
- As imagens são salvas no formato JPEG.

Claro, aqui está a documentação em Markdown para o código fornecido:

## Extração de Características dos Sinais de Libras usando MediaPipe Hands

Este script Python utiliza o MediaPipe Hands para extrair características das mãos detectadas em imagens de sinais de Libras. As características normalizadas são então armazenadas em um arquivo pickle para posterior treinamento de um modelo de interpretação de sinais de Libras.

### Utilização

1. **Preparação do Conjunto de Dados**: Certifique-se de ter um conjunto de imagens de sinais de Libras organizado em diretórios, onde cada diretório representa uma classe de sinal de Libras.

2. **Executar o Script**: Execute o script Python em um ambiente adequado.

3. **Extração de Características**: O script processará cada imagem do conjunto de dados, detectando as mãos e extraindo landmarks (pontos de referência) das mesmas.

4. **Armazenamento dos Dados**: As características normalizadas, juntamente com as labels correspondentes, serão armazenadas em um arquivo pickle para uso posterior no treinamento de um modelo de interpretação de sinais de Libras.

### Parâmetros

- `mp_hands`: Objeto MediaPipe Hands para detecção de mãos.
- `DATA_DIR`: Diretório onde os dados estão armazenados.
- `data`: Lista para armazenar as características normalizadas das mãos.
- `labels`: Lista para armazenar as labels correspondentes aos sinais de Libras.
- `hands`: Configuração do MediaPipe Hands.

### Estrutura do Arquivo Pickle

O arquivo pickle contém um dicionário com duas chaves:
- `data`: Lista de arrays numpy contendo as características normalizadas das mãos.
- `labels`: Lista de strings contendo as labels correspondentes aos sinais de Libras.

### Observações

- Certifique-se de possuir permissões de leitura e escrita nos diretórios de imagens e no diretório onde o arquivo pickle será salvo.
- O MediaPipe Hands pode não detectar mãos em todas as imagens, dependendo da qualidade e orientação das mesmas.

## Classificador de Mãos com Random Forest

Este é um script Python que carrega dados de landmarks de mãos, treina um classificador RandomForest e salva o modelo treinado em um arquivo.

### Random Forest

Random Forest é um algoritmo de aprendizado de máquina que combina múltiplas árvores de decisão durante o treinamento para realizar tarefas de classificação e regressão. É conhecido por sua capacidade de lidar com conjuntos de dados grandes e complexos, sua resistência ao overfitting e sua habilidade de fornecer insights sobre a importância das características para a classificação. Ajustar os hiperparâmetros, como o número de árvores na floresta e a profundidade máxima das árvores, é importante para obter o melhor desempenho possível no problema específico em questão.

### Uso

1. Certifique-se de ter os dados de entrada necessários no formato correto. Os dados de entrada devem estar armazenados em um arquivo `data.pickle` contendo dois elementos: `data`, que é uma matriz NumPy contendo os dados de entrada, e `labels`, que é uma matriz NumPy contendo as classes correspondentes.

2. Execute o script `train_classifier.py`:

```
python train_classifier.py
```

3. O modelo treinado será salvo em um arquivo `model.p` no diretório atual.

Aqui está a documentação em Markdown para o último trecho de código:

## Reconhecimento de Sinais de Libras em Tempo Real usando MediaPipe Hands e Modelo Treinado

Este script Python utiliza o MediaPipe Hands para detectar mãos em tempo real em um fluxo de vídeo da webcam. Em seguida, pré-processa as landmarks das mãos detectadas e as alimenta em um modelo treinado para prever o sinal de Libras correspondente.

### Pré-Requisitos

Certifique-se de ter as seguintes bibliotecas instaladas:

- `opencv-python`: para captura de vídeo da webcam e processamento de imagens.
- `mediapipe`: para detecção e rastreamento de mãos.
- `numpy`: para manipulação de arrays.

Você pode instalar as dependências executando o seguinte comando:
```bash
pip install opencv-python mediapipe numpy
```

### Utilização

1. **Executar o Script**: Execute o script Python em um ambiente com uma webcam disponível.

2. **Reconhecimento em Tempo Real**: O script abrirá a webcam e exibirá o fluxo de vídeo em tempo real. Ele detectará mãos na cena e exibirá o sinal de Libras correspondente ao sinal reconhecido.

3. **Concluir**: Pressione a tecla "Q" para encerrar o programa e fechar a janela da webcam.

### Parâmetros

- `mp_hands`: Objeto MediaPipe Hands para detecção de mãos.
- `model_dict`: Dicionário contendo o modelo treinado.
- `hands`: Configuração do MediaPipe Hands.
- `labels_dict`: Dicionário contendo os rótulos dos sinais de Libras.
- `preprocess_hand_data()`: Função para pré-processar os dados das landmarks das mãos.

### Observações

- Certifique-se de possuir permissões de acesso à webcam.
- O MediaPipe Hands pode detectar até duas mãos simultaneamente.
- O modelo treinado deve estar disponível no arquivo 'model.p' no diretório atual.
- As landmarks das mãos são desenhadas sobre o vídeo da webcam, e o sinal de Libras reconhecido é exibido acima da mão correspondente.
- Pressione a tecla "Q" para encerrar o programa.

# Referências

1. **Documentações**:
- [MediaPipe Solutions guide](https://developers.google.com/mediapipe/solutions/guide)

2. **Artigos**:
- [What is random forest?](https://www.ibm.com/topics/random-forest#:~:text=Random%20forest%20is%20a%20commonly,Decision%20trees)
- [Custom models: accuracy and confidence scores](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-accuracy-confidence?view=doc-intel-4.0.0)

3. **Vídeos**:
- [Sign language detection with Python and Scikit Learn | Landmark detection | Computer vision tutorial](https://www.youtube.com/watch?v=MJCSjXepaAM&t)
