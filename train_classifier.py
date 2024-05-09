import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Carrega os dados de entrada e as labels a partir do arquivo pickle."""
    data_dict = pickle.load(open(file_path, 'rb'))
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    return data, labels

def train_model(data, labels):
    """Treina o modelo RandomForestClassifier."""
    model = RandomForestClassifier()
    model.fit(data, labels)
    return model

def evaluate_model(model, x_test, y_test):
    """Avalia o modelo usando os dados de teste e retorna a acurácia, o F-score e a matriz de confusão."""
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    f_score = f1_score(y_test, y_predict, average='weighted')
    confusion = confusion_matrix(y_test, y_predict)
    return accuracy, f_score, confusion

def save_model(model, file_path):
    """Salva o modelo treinado em um arquivo pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

def plot_confusion_matrix(confusion_matrix, labels):
    """Plota a matriz de confusão."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    # Carrega os dados
    data, labels = load_data('./data.pickle')

    # Divide os dados em conjuntos de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Treina o modelo
    trained_model = train_model(x_train, y_train)

    # Salva o modelo treinado
    save_model(trained_model, 'model.p')

    # Avalia o modelo
    accuracy, f_score, confusion = evaluate_model(trained_model, x_test, y_test)

    # Imprime as métricas do modelo
    print(f'Accuracy Score: {accuracy:.2f}')
    print(f'F-score: {f_score:.2f}')

    # Plota a matriz de confusão
    labels = np.unique(labels)
    plot_confusion_matrix(confusion, labels)
