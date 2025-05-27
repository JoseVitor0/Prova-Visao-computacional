import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

caminho = '/content/imagens'
imagens = []
rotulos = []
classe_map = {'sedan': 0, 'suv': 1}
TARGET_SIZE = (128, 128)

for classe in os.listdir(caminho):
    if not os.path.isdir(os.path.join(caminho, classe)):
        continue

    pasta_classe = os.path.join(caminho, classe)

    if not os.path.isdir(pasta_classe):
        continue

    for nome_arquivo in os.listdir(pasta_classe):
        if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.png'):
            caminho_img = os.path.join(pasta_classe, nome_arquivo)
            
            img = cv2.imread(caminho_img, cv2.IMREAD_COLOR)

            if img is None:
                continue

           
            img_resized = cv2.resize(img, TARGET_SIZE)

            imagens.append(img_resized)
            rotulos.append(classe_map[classe.lower()])


# Visualizar 5 exemplos de cada classe
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
classes_vistas = [0, 1]  
exibidos = {0: 0, 1: 0}

for i, (img, r) in enumerate(zip(imagens, rotulos)):
    if exibidos[r] < 5:
        axs[r, exibidos[r]].imshow(img)
        axs[r, exibidos[r]].axis('off')
        axs[r, exibidos[r]].set_title('sedan' if r == 0 else 'suv')
        exibidos[r] += 1
    if all(v == 5 for v in exibidos.values()):
        break

plt.tight_layout()
plt.show()


imagens = np.array(imagens) / 255.0
rotulos = np.array(rotulos)

x_train, x_test, y_train, y_test = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)

import numpy as np
import collections

print("Distribuição de rótulos no treino:", collections.Counter(y_train))
print("Distribuição de rótulos no teste:", collections.Counter(y_test))

print("Rótulos de treino únicos:", np.unique(y_train, return_counts=True))

y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

print("Shape de y_train após one-hot encoding:", y_train_one_hot.shape)
print("Shape de y_test após one-hot encoding:", y_test_one_hot.shape)

class_names = ['sedan', 'suv']

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train_one_hot, epochs=10, validation_data=(x_test, y_test_one_hot))

test_loss, test_acc = model.evaluate(x_test, y_test_one_hot, verbose=2) # Use the one-hot encoded labels here
print(f'\nAcurácia no conjunto de teste: {test_acc:.4f}')

y_pred = np.argmax(model.predict(x_test), axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))

plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.title('Desempenho da CNN - Hatch vs Caminhonete')
plt.show()

import os

caminho = '/content/imagens/testes'
print("Arquivos encontrados na pasta:")
for f in os.listdir(caminho):
    print(f)


from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError

caminho_teste = '/content/imagens/testes'

for nome_arquivo in os.listdir(caminho_teste):
    if nome_arquivo.endswith('.jpg') or nome_arquivo.endswith('.png'):
        caminho_img = os.path.join(caminho_teste, nome_arquivo)

        try:
            img = image.load_img(caminho_img, target_size=(128, 128))
        except UnidentifiedImageError:
            print(f"Erro ao abrir imagem: {nome_arquivo} (arquivo inválido)")
            continue

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = int(predictions[0][0] > 0.5)


        print(f"Imagem: {nome_arquivo} - Classe prevista: {class_names[predicted_class]}")
