# Code Python commenté pour la partie 4.4 Réseaux de neurones (de Entrainement des modèles et résultats)

# ==========================================
# Importation des bibliothèques et des fonctions nécessaires
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ==========================================
# 1. Chargement et préparation des données
# ==========================================
data = pd.read_csv('../DataBases/Database_propre.csv')

X = data.drop('prediction', axis=1)
y = data['prediction']

# Séparation en données d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardisation des données
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ==========================================
# 2. Modèle DNN - Version 1 (2 couches cachées)
# ==========================================
model1 = Sequential()
model1.add(Input(shape=(27,)))
model1.add(Dense(26, activation='relu'))  #Première couche cachée
model1.add(Dense(13, activation='relu'))  #Deuxième couche cachée
model1.add(Dense(1, activation="sigmoid")) #Couche de sortie

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history1 = model1.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)

# ==========================================
# 3. Visualisation des courbes d'entraînement
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], label='Perte entraînement')
plt.plot(history1.history['val_loss'], label='Perte validation')
plt.title('Fonction de perte')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history1.history['accuracy'], label='Précision entraînement')
plt.plot(history1.history['val_accuracy'], label='Précision validation')
plt.title('Précision')
plt.xlabel('Époque')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 4. Évaluation du modèle sur la base de test
# ==========================================
y_pred = model1.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Évaluation du modèle 1 (2 couches cachées) :")
print("Recall     :", recall)
print("Precision  :", precision)
print("ROC-AUC    :", roc_auc)

# ==========================================
# 5. Modèle DNN - Version 2 (3 couches cachées)
# ==========================================
model2 = Sequential([
    Input(shape=(27,))])

model2.add(Dense(26, activation='relu'))  #Première couche cachée
model2.add(Dense(13, activation='relu'))  #Deuxième couche cachée
model2.add(Dense(6, activation='relu'))   #Troisième couche cachée
model2.add(Dense(1, activation="sigmoid")) #Couche de sortie

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history2 = model2.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)
