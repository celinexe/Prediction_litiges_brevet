# ----------------------------
# Importation des librairies et des fonctions nécessaires
# ----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# ----------------------------
# Chargement des données
# ----------------------------
data = pd.read_csv('../DataBases/Database_propre.csv')

# Extraction de la variable cible
prediction = data['prediction']


# ----------------------------
# Analyse exploratoire : répartition des classes
# ----------------------------
def counts_litige(x):
    return np.sum(x == 1)

nb_litiges = counts_litige(prediction)
nb_total = prediction.size

print(f"Nombre de litiges : {nb_litiges}")
print(f"Nombre total d'observations : {nb_total}")
print(f"Proportion de litiges : {nb_litiges / nb_total:.2%}")

# Histogramme des classes
plt.hist(prediction, bins=2, color='skyblue', edgecolor='black')
plt.xticks([0, 1], ['Non-litige', 'Litige'])
plt.title('Répartition des classes')
plt.ylabel('Nombre d\'observations')
plt.show()


# ----------------------------
# Visualisation des corrélations
# ----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.show()

# ----------------------------
# Séparation des variables explicatives en deux groupes
# ----------------------------
dataCopie = data.drop('prediction', axis=1)

dataCopie.head().T
groupe1 = dataCopie.iloc[:, :14]
groupe2 = dataCopie.iloc[:, 14:28]
y = data['prediction']

# ----------------------------
# Création des jeux de données d'entraînement et de test
# ----------------------------
x_train1, x_test1, y_train1, y_test1 = train_test_split(groupe1, y, test_size=0.2, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(groupe2, y, test_size=0.2, random_state=42)

# ----------------------------
# Modélisation des deux groupes : Random Forest
# ----------------------------
model1 = RandomForestClassifier(max_depth=10, min_samples_split=5, random_state=42)
model1.fit(x_train1, y_train1)
model2 = RandomForestClassifier(max_depth=10, min_samples_split=5, random_state=42)
model2.fit(x_train2, y_train2)

# ----------------------------
# Importance des variables : Groupe 1
# ----------------------------
importances1 = model1.feature_importances_

plt.figure(figsize=(20, 5))
plt.bar(groupe1.columns, importances1, color='skyblue')
plt.xlabel("Variables du groupe 1")
plt.ylabel("Importance")
plt.ylim(0, 0.3)
plt.title("Importance des variables dans RandomForestClassifier (Groupe 1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Importance des variables : Groupe 2
# ----------------------------
importances2 = model2.feature_importances_

plt.figure(figsize=(20, 5))
plt.bar(groupe2.columns, importances2, color='skyblue')
plt.xlabel("Variables du groupe 2")
plt.ylabel("Importance")
plt.ylim(0, 0.3)
plt.title("Importance des variables dans RandomForestClassifier (Groupe 2)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------
# Évaluation des performances
# ----------------------------
# Évaluation du modèle 1
y_pred1 = model1.predict(x_test1)
recall1 = recall_score(y_test1, y_pred1)
precision1 = precision_score(y_test1, y_pred1)
roc_auc1 = roc_auc_score(y_test1, y_pred1)

print('\nPerformances du modèle sur le groupe 1 :')
print("Recall :", recall1)
print("Precision :", precision1)
print("ROC-AUC :", roc_auc1)

# Évaluation du modèle 2
y_pred2 = model2.predict(x_test2)
recall2 = recall_score(y_test2, y_pred2)
precision2 = precision_score(y_test2, y_pred2)
roc_auc2 = roc_auc_score(y_test2, y_pred2)

print('\nPerformances du modèle sur le groupe 2 :')
print("Recall :", recall2)
print("Precision :", precision2)
print("ROC-AUC :", roc_auc2)

# ----------------------------
# Sélection des variables importantes à partir des deux groupes
# ----------------------------
selected_features1 = groupe1.columns[model1.feature_importances_ > 0.05]
selected_features2 = groupe2.columns[model2.feature_importances_ > 0.05]

print("Variables importantes du groupe 1 :", selected_features1)
print("Variables importantes du groupe 2 :", selected_features2)

all_selected_features = np.unique(np.concatenate((selected_features1, selected_features2)))
dfFiltre = data[all_selected_features]

# ----------------------------
# Nouveau modèle avec réduction et sélection des variables
# ----------------------------
x_train, x_test, y_train, y_test = train_test_split(dfFiltre, y, test_size=0.2, random_state=42)

modeleReduit = RandomForestClassifier(max_depth=10, min_samples_split=5, random_state=42)
modeleReduit.fit(x_train, y_train)

y_pred = modeleReduit.predict(x_test)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print('\nPerformances du modèle avec sélection des variables issues des deux groupes :')
print("Recall :", recall)
print("Precision :", precision)
print("ROC-AUC :", roc_auc)
