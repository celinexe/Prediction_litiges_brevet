# Code Python commenté pour la partie 4.2 Random Forests (de Entrainement des modèles et résultats)
# 1. Import des bibliothèques
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score
from sklearn.metrics import roc_auc_score



data=pd.read_csv('../DataBases/Database_propre.csv')

prediction=data['prediction']
data= data.drop('prediction', axis=1)


X_train, X_test, y_train, y_test = train_test_split(data, prediction, test_size=0.3, random_state=42)

##RANDOM FOREST PARAMETRE (Défaut )


# 4. Définition du modèle et de la grille
rfc = RandomForestClassifier(random_state=42)


rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

# 8. Évaluation des performances
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_proba = rfc.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print("\nClassification report :\n", classification_report(y_test, y_pred))

print(f"Recall    : {recall}")
print(f"Precision : {precision}")
print('f1_score',f1)
print("roc_auc",auc)



##GRID SERACH CV pr RANDOM FOREST

# 4. Définition du modèle et de la grille
rfc = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100,150],
    'max_depth': [5,10],
    'min_samples_split': [10, 50]
}

# 5. Recherche des meilleurs hyperparamètres avec validation croisée
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid,
                           cv=3, scoring='f1', verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

# 6. Meilleurs paramètres
print("Meilleurs hyperparamètres :", grid_search.best_params_)



# 7. Prédiction avec le meilleur modèle
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 8. Évaluation des performances
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Precision : {precision}")
print(f"Recall    : {recall}")
print("\nClassification report :\n", classification_report(y_test, y_pred))


# Probabilités prédites pour la classe positive
y_proba = best_model.predict_proba(X_test)[:, 1]

# Calcul de l'AUC
auc = roc_auc_score(y_test, y_proba)
print(f"Aire sous la courbe ROC (AUC) : {auc:.3f}")
