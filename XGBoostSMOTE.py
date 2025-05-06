# Code Python commenté pour la partie: Modélisation avec XGBoost, gestion du déséquilibre et calibration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

############################################
# 1) CHARGEMENT ET PRÉPARATION DES DONNÉES
############################################

# Charger la base de données
file_path = "../DataBases/Database_propre.csv"
data = pd.read_csv(file_path)

# Séparation des variables explicatives (X) et de la cible (y)
X = data.drop("prediction", axis=1)
y = data["prediction"]

# Division en ensembles d'entraînement et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

############################################
# 2) PREMIER MODÈLE XGBoost AVEC scale_pos_weight
############################################

# Calcul du ratio entre classe majoritaire et classe minoritaire
gamma_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print("scale_pos_weight:", gamma_ratio)

# Initialisation du modèle XGBClassifier
model = xgb.XGBClassifier(
    scale_pos_weight=gamma_ratio,  # Gestion du déséquilibre
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# Entraînement du modèle
model.fit(X_train, y_train)

# Évaluation sur l'ensemble de test
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Précision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))

############################################
# 3) DÉTERMINATION DU MEILLEUR SEUIL BASÉ SUR LE F1
############################################

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = f1_scores.argmax()
optimal_threshold = thresholds[optimal_idx]

print("Seuil optimal:", optimal_threshold)
print("Précision optimale:", precisions[optimal_idx])
print("Rappel optimal:", recalls[optimal_idx])

plt.plot(thresholds, precisions[:-1], label="Précision")
plt.plot(thresholds, recalls[:-1], label="Rappel")
plt.xlabel("Seuil")
plt.ylabel("Score")
plt.legend()
plt.show()

# Génération d'un nouveau vecteur de prédictions en fonction du seuil optimal
y_pred_adjusted = (y_prob >= optimal_threshold).astype(int)
print("F1-score ajusté:", f1_score(y_test, y_pred_adjusted))

############################################
# 4) SMOTE POUR RÉÉQUILIBRAGE DE LA CLASSE MINORITAIRE
############################################

# Séparation initiale
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Distribution originale :", y_train.value_counts())

# Application de SMOTE
# par cette nouvelle ligne réduisant la taille mémoire :
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Distribution après SMOTE :", pd.Series(y_train_res).value_counts())

############################################
# 5) GRID SEARCH POUR TROUVER LES MEILLEURS HYPERPARAMÈTRES XGBoost
############################################

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'scale_pos_weight': [1]  # On peut ajuster si besoin
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train_res, y_train_res)
print("Meilleurs paramètres :", grid_search.best_params_)

best_model = grid_search.best_estimator_

############################################
# 6) ÉVALUATION DU MODÈLE OPTIMAL
############################################

# Prédiction des probabilités sur l'ensemble de test
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Courbe Precision-Recall
precisions_best, recalls_best, thresholds_best = precision_recall_curve(y_test, y_prob_best)
f1_scores_best = 2 * precisions_best * recalls_best / (precisions_best + recalls_best + 1e-8)
optimal_idx_best = f1_scores_best.argmax()
optimal_threshold_best = thresholds_best[optimal_idx_best]

print("Seuil optimal :", optimal_threshold_best)
print("Précision au seuil optimal :", precisions_best[optimal_idx_best])
print("Rappel au seuil optimal :", recalls_best[optimal_idx_best])
print("F1-score optimal :", f1_scores_best[optimal_idx_best])

# Visualisation
plt.plot(thresholds_best, precisions_best[:-1], label="Précision")
plt.plot(thresholds_best, recalls_best[:-1], label="Rappel")
plt.xlabel("Seuil")
plt.ylabel("Score")
plt.legend()
plt.show()

# Application du seuil optimal
y_pred_adjusted_best = (y_prob_best >= optimal_threshold_best).astype(int)

print("Évaluation finale :")
print("Précision :", precision_score(y_test, y_pred_adjusted_best))
print("Rappel :", recall_score(y_test, y_pred_adjusted_best))
print("F1-score :", f1_score(y_test, y_pred_adjusted_best))
print("AUC-ROC :", roc_auc_score(y_test, y_prob_best))

############################################
# 7) CALIBRATION DU MODÈLE (ISOTONIC)
############################################

calibrated_model = CalibratedClassifierCV(best_model, cv='prefit', method='isotonic')
calibrated_model.fit(X_train_res, y_train_res)

y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

# Nouveau seuil optimal sur probabilités calibrées
precisions_cal, recalls_cal, thresholds_cal = precision_recall_curve(y_test, y_prob_calibrated)
f1_scores_cal = 2 * precisions_cal * recalls_cal / (precisions_cal + recalls_cal + 1e-8)
optimal_idx_cal = f1_scores_cal.argmax()
optimal_threshold_cal = thresholds_cal[optimal_idx_cal]

print("Seuil optimal (calibré) :", optimal_threshold_cal)
print("Précision (calibrée) :", precisions_cal[optimal_idx_cal])
print("Rappel (calibré) :", recalls_cal[optimal_idx_cal])
print("F1-score (calibré) :", f1_scores_cal[optimal_idx_cal])
