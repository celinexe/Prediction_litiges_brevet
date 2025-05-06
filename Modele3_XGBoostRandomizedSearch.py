# Code Python commenté pour la partie 4.3 XGBoost (de Entrainement des modèles et résultats)

# =========================================================
# Imports des bibliothèques et des fonctions nécessaires
# =========================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
""" pip install lightgbm
(si nécessaire)"""

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE

# =========================================================
# Chargement des données
# =========================================================

data = pd.read_csv("../DataBases/Database_propre.csv")
X = data.drop("prediction", axis=1)
y = data["prediction"]

# Séparation en ensembles d'entraînement et de test avec stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# =========================================================
# Modèle XGBoost avec pondération manuelle
# =========================================================

# Création des poids d'échantillons pour prendre en compte le déséquilibre
sample_weights = np.where(y_train == 0, 1, 10)

model_smote = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

model_smote.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model_smote.predict(X_test)
print("\nClassification report - XGBoost (avec pondération manuelle) :")
print(classification_report(y_test, y_pred))

# =========================================================
# Modèle LightGBM avec gestion automatique du déséquilibre
# =========================================================

model_lgb = lgb.LGBMClassifier(
    objective='binary',
    is_unbalance=True,
    random_state=42
)

model_lgb.fit(X_train, y_train)
y_pred_lgb = model_lgb.predict(X_test)

print("\nClassification report - LightGBM :")
print(classification_report(y_test, y_pred_lgb))

# =========================================================
# Modèle de stacking (XGBoost + Régression Logistique)
# =========================================================

base_learners = [
    ('xgb', xgb.XGBClassifier(objective='binary:logistic', random_state=42)),
    ('lr', LogisticRegression())
]

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression()
)

stacking_model.fit(X_train, y_train)
y_pred_stack = stacking_model.predict(X_test)

print("\nClassification report - Stacking (XGB + LogisticRegression) :")
print(classification_report(y_test, y_pred_stack))

# =========================================================
# Recherche d'hyperparamètres avec RandomizedSearchCV (XGBoost)
# =========================================================

param_dist = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'subsample': [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    scoring='f1',
    cv=3,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nMeilleurs paramètres :")
print(random_search.best_params_)

# Prédiction finale avec le meilleur modèle trouvé
y_pred_random = random_search.predict(X_test)

print("\nClassification report - XGBoost (meilleur modèle RandomizedSearchCV) :")
print(classification_report(y_test, y_pred_random))
