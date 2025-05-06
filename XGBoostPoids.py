# =========================================================
# Imports et chargement unique des données
# =========================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

from imblearn.over_sampling import SMOTE

# Lecture des données une seule fois
data = pd.read_csv(r'../DataBases/Database_propre.csv')
X = data.drop("prediction", axis=1)
y = data["prediction"]

# =========================================================
# 1) Recherche de poids optimaux (poids_0_values, poids_1_values)
# =========================================================
# Création d'un jeu de données pour cette expérience
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

poids_0_values = np.logspace(-2, 2, num=5)
poids_1_values = np.logspace(1, 3, num=5)

best_f1 = 0
best_weights = None

for poids_0 in poids_0_values:
    for poids_1 in poids_1_values:
        print(f"Test avec poids_0 = {poids_0:.2f}, poids_1 = {poids_1:.2f}")
        
        # Construction des poids pour la classe minoritaire vs majoritaire
        sample_weights = np.where(y_train1 == 0, poids_0, poids_1)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42
        )
        
        model.fit(X_train1, y_train1, sample_weight=sample_weights)
        
        y_pred1 = model.predict(X_test1)
        report1 = classification_report(y_test1, y_pred1, output_dict=True)
        f1_class_1 = report1['1']['f1-score']
        
        print(f"F1-score classe 1 : {f1_class_1:.5f}\n")
        
        if f1_class_1 > best_f1:
            best_f1 = f1_class_1
            best_weights = (poids_0, poids_1)

print(f"Meilleur couple de poids trouvé : {best_weights} avec F1-score = {best_f1:.5f}")

# =========================================================
# 2) Modèle XGBoost avec un jeu de données différent (test_size=0.3)
# =========================================================
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Exemple de pondération simple : class 0 = 1, class 1 = 10
sample_weights2 = np.where(y_train2 == 0, 1, 10)

model2 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

model2.fit(X_train2, y_train2, sample_weight=sample_weights2)

y_pred2 = model2.predict(X_test2)
report2 = classification_report(y_test2, y_pred2, output_dict=True)

precision_1 = report2['1']['precision']
recall_1 = report2['1']['recall']
f1_class_1 = report2['1']['f1-score']

print(f"\nRésultats (test_size=0.3, sample_weights=1 vs 10) :")
print(f"Précision (classe 1) : {precision_1:.5f}")
print(f"Rappel (classe 1)    : {recall_1:.5f}")
print(f"F1-score (classe 1)  : {f1_class_1:.5f}")

# =========================================================
# 3) Ajustement du seuil de décision (random_state=84)
# =========================================================
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=84
)

poids_0 = 1
poids_1 = 10
sample_weights3 = np.where(y_train3 == 0, poids_0, poids_1)

model3 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

model3.fit(X_train3, y_train3, sample_weight=sample_weights3)

y_prob3 = model3.predict_proba(X_test3)[:, 1]

precisions, rappels, seuils = precision_recall_curve(y_test3, y_prob3)
f1_scores = 2 * (precisions * rappels) / (precisions + rappels + 1e-10)

best_index = np.argmax(f1_scores)
seuil_optimal = seuils[best_index]
f1_optimal = f1_scores[best_index]
precision_opt = precisions[best_index]
recall_opt = rappels[best_index]

y_pred_opt = (y_prob3 >= seuil_optimal).astype(int)

print(f"\nRéglage du seuil (random_state=84, poids_0=1, poids_1=10) :")
print(f"Seuil optimal : {seuil_optimal:.5f}")
print(f"Précision optimale : {precision_opt:.5f}")
print(f"Rappel optimal    : {recall_opt:.5f}")
print(f"F1-score optimal  : {f1_optimal:.5f}")
