# Prédiction de litige de brevet 


## Introduction 

Ici, à travers ce projet de groupe, je souhaiterais mettre en avant un problème souvent rencontré lors des projets de Data Science de classification : celui du problème de déséquilibre de classe. 

### Contexte: 
L’innovation technologique moderne repose largement sur l’accumulation de découvertes successives, notamment dans des secteurs comme les technologies de l’information et les biotechnologies. Dans ces écosystèmes complexes, les brevets sont fortement interdépendants, exposant ainsi les innovateurs à des risques accrus de litiges juridiques. 
L'objectif de ce projet est d'établir un modèle automatique capable de prédire la probabilité, soit classifier si un brevet est susceptible d’être impliqué dans un litige ou non, afin de permettre aux entreprises et aux PME d'anticiper les risques juridiques et de mieux protéger leurs investissements en R&D. 



Nous disposons d’une base de données brut d’environs 644 000 brevets déposés aux États-Unis et au Japon entre 2002 et 2005, nous exploitons des indicateurs variés (total de 30 variables )  — profil du titulaire, stratégie de dépôt, transferts, etc… 

Après analyse et manipulation, nous observons que seulement 1,2% de l’ensemble des brevets sont litigieux ! Nous avons remarqué que ce déséquilibre de classe était très problématique et empêchait nos modèles de Machine Learning à apprendre correctement et des performances médiocres nous empêchant de mettre ce modèle en production pour les entreprises. 

<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/desequilibre.png" width="600" height="400">


## Points clés

Je souhaiterais ici uniquement, mettre en avant quelques parties du travail qui m’ont particulièrement plu : 

### Data Pre-Processing (à l’aide de  la librarie panda  de python): <br>
  Pipeline <br>
  Suppression des variables qui ne contribuent pas à la prédiction (ex: les numéros d'identifiant) <br>
	Data Engineering : création de la variable de cible (étude aggrégation inclusif OU logique ) <br>
	Nettoyage de la base de données (Suppression des NA's ou remplacement par moyenne) <br>
	Creation d’une base de données propres et commune utilisable par tous les membre du groupe

### Analyse : 
	Corrélation <br>
	Feature importance à travers les forêts aléatoires (Random Forest) <br>

### Problème du Déquilibre de Classes : 

   •Méthode 1 : Rééchantillonnage 
Sur-échantillonage  à l’aide de SMOTE de la bibliothèque python ’imblearn'. <br>
Sous-échantillonage à l’aide  UnderSampling de la  bibliothèque python 'RandomUnderSampler’. <br> 

Ces deux méthodes sont testées sur 2 modèles de machine learning : XGBoost et DNN (Dense Neural network). 


   •Méthode 2 : Pondération des classes 

Dans le modèle RandomForest de la bibliothèque scikit-learn, l’argument class_weight permet d’attribuer un poids à chaque classe. En augmentant le poids de la classe minoritaire, on incite le modèle à lui accorder davantage d’attention, ce qui peut améliorer ses performances, notamment en termes de détection de cette classe.


## Quelques détails, résumés, commentaires...





