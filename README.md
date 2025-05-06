# Prédiction de litige de brevet 


## Introduction 

Ici, à travers ce projet de groupe, je souhaiterai mettre en avant un problème souvent rencontrées lors des projets de Data Science de classification : celui du problème de déséquilibre de classe. 
Ce projet académique se porte sur la prédiction de litiges de brevet : établir un modèle automatique capable de prédire la probabilité, soit classifier si un brevet est susceptible d’être impliquée dans un litige ou non. 

Nous disposons d’une base de données brut d’environs 644 000 brevets déposés aux États-Unis et au Japon entre 2002 et 2005, nous exploitons des indicateurs variés — profil du titulaire, stratégie de dépôt, transferts, etc… 

Après analyse et manipulation, nous observons que seulement 1,2% de l’ensemble des brevets sont litigieux ! Nous avons remarqué que ce déséquilibre de classe était très problématique et empêchait nos modèles de Machine Learning à apprendre correctement et des performances médiocres nous empêchant de mettre ce modèle en production pour les entreprises. 

![ ] (image déséquilibre de classe) 


## Points clé

Ce projet académique a été fait en groupe (4 personnes). Je souhaiterais ici uniquement, mettre en avant quelques parties de mon travail et ma contribution dans le projet et qui m’a particulièrement plu : 

-Data Pre-Processing (à l’aide de  la librarie panda  de python): <br>
  Pipeline 
  Suppression des variables qui ne contribuent pas à la prédiction (ex: les numéros d'identifiant) 
	Data Engineering : création de la variable de cible (étude aggrégation inclusif OU logique ) 
	Nettoyage de la base de données (Suppression des NA's ou remplacement par moyenne) 
	Creation d’une base de données propres et commune utilisable par tous les membre du groupe

-Analyse : 
	Corrélation 
	Feature importance à travers les forêts aléatoires (Random Forest) 

-Déquilibre de Classes : 

   Méthode 1 : Rééchantillonnage 
Sur-échantillonage et Sous-échantillonage à l’aide de SMOTE, et UnderSampling des bibliothèques ‘’imblearn’’ et ‘’RandomUnderSampler’’ des libraries Python. 
Ces deux méthodes sont testées sur 2 modèles de machine learning : XGBoost et DNN (Dense Neural network). 


   Méthode 2 : Pondération des classes 

Dans le Modèle RandomForest de la librairie sklearn, il est possible de jouer avec l’argument ‘classe_weight’ nous permettant ainsi d’attribuer un poid pour chaque classe. 
Nous verrons comment cette méthode impactent l’apprentissage et les résultats. 


## Quelques résumés commentaires
