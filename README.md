# Prédiction de litige de brevet 


## Introduction 

Ici, à travers ce projet de groupe, je souhaiterais mettre en avant un problème souvent rencontré lors des projets de Data Science de classification : celui du problème de déséquilibre de classe. 

### Contexte: 
L’innovation technologique moderne repose largement sur l’accumulation de découvertes successives, notamment dans des secteurs comme les technologies de l’information et les biotechnologies. Dans ces écosystèmes complexes, les brevets sont fortement interdépendants, exposant ainsi les innovateurs à des risques accrus de litiges juridiques. 
L'objectif de ce projet est d'établir un modèle automatique capable de prédire la probabilité, soit classifier si un brevet est susceptible d’être impliqué dans un litige ou non, afin de permettre aux entreprises et aux PME d'anticiper les risques juridiques et de mieux protéger leurs investissements en R&D. 

<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/intro.png" width="500" height="300">



Nous disposons d’une base de données brut d’environs 644 000 brevets déposés aux États-Unis et au Japon entre 2002 et 2005, nous exploitons des indicateurs variés (total de 30 variables )  — profil du titulaire, stratégie de dépôt, transferts, etc… 

Après analyse et manipulation, nous observons que seulement 1,2% de l’ensemble des brevets sont litigieux ! Nous avons remarqué que ce déséquilibre de classe était très problématique et empêchait nos modèles de Machine Learning à apprendre correctement et des performances médiocres nous empêchant de mettre ce modèle en production pour les entreprises. 

<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/desequilibre.png" width="500" height="300">


## Points clés

Je souhaiterais ici uniquement, mettre en avant quelques parties du travail qui m’ont particulièrement plu : 

### Data Pre-Processing (à l’aide de  la librarie panda  de python): 
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

Data-preprocessing : 
Le Jupyter Notebook contient les détails de chaque étape du code, notamment le nettoyage de la base de données, la définition de la variable cible, etc. 
[](lien) 


#### Construction de la variable cible : 

Deux variables binaires étaient disponibles pour caractériser les litiges : Infringement (procédures pour contrefaçon) et Invalidity (procédures pour invalidité). Ces deux types de litige, bien que juridiquement distincts, relèvent du même phénomène économique d’exposition au risque juridique. En pratique, ils traduisent tous deux une contestation formelle de la validité ou de l’usage exclusif du brevet. <br>
Construire une variable cible synthétique définie comme le OU logique (Infringement OR Invalidity) entre les deux variables, de façon à capter tous les types de litiges. Cela permet d’élargir le nombre d’exemples positifs et de rendre le modèle
plus sensible à des formes variées de contentieux. 

#### Sélection des variables d'entrainement 

L’analyse de la matrice de corrélation n’a pas permis d’identifier clairement des variables explicatives pertinentes, car la corrélation entre les variables explicatives et la variable cible (‘prediction’) était très faible. J’ai donc utilisé un modèle Random Forest pour estimer l’importance des variables, en m’appuyant sur la fréquence d’utilisation des variables dans les arbres et leur impact sur la réduction de l’impureté (indice de Gini).

<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/matrice_cor.png" width="500" height="300">


<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/feature_impo_rf.png" width="500" height="300">

Dans un second temps, j’ai fixé un seuil d’importance et réentraîné le modèle en ne conservant que les variables dépassant ce seuil. Cependant, cette approche n’a pas permis d’obtenir une amélioration significative des performances. Nous avons donc finalement décidé de conserver l’ensemble des variables pour l’entraînement des modèles.


#### Réechantillonage 

Le principe de SMOTE est de générer de nouvelles observations synthétiques appartenant à la classe minoritaire, dans notre cas, la classe positive correspondant aux li- tiges. Ces nouvelles observations sont créées en interpolant les caractéristiques d’exemples existants de la classe minoritaire. En augmentant ainsi le nombre d’exemples positifs, on parvient à rééquilibrer les proportions entre les classes, ce qui améliore la capacité du mo- dèle à reconnaître les cas rares. Il est également possible de contrôler le degré de rééquili- brage, en choisissant dans quelle proportion on souhaite augmenter la classe minoritaire par rapport à la classe majoritaire. 

Si l’oversampling parvient à régler le souci de déséquilibre, l’undersampling est lui aussi une autre méthode pour atténuer ce déséquilibre. Au lieu de générer synthétiquement des observations, l’undersampling réduit la base de données en supprimant les observations appartenant à la classe majoritaire. Il est possible notamment de contrôler le degré de rééquilibrage. 

Le désavantage de l’oversampling est la fiabilité des observations générées synthéti- quement, tandis que pour l’undersampling, l‘inconvénient se trouve dans la perte d’in- formations puisque l’on supprime des observations. Si la taille du jeu de données n’est pas importante, on risque de perdre trop de données et d’avoir un modèle dysfonctionnel. Toutefois, même si la taille du jeu de données est assez importante, mais que le déséqui- libre de classes est trop élevé, en se ramenant à une proportion correcte de classes positive et négative, on risque de perdre trop d’informations car on supprime trop d’observations. 


La méthode de réechantillonage Smote et UnderSampling sont appliquées sur dans un premier modèle de réseau de neurones DNN et puis sur un second modèle XGboost. Ces deux modèles de machine learning, XGBoost et DNN, dont l’optimisation (architecture et choix des hyperparamètres) a déjà été réalisée au préalable.

On choisit un paramètre de rééquilibrage de 0,5, c’est-à-dire qu’on augmente ou réduit les observations de la classe minoritaire pour qu’elles représentent 50 % des observations de la classe majoritaire.

<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/dnn_score.png" width="500" height="300">

<img src="https://github.com/celinexe/Prediction_litiges_brevets/blob/main/images/xgboost_score.png" width="500" height="300">




#### Conclusion Resampling 

Après l’application des techniques de rééchantillonnage, on observe une amélioration significative du taux de rappel (recall) ainsi que du score F1, ce qui indique que le mo- dèle parvient mieux à détecter les cas de litige. Cependant, cette amélioration se fait au détriment de la précision, qui chute fortement. Avant le rééchantillonnage, 55 % des brevets prédits comme étant des litiges étaient effectivement des litiges. Après rééchan- tillonnage, ce taux tombe en moyenne à seulement 7.5%, ce qui signifie que le modèle génère beaucoup plus de faux positifs. 
Cette situation illustre bien le compromis entre précision et rappel : en augmentant la capacité du modèle à détecter les litiges (rappel), on augmente également le risque de prédire des litiges là où il n’y en a pas (baisse de précision). 








