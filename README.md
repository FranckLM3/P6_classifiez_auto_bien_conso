# Projet : Classifiez automatiquement des biens de consommation

**Autor :** Franck LE MAT 

**Date :** 26/01/2022

**Durée totale :** 100 heures

## Background du projet :
Vous êtes Data Scientist au sein de l’entreprise "Place de marché”, qui souhaite lancer une marketplace e-commerce.
Sur la place de marché, des vendeurs proposent des articles à des acheteurs en postant une photo et une description.
Pour l'instant, l'attribution de la catégorie d'un article est effectuée manuellement par les vendeurs et est donc peu fiable. De plus, le volume des articles est pour l’instant très petit.
Pour rendre l’expérience utilisateur des vendeurs (faciliter la mise en ligne de nouveaux articles) et des acheteurs (faciliter la recherche de produits) la plus fluide possible et dans l'optique d'un passage à l'échelle, il devient nécessaire d'automatiser cette tâche.
Linda, lead data scientist, vous demande donc d'étudier la faisabilité d'un moteur de classification des articles en différentes catégories, avec un niveau de précision suffisant.
- Réaliser une première étude de faisabilité d'un moteur de classification d'articles basé sur une image et une description pour l'automatisation de l'attribution de la catégorie de l'article.
- Ce projet est découpé en 2 grosses parties : Partie NLP & Partie Computer Vision

## Key point du projet :
- Preprocessing & Vectorization du corpus (création d'un Bag of Word avec NLTK)
- Réduction dimensionnelle (PCA)
- Clustering (Kmeans, Clustering hiérarchique) et mesure de qualité des clusters avec le coefficient de silhouette
- Mesure de similarité entre du Clustering via Indice de Rand Ajusté (ARI)
- Preprocessing des images et création d'un Bag of Visual Word (BOVW) via l'algorithme ORB
- Partie Deep Learning : Transfer Learning sur VGG16 (CNN)
- Stacking de l'approche NLP & Computer Vision et création de features puis utilisation d'une RandomForest, XGboost

## Livrables :
- Notebooks contenant les fonctions permettant le prétraitement des données textes et images ainsi que les résultats du clustering (en y incluant des représentations graphiques au besoin).
- Un support de présentation qui présente la démarche et les résultats du clustering.


## Compétences évaluées :
- Prétraiter des données image pour obtenir un jeu de données exploitable
- Mettre en œuvre des techniques de réduction de dimension
- Représenter graphiquement des données à grandes dimensions
- Prétraiter des données texte pour obtenir un jeu de données exploitable

