# Car Insurance Feature Analysis

## Description
Projet de machine learning pour identifier la **feature la plus prédictive** d’un sinistre automobile (`outcome`). Chaque feature est testée individuellement avec un pipeline `SimpleImputer + LogisticRegression` et cross-validation. La meilleure feature est stockée dans `best_feature_df`.

## Dataset
`car_insurance.csv` : contient des informations personnelles et véhicules des clients.

## Méthodologie
1. Nettoyage et transformation des colonnes catégorielles en numériques.
2. Sélection des features pertinentes.
3. Pipeline ML : imputation + logistic regression.
4. Cross-validation pour mesurer l’accuracy.
5. Identification de la feature la plus performante.

## Résultat
DataFrame final `best_feature_df` avec la feature la plus prédictive et son accuracy.
