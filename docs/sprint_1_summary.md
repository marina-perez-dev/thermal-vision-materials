# Résumé du Sprint 1 - Classification de Matériaux Thermiques

Ce document résume les réalisations du Sprint 1, qui a établi les fondations du projet de classification de matériaux thermiques avec un modèle baseline et un pipeline de traitement de données complet.

## Vue d'ensemble

Le Sprint 1 a permis de mettre en place un pipeline complet de traitement de données, d'entraînement et d'évaluation pour la classification de matériaux à partir d'images thermiques. Un modèle baseline CNN a été développé et entraîné, établissant une référence de performance pour les itérations futures.

## 1. Jeu de données

### Description du dataset

- **Source**: Roboflow Universe - Thermal Waste Detection 02
- **Organisation**: UVA Wellassa University
- **Format**: YOLO v3 (images + annotations)
- **Type**: Classification multiclasse
- **Total d'images**: 707 images

### Répartition des données

| Split | Nombre | Pourcentage |
|-------|--------|-------------|
| Train | 495 | 70.01% |
| Validation | 141 | 19.94% |
| Test | 71 | 10.04% |

### Distribution des classes

| Classe | Nombre | Pourcentage |
|--------|--------|-------------|
| glass | 155 | 21.38% |
| organic | 134 | 18.48% |
| steel | 111 | 15.31% |
| polythene | 105 | 14.48% |
| plastic | 102 | 14.07% |
| paper | 66 | 9.10% |
| aluminium | 52 | 7.17% |

**Total**: 7 classes de matériaux à classifier

### Caractéristiques

- Images thermiques avec annotations YOLO
- Déséquilibre modéré entre les classes (ratio max/min ≈ 3:1)
- Splits pré-définis par Roboflow
- Métadonnées extraites et stockées dans `data/raw/roboflow_metadata.csv`

## 2. Pipeline de preprocessing

### Étapes de traitement

1. **Ingestion** (`ingest_roboflow`)
   - Extraction du dataset depuis le fichier ZIP Roboflow
   - Génération de métadonnées (`roboflow_metadata.csv`)
   - Calcul de statistiques (`roboflow_statistics.json`)
   - Versioning avec DVC

2. **Preprocessing** (`preprocess`)
   - Conversion des images en format numpy (64x64 pixels, single-channel)
   - Normalisation min-max par frame
   - Conservation des splits train/validation/test
   - Génération de métadonnées préprocessées (`processed_metadata.csv`)
   - Statistiques de preprocessing (`preprocessing_stats.json`)

### Paramètres de preprocessing

- **Taille cible**: 64x64 pixels
- **Normalisation**: Min-Max par frame
- **Augmentation**: Désactivée (à activer dans les sprints futurs)
- **Format de sortie**: NumPy arrays (.npy)

### Fichiers générés

- `data/processed/train/` : 495 images préprocessées
- `data/processed/validation/` : 141 images préprocessées
- `data/processed/test/` : 71 images préprocessées
- `data/processed/processed_metadata.csv` : Métadonnées enrichies
- `data/processed/preprocessing_stats.json` : Statistiques de preprocessing

## 3. Architecture du modèle baseline

### Structure du modèle

Le modèle baseline est une CNN simple avec l'architecture suivante :

```
Input (64x64x1)
  ↓
Conv2D(32) + ReLU + MaxPooling2D
  ↓
Conv2D(64) + ReLU + MaxPooling2D
  ↓
Conv2D(128) + ReLU + MaxPooling2D
  ↓
Flatten
  ↓
Dense(128) + ReLU + Dropout(0.5)
  ↓
Dense(64) + ReLU + Dropout(0.5)
  ↓
Dense(7) + Softmax
  ↓
Output (probabilités de classes)
```

### Caractéristiques

- **Paramètres**: ~500K paramètres entraînables
- **Input shape**: (64, 64, 1) - images thermiques en niveaux de gris
- **Output**: 7 classes (aluminium, glass, organic, paper, plastic, polythene, steel)
- **Optimiseur**: Adam (learning rate: 0.001)
- **Loss**: Categorical Cross-Entropy
- **Métriques**: Accuracy, Precision, Recall

### Hyperparamètres d'entraînement

- **Batch size**: 32
- **Epochs maximum**: 50
- **Early stopping**: Patience de 10 epochs (surveille `val_loss`)
- **Learning rate reduction**: Réduction automatique sur plateau (facteur 0.5)
- **Checkpointing**: Sauvegarde du meilleur modèle basé sur `val_loss`

## 4. Résultats d'entraînement

### Métriques d'entraînement

- **Epochs effectués**: 28 (arrêt anticipé par early stopping)
- **Meilleur epoch**: 18
- **Meilleure val_loss**: 1.399
- **Meilleure val_accuracy**: 0.496 (49.6%)

### Courbes d'entraînement

- **Loss d'entraînement**: Décroissance de 1.93 → 0.47
- **Accuracy d'entraînement**: Croissance de 21% → 85%
- **Val_loss**: Minimum à l'epoch 18, puis légère augmentation (signe de surapprentissage)
- **Val_accuracy**: Stagnation autour de 50% après l'epoch 18

**Observation**: Le modèle montre des signes de surapprentissage (écart entre train et validation), ce qui est attendu pour un modèle baseline simple.

## 5. Résultats d'évaluation sur le test set

### Métriques globales

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 39.4% |
| **Precision (macro)** | 31.7% |
| **Recall (macro)** | 35.1% |
| **F1-score (macro)** | 31.0% |

### Performance par classe

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| aluminium | 0.00 | 0.00 | 0.00 | 6 |
| glass | 0.37 | 0.70 | 0.48 | 10 |
| organic | 0.38 | 0.64 | 0.47 | 14 |
| paper | 0.00 | 0.00 | 0.00 | 10 |
| plastic | 0.67 | 0.29 | 0.40 | 7 |
| polythene | 0.50 | 0.46 | 0.48 | 13 |
| steel | 0.31 | 0.36 | 0.33 | 11 |

### Analyse des résultats

**Points forts**:
- Classes `glass` et `organic` : Performance modérée (F1 ≈ 0.47-0.48)
- Classe `plastic` : Bonne précision (0.67) mais faible rappel (0.29)

**Points faibles**:
- Classes `aluminium` et `paper` : Aucune prédiction correcte (F1 = 0.00)
- Performance globale faible (accuracy 39.4%)
- Déséquilibre entre précision et rappel pour plusieurs classes

**Causes probables**:
- Dataset petit (71 images de test)
- Déséquilibre des classes (aluminium et paper sous-représentés)
- Architecture baseline simple (3 blocs convolutifs)
- Pas d'augmentation de données
- Surapprentissage visible

## 6. Artéfacts générés

### Modèles

- `models/baseline/baseline_model.keras` : Modèle final
- `models/baseline/baseline_model_best.keras` : Meilleur modèle (epoch 18)
- `models/baseline/baseline_model_config.json` : Configuration
- `models/baseline/baseline_model_history.json` : Historique complet d'entraînement
- `models/baseline/baseline_model_class_mapping.json` : Mapping classes → indices
- `models/baseline/baseline_model_summary.json` : Résumé de l'entraînement

### Métriques et visualisations

- `outputs/baseline_model_metrics.json` : Métriques d'évaluation complètes
- `outputs/baseline_model_confusion_matrix.png` : Matrice de confusion
- `outputs/baseline_model_classification_report.txt` : Rapport de classification

### Documentation

- `docs/BASELINE_MODEL.md` : Documentation du modèle
- `docs/DVC.md` : Guide DVC
- `docs/ROBOFLOW_DATASET.md` : Documentation du dataset
- `notebooks/evaluation_report.ipynb` : Notebook d'évaluation avec visualisations

## 7. Pipeline DVC

### Étapes validées

Le pipeline DVC est complet et reproductible :

1. **ingest_roboflow** : Ingestion du dataset Roboflow
2. **preprocess** : Preprocessing des images
3. **train** : Entraînement du modèle baseline
4. **evaluate** : Évaluation et génération de métriques

### Validation

- `dvc.yaml` : Pipeline complet avec dépendances
- `dvc.lock` : Toutes les étapes verrouillées
- Reproducibilité : `dvc repro` fonctionne de bout en bout
- Versioning : Tous les artéfacts trackés par DVC

## 8. Tests unitaires

### Tests implémentés

- `test_preprocess_shapes.py` : Validation des formes de sortie
- `test_data_loader_mock.py` : Tests des loaders avec données mockées
- `test_model_train_smoke.py` : Tests de fumée pour l'entraînement
- `test_baseline_model.py` : Tests du modèle baseline
- `test_preprocess_roboflow.py` : Tests du preprocessing Roboflow

**Statut**: Tous les tests passent

## 9. Prochaines étapes et recommandations

### Améliorations prioritaires pour Sprint 2

1. **Augmentation de données**
   - Activer l'augmentation (rotation, flip, brightness)
   - Réduire le surapprentissage
   - Améliorer la généralisation

2. **Architecture du modèle**
   - Architecture plus profonde (ResNet, EfficientNet)
   - Transfer learning depuis modèles pré-entraînés
   - Attention aux classes sous-représentées

3. **Gestion du déséquilibre**
   - Weighted loss pour pénaliser les classes rares
   - Oversampling des classes minoritaires (aluminium, paper)
   - Focal loss pour classes difficiles

4. **Hyperparamètres**
   - Grid search ou random search
   - Ajustement du learning rate
   - Optimisation de la régularisation (dropout, L2)

5. **Dataset**
   - Collecte de plus de données pour classes rares
   - Équilibrage des classes
   - Validation croisée pour évaluation plus robuste

6. **Métriques**
   - Suivi de métriques par classe
   - Visualisation des erreurs (confusion matrix améliorée)
   - Analyse des faux positifs/négatifs

### Objectifs de performance

- **Cible Sprint 2**: Accuracy > 60%, F1-score macro > 0.55
- **Cible Sprint 3**: Accuracy > 75%, F1-score macro > 0.70
- **Production**: Accuracy > 85%, F1-score macro > 0.80

## 10. Conclusion

Le Sprint 1 a établi une base solide pour le projet :

Pipeline de données complet et reproductible  
Modèle baseline fonctionnel  
Infrastructure de test et validation  
Documentation et artéfacts versionnés  
Métriques de référence établies  

Le modèle baseline atteint 39.4% d'accuracy, ce qui est attendu pour une première itération. Les résultats montrent clairement les axes d'amélioration pour les sprints suivants, notamment l'augmentation de données, l'amélioration de l'architecture, et la gestion du déséquilibre des classes.

---

**Date de réalisation**: Décembre 2024  
**Sprint**: Sprint 1  
**Statut**: Complété

