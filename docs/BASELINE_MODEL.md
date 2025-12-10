# Modèle Baseline - Classification de Matériaux Thermiques

Ce document décrit l'implémentation et l'utilisation du modèle baseline pour la classification de matériaux thermiques.

## Vue d'ensemble

Le modèle baseline est une architecture CNN simple conçue pour établir une référence de performance initiale. Il est composé de 3 blocs convolutifs suivis de couches denses pour la classification multi-classes.

## Architecture

### Structure du modèle

```
Input (64x64x1)
  ↓
Conv2D(32) + ReLU + MaxPooling
  ↓
Conv2D(64) + ReLU + MaxPooling
  ↓
Conv2D(128) + ReLU + MaxPooling
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

### Classes de classification

Le modèle classifie 7 types de matériaux:
- glass
- organic
- steel
- polythene
- plastic
- paper
- aluminium

## Utilisation

### Prérequis

1. **Données préprocessées**: Les données doivent avoir été préprocessées avec le pipeline de preprocessing (voir `docs/README.md`).

2. **Dépendances**: Installer les dépendances requises:
```bash
pip install -r requirements.txt
```

### Entraînement

#### Via DVC (recommandé)

Pour exécuter le pipeline complet incluant l'entraînement:

```bash
dvc repro train
```

#### Via script Python

Pour entraîner directement le modèle:

```bash
python scripts/train.py \
    --processed-data-dir data/processed \
    --metadata data/processed/processed_metadata.csv \
    --model-dir models/baseline \
    --batch-size 32 \
    --epochs 50 \
    --learning-rate 0.001 \
    --patience 10
```

#### Paramètres d'entraînement

- `--processed-data-dir`: Répertoire contenant les données préprocessées (défaut: `data/processed`)
- `--metadata`: Chemin vers le fichier CSV de métadonnées (défaut: `data/processed/processed_metadata.csv`)
- `--model-dir`: Répertoire pour sauvegarder le modèle (défaut: `models/baseline`)
- `--input-shape`: Forme d'entrée HEIGHT WIDTH CHANNELS (défaut: 64 64 1)
- `--num-classes`: Nombre de classes (défaut: 7)
- `--batch-size`: Taille des batches (défaut: 32)
- `--epochs`: Nombre maximum d'epochs (défaut: 50)
- `--learning-rate`: Taux d'apprentissage (défaut: 0.001)
- `--patience`: Patience pour l'early stopping (défaut: 10)
- `--model-name`: Nom du modèle (défaut: `baseline_model`)

### Fonctionnalités d'entraînement

#### Early Stopping

Le modèle utilise l'early stopping pour éviter le surapprentissage:
- Surveille la `val_loss`
- Arrête l'entraînement si aucune amélioration pendant `patience` epochs
- Restaure automatiquement les meilleurs poids

#### Model Checkpointing

Le meilleur modèle (basé sur `val_loss`) est sauvegardé automatiquement:
- Fichier: `{model_name}_best.keras`
- Sauvegarde uniquement si amélioration de la métrique surveillée

#### Réduction du Learning Rate

Le learning rate est réduit automatiquement sur plateau:
- Facteur de réduction: 0.5
- Patience: `patience / 2`
- Learning rate minimum: 1e-7

## Artéfacts générés

Après l'entraînement, les fichiers suivants sont créés dans `models/baseline/`:

1. **`baseline_model.keras`**: Modèle Keras sauvegardé (format .keras)
2. **`baseline_model_best.keras`**: Meilleur modèle (basé sur val_loss)
3. **`baseline_model_config.json`**: Configuration du modèle (hyperparamètres, forme d'entrée, etc.)
4. **`baseline_model_history.json`**: Historique d'entraînement (loss, accuracy par epoch)
5. **`baseline_model_class_mapping.json`**: Mapping des classes vers les indices
6. **`baseline_model_summary.json`**: Résumé de l'entraînement (métriques finales, date, config)

## Métriques

Le modèle suit les métriques suivantes pendant l'entraînement:

- **Loss**: Categorical cross-entropy
- **Accuracy**: Précision de classification
- **Precision**: Précision par classe
- **Recall**: Rappel par classe

## Hyperparamètres par défaut

- **Learning rate**: 0.001
- **Batch size**: 32
- **Epochs**: 50 (avec early stopping)
- **Optimizer**: Adam
- **Loss**: Categorical cross-entropy
- **Input shape**: (64, 64, 1)

## Intégration DVC

Le modèle est intégré dans le pipeline DVC:

```yaml
train:
  cmd: python scripts/train.py ...
  deps:
    - data/processed/train
    - data/processed/validation
    - data/processed/processed_metadata.csv
    - src/thermal_sensors/models/baseline_model.py
    - src/thermal_sensors/data_generator.py
    - src/thermal_sensors/scripts/train.py
  outs:
    - models/baseline
```

Les modèles sont versionnés avec DVC pour assurer la reproductibilité.

## Améliorations futures

- Architecture plus profonde (ResNet, EfficientNet)
- Augmentation de données plus agressive
- Fine-tuning de hyperparamètres
- Transfer learning depuis des modèles pré-entraînés
- Ensemble de modèles
