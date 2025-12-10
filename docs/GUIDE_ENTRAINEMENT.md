# Guide d'entraînement du modèle baseline

Ce guide vous explique étape par étape comment tester et exécuter l'entraînement du modèle baseline.

## Prérequis

### 1. Installation des dépendances

Assurez-vous d'avoir installé toutes les dépendances:

```bash
pip install -r requirements.txt
```

Cela installera notamment TensorFlow/Keras nécessaire pour l'entraînement.

### 2. Vérification des données préprocessées

Les données doivent avoir été préprocessées. Vérifiez que les répertoires suivants existent:

```bash
# Vérifier la structure des données préprocessées
ls data/processed/
# Doit contenir: train/, validation/, test/, processed_metadata.csv, preprocessing_stats.json
```

Si les données ne sont pas préprocessées, exécutez d'abord:

```bash
dvc repro preprocess
```

## Étape 1: Test de création du modèle

Avant d'entraîner, testons que le modèle peut être créé correctement.

### Option A: Script de test Python

```bash
python scripts/test_model_creation.py
```

Ce script va:
- Créer un modèle baseline
- Afficher son architecture
- Tester un forward pass
- Tester la sauvegarde/chargement

## Étape 2: Vérification du générateur de données

Testez que le générateur de données fonctionne correctement:

```bash
pytest tests/test_data_generator.py -v
```
## Étape 3: Entraînement du modèle

### Méthode 1: Via DVC (recommandé)

Pour exécuter le pipeline complet avec reproductibilité:

```bash
# Exécuter uniquement l'entraînement (si preprocessing déjà fait)
dvc repro train

# Ou exécuter tout le pipeline depuis le début
dvc repro
```

## Étape 4: Suivi de l'entraînement

Pendant l'entraînement, vous verrez:

```
Epoch 1/50
...
loss: 1.9456 - accuracy: 0.2857 - val_loss: 1.8234 - val_accuracy: 0.3500
...
Epoch 2/50
...
```

### Interprétation

- **loss**: Perte d'entraînement (doit diminuer)
- **accuracy**: Précision d'entraînement (doit augmenter)
- **val_loss**: Perte de validation (doit diminuer)
- **val_accuracy**: Précision de validation (doit augmenter)

### Early Stopping

Si l'entraînement s'arrête avant 50 epochs, c'est normal! L'early stopping détecte qu'il n'y a plus d'amélioration et arrête l'entraînement pour éviter le surapprentissage.
