# Guide DVC - Pipeline de Données et Modèles

Ce guide explique comment utiliser DVC (Data Version Control) pour gérer le pipeline de données, l'entraînement et l'évaluation des modèles dans ce projet.

## Vue d'ensemble

DVC permet de versionner les données, les modèles et les résultats de manière reproductible. Le pipeline complet comprend 4 étapes principales :

1. **Ingestion** : Collecte et préparation des données brutes
2. **Preprocessing** : Normalisation et transformation des données
3. **Entraînement** : Entraînement du modèle baseline
4. **Évaluation** : Calcul des métriques et génération de rapports

## État attendu du projet

### Structure DVC

```
.dvc/                      # Configuration DVC (dvc init)
├── config                 # Configuration DVC
└── cache/                 # Cache local des données versionnées

dvc.yaml                   # Définition du pipeline
dvc.lock                   # Verrouillage des versions (généré après dvc repro)

data/
├── raw/                   # Données brutes (trackées par DVC)
│   └── roboflow.dvc       # Référence DVC vers les données
└── processed/             # Données préprocessées (trackées par DVC)

models/
└── baseline/              # Modèles entraînés (trackés par DVC)

outputs/                   # Résultats d'évaluation (trackés par DVC)
```

## Pipeline DVC

### Structure du pipeline

Le pipeline est défini dans `dvc.yaml` avec les étapes suivantes :

```yaml
stages:
  ingest_roboflow:    # Ingestion du dataset Roboflow
  preprocess:         # Preprocessing des images
  train:             # Entraînement du modèle
  evaluate:          # Évaluation et métriques
```

### Dépendances entre étapes

```
ingest_roboflow → preprocess → train → evaluate
```

Chaque étape dépend des sorties de l'étape précédente.

## Commandes essentielles

### Exécution du pipeline

#### Exécuter tout le pipeline

```powershell
# Exécuter toutes les étapes nécessaires
dvc repro
```

#### Exécuter une étape spécifique

```powershell
# Exécuter uniquement l'ingestion
dvc repro ingest_roboflow

# Exécuter le preprocessing (et ses dépendances si nécessaire)
dvc repro preprocess

# Exécuter l'entraînement (et ses dépendances)
dvc repro train

# Exécuter l'évaluation (et ses dépendances)
dvc repro evaluate
```

#### Forcer la réexécution

```powershell
# Forcer la réexécution même si les outputs existent
dvc repro --force

# Forcer une étape spécifique
dvc repro train --force
```

### Vérification de l'état

#### Vérifier l'état du pipeline

```powershell
# Vérifier quelles étapes doivent être réexécutées
dvc status

# Vérifier l'état détaillé
dvc status --verbose
```

#### Vérifier les dépendances

```powershell
# Afficher le graphe de dépendances
dvc dag

# Afficher le graphe avec visualisation
dvc dag --dot | dot -Tpng -o pipeline.png
```

### Gestion des données

#### Ajouter des données au tracking DVC

```powershell
# Ajouter un répertoire de données
dvc add data/raw/roboflow

# Ajouter un fichier
dvc add data/raw/roboflow_metadata.csv
```

#### Récupérer les données versionnées

```powershell
# Récupérer toutes les données depuis le remote
dvc pull

# Récupérer des données spécifiques
dvc pull data/raw/roboflow.dvc

# Récupérer avec vérification
dvc pull --verify
```

#### Pousser les données vers le remote

```powershell
# Pousser toutes les données
dvc push

# Pousser des données spécifiques
dvc push data/raw/roboflow.dvc

# Pousser avec verbose
dvc push -v
```

### Inspection des données

#### Lister les fichiers trackés

```powershell
# Lister tous les fichiers trackés
dvc list .

# Lister un répertoire spécifique
dvc list data/processed
```

#### Vérifier l'intégrité

```powershell
# Vérifier l'intégrité des fichiers trackés
dvc cache dir

# Nettoyer le cache
dvc cache clean
```

## Exemples d'utilisation

### Scénario 1 : Premier setup du projet

```powershell
# 1. Cloner le repository
git clone <repository-url>
cd thermal-vision-materials

# 2. Créer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Récupérer les données versionnées
dvc pull

# 5. Exécuter le pipeline complet
dvc repro
```

### Scénario 2 : Réentraîner le modèle après modification

```powershell
# 1. Modifier le code du modèle (ex: src/thermal_sensors/models/baseline_model.py)
# ... faire les modifications ...

# 2. Vérifier quelles étapes seront réexécutées
dvc status

# 3. Réexécuter le pipeline (DVC détectera automatiquement les changements)
dvc repro

# Ou forcer la réexécution depuis l'entraînement
dvc repro train --force
```

### Scénario 3 : Tester une modification du preprocessing

```powershell
# 1. Modifier le script de preprocessing
# ... modifier src/thermal_sensors/scripts/preprocess.py ...

# 2. Réexécuter le preprocessing et les étapes suivantes
dvc repro preprocess

# DVC réexécutera automatiquement train et evaluate
```

### Scénario 4 : Comparer deux versions du modèle

```powershell
# 1. Voir l'historique des métriques
git log --oneline outputs/baseline_model_metrics.json

# 2. Checkout une version précédente
git checkout <commit-hash> dvc.lock

# 3. Récupérer les données de cette version
dvc checkout

# 4. Comparer les métriques
# ... comparer les fichiers outputs/baseline_model_metrics.json ...
```

### Scénario 5 : Nettoyer et recommencer

```powershell
# 1. Supprimer tous les outputs (garder les données brutes)
dvc remove dvc.yaml --outs

# 2. Réexécuter le pipeline
dvc repro
```

## Configuration du remote

### Configuration d'un remote local

```powershell
# Ajouter un remote local (pour partage entre machines)
dvc remote add -d local-storage /path/to/shared/storage

# Vérifier la configuration
dvc remote list
```

### Configuration d'un remote cloud (ex: S3, GCS, Azure)

```powershell
# Exemple avec S3
dvc remote add -d myremote s3://mybucket/dvc-cache

# Configurer les credentials (dans .dvc/config.local, non versionné)
# Ou utiliser des variables d'environnement
$env:AWS_ACCESS_KEY_ID = "your-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret"
```

## Détails des étapes du pipeline

### Étape 1: ingest_roboflow

**Commande**:
```powershell
dvc repro ingest_roboflow
```

**Description**: 
- Extrait le dataset Roboflow depuis le fichier ZIP
- Génère les métadonnées (`roboflow_metadata.csv`)
- Calcule les statistiques (`roboflow_statistics.json`)

**Inputs**:
- `data/raw/roboflow/` (dataset extrait)
- `src/thermal_sensors/scripts/ingest_roboflow_dataset.py`

**Outputs**:
- `data/raw/roboflow_metadata.csv`
- `data/raw/roboflow_statistics.json`

### Étape 2: preprocess

**Commande**:
```powershell
dvc repro preprocess
```

**Description**:
- Normalise les images (min-max par frame)
- Redimensionne à 64x64 pixels
- Génère les splits train/validation/test
- Crée les métadonnées préprocessées

**Inputs**:
- `data/raw/roboflow_metadata.csv`
- `data/raw/roboflow/`
- `src/thermal_sensors/scripts/preprocess.py`

**Outputs**:
- `data/processed/train/`
- `data/processed/validation/`
- `data/processed/test/`
- `data/processed/processed_metadata.csv`
- `data/processed/preprocessing_stats.json`

### Étape 3: train

**Commande**:
```powershell
dvc repro train
```

**Description**:
- Entraîne le modèle baseline CNN
- Utilise early stopping et checkpointing
- Sauvegarde le meilleur modèle et l'historique

**Inputs**:
- `data/processed/train/`
- `data/processed/validation/`
- `data/processed/processed_metadata.csv`
- `src/thermal_sensors/models/baseline_model.py`
- `src/thermal_sensors/data_generator.py`
- `src/thermal_sensors/scripts/train.py`

**Outputs**:
- `models/baseline/` (contenant tous les fichiers du modèle)

### Étape 4: evaluate

**Commande**:
```powershell
dvc repro evaluate
```

**Description**:
- Évalue le modèle sur le test set
- Calcule les métriques (accuracy, precision, recall, F1)
- Génère la matrice de confusion
- Crée le rapport de classification

**Inputs**:
- `models/baseline/`
- `data/processed/test/`
- `data/processed/processed_metadata.csv`
- Scripts d'évaluation

**Outputs**:
- `outputs/baseline_model_metrics.json`
- `outputs/baseline_model_confusion_matrix.png`
- `outputs/baseline_model_classification_report.txt`

## Bonnes pratiques

### Versioning

1. **Toujours committer `dvc.yaml` et `dvc.lock`**
   ```powershell
   git add dvc.yaml dvc.lock
   git commit -m "Update DVC pipeline"
   ```

2. **Ne pas committer les fichiers volumineux**
   - Les fichiers trackés par DVC sont dans `.gitignore`
   - Seuls les fichiers `.dvc` (références) sont versionnés

3. **Taguer les versions importantes**
   ```powershell
   git tag -a v1.0 -m "Baseline model v1.0"
   dvc tag v1.0
   ```

### Reproducibilité

1. **Vérifier la reproductibilité**
   ```powershell
   # Nettoyer et réexécuter
   dvc remove dvc.yaml --outs
   dvc repro
   ```

2. **Documenter les changements**
   - Mettre à jour les commits avec des messages clairs
   - Documenter les modifications dans les journaux de sprint

### Performance

1. **Utiliser le cache efficacement**
   - DVC ne réexécute que les étapes modifiées
   - Le cache évite les recalculs inutiles

2. **Nettoyer le cache si nécessaire**
   ```powershell
   # Voir la taille du cache
   dvc cache dir
   
   # Nettoyer les fichiers non utilisés
   dvc cache clean
   ```

## Dépannage

### Problème : `dvc repro` ne détecte pas les changements

**Solution**:
```powershell
# Forcer la réexécution
dvc repro --force

# Vérifier les dépendances
dvc status --verbose
```

### Problème : Erreur "Output already exists"

**Solution**:
```powershell
# Supprimer les outputs et réexécuter
dvc remove dvc.yaml --outs
dvc repro
```

### Problème : Données manquantes après `git pull`

**Solution**:
```powershell
# Récupérer les données depuis le remote
dvc pull

# Vérifier l'état
dvc status
```

### Problème : Cache corrompu

**Solution**:
```powershell
# Nettoyer le cache
dvc cache clean

# Réexécuter
dvc repro
```

## Intégration avec Git

### Workflow recommandé

1. **Développement local**
   ```powershell
   # Modifier le code
   # ... modifications ...
   
   # Tester le pipeline
   dvc repro
   
   # Committer les changements
   git add dvc.yaml dvc.lock src/...
   git commit -m "Description des changements"
   ```

2. **Partage avec l'équipe**
   ```powershell
   # Pousser le code
   git push
   
   # Pousser les données (si remote configuré)
   dvc push
   ```

3. **Récupération sur une autre machine**
   ```powershell
   git pull
   dvc pull
   dvc repro  # Si nécessaire
   ```
