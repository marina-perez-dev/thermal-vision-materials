"""Script d'ingestion pour le dataset Roboflow 'Thermal Waste Detection'.

Ce script traite le dataset téléchargé depuis Roboflow et le prépare pour le pipeline.
Le dataset peut être au format YOLO ou Classification multiclass (fichiers _classes.csv).
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
import zipfile
import shutil
import pandas as pd
from PIL import Image

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_roboflow_zip(zip_path: Path, output_dir: Path) -> Optional[Path]:
    """Extrait l'archive Roboflow.

    Args:
        zip_path: Chemin vers le fichier ZIP
        output_dir: Répertoire de destination

    Returns:
        Chemin vers le répertoire extrait ou None
    """
    try:
        logger.info(f"Extraction de {zip_path} vers {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        # Chercher le répertoire principal (généralement nommé comme le dataset)
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if extracted_dirs:
            main_dir = extracted_dirs[0]
            logger.info(f"Dataset extrait dans: {main_dir}")
            return main_dir

        return output_dir

    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {e}")
        return None


def find_dataset_structure(dataset_dir: Path) -> Dict[str, Path]:
    """Trouve la structure du dataset Roboflow.

    Args:
        dataset_dir: Répertoire du dataset extrait

    Returns:
        Dictionnaire avec les chemins (train, valid, test, data.yaml)
    """
    structure = {
        "train": None,
        "valid": None,
        "test": None,
        "data_yaml": None,
    }

    # Chercher les répertoires train/valid/test
    for split in ["train", "valid", "test", "val"]:
        split_dir = dataset_dir / split
        if split_dir.exists():
            if split == "val":
                structure["valid"] = split_dir
            else:
                structure[split] = split_dir

    # Chercher le fichier data.yaml (format YOLO)
    yaml_files = list(dataset_dir.glob("data.yaml")) + list(dataset_dir.glob("*.yaml"))
    if yaml_files:
        structure["data_yaml"] = yaml_files[0]

    return structure


def load_yolo_classes(yaml_path: Path) -> Optional[List[str]]:
    """Charge les classes depuis le fichier data.yaml.

    Args:
        yaml_path: Chemin vers le fichier data.yaml

    Returns:
        Liste des noms de classes ou None
    """
    try:
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        classes = data.get("names", [])
        if isinstance(classes, dict):
            # Si c'est un dictionnaire {0: 'class1', 1: 'class2', ...}
            classes = [classes[i] for i in sorted(classes.keys())]
        elif isinstance(classes, list):
            # Si c'est déjà une liste
            pass

        logger.info(f"Classes trouvées: {classes}")
        return classes

    except ImportError:
        logger.warning("PyYAML non installé, tentative de parsing manuel")
        # Parsing manuel basique
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Recherche simple des noms
                if "names:" in content:
                    # Extraction basique
                    pass
        except Exception as e:
            logger.error(f"Erreur lors du parsing YAML: {e}")

    except Exception as e:
        logger.error(f"Erreur lors du chargement des classes: {e}")

    return None


def scan_classification_dataset(split_dir: Path) -> List[Dict]:
    """Scanne un répertoire de classification multiclass et collecte les métadonnées.

    Args:
        split_dir: Répertoire du split (train/valid/test)

    Returns:
        Liste des métadonnées des images
    """
    records = []
    
    # Chercher le fichier _classes.csv
    classes_csv = split_dir / "_classes.csv"
    
    if not classes_csv.exists():
        logger.warning(f"Fichier _classes.csv introuvable: {classes_csv}")
        return records
    
    # Charger les annotations depuis le CSV
    try:
        df_annotations = pd.read_csv(classes_csv)
        logger.info(f"Chargement des annotations depuis {classes_csv}")
        
        # Les colonnes de classes sont toutes sauf 'filename'
        class_columns = [col for col in df_annotations.columns if col != "filename"]
        
        # Formats d'images supportés
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        
        logger.info(f"Scan de {split_dir.name}...")
        
        # Parcourir toutes les images du répertoire
        for image_path in split_dir.rglob("*"):
            if image_path.suffix.lower() in image_extensions and image_path.is_file():
                filename = image_path.name
                
                # Chercher les annotations pour cette image
                image_annotations = df_annotations[df_annotations["filename"] == filename]
                
                image_classes = []
                if not image_annotations.empty:
                    row = image_annotations.iloc[0]
                    for class_col in class_columns:
                        if row[class_col] == 1:
                            image_classes.append(class_col)
                
                records.append(
                    {
                        "image_path": str(image_path.relative_to(split_dir.parent)),
                        "split": split_dir.name,
                        "filename": filename,
                        "classes": ", ".join(image_classes) if image_classes else "unknown",
                        "class_count": len(image_classes),
                        "label_path": str(classes_csv.relative_to(split_dir.parent)),
                    }
                )
        
        logger.info(f"  {split_dir.name}: {len(records)} images trouvées")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des annotations: {e}")
    
    return records


def scan_yolo_dataset(split_dir: Path, classes: Optional[List[str]] = None) -> List[Dict]:
    """Scanne un répertoire YOLO et collecte les métadonnées.

    Args:
        split_dir: Répertoire du split (train/valid/test)
        classes: Liste des classes (optionnel)

    Returns:
        Liste des métadonnées des images
    """
    records = []
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        logger.warning(f"Répertoire images introuvable: {images_dir}")
        return records

    # Formats d'images supportés
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    logger.info(f"Scan de {split_dir.name}: {images_dir}")

    for image_path in images_dir.rglob("*"):
        if image_path.suffix.lower() in image_extensions and image_path.is_file():
            # Chercher le fichier d'annotation correspondant
            label_path = None
            if labels_dir.exists():
                label_file = labels_dir / f"{image_path.stem}.txt"
                if label_file.exists():
                    label_path = label_file

            # Extraire les classes depuis les annotations
            image_classes = []
            if label_path and label_path.exists():
                try:
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if classes and class_id < len(classes):
                                    class_name = classes[class_id]
                                    if class_name not in image_classes:
                                        image_classes.append(class_name)
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture de {label_path}: {e}")

            records.append(
                {
                    "image_path": str(image_path.relative_to(split_dir.parent)),
                    "split": split_dir.name,
                    "filename": image_path.name,
                    "classes": ", ".join(image_classes) if image_classes else "unknown",
                    "class_count": len(image_classes),
                    "label_path": str(label_path.relative_to(split_dir.parent))
                    if label_path
                    else None,
                }
            )

    logger.info(f"  {split_dir.name}: {len(records)} images trouvées")
    return records


def create_metadata_file(records: List[Dict], output_path: Path) -> None:
    """Crée un fichier CSV avec les métadonnées.

    Args:
        records: Liste des métadonnées
        output_path: Chemin de sortie
    """
    os.makedirs(output_path.parent, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"Métadonnées sauvegardées: {output_path} ({len(records)} entrées)")


def create_statistics(records: List[Dict], output_path: Path) -> None:
    """Crée un fichier JSON avec les statistiques.

    Args:
        records: Liste des métadonnées
        output_path: Chemin de sortie
    """
    stats = {
        "total_images": len(records),
        "splits": {},
        "classes": {},
    }

    df = pd.DataFrame(records)

    # Statistiques par split
    if "split" in df.columns:
        split_counts = df["split"].value_counts()
        for split, count in split_counts.items():
            stats["splits"][split] = {
                "count": int(count),
                "percentage": round(100 * count / len(records), 2),
            }

    # Statistiques par classe
    if "classes" in df.columns:
        all_classes = []
        for classes_str in df["classes"]:
            if classes_str and classes_str != "unknown":
                all_classes.extend([c.strip() for c in classes_str.split(",")])

        if all_classes:
            class_counts = pd.Series(all_classes).value_counts()
            for class_name, count in class_counts.items():
                stats["classes"][class_name] = {
                    "count": int(count),
                    "percentage": round(100 * count / len(all_classes), 2),
                }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Statistiques sauvegardées: {output_path}")


def main(
    zip_path: Optional[str] = None,
    output_dir: str = "data/raw/roboflow",
    target_dir: str = "data/raw",
) -> int:
    """Fonction principale d'ingestion du dataset Roboflow.

    Args:
        zip_path: Chemin vers le fichier ZIP téléchargé (optionnel si dataset déjà extrait)
        output_dir: Répertoire pour extraire le dataset ou contenant le dataset déjà extrait
        target_dir: Répertoire de destination pour les métadonnées

    Returns:
        Code de sortie (0 = succès, 1 = erreur)
    """
    output_path = Path(output_dir)
    target_path = Path(target_dir)

    logger.info("=" * 60)
    logger.info("Ingestion du dataset Roboflow: Thermal Waste Detection")
    logger.info("=" * 60)

    # Étape 1: Extraire le ZIP ou utiliser le répertoire existant
    extracted_dir = None
    if zip_path:
        zip_file = Path(zip_path)
        if not zip_file.exists():
            logger.error(f"Fichier ZIP introuvable: {zip_file}")
            logger.info(f"Assurez-vous que le fichier est dans: {zip_file.absolute()}")
            return 1
        
        logger.info("\n1. Extraction de l'archive...")
        extracted_dir = extract_roboflow_zip(zip_file, output_path)
        
        if not extracted_dir:
            logger.error("Échec de l'extraction")
            return 1
    else:
        # Utiliser le répertoire existant
        logger.info("\n1. Utilisation du dataset déjà extrait...")
        if output_path.exists() and output_path.is_dir():
            # Chercher les sous-répertoires train/valid/test
            if any((output_path / split).exists() for split in ["train", "valid", "test"]):
                extracted_dir = output_path
            else:
                # Peut-être qu'il y a un sous-répertoire
                subdirs = [d for d in output_path.iterdir() if d.is_dir()]
                if subdirs:
                    extracted_dir = subdirs[0]
                else:
                    extracted_dir = output_path
        else:
            logger.error(f"Répertoire introuvable: {output_path}")
            return 1
    
    logger.info(f"Dataset trouvé dans: {extracted_dir}")

    # Étape 2: Analyser la structure
    logger.info("\n2. Analyse de la structure du dataset...")
    structure = find_dataset_structure(extracted_dir)

    # Étape 3: Charger les classes (pour format YOLO)
    classes = None
    if structure["data_yaml"]:
        logger.info("\n3. Chargement des classes depuis data.yaml...")
        classes = load_yolo_classes(structure["data_yaml"])
    else:
        logger.info("\n3. Format classification détecté (pas de data.yaml)")

    # Étape 4: Scanner les images
    logger.info("\n4. Scan des images...")
    all_records = []
    
    # Détecter le format du dataset
    # Vérifier si c'est un format classification (fichiers _classes.csv)
    is_classification_format = False
    for split_name, split_dir in structure.items():
        if split_name != "data_yaml" and split_dir and split_dir.exists():
            classes_csv = split_dir / "_classes.csv"
            if classes_csv.exists():
                is_classification_format = True
                break
    
    # Scanner selon le format détecté
    for split_name, split_dir in structure.items():
        if split_name != "data_yaml" and split_dir and split_dir.exists():
            if is_classification_format:
                records = scan_classification_dataset(split_dir)
            else:
                records = scan_yolo_dataset(split_dir, classes)
            all_records.extend(records)

    if not all_records:
        logger.error("Aucune image trouvée dans le dataset")
        return 1

    # Étape 5: Créer les fichiers de métadonnées
    logger.info("\n5. Création des métadonnées...")
    metadata_file = target_path / "roboflow_metadata.csv"
    create_metadata_file(all_records, metadata_file)

    # Étape 6: Créer les statistiques
    stats_file = target_path / "roboflow_statistics.json"
    create_statistics(all_records, stats_file)

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion terminée avec succès!")
    logger.info(f"Dataset extrait dans: {extracted_dir}")
    logger.info(f"Métadonnées: {metadata_file}")
    logger.info(f"Statistiques: {stats_file}")
    logger.info(f"Total: {len(all_records)} images")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingérer le dataset Roboflow 'Thermal Waste Detection'"
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Chemin vers le fichier ZIP téléchargé (optionnel si dataset déjà extrait)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/roboflow",
        help="Répertoire pour extraire le dataset",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw",
        help="Répertoire de destination pour les métadonnées",
    )

    args = parser.parse_args()

    exit_code = main(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        target_dir=args.target_dir,
    )
    sys.exit(exit_code)

