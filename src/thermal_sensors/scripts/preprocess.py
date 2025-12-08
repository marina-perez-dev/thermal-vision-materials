"""Pipeline de preprocessing pour les images thermiques.

Ce module implémente les fonctions de preprocessing pour les données de capteurs thermiques,
incluant la normalisation, le redimensionnement, la calibration et l'augmentation de données.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Types de normalisation supportés
NormalizationMethod = Literal["min_max", "z_score"]
NormalizationScope = Literal["per_sensor", "per_frame"]


def load_thermal_image(image_path: Path) -> np.ndarray:
    """Charge une image thermique depuis le disque.

    Args:
        image_path: Chemin vers l'image

    Returns:
        Array numpy représentant l'image (H, W) ou (H, W, C)

    Raises:
        FileNotFoundError: Si l'image n'existe pas
        ValueError: Si l'image ne peut pas être chargée
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    try:
        # Charger l'image avec PIL
        img = Image.open(image_path)
        # Convertir en array numpy
        img_array = np.array(img, dtype=np.float32)

        # Si l'image est en niveaux de gris (2D), on la garde telle quelle
        # Si elle est en couleur (3D), on peut la convertir en niveaux de gris
        if len(img_array.shape) == 3:
            # Convertir RGB en niveaux de gris (moyenne pondérée)
            if img_array.shape[2] == 3:
                img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
            elif img_array.shape[2] == 4:
                # RGBA: prendre seulement RGB
                img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])

        logger.debug(f"Image chargée: {img_array.shape}, dtype: {img_array.dtype}, range: [{img_array.min():.2f}, {img_array.max():.2f}]")
        return img_array

    except Exception as e:
        raise ValueError(f"Erreur lors du chargement de l'image {image_path}: {e}") from e


def normalize_min_max(
    image: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Normalise une image avec la méthode min-max.

    La normalisation min-max transforme les valeurs dans la plage [0, 1] :
    normalized = (x - min) / (max - min)

    Args:
        image: Array numpy de l'image
        min_val: Valeur minimale (si None, utilise le min de l'image)
        max_val: Valeur maximale (si None, utilise le max de l'image)

    Returns:
        Image normalisée dans [0, 1]
    """
    if min_val is None:
        min_val = image.min()
    if max_val is None:
        max_val = image.max()

    # Éviter la division par zéro
    if max_val == min_val:
        logger.warning("Image avec valeurs constantes, retour de zéros")
        return np.zeros_like(image)

    normalized = (image - min_val) / (max_val - min_val)
    # S'assurer que les valeurs sont bien dans [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized


def normalize_z_score(
    image: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """Normalise une image avec la méthode z-score (standardisation).

    La normalisation z-score centre et réduit les données :
    normalized = (x - mean) / std

    Args:
        image: Array numpy de l'image
        mean: Moyenne (si None, calcule la moyenne de l'image)
        std: Écart-type (si None, calcule l'écart-type de l'image)

    Returns:
        Image normalisée (moyenne ~0, écart-type ~1)
    """
    if mean is None:
        mean = image.mean()
    if std is None:
        std = image.std()

    # Éviter la division par zéro
    if std == 0:
        logger.warning("Image avec écart-type nul, retour de zéros")
        return np.zeros_like(image)

    normalized = (image - mean) / std
    return normalized


def normalize_image(
    image: np.ndarray,
    method: NormalizationMethod = "min_max",
    scope: NormalizationScope = "per_frame",
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalise une image thermique selon la méthode et la portée spécifiées.

    Args:
        image: Array numpy de l'image
        method: Méthode de normalisation ("min_max" ou "z_score")
        scope: Portée de normalisation ("per_frame" ou "per_sensor")
        stats: Statistiques pré-calculées (pour normalisation per_sensor)

    Returns:
        Tuple (image normalisée, statistiques utilisées)
    """
    if scope == "per_frame":
        # Normalisation par frame: utiliser les stats de cette image
        if method == "min_max":
            normalized = normalize_min_max(image)
            stats_used = {
                "min": float(image.min()),
                "max": float(image.max()),
                "method": "min_max",
                "scope": "per_frame",
            }
        elif method == "z_score":
            normalized = normalize_z_score(image)
            stats_used = {
                "mean": float(image.mean()),
                "std": float(image.std()),
                "method": "z_score",
                "scope": "per_frame",
            }
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")

    elif scope == "per_sensor":
        # Normalisation par capteur: utiliser les stats globales
        if stats is None:
            raise ValueError("stats doit être fourni pour normalisation per_sensor")

        if method == "min_max":
            normalized = normalize_min_max(
                image, min_val=stats["min"], max_val=stats["max"]
            )
            stats_used = stats.copy()
        elif method == "z_score":
            normalized = normalize_z_score(
                image, mean=stats["mean"], std=stats["std"]
            )
            stats_used = stats.copy()
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
    else:
        raise ValueError(f"Portée de normalisation inconnue: {scope}")

    return normalized, stats_used


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int], method: str = "bilinear"
) -> np.ndarray:
    """Redimensionne une image thermique.

    Args:
        image: Array numpy de l'image (H, W)
        target_size: Taille cible (height, width)
        method: Méthode d'interpolation ("bilinear", "nearest", "bicubic")

    Returns:
        Image redimensionnée
    """
    # Convertir en PIL Image pour le redimensionnement
    pil_image = Image.fromarray(image.astype(np.uint8) if image.max() <= 255 else image)

    # Mapper les méthodes d'interpolation
    interpolation_map = {
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "bicubic": Image.BICUBIC,
    }

    interpolation = interpolation_map.get(method, Image.BILINEAR)

    # Redimensionner
    resized = pil_image.resize(target_size[::-1], interpolation)  # PIL utilise (width, height)

    # Reconvertir en numpy
    resized_array = np.array(resized, dtype=np.float32)

    # Préserver la plage de valeurs originale si nécessaire
    if image.max() > 255:
        # Si l'image originale était en float, normaliser la sortie
        resized_array = resized_array / 255.0 * image.max()

    return resized_array


def calibrate_thermal_image(
    image: np.ndarray,
    offset: float = 0.0,
    gain: float = 1.0,
    temperature_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Applique une calibration thermique à l'image.

    La calibration permet d'ajuster les valeurs pour correspondre à des températures réelles.
    calibration = (image * gain) + offset

    Args:
        image: Array numpy de l'image
        offset: Décalage à appliquer
        gain: Facteur de gain à appliquer
        temperature_range: Plage de température (min, max) pour normalisation optionnelle

    Returns:
        Image calibrée
    """
    calibrated = (image * gain) + offset

    # Si une plage de température est fournie, normaliser dans cette plage
    if temperature_range is not None:
        temp_min, temp_max = temperature_range
        calibrated = np.clip(calibrated, temp_min, temp_max)
        # Normaliser dans [0, 1] basé sur la plage de température
        calibrated = (calibrated - temp_min) / (temp_max - temp_min)

    return calibrated


def augment_flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Applique un retournement horizontal à l'image.

    Args:
        image: Array numpy de l'image (H, W)

    Returns:
        Image retournée horizontalement
    """
    return np.fliplr(image)


def augment_flip_vertical(image: np.ndarray) -> np.ndarray:
    """Applique un retournement vertical à l'image.

    Args:
        image: Array numpy de l'image (H, W)

    Returns:
        Image retournée verticalement
    """
    return np.flipud(image)


def augment_random_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int],
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Applique un crop aléatoire à l'image.

    Args:
        image: Array numpy de l'image (H, W)
        crop_size: Taille du crop (height, width)
        random_state: Seed pour la reproductibilité

    Returns:
        Image croppée
    """
    if random_state is not None:
        np.random.seed(random_state)

    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Vérifier que le crop est plus petit que l'image
    if crop_h > h or crop_w > w:
        logger.warning(
            f"Crop size {crop_size} plus grand que l'image {image.shape[:2]}, "
            "retour de l'image originale"
        )
        return image

    # Position aléatoire pour le crop
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)

    # Extraire le crop
    if len(image.shape) == 2:
        cropped = image[top : top + crop_h, left : left + crop_w]
    else:
        cropped = image[top : top + crop_h, left : left + crop_w, :]

    return cropped


def apply_augmentation(
    image: np.ndarray,
    augmentations: List[str],
    crop_size: Optional[Tuple[int, int]] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Applique une série d'augmentations à l'image.

    Args:
        image: Array numpy de l'image
        augmentations: Liste des augmentations à appliquer ("flip_h", "flip_v", "crop")
        crop_size: Taille du crop (requis si "crop" est dans augmentations)
        random_state: Seed pour la reproductibilité

    Returns:
        Image augmentée
    """
    augmented = image.copy()

    for aug in augmentations:
        if aug == "flip_h":
            augmented = augment_flip_horizontal(augmented)
        elif aug == "flip_v":
            augmented = augment_flip_vertical(augmented)
        elif aug == "crop":
            if crop_size is None:
                raise ValueError("crop_size doit être fourni pour l'augmentation 'crop'")
            augmented = augment_random_crop(augmented, crop_size, random_state)
        else:
            logger.warning(f"Augmentation inconnue ignorée: {aug}")

    return augmented


def stratified_split(
    df: pd.DataFrame,
    target_column: str = "classes",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Effectue un split stratifié train/validation/test en préservant la distribution des classes.

    Le split stratifié garantit que chaque split contient approximativement la même
    proportion de chaque classe que le dataset original.

    Args:
        df: DataFrame avec les métadonnées des images
        target_column: Colonne contenant les classes (ou liste de classes)
        train_ratio: Proportion pour l'entraînement (défaut: 0.7)
        val_ratio: Proportion pour la validation (défaut: 0.15)
        test_ratio: Proportion pour le test (défaut: 0.15)
        random_state: Seed pour la reproductibilité

    Returns:
        Tuple (train_df, val_df, test_df)

    Raises:
        ValueError: Si les ratios ne somment pas à 1.0
    """
    # Vérifier que les ratios somment à 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Les ratios doivent sommer à 1.0, mais somment à {total_ratio}"
        )

    # Extraire la première classe de chaque ligne (pour le split stratifié)
    # Si plusieurs classes, on prend la première
    def get_primary_class(classes_str: str) -> str:
        if pd.isna(classes_str) or classes_str == "unknown":
            return "unknown"
        # Prendre la première classe si plusieurs classes séparées par ", "
        return classes_str.split(", ")[0].strip()

    df = df.copy()
    df["primary_class"] = df[target_column].apply(get_primary_class)

    # Premier split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["primary_class"],
        random_state=random_state,
    )

    # Deuxième split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df["primary_class"],
        random_state=random_state,
    )

    # Supprimer la colonne temporaire
    train_df = train_df.drop(columns=["primary_class"])
    val_df = val_df.drop(columns=["primary_class"])
    test_df = test_df.drop(columns=["primary_class"])

    logger.info("Split stratifié effectué:")
    logger.info(f"  Train: {len(train_df)} images ({100*len(train_df)/len(df):.1f}%)")
    logger.info(f"  Validation: {len(val_df)} images ({100*len(val_df)/len(df):.1f}%)")
    logger.info(f"  Test: {len(test_df)} images ({100*len(test_df)/len(df):.1f}%)")

    return train_df, val_df, test_df


def compute_global_stats(
    images: List[np.ndarray],
    method: NormalizationMethod = "min_max",
) -> Dict[str, float]:
    """Calcule les statistiques globales pour la normalisation per_sensor.

    Args:
        images: Liste des arrays d'images
        method: Méthode de normalisation

    Returns:
        Dictionnaire avec les statistiques
    """
    if method == "min_max":
        all_mins = [img.min() for img in images]
        all_maxs = [img.max() for img in images]
        return {
            "min": float(np.min(all_mins)),
            "max": float(np.max(all_maxs)),
            "method": "min_max",
            "scope": "per_sensor",
        }
    elif method == "z_score":
        # Moyenne globale et écart-type global
        all_values = np.concatenate([img.flatten() for img in images])
        return {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "method": "z_score",
            "scope": "per_sensor",
        }
    else:
        raise ValueError(f"Méthode de normalisation inconnue: {method}")


def save_processed_image(
    image: np.ndarray, output_path: Path, format: str = "npy"
) -> None:
    """Sauvegarde une image préprocessée.

    Args:
        image: Array numpy de l'image
        output_path: Chemin de sortie
        format: Format de sauvegarde ("npy" pour numpy, "png" pour image)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npy":
        # Sauvegarder en format numpy (préserve les valeurs float)
        np.save(output_path, image)
    elif format == "png":
        # Sauvegarder en PNG (convertir en uint8)
        if image.max() <= 1.0:
            # Image normalisée [0, 1], convertir en [0, 255]
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image_uint8, mode="L")
        pil_image.save(output_path)
    else:
        raise ValueError(f"Format de sauvegarde inconnu: {format}")


def main(
    input_metadata: str = "data/raw/roboflow_metadata.csv",
    output_dir: str = "data/processed",
    raw_data_dir: str = "data/raw/roboflow",
    normalization_method: NormalizationMethod = "min_max",
    normalization_scope: NormalizationScope = "per_frame",
    target_size: Optional[Tuple[int, int]] = (64, 64),
    enable_augmentation: bool = False,
    augmentation_list: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    output_format: str = "npy",
) -> int:
    """Fonction principale du pipeline de preprocessing.

    Args:
        input_metadata: Chemin vers le fichier CSV de métadonnées Roboflow
        output_dir: Répertoire de sortie pour les données préprocessées
        raw_data_dir: Répertoire contenant les images brutes
        normalization_method: Méthode de normalisation ("min_max" ou "z_score")
        normalization_scope: Portée de normalisation ("per_frame" ou "per_sensor")
        target_size: Taille cible pour le redimensionnement (None pour garder la taille originale)
        enable_augmentation: Activer l'augmentation de données
        augmentation_list: Liste des augmentations à appliquer
        train_ratio: Proportion pour l'entraînement
        val_ratio: Proportion pour la validation
        test_ratio: Proportion pour le test
        random_state: Seed pour la reproductibilité
        output_format: Format de sauvegarde ("npy" ou "png")

    Returns:
        Code de sortie (0 = succès, 1 = erreur)
    """
    logger.info("=" * 60)
    logger.info("Pipeline de Preprocessing - Images Thermiques")
    logger.info("=" * 60)

    # Étape 1: Charger les métadonnées
    logger.info("\n1. Chargement des métadonnées...")
    metadata_path = Path(input_metadata)
    if not metadata_path.exists():
        logger.error(f"Fichier de métadonnées introuvable: {metadata_path}")
        return 1

    try:
        df = pd.read_csv(metadata_path)
        logger.info(f"  {len(df)} images chargées depuis {metadata_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées: {e}")
        return 1

    # Étape 2: Split stratifié (si les splits ne sont pas déjà définis)
    logger.info("\n2. Préparation des splits...")
    if "split" in df.columns:
        # Les splits sont déjà définis (Roboflow)
        logger.info("  Utilisation des splits existants du dataset Roboflow")
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "valid"].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        # Créer un split stratifié
        logger.info("  Création d'un split stratifié...")
        train_df, val_df, test_df = stratified_split(
            df,
            target_column="classes",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

    # Étape 3: Calculer les statistiques globales si normalisation per_sensor
    global_stats = None
    if normalization_scope == "per_sensor":
        logger.info("\n3. Calcul des statistiques globales (per_sensor)...")
        raw_data_path = Path(raw_data_dir)
        sample_images = []
        sample_count = min(100, len(df))  # Échantillonner pour accélérer

        for idx, row in df.head(sample_count).iterrows():
            image_path = raw_data_path / row["image_path"]
            if image_path.exists():
                try:
                    img = load_thermal_image(image_path)
                    sample_images.append(img)
                except Exception as e:
                    logger.warning(f"Impossible de charger {image_path}: {e}")

        if sample_images:
            global_stats = compute_global_stats(sample_images, normalization_method)
            logger.info(f"  Statistiques calculées: {global_stats}")
        else:
            logger.warning("  Aucune image trouvée pour calculer les stats, passage en per_frame")
            normalization_scope = "per_frame"

    # Étape 4: Preprocessing des images
    logger.info("\n4. Preprocessing des images...")
    output_path = Path(output_dir)
    raw_data_path = Path(raw_data_dir)

    # Préparer les listes pour les nouvelles métadonnées
    processed_metadata = []

    def process_split(split_name: str, split_df: pd.DataFrame) -> None:
        """Traite un split d'images."""
        logger.info(f"\n  Traitement du split '{split_name}' ({len(split_df)} images)...")
        split_output_dir = output_path / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in split_df.iterrows():
            try:
                # Charger l'image
                image_path = raw_data_path / row["image_path"]
                if not image_path.exists():
                    logger.warning(f"Image introuvable: {image_path}, ignorée")
                    continue

                image = load_thermal_image(image_path)

                # Redimensionner si nécessaire
                if target_size is not None:
                    image = resize_image(image, target_size)

                # Normaliser
                normalized, stats_used = normalize_image(
                    image,
                    method=normalization_method,
                    scope=normalization_scope,
                    stats=global_stats,
                )

                # Calibration (optionnelle, par défaut pas de calibration)
                processed_image = normalized

                # Augmentation (optionnelle)
                if enable_augmentation and augmentation_list:
                    processed_image = apply_augmentation(
                        processed_image,
                        augmentation_list,
                        crop_size=target_size,
                        random_state=random_state + idx,
                    )

                # Sauvegarder l'image préprocessée
                output_filename = f"{Path(row['filename']).stem}.{output_format}"
                output_file = split_output_dir / output_filename
                save_processed_image(processed_image, output_file, format=output_format)

                # Enregistrer les métadonnées
                processed_metadata.append(
                    {
                        "original_path": str(row["image_path"]),
                        "processed_path": str(split_output_dir.relative_to(output_path) / output_filename),
                        "split": split_name,
                        "filename": output_filename,
                        "classes": row.get("classes", "unknown"),
                        "normalization_method": stats_used.get("method", normalization_method),
                        "normalization_scope": stats_used.get("scope", normalization_scope),
                        "target_size": f"{target_size[0]}x{target_size[1]}" if target_size else "original",
                        "augmentation": str(augmentation_list) if enable_augmentation else "none",
                    }
                )

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {row.get('image_path', 'unknown')}: {e}")
                continue

    # Traiter chaque split
    process_split("train", train_df)
    process_split("validation", val_df)
    process_split("test", test_df)

    # Étape 5: Sauvegarder les métadonnées préprocessées
    logger.info("\n5. Sauvegarde des métadonnées préprocessées...")
    processed_metadata_df = pd.DataFrame(processed_metadata)
    processed_metadata_path = output_path / "processed_metadata.csv"
    processed_metadata_df.to_csv(processed_metadata_path, index=False)
    logger.info(f"  Métadonnées sauvegardées: {processed_metadata_path}")

    # Étape 6: Créer un fichier de statistiques
    logger.info("\n6. Création des statistiques...")
    stats = {
        "total_processed": len(processed_metadata),
        "splits": {},
        "normalization": {
            "method": normalization_method,
            "scope": normalization_scope,
            "global_stats": global_stats,
        },
        "target_size": f"{target_size[0]}x{target_size[1]}" if target_size else "original",
        "augmentation": {
            "enabled": enable_augmentation,
            "augmentations": augmentation_list if enable_augmentation else [],
        },
    }

    for split_name in ["train", "validation", "test"]:
        split_count = len([m for m in processed_metadata if m["split"] == split_name])
        stats["splits"][split_name] = {
            "count": split_count,
            "percentage": round(100 * split_count / len(processed_metadata), 2) if processed_metadata else 0,
        }

    stats_path = output_path / "preprocessing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"  Statistiques sauvegardées: {stats_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing terminé avec succès!")
    logger.info(f"Images préprocessées: {output_path}")
    logger.info(f"Total: {len(processed_metadata)} images")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline pour images thermiques"
    )
    parser.add_argument(
        "--input-metadata",
        type=str,
        default="data/raw/roboflow_metadata.csv",
        help="Chemin vers le fichier CSV de métadonnées",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Répertoire de sortie pour les données préprocessées",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw/roboflow",
        help="Répertoire contenant les images brutes",
    )
    parser.add_argument(
        "--normalization-method",
        type=str,
        choices=["min_max", "z_score"],
        default="min_max",
        help="Méthode de normalisation",
    )
    parser.add_argument(
        "--normalization-scope",
        type=str,
        choices=["per_frame", "per_sensor"],
        default="per_frame",
        help="Portée de normalisation",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[64, 64],
        metavar=("HEIGHT", "WIDTH"),
        help="Taille cible pour le redimensionnement (défaut: 64 64)",
    )
    parser.add_argument(
        "--enable-augmentation",
        action="store_true",
        help="Activer l'augmentation de données",
    )
    parser.add_argument(
        "--augmentation-list",
        type=str,
        nargs="+",
        choices=["flip_h", "flip_v", "crop"],
        help="Liste des augmentations à appliquer",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion pour l'entraînement (défaut: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion pour la validation (défaut: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion pour le test (défaut: 0.15)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed pour la reproductibilité (défaut: 42)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["npy", "png"],
        default="npy",
        help="Format de sauvegarde (défaut: npy)",
    )

    args = parser.parse_args()

    # Convertir target_size en tuple
    target_size = tuple(args.target_size) if args.target_size else None

    exit_code = main(
        input_metadata=args.input_metadata,
        output_dir=args.output_dir,
        raw_data_dir=args.raw_data_dir,
        normalization_method=args.normalization_method,
        normalization_scope=args.normalization_scope,
        target_size=target_size,
        enable_augmentation=args.enable_augmentation,
        augmentation_list=args.augmentation_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        output_format=args.output_format,
    )

    import sys

    sys.exit(exit_code)

