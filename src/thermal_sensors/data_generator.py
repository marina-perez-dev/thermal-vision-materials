"""Générateur de données pour l'entraînement du modèle baseline.

Ce module fournit un générateur de données pour charger les images
préprocessées et les préparer pour l'entraînement.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tensorflow import keras

logger = logging.getLogger(__name__)


class ThermalDataGenerator(keras.utils.Sequence):
    """Générateur de données pour les images thermiques préprocessées.

    Charge les images depuis les fichiers .npy et génère des batches
    pour l'entraînement du modèle.

    Attributes:
        data_dir: Répertoire contenant les données préprocessées
        metadata: DataFrame avec les métadonnées des images
        batch_size: Taille des batches
        input_shape: Forme d'entrée attendue (height, width, channels)
        num_classes: Nombre de classes
        class_to_idx: Mapping des classes vers les indices
        shuffle: Si True, mélange les données à chaque epoch
    """

    def __init__(
        self,
        data_dir: Path,
        metadata: pd.DataFrame,
        batch_size: int = 32,
        input_shape: Tuple[int, int, int] = (64, 64, 1),
        num_classes: int = 7,
        class_to_idx: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
    ):
        """Initialise le générateur de données.

        Args:
            data_dir: Répertoire contenant les données préprocessées
            metadata: DataFrame avec les métadonnées (doit contenir 'processed_path' et 'classes')
            batch_size: Taille des batches
            input_shape: Forme d'entrée (height, width, channels)
            num_classes: Nombre de classes
            class_to_idx: Mapping des classes vers les indices (si None, créé automatiquement)
            shuffle: Si True, mélange les données à chaque epoch
        """
        self.data_dir = Path(data_dir)
        self.metadata = metadata.copy()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shuffle = shuffle

        # Créer le mapping des classes si non fourni
        if class_to_idx is None:
            self.class_to_idx = self._create_class_mapping()
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Filtrer les métadonnées pour ne garder que les images valides
        self.metadata = self._filter_valid_images()

        logger.info(
            f"Générateur initialisé: {len(self.metadata)} images, "
            f"{len(self.class_to_idx)} classes, batch_size={batch_size}"
        )

    def _create_class_mapping(self) -> Dict[str, int]:
        """Crée le mapping des classes vers les indices.

        Returns:
            Dictionnaire {class_name: index}
        """
        # Extraire toutes les classes uniques
        all_classes = set()
        for classes_str in self.metadata["classes"]:
            if pd.notna(classes_str) and classes_str != "unknown":
                # Les classes peuvent être séparées par ", "
                for cls in str(classes_str).split(", "):
                    all_classes.add(cls.strip())

        # Créer le mapping (trié pour la reproductibilité)
        sorted_classes = sorted(all_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}

        logger.info(f"Mapping des classes créé: {class_to_idx}")

        return class_to_idx

    def _filter_valid_images(self) -> pd.DataFrame:
        """Filtre les images valides (qui existent et ont une classe valide).

        Returns:
            DataFrame filtré
        """
        valid_indices = []
        for idx, row in self.metadata.iterrows():
            image_path = self.data_dir / row["processed_path"]
            if not image_path.exists():
                logger.warning(f"Image introuvable: {image_path}")
                continue

            # Vérifier que la classe est valide
            classes_str = row.get("classes", "unknown")
            if pd.isna(classes_str) or classes_str == "unknown":
                logger.warning(f"Classe inconnue pour: {row.get('processed_path', 'unknown')}")
                continue

            # Prendre la première classe si plusieurs classes
            primary_class = str(classes_str).split(", ")[0].strip()
            if primary_class not in self.class_to_idx:
                logger.warning(f"Classe non mappée: {primary_class}")
                continue

            valid_indices.append(idx)

        return self.metadata.loc[valid_indices].reset_index(drop=True)

    def __len__(self) -> int:
        """Retourne le nombre de batches par epoch."""
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne un batch de données.

        Args:
            idx: Index du batch

        Returns:
            Tuple (X, y) où X est le batch d'images et y les labels encodés
        """
        # Calculer les indices pour ce batch
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.metadata))

        batch_metadata = self.metadata.iloc[start_idx:end_idx]

        # Charger les images et labels
        X = []
        y = []

        for _, row in batch_metadata.iterrows():
            # Charger l'image
            image_path = self.data_dir / row["processed_path"]
            try:
                image = np.load(image_path)
                # Ajouter la dimension de canal si nécessaire
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                # Redimensionner si nécessaire
                if image.shape[:2] != self.input_shape[:2]:
                    from PIL import Image
                    pil_img = Image.fromarray((image.squeeze() * 255).astype(np.uint8))
                    pil_img = pil_img.resize(self.input_shape[:2][::-1])
                    image = np.array(pil_img, dtype=np.float32) / 255.0
                    image = np.expand_dims(image, axis=-1)

                X.append(image)

                # Encoder le label (one-hot)
                classes_str = row["classes"]
                primary_class = str(classes_str).split(", ")[0].strip()
                class_idx = self.class_to_idx[primary_class]
                label = np.zeros(self.num_classes)
                label[class_idx] = 1.0
                y.append(label)

            except Exception as e:
                logger.error(f"Erreur lors du chargement de {image_path}: {e}")
                continue

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y

    def on_epoch_end(self) -> None:
        """Appelé à la fin de chaque epoch pour mélanger les données."""
        if self.shuffle:
            self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)

    def get_class_names(self) -> List[str]:
        """Retourne la liste des noms de classes dans l'ordre des indices.

        Returns:
            Liste des noms de classes
        """
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]


def load_processed_metadata(metadata_path: Path) -> pd.DataFrame:
    """Charge les métadonnées préprocessées.

    Args:
        metadata_path: Chemin vers le fichier CSV de métadonnées

    Returns:
        DataFrame avec les métadonnées
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables: {metadata_path}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Métadonnées chargées: {len(df)} images")

    return df

