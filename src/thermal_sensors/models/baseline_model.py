"""Architecture du modèle baseline pour la classification de matériaux thermiques.

Ce module implémente une architecture CNN simple avec 3 blocs convolutifs
suivis de couches denses pour la classification multi-classes.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics

logger = logging.getLogger(__name__)


class BaselineModel:
    """Modèle baseline CNN pour la classification de matériaux thermiques.

    Architecture:
    - Input: Images thermiques (64x64, single-channel par défaut)
    - 3 blocs convolutifs (Conv2D + ReLU + MaxPooling)
    - Couches denses pour la classification
    - Output: Probabilités de classes (softmax)

    Attributes:
        model: Modèle Keras compilé
        input_shape: Forme d'entrée (height, width, channels)
        num_classes: Nombre de classes de classification
        config: Configuration du modèle (hyperparamètres)
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (64, 64, 1),
        num_classes: int = 7,
        learning_rate: float = 0.001,
        config: Optional[Dict] = None,
    ):
        """Initialise le modèle baseline.

        Args:
            input_shape: Forme d'entrée (height, width, channels)
            num_classes: Nombre de classes de classification
            learning_rate: Taux d'apprentissage pour l'optimiseur
            config: Configuration additionnelle (optionnel)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.config = config or {}
        self.model = None

        self._build_model()

    def _build_model(self) -> None:
        """Construit l'architecture du modèle baseline."""
        logger.info(
            f"Construction du modèle baseline: input_shape={self.input_shape}, "
            f"num_classes={self.num_classes}"
        )

        # Couche d'entrée
        inputs = layers.Input(shape=self.input_shape, name="input")

        # Bloc convolutif 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="conv1",
        )(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

        # Bloc convolutif 2
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="conv2",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

        # Bloc convolutif 3
        x = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            name="conv3",
        )(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

        # Aplatir pour les couches denses
        x = layers.Flatten(name="flatten")(x)

        # Couches denses
        x = layers.Dense(128, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        x = layers.Dense(64, activation="relu", name="dense2")(x)
        x = layers.Dropout(0.5, name="dropout2")(x)

        # Couche de sortie (softmax pour classification multi-classes)
        outputs = layers.Dense(
            self.num_classes, activation="softmax", name="output"
        )(x)

        # Créer le modèle
        self.model = models.Model(inputs=inputs, outputs=outputs, name="baseline_model")

        # Compiler le modèle
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses.CategoricalCrossentropy(),
            metrics=[
                metrics.CategoricalAccuracy(name="accuracy"),
                metrics.Precision(name="precision"),
                metrics.Recall(name="recall"),
            ],
        )

        logger.info("Modèle baseline compilé avec succès")
        logger.info(f"Nombre de paramètres: {self.model.count_params():,}")

    def get_model(self) -> models.Model:
        """Retourne le modèle Keras.

        Returns:
            Modèle Keras compilé
        """
        return self.model

    def save(self, model_dir: Path, model_name: str = "baseline_model") -> None:
        """Sauvegarde le modèle et sa configuration.

        Args:
            model_dir: Répertoire de sauvegarde
            model_name: Nom du modèle (sans extension)
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le modèle Keras
        model_path = model_dir / f"{model_name}.keras"
        self.model.save(model_path)
        logger.info(f"Modèle sauvegardé: {model_path}")

        # Sauvegarder la configuration
        config = {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "total_params": int(self.model.count_params()),
            "model_name": model_name,
            **self.config,
        }

        config_path = model_dir / f"{model_name}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration sauvegardée: {config_path}")

    @classmethod
    def load(cls, model_dir: Path, model_name: str = "baseline_model") -> "BaselineModel":
        """Charge un modèle sauvegardé.

        Args:
            model_dir: Répertoire contenant le modèle
            model_name: Nom du modèle (sans extension)

        Returns:
            Instance de BaselineModel avec le modèle chargé
        """
        model_dir = Path(model_dir)
        model_path = model_dir / f"{model_name}.keras"
        config_path = model_dir / f"{model_name}_config.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable: {model_path}")

        # Charger la configuration
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        # Créer l'instance
        instance = cls(
            input_shape=tuple(config.get("input_shape", (64, 64, 1))),
            num_classes=config.get("num_classes", 7),
            learning_rate=config.get("learning_rate", 0.001),
            config=config,
        )

        # Charger les poids
        instance.model = models.load_model(model_path)
        logger.info(f"Modèle chargé depuis: {model_path}")

        return instance

    def summary(self) -> None:
        """Affiche un résumé de l'architecture du modèle."""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Modèle non construit")

