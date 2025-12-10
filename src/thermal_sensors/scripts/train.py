"""Script d'entraînement pour le modèle baseline.

Ce script implémente le pipeline d'entraînement complet avec:
- Early stopping pour éviter le surapprentissage
- Model checkpointing pour sauvegarder le meilleur modèle
- Suivi des métriques d'entraînement
- Sauvegarde de l'historique d'entraînement
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from thermal_sensors.models.baseline_model import BaselineModel
from thermal_sensors.data_generator import ThermalDataGenerator, load_processed_metadata

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_callbacks(
    model_dir: Path,
    model_name: str = "baseline_model",
    patience: int = 10,
    monitor: str = "val_loss",
    mode: str = "min",
) -> list:
    """Crée les callbacks pour l'entraînement.

    Args:
        model_dir: Répertoire pour sauvegarder les modèles
        model_name: Nom du modèle
        patience: Patience pour l'early stopping (nombre d'epochs sans amélioration)
        monitor: Métrique à surveiller
        mode: Mode de surveillance ("min" ou "max")

    Returns:
        Liste des callbacks
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stopping)

    # Model checkpointing
    checkpoint_path = model_dir / f"{model_name}_best.keras"
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        verbose=1,
    )
    callbacks.append(checkpoint)

    # Réduction du learning rate sur plateau
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-7,
        verbose=1,
    )
    callbacks.append(reduce_lr)

    logger.info(f"Callbacks créés: early_stopping (patience={patience}), checkpoint, reduce_lr")

    return callbacks


def train_model(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    data_dir: Path,
    model_dir: Path,
    input_shape: tuple = (64, 64, 1),
    num_classes: int = 7,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 10,
    model_name: str = "baseline_model",
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Dict:
    """Entraîne le modèle baseline.

    Args:
        train_metadata: Métadonnées d'entraînement
        val_metadata: Métadonnées de validation
        data_dir: Répertoire contenant les données préprocessées
        model_dir: Répertoire pour sauvegarder le modèle
        input_shape: Forme d'entrée (height, width, channels)
        num_classes: Nombre de classes
        batch_size: Taille des batches
        epochs: Nombre maximum d'epochs
        learning_rate: Taux d'apprentissage
        patience: Patience pour l'early stopping
        model_name: Nom du modèle
        class_to_idx: Mapping des classes vers les indices

    Returns:
        Dictionnaire avec l'historique d'entraînement et les métriques
    """
    logger.info("=" * 60)
    logger.info("Démarrage de l'entraînement du modèle baseline")
    logger.info("=" * 60)

    # Créer les générateurs de données
    logger.info("\n1. Création des générateurs de données...")
    train_gen = ThermalDataGenerator(
        data_dir=data_dir,
        metadata=train_metadata,
        batch_size=batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
        shuffle=True,
    )

    val_gen = ThermalDataGenerator(
        data_dir=data_dir,
        metadata=val_metadata,
        batch_size=batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        class_to_idx=train_gen.class_to_idx,  # Utiliser le même mapping
        shuffle=False,
    )

    logger.info(f"  Train: {len(train_gen)} batches ({len(train_gen.metadata)} images)")
    logger.info(f"  Validation: {len(val_gen)} batches ({len(val_gen.metadata)} images)")

    # Créer le modèle
    logger.info("\n2. Création du modèle...")
    baseline_model = BaselineModel(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )
    model = baseline_model.get_model()
    baseline_model.summary()

    # Créer les callbacks
    logger.info("\n3. Configuration des callbacks...")
    callbacks = create_callbacks(
        model_dir=model_dir,
        model_name=model_name,
        patience=patience,
    )

    # Entraîner le modèle
    logger.info("\n4. Démarrage de l'entraînement...")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Sauvegarder le modèle final
    logger.info("\n5. Sauvegarde du modèle...")
    baseline_model.save(model_dir, model_name=model_name)

    # Sauvegarder le mapping des classes
    class_mapping_path = model_dir / f"{model_name}_class_mapping.json"
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_to_idx, f, indent=2, ensure_ascii=False)
    logger.info(f"  Mapping des classes sauvegardé: {class_mapping_path}")

    # Préparer l'historique pour la sauvegarde
    history_dict = {
        "epoch": list(range(1, len(history.history["loss"]) + 1)),
        "loss": [float(x) for x in history.history["loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    }

    # Ajouter les autres métriques si présentes
    for key in history.history.keys():
        if key not in history_dict:
            history_dict[key] = [float(x) for x in history.history[key]]

    # Sauvegarder l'historique
    history_path = model_dir / f"{model_name}_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"  Historique sauvegardé: {history_path}")

    # Calculer les métriques finales
    final_metrics = {
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_accuracy": float(
            history.history["val_accuracy"][np.argmin(history.history["val_loss"])]
        ),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "total_epochs": len(history.history["loss"]),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Entraînement terminé!")
    logger.info(f"  Meilleur epoch: {final_metrics['best_epoch']}")
    logger.info(f"  Meilleure val_loss: {final_metrics['best_val_loss']:.4f}")
    logger.info(f"  Meilleure val_accuracy: {final_metrics['best_val_accuracy']:.4f}")
    logger.info("=" * 60)

    return {
        "history": history_dict,
        "metrics": final_metrics,
        "class_mapping": train_gen.class_to_idx,
    }


def main() -> int:
    """Fonction principale du script d'entraînement."""
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle baseline pour classification de matériaux thermiques"
    )
    parser.add_argument(
        "--processed-data-dir",
        type=str,
        default="data/processed",
        help="Répertoire contenant les données préprocessées",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/processed_metadata.csv",
        help="Chemin vers le fichier CSV de métadonnées préprocessées",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/baseline",
        help="Répertoire pour sauvegarder le modèle",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=[64, 64, 1],
        metavar=("HEIGHT", "WIDTH", "CHANNELS"),
        help="Forme d'entrée (défaut: 64 64 1)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Nombre de classes (défaut: 7)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille des batches (défaut: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre maximum d'epochs (défaut: 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d'apprentissage (défaut: 0.001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience pour l'early stopping (défaut: 10)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_model",
        help="Nom du modèle (défaut: baseline_model)",
    )

    args = parser.parse_args()

    # Convertir les chemins
    data_dir = Path(args.processed_data_dir)
    metadata_path = Path(args.metadata)
    model_dir = Path(args.model_dir)
    input_shape = tuple(args.input_shape)

    # Charger les métadonnées
    logger.info("Chargement des métadonnées...")
    metadata = load_processed_metadata(metadata_path)

    # Séparer train et validation
    train_metadata = metadata[metadata["split"] == "train"].copy()
    val_metadata = metadata[metadata["split"] == "validation"].copy()

    if len(train_metadata) == 0:
        logger.error("Aucune donnée d'entraînement trouvée!")
        return 1

    if len(val_metadata) == 0:
        logger.error("Aucune donnée de validation trouvée!")
        return 1

    logger.info(f"Train: {len(train_metadata)} images")
    logger.info(f"Validation: {len(val_metadata)} images")

    # Entraîner le modèle
    try:
        results = train_model(
            train_metadata=train_metadata,
            val_metadata=val_metadata,
            data_dir=data_dir,
            model_dir=model_dir,
            input_shape=input_shape,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            model_name=args.model_name,
        )

        # Sauvegarder un résumé des résultats
        summary_path = model_dir / f"{args.model_name}_summary.json"
        summary = {
            "training_date": datetime.now().isoformat(),
            "config": {
                "input_shape": input_shape,
                "num_classes": args.num_classes,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "patience": args.patience,
            },
            "metrics": results["metrics"],
            "class_mapping": results["class_mapping"],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Résumé sauvegardé: {summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

