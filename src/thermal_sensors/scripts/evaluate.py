"""Script d'évaluation pour le modèle baseline.

Ce script évalue un modèle entraîné sur le jeu de test et génère
toutes les métriques d'évaluation : accuracy, precision, recall,
F1-score, matrice de confusion et rapport de classification.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from thermal_sensors.models.baseline_model import BaselineModel
from thermal_sensors.data_generator import ThermalDataGenerator, load_processed_metadata
from thermal_sensors.evaluation import evaluate_model

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Fonction principale du script d'évaluation."""
    parser = argparse.ArgumentParser(
        description="Évaluation du modèle baseline pour classification de matériaux thermiques"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/baseline",
        help="Répertoire contenant le modèle entraîné",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_model",
        help="Nom du modèle (sans extension)",
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
        "--output-dir",
        type=str,
        default="outputs",
        help="Répertoire de sortie pour les métriques",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille des batches pour l'évaluation (défaut: 32)",
    )

    args = parser.parse_args()

    # Convertir les chemins
    model_dir = Path(args.model_dir)
    metadata_path = Path(args.metadata)
    data_dir = Path(args.processed_data_dir)
    output_dir = Path(args.output_dir)

    # Vérifier que le modèle existe
    model_path = model_dir / f"{args.model_name}.keras"
    if not model_path.exists():
        logger.error(f"Modèle introuvable: {model_path}")
        logger.error("Assurez-vous que le modèle a été entraîné avant l'évaluation.")
        return 1

    # Charger le modèle
    logger.info(f"Chargement du modèle depuis: {model_dir}")
    try:
        baseline_model = BaselineModel.load(model_dir, model_name=args.model_name)
        model = baseline_model.get_model()
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}", exc_info=True)
        return 1

    # Charger la configuration du modèle pour obtenir input_shape et num_classes
    config_path = model_dir / f"{args.model_name}_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        input_shape = tuple(config.get("input_shape", (64, 64, 1)))
        num_classes = config.get("num_classes", 7)
    else:
        logger.warning("Fichier de configuration non trouvé, utilisation des valeurs par défaut")
        input_shape = (64, 64, 1)
        num_classes = 7

    # Charger le mapping des classes si disponible
    class_mapping_path = model_dir / f"{args.model_name}_class_mapping.json"
    class_to_idx: Optional[dict] = None
    if class_mapping_path.exists():
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        logger.info(f"Mapping des classes chargé: {len(class_to_idx)} classes")
    else:
        logger.warning("Fichier de mapping des classes non trouvé")

    # Créer la liste des noms de classes
    class_names = None
    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Charger les métadonnées
    logger.info("Chargement des métadonnées...")
    try:
        metadata = load_processed_metadata(metadata_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées: {e}", exc_info=True)
        return 1

    # Filtrer pour le jeu de test uniquement
    test_metadata = metadata[metadata["split"] == "test"].copy()

    if len(test_metadata) == 0:
        logger.error("Aucune donnée de test trouvée!")
        logger.error("Assurez-vous que le split 'test' existe dans les métadonnées.")
        return 1

    logger.info(f"Données de test: {len(test_metadata)} images")

    # Créer le générateur de données de test
    logger.info("Création du générateur de données de test...")
    test_gen = ThermalDataGenerator(
        data_dir=data_dir,
        metadata=test_metadata,
        batch_size=args.batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        class_to_idx=class_to_idx,
        shuffle=False,
    )

    logger.info(f"Nombre de batches de test: {len(test_gen)}")

    # Évaluer le modèle
    try:
        metrics = evaluate_model(
            model=model,
            test_generator=test_gen,
            output_dir=output_dir,
            class_names=class_names,
            model_name=args.model_name,
        )

        # Afficher un résumé
        logger.info("\n" + "=" * 60)
        logger.info("Résumé des métriques")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
        logger.info(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

