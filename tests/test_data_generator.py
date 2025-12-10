"""Tests pour le générateur de données."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from thermal_sensors.data_generator import ThermalDataGenerator, load_processed_metadata


def test_data_generator_creation():
    """Test la création d'un générateur de données."""
    # Créer des données de test
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        # Créer quelques images factices
        metadata_records = []
        for i in range(10):
            image_path = train_dir / f"image_{i}.npy"
            # Créer une image 64x64 normalisée
            image = np.random.rand(64, 64).astype(np.float32)
            np.save(image_path, image)

            metadata_records.append({
                "processed_path": f"train/image_{i}.npy",
                "classes": "glass" if i % 2 == 0 else "plastic",
                "split": "train",
            })

        metadata = pd.DataFrame(metadata_records)

        # Créer le générateur
        generator = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=metadata,
            batch_size=4,
            input_shape=(64, 64, 1),
            num_classes=7,
            shuffle=False,
        )

        assert len(generator) > 0
        assert len(generator.class_to_idx) > 0


def test_data_generator_batch():
    """Test qu'un batch peut être généré."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        # Créer quelques images factices
        metadata_records = []
        for i in range(5):
            image_path = train_dir / f"image_{i}.npy"
            image = np.random.rand(64, 64).astype(np.float32)
            np.save(image_path, image)

            metadata_records.append({
                "processed_path": f"train/image_{i}.npy",
                "classes": "glass",
                "split": "train",
            })

        metadata = pd.DataFrame(metadata_records)

        generator = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=metadata,
            batch_size=2,
            input_shape=(64, 64, 1),
            num_classes=7,
            shuffle=False,
        )

        # Obtenir un batch
        X, y = generator[0]

        # Vérifier les formes
        assert X.shape[0] <= 2  # batch_size
        assert X.shape[1:] == (64, 64, 1)
        assert y.shape[0] == X.shape[0]
        assert y.shape[1] == 7

        # Vérifier que les labels sont one-hot
        assert (y.sum(axis=1) == 1.0).all()


def test_data_generator_class_mapping():
    """Test que le mapping des classes est créé correctement."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        train_dir = data_dir / "train"
        train_dir.mkdir(parents=True)

        classes = ["glass", "plastic", "steel"]
        metadata_records = []

        for i, cls in enumerate(classes * 2):
            image_path = train_dir / f"image_{i}.npy"
            image = np.random.rand(64, 64).astype(np.float32)
            np.save(image_path, image)

            metadata_records.append({
                "processed_path": f"train/image_{i}.npy",
                "classes": cls,
                "split": "train",
            })

        metadata = pd.DataFrame(metadata_records)

        generator = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=metadata,
            batch_size=2,
            input_shape=(64, 64, 1),
            num_classes=len(classes),
            shuffle=False,
        )

        # Vérifier que toutes les classes sont mappées
        assert len(generator.class_to_idx) == len(classes)
        for cls in classes:
            assert cls in generator.class_to_idx

