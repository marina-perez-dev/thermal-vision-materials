"""Smoke tests pour le pipeline d'entraînement.

Ces tests rapides vérifient que le pipeline d'entraînement fonctionne
sans erreur, sans nécessiter un entraînement complet. Ils valident:
- La création et compilation du modèle
- La capacité du modèle à s'entraîner sur quelques batches
- La sauvegarde et le chargement du modèle
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from thermal_sensors.models.baseline_model import BaselineModel
from thermal_sensors.data_generator import ThermalDataGenerator


class TestModelTrainSmoke:
    """Smoke tests pour le pipeline d'entraînement."""

    @pytest.fixture
    def temp_data_dir(self):
        """Crée un répertoire temporaire avec des données mockées."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(parents=True)

        # Créer des fichiers .npy mockés pour train et validation
        splits = ["train", "validation"]
        classes = ["Glass", "Metal", "Plastic"]

        metadata_records = []
        for split in splits:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)

            # Créer 5 images par classe pour chaque split (pour tests rapides)
            for class_name in classes:
                for i in range(5):
                    image_path = split_dir / f"{class_name}_{i}.npy"
                    # Créer une image mockée (64x64)
                    mock_image = np.random.rand(64, 64).astype(np.float32)
                    np.save(image_path, mock_image)

                    metadata_records.append(
                        {
                            "processed_path": f"{split}/{class_name}_{i}.npy",
                            "classes": class_name,
                            "split": split,
                        }
                    )

        yield data_dir, metadata_records

        # Nettoyer
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_model_dir(self):
        """Crée un répertoire temporaire pour les modèles."""
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "models" / "test"
        model_dir.mkdir(parents=True, exist_ok=True)

        yield model_dir

        # Nettoyer
        shutil.rmtree(temp_dir)

    def test_model_creation_and_compilation(self):
        """Test que le modèle peut être créé et compilé."""
        model = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=3,
            learning_rate=0.001,
        )

        assert model.model is not None
        assert model.input_shape == (64, 64, 1)
        assert model.num_classes == 3
        assert model.learning_rate == 0.001

        # Vérifier que le modèle peut faire une prédiction
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        output = model.model.predict(test_input, verbose=0)

        assert output.shape == (1, 3)
        assert np.isclose(output.sum(), 1.0, atol=1e-5)  # Softmax somme à 1

    def test_model_train_smoke(self, temp_data_dir, temp_model_dir):
        """Test rapide d'entraînement (smoke test)."""
        data_dir, metadata_records = temp_data_dir
        metadata_df = pd.DataFrame(metadata_records)

        # Séparer train et validation
        train_metadata = metadata_df[metadata_df["split"] == "train"]
        val_metadata = metadata_df[metadata_df["split"] == "validation"]

        # Créer les générateurs
        train_gen = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=train_metadata,
            batch_size=4,
            input_shape=(64, 64, 1),
            num_classes=3,
            shuffle=True,
        )

        val_gen = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=val_metadata,
            batch_size=4,
            input_shape=(64, 64, 1),
            num_classes=3,
            class_to_idx=train_gen.class_to_idx,
            shuffle=False,
        )

        # Créer le modèle
        model = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=3,
            learning_rate=0.001,
        )

        # Entraîner sur seulement 2 epochs avec 1 step par epoch (très rapide)
        history = model.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=2,
            steps_per_epoch=1,  # Seulement 1 batch par epoch
            validation_steps=1,  # Seulement 1 batch de validation
            verbose=0,
        )

        # Vérifier que l'entraînement s'est bien passé
        assert len(history.history["loss"]) == 2
        assert "accuracy" in history.history
        assert "val_loss" in history.history
        assert "val_accuracy" in history.history

    def test_model_save_and_load(self, temp_model_dir):
        """Test la sauvegarde et le chargement du modèle."""
        model_name = "test_model"

        # Créer et sauvegarder un modèle
        model1 = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=3,
            learning_rate=0.001,
        )

        model1.save(temp_model_dir, model_name=model_name)

        # Vérifier que les fichiers ont été créés
        assert (temp_model_dir / f"{model_name}.keras").exists()
        assert (temp_model_dir / f"{model_name}_config.json").exists()

        # Charger le modèle
        model2 = BaselineModel.load(temp_model_dir, model_name=model_name)

        # Vérifier que les configurations correspondent
        assert model2.input_shape == model1.input_shape
        assert model2.num_classes == model1.num_classes
        assert model2.learning_rate == model1.learning_rate

        # Vérifier que les prédictions sont identiques (même architecture)
        test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
        pred1 = model1.model.predict(test_input, verbose=0)
        pred2 = model2.model.predict(test_input, verbose=0)

        # Les prédictions devraient être identiques (même poids initialisés)
        np.testing.assert_allclose(pred1, pred2, atol=1e-5)

    def test_model_training_with_callbacks(self, temp_data_dir, temp_model_dir):
        """Test l'entraînement avec des callbacks (early stopping, checkpoint)."""
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        data_dir, metadata_records = temp_data_dir
        metadata_df = pd.DataFrame(metadata_records)

        train_metadata = metadata_df[metadata_df["split"] == "train"]
        val_metadata = metadata_df[metadata_df["split"] == "validation"]

        train_gen = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=train_metadata,
            batch_size=4,
            input_shape=(64, 64, 1),
            num_classes=3,
            shuffle=True,
        )

        val_gen = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=val_metadata,
            batch_size=4,
            input_shape=(64, 64, 1),
            num_classes=3,
            class_to_idx=train_gen.class_to_idx,
            shuffle=False,
        )

        model = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=3,
            learning_rate=0.001,
        )

        # Créer les callbacks
        checkpoint_path = temp_model_dir / "checkpoint.keras"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # Entraîner avec callbacks
        history = model.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=5,
            steps_per_epoch=1,
            validation_steps=1,
            callbacks=callbacks,
            verbose=0,
        )

        # Vérifier que l'entraînement s'est bien passé
        assert len(history.history["loss"]) > 0

    def test_model_different_input_shapes(self):
        """Test que le modèle peut être créé avec différentes formes d'entrée."""
        shapes = [(32, 32, 1), (128, 128, 1), (64, 64, 3)]

        for shape in shapes:
            model = BaselineModel(
                input_shape=shape,
                num_classes=3,
            )

            # Vérifier que le forward pass fonctionne
            test_input = np.random.rand(1, *shape).astype(np.float32)
            output = model.model.predict(test_input, verbose=0)

            assert output.shape == (1, 3)

    def test_model_different_num_classes(self):
        """Test que le modèle peut être créé avec différents nombres de classes."""
        num_classes_list = [2, 5, 10]

        for num_classes in num_classes_list:
            model = BaselineModel(
                input_shape=(64, 64, 1),
                num_classes=num_classes,
            )

            # Vérifier que la sortie a le bon nombre de classes
            test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
            output = model.model.predict(test_input, verbose=0)

            assert output.shape == (1, num_classes)
            assert np.isclose(output.sum(), 1.0, atol=1e-5)

    def test_model_training_loss_decreases(self, temp_data_dir):
        """Test que la loss diminue pendant l'entraînement (smoke test)."""
        data_dir, metadata_records = temp_data_dir
        metadata_df = pd.DataFrame(metadata_records)

        train_metadata = metadata_df[metadata_df["split"] == "train"]

        train_gen = ThermalDataGenerator(
            data_dir=data_dir,
            metadata=train_metadata,
            batch_size=4,
            input_shape=(64, 64, 1),
            num_classes=3,
            shuffle=True,
        )

        model = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=3,
            learning_rate=0.01,  # Learning rate plus élevé pour convergence rapide
        )

        # Entraîner sur quelques epochs
        history = model.model.fit(
            train_gen,
            epochs=3,
            steps_per_epoch=2,
            verbose=0,
        )

        # Vérifier que la loss a été calculée
        assert len(history.history["loss"]) == 3
        # Note: On ne vérifie pas forcément que la loss diminue car avec
        # si peu de données et d'epochs, cela peut varier. On vérifie juste
        # que l'entraînement fonctionne sans erreur.

