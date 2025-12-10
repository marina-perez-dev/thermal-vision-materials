"""Tests pour le modèle baseline.

Ces tests vérifient que le modèle peut être créé, compilé et utilisé correctement.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from thermal_sensors.models.baseline_model import BaselineModel


def test_baseline_model_creation():
    """Test la création du modèle baseline."""
    model = BaselineModel(
        input_shape=(64, 64, 1),
        num_classes=7,
        learning_rate=0.001,
    )

    assert model.model is not None
    assert model.input_shape == (64, 64, 1)
    assert model.num_classes == 7
    assert model.learning_rate == 0.001


def test_baseline_model_forward_pass():
    """Test qu'un forward pass fonctionne correctement."""
    model = BaselineModel(
        input_shape=(64, 64, 1),
        num_classes=7,
    )

    # Créer une image d'entrée factice
    test_image = np.random.rand(1, 64, 64, 1).astype(np.float32)

    # Faire une prédiction
    predictions = model.model.predict(test_image, verbose=0)

    # Vérifier la forme de la sortie
    assert predictions.shape == (1, 7)
    # Vérifier que les probabilités somment à ~1
    assert np.isclose(predictions.sum(), 1.0, atol=1e-5)
    # Vérifier que toutes les probabilités sont positives
    assert (predictions >= 0).all()


def test_baseline_model_save_load():
    """Test la sauvegarde et le chargement du modèle."""
    # Créer un répertoire temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"

        # Créer et sauvegarder un modèle
        model1 = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=7,
        )
        model1.save(model_dir, model_name="test_model")

        # Vérifier que les fichiers ont été créés
        assert (model_dir / "test_model.keras").exists()
        assert (model_dir / "test_model_config.json").exists()

        # Charger le modèle
        model2 = BaselineModel.load(model_dir, model_name="test_model")

        # Vérifier que les configurations correspondent
        assert model2.input_shape == model1.input_shape
        assert model2.num_classes == model1.num_classes
        assert model2.learning_rate == model1.learning_rate

        # Vérifier que les prédictions sont identiques
        test_image = np.random.rand(1, 64, 64, 1).astype(np.float32)
        pred1 = model1.model.predict(test_image, verbose=0)
        pred2 = model2.model.predict(test_image, verbose=0)

        np.testing.assert_allclose(pred1, pred2, atol=1e-5)


def test_baseline_model_summary():
    """Test que le modèle peut afficher son résumé."""
    model = BaselineModel(
        input_shape=(64, 64, 1),
        num_classes=7,
    )

    # Vérifier que le modèle a des paramètres
    assert model.model.count_params() > 0


def test_baseline_model_different_input_shapes():
    """Test que le modèle peut être créé avec différentes formes d'entrée."""
    shapes = [(32, 32, 1), (128, 128, 1), (64, 64, 3)]

    for shape in shapes:
        model = BaselineModel(
            input_shape=shape,
            num_classes=7,
        )

        # Vérifier que le forward pass fonctionne
        test_image = np.random.rand(1, *shape).astype(np.float32)
        predictions = model.model.predict(test_image, verbose=0)

        assert predictions.shape == (1, 7)


def test_baseline_model_different_num_classes():
    """Test que le modèle peut être créé avec différents nombres de classes."""
    for num_classes in [3, 5, 10]:
        model = BaselineModel(
            input_shape=(64, 64, 1),
            num_classes=num_classes,
        )

        # Vérifier que la sortie a le bon nombre de classes
        test_image = np.random.rand(1, 64, 64, 1).astype(np.float32)
        predictions = model.model.predict(test_image, verbose=0)

        assert predictions.shape == (1, num_classes)

