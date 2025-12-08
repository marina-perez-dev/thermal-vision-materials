"""Tests unitaires pour le pipeline de preprocessing.

Ces tests valident les fonctions de preprocessing, notamment:
- Les formes de sortie après preprocessing
- Les plages de valeurs après normalisation
- La cohérence des types de données
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from thermal_sensors.scripts.preprocess import (
    load_thermal_image,
    normalize_min_max,
    normalize_z_score,
    normalize_image,
    resize_image,
    calibrate_thermal_image,
    augment_flip_horizontal,
    augment_flip_vertical,
    augment_random_crop,
    apply_augmentation,
    stratified_split,
    compute_global_stats,
)


class TestNormalization:
    """Tests pour les fonctions de normalisation."""

    def test_normalize_min_max_basic(self):
        """Test de normalisation min-max basique."""
        # Créer une image de test avec valeurs [10, 20, 30, 40, 50]
        image = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        normalized = normalize_min_max(image)

        # Vérifier que les valeurs sont dans [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)

        # Vérifier que la forme est préservée
        assert normalized.shape == image.shape

    def test_normalize_min_max_with_constants(self):
        """Test de normalisation min-max avec valeurs constantes."""
        # Image avec valeurs constantes
        image = np.ones((10, 10)) * 5.0

        normalized = normalize_min_max(image)

        # Devrait retourner des zéros (évite division par zéro)
        assert np.allclose(normalized, 0.0)

    def test_normalize_min_max_custom_range(self):
        """Test de normalisation min-max avec plage personnalisée."""
        image = np.array([[10.0, 20.0], [30.0, 40.0]])

        normalized = normalize_min_max(image, min_val=10.0, max_val=40.0)

        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)

    def test_normalize_z_score_basic(self):
        """Test de normalisation z-score basique."""
        image = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])

        normalized = normalize_z_score(image)

        # Vérifier que la moyenne est proche de 0
        assert normalized.mean() == pytest.approx(0.0, abs=1e-3)
        # Vérifier que l'écart-type est proche de 1
        assert normalized.std() == pytest.approx(1.0, abs=1e-3)

        # Vérifier que la forme est préservée
        assert normalized.shape == image.shape

    def test_normalize_z_score_with_constants(self):
        """Test de normalisation z-score avec valeurs constantes."""
        image = np.ones((10, 10)) * 5.0

        normalized = normalize_z_score(image)

        # Devrait retourner des zéros (évite division par zéro)
        assert np.allclose(normalized, 0.0)

    def test_normalize_image_per_frame_min_max(self):
        """Test de normalisation per_frame avec min_max."""
        image = np.array([[10.0, 20.0], [30.0, 40.0]])

        normalized, stats = normalize_image(image, method="min_max", scope="per_frame")

        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max() == pytest.approx(1.0, abs=1e-6)
        assert stats["method"] == "min_max"
        assert stats["scope"] == "per_frame"
        assert "min" in stats
        assert "max" in stats

    def test_normalize_image_per_frame_z_score(self):
        """Test de normalisation per_frame avec z_score."""
        image = np.array([[10.0, 20.0], [30.0, 40.0]])

        normalized, stats = normalize_image(image, method="z_score", scope="per_frame")

        assert normalized.mean() == pytest.approx(0.0, abs=1e-3)
        assert stats["method"] == "z_score"
        assert stats["scope"] == "per_frame"
        assert "mean" in stats
        assert "std" in stats

    def test_normalize_image_per_sensor(self):
        """Test de normalisation per_sensor."""
        image = np.array([[10.0, 20.0], [30.0, 40.0]])
        stats = {"min": 0.0, "max": 100.0, "method": "min_max", "scope": "per_sensor"}

        normalized, stats_used = normalize_image(
            image, method="min_max", scope="per_sensor", stats=stats
        )

        assert stats_used == stats
        # Les valeurs devraient être normalisées selon les stats globales
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestResize:
    """Tests pour le redimensionnement d'images."""

    def test_resize_image_basic(self):
        """Test de redimensionnement basique."""
        # Créer une image 100x100
        image = np.random.rand(100, 100).astype(np.float32)

        resized = resize_image(image, target_size=(64, 64))

        assert resized.shape == (64, 64)
        assert resized.dtype == np.float32

    def test_resize_image_preserves_range(self):
        """Test que le redimensionnement préserve la plage de valeurs."""
        # Image normalisée [0, 1]
        image = np.random.rand(50, 50).astype(np.float32)

        resized = resize_image(image, target_size=(32, 32))

        # Les valeurs devraient rester dans une plage raisonnable
        assert resized.min() >= -1.0  # Peut être légèrement négatif après interpolation
        assert resized.max() <= 2.0  # Peut être légèrement supérieur à 1


class TestCalibration:
    """Tests pour la calibration thermique."""

    def test_calibrate_thermal_image_basic(self):
        """Test de calibration basique."""
        image = np.array([[10.0, 20.0], [30.0, 40.0]])

        calibrated = calibrate_thermal_image(image, offset=5.0, gain=2.0)

        # Vérifier la formule: (image * gain) + offset
        expected = (image * 2.0) + 5.0
        assert np.allclose(calibrated, expected)

    def test_calibrate_thermal_image_with_range(self):
        """Test de calibration avec plage de température."""
        image = np.array([[10.0, 20.0], [30.0, 40.0]])

        calibrated = calibrate_thermal_image(
            image, temperature_range=(0.0, 50.0)
        )

        # Les valeurs devraient être dans [0, 1] après normalisation
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0


class TestAugmentation:
    """Tests pour l'augmentation de données."""

    def test_augment_flip_horizontal(self):
        """Test de retournement horizontal."""
        image = np.array([[1, 2, 3], [4, 5, 6]])

        flipped = augment_flip_horizontal(image)

        expected = np.array([[3, 2, 1], [6, 5, 4]])
        assert np.array_equal(flipped, expected)

    def test_augment_flip_vertical(self):
        """Test de retournement vertical."""
        image = np.array([[1, 2], [3, 4], [5, 6]])

        flipped = augment_flip_vertical(image)

        expected = np.array([[5, 6], [3, 4], [1, 2]])
        assert np.array_equal(flipped, expected)

    def test_augment_random_crop(self):
        """Test de crop aléatoire."""
        image = np.random.rand(100, 100).astype(np.float32)

        cropped = augment_random_crop(image, crop_size=(50, 50), random_state=42)

        assert cropped.shape == (50, 50)

    def test_augment_random_crop_too_large(self):
        """Test de crop avec taille trop grande."""
        image = np.random.rand(50, 50).astype(np.float32)

        # Crop plus grand que l'image
        cropped = augment_random_crop(image, crop_size=(100, 100), random_state=42)

        # Devrait retourner l'image originale
        assert cropped.shape == image.shape

    def test_apply_augmentation_multiple(self):
        """Test d'application de plusieurs augmentations."""
        image = np.array([[1, 2], [3, 4]])

        augmented = apply_augmentation(image, ["flip_h", "flip_v"])

        # Vérifier que les augmentations ont été appliquées
        assert augmented.shape == image.shape


class TestDataSplitting:
    """Tests pour le split stratifié."""

    def test_stratified_split_basic(self):
        """Test de split stratifié basique."""
        import pandas as pd

        # Créer un DataFrame de test avec des classes
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(100)],
            "classes": ["class_a"] * 50 + ["class_b"] * 50,
        }
        df = pd.DataFrame(data)

        train_df, val_df, test_df = stratified_split(
            df, target_column="classes", random_state=42
        )

        # Vérifier les proportions approximatives
        assert len(train_df) == pytest.approx(70, abs=5)
        assert len(val_df) == pytest.approx(15, abs=5)
        assert len(test_df) == pytest.approx(15, abs=5)

        # Vérifier que toutes les images sont présentes
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df)

    def test_stratified_split_ratios(self):
        """Test de split stratifié avec ratios personnalisés."""
        import pandas as pd

        data = {
            "image_path": [f"img_{i}.jpg" for i in range(100)],
            "classes": ["class_a"] * 100,
        }
        df = pd.DataFrame(data)

        train_df, val_df, test_df = stratified_split(
            df,
            target_column="classes",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=42,
        )

        assert len(train_df) == pytest.approx(80, abs=5)
        assert len(val_df) == pytest.approx(10, abs=5)
        assert len(test_df) == pytest.approx(10, abs=5)


class TestGlobalStats:
    """Tests pour le calcul des statistiques globales."""

    def test_compute_global_stats_min_max(self):
        """Test de calcul des stats globales avec min_max."""
        images = [
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            np.array([[5.0, 15.0], [25.0, 35.0]]),
            np.array([[15.0, 25.0], [35.0, 45.0]]),
        ]

        stats = compute_global_stats(images, method="min_max")

        assert stats["method"] == "min_max"
        assert stats["scope"] == "per_sensor"
        assert stats["min"] == 5.0
        assert stats["max"] == 45.0

    def test_compute_global_stats_z_score(self):
        """Test de calcul des stats globales avec z_score."""
        images = [
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            np.array([[5.0, 15.0], [25.0, 35.0]]),
        ]

        stats = compute_global_stats(images, method="z_score")

        assert stats["method"] == "z_score"
        assert stats["scope"] == "per_sensor"
        assert "mean" in stats
        assert "std" in stats


class TestImageLoading:
    """Tests pour le chargement d'images."""

    def test_load_thermal_image_from_file(self):
        """Test de chargement d'image depuis un fichier."""
        # Créer une image temporaire
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_image.png"
            # Créer une image de test
            test_image = Image.new("L", (64, 64), color=128)
            test_image.save(img_path)

            # Charger l'image
            loaded = load_thermal_image(img_path)

            assert loaded.shape == (64, 64)
            assert loaded.dtype == np.float32

    def test_load_thermal_image_not_found(self):
        """Test de chargement d'image inexistante."""
        fake_path = Path("/nonexistent/path/image.png")

        with pytest.raises(FileNotFoundError):
            load_thermal_image(fake_path)

    def test_load_thermal_image_rgb_conversion(self):
        """Test de conversion RGB en niveaux de gris."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_rgb.png"
            # Créer une image RGB
            test_image = Image.new("RGB", (32, 32), color=(100, 150, 200))
            test_image.save(img_path)

            # Charger l'image
            loaded = load_thermal_image(img_path)

            # Devrait être convertie en 2D (niveaux de gris)
            assert len(loaded.shape) == 2
            assert loaded.shape == (32, 32)

