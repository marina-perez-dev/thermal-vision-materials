"""Script de test pour vérifier que le modèle peut être créé et utilisé.

Ce script teste la création du modèle sans nécessiter de données d'entraînement.
"""

import sys
from pathlib import Path
import numpy as np

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermal_sensors.models.baseline_model import BaselineModel


def test_model_creation():
    """Test la création et l'utilisation du modèle."""
    print("=" * 60)
    print("Test de création du modèle baseline")
    print("=" * 60)

    # 1. Créer le modèle
    print("\n1. Création du modèle...")
    model = BaselineModel(
        input_shape=(64, 64, 1),
        num_classes=7,
        learning_rate=0.001,
    )
    print("✓ Modèle créé avec succès")

    # 2. Afficher le résumé
    print("\n2. Résumé de l'architecture:")
    model.summary()

    # 3. Test d'un forward pass
    print("\n3. Test d'un forward pass...")
    test_image = np.random.rand(1, 64, 64, 1).astype(np.float32)
    predictions = model.model.predict(test_image, verbose=0)
    print(f"✓ Forward pass réussi")
    print(f"  Forme de sortie: {predictions.shape}")
    print(f"  Probabilités: {predictions[0]}")
    print(f"  Classe prédite: {np.argmax(predictions[0])}")

    # 4. Test de sauvegarde/chargement
    print("\n4. Test de sauvegarde/chargement...")
    test_model_dir = Path("models/test")
    test_model_dir.mkdir(parents=True, exist_ok=True)

    model.save(test_model_dir, model_name="test_model")
    print("✓ Modèle sauvegardé")

    loaded_model = BaselineModel.load(test_model_dir, model_name="test_model")
    print("✓ Modèle chargé")

    # Vérifier que les prédictions sont identiques
    pred1 = model.model.predict(test_image, verbose=0)
    pred2 = loaded_model.model.predict(test_image, verbose=0)
    assert np.allclose(pred1, pred2), "Les prédictions doivent être identiques"
    print("✓ Prédictions identiques après chargement")

    print("\n" + "=" * 60)
    print("Tous les tests sont passés avec succès!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = test_model_creation()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

