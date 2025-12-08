"""Compatibility wrapper: call package-local entrypoint for preprocess."""

from thermal_sensors.scripts.preprocess import main

if __name__ == "__main__":
    import sys

    # Utiliser les arguments de ligne de commande ou les valeurs par défaut pour Roboflow
    input_metadata = sys.argv[1] if len(sys.argv) > 1 else "data/raw/roboflow_metadata.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    
    # Appeler la fonction main avec les paramètres appropriés
    exit_code = main(
        input_metadata=input_metadata,
        output_dir=output_dir,
        raw_data_dir="data/raw/roboflow",
    )
    sys.exit(exit_code)