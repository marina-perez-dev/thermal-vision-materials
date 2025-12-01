"""Compatibility wrapper: call package-local entrypoint for preprocess."""

from thermal_sensors.scripts.preprocess import main

if __name__ == "__main__":
    import sys

    inp = sys.argv[1] if len(sys.argv) > 1 else "data/raw/data_raw.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/data_processed.csv"
    main(inp, out)