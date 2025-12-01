"""Compatibility wrapper: call package-local entrypoint.

Keeps the top-level `scripts/` CLI compatible during development while the
actual implementation lives under `thermal_sensors.scripts` (installed with
the package and available via entry points).
"""

from thermal_sensors.scripts.ingest import main

if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data_src"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/raw/data_raw.csv"
    main(src, out)