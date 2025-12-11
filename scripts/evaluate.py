"""Compatibility wrapper: call package-local entrypoint for evaluate."""

from thermal_sensors.scripts.evaluate import main

if __name__ == "__main__":
    import sys

    sys.exit(main())

