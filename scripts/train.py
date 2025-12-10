"""Compatibility wrapper: call package-local entrypoint for train."""

from thermal_sensors.scripts.train import main

if __name__ == "__main__":
    import sys

    sys.exit(main())

