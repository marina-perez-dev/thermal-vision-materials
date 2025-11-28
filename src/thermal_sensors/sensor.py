"""Minimal sensor API for tests and package structure.

This module provides a small adapter used by tests. The real
implementation should interact with hardware drivers and return
temperature values.
"""

def _read_from_driver():
    """Read temperature from the hardware driver.

    In production this will access the sensor; tests patch this
    function to simulate driver values.
    """
    raise NotImplementedError("Hardware driver not implemented")


def read_temperature():
    """Return the latest temperature reading from the sensor."""
    return _read_from_driver()
