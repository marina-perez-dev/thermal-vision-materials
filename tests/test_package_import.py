def test_package_and_modules_importable():
    import importlib

    mod = importlib.import_module("thermal_sensors")
    assert mod is not None

    sensor = importlib.import_module("thermal_sensors.sensor")
    assert hasattr(sensor, "read_temperature")

    cli_ingest = importlib.import_module("thermal_sensors.scripts.ingest_roboflow_dataset")
    assert hasattr(cli_ingest, "main")

    try:
        cli_pre = importlib.import_module("thermal_sensors.scripts.preprocess")
        assert hasattr(cli_pre, "main")
    except ModuleNotFoundError:

        pass
