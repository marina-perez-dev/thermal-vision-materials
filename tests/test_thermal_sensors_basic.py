from unittest.mock import patch

# Adapter 'thermal_sensors.sensor.read_temperature' au chemin réel du projet.
def test_read_temperature_uses_driver():
    with patch('thermal_sensors.sensor._read_from_driver') as mock_driver:
        mock_driver.return_value = 36.5
        from thermal_sensors.sensor import read_temperature
        assert read_temperature() == 36.5
        mock_driver.assert_called_once()