# thermal-vision-materials

Small package for working with simulated thermal sensor data and helper CLIs.

Installation
 - Editable install for development:

```powershell
python -m pip install -e .
```

Basic usage

 - Import API:

```python
from thermal_sensors.sensor import read_temperature
```

 - CLI (after install):

```powershell
thermal-ingest <src_dir> <out_path>
thermal-preprocess <in_path> <out_path>
```

Development notes

 - Project uses `src/` layout. Tests use `tests/conftest.py` to add `src/` to
   `sys.path` during test runs when not installed.
 - CI installs the package via `pip install -e .` before running tests.
