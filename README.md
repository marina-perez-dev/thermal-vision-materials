# Thermal Vision Materials

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for working with thermal sensor data, dataset ingestion, and preprocessing for material classification and waste detection applications.

## Overview

This project provides tools and utilities for processing thermal imaging data, with support for:
- Thermal sensor data ingestion and preprocessing
- Roboflow dataset integration (YOLO format)
- Data versioning with DVC (Data Version Control)
- CLI tools for data pipeline management
- Integration with ROS2/Gazebo for robotic simulations

## Features

- **Thermal Sensor API**: Simple interface for reading temperature data from hardware sensors
- **Dataset Ingestion**: Automated ingestion of Roboflow datasets with metadata extraction
- **Data Preprocessing**: Pipeline for preparing thermal images for machine learning
- **DVC Integration**: Version control for datasets and models
- **CLI Tools**: Command-line utilities for common data operations
- **Modular Architecture**: Clean `src/` layout with comprehensive test coverage

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Development Installation

For development, install the package in editable mode:

```powershell
# Create and activate virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in editable mode
python -m pip install -e .

# Install development dependencies
pip install -r requirements.txt
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pip install -r requirements.txt
```

## Usage

#### Data Preprocessing

Preprocess thermal images:

```powershell
thermal-preprocess <input_path> <output_path>
```

## Project Structure

```
thermal-vision-materials/
├── src/
│   └── thermal_sensors/          # Main package
│       ├── sensor.py              # Sensor API
│       └── scripts/               # CLI scripts
├── data/                         # Data directory (DVC managed)
│   └── raw/                      # Raw datasets
├── tests/                        # Test suite
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── examples/                     # Example usage
```

## Development

### Running Tests

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=src/thermal_sensors
```

### Code Quality

The project uses `black` for code formatting and `ruff` for linting:

```powershell
# Format code
black .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```powershell
.\scripts\install-hooks.ps1
```

## Documentation

- [Setup Guide](docs/SETUP.md) - Development environment setup
- [Testing Guide](docs/TESTS.md) - Running and writing tests
- [DVC Guide](docs/DVC.md) - Data version control workflow
- [Roboflow Dataset](docs/ROBOFLOW_DATASET.md) - Dataset ingestion guide
- [Contributing](docs/CONTRIBUTING.md) - Contribution guidelines

## Data Version Control

This project uses DVC (Data Version Control) for managing datasets and model artifacts. See [DVC.md](docs/DVC.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

MarinaP

## Acknowledgments

- Dataset: [Thermal Waste Detection](https://universe.roboflow.com/uva-wellassa-university-y1g0q/thermal-waste-detection-02/dataset/3) by UVA Wellassa University
- Built with Python, DVC, and modern MLOps practices
