from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent


def read_requirements():
    req = root / "requirements.txt"
    if req.exists():
        return [
            r.strip()
            for r in req.read_text(encoding="utf-8").splitlines()
            if r.strip() and not r.strip().startswith("#")
        ]
    return []


long_description = (
    (root / "README.md").read_text(encoding="utf-8")
    if (root / "README.md").exists()
    else ""
)

setup(
    name="thermal-vision-materials",
    version="0.1.0",
    description="Thermal vision materials, dataset simulation and robot-simulation",
    long_description=long_description,
    packages=find_packages(exclude=["tests*", "docs*", "outputs*", "data*"]),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.8",
    license="MIT",
    author="MarinaP",
)