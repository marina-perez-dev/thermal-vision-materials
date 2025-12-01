from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent


def _read_text_robust(path: Path) -> str:
    b = path.read_bytes()
    # BOM detection
    if b.startswith(b'\xff\xfe') or b.startswith(b'\xfe\xff'):
        return b.decode('utf-16')
    if b.startswith(b'\xef\xbb\xbf'):
        return b.decode('utf-8-sig')
    # try common encodings
    for enc in ('utf-8', 'latin-1'):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    # fallback
    return b.decode('latin-1', errors='replace')

def read_requirements():
    req = root / "requirements.txt"
    if req.exists():
        text = _read_text_robust(req)
        return [
            r.strip()
            for r in text.splitlines()
            if r.strip() and not r.strip().startswith("#")
        ]
    return []

long_description = (
    _read_text_robust(root / "README.md") if (root / "README.md").exists() else ""
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