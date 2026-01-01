# THEMAP

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs--jcim--3c01774-blue)](https://doi.org/10.1021/acs.jcim.4c00160)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/themap.svg)](https://badge.fury.io/py/themap)

<p align="center">
  <img src="assets/banner.png" alt="THEMAP Banner" style="max-width:100%;">
</p>

**T**ask **H**ardness **E**stimation for **M**olecular **A**ctivity **P**rediction

A Python library for calculating distances between chemical datasets to enable intelligent dataset selection for molecular activity prediction tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Reproducing FS-Mol Experiments](#reproducing-fs-mol-experiments)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

THEMAP is a Python library designed to calculate distances between chemical datasets for molecular activity prediction tasks. The primary goal is to enable intelligent dataset selection for:

- **Transfer Learning**: Identify the most relevant source datasets for your target prediction task
- **Domain Adaptation**: Measure dataset similarity to guide model adaptation strategies
- **Task Hardness Assessment**: Quantify how difficult a prediction task will be based on dataset characteristics
- **Dataset Curation**: Select optimal training datasets from large chemical databases like ChEMBL


## Installation

### Quick Start (Recommended)

The easiest way to install THEMAP with all features:

```bash
git clone https://github.com/HFooladi/THEMAP.git
cd THEMAP
source env.sh
```

This automatically:
- Installs `uv` (fast Python package manager) if needed
- Creates a virtual environment in `.venv`
- Installs all dependencies
- Activates the environment

After installation, try an example:
```bash
python examples/basic/molecule_datasets_demo.py
```

To reactivate the environment later:
```bash
source .venv/bin/activate
```

### Manual Installation

For more control, install with pip:

```bash
pip install themap                # Basic installation from PyPI
pip install -e ".[all]"           # Full installation (editable)
pip install -e ".[protein]"       # Protein analysis only
pip install -e ".[otdd]"          # Optimal transport only
pip install -e ".[dev,test]"      # Development + testing
```

### Conda Alternative

For GPU support with specific CUDA versions:

```bash
conda env create -f environment.yml
conda activate themap
pip install -e . --no-deps
```

### Prerequisites

- Python 3.10 or higher
- For GPU features: CUDA-compatible GPU and drivers

## Quick Start

### Compute Dataset Distances

The simplest way to compute distances between molecular datasets:

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",          # Directory with train/ and test/ folders
    output_dir="output",          # Where to save results
    molecule_featurizer="ecfp",   # Fingerprint type (ecfp, maccs, etc.)
    molecule_method="euclidean",  # Distance metric
)

# Results saved to output/molecule_distances.csv
```

### Using a Config File

For reproducible experiments, use a YAML configuration:

```python
from themap import run_pipeline

results = run_pipeline("config.yaml")
```

Example `config.yaml`:
```yaml
data:
  directory: "datasets"

molecule:
  enabled: true
  featurizer: "ecfp"
  method: "euclidean"

output:
  directory: "output"
  format: "csv"
```

### Data Format

Organize your data in this structure:

```
datasets/
├── train/                        # Source datasets
│   ├── CHEMBL123456.jsonl.gz
│   └── ...
└── test/                         # Target datasets
    ├── CHEMBL111111.jsonl.gz
    └── ...
```

Each `.jsonl.gz` file contains molecules in JSON lines format:
```json
{"SMILES": "CCO", "Property": 1}
{"SMILES": "CCCO", "Property": 0}
```


## Usage Examples

### Analyzing Distance Results

```python
import pandas as pd

# Load computed distances
distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

# Find closest source for each target (transfer learning selection)
for target in distances.columns:
    closest = distances[target].idxmin()
    dist = distances[target].min()
    print(f"{target} <- {closest} (distance: {dist:.4f})")

# Estimate task hardness (average distance to k-nearest sources)
k = 3
for target in distances.columns:
    hardness = distances[target].nsmallest(k).mean()
    print(f"Task hardness for {target}: {hardness:.4f}")
```

## Reproducing FS-Mol Experiments

Pre-computed molecular embeddings and distance matrices for the FS-Mol dataset are available on [Zenodo](https://zenodo.org/records/10605093).

### Setup
1. Download data from [Zenodo](https://zenodo.org/records/10605093)
2. Extract to `datasets/fsmol_hardness/`
3. See `examples/` directory for usage examples

## Documentation

Full documentation is available at [themap.readthedocs.io](https://hfooladi.github.io/THEMAP/) or can be built locally:

```bash
mkdocs serve  # Serve locally at http://127.0.0.1:8000
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/HFooladi/THEMAP.git
cd THEMAP
pip install -e ".[dev,test]"
```

### Running Tests

```bash
pytest
pytest --cov=themap  # with coverage
```

### Code Quality

```bash
ruff check && ruff format  # linting and formatting
mypy themap/               # type checking
```

## Citation

If you use THEMAP in your research, please cite our paper:

```bibtex
@article{fooladi2024quantifying,
  title={Quantifying the hardness of bioactivity prediction tasks for transfer learning},
  author={Fooladi, Hosein and Hirte, Steffen and Kirchmair, Johannes},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={10},
  pages={4031-4046},
  year={2024},
  publisher={ACS Publications}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [Documentation](https://hfooladi.github.io/THEMAP/)
- [Issue Tracker](https://github.com/HFooladi/THEMAP/issues)
- [Discussions](https://github.com/HFooladi/THEMAP/discussions)
