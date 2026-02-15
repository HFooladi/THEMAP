# Getting Started

This guide covers installation and three ways to compute distances between molecular datasets.

## Installation

### Prerequisites

- Python 3.10 or higher

### Quick Install

```bash
git clone https://github.com/HFooladi/THEMAP.git
cd THEMAP
source install.sh   # creates .venv with uv
```

To reactivate later:

```bash
source .venv/bin/activate
```

### Optional Dependencies

THEMAP has optional dependency groups for different functionality:

```bash
pip install -e ".[ml]"       # molecular analysis
pip install -e ".[protein]"  # protein analysis (ESM2)
pip install -e ".[otdd]"     # OTDD distance computation
pip install -e ".[all]"      # everything
pip install -e ".[dev,test]" # development
```

### Verify Installation

```python
import themap
print(f"THEMAP version: {themap.__version__}")
```

## Quick Start

### Option 1: Command Line

The fastest way to compute distances:

```bash
themap quick datasets/ -f ecfp -m euclidean -o output/
```

This computes Euclidean distances between all train and test datasets using ECFP fingerprints and saves results to `output/molecule_distances.csv`.

### Option 2: Python One-Liner

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",
    output_dir="output",
    molecule_featurizer="ecfp",
    molecule_method="euclidean",
)
```

### Option 3: Config File

For reproducible experiments, use a YAML configuration:

```bash
themap init          # generates a config.yaml template
themap run config.yaml
```

Or from Python:

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

## Understanding Results

The output is a CSV distance matrix with source tasks as rows and target tasks as columns:

```python
import pandas as pd

distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

# Find closest source for each target
for target in distances.columns:
    closest = distances[target].idxmin()
    dist = distances[target].min()
    print(f"{target} <- {closest} (distance: {dist:.4f})")
```

Lower distance values indicate more similar datasets (easier transfer learning).

## Data Format

Organize your data in this structure:

```
datasets/
├── train/
│   ├── CHEMBL123456.jsonl.gz
│   └── ...
└── test/
    ├── CHEMBL111111.jsonl.gz
    └── ...
```

Each `.jsonl.gz` file contains molecules in JSON lines format:

```json
{"SMILES": "CCO", "Property": 1}
{"SMILES": "CCCO", "Property": 0}
```

!!! tip
    Use `themap convert` to convert CSV files to the required JSONL.GZ format:
    ```bash
    themap convert data.csv CHEMBL123456
    ```

## Next Steps

- [Distance Computation Guide](distance-computation.md) - detailed explanation of all distance methods
- [Tutorials](../tutorials/index.md) - step-by-step walkthroughs
- [CLI Reference](cli.md) - all command-line options
