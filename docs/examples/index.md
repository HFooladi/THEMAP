# Examples

This section provides practical, runnable examples for common THEMAP use cases.

## Quick Start

### One-liner Distance Computation

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",          # Directory with train/ and test/ folders
    output_dir="output",          # Where to save results
    molecule_featurizer="ecfp",   # Fingerprint type
    molecule_method="euclidean",  # Distance metric
)

# Results saved to output/molecule_distances.csv
```

### Using a Config File

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
  save_features: true

compute:
  n_jobs: 8
```

### Full Programmatic Control

```python
from themap import Pipeline, PipelineConfig
from themap.config import DataConfig, MoleculeDistanceConfig, OutputConfig

config = PipelineConfig(
    data=DataConfig(directory=Path("datasets")),
    molecule=MoleculeDistanceConfig(
        enabled=True,
        featurizer="ecfp",
        method="euclidean"
    ),
    output=OutputConfig(
        directory=Path("output"),
        format="csv"
    ),
)

pipeline = Pipeline(config)
results = pipeline.run()
```

## Analyzing Results

### Load and Explore Distance Matrix

```python
import pandas as pd

# Load computed distances
distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

print(f"Shape: {distances.shape}")
print(f"Sources: {list(distances.index)}")
print(f"Targets: {list(distances.columns)}")
```

### Find Closest Source for Each Target

```python
import pandas as pd

distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

for target in distances.columns:
    closest = distances[target].idxmin()
    dist = distances[target].min()
    print(f"{target} <- {closest} (distance: {dist:.4f})")
```

### Estimate Task Hardness

```python
import pandas as pd

distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

# Task hardness = average distance to k-nearest sources
k = 3
for target in distances.columns:
    hardness = distances[target].nsmallest(k).mean()
    print(f"Hardness for {target}: {hardness:.4f}")
```

## Comparing Distance Methods

```python
import time
from themap import quick_distance

methods = ["euclidean", "cosine"]

for method in methods:
    start = time.time()
    results = quick_distance(
        data_dir="datasets",
        molecule_method=method,
    )
    elapsed = time.time() - start
    print(f"{method}: {elapsed:.2f}s")
```

## Visualization

### Distance Matrix Heatmap

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load distances
distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    distances,
    annot=True,
    fmt='.3f',
    cmap='viridis',
    cbar_kws={'label': 'Distance'},
)

plt.title("Dataset Distances")
plt.xlabel("Target Tasks")
plt.ylabel("Source Tasks")
plt.tight_layout()
plt.savefig("distance_heatmap.png", dpi=300)
plt.show()
```

## Data Format

### Directory Structure

```
datasets/
├── train/                        # Source datasets
│   ├── CHEMBL123456.jsonl.gz
│   └── ...
└── test/                         # Target datasets
    ├── CHEMBL111111.jsonl.gz
    └── ...
```

### JSONL.GZ File Format

Each file contains molecules in JSON lines format:
```json
{"SMILES": "CCO", "Property": 1}
{"SMILES": "CCCO", "Property": 0}
```

## Command Line Usage

```bash
# Run quickstart example
python examples/quickstart.py --data datasets --featurizer ecfp --method euclidean

# With config file
python examples/quickstart.py --config config.yaml
```

## Available Options

### Featurizers

| Featurizer | Description |
|------------|-------------|
| `ecfp` | Extended Connectivity Fingerprints (fast) |
| `maccs` | MACCS structural keys (fast) |
| `desc2D` | 2D molecular descriptors |
| `desc3D` | 3D molecular descriptors |

### Distance Methods

| Method | Description |
|--------|-------------|
| `euclidean` | Fast Euclidean distance |
| `cosine` | Cosine distance |
| `otdd` | Optimal Transport Dataset Distance (slow but accurate) |

For more detailed explanations, see our [tutorials](../tutorials/index.md).
