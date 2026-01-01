# Getting Started Tutorial

This tutorial will walk you through the basic concepts and usage of THEMAP for task hardness estimation and distance computation.

## Prerequisites

- THEMAP installed with basic dependencies
- Python 3.10+
- Basic knowledge of molecular datasets

## Tutorial Overview

In this tutorial, you will learn how to:

1. Set up your data directory
2. Compute distances between datasets
3. Analyze the results
4. Estimate task hardness

## Step 1: Setting Up Your Data

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

## Step 2: Computing Distances (One-liner)

The simplest way to compute distances:

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",
    output_dir="output",
    molecule_featurizer="ecfp",
    molecule_method="euclidean",
)

print("Results saved to output/molecule_distances.csv")
```

## Step 3: Using a Config File

For reproducible experiments:

```python
from themap import run_pipeline

# Create config file
config_content = """
data:
  directory: "datasets"

molecule:
  enabled: true
  featurizer: "ecfp"
  method: "euclidean"

output:
  directory: "output"
  format: "csv"
"""

# Save and run
with open("config.yaml", "w") as f:
    f.write(config_content)

results = run_pipeline("config.yaml")
```

## Step 4: Analyzing Results

```python
import pandas as pd

# Load computed distances
distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

print(f"Distance matrix shape: {distances.shape}")
print(f"Sources (rows): {list(distances.index)}")
print(f"Targets (columns): {list(distances.columns)}")

# Find closest source for each target
for target in distances.columns:
    closest = distances[target].idxmin()
    dist = distances[target].min()
    print(f"{target} <- {closest} (distance: {dist:.4f})")
```

## Step 5: Estimating Task Hardness

Task hardness is estimated from the average distance to the k-nearest source tasks:

```python
# Compute task hardness for each target
k = 3
for target in distances.columns:
    k_nearest = distances[target].nsmallest(k).mean()
    print(f"Task hardness for {target}: {k_nearest:.4f}")
```

Higher hardness values indicate tasks that are more different from available training data.

## Understanding Results

Distance values have the following interpretations:

- **Lower values**: More similar datasets (easier transfer learning)
- **Higher values**: More different datasets (harder transfer learning)
- **Scale**: Depends on the method and featurizer used

## Next Steps

Now that you understand the basics:

1. Try different distance methods (`cosine`, `otdd`)
2. Try different featurizers (`maccs`, `desc2D`)
3. Learn about [performance optimization](performance-optimization.md)
4. Check out the [examples](../examples/index.md)

## Common Issues

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -e ".[all]"
```

### Memory Issues
Use euclidean distance for large datasets instead of OTDD.

### File Not Found
Ensure your data files are in the correct directory structure.
