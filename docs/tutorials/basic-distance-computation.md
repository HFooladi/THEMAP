# Basic Distance Computation

This tutorial covers the fundamentals of computing distances between molecular datasets using THEMAP.

## Overview

Distance computation is at the core of THEMAP's functionality. This tutorial will show you how to:

1. Choose appropriate distance metrics
2. Configure distance calculations
3. Analyze results
4. Optimize performance

## Quick Start

The simplest way to compute distances:

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",
    output_dir="output",
    molecule_featurizer="ecfp",
    molecule_method="euclidean",
)
```

## Distance Methods Comparison

### Euclidean Distance
- **Fast** and memory-efficient
- Good for initial exploration
- Works with any embedding

```python
results = quick_distance(
    data_dir="datasets",
    molecule_method="euclidean"
)
```

### Cosine Distance
- Focuses on feature orientation
- Good for high-dimensional embeddings
- Normalized by vector magnitude

```python
results = quick_distance(
    data_dir="datasets",
    molecule_method="cosine"
)
```

### OTDD (Optimal Transport Dataset Distance)
- Most comprehensive but computationally expensive
- Considers both features and labels
- Best for detailed analysis

```python
results = quick_distance(
    data_dir="datasets",
    molecule_method="otdd"
)
```

## Featurizer Options

Different molecular representations:

| Featurizer | Description | Speed |
|------------|-------------|-------|
| `ecfp` | Extended Connectivity Fingerprints | Fast |
| `maccs` | MACCS structural keys | Fast |
| `desc2D` | 2D molecular descriptors | Medium |
| `desc3D` | 3D molecular descriptors | Slow |

```python
# Using different featurizers
results = quick_distance(
    data_dir="datasets",
    molecule_featurizer="maccs",  # or "ecfp", "desc2D"
    molecule_method="euclidean",
)
```

## Using Config Files

For reproducible experiments:

```yaml
# config.yaml
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
```

```python
from themap import run_pipeline

results = run_pipeline("config.yaml")
```

## Analyzing Results

### Loading Distance Matrix

```python
import pandas as pd

distances = pd.read_csv("output/molecule_distances.csv", index_col=0)
print(f"Matrix shape: {distances.shape}")
```

### Finding Most Similar Datasets

```python
# Find closest source for each target
for target in distances.columns:
    closest = distances[target].idxmin()
    dist = distances[target].min()
    print(f"{target} <- {closest} (distance: {dist:.4f})")
```

### Task Hardness Estimation

```python
# Estimate hardness as average distance to k-nearest sources
k = 3
for target in distances.columns:
    hardness = distances[target].nsmallest(k).mean()
    print(f"Hardness for {target}: {hardness:.4f}")
```

## Distance Interpretation

- **0.0**: Identical datasets
- **Low values (< 1.0)**: Very similar datasets
- **Medium values (1.0-10.0)**: Moderately similar
- **High values (> 10.0)**: Very different datasets

Note: Actual ranges depend on the distance method and data characteristics.

## Performance Optimization

### For Large Datasets

```python
# Use fast featurizer and method
results = quick_distance(
    data_dir="datasets",
    molecule_featurizer="ecfp",   # Fast fingerprints
    molecule_method="euclidean",  # Faster than OTDD
    n_jobs=8,                     # Parallel processing
)
```

### Caching Features

```yaml
# config.yaml
output:
  save_features: true  # Cache features for reuse
```

## Troubleshooting

### Common Errors

1. **Memory Error**: Use euclidean distance or reduce dataset size
2. **Import Error**: Install required dependencies with `pip install -e ".[all]"`
3. **File Not Found**: Check file paths and data structure

### Performance Tips

1. Start with euclidean distance for exploration
2. Use OTDD only for final analysis
3. Enable feature caching for repeated computations
4. Use parallel processing (`n_jobs` parameter)

## Next Steps

- Learn about [performance optimization](performance-optimization.md)
- Check out the [examples](../examples/index.md)
