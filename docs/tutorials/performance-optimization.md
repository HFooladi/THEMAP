# Performance Optimization

This tutorial covers best practices for optimizing THEMAP performance when working with large datasets.

## Overview

THEMAP offers several strategies to optimize performance:

1. **Method selection**: Choose appropriate distance metrics
2. **Featurizer selection**: Choose fast vs. accurate featurizers
3. **Parallel processing**: Use multiple cores
4. **Caching**: Save features to disk for reuse

## Distance Method Performance

### Speed Comparison

| Method | Speed | Memory | Accuracy | Best For |
|--------|-------|--------|----------|----------|
| **Euclidean** | Fast | Low | Good | Initial exploration, large datasets |
| **Cosine** | Fast | Low | Good | High-dimensional features |
| **OTDD** | Slow | High | Best | Detailed analysis, small datasets |

### Choosing the Right Method

```python
from themap import quick_distance

# For large datasets or initial exploration
results = quick_distance(
    data_dir="datasets",
    molecule_method="euclidean",  # Fast
)

# For final analysis on small datasets
results = quick_distance(
    data_dir="datasets",
    molecule_method="otdd",  # Most accurate but slow
)
```

## Featurizer Performance

### Speed Comparison

| Featurizer | Speed | Quality |
|------------|-------|---------|
| `ecfp` | Fast | Good |
| `maccs` | Fast | Good |
| `desc2D` | Medium | Good |
| `desc3D` | Slow | Better |

```python
# Fast featurizer for large datasets
results = quick_distance(
    data_dir="datasets",
    molecule_featurizer="ecfp",  # Fast fingerprints
)
```

## Parallel Processing

Use the `n_jobs` parameter:

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",
    n_jobs=8,  # Use 8 parallel workers
)
```

## Caching Features

Save computed features to disk for reuse:

```yaml
# config.yaml
output:
  save_features: true  # Cache features for reuse
```

```python
from themap import run_pipeline

# First run: computes and caches features
results = run_pipeline("config.yaml")

# Subsequent runs: loads cached features (faster)
results = run_pipeline("config.yaml")
```

## Memory Management

For large datasets:

```python
from themap import quick_distance

# Use euclidean instead of OTDD (much less memory)
results = quick_distance(
    data_dir="datasets",
    molecule_featurizer="ecfp",
    molecule_method="euclidean",
)
```

## Best Practices

### Quick Optimization Checklist

1. **Start with fast methods**: Use euclidean for exploration
2. **Use fast featurizers**: ecfp or maccs for initial runs
3. **Enable caching**: Set `save_features: true`
4. **Use parallel processing**: Set `n_jobs` to available cores
5. **Reserve OTDD for final analysis**: Only use on small datasets

### Recommended Configuration

```yaml
# config.yaml - optimized for performance
data:
  directory: "datasets"

molecule:
  enabled: true
  featurizer: "ecfp"      # Fast fingerprints
  method: "euclidean"     # Fast distance

output:
  directory: "output"
  format: "csv"
  save_features: true     # Cache for reuse

compute:
  n_jobs: 8               # Parallel processing
  device: "auto"          # Use GPU if available
```

## Benchmarking

Compare methods on your data:

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

## Next Steps

- Check out the [examples](../examples/index.md)
- Read the [API documentation](../api/distance.md)
