# Distance Computation Guide

Distance computation is central to THEMAP, enabling dataset similarity assessment, transfer learning guidance, and task hardness estimation.

## Molecular Dataset Distances

THEMAP supports three methods for comparing molecular datasets via `DatasetDistance`:

### OTDD (Optimal Transport Dataset Distance)

The most comprehensive method. OTDD considers both feature distributions and label relationships using optimal transport theory.

```python
from themap import DatasetDistance

dd = DatasetDistance(
    train_datasets=train_datasets,
    test_datasets=test_datasets,
    featurizer="ecfp",
    method="otdd",
)
distance_matrix = dd.compute()
```

**When to use:**

- High accuracy requirements
- Moderate dataset sizes (< 10,000 molecules per task)
- Both features and labels matter

**Limitations:** Computationally expensive and memory-intensive for large datasets.

### Euclidean Distance

Fast and interpretable distance based on feature vector similarity.

```python
dd = DatasetDistance(
    train_datasets=train_datasets,
    test_datasets=test_datasets,
    featurizer="ecfp",
    method="euclidean",
)
distance_matrix = dd.compute()
```

**When to use:**

- Large datasets (> 10,000 molecules)
- Fast computation needed
- Feature magnitude is meaningful

### Cosine Distance

Measures angular similarity, ignoring vector magnitude. Good for high-dimensional sparse features.

```python
dd = DatasetDistance(
    train_datasets=train_datasets,
    test_datasets=test_datasets,
    featurizer="ecfp",
    method="cosine",
)
distance_matrix = dd.compute()
```

**When to use:**

- High-dimensional features
- Sparse feature vectors (fingerprints)
- Feature orientation matters more than magnitude

## Method Comparison

| Method | Speed | Memory | Accuracy | Best For |
|--------|-------|--------|----------|----------|
| **OTDD** | Slow | High | Highest | Small-medium datasets |
| **Euclidean** | Fast | Low | Good | Large datasets, magnitude-sensitive |
| **Cosine** | Fast | Low | Good | Sparse, high-dimensional features |

## Metadata Distances

`MetadataDistance` computes distances between single-vector task metadata (e.g., assay descriptions, protein embeddings).

Available methods: `euclidean`, `cosine`, `manhattan`, `jaccard`.

```python
from themap import MetadataDistance

md = MetadataDistance(
    train_metadata=train_metadata,
    test_metadata=test_metadata,
    method="cosine",
)
metadata_distances = md.compute()
```

## Combined Distances

Use `TaskDistanceCalculator` to combine molecule, protein, and metadata distances into a single score.

```python
from themap.distance import TaskDistanceCalculator

calc = TaskDistanceCalculator(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean",
    metadata_method="jaccard",
)

all_distances = calc.compute_all_distances(
    combination_strategy="weighted_average",
    molecule_weight=0.5,
    protein_weight=0.3,
    metadata_weight=0.2,
)
```

**Combination strategies:**

| Strategy | Description |
|----------|-------------|
| `"average"` | Simple arithmetic mean of all modalities |
| `"weighted_average"` | Weighted combination with user-specified weights |
| `"separate"` | Return each modality's distances separately |

## Featurizer Options

Different molecular representations affect distance quality and speed:

| Category | Featurizers | Speed |
|----------|-------------|-------|
| **Fingerprints** | ecfp, fcfp, maccs, avalon, topological, atompair, pattern, layered, secfp, erg, estate, rdkit | Fast |
| **Count fingerprints** | ecfp-count, fcfp-count, topological-count, atompair-count, rdkit-count, avalon-count | Fast |
| **Descriptors** | desc2D, mordred, cats2D, pharm2D, scaffoldkeys | Medium |
| **Neural** | ChemBERTa-77M-MLM, ChemBERTa-77M-MTR, MolT5, Roberta-Zinc480M-102M, GIN variants | Slow |

Run `themap list-featurizers` to see all available featurizers.

## Interpreting Results

- **0.0**: Identical datasets
- **Low values**: Very similar datasets (good transfer learning candidates)
- **High values**: Very different datasets (poor transfer learning candidates)

!!! note
    Absolute distance values depend on the method and featurizer. Compare distances within the same configuration, not across different methods.

## Next Steps

- [Basic Distance Computation Tutorial](../tutorials/basic-distance-computation.md) - hands-on walkthrough
- [Performance Optimization](../tutorials/performance-optimization.md) - caching and parallelism
- [API Reference](../api/distance.md) - full API documentation
