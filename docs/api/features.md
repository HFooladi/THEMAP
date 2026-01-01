# Features Module

The features module provides unified feature extraction for molecules and proteins. It handles featurization, caching, and batch processing.

## Overview

The features system consists of three main components:

- **`MoleculeFeaturizer`** - Extract molecular representations (fingerprints, descriptors, embeddings)
- **`ProteinFeaturizer`** - Extract protein sequence embeddings (ESM2, ESM3)
- **`FeatureCache`** - Efficient caching for expensive feature computations

## Molecule Featurizer

### MoleculeFeaturizer

::: themap.features.molecule.MoleculeFeaturizer
    options:
      show_root_heading: true
      heading_level: 3

### Available Featurizers

#### Fingerprints (Fast)

| Featurizer | Description | Dimensions |
|------------|-------------|------------|
| `ecfp` | Extended Connectivity Fingerprints | 2048 |
| `maccs` | MACCS Structural Keys | 167 |
| `topological` | Topological Fingerprints | 2048 |
| `avalon` | Avalon Fingerprints | 512 |

#### Descriptors (Medium Speed)

| Featurizer | Description | Dimensions |
|------------|-------------|------------|
| `desc2D` | 2D Molecular Descriptors | ~200 |
| `mordred` | Mordred Descriptors | ~1600 |

#### Neural Embeddings (Slow, GPU Recommended)

| Featurizer | Description | Dimensions |
|------------|-------------|------------|
| `ChemBERTa-77M-MLM` | ChemBERTa masked language model | 384 |
| `ChemBERTa-77M-MTR` | ChemBERTa multi-task regression | 384 |
| `MolT5` | Molecular T5 embeddings | 768 |
| `Roberta-Zinc480M-102M` | RoBERTa trained on ZINC | 768 |
| `gin_supervised_*` | Graph neural network embeddings | 300 |

### Usage Examples

```python
from themap.features import MoleculeFeaturizer

# Initialize featurizer
featurizer = MoleculeFeaturizer(
    featurizer_name="ecfp",
    n_jobs=8
)

# Featurize a list of SMILES
smiles_list = ["CCO", "CCCO", "CC(=O)O"]
features = featurizer.featurize(smiles_list)

print(f"Features shape: {features.shape}")
# Features shape: (3, 2048)
```

#### Batch Processing with Deduplication

```python
from themap.features import MoleculeFeaturizer

featurizer = MoleculeFeaturizer(featurizer_name="ecfp")

# Featurize multiple datasets with global deduplication
datasets = {
    "task1": dataset1,  # MoleculeDataset objects
    "task2": dataset2,
}

features = featurizer.featurize_datasets(
    datasets,
    deduplicate=True  # Avoid re-computing for duplicate SMILES
)

for task_id, task_features in features.items():
    print(f"{task_id}: {task_features.shape}")
```

## Protein Featurizer

### ProteinFeaturizer

::: themap.features.protein.ProteinFeaturizer
    options:
      show_root_heading: true
      heading_level: 3

### Available Models

#### ESM2 Models

| Model | Parameters | Layers | Embedding Dim |
|-------|------------|--------|---------------|
| `esm2_t6_8M_UR50D` | 8M | 6 | 320 |
| `esm2_t12_35M_UR50D` | 35M | 12 | 480 |
| `esm2_t30_150M_UR50D` | 150M | 30 | 640 |
| `esm2_t33_650M_UR50D` | 650M | 33 | 1280 |

#### ESM3 Models

| Model | Description |
|-------|-------------|
| `esm3_sm_open_v1` | ESM3 small open model |

### Usage Examples

```python
from themap.features import ProteinFeaturizer

# Initialize with ESM2
featurizer = ProteinFeaturizer(
    model_name="esm2_t33_650M_UR50D",
    device="cuda"  # Use GPU if available
)

# Featurize protein sequences
sequences = [
    "MKTVRQERLKSIVRILERSKEPVSG",
    "MGSSHHHHHHSSGLVPRGSHM"
]

embeddings = featurizer.featurize(sequences)
print(f"Embeddings shape: {embeddings.shape}")
# Embeddings shape: (2, 1280)
```

#### Reading from FASTA Files

```python
from themap.features.protein import read_fasta_file

# Read sequences from FASTA
sequences = read_fasta_file("proteins.fasta")

for seq_id, sequence in sequences.items():
    print(f"{seq_id}: {len(sequence)} residues")
```

## Feature Cache

### FeatureCache

::: themap.features.cache.FeatureCache
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.features import FeatureCache

# Initialize cache
cache = FeatureCache(cache_dir="cache/features")

# Check if features are cached
cache_key = "ecfp_task1"
if cache.has(cache_key):
    features = cache.load(cache_key)
else:
    features = compute_features()
    cache.save(cache_key, features)
```

#### Automatic Caching

```python
from themap.features import MoleculeFeaturizer, FeatureCache

cache = FeatureCache(cache_dir="cache/")
featurizer = MoleculeFeaturizer(
    featurizer_name="ecfp",
    cache=cache  # Enable automatic caching
)

# First call computes and caches
features1 = featurizer.featurize(smiles_list)

# Second call loads from cache (fast)
features2 = featurizer.featurize(smiles_list)
```

## Performance Optimization

### Choosing the Right Featurizer

```python
def choose_featurizer(dataset_size: int, accuracy_priority: bool) -> str:
    """Choose appropriate featurizer based on requirements."""
    if dataset_size > 100000:
        return "ecfp"  # Fast fingerprints for large datasets
    elif accuracy_priority:
        return "ChemBERTa-77M-MLM"  # Neural embeddings for accuracy
    else:
        return "desc2D"  # Good balance of speed and quality
```

### Parallel Processing

```python
from themap.features import MoleculeFeaturizer

# Use multiple CPU cores
featurizer = MoleculeFeaturizer(
    featurizer_name="mordred",
    n_jobs=16  # Use 16 parallel workers
)
```

### GPU Acceleration

```python
from themap.features import ProteinFeaturizer

# Use GPU for neural models
featurizer = ProteinFeaturizer(
    model_name="esm2_t33_650M_UR50D",
    device="cuda:0"  # Specific GPU
)

# Batch processing for efficiency
embeddings = featurizer.featurize(
    sequences,
    batch_size=32  # Process 32 sequences at a time
)
```

## Error Handling

```python
from themap.features import MoleculeFeaturizer

featurizer = MoleculeFeaturizer(featurizer_name="ecfp")

# Handle invalid SMILES
smiles_list = ["CCO", "invalid_smiles", "CCCO"]

try:
    features = featurizer.featurize(smiles_list)
except ValueError as e:
    print(f"Invalid SMILES: {e}")
    
# Or use safe mode
features = featurizer.featurize(
    smiles_list,
    on_error="skip"  # Skip invalid molecules
)
```

## Integration with Distance Computation

```python
from themap.features import MoleculeFeaturizer
from themap.distance import compute_dataset_distance_matrix
import numpy as np

# Featurize datasets
featurizer = MoleculeFeaturizer(featurizer_name="ecfp")

source_features = featurizer.featurize(source_smiles)
target_features = featurizer.featurize(target_smiles)

# Compute distances
distances = compute_dataset_distance_matrix(
    source_features,
    target_features,
    method="euclidean"
)
```

## Constants

### Available Featurizer Names

```python
from themap.features.molecule import (
    FINGERPRINT_FEATURIZERS,
    DESCRIPTOR_FEATURIZERS,
    NEURAL_FEATURIZERS,
)

print("Fingerprints:", FINGERPRINT_FEATURIZERS)
print("Descriptors:", DESCRIPTOR_FEATURIZERS)
print("Neural:", NEURAL_FEATURIZERS)
```

### ESM Model Names

```python
from themap.features.protein import ESM2_MODELS, ESM3_MODELS

print("ESM2 models:", ESM2_MODELS)
print("ESM3 models:", ESM3_MODELS)
```

