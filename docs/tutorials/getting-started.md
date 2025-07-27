# Getting Started Tutorial

This tutorial will walk you through the basic concepts and usage of THEMAP for task hardness estimation and distance computation.

## Prerequisites

- THEMAP installed with basic dependencies
- Python 3.10+
- Basic knowledge of molecular datasets

## Tutorial Overview

In this tutorial, you will learn how to:

1. Load molecular and protein datasets
2. Compute basic distances between datasets
3. Work with the unified task system
4. Interpret distance results

## Step 1: Loading Your First Dataset

```python
from themap.data import MoleculeDataset
from dpu_utils.utils.richpath import RichPath

# Load a molecular dataset
dataset_path = RichPath.create("datasets/train/CHEMBL1023359.jsonl.gz")
dataset = MoleculeDataset.load_from_file(dataset_path)

print(f"Loaded dataset with {len(dataset)} molecules")
print(f"Task ID: {dataset.task_id}")
print(f"Sample molecule: {dataset[0].smiles}")
```

## Step 2: Computing Simple Distances

```python
from themap.distance import MoleculeDatasetDistance

# Load two datasets to compare
source_dataset = MoleculeDataset.load_from_file(
    RichPath.create("datasets/train/CHEMBL1023359.jsonl.gz")
)
target_dataset = MoleculeDataset.load_from_file(
    RichPath.create("datasets/test/CHEMBL2219358.jsonl.gz")
)

# Create distance calculator
distance_calc = MoleculeDatasetDistance(
    tasks=None,
    molecule_method="euclidean"  # Start with fastest method
)

# Set up the comparison
distance_calc.source_molecule_datasets = [source_dataset]
distance_calc.target_molecule_datasets = [target_dataset]
distance_calc.source_task_ids = [source_dataset.task_id]
distance_calc.target_task_ids = [target_dataset.task_id]

# Compute distance
result = distance_calc.get_distance()
print(f"Distance: {result}")
```

## Step 3: Working with Protein Data

```python
from themap.data import ProteinDataset
from themap.distance import ProteinDatasetDistance

# Load protein sequences
proteins = ProteinDataset.load_from_file("datasets/train/train_proteins.fasta")
print(f"Loaded {len(proteins)} protein sequences")

# Compute protein similarities
protein_distance = ProteinDatasetDistance(
    tasks=None,
    protein_method="euclidean"
)

protein_distance.source_protein_datasets = proteins
protein_distance.target_protein_datasets = proteins

distances = protein_distance.get_distance()
print("Protein distance matrix computed")
```

## Step 4: Understanding Results

Distance values have the following interpretations:

- **Lower values**: More similar datasets
- **Higher values**: More different datasets
- **Scale**: Depends on the method used

```python
# Analyze distance results
for target_id, source_distances in result.items():
    print(f"Target task: {target_id}")
    for source_id, distance in source_distances.items():
        print(f"  Distance from {source_id}: {distance:.3f}")
```

## Next Steps

Now that you understand the basics:

1. Try different distance methods (`cosine`, `otdd`)
2. Explore the unified task system
3. Learn about task hardness estimation
4. Check out the advanced examples

## Common Issues

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -e ".[ml]"
```

### Memory Issues
Use euclidean distance for large datasets instead of OTDD.

### File Not Found
Ensure your data files are in the correct directory structure.
