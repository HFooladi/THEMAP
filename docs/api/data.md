# Data Module

The data module provides classes and utilities for loading, managing, and converting molecular and protein datasets.

## Overview

The data system consists of these main components:

- **`MoleculeDataset`** - Container for molecular data (SMILES, labels)
- **`DatasetLoader`** - Load datasets from directory structures
- **`CSVConverter`** - Convert CSV files to JSONL.GZ format
- **`Tasks`** - Unified task management across train/test/valid splits

## MoleculeDataset

### MoleculeDataset

::: themap.data.molecule_dataset.MoleculeDataset
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.data import MoleculeDataset

# Load from JSONL.GZ file
dataset = MoleculeDataset.from_jsonl_gz("datasets/train/CHEMBL123456.jsonl.gz")

print(f"Number of molecules: {len(dataset)}")
print(f"SMILES: {dataset.smiles_list[:3]}")
print(f"Labels: {dataset.labels[:3]}")
```

#### Creating from Data

```python
from themap.data import MoleculeDataset

# Create from lists
dataset = MoleculeDataset(
    smiles_list=["CCO", "CCCO", "CC(=O)O"],
    labels=[1, 0, 1],
    task_id="my_task"
)

# Save to file
dataset.to_jsonl_gz("output/my_task.jsonl.gz")
```

## DatasetLoader

### DatasetLoader

::: themap.data.loader.DatasetLoader
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.data import DatasetLoader

# Initialize loader
loader = DatasetLoader(
    data_dir="datasets",
    task_list_file="datasets/sample_tasks_list.json"
)

# Load all datasets
train_datasets = loader.load_fold("train")
test_datasets = loader.load_fold("test")

print(f"Loaded {len(train_datasets)} training datasets")
print(f"Loaded {len(test_datasets)} test datasets")
```

#### Loading Specific Tasks

```python
from themap.data import DatasetLoader

loader = DatasetLoader(data_dir="datasets")

# Load specific dataset
dataset = loader.load_dataset(
    task_id="CHEMBL123456",
    fold="train"
)

# Load multiple datasets
datasets = loader.load_datasets(
    task_ids=["CHEMBL123456", "CHEMBL789012"],
    fold="train"
)
```

#### Get Dataset Statistics

```python
from themap.data import DatasetLoader

loader = DatasetLoader(data_dir="datasets")
stats = loader.get_statistics()

print(f"Data directory: {stats['data_dir']}")
for fold, fold_stats in stats['folds'].items():
    print(f"  {fold}: {fold_stats['task_count']} tasks")
```

## CSVConverter

### CSVConverter

::: themap.data.converter.CSVConverter
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.data.converter import CSVConverter
from pathlib import Path

# Initialize converter
converter = CSVConverter(
    validate_smiles=True,
    auto_detect_columns=True
)

# Convert a CSV file
stats = converter.convert(
    input_path=Path("data/raw.csv"),
    output_path=Path("datasets/train/CHEMBL123456.jsonl.gz"),
    task_id="CHEMBL123456"
)

print(f"Converted {stats.valid_molecules}/{stats.total_rows} molecules")
print(f"Success rate: {stats.success_rate:.1f}%")
```

#### Specifying Column Names

```python
from themap.data.converter import CSVConverter
from pathlib import Path

converter = CSVConverter(validate_smiles=True)

stats = converter.convert(
    input_path=Path("data.csv"),
    output_path=Path("output.jsonl.gz"),
    task_id="my_task",
    smiles_column="canonical_smiles",
    activity_column="pIC50"
)
```

#### Batch Conversion

```python
from themap.data.converter import CSVConverter
from pathlib import Path

converter = CSVConverter()

# Convert multiple files
csv_files = Path("raw_data").glob("*.csv")

for csv_file in csv_files:
    task_id = csv_file.stem
    output_path = Path(f"datasets/train/{task_id}.jsonl.gz")
    
    stats = converter.convert(csv_file, output_path, task_id)
    print(f"{task_id}: {stats.valid_molecules} molecules")
```

## Tasks

### Tasks

::: themap.data.tasks.Tasks
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.data.tasks import Tasks

# Load tasks from directory
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True
)

print(f"Train tasks: {tasks.get_num_fold_tasks('TRAIN')}")
print(f"Test tasks: {tasks.get_num_fold_tasks('TEST')}")
```

#### Accessing Tasks

```python
# Get task IDs by fold
train_ids = tasks.get_task_ids(fold="TRAIN")
test_ids = tasks.get_task_ids(fold="TEST")

# Get specific task
task = tasks.get_task("CHEMBL123456")
print(f"Task {task.task_id}: {len(task.molecule_dataset)} molecules")
```

#### Working with Features

```python
# Compute features for all tasks
all_features = tasks.compute_all_task_features(
    molecule_featurizer="ecfp",
    protein_featurizer="esm2_t33_650M_UR50D",
    folds=["TRAIN", "TEST"]
)

# Get features ready for distance computation
source_features, target_features, source_names, target_names = (
    tasks.get_distance_computation_ready_features(
        molecule_featurizer="ecfp",
        source_fold="TRAIN",
        target_folds=["TEST"]
    )
)
```

## Task Class

### Task

::: themap.data.tasks.Task
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.data.tasks import Task

# Access task data
task = tasks.get_task("CHEMBL123456")

# Molecular data
if task.molecule_dataset:
    smiles = task.molecule_dataset.smiles_list
    labels = task.molecule_dataset.labels
    
# Protein data
if task.protein_dataset:
    sequences = task.protein_dataset.sequences

# Get features
mol_features = task.get_molecule_features("ecfp")
prot_features = task.get_protein_features("esm2_t33_650M_UR50D")
```

## TorchDataset

### TorchMoleculeDataset

::: themap.data.torch_dataset.TorchMoleculeDataset
    options:
      show_root_heading: true
      heading_level: 3

### Usage Examples

```python
from themap.data.torch_dataset import TorchMoleculeDataset
from torch.utils.data import DataLoader

# Create PyTorch dataset
torch_dataset = TorchMoleculeDataset(
    dataset=molecule_dataset,
    featurizer="ecfp"
)

# Use with DataLoader
dataloader = DataLoader(
    torch_dataset,
    batch_size=32,
    shuffle=True
)

for batch in dataloader:
    features, labels = batch
    # Train your model...
```

## Data Format

### JSONL.GZ Format

THEMAP uses compressed JSON Lines format for molecular data:

```json
{"SMILES": "CCO", "Property": 1}
{"SMILES": "CCCO", "Property": 0}
{"SMILES": "CC(=O)O", "Property": 1}
```

### Directory Structure

```
datasets/
├── sample_tasks_list.json      # Task organization
├── train/
│   ├── CHEMBL123456.jsonl.gz   # Molecular data
│   ├── CHEMBL123456.fasta      # Protein sequences
│   └── ...
├── test/
│   └── ...
└── valid/
    └── ...
```

### Task List Format

```json
{
    "train": ["CHEMBL123456", "CHEMBL789012", ...],
    "test": ["CHEMBL111111", "CHEMBL222222", ...],
    "valid": ["CHEMBL333333", ...]
}
```

## Utility Functions

### Validation

```python
from themap.data.molecule_dataset import validate_smiles

# Validate a SMILES string
is_valid = validate_smiles("CCO")
print(f"Valid: {is_valid}")  # True

is_valid = validate_smiles("invalid")
print(f"Valid: {is_valid}")  # False
```

### Canonicalization

```python
from themap.data.molecule_dataset import canonicalize_smiles

# Canonicalize SMILES
canonical = canonicalize_smiles("C(C)O")
print(canonical)  # "CCO"
```

## Error Handling

```python
from themap.data import DatasetLoader, MoleculeDataset

try:
    loader = DatasetLoader(data_dir="datasets")
    dataset = loader.load_dataset("CHEMBL123456", fold="train")
except FileNotFoundError:
    print("Dataset file not found")
except ValueError as e:
    print(f"Invalid data format: {e}")
```

## Performance Tips

1. **Lazy loading**: Use `DatasetLoader` to load datasets on demand
2. **Caching**: Enable feature caching for repeated computations
3. **Batch processing**: Process datasets in batches for memory efficiency
4. **Parallel loading**: Use `n_jobs` parameter for parallel dataset loading

```python
from themap.data import DatasetLoader

# Parallel loading
loader = DatasetLoader(data_dir="datasets", n_jobs=8)
datasets = loader.load_all_folds()
```
