# Data Module

The data module provides classes for loading, managing, and converting molecular and protein datasets.

## MoleculeDataset

::: themap.data.molecule_dataset.MoleculeDataset
    options:
      show_root_heading: true
      heading_level: 3

## DatasetLoader

::: themap.data.loader.DatasetLoader
    options:
      show_root_heading: true
      heading_level: 3

## Tasks

::: themap.data.tasks.Tasks
    options:
      show_root_heading: true
      heading_level: 3

## Task

::: themap.data.tasks.Task
    options:
      show_root_heading: true
      heading_level: 3

## CSVConverter

::: themap.data.converter.CSVConverter
    options:
      show_root_heading: true
      heading_level: 3

## TorchMoleculeDataset

::: themap.data.torch_dataset.TorchMoleculeDataset
    options:
      show_root_heading: true
      heading_level: 3

## Data Format

### Directory Structure

```
datasets/
├── sample_tasks_list.json
├── train/
│   ├── CHEMBL123456.jsonl.gz
│   ├── CHEMBL123456.fasta
│   └── ...
├── test/
│   └── ...
└── valid/
    └── ...
```

### JSONL.GZ Format

Each file contains molecules as JSON lines:

```json
{"SMILES": "CCO", "Property": 1}
{"SMILES": "CCCO", "Property": 0}
```

### Task List Format

```json
{
    "train": ["CHEMBL123456", "CHEMBL789012"],
    "test": ["CHEMBL111111", "CHEMBL222222"],
    "valid": ["CHEMBL333333"]
}
```

See [Getting Started](../user-guide/getting-started.md) for setup instructions.
