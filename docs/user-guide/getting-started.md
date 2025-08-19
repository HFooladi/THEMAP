# Getting Started

Welcome to THEMAP! This guide will help you get up and running with task hardness estimation for molecular activity prediction.

## What is THEMAP?

THEMAP (Task Hardness Estimation for Molecular Activity Prediction) is a Python library designed to aid drug discovery by providing powerful methods for estimating the difficulty of bioactivity prediction tasks. It enables researchers to:

- **Compute distances between molecular datasets** using various metrics (OTDD, Euclidean, Cosine)
- **Analyze protein similarity** through sequence and structural features
- **Estimate task hardness** for transfer learning scenarios
- **Build transferability maps** for bioactivity prediction tasks
- **Integrate multi-modal data** (molecules, proteins, metadata)

## Installation

### Prerequisites

- Python 3.10 or higher
- conda (recommended) or pip

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/HFooladi/THEMAP.git
cd THEMAP

# Create and activate conda environment
conda env create -f environment.yml
conda activate themap

# Install THEMAP
pip install --no-deps -e .
```

### Installation with Optional Dependencies

THEMAP has several optional dependency groups for different functionality:

```bash
# For basic molecular analysis
pip install -e ".[ml]"

# For protein analysis
pip install -e ".[protein]"

# For OTDD distance computation
pip install -e ".[otdd]"

# For all features
pip install -e ".[all]"

# For development
pip install -e ".[dev,test]"
```

### Verify Installation

```python
import themap
print(f"THEMAP version: {themap.__version__}")

# Test basic functionality
from themap.data import MoleculeDataset
from themap.distance import MoleculeDatasetDistance
print("✅ Installation successful!")
```

## Quick Start

### 1. Molecular Dataset Distance

Compute distances between molecular datasets to understand chemical space similarity:

```python
import os
from dpu_utils.utils.richpath import RichPath
from themap.data import MoleculeDataset
from themap.distance import MoleculeDatasetDistance

# Load datasets
source_path = RichPath.create("datasets/train/CHEMBL1023359.jsonl.gz")
target_path = RichPath.create("datasets/test/CHEMBL2219358.jsonl.gz")

source_dataset = MoleculeDataset.load_from_file(source_path)
target_dataset = MoleculeDataset.load_from_file(target_path)

# Compute distance using OTDD
distance_calc = MoleculeDatasetDistance(
    tasks=None,  # Will be auto-extracted from datasets
    molecule_method="otdd"
)

distances = distance_calc.get_distance()
print(distances)
# Output: {'CHEMBL2219358': {'CHEMBL1023359': 7.074298858642578}}
```

### 2. Protein Dataset Distance

Analyze protein similarity using sequence features:

```python
from themap.data import ProteinMetadataDatasets
from themap.distance import ProteinDatasetDistance

# Load protein datasets
source_proteins = ProteinMetadataDatasets.from_directory("datasets/train/")
target_proteins = ProteinMetadataDatasets.from_directory("datasets/test/")

# Compute euclidean distance between protein features
protein_distance = ProteinDatasetDistance(
    tasks=None,  # Will be auto-extracted
    protein_method="euclidean"
)

distances = protein_distance.get_distance()
print(distances)
```

### 3. Unified Task Analysis

Work with the unified task system that integrates multiple data modalities:

```python
from themap.data.tasks import Tasks
from themap.distance import TaskDistance

# Load integrated tasks
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    load_metadata=True,
    cache_dir="cache/"
)

print(f"Loaded {len(tasks)} tasks")
print(f"Train tasks: {tasks.get_num_fold_tasks('TRAIN')}")
print(f"Test tasks: {tasks.get_num_fold_tasks('TEST')}")

# Compute combined distances
task_distance = TaskDistance(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean"
)

# Get all distance types
all_distances = task_distance.compute_all_distances()
```

## Core Concepts

### Tasks and Data Modalities

THEMAP organizes data around the concept of **Tasks** - individual bioactivity prediction problems that can contain:

- **Molecular data**: SMILES strings, molecular descriptors, embeddings
- **Protein data**: Sequences, structural features, embeddings
- **Metadata**: Assay descriptions, experimental conditions, bioactivity values

### Distance Metrics

Different distance metrics are optimized for different scenarios:

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **OTDD** | Comprehensive dataset comparison | Most accurate, considers both features and labels | Computationally expensive |
| **Euclidean** | Fast similarity estimation | Fast, interpretable | May miss complex relationships |
| **Cosine** | Feature orientation comparison | Good for high-dimensional data | Ignores magnitude differences |

### Task Hardness

Task hardness quantifies how difficult a prediction task is, which helps in:

- **Transfer learning**: Identify similar tasks for knowledge transfer
- **Model selection**: Choose appropriate models based on task complexity
- **Resource allocation**: Prioritize difficult tasks for more computational resources

## Directory Structure

Understanding the expected directory structure helps organize your data:

```
your_project/
├── datasets/
│   ├── train/
│   │   ├── CHEMBL123.jsonl.gz    # Molecular data
│   │   ├── CHEMBL123.fasta       # Protein sequences
│   │   └── ...
│   ├── test/
│   │   └── ...
│   ├── valid/
│   │   └── ...
│   └── sample_tasks_list.json    # Task configuration
├── cache/                        # Feature caching
└── results/                      # Output analyses
```

### Sample Task List Format

```json
{
  "train": ["CHEMBL1023359", "CHEMBL1613776", ...],
  "test": ["CHEMBL2219358", "CHEMBL1963831", ...],
  "valid": ["CHEMBL2219236", ...]
}
```

## Next Steps

Now that you have the basics:

1. **Explore the tutorials**: Check out [detailed tutorials](../tutorials/index.md) for step-by-step examples
2. **Explore the code**: Understand the codebase structure for advanced usage
3. **Run the examples**: Execute the provided [example scripts](../examples/index.md)
4. **Join the community**: Contribute to the project on [GitHub](https://github.com/HFooladi/THEMAP)

## Common Issues

### Import Errors

If you encounter import errors:

```python
# Check if optional dependencies are installed
try:
    from themap.distance import MoleculeDatasetDistance
    print("✅ Distance module available")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install -e '.[otdd]'")
```

### Memory Issues

For large datasets:

```python
# Reduce memory usage
distance_calc = MoleculeDatasetDistance(
    tasks=tasks,
    molecule_method="euclidean"  # Use instead of OTDD for large datasets
)

# Or limit OTDD samples
hopts = {"maxsamples": 500}  # Default is 1000
```

### Data Format Issues

Ensure your data follows the expected format:

```python
# Validate molecular data
dataset = MoleculeDataset.load_from_file("your_data.jsonl.gz")
print(f"Dataset contains {len(dataset)} molecules")
print(f"Sample molecule: {dataset[0].smiles}")

# Validate protein data
proteins = ProteinMetadataDatasets.from_directory("proteins/")
print(f"Loaded {len(proteins)} protein sequences")
```

Ready to dive deeper? Continue with our [comprehensive tutorials](../tutorials/index.md)!
