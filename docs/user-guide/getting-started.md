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
from themap import quick_distance
print("Installation successful!")
```

## Quick Start

### 1. One-liner Distance Computation

The simplest way to compute distances between molecular datasets:

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",          # Directory with train/ and test/ folders
    output_dir="output",          # Where to save results
    molecule_featurizer="ecfp",   # Fingerprint type (ecfp, maccs, etc.)
    molecule_method="euclidean",  # Distance metric
)

# Results saved to output/molecule_distances.csv
```

### 2. Config File Approach

For reproducible experiments, use a YAML configuration:

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
```

### 3. Analyzing Results

```python
import pandas as pd

# Load computed distances
distances = pd.read_csv("output/molecule_distances.csv", index_col=0)

# Find closest source for each target
for target in distances.columns:
    closest = distances[target].idxmin()
    dist = distances[target].min()
    print(f"{target} <- {closest} (distance: {dist:.4f})")

# Estimate task hardness (average distance to k-nearest sources)
k = 3
for target in distances.columns:
    hardness = distances[target].nsmallest(k).mean()
    print(f"Task hardness for {target}: {hardness:.4f}")
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
    from themap import quick_distance
    print("Distance module available")
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install -e '.[all]'")
```

### Memory Issues

For large datasets, use faster distance methods:

```python
from themap import quick_distance

# Use euclidean instead of OTDD for large datasets
results = quick_distance(
    data_dir="datasets",
    molecule_featurizer="ecfp",
    molecule_method="euclidean",  # Faster than OTDD
)
```

### Data Format Issues

Ensure your data follows the expected format:

```
datasets/
├── train/
│   └── CHEMBL123456.jsonl.gz
└── test/
    └── CHEMBL111111.jsonl.gz
```

Each `.jsonl.gz` file should contain JSON lines:
```json
{"SMILES": "CCO", "Property": 1}
{"SMILES": "CCCO", "Property": 0}
```

Ready to dive deeper? Continue with our [comprehensive tutorials](../tutorials/index.md)!
