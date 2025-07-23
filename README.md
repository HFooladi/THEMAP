[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs--jcim--3c01774-blue)](https://doi.org/10.1021/acs.jcim.4c00160)

<p align="center">
  <img src="assets/banner.png" alt="THEMAP Banner" style="max-width:100%;">
</p>

# THEMAP
**T**ask **H**ardness **E**stimation for **M**olecular **A**ctivity **P**rediction

## 🎯 Main Purpose

**THEMAP is a Python library designed to calculate distances between chemical datasets** for molecular activity prediction tasks. The primary goal is to enable intelligent dataset selection for:

- **Transfer Learning**: Identify the most relevant source datasets for your target prediction task
- **Domain Adaptation**: Measure dataset similarity to guide model adaptation strategies
- **Task Hardness Assessment**: Quantify how difficult a prediction task will be based on dataset characteristics
- **Dataset Curation**: Select optimal training datasets from large chemical databases like ChEMBL

## ✨ Key Features

### 🔬 **Multi-Level Distance Calculation**
- **Dataset-level distances**: Assess overall similarity between entire chemical datasets
- **Protein target distances**: Compare biological targets for bioactivity prediction tasks

### 📊 **Multiple Distance Metrics**
- **OTDD (Optimal Transport Dataset Distance)**: Advanced optimal transport-based dataset comparison
- **Protein distance**: Target-based similarity for bioactivity datasets
- **Method-specific calculations**: Different distance methods for molecules, proteins, and metadata
- **Combined distances**: Weighted combination of multiple data type distances

### 🧬 **Flexible Molecular Representations**
- **GIN (Graph Isomorphism Network)**: Deep learning-based molecular embeddings
- **Traditional fingerprints**: Support for various molecular fingerprint methods
- **Custom embeddings**: Integrate your own molecular representation methods

### 🏗️ **Ready-to-Use Datasets**
- **FS-Mol integration**: Pre-computed distances for Few-Shot Molecular property prediction datasets
- **ChEMBL compatibility**: Direct integration with ChEMBL bioactivity data
- **Custom dataset support**: Easy integration with your own chemical datasets

## 🚀 Quick Start

### Installation Options

THEMAP offers flexible installation options depending on your needs:

#### Option 1: Minimal Installation (Recommended for Getting Started)
For basic functionality without heavy ML dependencies:


#### Option 1: Full Installation with GPU Support
For complete functionality including OTDD and GPU acceleration:

```bash
# Create full environment (requires CUDA)
conda env create -f environment.yml
conda activate themap

# Install with all features
pip install -e . --no-deps
```

#### Option 2: Custom Installation
Install only the features you need:

```bash
# Minimal base
pip install -e .

# Add specific features as needed
pip install -e ".[otdd]"     # For optimal transport distances
pip install -e ".[protein]"  # For protein analysis
pip install -e ".[ml-gpu]"   # For GPU-accelerated ML
```


### 🧪 Testing Installation

Verify your installation works:

```bash
# Test basic functionality
python -c "from themap.data.molecule_dataset import MoleculeDataset; print('✅ Core functionality works')"
```

### 🆕 Enhanced Distance Calculation Interface

THEMAP now features an improved distance calculation system that supports:

- **Tasks-based approach**: Organize datasets into train/validation/test splits using the `Tasks` collection
- **Method-specific configuration**: Use different distance methods for molecules (`"otdd"`, `"euclidean"`, `"cosine"`), proteins (`"euclidean"`, `"cosine"`), and metadata (`"euclidean"`, `"cosine"`, `"jaccard"`, `"hamming"`)
- **Unified API**: All distance classes (`MoleculeDatasetDistance`, `ProteinDatasetDistance`, `TaskDistance`) share the same interface
- **Flexible combination strategies**: Combine distances from multiple data types using various strategies (`"average"`, `"weighted_average"`, `"min"`, `"max"`)
- **Backward compatibility**: Existing code continues to work with the legacy interface

### Basic Usage Examples

#### Example 1: Dataset Loading and Basic Analysis (Works with Minimal Installation)

```python
import os
from dpu_utils.utils.richpath import RichPath
from themap.data.molecule_dataset import MoleculeDataset

# Load datasets
source_dataset_path = RichPath.create(os.path.join("datasets", "train", "CHEMBL1023359.jsonl.gz"))
source_dataset = MoleculeDataset.load_from_file(source_dataset_path)

# Basic dataset analysis (works with minimal installation)
print(f"Dataset size: {len(source_dataset)}")
print(f"Positive ratio: {source_dataset.get_ratio}")
print(f"Dataset statistics: {source_dataset.get_statistics()}")

# Validate dataset integrity
try:
    source_dataset.validate_dataset_integrity()
    print("✅ Dataset is valid")
except ValueError as e:
    print(f"❌ Dataset validation failed: {e}")
```

#### Example 2: Advanced ML Features (Requires ML Installation)

```python
# Only works with pip install -e ".[ml]" or higher
from themap.data.molecule_dataset import MoleculeDataset
dataset_path = RichPath.create(os.path.join("datasets", "train", "CHEMBL1023359.jsonl.gz"))

# Load dataset
dataset = MoleculeDataset.load_from_file(dataset_path)

# Calculate molecular embeddings (requires ML dependencies)
try:
    features = dataset.get_dataset_embedding("ecfp")
    print(f"Features shape: {features.shape}")
except ImportError:
    print("❌ ML dependencies not installed. Use: pip install -e '.[ml]'")
```

#### Example 3: Advanced Distance Calculation (Requires Full Installation)

```python
# Only works with pip install -e ".[all]" 
from themap.data.tasks import Tasks, Task
from themap.distance.tasks_distance import MoleculeDatasetDistance, TaskDistance

# Create Tasks collection from your datasets
source_dataset_path = RichPath.create(os.path.join("datasets", "train", "CHEMBL1023359.jsonl.gz"))
source_dataset = MoleculeDataset.load_from_file(source_dataset_path)
target_dataset_path = RichPath.create(os.path.join("datasets", "test", "CHEMBL2219358.jsonl.gz"))
target_dataset = MoleculeDataset.load_from_file(source_dataset_path)
source_task = Task(task_id="CHEMBL1023359", molecule_dataset=source_dataset)
target_task = Task(task_id="CHEMBL2219358", molecule_dataset=target_dataset)

# Option 1: Create Tasks collection with train/test split
tasks = Tasks(train_tasks=[source_task], test_tasks=[target_task])

# Option 2: Compute molecule distance with method-specific configuration
try:
    # Use different methods for different data types
    mol_dist = MoleculeDatasetDistance(
        tasks=tasks,
        molecule_method="otdd",     # OTDD for molecules
    )
    mol_dist._compute_features()
    distance = mol_dist.get_distance()
    print(distance)
    # Output: {'CHEMBL2219358': {'CHEMBL1023359': 7.074298858642578}}
    
    # Option 3: Use TaskDistance for comprehensive analysis
    task_dist = TaskDistance(
        tasks=tasks,
        molecule_method="otdd",
        protein_method="cosine"
    )
    
    # Compute different types of distances
    molecule_distances = task_dist.compute_molecule_distance()
    # protein_distances = task_dist.compute_protein_distance()  # If protein data available
    
    # Or compute all distances at once
    all_distances = task_dist.compute_all_distances(
        molecule_method="otdd",
        combination_strategy="weighted_average",
        molecule_weight=0.7,
        protein_weight=0.3
    )
    print("Molecule distances:", all_distances["molecule"])
    print("Combined distances:", all_distances["combined"])
    
except ImportError:
    print("❌ Distance calculation dependencies not installed. Use: pip install -e '.[all]'")
```


## 📈 Use Cases

### 1. **Transfer Learning Dataset Selection**
```python
# Find the most similar training datasets for your target task
candidate_datasets = ["CHEMBL1023359", "CHEMBL2219358", "CHEMBL1243967"]
target_dataset = "my_target_assay"

distances = calculate_all_distances(candidate_datasets, target_dataset)
best_source = min(distances, key=distances.get)  # Closest dataset for transfer learning
```

### 2. **Domain Adaptation Assessment**
```python
# Assess how much domain shift exists between datasets
domain_gap = calculate_dataset_distance(source_domain, target_domain)
if domain_gap < threshold:
    print("Direct transfer likely to work well")
else:
    print("Domain adaptation strategies recommended")
```

### 3. **Task Hardness Prediction**
```python
# Predict task difficulty based on dataset characteristics
hardness_score = estimate_task_hardness(dataset, reference_datasets)
print(f"Predicted task difficulty: {hardness_score}")
```

## 🔬 Reproduce FS-Mol Experiments

For the FS-Mol dataset, molecular embeddings and distance matrices have been pre-computed and are available on [Zenodo](https://zenodo.org/records/10605093).

1. **Download pre-computed data**:
   ```bash
   # Download from https://zenodo.org/records/10605093
   # Unzip and place in datasets/fsmol_hardness
   ```

2. **Run analysis notebooks**:
   ```bash
   cd notebooks
   jupyter notebook  # Open and run the provided notebooks
   ```

## 🛠️ Development

### Setting Up Development Environment

For development, use the minimal installation with development tools:

```bash
# Create development environment
conda env create -f environment-minimal.yml
conda activate themap-minimal

# Install in development mode with dev tools
pip install -e ".[dev,test]"
```

### Running Tests

```bash
# Run pytest (if no import conflicts)
pytest

# Run tests with coverage
pytest --cov=themap
```

### Code Quality

```bash
# Check code style
ruff check
ruff format

# Type checking
mypy themap/

# Run all quality checks
ruff check && ruff format && mypy themap/
```

### Documentation

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### 🔧 Handling Import Issues During Development

If you encounter PyTorch/CUDA import issues during development:

1. **Use isolated testing**: `python run_isolated_tests.py`
2. **Avoid importing the full package**: Import modules directly
3. **Use minimal environment**: Develop with `environment-minimal.yml`
4. **Add lazy imports**: For heavy dependencies in your code

```python
# Example of lazy import pattern
def get_heavy_dependency():
    try:
        import heavy_ml_library
        return heavy_ml_library
    except ImportError:
        raise ImportError("Please install with: pip install -e '.[ml]'")
```

## 📚 Citation

If you use THEMAP in your research, please cite our paper:

```bibtex
@article{fooladi2024quantifying,
  title={Quantifying the hardness of bioactivity prediction tasks for transfer learning},
  author={Fooladi, Hosein and Hirte, Steffen and Kirchmair, Johannes},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={10},
  pages={4031-4046},
  year={2024},
  publisher={ACS Publications}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests to help improve THEMAP.

---

**Ready to optimize your chemical dataset selection for machine learning?** Start with THEMAP today! 🚀
