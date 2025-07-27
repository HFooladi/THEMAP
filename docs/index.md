# THEMAP: Task Hardness Estimation for Molecular Activity Prediction

**THEMAP** is a comprehensive Python library designed to aid drug discovery by providing powerful methods for estimating task hardness and computing transferability maps for bioactivity prediction tasks. It enables researchers and chemists to efficiently determine the similarity between molecular datasets and make informed decisions about transfer learning strategies.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline: **Quick Start**

    ---

    Get up and running with THEMAP in minutes

    [:octicons-arrow-right-24: Getting started](user-guide/getting-started.md)

-   :material-book-open-variant: **Tutorials**

    ---

    Step-by-step guides for common workflows

    [:octicons-arrow-right-24: View tutorials](tutorials/index.md)

-   :material-code-braces: **API Reference**

    ---

    Detailed documentation for all modules

    [:octicons-arrow-right-24: API docs](api/distance.md)

-   :material-script-text-outline: **Examples**

    ---

    Ready-to-use code examples and scripts

    [:octicons-arrow-right-24: Browse examples](examples/index.md)

</div>

## Key Features

### 🧪 Multi-Modal Distance Computation
- **Molecular datasets**: OTDD, Euclidean, and Cosine distances
- **Protein sequences**: ESM2-based embeddings and similarity metrics
- **Metadata integration**: Assay descriptions and experimental conditions
- **Combined analysis**: Multi-modal fusion strategies

### 🎯 Task Hardness Estimation
- **Transfer learning guidance**: Identify similar tasks for knowledge transfer
- **Difficulty quantification**: Estimate prediction task complexity
- **Resource optimization**: Prioritize computational resources effectively

### ⚡ Production-Ready Framework
- **Scalable architecture**: Handle large-scale dataset comparisons
- **Caching system**: Efficient feature storage and reuse
- **Error handling**: Robust validation and error recovery
- **GPU acceleration**: CUDA support for intensive computations

### 🔬 Unified Task System
- **Integrated data management**: Molecules, proteins, and metadata in one framework
- **Flexible organization**: Train/validation/test fold management
- **Feature extraction**: Unified API for multi-modal featurization

## Installation

=== "Basic Installation"

    ```bash
    # Clone repository
    git clone https://github.com/HFooladi/THEMAP.git
    cd THEMAP

    # Create conda environment
    conda env create -f environment.yml
    conda activate themap

    # Install THEMAP
    pip install --no-deps -e .
    ```

=== "With All Features"

    ```bash
    # Install with all optional dependencies
    pip install -e ".[all]"
    ```

=== "Development Setup"

    ```bash
    # Install development dependencies
    pip install -e ".[dev,test]"

    # Run tests
    python run_tests.py
    ```

## Quick Examples

### Molecular Dataset Distance

Compute distances between molecular datasets to understand chemical space similarity:

```python
from themap.data.tasks import Tasks
from themap.distance import MoleculeDatasetDistance

# Load tasks from directory structure
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True
)

# Compute molecular distances using OTDD
mol_distance = MoleculeDatasetDistance(
    tasks=tasks,
    molecule_method="otdd"
)

distances = mol_distance.get_distance()
print(distances)
# {'CHEMBL2219358': {'CHEMBL1023359': 7.074}}
```

### Unified Multi-Modal Analysis

Combine molecular, protein, and metadata information for comprehensive task comparison:

```python
from themap.distance import TaskDistance

# Compute combined distances from multiple modalities
task_distance = TaskDistance(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean"
)

# Get all distance types
all_distances = task_distance.compute_all_distances(
    combination_strategy="weighted_average",
    molecule_weight=0.7,
    protein_weight=0.3
)

print(f"Molecule distances: {len(all_distances['molecule'])} tasks")
print(f"Protein distances: {len(all_distances['protein'])} tasks")
print(f"Combined distances: {len(all_distances['combined'])} tasks")
```

### Protein Similarity Analysis

Analyze protein similarity using advanced sequence embeddings:

```python
from themap.distance import ProteinDatasetDistance

# Compute protein distances using ESM2 embeddings
prot_distance = ProteinDatasetDistance(
    tasks=tasks,
    protein_method="euclidean"
)

distances = prot_distance.get_distance()
# Organized as {target_task: {source_task: distance, ...}, ...}
```

## Use Cases

### Drug Discovery Workflows
- **Target identification**: Find similar protein targets for drug repurposing
- **Chemical space analysis**: Understand molecular diversity across datasets
- **Assay development**: Identify related bioactivity assays

### Transfer Learning Applications
- **Source task selection**: Choose optimal training data for new targets
- **Model adaptation**: Quantify domain shift between datasets
- **Performance prediction**: Estimate model performance on new tasks

### Computational Biology
- **Protein function prediction**: Leverage sequence similarity for annotation
- **Chemical-protein interaction**: Model molecular-target relationships
- **Multi-omics integration**: Combine molecular and protein data

## Performance

THEMAP is optimized for both accuracy and computational efficiency:

| Method | Speed | Memory | Accuracy | Best For |
|--------|-------|--------|----------|----------|
| **OTDD** | Slower | High | Highest | Small-medium datasets |
| **Euclidean** | Fast | Low | Good | Large datasets |
| **Cosine** | Fast | Low | Good | High-dimensional features |

### Scalability Features
- **Parallel processing**: Multi-core distance computations
- **Memory management**: Efficient handling of large datasets
- **Caching system**: Reuse expensive feature computations
- **Batch processing**: Handle thousands of dataset comparisons

## Citation

If you use THEMAP in your research, please cite:

```bibtex
@software{themap2024,
  title={THEMAP: Task Hardness Estimation for Molecular Activity Prediction},
  author={Hosein Fooladi},
  year={2024},
  url={https://github.com/HFooladi/THEMAP}
}
```

## Community and Support

### Get Help
- **Documentation**: Comprehensive guides and API reference
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and best practices

### Contributing
We welcome contributions! See our [contribution guidelines](https://github.com/HFooladi/THEMAP/blob/main/CONTRIBUTING.md) for:
- Code contributions
- Documentation improvements
- Bug reports and feature requests
- Example workflows and tutorials

### License
THEMAP is released under the MIT License. See [LICENSE](https://github.com/HFooladi/THEMAP/blob/main/LICENSE) for details.

---

## What's Next?

<div class="grid cards" markdown>

-   **New to THEMAP?**

    ---

    Start with our comprehensive getting started guide

    [:octicons-arrow-right-24: Getting started](user-guide/getting-started.md)

-   **Want to compute distances?**

    ---

    Learn about all available distance computation methods

    [:octicons-arrow-right-24: Distance computation](user-guide/distance-computation.md)

-   **Working with real data?**

    ---

    Understand the unified task system for multi-modal data

    [:octicons-arrow-right-24: Task system](UNIFIED_TASK_SYSTEM.md)

-   **Need inspiration?**

    ---

    Browse our collection of examples and use cases

    [:octicons-arrow-right-24: Examples](examples/index.md)

</div>
