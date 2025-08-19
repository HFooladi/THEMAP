# THEMAP Examples

This directory contains example scripts demonstrating how to use THEMAP's various features and capabilities. Examples are organized by complexity and topic to help you learn progressively.

## 📁 Directory Structure

```
examples/
├── basic/          # Start here - fundamental concepts
├── distance/       # Distance computation examples  
├── advanced/       # Complex workflows and analysis
└── README.md       # This guide
```

## 🚀 Getting Started

**New to THEMAP?** Start with the `basic/` examples to learn fundamental concepts, then progress to `distance/` and `advanced/` examples.

## 📚 Example Categories

### 🔤 Basic Examples (`basic/`)

**Start here if you're new to THEMAP.** These examples cover the fundamental data structures and operations.

| Script | Description | Key Concepts |
|--------|-------------|--------------|
| `molecule_datasets_demo.py` | Working with molecular datasets | MoleculeDataset, loading data, featurization |
| `protein_datasets_demo.py` | Working with protein datasets | ProteinMetadataDataset, sequence handling, embeddings |
| `tasks_demo.py` | Understanding the Task system | Task creation, multi-modal data integration |
| `protein_features_example.py` | Protein feature computation | ESM embeddings, feature caching |

**Example Usage:**
```bash
cd examples/basic/
python molecule_datasets_demo.py
python protein_datasets_demo.py
```

### 📏 Distance Examples (`distance/`)

**Learn distance computation** between molecular and protein datasets.

| Script | Description | Key Concepts |
|--------|-------------|--------------|
| `task_distance_demo.py` | Basic distance computation | MoleculeDatasetDistance, ProteinDatasetDistance |
| `example_otdd_usage.py` | Optimal Transport Dataset Distance | OTDD, advanced distance metrics |
| `metadata_datasets_demo.py` | Metadata-based distances | Incorporating assay metadata |

**Example Usage:**
```bash
cd examples/distance/
python task_distance_demo.py
python example_otdd_usage.py
```

### 🔬 Advanced Examples (`advanced/`)

**Complex workflows** for research and analysis applications.

| Script | Description | Key Concepts |
|--------|-------------|--------------|
| `task_embedding_molecules.py` | Molecular task embeddings | Task representation learning, high-dimensional analysis |
| `task_hardness_protein.py` | Task difficulty analysis | Hardness metrics, task complexity evaluation |
| `metalearning_example.py` | Meta-learning workflows | Few-shot learning, task similarity |

**Example Usage:**
```bash
cd examples/advanced/
python task_embedding_molecules.py
python task_hardness_protein.py
```

## 🎯 Quick Start Guide

### 1. **First Time Users**
```bash
# Start with basic molecule handling
cd examples/basic/
python molecule_datasets_demo.py

# Learn about the unified task system
python tasks_demo.py
```

### 2. **Distance Computation**
```bash
# Basic distance between datasets
cd examples/distance/
python task_distance_demo.py

# Advanced OTDD distances
python example_otdd_usage.py
```

### 3. **Research Applications**
```bash
# Task representation learning
cd examples/advanced/
python task_embedding_molecules.py

# Analyze task difficulty
python task_hardness_protein.py
```

## 📋 Prerequisites

Most examples require:
- **Basic setup**: `conda activate themap` 
- **Sample data**: Available in `datasets/` directory
- **Dependencies**: Install with `pip install -e ".[dev,test]"`

Some advanced examples may require:
- **GPU**: For protein embeddings and large-scale computations
- **Additional data**: Some examples download datasets automatically

## 🔧 Common Usage Patterns

### Loading Your Own Data
```python
# For CSV files - convert first
# python scripts/csv_to_jsonl.py your_data.csv YOUR_ASSAY_ID --auto-detect

# Then use in examples
from themap.data import MoleculeDataset
dataset = MoleculeDataset.load_from_file("YOUR_ASSAY_ID.jsonl.gz")
```

### Customizing Examples
Most examples accept command-line arguments:
```bash
# Use smaller sample sizes for testing
python molecule_datasets_demo.py --sample-size 100

# Enable debug logging
python task_distance_demo.py --log-level DEBUG

# Use specific datasets
python tasks_demo.py --dataset CHEMBL123456
```

### Performance Tips
- **Start small**: Use `--sample-size` for initial testing
- **Use caching**: Enable feature caching with `--cache-features`
- **Parallel processing**: Most examples support `--max-workers N`

## 🐛 Troubleshooting

### Common Issues

**Import errors**: Make sure THEMAP is installed
```bash
conda activate themap
pip install -e ".[dev,test]"
```

**Missing data**: Download sample datasets
```bash
# Check if you have sample data
ls datasets/train/
ls datasets/test/
```

**Memory errors**: Reduce sample sizes
```bash
python example_script.py --sample-size 50
```

**Slow execution**: Enable caching and use fewer workers initially
```bash
python example_script.py --cache-features --max-workers 2
```

## 📖 Learning Path

We recommend this progression:

1. **Week 1**: Master `basic/` examples
   - Understand MoleculeDataset and ProteinDataset
   - Learn the Task system
   - Get comfortable with data loading

2. **Week 2**: Explore `distance/` examples  
   - Compute basic distances
   - Try different distance metrics
   - Understand OTDD concepts

3. **Week 3+**: Advanced applications
   - Task embeddings and representations
   - Hardness analysis
   - Meta-learning workflows

## 📚 Additional Resources

- **Documentation**: See `docs/` directory
- **Tutorials**: Interactive Jupyter notebooks in `docs/tutorials/`
- **API Reference**: `docs/api/` 
- **Pipeline Guide**: `docs/PIPELINE_GUIDE.md`

## 🤝 Contributing

Found a bug in an example? Want to add a new example?

1. **Bug reports**: Create an issue describing the problem
2. **New examples**: Follow the existing structure and naming conventions
3. **Improvements**: Submit a pull request with clear documentation

## 💡 Tips for Success

- **Read the docstrings**: Each script has detailed documentation
- **Start with small datasets**: Use `--sample-size` for initial exploration  
- **Check the logs**: Use `--log-level DEBUG` to understand what's happening
- **Experiment**: Modify parameters and see how results change
- **Ask for help**: Check issues or create new ones for questions

Happy learning with THEMAP! 🚀