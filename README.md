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

### 🧬 **Flexible Molecular Representations**
- **GIN (Graph Isomorphism Network)**: Deep learning-based molecular embeddings
- **Traditional fingerprints**: Support for various molecular fingerprint methods
- **Custom embeddings**: Integrate your own molecular representation methods

### 🏗️ **Ready-to-Use Datasets**
- **FS-Mol integration**: Pre-computed distances for Few-Shot Molecular property prediction datasets
- **ChEMBL compatibility**: Direct integration with ChEMBL bioactivity data
- **Custom dataset support**: Easy integration with your own chemical datasets

## 🚀 Quick Start

### Installation
`THEMAP` can be installed using pip. First, clone this repository, create a new conda environment with the required packages, and finally, install the repository using pip.

```bash
conda env create -f environment.yml
conda activate themap

pip install --no-deps git+https://github.com/HFooladi/otdd.git  
pip install --no-deps -e .
```

## Getting Started

### Basic Usage - Calculate Dataset Distance
  
```python
import os
from dpu_utils.utils.richpath import RichPath

from themap.data import MoleculeDataset
from themap.distance import MoleculeDatasetDistance

# Load source and target datasets
source_dataset_path = RichPath.create(os.path.join("datasets", "train", "CHEMBL1023359.jsonl.gz"))
target_dataset_path = RichPath.create(os.path.join("datasets", "test", "CHEMBL2219358.jsonl.gz"))
source_dataset = MoleculeDataset.load_from_file(source_dataset_path)
target_dataset = MoleculeDataset.load_from_file(target_dataset_path)

# Calculate molecular embeddings
molecule_featurizer = "gin_supervised_infomax"
source_features = source_dataset.get_dataset_embedding(molecule_featurizer)
target_features = target_dataset.get_dataset_embedding(molecule_featurizer)

# Compute dataset distance using OTDD
Dist = MoleculeDatasetDistance(D1=source_dataset, D2=target_dataset, method="otdd")
distance = Dist.get_distance()

print(distance)
# Output: {'CHEMBL2219358': {'CHEMBL1023359': 7.074298858642578}}
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

### Tests
```bash
pytest
```

### Code Style
```bash
ruff check
ruff format
```

### Documentation
```bash
mkdocs serve
```

## 📚 Citation

If you use THEMAP in your research, please cite our paper:

## Citation <a name="citation"></a>
If you find the models useful in your research, we ask that you cite the following paper:

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