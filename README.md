[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs--jcim--3c01774-blue)](https://doi.org/10.1021/acs.jcim.4c00160)


# THEMAP
Task Hardness Estimation for Molecular Activity Prediction


## Installation
`THEMAP` can be installed using pip. First, clone this repository, create a new conda environment with the required packages, and finally, install the repository using pip.

```bash
conda env create -f environment.yml
conda activate themap

pip install --no-deps git+https://github.com/HFooladi/otdd.git  
pip install --no-deps -e .
```

## Getting Started

### Basic Usage
  
```python
import os
from dpu_utils.utils.richpath import RichPath

from themap.data import MoleculeDataset
from themap.data.distance import MoleculeDatasetDistance

source_dataset_path = RichPath.create(os.path.join("datasets", "train", "CHEMBL1023359.jsonl.gz"))
target_dataset_path = RichPath.create(os.path.join("datasets", "test", "CHEMBL2219358.jsonl.gz"))
source_dataset = MoleculeDataset.load_from_file(source_dataset_path)
target_dataset = MoleculeDataset.load_from_file(target_dataset_path)

molecule_feaurizer = "gin_supervised_infomax"
source_features = source_dataset.get_dataset_embedding(molecule_feaurizer)
target_features = target_dataset.get_dataset_embedding(molecule_feaurizer)

Dist = MoleculeDatasetDistance(D1=source_dataset, D2=target_dataset, method="otdd")

Dist.get_distance()
>>> {'CHEMBL2219358': {'CHEMBL1023359': 7.074298858642578}}
```


### Reproduce FS-Mol Experiments
For the FS-Mol dataset, molecular embedding for each assay (ChEMBL id) and also, chemical and protein distance have been calculated and deposited in the [zenodo](https://zenodo.org/records/10605093). 

1. Download it from [zenodo](https://zenodo.org/records/10605093)
2. Unzip the directory and place it into `datasets` such that you have the path `datasets/fsmol_hardness`

Then, you can go to the `notebooks` folder, and run the notebooks.


## Development
### Tests

You can run tests locally with:

```bash
pytest
```

### Code style
We use `ruff` as a linter and formatter. 

```bash
ruff check
ruff format
```

### Documentation

You can build and run a documentation server with:

```bash
mkdocs serve
```

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
