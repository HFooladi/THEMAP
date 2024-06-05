# Overview

THEMAP is a python library designed for aiding in drug discovery by providing powerful methods for estimating the hardness of bioactivity prediction task for transfer learning. It enables researchers and chemists to efficiently determine transferrabilty map for bioactivity prediction tasks. 


## Installation
`THEMAP` can be installed using pip. First, clone this repository, create a new conda environment with the required packages, and finally, install the repository using pip.

```bash
conda env create -f environment.yml
conda activate themap

pip install --no-deps git+https://github.com/HFooladi/otdd.git  
pip install --no-deps -e .
```

## Getting Started

You can calculate distance between chemical spces of two (or more) datasets using `MoleculeDatasetDistance` class.

```python
import os
from dpu_utils.utils.richpath import RichPath

from themap.data import MoleculeDataset
from themap.data.distance import MoleculeDatasetDistance

source_dataset_path = RichPath.create(os.path.join("datasets", "train", "CHEMBL1023359.jsonl.gz"))
target_dataset_path = RichPath.create(os.path.join("datasets", "test", "CHEMBL2219358.jsonl.gz"))

# load some datasets
source_dataset = MoleculeDataset.load_from_file(source_dataset_path)
target_dataset = MoleculeDataset.load_from_file(target_dataset_path)

# get the features
molecule_feaurizer = "gin_supervised_infomax"
source_features = source_dataset.get_dataset_embedding(molecule_feaurizer)
target_features = target_dataset.get_dataset_embedding(molecule_feaurizer)

# calculate the distance
Dist = MoleculeDatasetDistance(D1=source_dataset, D2=target_dataset, method="otdd")
Dist.get_distance()
>>> {'CHEMBL2219358': {'CHEMBL1023359': 7.074298858642578}}
```

You can calculate distance between protein spces of two (or more) proteins (metadata) using `ProteinDatasetDistance` class.
    
```python
from themap.data import ProteinDataset
from themap.data.distance import ProteinDatasetDistance

# load some datasets
source_protein = ProteinDataset.load_from_file("datasets/train/train_proteins.fasta")
source_protein = ProteinDataset.load_from_file("datasets/train/train_proteins.fasta")

# get the features
protein_featurizer = "esm2_t33_650M_UR50D"
source_protein_features = source_protein.get_features(protein_featurizer)
target_protein_features = target_protein.get_features(protein_featurizer)

# calculate the distance
Dist = ProteinDatasetDistance(source_protein, target_protein, "euclidean")
Dist.get_distance()
>>> {'CHEMBL2219236': {'CHEMBL2219236': 2.9516282297179703,
>>>  'CHEMBL2219358': 4.372332083302979,
>>>  'CHEMBL1963831': 4.258244298189887},
>>>  'CHEMBL2219358': {'CHEMBL2219236': 3.560959265946417,
>>>  'CHEMBL2219358': 2.005268985065835,
>>>  'CHEMBL1963831': 2.772209146380105},
>>>  'CHEMBL1963831': {'CHEMBL2219236': 3.3623606434721895,
>>>  'CHEMBL2219358': 1.9580669485355773,
>>>  'CHEMBL1963831': 2.452369399042511}}
```


## Tutorials

Check out the [tutorials](tutorials/Basics.ipynb) to get started.