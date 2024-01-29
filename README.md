# THEMAP
Task Hardness Estimation for Molecular Activity Predcition


## Installation
THEMAP can be installed using pip. First, clone this reposiroty, and then install the repository using pip.

```bash
git clone https://github.com/HFooladi/THEMAP.git
cd THEMAP 
pip install -e .
```

## Getting Started
For the FS-Mol dataset, moleuclar embedding for each assay (ChEMBL id) and also, chemical and protein distance have been calculated and deposited in the [zenodo](https://zenodo.org/records/10580588). 

1. Download it from [zenodo](https://zenodo.org/records/10580588)
2. Unzip the directory and place it into `datasets` such that you have the path `datasets/fsmol_hardness`

Then, you can go to the `notebooks` folder, and run the notebooks.

## Citation <a name="citation"></a>
If you find the models useful in your research, we ask that you cite the following paper:

```bibtex
@article{fooladi2024qth,
  author = {Fooladi, Hosein and Hirte, Steffen and Kirchmair, Johannes},
  title={Quantifying the hardness of bioactivity prediction tasks for transfer learning},
  year={2024},
  doi={10.26434/chemrxiv-2024-871mt},
  url={https://chemrxiv.org/engage/chemrxiv/article-details/65b3cafd9138d23161cc5ea4},
  journal={ChemRxiv}
}
```
