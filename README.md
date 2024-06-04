# THEMAP
Task Hardness Estimation for Molecular Activity Predcition


## Installation
`THEMAP` can be installed using pip. First, create a new conda environment with the required packages. Then, clone this repository, and finally, install the repository using pip.

```bash
conda env create -f environment.yml
conda activate themap
 
pip install --no-deps -e .
```

## Getting Started
For the FS-Mol dataset, moleuclar embedding for each assay (ChEMBL id) and also, chemical and protein distance have been calculated and deposited in the [zenodo](https://zenodo.org/records/10605093). 

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

You can build and run documentation server with:

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
  year={2024},
  publisher={ACS Publications}
}
```
