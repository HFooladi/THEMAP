# Run on Colab

Want to see what THEMAP does before installing anything? Run these notebooks
directly in your browser on Google Colab — no local setup needed.

| Notebook | What it covers | Runtime | Open |
| --- | --- | --- | --- |
| **5-minute quick tour** | Install THEMAP, download 5 sample ChEMBL datasets, compute a 2×3 distance matrix with `quick_distance`, and visualise it as a heatmap. | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HFooladi/THEMAP/blob/main/notebooks/colab/01_quick_tour.ipynb) |
| **API deep dive** | Walk through the building blocks: `DatasetLoader`, `MoleculeFeaturizer`, `DatasetDistance`, YAML pipelines, featurizer/metric comparison, and a PCA task landscape over 13 datasets. | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HFooladi/THEMAP/blob/main/notebooks/colab/02_api_deep_dive.ipynb) |
| **OTDD deep dive** | Run OTDD on the same ChEMBL assays with three featurizers (ECFP, `desc2D`, ChemBERTa). Compare OTDD against Euclidean/Cosine, and see why OTDD's Gaussian inner approximation rewards continuous representations and struggles with binary fingerprints. | GPU (T4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HFooladi/THEMAP/blob/main/notebooks/colab/03_otdd_deep_dive.ipynb) |

All three notebooks live in
[`notebooks/colab/`](https://github.com/HFooladi/THEMAP/tree/main/notebooks/colab).
The first two run end-to-end on a free Colab CPU runtime; the OTDD notebook needs
a GPU (a free T4 works) because OTDD's Wasserstein computation is GPU-bound at any
practical scale.

!!! tip
    On Colab, select **Runtime → Change runtime type → T4 GPU** before running the
    OTDD notebook.
