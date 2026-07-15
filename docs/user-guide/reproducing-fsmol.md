# Reproducing FS-Mol Experiments

The companion data for our paper *"Quantifying the hardness of bioactivity
prediction tasks for transfer learning"* (J. Chem. Inf. Model. 64(10), 4031–4046,
2024) is published on [Zenodo (record 10605093)](https://zenodo.org/records/10605093).
It contains pre-computed OTDD distance matrices across multiple molecular
featurizers, ESM-2 protein embeddings, internal chemical hardness measures, and
ProtoNet evaluation summaries on the FS-Mol benchmark — everything needed to
reproduce the figures and tables without re-running the expensive embedding
pipelines.

## 1. Install dependencies

```bash
source install.sh    # creates .venv and installs themap[all,dev,test]
```

The reproduction notebooks rely on the optional `ml` extras (torch, ESM, etc.);
the all-in-one install above covers them.

## 2. Download the dataset (~16 GB)

You need ~35 GB of free disk space (16 GB zip + ~16 GB extracted). The script
downloads with resume support, verifies the MD5 checksum, extracts into
`datasets/fsmol_hardness/`, and removes the zip when done.

```bash
make download-fsmol
# or, equivalently:
python scripts/download_fsmol_data.py
```

Useful flags: `--keep-zip` (don't delete the archive after extraction), `--force`
(re-download), `--no-verify` (skip MD5 — only if you've already verified
out-of-band), `--dest DIR` (custom location).

After it completes (~31 GB extracted) you should see:

```
datasets/fsmol_hardness/
├── ext_chem/                       # OTDD distance matrices per molecular featurizer
├── ext_prot/                       # ESM-2 protein-distance matrices (t6_8M ... t36_3B)
├── int_chem/{train,test}/          # Internal chemical hardness (RF baselines)
├── embeddings/                     # Per-task molecular embeddings used to compute the OTDDs
├── FSMol_Eval_ProtoNet/summary/    # ProtoNet performance per support-set size (16/32/64/128)
└── FSMol_Eval_randomForest/summary/  # Random-forest baseline performance summaries
```

The reproduction notebooks read from `ext_chem/`, `ext_prot/`, `int_chem/`, and
`FSMol_Eval_ProtoNet/`; the other two directories are provided so you can rebuild
the OTDD matrices from raw embeddings if desired.

??? note "Manual download (no Python)"

    ```bash
    mkdir -p datasets/fsmol_hardness
    cd datasets
    wget -c https://zenodo.org/records/10605093/files/fsmol_hardness.zip
    echo "10644660a53d8d106b6883cb53eb1f3b  fsmol_hardness.zip" | md5sum -c -
    unzip fsmol_hardness.zip -d fsmol_hardness/
    ```

## 3. Run the reproduction notebooks

```bash
cd notebooks
jupyter lab        # or: jupyter notebook
```

| Notebook | What it reproduces |
| --- | --- |
| [`external_chemical_hardness.ipynb`](https://github.com/HFooladi/THEMAP/blob/main/notebooks/external_chemical_hardness.ipynb) | External chemical-space hardness: correlation between k-nearest source-task OTDD distance and ProtoNet performance, across molecular featurizers (GIN, UniMol, ChemBERTa/Roberta-Zinc, desc2D). |
| [`external_protein_hardness.ipynb`](https://github.com/HFooladi/THEMAP/blob/main/notebooks/external_protein_hardness.ipynb) | External protein-space hardness: correlation between target/source protein-embedding distance and performance, across ESM-2 model sizes (t6_8M → t36_3B). |
| [`task_hardness.ipynb`](https://github.com/HFooladi/THEMAP/blob/main/notebooks/task_hardness.ipynb) | Combined task-hardness score (external chemical + external protein + internal chemical) and its correlation with ProtoNet performance at support-set sizes 16/32/64/128. |

Notebook paths are resolved relative to the `notebooks/` directory, so launch
Jupyter from there. Outputs are auto-stripped on commit by the pre-commit hook
(`nbstripout`).
