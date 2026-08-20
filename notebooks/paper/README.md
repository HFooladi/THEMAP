# Paper reproduction — JCIM 2024

These three notebooks reproduce the figures of:

> Hosein Fooladi, Steffen Hirte, Johannes Kirchmair.
> **Quantifying the hardness of bioactivity prediction tasks for transfer learning.**
> *Journal of Chemical Information and Modeling* **64**(10), 4031–4046 (2024).
> [doi:10.1021/acs.jcim.4c00160](https://doi.org/10.1021/acs.jcim.4c00160)

```bibtex
@article{fooladi2024quantifying,
  title={Quantifying the hardness of bioactivity prediction tasks for transfer learning},
  author={Fooladi, Hosein and Hirte, Steffen and Kirchmair, Johannes},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={10},
  pages={4031-4046},
  year={2024},
  publisher={ACS Publications},
  doi={10.1021/acs.jcim.4c00160}
}
```

This folder is **frozen**. It changes only when a fix is required to keep reproduction
working — a library rename, a dependency break. New experiments never land here; they go in
[`../research/`](../research/).

## What you need

**1. The code, at the pinned version.**

```bash
git clone https://github.com/HFooladi/THEMAP.git
cd THEMAP
git checkout v0.6.0
source install.sh    # creates .venv and installs themap[all,dev,test]
```

Python 3.10, 3.11 or 3.12. The notebooks need the optional `ml` extras (torch, ESM); the
install above covers them.

**2. A LaTeX installation.** Notebook `03` writes PDFs through matplotlib's `pgf` backend,
which shells out to LaTeX. Without it, `03` runs but fails at the final save. TeX Live or
MiKTeX both work.

**3. The companion data (~16 GB download, ~31 GB extracted).**

```bash
make download-fsmol
```

This fetches [Zenodo record 10605093](https://zenodo.org/records/10605093), verifies MD5
`10644660a53d8d106b6883cb53eb1f3b`, and extracts to `datasets/fsmol_hardness/`. It holds
precomputed OTDD distance matrices, ESM-2 protein embeddings, internal chemical-hardness
measures, and ProtoNet evaluation summaries — everything needed to rebuild the figures
without re-running the expensive embedding pipelines.

## Run order

Run the notebooks in numeric order; `03` composes the quantities the first two analyse. Each
locates the repository root on its own, so launch Jupyter from wherever you like.

### `01_external_chemical_hardness.ipynb`

External chemical-space hardness. Sweeps *k* ∈ {1, 10, 50, 100, 500, 1000, 2000, 4000}
nearest source tasks and correlates OTDD distance against ProtoNet ROC-AUC, then
cross-compares hardness computed from six molecular featurizers.

*Reads* `ext_chem/otdd_{gin_supervised_infomax,gin_supervised_masking,gin_supervised_contextpred,unimol,desc2D,Roberta-Zinc480M-102M}.pkl`,
`int_chem/train/rf_16.pkl`, `FSMol_Eval_ProtoNet/summary/ProtoNet_summary_num_train_requested_{16,32,64,128}.csv`.
*Writes* nothing — figures render inline.

### `02_external_protein_hardness.ipynb`

The protein-side mirror. Same *k*-sweep against ProtoNet ROC-AUC and ΔAUPRC, then compares
hardness across the five ESM-2 model sizes to test representation sensitivity.

*Reads* `ext_prot/esm2_t{6_8M,12_35M,30_150M,33_650M,36_3B}_UR50D.pkl`,
`FSMol_Eval_ProtoNet/summary/*.csv`.
*Writes* nothing — figures render inline.

### `03_task_hardness.ipynb`

The headline notebook. Combines external chemical, external protein and internal chemical
hardness into a composite score and correlates each component against ProtoNet performance
at support-set sizes 16/32/64/128.

*Reads* `ext_chem/otdd_gin_supervised_infomax.pkl`, `ext_prot/esm2_t*_UR50D.pkl`,
`int_chem/{train,test}/rf_16.pkl`, `FSMol_Eval_ProtoNet/summary/*.csv`.
*Writes* four files into `assets/`:

| File | Figure |
|---|---|
| `assets/EXT_CHEM_EXT_PROT_hist.pdf` | Hardness distributions, external chemical vs external protein |
| `assets/hardness_comparisson.pdf` | The three hardness measures side by side |
| `assets/Hardness_vs_ProtoNet.pdf` | Lettered-panel regression grid, hardness vs ProtoNet performance |
| `assets/Hardness_vs_ProtoNet.svg` | The same figure as SVG |

These are gitignored — reproduction regenerates them rather than diffing them.

It is the slow one: it loads several GB of pickles.

## Notes

- Notebook outputs are stripped on commit by the `nbstripout` pre-commit hook, so the
  notebooks are checked in clean. You will see the figures only after running them.
- `datasets/fsmol_hardness/` also contains `embeddings/` and
  `FSMol_Eval_randomForest/summary/`, which these notebooks do not read. They are shipped so
  you can rebuild the OTDD matrices from raw embeddings if you want to go a level deeper.

**Last verified reproduced:** 2026-08-20 — all three notebooks executed top to bottom on
Python 3.12.13 against the Zenodo archive, and `03` regenerated all four figures.
