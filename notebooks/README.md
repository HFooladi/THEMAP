# THEMAP notebooks

Three folders, three different promises.

| Folder | What is in it | What you can rely on |
|---|---|---|
| [`paper/`](paper/) | The three notebooks that reproduce the JCIM 2024 hardness paper | **Frozen.** Pinned to a release tag, verified end to end, changed only when a fix is needed to keep reproduction working. Start here if you came from the paper. |
| [`tutorials/`](tutorials/) | Getting-started material — a local walkthrough and three Colab notebooks that need no install | Maintained. Kept working against the current release. |
| [`research/`](research/) | Ongoing experiments and analysis that post-date the paper | **No guarantees.** These evolve with the library and may break. |

## Reproducing the paper

Go to **[`paper/`](paper/)** and read its README. Nothing else in this tree is part of the
published results.

## Running any of them

Every notebook locates the repository root on its own and `chdir`s there, so it does not
matter which directory you launch Jupyter from:

```python
repo_path = next(str(p) for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
os.chdir(repo_path)
```

Outputs are stripped automatically on commit by the `nbstripout` pre-commit hook. Never
commit a notebook with embedded outputs; run `nbstripout path/to/nb.ipynb` to clean one by
hand.

## A note on older links

Before this reorganisation the notebooks all sat flat in `notebooks/`. If you are following
a link that expects the old layout — `notebooks/task_hardness.ipynb`,
`notebooks/colab/01_quick_tour.ipynb`, and so on — those paths still resolve at commit
[`6e07d19`](https://github.com/HFooladi/THEMAP/tree/6e07d19/notebooks).

| Old path | New path |
|---|---|
| `notebooks/external_chemical_hardness.ipynb` | `notebooks/paper/01_external_chemical_hardness.ipynb` |
| `notebooks/external_protein_hardness.ipynb` | `notebooks/paper/02_external_protein_hardness.ipynb` |
| `notebooks/task_hardness.ipynb` | `notebooks/paper/03_task_hardness.ipynb` |
| `notebooks/example.ipynb` | `notebooks/tutorials/example.ipynb` |
| `notebooks/colab/` | `notebooks/tutorials/colab/` |
| `notebooks/fsmol_benchmark_explorer.ipynb` | `notebooks/research/fsmol_benchmark_explorer.ipynb` |
| `notebooks/metalearning_reproduction.ipynb` | `notebooks/research/metalearning_reproduction.ipynb` |
| `notebooks/figure_for_presentation.ipynb` | `notebooks/research/figure_for_presentation.ipynb` |
