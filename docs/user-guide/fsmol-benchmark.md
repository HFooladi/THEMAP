# FS-Mol Parity Benchmark

THEMAP's meta-learners (ProtoNet and MAML) are PyTorch re-implementations; the original
[FS-Mol](https://github.com/microsoft/FS-Mol) baselines are TensorFlow. Since THEMAP's
results are interpreted relative to FS-Mol, it is worth being able to confirm on demand
that the two land in the same ballpark.

`themap fsmol-benchmark` does that. It runs THEMAP's meta-learners on FS-Mol test tasks
under FS-Mol's own evaluation protocol and prints a side-by-side comparison with FS-Mol's
published per-task results.

This is a **parity check, not a leaderboard entry**. THEMAP encodes molecules with an MLP
over 2048-bit ECFP fingerprints, while FS-Mol's ProtoNet uses a GNN+ECFP+FC encoder with
Mahalanobis distance, so THEMAP is *expected* to score somewhat lower. The benchmark is
designed to tell "correct implementation, weaker encoder" apart from "something is broken".

## What you need

The FS-Mol dataset, laid out as `train/`, `valid/` and `test/` folders of `.jsonl.gz` task
files, plus a `fsmol_tasks_list.json` giving the canonical 4938/40/157 fold membership.
[Download it from FigShare](https://figshare.com/ndownloader/files/31345321).

FS-Mol's published baseline results are fetched automatically — seven CSVs, about 150 KB
total — and cached under `benchmarking_datasets/fsmol_reference/`. After the first run,
pass `--offline` to work without network access.

!!! note "This is not the `make download-fsmol` archive"
    That 16 GB Zenodo download contains the hardness companion data for the THEMAP paper
    (OTDD matrices, ESM-2 embeddings). It is unrelated to this benchmark.

## Getting to know the data first

Once the download is in place,
[`notebooks/research/fsmol_benchmark_explorer.ipynb`](https://github.com/HFooladi/THEMAP/blob/main/notebooks/research/fsmol_benchmark_explorer.ipynb)
is a visual tour of what is actually in the benchmark — task sizes, protein families,
chemical diversity, and how hard FS-Mol's own baselines find each task. It needs no GPU and
no extra download, and caches its scans under `notebooks/research/cache/` so only the first run is
slow (about four minutes).

It is worth reading before interpreting any benchmark output, because several structural
features of FS-Mol are not obvious from the paper and change how the numbers should be read:

- **Nearly half the test fold is one compound panel.** A single 157-molecule set is screened
  against 73 different kinases in `test` and 284 more in `train`. Those tasks are not
  independent evidence, and they are the reason the test-fold size distribution spikes at
  exactly 157.
- **The split is by assay, not by molecule.** 43% of test molecules also appear in a training
  assay, and 78 of the 157 test tasks have a training task with a byte-identical molecule set.
- **Train and test assays differ in kind.** Training tasks have a median of 46 molecules and a
  mean internal Tanimoto near 0.45 — single lead series. Test tasks have 157 molecules at
  around 0.12 — broad screening decks.
- **Per-task difficulty ranges from 0.008 to 0.449 ΔAUPRC**, so a single benchmark mean hides
  most of what is going on.

## Running it

Pick a representative subset of the test tasks, then benchmark against it:

```bash
# Deterministically choose 20 of the 157 test tasks
themap fsmol-subset benchmarking_datasets/fsmol_datasets \
    -o benchmarking_datasets/fsmol_subset_20.json

# Meta-train on the full FS-Mol training fold and evaluate the subset
themap fsmol-benchmark benchmarking_datasets/fsmol_datasets \
    --subset-file benchmarking_datasets/fsmol_subset_20.json \
    --algorithm proto --algorithm maml \
    --support-sizes 16,32,64,128,256 --seeds 10 --device cuda
```

Omit `--subset-file` to evaluate all 157 test tasks. Featurization is cached, so only the
first run pays for it (roughly five minutes for the whole training fold).

Expect around 35 minutes per algorithm on one GPU at FS-Mol's full meta-training budget
(`--num-epochs 100 --episodes-per-epoch 100 --meta-batch-size 16`, i.e. 10,000 outer steps
of 16 tasks each). Use `CUDA_VISIBLE_DEVICES` to pin a device and run algorithms in
parallel.

## How the subset is chosen

`themap fsmol-subset` stratifies deterministically along three axes: EC super-class of the
protein target (FS-Mol's test set is 125 kinases out of 157), task size, and difficulty as
measured by FS-Mol ProtoNet's ΔAUPRC at support size 16. Size and difficulty are not
independent — larger tasks are somewhat harder — so both are needed. Within each cell the
task closest to the cell centre is taken, and a quota ensures enough large tasks to make
the biggest support sizes meaningful.

Subsetting does not bias the comparison, because FS-Mol's results are published *per task*:
the reference means are recomputed on exactly the tasks being evaluated. The subset is
chosen well simply so those means stay close to the full-benchmark ones, which the command
reports.

## Protocol alignment

Several defaults differ from THEMAP's own meta-learning workflow. The benchmark changes
them to match FS-Mol; each is available as a flag.

| Aspect | THEMAP default | Benchmark | Why |
|---|---|---|---|
| Query set | fractional holdout, shared across support sizes | support of size N, query = the entire remainder (`--query-mode fsmol`) | Under the holdout, N=128 is infeasible for ~74% of FS-Mol test tasks |
| Support draw | forced 50/50 per class | proportionally stratified | Matches FS-Mol's `StratifiedTaskSampler`; 18 test tasks are not near-balanced |
| Episode sizes | requested shot is required | requested shot is a maximum (`--no-adaptive` to disable) | A 64-shot request would otherwise drop ~83% of FS-Mol's training assays, whose median size is 44 |
| Repeats | 5 | 10 (`--seeds`) | FS-Mol's `--num-runs` |
| Validation | one held-out source task | FS-Mol's own `valid` fold | Meaningful early stopping |

## Reading the output

Artifacts land in the output directory:

| File | Contents |
|---|---|
| `report.md` | The rendered comparison and the acceptance-criteria verdict |
| `comparison_summary.csv` | Mean ΔAUPRC ± standard error per arm and support size |
| `per_task_results.csv` | Long form: one row per task, support size, seed and arm |
| `correlations.csv` | Per-task Spearman/Pearson agreement with FS-Mol ProtoNet |
| `representativeness.csv` | Subset reference means vs the full 157-task benchmark |
| `acceptance_criteria.csv` | Each criterion, its threshold, and whether it passed |
| `comparison.png` | ΔAUPRC curves, and the per-task scatter against FS-Mol ProtoNet |
| `config.json` | Full configuration, resolved task ids, pool statistics, git SHA |

### Two things to know when interpreting the numbers

**ΔAUPRC recomputed from FS-Mol's per-task CSVs sits uniformly 0.005 above the paper's
Table 2**, for every method — a detail of how the paper averaged the prevalence term. The
report compares recomputed against recomputed so the offset cancels; don't compare THEMAP's
numbers to the printed table directly.

**Error bars are across tasks, not across seeds.** FS-Mol's ±0.008 for ProtoNet is the
spread over its 157 test tasks. The report averages seeds within a task first, then takes
the standard error over tasks, so the two are comparable.

### Acceptance criteria

The most informative check is whether THEMAP's meta-learner beats **FS-Mol's random
forest**. RF consumes the same ECFP fingerprints THEMAP does, so the encoder gap cannot
explain a loss to it — that would point at the meta-learning itself.

Alongside it: the meta-learner must beat its own from-scratch MLP on identical features,
ΔAUPRC must improve with support size, and per-task Spearman correlation with FS-Mol
ProtoNet should be at least 0.5. That last one is the sharpest signal, because it survives
a constant offset from the weaker encoder — a broken implementation has no reason to find
the same tasks easy that FS-Mol finds easy.
