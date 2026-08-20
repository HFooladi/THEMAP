# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The bibtex blocks in `README.md` and `notebooks/paper/README.md` now carry the paper's DOI, matching the structured `doi` field in `CITATION.cff`.

## [v0.6.0] - 2026-08-20

### Added
- **FS-Mol parity benchmark** (`themap fsmol-benchmark`): runs THEMAP's PyTorch meta-learners on FS-Mol test tasks under FS-Mol's own evaluation protocol and prints a side-by-side comparison with FS-Mol's published per-task baselines. Ships with `themap fsmol-subset` for deterministic selection of a representative 20-task subset, a torch-free reference-table loader, and acceptance criteria that separate "correct implementation, weaker encoder" from "something is broken".
- **FS-Mol benchmark explorer notebook**: a guided tour of the benchmark data itself — fold sizes, class balance, assay types, protein-family shift, molecule overlap between folds, internal chemical diversity, and how hard FS-Mol's own baselines find each task.
- `CITATION.cff`, so GitHub renders a "Cite this repository" entry for the JCIM 2024 paper.

### Changed
- **Notebooks reorganised by promise**: `notebooks/` is now split into `paper/` (frozen reproduction set for the JCIM 2024 hardness paper), `tutorials/` (local + Colab onboarding), and `research/` (post-paper exploration). Each folder has a README stating what it guarantees. Old paths still resolve at commit `6e07d19`.

  | Old | New |
  |---|---|
  | `notebooks/external_chemical_hardness.ipynb` | `notebooks/paper/01_external_chemical_hardness.ipynb` |
  | `notebooks/external_protein_hardness.ipynb` | `notebooks/paper/02_external_protein_hardness.ipynb` |
  | `notebooks/task_hardness.ipynb` | `notebooks/paper/03_task_hardness.ipynb` |
  | `notebooks/example.ipynb` | `notebooks/tutorials/example.ipynb` |
  | `notebooks/colab/` | `notebooks/tutorials/colab/` |
  | `notebooks/fsmol_benchmark_explorer.ipynb` | `notebooks/research/fsmol_benchmark_explorer.ipynb` |
  | `notebooks/metalearning_reproduction.ipynb` | `notebooks/research/metalearning_reproduction.ipynb` |
  | `notebooks/figure_for_presentation.ipynb` | `notebooks/research/figure_for_presentation.ipynb` |

- **Documentation site modernised**: refreshed theme, automated deployment, and updated content.

### Fixed
- **Notebook repo-root bootstrap**: notebooks derived the repository root as the parent of the working directory (`os.path.dirname(os.path.abspath(""))`), which only worked when Jupyter was launched from `notebooks/` and failed *silently* at any other depth. They now search upward for `pyproject.toml`, so the working directory no longer matters.
- **Transposed distance matrix in `example.ipynb`**: the heatmap rendered the matrix with its axes swapped. Notebook plotting is now unified through the shared style helper.
- **Meta-learning demo failed obscurely when the base assay was missing**; it now fails with a clear message.
- **Unreadable Colab heatmaps**: annotations overlapped and tick labels overprinted. Three causes — cells too narrow for their numbers, 13-character CHEMBL ids rotated into rows taller than the row, and a fixed `.2f` that overflowed on six-figure OTDD distances while collapsing small cosine distances to an identical `0.00`. Heatmaps now size to their content, drop the constant `CHEMBL` prefix from tick labels, and choose a format that fits.

## [v0.5.0] - 2026-07-15

### Added
- **Distance-guided meta-learning** (`themap.metalearning`): Prototypical Networks and MAML in PyTorch, an episodic meta-trainer, a low-data evaluator, and a runner that selects source tasks by distance (`themap metalearn`). Extended over the cycle with train-shot modes, shared query sets, ΔAUPRC reporting, and a distance-vs-random source-selection comparison (`themap metalearn-compare`).
- **Featurizer catalogue expanded from 13 to 27**, unified into a single source of truth in `themap/utils/featurizer_utils.py`.
- **`themap featurize` CLI command** for pre-computing and caching features without running a distance computation.
- **GPU acceleration for OTDD**: `compute.device` is now wired through to the OTDD backend.
- **Colab onboarding notebooks**: a 5-minute quick tour, an API deep dive, and an OTDD deep dive demonstrating why OTDD's Gaussian inner approximation rewards continuous representations over binary fingerprints.
- **Presentation hero figure notebook**: UMAP landscape of FS-Mol task centroids with OTDD nearest-source arrows.
- **Shared notebook plot style** via `set_plot_style()` (Set2 palette + CMU Serif, with graceful fallback).
- **CLI reference documentation**, and a root `config.yaml` so the pipeline runs out of the box.
- **Streamlined FS-Mol paper reproduction**, with `nbstripout` auto-stripping notebook outputs on commit.

### Changed
- `env.sh` renamed to `install.sh`.
- **Imports made lazy throughout**: molfeat, protein datasets, and other heavy dependencies now load on demand, keeping `import themap` fast and the feature subsystems isolated.
- **Logging professionalised** across the package.
- **Documentation site cleaned up**: API reference pages reorganised and docstring autoref warnings resolved.
- **CI/CD hardened**: workflow permissions, dependabot configuration, and uv cache keys.
- Makefile improved and documented in CLAUDE.md.

### Fixed
- **OTDD Distance Returning All `inf`**: Replaced removed `torch.symeig` calls with `torch.linalg.eigh` in vendored OTDD code, fixing compatibility with PyTorch 2.0+
- **OTDD Error Reporting**: Upgraded silent `warning` to `error` level logging, added exception type to messages, and added a post-computation summary of failed pairs
- **NaN Feature Handling in OTDD**: Added validation that detects and replaces NaN values in feature arrays before OTDD computation, preventing silent numerical failures
- **Vendored OTDD repaired against geomloss 0.3.x**, including numerical-instability fixes.
- OTDD now surfaces the underlying `ImportError` so a missing dependency can be diagnosed, and lazy-loads its plotting dependencies so distance computation works without `adjustText`.
- **Colab installs fixed**: install from GitHub rather than PyPI, drop the `molfeat[transformer]` extra to avoid a `tokenizers` source build, and bypass molfeat's ModelStore for ChemBERTa.
- `numba>=0.59` floored so the `[dev]` extra resolves on Python 3.12.
- Assorted CI failures fixed and the codebase prepared for PyPI release.

### Removed
- `run_pipeline.py` and `run_pipeline.sh` — redundant wrapper scripts superseded by the `themap` CLI

## [v0.4.0] - 2026-01-01

Major repository overhaul.

### Added
- **`env.sh` installation script** with uv support.
- Pipeline configuration examples and refreshed sample datasets.

### Changed
- **Data layer simplified for fast N×M distance computation** — the refactor behind the current `MoleculeDataset` / `MoleculeDatasets` / `Tasks` abstractions.
- CI switched to uv for faster dependency installation.
- Documentation expanded across the data and utils modules.

### Fixed
- Molecule distance computation corrected; L2 and cosine distances now use class prototypes.
- Installation setup and package dependency declarations repaired.
- Examples updated to work with the current THEMAP API.

## [v0.3.0] - 2025-08-19
### Added
- **Pipeline Infrastructure**: Complete configuration-driven pipeline system for distance computation workflows
  - `themap.pipeline` module with CLI, configuration management, and execution engine
  - Support for both directory-based dataset discovery and explicit dataset specification
  - YAML/JSON configuration files with validation and comprehensive examples

- **Distance Computation Fixes**: Corrected dataset-level distance computation methodology
  - Fixed Euclidean and Cosine distance implementations to work properly with variable-sized datasets
  - Implemented proper pairwise distance computation between individual molecules across datasets
  - Fixed method naming inconsistencies (`compute_distance()` → `get_distance()`)

- **Utility Scripts**: New data processing and conversion tools
  - `scripts/csv_to_jsonl.py` - Convert CSV files to THEMAP's native JSONL.GZ format with SMILES validation
  - `scripts/clean_smiles.py` - SMILES validation and cleanup utility for datasets

- **Examples Reorganization**: Structured example system by complexity level
  - `examples/basic/` - Introductory examples for new users
  - `examples/distance/` - Distance computation workflows
  - `examples/advanced/` - Complex research applications
  - Comprehensive example configurations in `configs/examples/`

- **Enhanced Documentation**: Pipeline usage guides and workflow documentation
  - `docs/PIPELINE_GUIDE.md` - Complete pipeline usage documentation
  - `docs/README_PIPELINE_SCRIPTS.md` - Utility scripts documentation

- **New Test Coverage**: Comprehensive test suite for new functionality
  - Unit tests for distance computation modules
  - Pipeline component tests
  - Configuration validation tests

### Fixed
- **Critical Distance Computation Bug**: Resolved incorrect Euclidean/Cosine distance implementation
  - Previous implementation incorrectly flattened entire feature matrices from different-sized datasets
  - Now correctly computes pairwise distances between individual molecules across datasets
  - Uses mean of pairwise distances as dataset-level distance metric

- **Pipeline Method Calls**: Fixed incorrect method names in pipeline execution
  - Updated all distance computation calls to use correct `get_distance()` method
  - Fixed TaskDistance instantiation and usage patterns

- **Dataset Loading**: Fixed explicit dataset specification mode
  - Individual dataset loading now creates proper Task objects for distance computation
  - Both directory-based and explicit dataset modes now work correctly

### Changed
- **Examples Structure**: Moved examples from `scripts/` to organized `examples/` directory
- **Distance Module**: Refactored distance computation classes for better consistency and correctness
- **Configuration System**: Enhanced pipeline configuration with better validation and error handling

### Removed
- Deprecated example scripts from `scripts/` directory (moved to `examples/`)
- Legacy `tasks_distance.py` module (functionality integrated into new distance classes)

## [v0.2.0] - 2025-07-23
### Added
- Initial changelog entry. Describe new features, changes, and fixes here.

## [v0.1.0] - 2025-03-11
### Added
- Initial release.
