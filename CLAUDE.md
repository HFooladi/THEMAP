# CLAUDE.md

THEMAP: library for computing distances between chemical datasets for molecular activity prediction. Architecture optimized for N×M distance matrix computation.

## Setup

```bash
source install.sh          # first time (uv-based, creates .venv)
source .venv/bin/activate   # reactivate later
```

## Commands

### Testing
- `python run_tests.py` — all tests
- `python run_tests.py unit` — unit tests only
- `python run_tests.py integration` — integration tests only
- `python run_tests.py distance` — distance module tests (`tests/unit/distance/`)
- `python run_tests.py fast` — skip `@pytest.mark.slow` tests
- `python run_tests.py coverage` — tests with coverage
- `pytest tests/unit/data/test_molecule_dataset_isolated.py -v` — specific file
- `pytest -k "test_name"` — keyword match
- `pytest -m "unit and not slow"` — fast unit tests

### Code Quality
- `ruff check --fix .` — lint with auto-fix
- `ruff format .` — format
- `mypy themap/` — type check

### Documentation
- `mkdocs serve` — local docs at http://127.0.0.1:8000
- `python build_docs.py build` — build static docs

### GitHub CLI
Use `gh` (not `gh_cli`) for GitHub interactions: `gh pr list`, `gh issue list`, `gh run list`.

## CI Requirements

CI runs these exact commands — verify locally before pushing:
```bash
ruff check .
ruff format --check .
mypy -p themap
pytest tests/ -m "not slow" --cov=themap --cov-report=xml
mkdocs build --strict
```
Tests run across Python 3.10, 3.11, 3.12.

## Project Structure

```
themap/
├── __init__.py              # lazy imports via __getattr__ (see Gotchas)
├── cli.py                   # click CLI: run, quick, featurize, init, convert, info, list-featurizers
├── config.py                # PipelineConfig, DataConfig, COMBINATION_STRATEGIES
├── data/
│   ├── molecule_dataset.py  # MoleculeDataset (SMILES + labels as numpy arrays)
│   ├── molecule_datasets.py # MoleculeDatasets (train/val/test fold manager)
│   ├── protein_datasets.py  # ProteinMetadataDataset
│   ├── task.py              # Task, Tasks (unified multi-modal abstraction)
│   ├── converter.py         # [mypy ignored]
│   ├── torch_dataset.py     # [mypy ignored]
│   └── exceptions.py        # FeaturizationError, InvalidSMILESError
├── distance/
│   ├── base.py              # DatasetDistance, MetadataDistance, DATASET_DISTANCE_METHODS, METADATA_DISTANCE_METHODS
│   ├── calculator.py        # TaskDistanceCalculator, combine_distance_matrices()
│   └── exceptions.py        # DistanceComputationError
├── pipeline/
│   ├── feature_store.py     # FeatureStore (disk cache, .npz format)
│   ├── featurization.py     # FeaturizationPipeline (batch featurize with SMILES dedup)
│   ├── orchestrator.py      # Pipeline (top-level entry point)
│   ├── output.py            # [mypy ignored]
│   ├── runner.py            # [mypy ignored]
│   └── cli.py               # [mypy ignored]
├── models/otdd/             # Optimal Transport Dataset Distance implementation
├── metalearning/            # [mypy ignored — entire subpackage]
├── features/cache.py        # [mypy ignored]
└── utils/
    ├── featurizer_utils.py  # get_featurizer(), AVAILABLE_FEATURIZERS
    ├── logging.py           # logging config
    └── config.py            # utility config helpers
```

## Key Constants

| Constant | Location | Values |
|---|---|---|
| `DATASET_DISTANCE_METHODS` | `themap/distance/base.py` | `["otdd", "euclidean", "cosine"]` |
| `METADATA_DISTANCE_METHODS` | `themap/distance/base.py` | `["euclidean", "cosine", "manhattan"]` |
| `AVAILABLE_FEATURIZERS` | `themap/utils/featurizer_utils.py` | `["ecfp", "fcfp", "maccs", "desc2D", "mordred", "ChemBERTa-77M-MLM", "ChemBERTa-77M-MTR", "MolT5", "Roberta-Zinc480M-102M", "gin_supervised_infomax", "gin_supervised_contextpred", "gin_supervised_edgepred", "gin_supervised_masking"]` |
| `COMBINATION_STRATEGIES` | `themap/config.py` | `["average", "weighted_average", "separate"]` |

## Testing

### Structure
```
tests/
├── conftest.py                              # shared fixtures
├── unit/
│   ├── data/                                # MoleculeDataset, MoleculeDatasets, protein tests
│   ├── distance/                            # test_base.py, test_protein_distance.py
│   └── pipeline/                            # test_config.py, test_output.py
└── integration/
    ├── test_pipeline_integration.py         # end-to-end pipeline
    └── test_distance_computation.py         # distance matrix computation
```

### Markers
- `@pytest.mark.unit` — isolated component tests (extensive mocking)
- `@pytest.mark.integration` — multi-component workflows
- `@pytest.mark.slow` — computationally expensive (excluded in CI)

### Conventions
- Tests use fixtures from `tests/conftest.py`
- Mock external dependencies (RDKit, network calls)
- Distance tests are in `tests/unit/distance/` (not `tests/distance/`)

## Code Standards

- **Line length**: 110 chars (ruff)
- **Import sorting**: automatic via ruff
- **Type hints**: required for public APIs, enforced by mypy
- **Docstrings**: Google-style for public functions
- **Pre-commit hooks**: ruff, mypy, trailing whitespace, end-of-file fixer

## Gotchas and Conventions

### Lazy imports in `themap/__init__.py`
`__getattr__` lazily imports heavy modules (`Pipeline`, `PipelineConfig`, `DatasetDistance`, `MoleculeDataset`, etc.) to keep `import themap` fast. When adding new public API symbols, add them to `__getattr__` and to the `__all__` list.

### Literal types and `typing.cast()`
Distance method parameters use `Literal` types (e.g., `Literal["euclidean", "cosine"]`). When passing dynamic strings to these functions, use `typing.cast()` to satisfy mypy.

### Mypy ignore_errors modules
These modules have `ignore_errors = true` in `pyproject.toml` — mypy won't flag errors in them:
`themap.metalearning.*`, `themap.data.converter`, `themap.data.torch_dataset`, `themap.features.cache`, `themap.pipeline.output`, `themap.pipeline.runner`, `themap.pipeline.cli`

### Custom exceptions
- `themap/data/exceptions.py`: `FeaturizationError`, `InvalidSMILESError`
- `themap/distance/exceptions.py`: `DistanceComputationError`

### Feature caching
- Molecule features: `.npz` format (features + labels per task)
- Metadata: `.npy` format (single vector per task)
- SMILES are deduplicated before featurization
- Cache dir defaults to `./feature_cache/`

### Dataset format
- Stored in `datasets/train/`, `datasets/test/`, `datasets/valid/`
- Molecules: `{CHEMBL_ID}.jsonl.gz`
- Proteins: `{CHEMBL_ID}.fasta`

### Optional dependencies
Heavy ML libraries (torch, molfeat, esm, etc.) are optional. Install groups are defined in `pyproject.toml` under `[project.optional-dependencies]`. Use `pip install -e ".[all]"` for everything, or selective groups like `.[ml]`, `.[protein]`, `.[otdd]`.

## Common Development Workflows

### Adding a new featurizer
1. Add to `get_featurizer()` in `themap/utils/featurizer_utils.py`
2. Add name to `AVAILABLE_FEATURIZERS` in same file
3. Add tests in `tests/unit/data/`

### Adding a new distance method
1. Implement in `DatasetDistance` or `MetadataDistance` in `themap/distance/base.py`
2. Add to `DATASET_DISTANCE_METHODS` or `METADATA_DISTANCE_METHODS`
3. Add tests in `tests/unit/distance/`

### Adding a new CLI command
1. Add click command in `themap/cli.py`
2. Register with the `@cli.command()` decorator
3. Add tests in `tests/unit/pipeline/`
