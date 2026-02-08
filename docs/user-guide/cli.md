# Command Line Interface

THEMAP provides a powerful command-line interface for running distance computations without writing Python code. This guide covers all available commands and their options.

## Installation

The CLI is automatically installed when you install THEMAP:

```bash
pip install -e .
```

Verify the installation:

```bash
themap --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `themap run` | Run pipeline with a YAML configuration file |
| `themap quick` | Quick distance computation with minimal configuration |
| `themap init` | Create a sample configuration file |
| `themap convert` | Convert CSV files to JSONL.GZ format |
| `themap info` | Show information about a dataset directory |
| `themap list-featurizers` | List available molecule and protein featurizers |

## Quick Distance Computation

The fastest way to compute distances:

```bash
themap quick datasets/
```

### Options

```bash
themap quick [OPTIONS] DATA_DIR

Options:
  -o, --output TEXT      Output directory (default: output)
  -f, --featurizer TEXT  Molecule featurizer (default: ecfp)
  -m, --method TEXT      Distance method (default: euclidean)
  -j, --n-jobs INTEGER   Number of parallel jobs (default: 8)
```

### Examples

```bash
# Basic usage with defaults
themap quick datasets/

# Custom featurizer and method
themap quick datasets/ --featurizer maccs --method cosine

# Specify output directory
themap quick datasets/ --output results/my_experiment

# Use more parallel workers
themap quick datasets/ --n-jobs 16
```

## Run Pipeline with Configuration

For reproducible experiments, use a YAML configuration file:

```bash
themap run config.yaml
```

### Options

```bash
themap run [OPTIONS] CONFIG

Arguments:
  CONFIG  Path to YAML configuration file (required)

Options:
  -o, --output PATH      Output directory (overrides config)
  --molecule-only        Only compute molecule distances
  --protein-only         Only compute protein distances
  -j, --n-jobs INTEGER   Number of parallel jobs
  -v, --verbose          Enable verbose output
```

### Examples

```bash
# Run with config file
themap run config.yaml

# Override output directory
themap run config.yaml --output results/experiment_01

# Only compute molecular distances
themap run config.yaml --molecule-only

# Verbose output for debugging
themap run config.yaml --verbose
```

## Initialize Configuration

Create a sample configuration file to get started:

```bash
themap init
```

### Options

```bash
themap init [OPTIONS]

Options:
  -o, --output TEXT      Output file path (default: config.yaml)
  --data-dir PATH        Data directory to use in config
```

### Examples

```bash
# Create default config.yaml
themap init

# Create config with custom name
themap init --output my_config.yaml

# Create config pointing to specific data directory
themap init --data-dir /path/to/datasets
```

### Generated Configuration

The generated configuration file looks like:

```yaml
data:
  directory: "datasets"
  task_list: null  # Auto-discover all files

molecule:
  enabled: true
  featurizer: "ecfp"
  method: "euclidean"

protein:
  enabled: false

output:
  directory: "output"
  format: "csv"
  save_features: true

compute:
  n_jobs: 8
  device: "auto"
```

## Convert CSV to JSONL.GZ

Convert CSV files to THEMAP's native JSONL.GZ format:

```bash
themap convert data.csv CHEMBL123456
```

### Options

```bash
themap convert [OPTIONS] INPUT_CSV TASK_ID

Arguments:
  INPUT_CSV  Path to the CSV file (required)
  TASK_ID    Task identifier, e.g., CHEMBL123456 (required)

Options:
  -o, --output PATH          Output file path
  --smiles-column TEXT       SMILES column name (auto-detected if not specified)
  --activity-column TEXT     Activity column name (auto-detected if not specified)
  --no-validate              Skip SMILES validation
```

### Examples

```bash
# Basic conversion (auto-detect columns)
themap convert data.csv CHEMBL123456

# Specify output path
themap convert data.csv CHEMBL123456 --output datasets/train/CHEMBL123456.jsonl.gz

# Specify column names
themap convert data.csv CHEMBL123456 --smiles-column SMILES --activity-column pIC50

# Skip SMILES validation (faster but less safe)
themap convert data.csv CHEMBL123456 --no-validate
```

### Expected CSV Format

The CSV file should have at minimum:

- A SMILES column (auto-detected names: `SMILES`, `smiles`, `Smiles`, `canonical_smiles`)
- An activity column (auto-detected names: `Property`, `Activity`, `pIC50`, `Label`)

```csv
SMILES,Property
CCO,1
CCCO,0
CC(=O)O,1
```

## Dataset Information

Show information about a dataset directory:

```bash
themap info datasets/
```

### Output

```
Dataset Directory: /path/to/datasets
Task list provided: True

Folds:
  train:
    Tasks: 10
    CSV files: 0
    JSONL.GZ files: 10
  test:
    Tasks: 3
    CSV files: 0
    JSONL.GZ files: 3

Proteins: 13 FASTA files
```

## List Available Featurizers

View all available molecule and protein featurizers:

```bash
themap list-featurizers
```

### Output

```
Molecule Featurizers:

  Fingerprints (fast):
    - ecfp
    - maccs
    - topological
    - avalon

  Descriptors (medium):
    - desc2D
    - mordred

  Neural Embeddings (slow, requires GPU):
    - ChemBERTa-77M-MLM
    - ChemBERTa-77M-MTR
    - MolT5
    - Roberta-Zinc480M-102M


Protein Featurizers:

  ESM2 Models:
    - esm2_t6_8M_UR50D
    - esm2_t12_35M_UR50D
    - esm2_t30_150M_UR50D
    - esm2_t33_650M_UR50D

  ESM3 Models:
    - esm3_sm_open_v1
```

## Global Options

All commands support these global options:

```bash
themap [OPTIONS] COMMAND

Options:
  -v, --verbose  Enable verbose output
  --help         Show help message
```

## Environment Variables

THEMAP respects these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `THEMAP_DATA_DIR` | Default data directory | `datasets/` |
| `THEMAP_CACHE_DIR` | Feature cache directory | `cache/` |
| `CUDA_VISIBLE_DEVICES` | GPU devices for neural featurizers | All GPUs |

## Workflow Examples

### Complete Workflow

```bash
# 1. Check available featurizers
themap list-featurizers

# 2. Initialize configuration
themap init --output my_experiment.yaml --data-dir datasets/TDC

# 3. Edit the configuration file as needed
# (use your favorite editor)

# 4. Validate your data
themap info datasets/TDC

# 5. Run the pipeline
themap run my_experiment.yaml --verbose

# 6. Results are saved to output directory
```

### Converting External Data

```bash
# 1. Convert multiple CSV files
for csv in raw_data/*.csv; do
    task_id=$(basename "$csv" .csv)
    themap convert "$csv" "$task_id" --output datasets/train/"$task_id".jsonl.gz
done

# 2. Verify the conversion
themap info datasets/

# 3. Run distance computation
themap quick datasets/ --output results/
```

## Troubleshooting

### Common Issues

**Command not found:**
```bash
# Ensure THEMAP is installed
pip install -e .

# Or use python -m
python -m themap.cli --help
```

**Missing dependencies:**
```bash
# Install all optional dependencies
pip install -e ".[all]"
```

**Memory errors:**
```bash
# Use faster methods for large datasets
themap quick datasets/ --method euclidean --featurizer ecfp
```

**GPU not detected:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
CUDA_VISIBLE_DEVICES="" themap run config.yaml
```

## Next Steps

- Learn about [distance computation](distance-computation.md)
- Explore the [Python API](../api/distance.md)
- Check out [examples](../examples/index.md)
