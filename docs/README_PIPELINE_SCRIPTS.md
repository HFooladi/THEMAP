# THEMAP Pipeline Runner Scripts

This directory contains convenient scripts to run the THEMAP pipeline with configuration files. You can choose between a bash script or a Python script based on your preference.

## Quick Start

### Option 1: Bash Script (Recommended)
```bash
# Make the script executable (if not already)
chmod +x run_pipeline.sh

# Run a simple pipeline
./run_pipeline.sh configs/examples/simple_directory_discovery.yaml

# Run with custom settings
./run_pipeline.sh --data-path /path/to/data --sample-size 100 configs/my_config.yaml
```

### Option 2: Python Script
```bash
# Run a simple pipeline
python run_pipeline.py configs/examples/simple_directory_discovery.yaml

# Run with custom settings
python run_pipeline.py --data-path /path/to/data --sample-size 100 configs/my_config.yaml
```

## Available Options

Both scripts support the same options:

| Option | Description | Example |
|--------|-------------|---------|
| `--data-path PATH` | Base path to dataset files | `--data-path /path/to/datasets` |
| `--output-dir PATH` | Override output directory | `--output-dir results/my_experiment` |
| `--log-level LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `--log-level DEBUG` |
| `--sample-size N` | Use only N samples (for testing) | `--sample-size 100` |
| `--max-workers N` | Number of parallel workers | `--max-workers 8` |
| `--validate-only` | Only validate config without running | `--validate-only` |
| `--dry-run` | Show what would be computed | `--dry-run` |
| `--list-examples` | List available example configs | `--list-examples` |
| `--help` | Show help message | `--help` |

## Example Usage

### 1. Basic Pipeline Run
```bash
# Using the new directory-based discovery
./run_pipeline.sh configs/examples/simple_directory_discovery.yaml
```

### 2. Testing with Small Sample
```bash
# Test with just 50 samples and debug logging
./run_pipeline.sh --sample-size 50 --log-level DEBUG configs/examples/comprehensive_multimodal.yaml
```

### 3. Custom Data Location
```bash
# Run with data in a different location
./run_pipeline.sh --data-path /mnt/my_datasets configs/my_config.yaml
```

### 4. Validation and Planning
```bash
# Check if your config is valid
./run_pipeline.sh --validate-only configs/my_config.yaml

# See what computations would be performed
./run_pipeline.sh --dry-run configs/my_config.yaml
```

### 5. Production Run with Custom Output
```bash
# Full run with custom output directory
./run_pipeline.sh --output-dir results/experiment_2025_01 --max-workers 8 configs/production_config.yaml
```

## Configuration Files

The scripts work with both the old explicit dataset format and the new directory-based discovery format:

### Directory-Based (Recommended for many datasets)
```yaml
name: "my_pipeline"
description: "Automatic dataset discovery"

molecule:
  directory:
    root_path: "datasets"
    task_list_file: "sample_tasks_list.json"
    load_molecules: true
    load_proteins: false
    load_metadata: true
  featurizers: ["ecfp", "maccs"]
  distance_methods: ["euclidean", "cosine"]

output:
  directory: "results/my_experiment"
  formats: ["json", "csv"]

compute:
  max_workers: 4
  sample_size: null  # Use all data
```

### Explicit Dataset Format (Still supported)
```yaml
name: "my_pipeline"
description: "Explicit dataset specification"

molecule:
  datasets:
    - name: "CHEMBL123"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL456"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
  featurizers: ["ecfp"]
  distance_methods: ["euclidean"]
```

## Environment Setup

The bash script automatically handles conda environment activation. Make sure you have:

1. **Conda environment named 'themap'** with THEMAP installed
2. **Datasets** in the expected directory structure
3. **Task list file** (for directory-based configs)

## Troubleshooting

### Common Issues

1. **"Conda environment 'themap' not found"**
   - Create the environment: `conda create -n themap python=3.10`
   - Install THEMAP in the environment

2. **"Configuration file not found"**
   - Check the path to your config file
   - Use absolute paths if needed

3. **"Dataset validation failed"**
   - Check if your datasets directory exists
   - Verify the task list file format
   - Use `--validate-only` to debug

4. **Python script fails but bash script works**
   - Make sure you're in the correct conda environment
   - Use the bash script which handles environment activation automatically

### Getting Help

```bash
# Show detailed help
./run_pipeline.sh --help
python run_pipeline.py --help

# List available example configurations
./run_pipeline.sh --list-examples

# Check what would be computed without running
./run_pipeline.sh --dry-run configs/my_config.yaml
```

## Script Differences

| Feature | Bash Script | Python Script |
|---------|-------------|---------------|
| Environment handling | Automatic conda activation | Manual environment required |
| Error handling | Comprehensive with colored output | Basic error reporting |
| Platform support | Linux/macOS | Cross-platform |
| Dependencies | Requires conda | Requires THEMAP installed |

Choose the bash script for automatic environment handling, or the Python script if you prefer to manage environments manually.
