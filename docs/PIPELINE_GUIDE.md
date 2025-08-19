# THEMAP Pipeline System

The THEMAP pipeline system provides a configuration-driven approach to running large-scale distance computation benchmarks across molecular and protein datasets. This system allows you to define complex benchmarking workflows through simple YAML or JSON configuration files.

## Quick Start

### 1. Install Dependencies

```bash
# Activate conda environment
eval "$(conda shell.bash hook)" && conda activate themap

# Ensure pipeline dependencies are available
pip install pyyaml pandas numpy
```

### 2. Run a Simple Example

```bash
# List available example configurations
python -m themap.pipeline --list-examples

# Run a quick test
python -m themap.pipeline configs/examples/quick_test.yaml

# Run with custom settings
python -m themap.pipeline configs/examples/simple_molecule_benchmark.yaml --log-level DEBUG
```

### 3. Create Your Own Configuration

```yaml
# my_benchmark.yaml
name: "my_custom_benchmark"
description: "Custom molecular distance benchmark"

molecule:
  datasets:
    - name: "CHEMBL1963831"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL2219236"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
  featurizers: ["ecfp", "maccs"]
  distance_methods: ["euclidean", "cosine"]

output:
  directory: "my_results"
  formats: ["json", "csv"]

compute:
  sample_size: 500  # For faster testing
```

## Configuration Schema

### Pipeline Structure

```yaml
name: "pipeline_name"           # Required: Pipeline identifier
description: "Description"      # Optional: Pipeline description

molecule:                       # Optional: Molecular data configuration
  datasets: [...]              # List of molecular datasets
  featurizers: [...]           # Molecular featurization methods
  distance_methods: [...]      # Distance computation methods

protein:                        # Optional: Protein data configuration
  datasets: [...]              # List of protein datasets
  featurizers: [...]           # Protein featurization methods
  distance_methods: [...]      # Distance computation methods

metadata:                       # Optional: Metadata configuration
  datasets: [...]              # List of metadata datasets
  features: [...]              # Custom metadata features
  distance_methods: [...]      # Distance computation methods

task_distance:                  # Optional: Combined task distance
  combination_strategy: "..."   # How to combine modalities
  weights: {...}               # Weights for each modality

output:                         # Output configuration
  directory: "results"         # Output directory
  formats: ["json", "csv"]     # Output formats
  save_intermediate: true      # Save intermediate results
  save_matrices: false        # Save full distance matrices

compute:                        # Computation configuration
  max_workers: 4               # Parallel workers
  cache_features: true         # Cache computed features
  gpu_if_available: true       # Use GPU if available
  sample_size: null            # Sample size (null = full dataset)
  seed: 42                     # Random seed
```

### Dataset Configuration

Each dataset is specified with:

```yaml
- name: "CHEMBL123456"          # Dataset identifier (matches filename)
  source_fold: "TRAIN"          # Source data fold: TRAIN, TEST, VALIDATION
  target_folds: ["TEST"]        # Target folds for comparison
  path: null                    # Optional: Custom file path override
```

### Available Options

**Molecular Featurizers:**
- `ecfp`: Extended Connectivity Fingerprints
- `maccs`: MACCS Keys
- `desc2D`: 2D molecular descriptors
- `mordred`: Mordred descriptors
- Neural embeddings: `ChemBERTa-77M-MLM`, `MolT5`, etc.

**Protein Featurizers:**
- `esm`: ESM protein language model embeddings
- `esm2`: ESM-2 embeddings

**Distance Methods:**
- `euclidean`: Euclidean distance
- `cosine`: Cosine distance
- `otdd`: Optimal Transport Dataset Distance (molecules only)

**Output Formats:**
- `json`: JSON format
- `csv`: CSV format
- `parquet`: Parquet format
- `pickle`: Python pickle format

## Command Line Interface

### Basic Usage

```bash
# Run pipeline with configuration file
python -m themap.pipeline config.yaml

# Specify custom data path
python -m themap.pipeline config.yaml --data-path /path/to/datasets

# Override output directory
python -m themap.pipeline config.yaml --output-dir custom_results

# Set logging level
python -m themap.pipeline config.yaml --log-level DEBUG
```

### Validation and Testing

```bash
# Validate configuration without running
python -m themap.pipeline config.yaml --validate-only

# Dry run to see what would be computed
python -m themap.pipeline config.yaml --dry-run

# Override sample size for testing
python -m themap.pipeline config.yaml --sample-size 100
```

### Advanced Options

```bash
# Override max workers
python -m themap.pipeline config.yaml --max-workers 8

# List available examples
python -m themap.pipeline --list-examples
```

## Example Configurations

### 1. Simple Molecular Benchmark

```yaml
name: "simple_molecule_benchmark"
description: "Basic molecular distance computation"

molecule:
  datasets:
    - name: "CHEMBL1963831"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL2219236"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
  featurizers: ["ecfp"]
  distance_methods: ["euclidean"]

output:
  directory: "results/simple"
  formats: ["json", "csv"]

compute:
  sample_size: 1000
```

### 2. Comprehensive Multi-Modal

```yaml
name: "comprehensive_benchmark"
description: "Full multi-modal benchmarking"

molecule:
  datasets:
    - name: "CHEMBL1963831"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL2219236"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
  featurizers: ["ecfp", "maccs", "desc2D"]
  distance_methods: ["euclidean", "cosine"]

protein:
  datasets:
    - name: "CHEMBL1963831"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL2219236"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
  featurizers: ["esm"]
  distance_methods: ["euclidean", "cosine"]

task_distance:
  combination_strategy: "weighted_average"
  weights:
    molecule: 0.7
    protein: 0.3

output:
  directory: "results/comprehensive"
  formats: ["json", "csv", "parquet"]
  save_intermediate: true
  save_matrices: true
```

### 3. High-Throughput Screening

```yaml
name: "high_throughput_benchmark"
description: "Large-scale benchmarking with multiple methods"

molecule:
  datasets:
    - name: "CHEMBL1006005"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL1119333"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    - name: "CHEMBL1243967"
      source_fold: "TRAIN"
      target_folds: ["TEST"]
    # ... more datasets
  featurizers: ["ecfp", "maccs", "desc2D", "mordred"]
  distance_methods: ["euclidean", "cosine", "otdd"]

output:
  directory: "results/high_throughput"
  formats: ["json", "parquet"]
  save_intermediate: false
  save_matrices: true

compute:
  max_workers: 8
  gpu_if_available: true
```

## Output Structure

### Results Directory

```
results/
├── pipeline_results_20240117_143022.json    # Main results file
├── pipeline_results_20240117_143022.csv     # CSV format results
├── pipeline_summary_20240117_143022.json    # Execution summary
├── distance_matrix_20240117_143022.csv      # Distance matrices (if enabled)
└── intermediate_*/                           # Intermediate results (if enabled)
    ├── molecule_distance_*.json
    ├── protein_distance_*.json
    └── task_distance_*.json
```

### Results Format

**Main Results JSON:**
```json
{
  "config": {...},                    // Pipeline configuration
  "datasets_info": [...],             // Information about loaded datasets
  "distance_results": [               // Distance computation results
    {
      "source_dataset": "CHEMBL123456",
      "target_dataset": "CHEMBL789012",
      "modality": "molecule",
      "featurizer": "ecfp",
      "method": "euclidean",
      "distance": 0.753,
      "computation_time": 1.23
    }
  ],
  "runtime_seconds": 45.67,
  "errors": []                        // Any errors encountered
}
```

**Summary Report:**
```json
{
  "pipeline_name": "my_benchmark",
  "execution_timestamp": "20240117_143022",
  "total_runtime_seconds": 45.67,
  "datasets_processed": {
    "total_datasets": 5,
    "molecule_datasets": 3,
    "protein_datasets": 2
  },
  "distance_computations": {
    "total_computations": 15,
    "by_modality": {"molecule": 10, "protein": 5},
    "by_method": {"euclidean": 8, "cosine": 7},
    "average_computation_time": 2.34
  },
  "files_generated": [...],
  "config": {...},
  "errors": []
}
```

## Programming Interface

### Python API

```python
from themap.pipeline import PipelineConfig, PipelineRunner

# Load configuration
config = PipelineConfig.from_file("my_config.yaml")

# Create and run pipeline
runner = PipelineRunner(config)
results = runner.run(base_data_path="datasets")

# Access results
print(f"Computed {len(results['distance_results'])} distances")
print(f"Runtime: {results['runtime_seconds']:.2f} seconds")
```

### Custom Configuration

```python
from themap.pipeline.config import (
    PipelineConfig, MoleculeConfig, DatasetConfig
)

# Create configuration programmatically
molecule_config = MoleculeConfig(
    datasets=[
        DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"]),
        DatasetConfig("CHEMBL789012", "TRAIN", ["TEST"])
    ],
    featurizers=["ecfp", "maccs"],
    distance_methods=["euclidean", "cosine"]
)

config = PipelineConfig(
    name="programmatic_pipeline",
    molecule=molecule_config
)

# Save configuration
config.save("my_config.yaml")
```

## Best Practices

### Performance Optimization

1. **Use Sampling for Development:**
   ```yaml
   compute:
     sample_size: 100  # Use small samples during development
   ```

2. **Enable Feature Caching:**
   ```yaml
   compute:
     cache_features: true  # Cache expensive feature computations
   ```

3. **Parallel Processing:**
   ```yaml
   compute:
     max_workers: 8  # Use multiple workers for distance computation
   ```

4. **GPU Utilization:**
   ```yaml
   compute:
     gpu_if_available: true  # Use GPU for protein embeddings
   ```

### Memory Management

1. **Disable Matrix Saving for Large Benchmarks:**
   ```yaml
   output:
     save_matrices: false  # Avoid large matrix files
   ```

2. **Selective Intermediate Saving:**
   ```yaml
   output:
     save_intermediate: false  # Disable for production runs
   ```

### Error Handling

- Pipeline continues execution even if some distances fail to compute
- All errors are logged and included in final results
- Use `--validate-only` to check configuration before running
- Use `--dry-run` to preview computations

### Reproducibility

```yaml
compute:
  seed: 42              # Set random seed
  sample_size: 1000     # Fix sample size
```

## Troubleshooting

### Common Issues

1. **Dataset Not Found:**
   ```
   FileNotFoundError: Molecule dataset not found: datasets/train/CHEMBL123456.jsonl.gz
   ```
   - Check dataset files exist in correct directories
   - Verify dataset names match configuration

2. **Invalid Featurizer:**
   ```
   ValueError: Unknown featurizer: invalid_featurizer
   ```
   - Check available featurizers in documentation
   - Ensure required dependencies are installed

3. **Memory Issues:**
   - Reduce `sample_size`
   - Disable `save_matrices`
   - Reduce `max_workers`

4. **GPU Issues:**
   ```yaml
   compute:
     gpu_if_available: false  # Disable GPU usage
   ```

### Debug Mode

```bash
# Enable detailed logging
python -m themap.pipeline config.yaml --log-level DEBUG

# Validate configuration
python -m themap.pipeline config.yaml --validate-only

# Preview computations
python -m themap.pipeline config.yaml --dry-run
```

## Contributing

The pipeline system is designed to be extensible:

1. **New Featurizers:** Add to `featurizer_utils.py`
2. **New Distance Methods:** Extend distance classes
3. **New Output Formats:** Extend `OutputManager`
4. **New Combination Strategies:** Extend `TaskDistance`

See the main project documentation for detailed contribution guidelines.
