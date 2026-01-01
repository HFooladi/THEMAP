# Pipeline Module

The pipeline module provides a high-level API for running distance computation workflows. It supports both programmatic and configuration-based approaches.

## Overview

The pipeline system offers three levels of usage:

1. **`quick_distance`** - One-liner for simple computations
2. **`run_pipeline`** - Configuration file-based execution
3. **`Pipeline`** - Full programmatic control

## Quick Distance

The simplest way to compute distances:

```python
from themap import quick_distance

results = quick_distance(
    data_dir="datasets",
    output_dir="output",
    molecule_featurizer="ecfp",
    molecule_method="euclidean",
)
```

### Function Reference

::: themap.pipeline.orchestrator.quick_distance
    options:
      show_root_heading: true
      heading_level: 3

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | Required | Directory with train/test folders |
| `output_dir` | str | `"output"` | Output directory for results |
| `molecule_featurizer` | str | `"ecfp"` | Molecular fingerprint type |
| `molecule_method` | str | `"euclidean"` | Distance metric |
| `n_jobs` | int | `8` | Parallel workers |

### Returns

Dictionary with distance matrices:

```python
{
    "molecule": {
        "target_task_1": {
            "source_task_1": 0.75,
            "source_task_2": 1.23,
        },
        "target_task_2": {...}
    }
}
```

## Run Pipeline

Execute a pipeline from a YAML configuration file:

```python
from themap import run_pipeline

results = run_pipeline("config.yaml")
```

### Function Reference

::: themap.pipeline.orchestrator.run_pipeline
    options:
      show_root_heading: true
      heading_level: 3

### Configuration File Format

```yaml
# config.yaml
data:
  directory: "datasets"
  task_list: null  # Auto-discover tasks

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

## Pipeline Class

For full programmatic control:

### Pipeline

::: themap.pipeline.orchestrator.Pipeline
    options:
      show_root_heading: true
      heading_level: 3

### Usage Example

```python
from themap import Pipeline, PipelineConfig
from themap.config import (
    DataConfig,
    MoleculeDistanceConfig,
    OutputConfig,
    ComputeConfig,
)
from pathlib import Path

# Build configuration
config = PipelineConfig(
    data=DataConfig(
        directory=Path("datasets"),
        task_list=None,
    ),
    molecule=MoleculeDistanceConfig(
        enabled=True,
        featurizer="ecfp",
        method="euclidean",
    ),
    output=OutputConfig(
        directory=Path("output"),
        format="csv",
        save_features=True,
    ),
    compute=ComputeConfig(
        n_jobs=8,
        device="auto",
    ),
)

# Create and run pipeline
pipeline = Pipeline(config)
results = pipeline.run()

# Access results
print(f"Computed distances for {len(results['molecule'])} target tasks")
```

## Configuration Classes

### PipelineConfig

::: themap.config.PipelineConfig
    options:
      show_root_heading: true
      heading_level: 3

### DataConfig

```python
from themap.config import DataConfig
from pathlib import Path

data_config = DataConfig(
    directory=Path("datasets"),
    task_list=["CHEMBL123", "CHEMBL456"],  # Optional: specific tasks
    source_fold="train",
    target_fold="test",
)
```

### MoleculeDistanceConfig

```python
from themap.config import MoleculeDistanceConfig

mol_config = MoleculeDistanceConfig(
    enabled=True,
    featurizer="ecfp",      # Fingerprint type
    method="euclidean",     # Distance method
    cache_features=True,    # Cache computed features
)
```

### ProteinDistanceConfig

```python
from themap.config import ProteinDistanceConfig

prot_config = ProteinDistanceConfig(
    enabled=True,
    featurizer="esm2_t33_650M_UR50D",
    method="euclidean",
)
```

### OutputConfig

```python
from themap.config import OutputConfig
from pathlib import Path

output_config = OutputConfig(
    directory=Path("output"),
    format="csv",           # csv, json, or parquet
    save_features=True,     # Save computed features
    save_matrices=True,     # Save distance matrices
)
```

### ComputeConfig

```python
from themap.config import ComputeConfig

compute_config = ComputeConfig(
    n_jobs=8,               # Parallel workers
    device="auto",          # auto, cpu, or cuda
    batch_size=1000,        # Batch size for processing
)
```

## Configuration from YAML

### Loading Configuration

```python
from themap.config import PipelineConfig

# Load from file
config = PipelineConfig.from_yaml("config.yaml")

# Validate configuration
issues = config.validate()
if issues:
    for issue in issues:
        print(f"Warning: {issue}")
```

### Saving Configuration

```python
from themap.config import PipelineConfig

config = PipelineConfig(...)

# Save to file
config.to_yaml("my_config.yaml")
```

## Output Files

The pipeline generates these output files:

```
output/
├── molecule_distances.csv       # Distance matrix
├── molecule_distances.json      # JSON format (if enabled)
├── features/                    # Cached features (if enabled)
│   ├── ecfp_source.npz
│   └── ecfp_target.npz
└── pipeline_summary.json        # Execution summary
```

### Distance Matrix Format (CSV)

```csv
,CHEMBL111111,CHEMBL222222,CHEMBL333333
CHEMBL123456,0.75,1.23,0.89
CHEMBL789012,1.45,0.67,1.12
```

### Summary File

```json
{
    "pipeline_name": "molecule_distance",
    "execution_time": "2.34s",
    "datasets_processed": {
        "source": 10,
        "target": 3
    },
    "distance_computations": 30,
    "config": {...}
}
```

## Advanced Usage

### Custom Pipeline Steps

```python
from themap import Pipeline, PipelineConfig

class CustomPipeline(Pipeline):
    def run(self):
        # Pre-processing
        self.validate_data()
        
        # Run standard pipeline
        results = super().run()
        
        # Post-processing
        results = self.analyze_results(results)
        
        return results
    
    def validate_data(self):
        """Custom validation logic."""
        pass
    
    def analyze_results(self, results):
        """Custom analysis."""
        return results
```

### Combining Multiple Runs

```python
from themap import quick_distance

# Run with different methods
methods = ["euclidean", "cosine"]
all_results = {}

for method in methods:
    results = quick_distance(
        data_dir="datasets",
        molecule_method=method,
        output_dir=f"output_{method}",
    )
    all_results[method] = results

# Compare methods
for method, results in all_results.items():
    print(f"\n{method.upper()} distances:")
    # Analyze results...
```

### Incremental Processing

```python
from themap import Pipeline, PipelineConfig

# Process in batches
config = PipelineConfig(...)
pipeline = Pipeline(config)

# Get task lists
source_tasks = pipeline.get_source_tasks()
target_tasks = pipeline.get_target_tasks()

# Process incrementally
batch_size = 10
all_results = {}

for i in range(0, len(target_tasks), batch_size):
    batch_targets = target_tasks[i:i+batch_size]
    
    batch_results = pipeline.run_for_targets(batch_targets)
    all_results.update(batch_results)
    
    print(f"Processed {i+batch_size}/{len(target_tasks)} targets")
```

## Error Handling

```python
from themap import run_pipeline
from themap.pipeline import PipelineError

try:
    results = run_pipeline("config.yaml")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")
except PipelineError as e:
    print(f"Pipeline execution failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Use caching**: Set `save_features: true` for repeated runs
2. **Choose fast methods**: Use `euclidean` for exploration, `otdd` for final analysis
3. **Parallel processing**: Increase `n_jobs` for multi-core systems
4. **GPU acceleration**: Use `device: cuda` for protein featurizers

```yaml
# Optimized configuration
compute:
  n_jobs: 16
  device: "cuda"
  batch_size: 2000

output:
  save_features: true  # Cache for reuse
```

