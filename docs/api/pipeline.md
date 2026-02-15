# Pipeline Module

The pipeline module provides a high-level API for running distance computation workflows.

## Entry Points

### quick_distance

::: themap.pipeline.orchestrator.quick_distance
    options:
      show_root_heading: true
      heading_level: 3

### run_pipeline

::: themap.pipeline.orchestrator.run_pipeline
    options:
      show_root_heading: true
      heading_level: 3

### Pipeline

::: themap.pipeline.orchestrator.Pipeline
    options:
      show_root_heading: true
      heading_level: 3

## Configuration Classes

### PipelineConfig

::: themap.config.PipelineConfig
    options:
      show_root_heading: true
      heading_level: 3

### DataConfig

::: themap.config.DataConfig
    options:
      show_root_heading: true
      heading_level: 3

### MoleculeDistanceConfig

::: themap.config.MoleculeDistanceConfig
    options:
      show_root_heading: true
      heading_level: 3

### ProteinDistanceConfig

::: themap.config.ProteinDistanceConfig
    options:
      show_root_heading: true
      heading_level: 3

### OutputConfig

::: themap.config.OutputConfig
    options:
      show_root_heading: true
      heading_level: 3

### ComputeConfig

::: themap.config.ComputeConfig
    options:
      show_root_heading: true
      heading_level: 3

### CombinationConfig

::: themap.config.CombinationConfig
    options:
      show_root_heading: true
      heading_level: 3

## Featurization Pipeline

### FeatureStore

::: themap.pipeline.featurization.FeatureStore
    options:
      show_root_heading: true
      heading_level: 3

### FeaturizationPipeline

::: themap.pipeline.featurization.FeaturizationPipeline
    options:
      show_root_heading: true
      heading_level: 3

## Configuration File Format

```yaml
data:
  directory: "datasets"

distances:
  molecule:
    enabled: true
    featurizer: "ecfp"
    method: "euclidean"
  protein:
    enabled: false
    featurizer: "esm2_t33_650M_UR50D"
    method: "euclidean"

output:
  directory: "output"
  format: "csv"

compute:
  n_jobs: 8
  device: "auto"
```

## Output Files

```
output/
├── molecule_distances.csv
├── features/
│   ├── ecfp_source.npz
│   └── ecfp_target.npz
└── pipeline_summary.json
```

See [Getting Started](../user-guide/getting-started.md) for usage examples and [CLI Reference](../user-guide/cli.md) for command-line usage.
