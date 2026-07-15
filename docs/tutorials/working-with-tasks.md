# Working with Tasks

This tutorial covers THEMAP's unified task system that integrates molecular data,
protein data, and other metadata for comprehensive analysis.

!!! note
    The code on this page is written against the current THEMAP API. The examples
    are illustrative — running them requires a real dataset directory laid out as
    shown below.

## Understanding the Task System

The Task system in THEMAP provides a unified interface for working with
multi-modal bioactivity prediction data:

- **Tasks**: individual prediction problems (e.g. `CHEMBL1023359`)
- **Data modalities**: molecules (`molecule_dataset`) and metadata such as protein
  embeddings (`metadata_datasets`)
- **Folds**: train / validation / test divisions, addressed with the `DataFold` enum
- **Caching**: efficient feature storage and retrieval

## Loading Tasks

### Basic Task Loading

```python
from themap.data.tasks import Tasks

# Load tasks from a directory structure
tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    load_metadata=True,
    cache_dir="cache/",
)

print(f"Loaded {len(tasks)} tasks")
```

### Task Configuration File

The `sample_tasks_list.json` file defines which tasks belong to which fold:

```json
{
  "train": ["CHEMBL1023359", "CHEMBL1613776", "CHEMBL4078627"],
  "test": ["CHEMBL2219358", "CHEMBL1963831"],
  "valid": ["CHEMBL2219236"]
}
```

### Directory Structure

```
datasets/
├── train/
│   ├── CHEMBL1023359.jsonl.gz    # Molecular data
│   ├── CHEMBL1023359.fasta       # Protein sequence
│   └── ...
├── test/
│   └── ...
├── valid/
│   └── ...
└── sample_tasks_list.json
```

## Exploring Tasks

Fold-aware methods take a `DataFold` value (`TRAIN`, `VALIDATION`, or `TEST`), not
a string. There is no single "all folds" accessor, so concatenate the three folds
when you need every task ID.

```python
from themap.data.enums import DataFold

# Per-fold counts
print(f"Total tasks: {len(tasks)}")
print(f"Train tasks: {tasks.get_num_fold_tasks(DataFold.TRAIN)}")
print(f"Test tasks:  {tasks.get_num_fold_tasks(DataFold.TEST)}")
print(f"Valid tasks: {tasks.get_num_fold_tasks(DataFold.VALIDATION)}")

# Task IDs for a single fold
train_task_ids = tasks.get_task_ids(DataFold.TRAIN)
print(f"Training task IDs: {train_task_ids}")


def all_task_ids(tasks):
    """Return task IDs across every fold."""
    return (
        tasks.get_task_ids(DataFold.TRAIN)
        + tasks.get_task_ids(DataFold.VALIDATION)
        + tasks.get_task_ids(DataFold.TEST)
    )


# Access an individual task
task = tasks.get_task_by_id("CHEMBL1023359")
print(f"Task {task.task_id} has {len(task.molecule_dataset)} molecules")
```

### Data Modality Access

A `Task` exposes its molecules through `molecule_dataset` and any other modality
(such as protein embeddings) through the `metadata_datasets` dictionary, where
protein data lives under the `"protein"` key.

```python
for task_id in all_task_ids(tasks)[:3]:  # first 3 tasks
    task = tasks.get_task_by_id(task_id)

    print(f"\nTask: {task_id}")

    # Molecular data
    if task.molecule_dataset is not None:
        print(f"  Molecules: {len(task.molecule_dataset)}")
        print(f"  Sample SMILES: {task.molecule_dataset.smiles[0]}")

    # Metadata modalities (e.g. protein embeddings)
    if task.metadata_datasets:
        print(f"  Metadata modalities: {list(task.metadata_datasets.keys())}")
        if "protein" in task.metadata_datasets:
            print("  Has protein metadata")
```

## Task-Based Distance Computation

### Unified Distance Calculation

`TaskDistanceCalculator` orchestrates molecule, protein, and combined distances.
`compute_all_distances` returns a dictionary keyed by `"molecules"`, `"protein"`,
and `"combined"`; each value is a nested `{target_id: {source_id: distance}}` map.

```python
from themap.distance import TaskDistanceCalculator

# Create the calculator (dataset_method drives molecule distances,
# metadata_method drives protein/metadata distances)
task_distance = TaskDistanceCalculator(
    tasks=tasks,
    dataset_method="cosine",
    metadata_method="euclidean",
)

# Compute every distance type at once
all_distances = task_distance.compute_all_distances(
    molecule_featurizer="ecfp",
    combination="weighted_average",
    weights={"molecules": 0.7, "protein": 0.3},
)

print("Distance types computed:")
print(f"  Molecule distances: {len(all_distances['molecules'])} target tasks")
print(f"  Protein distances:  {len(all_distances['protein'])} target tasks")
print(f"  Combined distances: {len(all_distances['combined'])} target tasks")
```

### Specific Distance Types

```python
# Molecule distances only
molecule_distances = task_distance.compute_molecule_distance(
    molecule_featurizer="ecfp",
)

# Protein distances only
protein_distances = task_distance.compute_protein_distance(
    protein_featurizer="esm2_t33_650M_UR50D",
)

# Combined distances with custom weights
combined_distances = task_distance.compute_combined_distance(
    molecule_featurizer="ecfp",
    weights={"molecules": 0.6, "protein": 0.4},
    combination="weighted_average",
)
```

## Working with Folds

### Train–Test Analysis

Each distance matrix is indexed as `distances[target_id][source_id]`. With the
default `source_fold=DataFold.TRAIN`, train tasks are the sources (inner keys) and
the evaluated tasks are the targets (outer keys).

```python
import pandas as pd
from themap.data.enums import DataFold


def analyze_train_test_distances(tasks, distance_type="molecules"):
    """Collect distances between train (source) and test (target) tasks."""
    train_ids = set(tasks.get_task_ids(DataFold.TRAIN))
    test_ids = set(tasks.get_task_ids(DataFold.TEST))

    task_distance = TaskDistanceCalculator(tasks=tasks)
    distances = task_distance.compute_all_distances()[distance_type]

    pairs = []
    for test_id in test_ids:
        if test_id not in distances:
            continue
        for train_id in train_ids:
            if train_id in distances[test_id]:
                pairs.append({
                    "test_task": test_id,
                    "train_task": train_id,
                    "distance": distances[test_id][train_id],
                })
    return pairs


tt_distances = analyze_train_test_distances(tasks, "molecules")
print(f"Found {len(tt_distances)} train-test pairs")

# Closest training tasks for each test task
df = pd.DataFrame(tt_distances)
for test_task in df["test_task"].unique():
    closest = df[df["test_task"] == test_task].nsmallest(3, "distance")
    print(f"\nTest task {test_task} - closest training tasks:")
    for _, row in closest.iterrows():
        print(f"  {row['train_task']}: {row['distance']:.3f}")
```

### Cross-Validation Support

```python
import random
from themap.data.enums import DataFold


def create_cv_folds(tasks, n_folds=5):
    """Create cross-validation folds from all task IDs."""
    task_ids = (
        tasks.get_task_ids(DataFold.TRAIN)
        + tasks.get_task_ids(DataFold.VALIDATION)
        + tasks.get_task_ids(DataFold.TEST)
    )
    random.shuffle(task_ids)

    fold_size = len(task_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(task_ids)
        test_tasks = task_ids[start:end]
        train_tasks = [tid for tid in task_ids if tid not in test_tasks]
        folds.append({"fold": i, "train": train_tasks, "test": test_tasks})
    return folds


cv_folds = create_cv_folds(tasks, n_folds=5)
print(f"Created {len(cv_folds)} CV folds")
```

## Caching and Performance

### Feature Caching

Pass a `cache_dir` and THEMAP caches computed features there, reusing them on
subsequent loads.

```python
tasks_with_cache = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    cache_dir="cache/",   # features are cached here
)

print(f"Cache directory: {tasks_with_cache.cache_dir}")
```

### Selective Loading

```python
# Load only what you need for faster processing
molecule_only_tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=False,   # skip protein loading
    load_metadata=False,   # skip other metadata
)

print("Loaded molecular data only for faster processing")
```

## Advanced Task Operations

### Task Filtering

```python
def filter_tasks_by_size(tasks, min_molecules=10, max_molecules=1000):
    """Filter tasks by number of molecules."""
    filtered = []
    for task_id in all_task_ids(tasks):
        task = tasks.get_task_by_id(task_id)
        if task.molecule_dataset is None:
            continue
        n_molecules = len(task.molecule_dataset)
        if min_molecules <= n_molecules <= max_molecules:
            filtered.append(task_id)
    return filtered


good_tasks = filter_tasks_by_size(tasks, min_molecules=20, max_molecules=500)
print(f"Found {len(good_tasks)} tasks with an appropriate size")
```

### Inspecting Modalities

```python
def summarize_modalities(tasks):
    """Count how many tasks carry each metadata modality."""
    counts = {}
    for task_id in all_task_ids(tasks):
        task = tasks.get_task_by_id(task_id)
        for modality in (task.metadata_datasets or {}):
            counts[modality] = counts.get(modality, 0) + 1
    for modality, n in counts.items():
        print(f"Modality '{modality}': present in {n} tasks")
    return counts


summarize_modalities(tasks)
```

## Integration with External Tools

### Export for Analysis

```python
import pandas as pd


def export_task_summary(tasks, output_file="task_summary.csv"):
    """Export per-task information to CSV for external analysis."""
    rows = []
    for task_id in all_task_ids(tasks):
        task = tasks.get_task_by_id(task_id)
        rows.append({
            "task_id": task_id,
            "n_molecules": len(task.molecule_dataset) if task.molecule_dataset else 0,
            "modalities": ",".join((task.metadata_datasets or {}).keys()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Task summary exported to {output_file}")
    return df


task_df = export_task_summary(tasks)
```

## Best Practices

### Performance Tips

1. **Use caching**: always specify a `cache_dir`.
2. **Load selectively**: only load the modalities you need.
3. **Filter early**: remove unsuitable tasks before distance computation.
4. **Batch processing**: process tasks in groups for large datasets.

### Memory Management

For large task sets, build smaller `Tasks` collections with the constructor
(there is no incremental `add_task`; pass fold lists instead).

```python
from themap.data.enums import DataFold


def process_train_tasks_in_batches(tasks, batch_size=10):
    """Yield smaller Tasks collections over the train fold."""
    train_ids = tasks.get_task_ids(DataFold.TRAIN)
    for i in range(0, len(train_ids), batch_size):
        batch_ids = train_ids[i:i + batch_size]
        batch_tasks = Tasks(
            train_tasks=[tasks.get_task_by_id(tid) for tid in batch_ids],
        )
        yield batch_tasks


for batch_tasks in process_train_tasks_in_batches(tasks, batch_size=5):
    print(f"Processing batch with {len(batch_tasks)} tasks")
    # Perform distance computation on the batch
```

## Next Steps

- Explore [performance optimization](performance-optimization.md)
- Learn about task hardness estimation
- Try advanced distance combination strategies
- Integrate with your machine learning pipeline
