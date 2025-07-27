# Working with Tasks

This tutorial covers THEMAP's unified task system that integrates molecular data, protein data, and metadata for comprehensive analysis.

## Understanding the Task System

The Task system in THEMAP provides a unified interface for working with multi-modal bioactivity prediction data:

- **Tasks**: Individual prediction problems (e.g., CHEMBL1023359)
- **Data modalities**: Molecules, proteins, metadata
- **Splits**: Train/test/validation divisions
- **Caching**: Efficient feature storage and retrieval

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
    cache_dir="cache/"
)

print(f"Loaded {len(tasks)} tasks")
```

### Task Configuration File

The `sample_tasks_list.json` file defines which tasks belong to which split:

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
│   ├── CHEMBL1023359.fasta       # Protein sequences
│   └── CHEMBL1023359_metadata.json
├── test/
│   └── ...
├── valid/
│   └── ...
└── sample_tasks_list.json
```

## Exploring Tasks

### Basic Task Information

```python
# Get task statistics
print(f"Total tasks: {len(tasks)}")
print(f"Train tasks: {tasks.get_num_fold_tasks('TRAIN')}")
print(f"Test tasks: {tasks.get_num_fold_tasks('TEST')}")
print(f"Valid tasks: {tasks.get_num_fold_tasks('VALID')}")

# Get task IDs
train_task_ids = tasks.get_task_ids(fold='TRAIN')
print(f"Training task IDs: {train_task_ids}")

# Access individual tasks
task = tasks.get_task("CHEMBL1023359")
print(f"Task {task.task_id} has {len(task.molecules)} molecules")
```

### Data Modality Access

```python
# Access different data types
for task_id in tasks.get_task_ids()[:3]:  # First 3 tasks
    task = tasks.get_task(task_id)

    print(f"\nTask: {task_id}")

    # Molecular data
    if hasattr(task, 'molecules') and task.molecules:
        print(f"  Molecules: {len(task.molecules)}")
        print(f"  Sample SMILES: {task.molecules[0].smiles}")

    # Protein data
    if hasattr(task, 'proteins') and task.proteins:
        print(f"  Proteins: {len(task.proteins)}")
        print(f"  Sample protein length: {len(task.proteins[0].sequence)}")

    # Metadata
    if hasattr(task, 'metadata') and task.metadata:
        print(f"  Metadata keys: {list(task.metadata.keys())}")
```

## Task-Based Distance Computation

### Unified Distance Calculation

```python
from themap.distance import TaskDistance

# Create task distance calculator
task_distance = TaskDistance(
    tasks=tasks,
    molecule_method="cosine",
    protein_method="euclidean"
)

# Compute all distance types
all_distances = task_distance.compute_all_distances(
    combination_strategy="weighted_average",
    molecule_weight=0.7,
    protein_weight=0.3
)

print("Distance types computed:")
print(f"  Molecule distances: {len(all_distances['molecule'])} tasks")
print(f"  Protein distances: {len(all_distances['protein'])} tasks")
print(f"  Combined distances: {len(all_distances['combined'])} tasks")
```

### Specific Distance Types

```python
# Compute only molecular distances
molecule_distances = task_distance.get_molecule_distances()

# Compute only protein distances
protein_distances = task_distance.get_protein_distances()

# Compute combined distances with custom weights
combined_distances = task_distance.get_combined_distances(
    molecule_weight=0.6,
    protein_weight=0.4
)
```

## Working with Folds

### Train-Test Analysis

```python
def analyze_train_test_distances(tasks, distance_type="molecule"):
    """Analyze distances between train and test tasks."""

    # Get task IDs by fold
    train_ids = set(tasks.get_task_ids(fold='TRAIN'))
    test_ids = set(tasks.get_task_ids(fold='TEST'))

    # Compute distances
    task_distance = TaskDistance(tasks=tasks)
    distances = task_distance.compute_all_distances()[distance_type]

    # Extract train-test distances
    train_test_distances = []

    for test_id in test_ids:
        if test_id in distances:
            for train_id in train_ids:
                if train_id in distances[test_id]:
                    train_test_distances.append({
                        'test_task': test_id,
                        'train_task': train_id,
                        'distance': distances[test_id][train_id]
                    })

    return train_test_distances

# Analyze train-test relationships
tt_distances = analyze_train_test_distances(tasks, "molecule")
print(f"Found {len(tt_distances)} train-test pairs")

# Find closest training tasks for each test task
import pandas as pd

df = pd.DataFrame(tt_distances)
for test_task in df['test_task'].unique():
    task_distances = df[df['test_task'] == test_task]
    closest = task_distances.nsmallest(3, 'distance')

    print(f"\nTest task {test_task} - closest training tasks:")
    for _, row in closest.iterrows():
        print(f"  {row['train_task']}: {row['distance']:.3f}")
```

### Cross-Validation Support

```python
def create_cv_folds(tasks, n_folds=5):
    """Create cross-validation folds from tasks."""

    all_task_ids = tasks.get_task_ids()
    random.shuffle(all_task_ids)  # Randomize

    fold_size = len(all_task_ids) // n_folds
    folds = []

    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(all_task_ids)

        test_tasks = all_task_ids[start_idx:end_idx]
        train_tasks = [tid for tid in all_task_ids if tid not in test_tasks]

        folds.append({
            'fold': i,
            'train': train_tasks,
            'test': test_tasks
        })

    return folds

# Create CV folds
cv_folds = create_cv_folds(tasks, n_folds=5)
print(f"Created {len(cv_folds)} CV folds")
```

## Caching and Performance

### Feature Caching

```python
# Tasks automatically cache computed features
tasks_with_cache = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=True,
    cache_dir="cache/",  # Features will be cached here
    force_reload=False   # Use cached features if available
)

# Check cache status
print(f"Cache directory: {tasks_with_cache.cache_dir}")
```

### Selective Loading

```python
# Load only specific data types for better performance
molecule_only_tasks = Tasks.from_directory(
    directory="datasets/",
    task_list_file="datasets/sample_tasks_list.json",
    load_molecules=True,
    load_proteins=False,  # Skip protein loading
    load_metadata=False   # Skip metadata loading
)

print("Loaded molecular data only for faster processing")
```

## Advanced Task Operations

### Task Filtering

```python
def filter_tasks_by_size(tasks, min_molecules=10, max_molecules=1000):
    """Filter tasks by number of molecules."""

    filtered_task_ids = []

    for task_id in tasks.get_task_ids():
        task = tasks.get_task(task_id)
        if hasattr(task, 'molecules') and task.molecules:
            n_molecules = len(task.molecules)
            if min_molecules <= n_molecules <= max_molecules:
                filtered_task_ids.append(task_id)

    return filtered_task_ids

# Filter tasks
good_tasks = filter_tasks_by_size(tasks, min_molecules=20, max_molecules=500)
print(f"Found {len(good_tasks)} tasks with appropriate size")
```

### Task Metadata Analysis

```python
def analyze_task_metadata(tasks):
    """Analyze metadata across all tasks."""

    metadata_summary = {}

    for task_id in tasks.get_task_ids():
        task = tasks.get_task(task_id)

        if hasattr(task, 'metadata') and task.metadata:
            for key, value in task.metadata.items():
                if key not in metadata_summary:
                    metadata_summary[key] = []
                metadata_summary[key].append(value)

    # Print summary
    for key, values in metadata_summary.items():
        unique_values = len(set(str(v) for v in values))
        print(f"Metadata '{key}': {len(values)} tasks, {unique_values} unique values")

    return metadata_summary

# Analyze metadata
metadata = analyze_task_metadata(tasks)
```

## Integration with External Tools

### Export for Analysis

```python
def export_task_summary(tasks, output_file="task_summary.csv"):
    """Export task information to CSV for external analysis."""

    import pandas as pd

    summary_data = []

    for task_id in tasks.get_task_ids():
        task = tasks.get_task(task_id)

        row = {'task_id': task_id}

        # Add molecule info
        if hasattr(task, 'molecules') and task.molecules:
            row['n_molecules'] = len(task.molecules)
        else:
            row['n_molecules'] = 0

        # Add protein info
        if hasattr(task, 'proteins') and task.proteins:
            row['n_proteins'] = len(task.proteins)
            row['avg_protein_length'] = sum(len(p.sequence) for p in task.proteins) / len(task.proteins)
        else:
            row['n_proteins'] = 0
            row['avg_protein_length'] = 0

        # Add metadata
        if hasattr(task, 'metadata') and task.metadata:
            for key, value in task.metadata.items():
                row[f'metadata_{key}'] = value

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Task summary exported to {output_file}")

    return df

# Export summary
task_df = export_task_summary(tasks)
```

## Best Practices

### Performance Tips

1. **Use caching**: Always specify a cache directory
2. **Load selectively**: Only load needed data types
3. **Filter early**: Remove unsuitable tasks before distance computation
4. **Batch processing**: Process tasks in groups for large datasets

### Memory Management

```python
# For large task sets, process in batches
def process_tasks_in_batches(tasks, batch_size=10):
    """Process tasks in smaller batches to manage memory."""

    task_ids = tasks.get_task_ids()

    for i in range(0, len(task_ids), batch_size):
        batch_ids = task_ids[i:i + batch_size]

        # Create subset tasks
        batch_tasks = Tasks()
        for task_id in batch_ids:
            batch_tasks.add_task(tasks.get_task(task_id))

        # Process batch
        yield batch_tasks

# Example usage
for batch_tasks in process_tasks_in_batches(tasks, batch_size=5):
    print(f"Processing batch with {len(batch_tasks)} tasks")
    # Perform distance computation on batch
```

## Troubleshooting

### Common Issues

1. **Missing files**: Ensure all task files exist in expected directories
2. **Cache conflicts**: Use `force_reload=True` to refresh cached features
3. **Memory errors**: Reduce batch size or load fewer data modalities
4. **Inconsistent data**: Validate that all tasks have required data types

### Validation

```python
def validate_tasks(tasks):
    """Validate task data consistency."""

    issues = []

    for task_id in tasks.get_task_ids():
        task = tasks.get_task(task_id)

        # Check for required data
        if not hasattr(task, 'molecules') or not task.molecules:
            issues.append(f"Task {task_id}: No molecular data")

        if not hasattr(task, 'proteins') or not task.proteins:
            issues.append(f"Task {task_id}: No protein data")

    if issues:
        print("Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All tasks validated successfully")

    return issues

# Validate loaded tasks
validation_issues = validate_tasks(tasks)
```

## Next Steps

- Explore [performance optimization](performance-optimization.md)
- Learn about task hardness estimation
- Try advanced distance combination strategies
- Integrate with your machine learning pipeline
