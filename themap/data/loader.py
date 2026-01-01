"""
Data loader for THEMAP pipeline.

This module provides utilities for loading datasets from directory structures,
auto-discovering files, and converting between formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """Load datasets from directory structure.

    Supports the following directory structure:
    ```
    data_dir/
    ├── train/           # Training tasks (source)
    │   ├── TASK1.csv
    │   ├── TASK2.jsonl.gz
    │   └── ...
    ├── test/            # Test tasks (target)
    │   ├── TASK3.csv
    │   └── ...
    ├── valid/           # Optional validation tasks
    │   └── ...
    ├── proteins/        # Optional protein FASTA files
    │   ├── TASK1.fasta
    │   └── ...
    └── tasks.json       # Optional task list
    ```

    If tasks.json is not provided, all CSV/JSONL.GZ files are auto-discovered.

    Attributes:
        data_dir: Root directory containing train/test/valid folders.
        task_list: Optional task list loaded from tasks.json.

    Examples:
        >>> loader = DatasetLoader(Path("datasets/TDC"))
        >>> train_datasets = loader.load_datasets("train")
        >>> test_datasets = loader.load_datasets("test")
        >>> # Get task IDs
        >>> train_ids = list(train_datasets.keys())
    """

    SUPPORTED_EXTENSIONS = {".jsonl.gz", ".csv"}
    FOLDS = {"train", "test", "valid"}

    def __init__(
        self,
        data_dir: Union[str, Path],
        task_list_file: Optional[str] = None,
    ):
        """Initialize the dataset loader.

        Args:
            data_dir: Root directory containing train/test/valid folders.
            task_list_file: Optional name of task list JSON file in data_dir.
                           If None, all files are auto-discovered.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.task_list: Optional[Dict[str, List[str]]] = None
        if task_list_file:
            task_list_path = self.data_dir / task_list_file
            if task_list_path.exists():
                self.task_list = self._load_task_list(task_list_path)
                logger.info(f"Loaded task list from {task_list_path}")
            else:
                logger.warning(f"Task list file not found: {task_list_path}. Auto-discovering all files.")

    def _load_task_list(self, path: Path) -> Dict[str, List[str]]:
        """Load task list from JSON file.

        Expected format:
        {
            "train": ["TASK1", "TASK2"],
            "test": ["TASK3", "TASK4"]
        }

        Args:
            path: Path to the JSON file.

        Returns:
            Dictionary mapping fold names to task ID lists.
        """
        with open(path, "r") as f:
            data = json.load(f)

        # Validate structure
        result = {}
        for fold in self.FOLDS:
            if fold in data:
                if not isinstance(data[fold], list):
                    raise ValueError(f"Task list '{fold}' must be a list")
                result[fold] = data[fold]

        return result

    def _discover_files(self, fold_dir: Path) -> List[Path]:
        """Discover all supported dataset files in a directory.

        Args:
            fold_dir: Directory to search.

        Returns:
            List of paths to dataset files.
        """
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            if ext == ".jsonl.gz":
                # Only include files that look like molecule datasets (CHEMBL IDs)
                for f in fold_dir.glob("*.jsonl.gz"):
                    # Skip files that contain "protein" in name
                    if "protein" not in f.name.lower():
                        files.append(f)
            else:
                for f in fold_dir.glob(f"*{ext}"):
                    # Skip files that contain "protein" in name
                    if "protein" not in f.name.lower():
                        files.append(f)

        return sorted(files)

    def _get_task_id_from_path(self, path: Path) -> str:
        """Extract task ID from file path.

        Args:
            path: Path to the dataset file.

        Returns:
            Task ID string.
        """
        name = path.name
        # Remove extensions
        if name.endswith(".jsonl.gz"):
            return name[: -len(".jsonl.gz")]
        elif name.endswith(".csv"):
            return name[: -len(".csv")]
        return name

    def _find_file_for_task(self, fold_dir: Path, task_id: str) -> Optional[Path]:
        """Find the dataset file for a given task ID.

        Args:
            fold_dir: Directory to search.
            task_id: Task ID to find.

        Returns:
            Path to the dataset file, or None if not found.
        """
        # Check JSONL.GZ first (preferred format)
        jsonl_path = fold_dir / f"{task_id}.jsonl.gz"
        if jsonl_path.exists():
            return jsonl_path

        # Check CSV
        csv_path = fold_dir / f"{task_id}.csv"
        if csv_path.exists():
            return csv_path

        return None

    def get_fold_dir(self, fold: str) -> Path:
        """Get the directory for a specific fold.

        Args:
            fold: Fold name (train, test, or valid).

        Returns:
            Path to the fold directory.

        Raises:
            ValueError: If fold name is invalid.
        """
        if fold not in self.FOLDS:
            raise ValueError(f"Invalid fold '{fold}'. Must be one of {self.FOLDS}")
        return self.data_dir / fold

    def get_task_ids(self, fold: str) -> List[str]:
        """Get list of task IDs for a fold.

        If task_list is provided, uses that. Otherwise auto-discovers files.

        Args:
            fold: Fold name (train, test, or valid).

        Returns:
            List of task IDs.
        """
        if self.task_list and fold in self.task_list:
            return self.task_list[fold]

        # Auto-discover
        fold_dir = self.get_fold_dir(fold)
        if not fold_dir.exists():
            logger.warning(f"Fold directory not found: {fold_dir}")
            return []

        files = self._discover_files(fold_dir)
        return [self._get_task_id_from_path(f) for f in files]

    def load_dataset(
        self,
        fold: str,
        task_id: str,
        convert_csv: bool = True,
    ) -> "MoleculeDataset":
        """Load a single dataset.

        Args:
            fold: Fold name (train, test, or valid).
            task_id: Task ID to load.
            convert_csv: If True, convert CSV to JSONL.GZ format automatically.

        Returns:
            MoleculeDataset instance.

        Raises:
            FileNotFoundError: If dataset file not found.
        """
        from .molecule_dataset import MoleculeDataset

        fold_dir = self.get_fold_dir(fold)
        file_path = self._find_file_for_task(fold_dir, task_id)

        if file_path is None:
            raise FileNotFoundError(f"Dataset file not found for task '{task_id}' in {fold_dir}")

        # Check if we need to convert CSV
        if file_path.suffix == ".csv":
            if convert_csv:
                logger.info(f"Converting CSV to JSONL.GZ: {file_path}")
                jsonl_path = self._convert_csv_to_jsonl(file_path)
                return MoleculeDataset.load_from_file(jsonl_path)
            else:
                # Load directly from CSV
                return self._load_from_csv(file_path, task_id)

        return MoleculeDataset.load_from_file(file_path)

    def load_datasets(
        self,
        fold: str,
        task_ids: Optional[List[str]] = None,
        convert_csv: bool = True,
    ) -> Dict[str, "MoleculeDataset"]:
        """Load all datasets for a fold.

        Args:
            fold: Fold name (train, test, or valid).
            task_ids: Optional list of specific task IDs to load.
                     If None, loads all tasks in the fold.
            convert_csv: If True, convert CSV to JSONL.GZ format automatically.

        Returns:
            Dictionary mapping task IDs to MoleculeDataset instances.
        """
        if task_ids is None:
            task_ids = self.get_task_ids(fold)

        if not task_ids:
            logger.warning(f"No tasks found for fold '{fold}'")
            return {}

        logger.info(f"Loading {len(task_ids)} datasets for fold '{fold}'")

        datasets = {}
        for task_id in task_ids:
            try:
                datasets[task_id] = self.load_dataset(fold, task_id, convert_csv)
            except Exception as e:
                logger.error(f"Failed to load dataset {task_id}: {e}")
                continue

        logger.info(f"Successfully loaded {len(datasets)}/{len(task_ids)} datasets")
        return datasets

    def load_all_folds(
        self,
        convert_csv: bool = True,
    ) -> Dict[str, Dict[str, "MoleculeDataset"]]:
        """Load datasets from all available folds.

        Returns:
            Dictionary mapping fold names to dictionaries of datasets.
        """
        results = {}
        for fold in self.FOLDS:
            fold_dir = self.get_fold_dir(fold)
            if fold_dir.exists():
                results[fold] = self.load_datasets(fold, convert_csv=convert_csv)

        return results

    def _load_from_csv(self, path: Path, task_id: str) -> "MoleculeDataset":
        """Load dataset directly from CSV without conversion.

        Args:
            path: Path to the CSV file.
            task_id: Task ID for the dataset.

        Returns:
            MoleculeDataset instance.
        """
        from .converter import CSVConverter
        from .molecule_dataset import MoleculeDataset

        converter = CSVConverter()
        data = converter.read_csv(path)

        return MoleculeDataset(
            task_id=task_id,
            smiles_list=data["smiles"],
            labels=np.array(data["labels"], dtype=np.int32),
            numeric_labels=np.array(data["numeric_labels"], dtype=np.float32)
            if data.get("numeric_labels")
            else None,
        )

    def _convert_csv_to_jsonl(self, csv_path: Path) -> Path:
        """Convert CSV file to JSONL.GZ format.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            Path to the converted JSONL.GZ file.
        """
        from .converter import CSVConverter

        converter = CSVConverter()
        task_id = self._get_task_id_from_path(csv_path)
        output_path = csv_path.parent / f"{task_id}.jsonl.gz"

        converter.convert(csv_path, output_path, task_id)
        return output_path

    def get_protein_file(self, task_id: str) -> Optional[Path]:
        """Get path to protein FASTA file for a task.

        Args:
            task_id: Task ID to find protein for.

        Returns:
            Path to FASTA file, or None if not found.
        """
        proteins_dir = self.data_dir / "proteins"
        if not proteins_dir.exists():
            return None

        fasta_path = proteins_dir / f"{task_id}.fasta"
        if fasta_path.exists():
            return fasta_path

        return None

    def load_protein_sequences(self) -> Dict[str, str]:
        """Load all protein sequences from the proteins directory.

        Returns:
            Dictionary mapping task IDs to protein sequences.
        """
        proteins_dir = self.data_dir / "proteins"
        if not proteins_dir.exists():
            logger.warning(f"Proteins directory not found: {proteins_dir}")
            return {}

        sequences = {}
        for fasta_path in proteins_dir.glob("*.fasta"):
            task_id = fasta_path.stem
            sequence = self._read_fasta(fasta_path)
            if sequence:
                sequences[task_id] = sequence

        logger.info(f"Loaded {len(sequences)} protein sequences")
        return sequences

    def _read_fasta(self, path: Path) -> Optional[str]:
        """Read protein sequence from FASTA file.

        Args:
            path: Path to the FASTA file.

        Returns:
            Protein sequence string, or None if file is empty/invalid.
        """
        try:
            with open(path, "r") as f:
                lines = f.readlines()

            # Skip header lines (starting with >)
            sequence_lines = [line.strip() for line in lines if not line.startswith(">")]
            sequence = "".join(sequence_lines)

            if not sequence:
                logger.warning(f"Empty sequence in FASTA file: {path}")
                return None

            return sequence
        except Exception as e:
            logger.error(f"Failed to read FASTA file {path}: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about available datasets.

        Returns:
            Dictionary with dataset counts and information.
        """
        stats = {
            "data_dir": str(self.data_dir),
            "task_list_provided": self.task_list is not None,
            "folds": {},
        }

        for fold in self.FOLDS:
            fold_dir = self.get_fold_dir(fold)
            if fold_dir.exists():
                task_ids = self.get_task_ids(fold)
                files = self._discover_files(fold_dir)

                stats["folds"][fold] = {
                    "task_count": len(task_ids),
                    "file_count": len(files),
                    "csv_count": len([f for f in files if f.suffix == ".csv"]),
                    "jsonl_gz_count": len([f for f in files if f.name.endswith(".jsonl.gz")]),
                }

        # Check proteins
        proteins_dir = self.data_dir / "proteins"
        if proteins_dir.exists():
            protein_files = list(proteins_dir.glob("*.fasta"))
            stats["proteins"] = {"count": len(protein_files)}

        return stats


def load_datasets_for_distance(
    data_dir: Union[str, Path],
    task_list_file: Optional[str] = None,
    source_fold: str = "train",
    target_fold: str = "test",
) -> Tuple[
    Dict[str, "MoleculeDataset"],
    Dict[str, "MoleculeDataset"],
    List[str],
    List[str],
]:
    """Convenience function to load source and target datasets for distance computation.

    Args:
        data_dir: Root directory containing train/test/valid folders.
        task_list_file: Optional name of task list JSON file.
        source_fold: Fold to use as source (default: "train").
        target_fold: Fold to use as target (default: "test").

    Returns:
        Tuple of (source_datasets, target_datasets, source_ids, target_ids)
    """
    loader = DatasetLoader(data_dir, task_list_file)

    source_datasets = loader.load_datasets(source_fold)
    target_datasets = loader.load_datasets(target_fold)

    source_ids = list(source_datasets.keys())
    target_ids = list(target_datasets.keys())

    return source_datasets, target_datasets, source_ids, target_ids
