import json
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from themap.utils.cache_utils import GlobalMoleculeCache
from themap.utils.logging import get_logger
from themap.utils.protein_utils import (
    convert_fasta_to_dict,
    get_protein_accession,
    get_protein_features,
    get_protein_sequence,
)

logger = get_logger(__name__)


class DataFold(IntEnum):
    """Enum for data fold types."""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2


@dataclass
class ProteinDataset:
    """Single protein dataset representing one task.

    Args:
        task_id (str): Unique identifier for the task (CHEMBL ID)
        uniprot_id (str): UniProt accession ID for the protein
        sequence (str): Protein amino acid sequence
        features (Optional[NDArray[np.float32]]): Optional pre-computed protein features
    """

    task_id: str
    uniprot_id: str
    sequence: str
    features: Optional[NDArray[np.float32]] = None

    def __post_init__(self) -> None:
        """Validate initialization data."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")
        if not isinstance(self.uniprot_id, str):
            raise TypeError("uniprot_id must be a string")
        if not isinstance(self.sequence, str):
            raise TypeError("sequence must be a string")

    def __repr__(self) -> str:
        return f"ProteinDataset(task_id={self.task_id}, uniprot_id={self.uniprot_id}, seq_len={len(self.sequence)})"

    def get_features(
        self, featurizer_name: str = "esm3_sm_open_v1", layer: Optional[int] = None
    ) -> NDArray[np.float32]:
        """Get protein features using the specified featurizer.

        Args:
            featurizer_name: Name of the protein featurizer to use
            layer: Layer number for ESM models

        Returns:
            Computed protein features
        """
        if self.features is None:
            protein_dict = {self.uniprot_id: self.sequence}
            self.features = get_protein_features(protein_dict, featurizer_name, layer)

        return self.features


class ProteinDatasets:
    """Collection of protein datasets for different folds (train/validation/test).

    Similar to MoleculeDatasets but specifically designed for protein data management,
    including FASTA file downloading, caching, and feature computation.
    """

    def __init__(
        self,
        train_data_paths: Optional[List[RichPath]] = None,
        valid_data_paths: Optional[List[RichPath]] = None,
        test_data_paths: Optional[List[RichPath]] = None,
        num_workers: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize ProteinDatasets.

        Args:
            train_data_paths: List of paths to training FASTA files
            valid_data_paths: List of paths to validation FASTA files
            test_data_paths: List of paths to test FASTA files
            num_workers: Number of workers for data loading
            cache_dir: Directory for persistent caching
            uniprot_mapping_file: Path to CHEMBLID -> UNIPROT mapping file
        """
        logger.info("Initializing ProteinDatasets")
        self._fold_to_data_paths: Dict[DataFold, List[RichPath]] = {
            DataFold.TRAIN: train_data_paths or [],
            DataFold.VALIDATION: valid_data_paths or [],
            DataFold.TEST: test_data_paths or [],
        }
        self._num_workers = num_workers

        # Initialize global caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.global_cache = GlobalMoleculeCache(cache_dir) if cache_dir else None

        # Cache for loaded datasets
        self._loaded_datasets: Dict[str, ProteinDataset] = {}

        # UniProt mapping
        self.uniprot_mapping_file = uniprot_mapping_file or "datasets/uniprot_mapping.csv"
        self._uniprot_mapping: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized with {len(train_data_paths or [])} training, {len(valid_data_paths or [])} validation, and {len(test_data_paths or [])} test paths"
        )
        if cache_dir:
            logger.info(f"Global caching enabled at {cache_dir}")

    def __repr__(self) -> str:
        return f"ProteinDatasets(train={len(self._fold_to_data_paths[DataFold.TRAIN])}, valid={len(self._fold_to_data_paths[DataFold.VALIDATION])}, test={len(self._fold_to_data_paths[DataFold.TEST])})"

    @property
    def uniprot_mapping(self) -> pd.DataFrame:
        """Lazy load UniProt mapping dataframe."""
        if self._uniprot_mapping is None:
            self._uniprot_mapping = pd.read_csv(self.uniprot_mapping_file)
        return self._uniprot_mapping

    def get_uniprot_id_from_chembl(self, chembl_id: str) -> Optional[str]:
        """Get UniProt ID from ChEMBL ID using mapping file.

        Args:
            chembl_id: ChEMBL task ID

        Returns:
            UniProt accession ID if found, None otherwise
        """
        try:
            mapping_row = self.uniprot_mapping[self.uniprot_mapping["chembl_id"] == chembl_id]
            if not mapping_row.empty:
                return str(mapping_row.iloc[0]["target_accession_id"])
        except Exception as e:
            logger.warning(f"Failed to get UniProt ID for {chembl_id}: {e}")

        # Fallback: try API
        try:
            result = get_protein_accession(chembl_id)
            return result
        except Exception as e:
            logger.error(f"API fallback failed for {chembl_id}: {e}")
            return None

    def download_fasta_for_task(self, chembl_id: str, output_path: Path) -> bool:
        """Download FASTA file for a single task.

        Args:
            chembl_id: ChEMBL task ID
            output_path: Path where to save the FASTA file

        Returns:
            True if successful, False otherwise
        """
        uniprot_id = self.get_uniprot_id_from_chembl(chembl_id)
        if not uniprot_id:
            logger.error(f"Could not get UniProt ID for {chembl_id}")
            return False

        try:
            # Download protein sequence
            seq_records = get_protein_sequence(uniprot_id)
            if not seq_records:
                logger.error(f"Could not download sequence for UniProt ID {uniprot_id}")
                return False

            # Save to FASTA file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                SeqIO.write(seq_records, f, "fasta")

            logger.info(f"Downloaded FASTA for {chembl_id} -> {uniprot_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download FASTA for {chembl_id}: {e}")
            return False

    @staticmethod
    def create_fasta_files_from_task_list(
        task_list_file: Union[str, Path],
        output_dir: Union[str, Path],
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
    ) -> "ProteinDatasets":
        """Create FASTA files from a task list and return ProteinDatasets.

        Args:
            task_list_file: Path to JSON file containing fold-specific task lists
            output_dir: Base directory where to create train/test subdirectories
            uniprot_mapping_file: Path to CHEMBLID -> UNIPROT mapping file

        Returns:
            ProteinDatasets instance with paths to created FASTA files
        """
        logger.info(f"Creating FASTA files from task list {task_list_file}")

        # Load task list
        with open(task_list_file, "r") as f:
            task_data = json.load(f)

        output_dir = Path(output_dir)

        # Create ProteinDatasets instance for downloading
        protein_datasets = ProteinDatasets(uniprot_mapping_file=uniprot_mapping_file)

        # Process each fold
        fold_paths: Dict[DataFold, List[RichPath]] = {}
        fold_names = {"train": DataFold.TRAIN, "validation": DataFold.VALIDATION, "test": DataFold.TEST}

        for fold_name, fold_tasks in task_data.items():
            if fold_name in fold_names and fold_tasks:
                fold_dir = output_dir / fold_name
                fold_dir.mkdir(parents=True, exist_ok=True)

                fold_file_paths: List[RichPath] = []
                logger.info(f"Processing {len(fold_tasks)} tasks for {fold_name} fold")

                for task_id in fold_tasks:
                    fasta_path = fold_dir / f"{task_id}.fasta"

                    # Skip if file already exists
                    if fasta_path.exists():
                        logger.info(f"FASTA file already exists: {fasta_path}")
                        fold_file_paths.append(RichPath.create(str(fasta_path)))
                        continue

                    # Download FASTA file
                    if protein_datasets.download_fasta_for_task(task_id, fasta_path):
                        fold_file_paths.append(RichPath.create(str(fasta_path)))
                    else:
                        logger.warning(f"Failed to create FASTA for {task_id}")

                fold_paths[fold_names[fold_name]] = fold_file_paths
                logger.info(f"Created {len(fold_file_paths)} FASTA files for {fold_name}")

        # Create final ProteinDatasets instance
        return ProteinDatasets(
            train_data_paths=fold_paths.get(DataFold.TRAIN, []),
            valid_data_paths=fold_paths.get(DataFold.VALIDATION, []),
            test_data_paths=fold_paths.get(DataFold.TEST, []),
            uniprot_mapping_file=uniprot_mapping_file,
        )

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        uniprot_mapping_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> "ProteinDatasets":
        """Create ProteinDatasets from a directory containing FASTA files.

        Args:
            directory: Directory containing train/valid/test subdirectories with FASTA files
            task_list_file: File containing list of tasks to include
            cache_dir: Directory for persistent caching
            uniprot_mapping_file: Path to CHEMBLID -> UNIPROT mapping file
            **kwargs: Additional arguments

        Returns:
            ProteinDatasets instance
        """
        logger.info(f"Loading protein datasets from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        elif isinstance(directory, Path):
            directory = RichPath.create(str(directory))

        # Handle task list file
        fold_task_lists: Dict[str, List[str]] = {}
        task_list: Optional[List[str]] = None
        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            logger.info(f"Using task list file: {task_list_file}")

            with open(str(task_list_file), "r") as f:
                content = f.read().strip()

            try:
                task_data = json.loads(content)
                if isinstance(task_data, dict):
                    fold_task_lists = {
                        "train": task_data.get("train", []),
                        "valid": task_data.get("validation", []),
                        "test": task_data.get("test", []),
                    }
                    logger.info(
                        f"Loaded JSON task list: {len(fold_task_lists['train'])} train, "
                        f"{len(fold_task_lists['valid'])} validation, {len(fold_task_lists['test'])} test tasks"
                    )
                else:
                    task_list = task_data if isinstance(task_data, list) else [str(task_data)]
            except json.JSONDecodeError:
                task_list = [line.strip() for line in content.split("\n") if line.strip()]
                logger.info(f"Loaded text task list with {len(task_list)} tasks")

        def get_fold_file_names(data_fold_name: str) -> List[RichPath]:
            fold_dir = directory.join(data_fold_name)
            if not fold_dir.exists():
                logger.warning(f"Directory {fold_dir} does not exist")
                return []

            fold_path = Path(str(fold_dir))
            fasta_files = list(fold_path.glob("*.fasta"))

            rich_paths: List[RichPath] = []
            for file_path in fasta_files:
                rich_path = RichPath.create(str(file_path))
                task_name = file_path.stem  # Get filename without extension

                # Use fold-specific task list if available
                if fold_task_lists:
                    applicable_tasks = fold_task_lists.get(data_fold_name, [])
                    if not applicable_tasks or task_name in applicable_tasks:
                        rich_paths.append(rich_path)
                elif task_list is None or task_name in task_list:
                    rich_paths.append(rich_path)

            return rich_paths

        train_data_paths = get_fold_file_names("train")
        valid_data_paths = get_fold_file_names("valid")
        test_data_paths = get_fold_file_names("test")

        logger.info(
            f"Found {len(train_data_paths)} training, {len(valid_data_paths)} validation, and {len(test_data_paths)} test protein files"
        )

        return ProteinDatasets(
            train_data_paths=train_data_paths,
            valid_data_paths=valid_data_paths,
            test_data_paths=test_data_paths,
            cache_dir=cache_dir,
            uniprot_mapping_file=uniprot_mapping_file,
            **kwargs,
        )

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold."""
        return len(self._fold_to_data_paths[fold])

    def get_task_names(self, data_fold: DataFold) -> List[str]:
        """Get list of task names in a specific fold."""
        task_names: List[str] = []
        for path in self._fold_to_data_paths[data_fold]:
            # Extract task name from file path (assume CHEMBL format)
            file_name = path.basename()
            task_name = file_name.replace(".fasta", "")
            task_names.append(task_name)
        return task_names

    def load_datasets(self, folds: Optional[List[DataFold]] = None) -> Dict[str, ProteinDataset]:
        """Load all protein datasets from specified folds.

        Args:
            folds: List of folds to load. If None, loads all folds.

        Returns:
            Dictionary mapping dataset names to loaded ProteinDataset objects
        """
        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}

        datasets: Dict[str, ProteinDataset] = {}
        for fold in folds:
            fold_name = fold_names[fold]
            for path in self._fold_to_data_paths[fold]:
                file_name = path.basename()
                task_name = file_name.replace(".fasta", "")
                dataset_name = f"{fold_name}_{task_name}"

                # Check if already loaded
                if dataset_name not in self._loaded_datasets:
                    logger.info(f"Loading protein dataset {dataset_name}")

                    # Load FASTA file
                    fasta_dict = convert_fasta_to_dict(str(path))

                    if not fasta_dict:
                        logger.warning(f"Empty FASTA file: {path}")
                        continue

                    # Get first protein (assuming one protein per task)
                    uniprot_id = list(fasta_dict.keys())[0]
                    if "|" in uniprot_id:
                        uniprot_id = uniprot_id.split("|")[1]  # Extract UniProt ID from header

                    sequence = list(fasta_dict.values())[0]

                    protein_dataset = ProteinDataset(
                        task_id=task_name, uniprot_id=uniprot_id, sequence=sequence
                    )

                    self._loaded_datasets[dataset_name] = protein_dataset

                datasets[dataset_name] = self._loaded_datasets[dataset_name]

        logger.info(f"Loaded {len(datasets)} protein datasets")
        return datasets

    def compute_all_features_with_deduplication(
        self,
        featurizer_name: str = "esm3_sm_open_v1",
        layer: Optional[int] = None,
        folds: Optional[List[DataFold]] = None,
        batch_size: int = 100,
        force_recompute: bool = False,
    ) -> Dict[str, NDArray[np.float32]]:
        """Compute features for all protein datasets with UniProt ID deduplication.

        Args:
            featurizer_name: Name of protein featurizer to use
            layer: Layer number for ESM models
            folds: List of folds to process. If None, processes all folds
            batch_size: Batch size for feature computation
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Dictionary mapping dataset names to computed features
        """
        start_time = time.time()

        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        # Load all datasets
        datasets = self.load_datasets(folds)
        dataset_list = list(datasets.values())
        dataset_names = list(datasets.keys())

        logger.info(f"Computing protein features for {len(dataset_list)} datasets using {featurizer_name}")

        # Collect unique proteins by UniProt ID
        unique_proteins: Dict[str, str] = {}
        protein_to_datasets: Dict[str, List[Tuple[str, ProteinDataset]]] = {}

        for dataset_name, dataset in zip(dataset_names, dataset_list):
            uniprot_id = dataset.uniprot_id

            if uniprot_id not in unique_proteins:
                unique_proteins[uniprot_id] = dataset.sequence
                protein_to_datasets[uniprot_id] = []

            protein_to_datasets[uniprot_id].append((dataset_name, dataset))

        logger.info(f"Found {len(unique_proteins)} unique proteins across {len(dataset_list)} datasets")

        # Compute features for unique proteins
        results: Dict[str, NDArray[np.float32]] = {}
        if len(unique_proteins) > 0:
            logger.info(f"Computing features for {len(unique_proteins)} unique proteins")
            all_features = get_protein_features(unique_proteins, featurizer_name, layer)

            # Map features back to datasets
            for i, (uniprot_id, datasets_list) in enumerate(protein_to_datasets.items()):
                protein_features = all_features[i]

                for dataset_name, dataset in datasets_list:
                    dataset.features = protein_features
                    results[dataset_name] = protein_features
        else:
            logger.warning("No proteins found to compute features for")

        elapsed_time = time.time() - start_time
        total_datasets = len(dataset_list)
        unique_proteins_count = len(unique_proteins)

        if total_datasets > 0:
            deduplication_percentage = (total_datasets - unique_proteins_count) / total_datasets * 100
            logger.info(
                f"Computed protein features for {total_datasets} datasets "
                f"({unique_proteins_count} unique proteins) in {elapsed_time:.2f} seconds. "
                f"Deduplication saved {deduplication_percentage:.1f}% computation"
            )
        else:
            logger.info(f"No datasets to process - completed in {elapsed_time:.2f} seconds")

        return results

    def get_distance_computation_ready_features(
        self,
        featurizer_name: str = "esm3_sm_open_v1",
        layer: Optional[int] = None,
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[str], List[str]]:
        """Get protein features organized for efficient N×M distance matrix computation.

        Args:
            featurizer_name: Name of protein featurizer to use
            layer: Layer number for ESM models
            source_fold: Fold to use as source datasets (N)
            target_folds: Folds to use as target datasets (M)

        Returns:
            Tuple containing:
            - source_features: List of feature arrays for source datasets
            - target_features: List of feature arrays for target datasets
            - source_names: List of source dataset names
            - target_names: List of target dataset names
        """
        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]

        # Compute features for all relevant datasets
        all_folds = [source_fold] + target_folds
        all_features = self.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, layer=layer, folds=all_folds
        )

        # Create mapping from fold values to names
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}

        # Separate source and target features
        source_features: List[NDArray[np.float32]] = []
        source_names: List[str] = []
        target_features: List[NDArray[np.float32]] = []
        target_names: List[str] = []

        for dataset_name, features in all_features.items():
            fold_name = dataset_name.split("_")[0].lower()

            if fold_name == fold_names[source_fold]:
                source_features.append(features)
                source_names.append(dataset_name)
            elif any(fold_name == fold_names[fold] for fold in target_folds):
                target_features.append(features)
                target_names.append(dataset_name)

        logger.info(
            f"Prepared {len(source_features)} source and {len(target_features)} target protein datasets "
            f"for {len(source_features)}×{len(target_features)} distance matrix computation"
        )

        return source_features, target_features, source_names, target_names

    def save_features_to_file(
        self,
        output_path: Union[str, Path],
        featurizer_name: str = "esm3_sm_open_v1",
        layer: Optional[int] = None,
        folds: Optional[List[DataFold]] = None,
    ) -> None:
        """Save computed features to a pickle file for efficient loading.

        Args:
            output_path: Path where to save the features
            featurizer_name: Name of protein featurizer used
            layer: Layer number for ESM models
            folds: List of folds to save. If None, saves all folds
        """
        features = self.compute_all_features_with_deduplication(
            featurizer_name=featurizer_name, layer=layer, folds=folds
        )

        # Prepare data for saving
        save_data: Dict[str, Any] = {
            "features": features,
            "featurizer_name": featurizer_name,
            "layer": layer,
            "timestamp": time.time(),
            "num_datasets": len(features),
        }

        # Extract task IDs by fold
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}
        for fold, fold_name in fold_names.items():
            fold_tasks = [
                name.replace(f"{fold_name}_", "")
                for name in features.keys()
                if name.startswith(f"{fold_name}_")
            ]
            save_data[f"{fold_name}_task_ids"] = fold_tasks

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Saved protein features for {len(features)} datasets to {output_path}")

    @staticmethod
    def load_features_from_file(file_path: Union[str, Path]) -> Dict[str, NDArray[np.float32]]:
        """Load precomputed features from a pickle file.

        Args:
            file_path: Path to the saved features file

        Returns:
            Dictionary mapping dataset names to features
        """
        import pickle

        with open(file_path, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)

        logger.info(f"Loaded protein features for {data.get('num_datasets', 0)} datasets")
        logger.info(
            f"Featurizer: {data.get('featurizer_name', 'unknown')}, Layer: {data.get('layer', 'unknown')}"
        )

        features: Dict[str, NDArray[np.float32]] = data["features"]
        return features

    def get_global_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the global cache usage."""
        if self.global_cache is None or self.global_cache.persistent_cache is None:
            return None

        return {
            "persistent_cache_stats": self.global_cache.persistent_cache.get_stats(),
            "persistent_cache_size": self.global_cache.persistent_cache.get_cache_size_info(),
            "loaded_datasets": len(self._loaded_datasets),
        }


# Legacy compatibility: users can still import ProteinDatasets as old ProteinDataset
# But now we have both ProteinDataset (single) and ProteinDatasets (collection)
