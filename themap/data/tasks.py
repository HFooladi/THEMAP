import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from ..utils.cache_utils import GlobalMoleculeCache
from ..utils.logging import get_logger, setup_logging
from .metadata import DataFold, MetadataDatasets, TextMetadataDataset
from .molecule_dataset import MoleculeDataset
from .protein_datasets import ProteinMetadataDataset, ProteinMetadataDatasets

# Setup logging
setup_logging()
logger = get_logger(__name__)


@dataclass
class Task:
    """A task represents a complete molecular property prediction problem.

    Each task contains:
    - Dataset: MoleculeDataset (set of molecules with SMILES and labels)
    - Metadata: Various metadata types including protein (single vectors per task)

    Args:
        task_id (str): Unique identifier for the task (e.g., CHEMBL ID)
        molecule_dataset (Optional[MoleculeDataset]): THE dataset - set of molecules for this task
        metadata_datasets (Optional[Dict[str, Any]]): Dictionary of metadata by type
            - Can include "protein" for protein metadata (single vector per task)
            - Can include "assay_description", "target_info", etc.
        hardness (Optional[float]): Optional measure of task difficulty

    Note:
        protein_dataset is deprecated - protein data should be stored in metadata_datasets["protein"]
    """

    task_id: str
    molecule_dataset: Optional[MoleculeDataset] = None
    metadata_datasets: Optional[Dict[str, Any]] = None
    hardness: Optional[float] = None

    # Deprecated field for backward compatibility
    protein_dataset: Optional[ProteinMetadataDataset] = None

    def __post_init__(self) -> None:
        """Validate task initialization and handle backward compatibility."""
        if not isinstance(self.task_id, str):
            raise TypeError("task_id must be a string")

        # Handle backward compatibility: move protein_dataset to metadata_datasets
        if self.protein_dataset is not None:
            logger.warning(
                f"protein_dataset is deprecated for task {self.task_id}. Move to metadata_datasets['protein']"
            )
            if self.metadata_datasets is None:
                self.metadata_datasets = {}
            if "protein" not in self.metadata_datasets:
                self.metadata_datasets["protein"] = self.protein_dataset

        # At least one data type must be present
        if not any([self.molecule_dataset, self.metadata_datasets]):
            raise ValueError(
                "Task must contain at least one data type (molecule_dataset or metadata_datasets)"
            )

        if self.molecule_dataset is not None and not isinstance(self.molecule_dataset, MoleculeDataset):
            raise TypeError("molecule_dataset must be a MoleculeDataset or None")
        if self.protein_dataset is not None and not isinstance(self.protein_dataset, ProteinMetadataDataset):
            raise TypeError("protein_dataset must be a ProteinMetadataDataset or None")
        if self.hardness is not None and not isinstance(self.hardness, (int, float)):
            raise TypeError("hardness must be a number or None")

    def __repr__(self) -> str:
        components = []
        if self.molecule_dataset:
            components.append(f"dataset={len(self.molecule_dataset)} molecules")
        if self.metadata_datasets:
            metadata_types = list(self.metadata_datasets.keys())
            components.append(f"metadata={metadata_types}")

        component_str = ", ".join(components)
        return f"Task(task_id={self.task_id}, {component_str}, hardness={self.hardness})"

    def get_molecule_features(self, featurizer_name: str, **kwargs: Any) -> Optional[NDArray[np.float32]]:
        """Get molecular features for this task.

        Args:
            featurizer_name: Name of molecular featurizer to use
            **kwargs: Additional featurizer arguments

        Returns:
            Molecular features or None if no molecule data
        """
        if self.molecule_dataset is None:
            return None

        return self.molecule_dataset.get_features(featurizer_name=featurizer_name, **kwargs)

    def get_protein_features(
        self, featurizer_name: str = "esm2_t33_650M_UR50D", layer: int = 33, **kwargs: Any
    ) -> Optional[NDArray[np.float32]]:
        """Get protein features for this task.

        Args:
            featurizer_name: Name of protein featurizer to use
            layer: Layer number for ESM models
            **kwargs: Additional featurizer arguments

        Returns:
            Protein features or None if no protein data
        """
        if self.protein_dataset is None:
            return None

        return self.protein_dataset.get_features(featurizer_name=featurizer_name, layer=layer, **kwargs)

    def get_metadata_features(
        self, metadata_type: str, featurizer_name: str, **kwargs: Any
    ) -> Optional[NDArray[np.float32]]:
        """Get metadata features for this task.

        Args:
            metadata_type: Type of metadata to get features for
            featurizer_name: Name of metadata featurizer to use
            **kwargs: Additional featurizer arguments

        Returns:
            Metadata features or None if metadata type not available
        """
        if self.metadata_datasets is None or metadata_type not in self.metadata_datasets:
            return None

        metadata_dataset = self.metadata_datasets[metadata_type]
        return metadata_dataset.get_features(featurizer_name=featurizer_name, **kwargs)  # type: ignore

    def get_combined_features(
        self,
        molecule_featurizer: Optional[str] = None,
        protein_featurizer: Optional[str] = None,
        metadata_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        combination_method: str = "concatenate",
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Get combined features from all available data types.

        Args:
            molecule_featurizer: Molecular featurizer name
            protein_featurizer: Protein featurizer name
            metadata_configs: Dict mapping metadata types to featurizer configs
            combination_method: How to combine features ('concatenate', 'average', 'weighted_average')
            **kwargs: Additional arguments

        Returns:
            Combined feature vector
        """
        feature_components = []
        component_names = []

        # Get molecular features
        if molecule_featurizer and self.molecule_dataset:
            mol_features = self.get_molecule_features(molecule_featurizer, **kwargs)
            if mol_features is not None:
                feature_components.append(mol_features.flatten())
                component_names.append("molecules")

        # Get protein features
        if protein_featurizer and self.protein_dataset:
            prot_features = self.get_protein_features(protein_featurizer, **kwargs)
            if prot_features is not None:
                feature_components.append(prot_features.flatten())
                component_names.append("protein")

        # Get metadata features
        if metadata_configs and self.metadata_datasets:
            for metadata_type, config in metadata_configs.items():
                featurizer_name = config.get("featurizer_name")
                featurizer_kwargs = config.get("kwargs", {})

                meta_features = self.get_metadata_features(
                    metadata_type=metadata_type,
                    featurizer_name=featurizer_name,  # type: ignore
                    **featurizer_kwargs,
                )
                if meta_features is not None:
                    feature_components.append(meta_features.flatten())
                    component_names.append(f"metadata_{metadata_type}")

        if not feature_components:
            raise ValueError(f"No features could be computed for task {self.task_id}")

        logger.debug(
            f"Task {self.task_id}: combining {len(feature_components)} feature types: {component_names}"
        )

        # Combine features
        if combination_method == "concatenate":
            return np.concatenate(feature_components).astype(np.float32)
        elif combination_method == "average":
            # Pad to same length and average
            max_len = max(len(f) for f in feature_components)
            padded_features = []
            for features in feature_components:
                if len(features) < max_len:
                    padded = np.pad(features, (0, max_len - len(features)), mode="constant")
                    padded_features.append(padded)
                else:
                    padded_features.append(features[:max_len])
            return np.mean(padded_features, axis=0).astype(np.float32)  # type: ignore
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")

    def get_task_embedding(self, data_model: Any, metadata_model: Any) -> NDArray[np.float32]:
        """Legacy method for backward compatibility.

        Args:
            data_model: Model for data feature extraction
            metadata_model: Model for metadata feature extraction

        Returns:
            Combined feature vector
        """
        logger.warning("get_task_embedding is deprecated, use get_combined_features instead")

        return self.get_combined_features(
            molecule_featurizer="morgan_fingerprints" if self.molecule_dataset else None,
            protein_featurizer="esm2_t33_650M_UR50D" if self.protein_dataset else None,
        )

    def __len__(self) -> int:
        """Get number of tasks."""
        return len(self.molecule_dataset) if self.molecule_dataset else 0


class Tasks:
    """Collection of tasks for molecular property prediction across different folds.

    This class manages multiple Task objects and provides unified access to
    molecular, protein, and metadata features across train/validation/test splits.
    """

    def __init__(
        self,
        train_tasks: Optional[List[Task]] = None,
        valid_tasks: Optional[List[Task]] = None,
        test_tasks: Optional[List[Task]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize Tasks collection.

        Args:
            train_tasks: List of training tasks
            valid_tasks: List of validation tasks
            test_tasks: List of test tasks
            cache_dir: Directory for persistent caching
        """
        logger.info("Initializing Tasks collection")
        self._fold_to_tasks: Dict[DataFold, List[Task]] = {
            DataFold.TRAIN: train_tasks or [],
            DataFold.VALIDATION: valid_tasks or [],
            DataFold.TEST: test_tasks or [],
        }

        # Initialize caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.global_cache = GlobalMoleculeCache(cache_dir) if cache_dir else None

        # Cache for computed features
        self._feature_cache: Dict[str, Dict[str, NDArray[np.float32]]] = {}

        logger.info(
            f"Initialized with {len(train_tasks or [])} training, {len(valid_tasks or [])} validation, "
            f"and {len(test_tasks or [])} test tasks"
        )

    def __repr__(self) -> str:
        return f"Tasks(train={len(self._fold_to_tasks[DataFold.TRAIN])}, valid={len(self._fold_to_tasks[DataFold.VALIDATION])}, test={len(self._fold_to_tasks[DataFold.TEST])})"

    @staticmethod
    def from_directory(
        directory: Union[str, RichPath],
        task_list_file: Optional[Union[str, RichPath]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        load_molecules: bool = True,
        load_proteins: bool = True,
        load_metadata: bool = True,
        metadata_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "Tasks":
        """Create Tasks from a directory structure.

        Expected directory structure:
        directory/
        ├── train/
        │   ├── CHEMBL123.jsonl.gz (molecules)
        │   ├── CHEMBL123.fasta (proteins)
        │   ├── CHEMBL123_assay.json (metadata)
        │   └── ...
        ├── valid/
        └── test/

        Args:
            directory: Base directory containing task data
            task_list_file: JSON file with fold-specific task lists
            cache_dir: Directory for persistent caching
            load_molecules: Whether to load molecular data
            load_proteins: Whether to load protein data
            load_metadata: Whether to load metadata
            metadata_types: List of metadata types to load
            **kwargs: Additional arguments

        Returns:
            Tasks instance with loaded data
        """
        logger.info(f"Loading tasks from directory {directory}")
        if isinstance(directory, str):
            directory = RichPath.create(directory)
        elif isinstance(directory, Path):
            directory = RichPath.create(str(directory))

        # Load task list for filtering
        fold_task_lists: Dict[str, List[str]] = {}
        if task_list_file is not None:
            if isinstance(task_list_file, str):
                task_list_file = RichPath.create(task_list_file)
            logger.info(f"Using task list file: {task_list_file}")

            with open(str(task_list_file), "r") as f:
                task_data = json.load(f)
                fold_task_lists = {
                    "train": task_data.get("train", []),
                    "valid": task_data.get("validation", []),
                    "test": task_data.get("test", []),
                }

        # Initialize data loaders
        molecule_datasets = None
        protein_datasets = None
        metadata_datasets_by_type: Dict[str, MetadataDatasets] = {}

        if load_molecules:
            try:
                from themap.data.molecule_datasets import MoleculeDatasets

                molecule_datasets = MoleculeDatasets.from_directory(
                    directory=directory, task_list_file=task_list_file, cache_dir=cache_dir, **kwargs
                )
                logger.info("Loaded molecule datasets")
            except Exception as e:
                logger.warning(f"Failed to load molecule datasets: {e}")

        if load_proteins:
            try:
                protein_datasets = ProteinMetadataDatasets.from_directory(
                    directory=directory, task_list_file=task_list_file, cache_dir=cache_dir, **kwargs
                )
                logger.info("Loaded protein datasets")
            except Exception as e:
                logger.warning(f"Failed to load protein datasets: {e}")

        if load_metadata:
            if metadata_types is None:
                metadata_types = ["assay_description", "bioactivity", "target_info"]

            for metadata_type in metadata_types:
                try:
                    # Try to load metadata of this type
                    metadata_datasets_by_type[metadata_type] = MetadataDatasets.from_directory(
                        directory=directory,
                        metadata_type=metadata_type,
                        dataset_class=TextMetadataDataset,  # Default to text, can be customized
                        task_list_file=task_list_file,
                        cache_dir=cache_dir,
                        file_pattern=f"*_{metadata_type}.json",
                        **kwargs,
                    )
                    logger.info(f"Loaded {metadata_type} metadata")
                except Exception as e:
                    logger.warning(f"Failed to load {metadata_type} metadata: {e}")

        # Create tasks for each fold
        fold_names = {"train": DataFold.TRAIN, "valid": DataFold.VALIDATION, "test": DataFold.TEST}
        tasks_by_fold: Dict[DataFold, List[Task]] = {fold: [] for fold in DataFold}

        for fold_name, fold_enum in fold_names.items():
            if fold_task_lists:
                task_ids = fold_task_lists.get(fold_name, [])
            else:
                # Get task IDs from any available data source
                task_ids = set()  # type: ignore
                if molecule_datasets:
                    task_ids.update(molecule_datasets.get_task_names(fold_enum))  # type: ignore
                if protein_datasets:
                    protein_task_names = protein_datasets.get_task_names(fold_enum)
                    task_ids.update(protein_task_names)  # type: ignore
                task_ids = list(task_ids)

            logger.info(f"Creating {len(task_ids)} tasks for {fold_name} fold")

            for task_id in task_ids:
                try:
                    # Get molecular data
                    molecule_dataset = None
                    if molecule_datasets:
                        molecule_dataset_name = f"{fold_name}_{task_id}"
                        molecule_data = molecule_datasets.load_datasets([fold_enum])  # type: ignore
                        molecule_dataset = molecule_data.get(molecule_dataset_name)

                    # Get protein data
                    protein_dataset = None
                    if protein_datasets:
                        protein_dataset_name = f"{fold_name}_{task_id}"
                        protein_data = protein_datasets.load_datasets([fold_enum])
                        protein_dataset = protein_data.get(protein_dataset_name)

                    # Get metadata
                    task_metadata: Dict[str, Any] = {}
                    for metadata_type, metadata_datasets_obj in metadata_datasets_by_type.items():
                        metadata_dataset_name = f"{fold_name}_{task_id}_{metadata_type}"
                        metadata_data = metadata_datasets_obj.load_datasets([fold_enum])
                        metadata_dataset = metadata_data.get(metadata_dataset_name)
                        if metadata_dataset:
                            task_metadata[metadata_type] = metadata_dataset

                    # Create task if we have any data
                    if molecule_dataset or protein_dataset or task_metadata:
                        task = Task(
                            task_id=task_id,
                            molecule_dataset=molecule_dataset,
                            protein_dataset=protein_dataset,
                            metadata_datasets=task_metadata if task_metadata else None,
                        )
                        tasks_by_fold[fold_enum].append(task)
                        logger.debug(f"Created task {task_id} for {fold_name}")
                    else:
                        logger.warning(f"No data found for task {task_id} in {fold_name}")

                except Exception as e:
                    logger.error(f"Failed to create task {task_id} for {fold_name}: {e}")

        return Tasks(
            train_tasks=tasks_by_fold[DataFold.TRAIN],
            valid_tasks=tasks_by_fold[DataFold.VALIDATION],
            test_tasks=tasks_by_fold[DataFold.TEST],
            cache_dir=cache_dir,
        )

    def get_num_fold_tasks(self, fold: DataFold) -> int:
        """Get number of tasks in a specific fold."""
        return len(self._fold_to_tasks[fold])

    def get_task_ids(self, fold: DataFold) -> List[str]:
        """Get list of task IDs in a specific fold."""
        return [task.task_id for task in self._fold_to_tasks[fold]]

    def get_tasks(self, fold: DataFold) -> List[Task]:
        """Get list of tasks in a specific fold."""
        return self._fold_to_tasks[fold].copy()

    def __len__(self) -> int:
        """Get number of tasks."""
        return sum(len(tasks) for tasks in self._fold_to_tasks.values())

    def __getitem__(self, index: int) -> List[Task]:
        """Get a task by index.

        Args:
            index: int: index of the task

        Returns:
            List[Task]: list of tasks

        Note:
         index 0: Train Tasks
         index 1: Validation Tasks
         index 2: Test Tasks

        Raises:
            IndexError: if index is out of range
        """
        if index == 0:
            return self.get_tasks(DataFold.TRAIN)
        elif index == 1:
            return self.get_tasks(DataFold.VALIDATION)
        elif index == 2:
            return self.get_tasks(DataFold.TEST)
        else:
            raise IndexError(f"Index {index} is out of range")

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a specific task by its ID."""
        for tasks in self._fold_to_tasks.values():
            for task in tasks:
                if task.task_id == task_id:
                    return task
        return None

    def compute_all_task_features(
        self,
        molecule_featurizer: Optional[str] = None,
        protein_featurizer: Optional[str] = None,
        metadata_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        combination_method: str = "concatenate",
        folds: Optional[List[DataFold]] = None,
        force_recompute: bool = False,
        **kwargs: Any,
    ) -> Dict[str, NDArray[np.float32]]:
        """Compute combined features for all tasks.

        Args:
            molecule_featurizer: Molecular featurizer name
            protein_featurizer: Protein featurizer name
            metadata_configs: Metadata featurizer configurations
            combination_method: How to combine features
            folds: List of folds to process
            force_recompute: Whether to force recomputation
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping task names to combined features
        """
        start_time = time.time()

        if folds is None:
            folds = [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]

        # Create cache key
        cache_key = (
            f"{molecule_featurizer}_{protein_featurizer}_{hash(str(metadata_configs))}_{combination_method}"
        )

        if not force_recompute and cache_key in self._feature_cache:
            logger.info("Using cached task features")
            return self._feature_cache[cache_key]

        # Get all tasks to process
        all_tasks = []
        task_names = []
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "validation", DataFold.TEST: "test"}

        for fold in folds:
            fold_name = fold_names[fold]
            for task in self._fold_to_tasks[fold]:
                all_tasks.append(task)
                task_names.append(f"{fold_name}_{task.task_id}")

        logger.info(f"Computing combined features for {len(all_tasks)} tasks")

        # Compute features for each task
        results: Dict[str, NDArray[np.float32]] = {}
        for task, task_name in zip(all_tasks, task_names):
            try:
                logger.debug(f"Computing features for task {task.task_id}")
                features = task.get_combined_features(
                    molecule_featurizer=molecule_featurizer,
                    protein_featurizer=protein_featurizer,
                    metadata_configs=metadata_configs,
                    combination_method=combination_method,
                    **kwargs,
                )
                results[task_name] = features
            except Exception as e:
                logger.error(f"Failed to compute features for task {task.task_id}: {e}")
                # Use zero features as fallback
                results[task_name] = np.zeros(100, dtype=np.float32)

        # Cache results
        self._feature_cache[cache_key] = results

        elapsed_time = time.time() - start_time
        logger.info(f"Computed task features for {len(results)} tasks in {elapsed_time:.2f} seconds")

        return results

    def get_distance_computation_ready_features(
        self,
        molecule_featurizer: Optional[str] = None,
        protein_featurizer: Optional[str] = None,
        metadata_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        combination_method: str = "concatenate",
        source_fold: DataFold = DataFold.TRAIN,
        target_folds: Optional[List[DataFold]] = None,
        **kwargs: Any,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]], List[str], List[str]]:
        """Get task features organized for efficient N×M distance matrix computation.

        Args:
            molecule_featurizer: Molecular featurizer name
            protein_featurizer: Protein featurizer name
            metadata_configs: Metadata featurizer configurations
            combination_method: How to combine features
            source_fold: Fold to use as source tasks (N)
            target_folds: Folds to use as target tasks (M)
            **kwargs: Additional arguments

        Returns:
            Tuple containing:
            - source_features: List of feature arrays for source tasks
            - target_features: List of feature arrays for target tasks
            - source_names: List of source task names
            - target_names: List of target task names
        """
        if target_folds is None:
            target_folds = [DataFold.VALIDATION, DataFold.TEST]

        # Compute features for all relevant tasks
        all_folds = [source_fold] + target_folds
        all_features = self.compute_all_task_features(
            molecule_featurizer=molecule_featurizer,
            protein_featurizer=protein_featurizer,
            metadata_configs=metadata_configs,
            combination_method=combination_method,
            folds=all_folds,
            **kwargs,
        )

        # Create mapping from fold values to names
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "valid", DataFold.TEST: "test"}

        # Separate source and target features
        source_features: List[NDArray[np.float32]] = []
        source_names: List[str] = []
        target_features: List[NDArray[np.float32]] = []
        target_names: List[str] = []

        for task_name, features in all_features.items():
            fold_name = task_name.split("_")[0].lower()

            if fold_name == fold_names[source_fold]:
                source_features.append(features)
                source_names.append(task_name)
            elif any(fold_name == fold_names[fold] for fold in target_folds):
                target_features.append(features)
                target_names.append(task_name)

        logger.info(
            f"Prepared {len(source_features)} source and {len(target_features)} target tasks "
            f"for {len(source_features)}×{len(target_features)} distance matrix computation"
        )

        return source_features, target_features, source_names, target_names

    def save_task_features_to_file(
        self,
        output_path: Union[str, Path],
        molecule_featurizer: Optional[str] = None,
        protein_featurizer: Optional[str] = None,
        metadata_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        combination_method: str = "concatenate",
        folds: Optional[List[DataFold]] = None,
        **kwargs: Any,
    ) -> None:
        """Save computed task features to a pickle file for efficient loading."""
        features = self.compute_all_task_features(
            molecule_featurizer=molecule_featurizer,
            protein_featurizer=protein_featurizer,
            metadata_configs=metadata_configs,
            combination_method=combination_method,
            folds=folds,
            **kwargs,
        )

        # Prepare data for saving
        save_data: Dict[str, Any] = {
            "task_features": features,
            "molecule_featurizer": molecule_featurizer,
            "protein_featurizer": protein_featurizer,
            "metadata_configs": metadata_configs,
            "combination_method": combination_method,
            "timestamp": time.time(),
            "num_tasks": len(features),
        }

        # Extract task IDs by fold
        fold_names = {DataFold.TRAIN: "train", DataFold.VALIDATION: "valid", DataFold.TEST: "test"}
        for fold, fold_name in fold_names.items():
            fold_task_ids = [
                name.replace(f"{fold_name}_", "")
                for name in features.keys()
                if name.startswith(f"{fold_name}_")
            ]
            save_data[f"{fold_name}_task_ids"] = fold_task_ids

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Saved task features for {len(features)} tasks to {output_path}")

    @staticmethod
    def load_task_features_from_file(file_path: Union[str, Path]) -> Dict[str, NDArray[np.float32]]:
        """Load precomputed task features from a pickle file."""
        import pickle

        with open(file_path, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)

        logger.info(f"Loaded task features for {data.get('num_tasks', 0)} tasks")
        logger.info(f"Molecule featurizer: {data.get('molecule_featurizer', 'None')}")
        logger.info(f"Protein featurizer: {data.get('protein_featurizer', 'None')}")
        logger.info(f"Combination method: {data.get('combination_method', 'unknown')}")

        features: Dict[str, NDArray[np.float32]] = data["task_features"]
        return features

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about feature caching."""
        return {
            "feature_cache_entries": len(self._feature_cache),
            "global_cache_available": self.global_cache is not None,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
