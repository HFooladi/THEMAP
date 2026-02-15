"""
Pipeline runner for executing configuration-driven distance computation workflows.

This module provides the main pipeline execution engine that processes datasets
according to configuration specifications and computes distances between them.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Note: Individual dataset loading will be handled through Tasks.from_directory()
# or through the existing MoleculeDatasets and ProteinMetadataDatasets classes
from ..data.tasks import Task, Tasks
from ..distance import TaskDistance
from ..distance.molecule_distance import MoleculeDatasetDistance
from ..distance.protein_distance import ProteinDatasetDistance
from ..utils.logging import get_logger
from .config import PipelineConfig
from .output import OutputManager


class PipelineRunner:
    """Main pipeline execution engine."""

    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline runner.

        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        self.output_manager = OutputManager(config.output)

        # Track execution state
        self.start_time: Optional[float] = None
        self.results: Dict[str, Any] = {}
        self.errors: List[Dict[str, Any]] = []

    @property
    def cache_dir(self) -> Optional[str]:
        """
        Get the cache directory for feature storage.

        Uses the datasets directory as the cache location since that's where
        existing cached features are stored (embeddings/, protein_features_cache.pkl).

        Returns:
            Cache directory path if caching is enabled, None otherwise
        """
        if self.config.compute.cache_features:
            return "datasets"
        return None

    def run(self, base_data_path: str = "datasets") -> Dict[str, Any]:
        """
        Execute the complete pipeline.

        Args:
            base_data_path: Base path to dataset files

        Returns:
            Complete pipeline results
        """
        self.start_time = time.time()
        self.logger.info(f"Starting pipeline: {self.config.name}")

        try:
            # Validate configuration and datasets
            self._validate_pipeline(base_data_path)

            # Load datasets
            datasets_info = self._load_datasets(base_data_path)

            # Compute distances
            distance_results = self._compute_distances(datasets_info)

            # Compile final results
            self.results = {
                "config": self.config.to_dict(),
                "datasets_info": datasets_info,
                "distance_results": distance_results,
                "runtime_seconds": time.time() - self.start_time,
                "errors": self.errors,
            }

            # Save results
            saved_files = self.output_manager.save_results(self.results, "pipeline_results")
            summary_file = self.output_manager.create_summary_report(self.results)

            self.logger.info(
                f"Pipeline completed successfully in {self.results['runtime_seconds']:.2f} seconds"
            )
            self.logger.info(f"Results saved to: {saved_files}")
            self.logger.info(f"Summary report: {summary_file}")

            return self.results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.errors.append({"stage": "pipeline_execution", "error": str(e), "timestamp": time.time()})
            raise

    def _validate_pipeline(self, base_data_path: str) -> None:
        """Validate pipeline configuration and dataset availability."""
        self.logger.info("Validating pipeline configuration...")

        try:
            self.config.validate_datasets(base_data_path)
            self.logger.info("Configuration validation passed")
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

    def _load_datasets(self, base_data_path: str) -> List[Dict[str, Any]]:
        """
        Load all datasets specified in configuration.

        Args:
            base_data_path: Base path to dataset files

        Returns:
            List of dataset information dictionaries
        """
        self.logger.info("Loading datasets...")

        # Check if any modality uses directory-based configuration
        uses_directory_mode = (
            (self.config.molecule and self.config.molecule.directory)
            or (self.config.protein and self.config.protein.directory)
            or (self.config.metadata and self.config.metadata.directory)
        )

        if uses_directory_mode:
            # Use directory-based loading with Tasks.from_directory()
            return self._load_datasets_from_directory(base_data_path)
        else:
            # Use traditional individual dataset loading
            return self._load_datasets_individually(base_data_path)

    def _load_datasets_from_directory(self, base_data_path: str) -> List[Dict[str, Any]]:
        """Load datasets using the Tasks.from_directory() approach."""
        from ..data.tasks import Tasks

        self.logger.info("Loading datasets from directory structure...")

        # Get directory configuration from the first modality that has it
        directory_config = None
        if self.config.molecule and self.config.molecule.directory:
            directory_config = self.config.molecule.directory
        elif self.config.protein and self.config.protein.directory:
            directory_config = self.config.protein.directory
        elif self.config.metadata and self.config.metadata.directory:
            directory_config = self.config.metadata.directory

        if not directory_config:
            raise ValueError("No directory configuration found")

        # Determine the root path
        root_path = directory_config.root_path
        if not Path(root_path).is_absolute():
            root_path = str(Path(base_data_path) / root_path)

        # Determine task list file path
        task_list_file = directory_config.task_list_file
        if not Path(task_list_file).is_absolute():
            task_list_file = str(Path(root_path) / task_list_file)

        try:
            # Load tasks using from_directory
            tasks = Tasks.from_directory(
                directory=root_path,
                task_list_file=task_list_file,
                load_molecules=directory_config.load_molecules and bool(self.config.molecule),
                load_proteins=directory_config.load_proteins and bool(self.config.protein),
                load_metadata=directory_config.load_metadata and bool(self.config.metadata),
                metadata_types=directory_config.metadata_types,
                cache_dir=self.cache_dir,
            )

            self.logger.info(f"Loaded {len(tasks)} tasks from directory")

            # Convert Tasks object to the expected format for pipeline processing
            return self._convert_tasks_to_dataset_info(tasks)

        except Exception as e:
            error_info = {
                "stage": "dataset_loading",
                "dataset": "directory_based",
                "type": "all",
                "error": str(e),
                "timestamp": time.time(),
            }
            self.errors.append(error_info)
            self.logger.error(f"Failed to load datasets from directory: {e}")
            raise

    def _load_datasets_individually(self, base_data_path: str) -> List[Dict[str, Any]]:
        """Load datasets using individual dataset configurations."""
        self.logger.info("Loading individual datasets...")
        datasets_info = []
        base_path = Path(base_data_path)

        # Load molecule datasets
        if self.config.molecule and self.config.molecule.datasets:
            for dataset_config in self.config.molecule.datasets:
                try:
                    dataset_info = self._load_molecule_dataset(base_path, dataset_config)
                    datasets_info.append(dataset_info)
                except Exception as e:
                    error_info = {
                        "stage": "dataset_loading",
                        "dataset": dataset_config.name,
                        "type": "molecule",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    self.errors.append(error_info)
                    self.logger.warning(f"Failed to load molecule dataset {dataset_config.name}: {e}")

        # Load protein datasets
        if self.config.protein and self.config.protein.datasets:
            for dataset_config in self.config.protein.datasets:
                try:
                    dataset_info = self._load_protein_dataset(base_path, dataset_config)
                    datasets_info.append(dataset_info)
                except Exception as e:
                    error_info = {
                        "stage": "dataset_loading",
                        "dataset": dataset_config.name,
                        "type": "protein",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    self.errors.append(error_info)
                    self.logger.warning(f"Failed to load protein dataset {dataset_config.name}: {e}")

        self.logger.info(f"Loaded {len(datasets_info)} datasets")
        return datasets_info

    def _convert_tasks_to_dataset_info(self, tasks) -> List[Dict[str, Any]]:
        """Convert Tasks object to dataset info format expected by pipeline."""
        from ..data.tasks import DataFold

        datasets_info = []

        # Get task names from each fold
        train_tasks = [task.task_id for task in tasks._fold_to_tasks[DataFold.TRAIN]]
        validation_tasks = [task.task_id for task in tasks._fold_to_tasks[DataFold.VALIDATION]]
        test_tasks = [task.task_id for task in tasks._fold_to_tasks[DataFold.TEST]]

        # For directory-based loading, we create synthetic dataset configs
        # based on the tasks found in each fold
        all_task_names = set(train_tasks + validation_tasks + test_tasks)

        for task_name in all_task_names:
            # Determine source and target folds for this task
            source_fold = None
            target_folds = []

            if task_name in train_tasks:
                source_fold = "TRAIN"
                # Target folds are other folds where this task appears
                if task_name in validation_tasks:
                    target_folds.append("VALIDATION")
                if task_name in test_tasks:
                    target_folds.append("TEST")
            elif task_name in validation_tasks:
                source_fold = "VALIDATION"
                if task_name in test_tasks:
                    target_folds.append("TEST")
            elif task_name in test_tasks:
                source_fold = "TEST"

            if not target_folds:
                # If no target folds, use the same fold as target
                target_folds = [source_fold]

            # Get the actual task object
            task = None
            for fold in [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]:
                for t in tasks._fold_to_tasks[fold]:
                    if t.task_id == task_name:
                        task = t
                        break
                if task is not None:
                    break

            if task is None:
                continue

            # Determine available modalities
            modalities = []
            if hasattr(task, "molecule_dataset") and task.molecule_dataset is not None:
                modalities.append("molecule")
            if hasattr(task, "protein_dataset") and task.protein_dataset is not None:
                modalities.append("protein")
            if hasattr(task, "metadata_datasets") and task.metadata_datasets:
                modalities.append("metadata")

            # Create dataset info
            dataset_info = {
                "name": task_name,
                "type": "multimodal" if len(modalities) > 1 else modalities[0] if modalities else "unknown",
                "source_fold": source_fold,
                "target_folds": target_folds,
                "size": len(task.molecule_dataset)
                if hasattr(task, "molecule_dataset") and task.molecule_dataset
                else 0,
                "modalities": modalities,
                "task": task,  # Store the actual task object
            }

            datasets_info.append(dataset_info)

        return datasets_info

    def _load_molecule_dataset(self, base_path: Path, dataset_config) -> Dict[str, Any]:
        """Load a single molecule dataset."""
        from ..data.molecule_dataset import MoleculeDataset

        file_path = base_path / dataset_config.source_fold.lower() / f"{dataset_config.name}.jsonl.gz"

        # Load dataset using MoleculeDataset.load_from_file
        dataset = MoleculeDataset.load_from_file(str(file_path))

        # Sample dataset if configured
        sample_size = self.config.compute.sample_size
        if sample_size and len(dataset) > sample_size:
            # Simple sampling by taking first N samples
            dataset.data = dataset.data[:sample_size]

        # Create Task object from the loaded dataset
        from ..data.tasks import Task

        task = Task(
            task_id=dataset_config.name,
            molecule_dataset=dataset,
        )

        return {
            "name": dataset_config.name,
            "type": "molecule",
            "source_fold": dataset_config.source_fold,
            "target_folds": dataset_config.target_folds,
            "size": len(dataset),
            "modalities": ["molecule"],
            "dataset": dataset,
            "task": task,  # Add the Task object here
            "config": dataset_config,
        }

    def _load_protein_dataset(self, base_path: Path, dataset_config) -> Dict[str, Any]:
        """Load a single protein dataset."""
        # Note: Individual protein dataset loading is not currently supported.
        # Use directory-based loading with Tasks.from_directory() instead.
        raise NotImplementedError(
            "Individual protein dataset loading is not supported. "
            "Use directory-based configuration with 'directory' field instead of 'datasets' field."
        )

    def _compute_distances(self, datasets_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute all specified distances between datasets.

        Args:
            datasets_info: List of loaded dataset information

        Returns:
            List of distance computation results
        """
        self.logger.info("Computing distances...")
        distance_results = []

        # Group datasets by type
        molecule_datasets = [d for d in datasets_info if d["type"] == "molecule"]
        protein_datasets = [d for d in datasets_info if d["type"] == "protein"]

        # Compute molecule distances
        if self.config.molecule and molecule_datasets:
            molecule_results = self._compute_molecule_distances(molecule_datasets)
            distance_results.extend(molecule_results)

        # Compute protein distances
        if self.config.protein and protein_datasets:
            protein_results = self._compute_protein_distances(protein_datasets)
            distance_results.extend(protein_results)

        # Compute combined task distances if multiple modalities
        if self.config.task_distance and len(datasets_info) > 1:
            task_results = self._compute_task_distances(datasets_info)
            distance_results.extend(task_results)

        self.logger.info(f"Computed {len(distance_results)} distance measurements")
        return distance_results

    def _compute_molecule_distances(self, datasets_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute distances between molecule datasets."""
        results = []

        for featurizer in self.config.molecule.featurizers:
            for method in self.config.molecule.distance_methods:
                self.logger.info(f"Computing molecule distances: {featurizer} + {method}")

                start_time = time.time()
                try:
                    # Create Tasks collection from dataset info
                    # We need to extract the actual Task objects from datasets_info
                    train_tasks = []
                    test_tasks = []

                    for info in datasets_info:
                        if "task" in info and info["task"] is not None:
                            if info["source_fold"] == "TRAIN":
                                train_tasks.append(info["task"])
                            elif info["source_fold"] == "TEST":
                                test_tasks.append(info["task"])

                    # Create Tasks collection for distance computation
                    if train_tasks and test_tasks:
                        tasks = Tasks(train_tasks=train_tasks, test_tasks=test_tasks)

                        # Create distance computer
                        distance_computer = MoleculeDatasetDistance(tasks=tasks, molecule_method=method)

                        # Set the current featurizer
                        distance_computer._current_featurizer = featurizer

                        # Compute all pairwise distances at once
                        distance_results = distance_computer.get_distance()

                        # Parse results into expected format
                        for target_name, source_distances in distance_results.items():
                            for source_name, distance_value in source_distances.items():
                                result = {
                                    "source_dataset": source_name,
                                    "target_dataset": target_name,
                                    "source_fold": "TRAIN",  # Source is typically train
                                    "target_fold": "TEST",  # Target is typically test
                                    "modality": "molecule",
                                    "featurizer": featurizer,
                                    "method": method,
                                    "distance": float(distance_value),
                                    "computation_time": time.time() - start_time,
                                }
                                results.append(result)

                                # Save intermediate results if configured
                                if self.config.output.save_intermediate:
                                    self.output_manager.save_intermediate_results(
                                        result,
                                        "molecule_distance",
                                        f"{source_name}_{target_name}_{featurizer}_{method}",
                                    )
                    else:
                        self.logger.warning(
                            f"Insufficient tasks for distance computation: {len(train_tasks)} train, {len(test_tasks)} test"
                        )

                except Exception as e:
                    error_info = {
                        "stage": "distance_computation",
                        "modality": "molecule",
                        "featurizer": featurizer,
                        "method": method,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    self.errors.append(error_info)
                    self.logger.warning(f"Failed to compute distance: {error_info}")

        return results

    def _compute_protein_distances(self, datasets_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute distances between protein datasets."""
        results = []

        for featurizer in self.config.protein.featurizers:
            for method in self.config.protein.distance_methods:
                self.logger.info(f"Computing protein distances: {featurizer} + {method}")

                start_time = time.time()
                try:
                    # Group tasks by fold to create source/target collections
                    train_tasks = []
                    test_tasks = []

                    for info in datasets_info:
                        if info["task"].protein_dataset is not None:
                            if info["source_fold"] == "TRAIN":
                                train_tasks.append(info["task"])
                            elif info["source_fold"] == "TEST":
                                test_tasks.append(info["task"])

                    # Create Tasks collection for distance computation
                    if train_tasks and test_tasks:
                        tasks = Tasks(train_tasks=train_tasks, test_tasks=test_tasks)

                        # Create distance computer
                        distance_computer = ProteinDatasetDistance(
                            tasks=tasks,
                            protein_method=method,
                            embedding_method=featurizer,
                            max_workers=self.config.compute.max_workers,
                        )

                        # Compute all pairwise distances at once
                        distance_results = distance_computer.get_distance()

                        # Parse results into expected format
                        for target_name, source_distances in distance_results.items():
                            for source_name, distance_value in source_distances.items():
                                result = {
                                    "source_dataset": source_name,
                                    "target_dataset": target_name,
                                    "source_fold": "TRAIN",  # Source is typically train
                                    "target_fold": "TEST",  # Target is typically test
                                    "modality": "protein",
                                    "featurizer": featurizer,
                                    "method": method,
                                    "distance": float(distance_value),
                                    "computation_time": time.time() - start_time,
                                }
                                results.append(result)

                                # Save intermediate results if configured
                                if self.config.output.save_intermediate:
                                    self.output_manager.save_intermediate_results(
                                        result,
                                        "protein_distance",
                                        f"{source_name}_{target_name}_{featurizer}_{method}",
                                    )

                except Exception as e:
                    error_info = {
                        "stage": "distance_computation",
                        "modality": "protein",
                        "featurizer": featurizer,
                        "method": method,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    self.errors.append(error_info)
                    self.logger.warning(f"Failed to compute distance: {error_info}")

        return results

    def _compute_task_distances(self, datasets_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute combined task distances using multiple modalities."""
        results = []

        # Group datasets by name to create Tasks
        dataset_groups = {}
        for dataset_info in datasets_info:
            name = dataset_info["name"]
            if name not in dataset_groups:
                dataset_groups[name] = {}
            dataset_groups[name][dataset_info["type"]] = dataset_info

        # Only compute for datasets that have multiple modalities
        multi_modal_datasets = {name: info for name, info in dataset_groups.items() if len(info) > 1}

        if not multi_modal_datasets:
            self.logger.info("No multi-modal datasets found for task distance computation")
            return results

        self.logger.info(f"Computing task distances for {len(multi_modal_datasets)} multi-modal datasets")

        # Create Tasks objects
        tasks = []
        for name, modality_info in multi_modal_datasets.items():
            task_kwargs = {"task_id": name}

            if "molecule" in modality_info:
                task_kwargs["molecule_dataset"] = modality_info["molecule"]["dataset"]
            if "protein" in modality_info:
                task_kwargs["protein_dataset"] = modality_info["protein"]["dataset"]

            tasks.append(Task(**task_kwargs))

        # Create distance computer (kept for potential future use)
        _task_distance = TaskDistance(
            combination_strategy=self.config.task_distance.combination_strategy,
            weights=self.config.task_distance.weights,
            max_workers=self.config.compute.max_workers,
        )

        # Compute pairwise task distances
        for i, source_task in enumerate(tasks):
            for j, target_task in enumerate(tasks):
                if i <= j:  # Avoid duplicate computations
                    continue

                start_time = time.time()
                try:
                    # Create Tasks collection for this pair
                    task_collection = Tasks(train_tasks=[source_task], test_tasks=[target_task])

                    # Create new TaskDistance instance with tasks
                    task_distance_instance = TaskDistance(
                        tasks=task_collection,
                        combination_strategy=self.config.task_distance.combination_strategy,
                        weights=self.config.task_distance.weights,
                        max_workers=self.config.compute.max_workers,
                    )

                    distance_results = task_distance_instance.get_distance()
                    # Extract single distance value from results
                    distance = list(list(distance_results.values())[0].values())[0]

                    result = {
                        "source_dataset": source_task.task_id,
                        "target_dataset": target_task.task_id,
                        "modality": "combined_task",
                        "method": self.config.task_distance.combination_strategy,
                        "weights": self.config.task_distance.weights,
                        "distance": float(distance),
                        "computation_time": time.time() - start_time,
                    }
                    results.append(result)

                    # Save intermediate results if configured
                    if self.config.output.save_intermediate:
                        self.output_manager.save_intermediate_results(
                            result, "task_distance", f"{source_task.task_id}_{target_task.task_id}"
                        )

                except Exception as e:
                    error_info = {
                        "stage": "task_distance_computation",
                        "source_dataset": source_task.task_id,
                        "target_dataset": target_task.task_id,
                        "modality": "combined_task",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    self.errors.append(error_info)
                    self.logger.warning(f"Failed to compute task distance: {error_info}")

        return results

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for pipeline execution."""
        return get_logger(f"themap.pipeline.{self.config.name}")
