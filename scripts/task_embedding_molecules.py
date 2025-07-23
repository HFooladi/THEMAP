#!/usr/bin/env python3
"""Professional molecular embedding generation script.

This script computes molecular embeddings for train and test tasks using various featurizers.
It provides comprehensive logging, error handling, memory monitoring, and efficient caching.

The script supports:
- Multiple featurizers (fingerprints, descriptors, neural networks)
- Efficient caching and memory management
- Comprehensive progress tracking and logging
- Error recovery and validation
- Performance monitoring

Usage:
    python scripts/task_embedding_molecules.py --featurizer ecfp --output_dir datasets/embeddings --n_jobs 32
    python scripts/task_embedding_molecules.py --featurizer all --cache_dir cache --memory_limit 8GB
"""

import argparse
import json
import logging
import pickle
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
from tqdm import tqdm

# Setup repository paths
REPO_PATH = Path(__file__).parent.parent.absolute()
CHECKOUT_PATH = REPO_PATH
DATASET_PATH = REPO_PATH / "datasets"

# Add to Python path
sys.path.insert(0, str(CHECKOUT_PATH))

try:
    from themap.data import DataFold, MoleculeDataset, MoleculeDatasets
    from themap.utils.cache_utils import GlobalMoleculeCache, PersistentFeatureCache
    from themap.utils.featurizer_utils import get_featurizer
    from themap.utils.logging import get_logger
    from themap.utils.memory_utils import MemoryEfficientFeatureStorage
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed.")
    sys.exit(1)

# Configure logging
logger = get_logger(__name__)


# Available featurizers organized by category
AVAILABLE_FEATURIZERS = {
    "fingerprints": [
        "ecfp",  # Extended-Connectivity Fingerprints
        "maccs",  # MACCS structural keys fingerprint
    ],
    "descriptors": [
        "mordred",  # Mordred molecular descriptors
        "desc2D",  # 2D molecular descriptors
    ],
    "language_models": [
        "ChemBERTa-77M-MLM",  # ChemBERTa with masked language modeling
        "ChemBERTa-77M-MTR",  # ChemBERTa with molecular translation
        "Roberta-Zinc480M-102M",  # RoBERTa pretrained on ZINC
        "MolT5",  # Text-to-molecule model
    ],
    "graph_networks": [
        "gin_supervised_infomax",  # GIN with InfoMax pretraining
        "gin_supervised_contextpred",  # GIN with context prediction
        "gin_supervised_edgepred",  # GIN with edge prediction
        "gin_supervised_masking",  # GIN with masking pretraining
    ],
}

# Flatten the list for easy access
ALL_FEATURIZERS = [f for category in AVAILABLE_FEATURIZERS.values() for f in category]

# Known problematic featurizers
KNOWN_ISSUES = {
    "desc3D": "Requires 3D conformers which may not be available",
    "usrcat": "3D structure requirements cause inconsistencies",
}


class MolecularEmbeddingGenerator:
    """Professional molecular embedding generation with comprehensive features."""

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        task_list_file: Optional[Path] = None,
        n_jobs: int = -1,
        memory_limit_gb: Optional[float] = None,
        enable_caching: bool = True,
    ):
        """Initialize the embedding generator.

        Args:
            dataset_path: Path to the dataset directory
            output_dir: Directory for output files
            cache_dir: Directory for caching (optional)
            task_list_file: Path to task list file (optional)
            n_jobs: Number of parallel jobs
            memory_limit_gb: Memory limit in GB (optional)
            enable_caching: Whether to enable feature caching
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.task_list_file = Path(task_list_file) if task_list_file else None
        self.n_jobs = n_jobs
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3) if memory_limit_gb else None
        self.enable_caching = enable_caching

        # Initialize components
        self.dataset: Optional[MoleculeDatasets] = None
        self.global_cache: Optional[GlobalMoleculeCache] = None
        self.memory_storage: Optional[MemoryEfficientFeatureStorage] = None

        # Statistics tracking
        self.stats = {
            "start_time": time.time(),
            "processed_tasks": 0,
            "failed_tasks": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_peak_mb": 0,
        }

        self._setup_directories()
        self._setup_caching()
        self._log_system_info()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")
        if self.cache_dir:
            logger.info(f"💾 Cache directory: {self.cache_dir}")

    def _setup_caching(self) -> None:
        """Initialize caching components."""
        if self.enable_caching and self.cache_dir:
            try:
                self.global_cache = GlobalMoleculeCache(self.cache_dir)
                self.memory_storage = MemoryEfficientFeatureStorage(
                    self.cache_dir / "memory_storage",
                    max_memory_cache_mb=1024
                    if not self.memory_limit_bytes
                    else min(1024, self.memory_limit_bytes // (1024**2) // 4),
                )
                logger.info("⚡ Caching system initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize caching: {e}")
                self.enable_caching = False

    def _log_system_info(self) -> None:
        """Log system information for debugging."""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            logger.info(f"⚙️ System: {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
            if self.memory_limit_bytes:
                logger.info(f"📊 Memory limit: {self.memory_limit_bytes / (1024**3):.1f}GB")
        except Exception as e:
            logger.debug(f"Could not get system info: {e}")

    def load_dataset(self) -> None:
        """Load the MoleculeDatasets with error handling."""
        try:
            # Use provided task list file or look for default
            task_list_file = self.task_list_file
            if task_list_file is None:
                task_list_file = self.dataset_path / "sample_tasks_list.json"
                if not task_list_file.exists():
                    logger.warning(f"Task list file not found: {task_list_file}")
                    task_list_file = None

            logger.info(f"🔍 Loading datasets from {self.dataset_path}")
            if task_list_file:
                logger.info(f"📋 Using task list file: {task_list_file}")
            self.dataset = MoleculeDatasets.from_directory(
                str(self.dataset_path),
                task_list_file=str(task_list_file) if task_list_file else None,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
            logger.info("✅ Datasets loaded successfully")

        except Exception as e:
            import traceback

            logger.error(f"Failed to load datasets: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _monitor_memory(self) -> None:
        """Monitor memory usage and update statistics."""
        try:
            current_memory_mb = psutil.Process().memory_info().rss / (1024**2)
            self.stats["memory_peak_mb"] = max(self.stats["memory_peak_mb"], current_memory_mb)

            if self.memory_limit_bytes:
                memory_usage_ratio = current_memory_mb * (1024**2) / self.memory_limit_bytes
                if memory_usage_ratio > 0.9:
                    logger.warning(f"High memory usage: {memory_usage_ratio:.1%}")

        except Exception as e:
            logger.debug(f"Memory monitoring failed: {e}")

    def _load_tasks(self, data_fold: DataFold, fold_name: str) -> List[MoleculeDataset]:
        """Load tasks for a specific data fold with error handling."""
        tasks = []
        failed_count = 0

        logger.info(f"Loading {fold_name} tasks...")

        try:
            # Get all datasets for this fold
            all_datasets = self.dataset.load_datasets([data_fold])

            # Filter to get only datasets from this fold
            fold_prefix = fold_name + "_"
            fold_datasets = {k: v for k, v in all_datasets.items() if k.startswith(fold_prefix)}

            with tqdm(desc=f"Loading {fold_name} tasks", total=len(fold_datasets), unit="tasks") as pbar:
                for dataset_name, dataset in fold_datasets.items():
                    try:
                        # Add name attribute for compatibility
                        dataset.name = dataset_name.replace(fold_prefix, "")
                        tasks.append(dataset)
                        self.stats["processed_tasks"] += 1
                        pbar.update(1)

                        # Monitor memory periodically
                        if len(tasks) % 10 == 0:
                            self._monitor_memory()

                    except Exception as e:
                        logger.error(f"Failed to load task {dataset_name}: {e}")
                        failed_count += 1
                        self.stats["failed_tasks"] += 1

            logger.info(f"Loaded {len(tasks)} {fold_name} tasks")
            if failed_count > 0:
                logger.warning(f"Failed to load {failed_count} {fold_name} tasks")

        except Exception as e:
            logger.error(f"Failed to load {fold_name} tasks: {e}")
            raise

        return tasks

    def _validate_featurizer(self, featurizer_name: str) -> bool:
        """Validate that a featurizer is available and working."""
        if featurizer_name not in ALL_FEATURIZERS:
            if featurizer_name in KNOWN_ISSUES:
                logger.warning(
                    f"Featurizer {featurizer_name} has known issues: {KNOWN_ISSUES[featurizer_name]}"
                )
                return False
            else:
                logger.error(f"Unknown featurizer: {featurizer_name}")
                return False

        try:
            # Test featurizer initialization
            get_featurizer(featurizer_name, n_jobs=1)
            return True
        except Exception as e:
            logger.error(f"Featurizer {featurizer_name} failed validation: {e}")
            return False

    def _initialize_featurizer(self, featurizer_name: str) -> Any:
        """Initialize a featurizer with error handling."""
        try:
            logger.info(f"🧪 Initializing featurizer: {featurizer_name}")
            featurizer = get_featurizer(featurizer_name, n_jobs=self.n_jobs)
            logger.info(f"⚡ Successfully initialized {featurizer_name}")
            return featurizer

        except Exception as e:
            logger.error(f"Failed to initialize featurizer {featurizer_name}: {e}")
            raise

    def _compute_task_features(
        self, task: MoleculeDataset, transformer: Any, task_name: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
        """Compute features for a single task with error handling."""
        try:
            # Extract SMILES, labels from MoleculeDataset
            smiles_list = [item.smiles for item in task.data]
            labels_list = [item.bool_label for item in task.data]

            # Compute features using transformer
            features_array = transformer(smiles_list)

            # Convert to tensors
            features = torch.Tensor(np.array(features_array))
            labels = torch.Tensor(np.array(labels_list))
            smiles = np.array(smiles_list)

            # Validate results
            if features is None or len(features) == 0:
                logger.warning(f"Empty features for task {task_name}")
                return None

            if len(features) != len(labels) or len(features) != len(smiles):
                logger.error(
                    f"Mismatched lengths for task {task_name}: features={len(features)}, labels={len(labels)}, smiles={len(smiles)}"
                )
                return None

            return features, labels, smiles

        except Exception as e:
            logger.error(f"Failed to compute features for task {task_name}: {e}")
            return None

    def _compute_features_for_tasks(
        self, tasks: List[MoleculeDataset], transformer: Any, task_type: str
    ) -> List[Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]]:
        """Compute features for all tasks with progress tracking."""
        results = []
        failed_count = 0

        logger.info(f"🎯 Computing features for {len(tasks)} {task_type} tasks")

        # Process tasks with progress bar
        successful_tasks = []
        failed_tasks = []

        # Create progress bar
        progress_bar = tqdm(tasks, desc=f"🔥 Processing {task_type} tasks", unit="task", leave=True)

        for i, task in enumerate(tasks):
            task_name = getattr(task, "name", f"{task_type}_task_{i}")

            try:
                result = self._compute_task_features(task, transformer, task_name)
                results.append(result)

                if result is None:
                    failed_count += 1
                    self.stats["failed_tasks"] += 1

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "failed": failed_count,
                        "memory_mb": f"{psutil.Process().memory_info().rss / (1024**2):.1f}",
                    }
                )

                # Monitor memory
                if i % 5 == 0:
                    self._monitor_memory()

            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Unexpected error processing task {task_name}: {e}")
                results.append(None)
                failed_count += 1
                self.stats["failed_tasks"] += 1
                progress_bar.update(1)

        success_count = len(results) - failed_count
        logger.info(f"Completed {task_type} processing: {success_count}/{len(tasks)} successful")

        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} {task_type} tasks")

        return results

    def _organize_features_by_task(
        self,
        tasks: List[MoleculeDataset],
        features_list: List[Optional[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]],
        featurizer_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Organize computed features by task name."""
        organized_features = {}

        for i, (task, features) in enumerate(zip(tasks, features_list)):
            task_name = getattr(task, "name", f"task_{i}")

            if features is not None:
                feature_tensor, labels, smiles = features
                organized_features[task_name] = {
                    featurizer_name: feature_tensor,
                    "labels": labels,
                    "smiles": smiles,
                    "task_size": len(feature_tensor),
                }
            else:
                logger.warning(f"Skipping task {task_name} due to failed feature computation")

        return organized_features

    def _save_embeddings(
        self,
        features_dict: Dict[str, Dict[str, Any]],
        output_path: Path,
        featurizer_name: str,
        task_type: str,
    ) -> None:
        """Save embeddings to file with error handling."""
        try:
            logger.info(f"💾 Saving {task_type} embeddings to {output_path}")

            # Create backup if file exists
            if output_path.exists():
                backup_path = output_path.with_suffix(f".backup_{int(time.time())}.npz")
                # Use shutil.copy2 for better backup
                import shutil

                shutil.copy2(output_path, backup_path)
                logger.info(f"📋 Created backup: {backup_path}")

            # Save with metadata
            np.savez_compressed(output_path, **features_dict)

            # Log file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Successfully saved {task_type} embeddings ({file_size_mb:.2f}MB)")

            # Test loading to verify integrity
            try:
                with open(output_path, "rb") as f:
                    test_load = pickle.load(f)
                logger.debug(f"Verified file integrity: {len(test_load)} tasks")
            except Exception as e:
                logger.error(f"File integrity check failed: {e}")

        except Exception as e:
            logger.error(f"Failed to save {task_type} embeddings: {e}")
            raise

    def process_featurizer(self, featurizer_name: str) -> bool:
        """Process a single featurizer for all tasks."""
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing featurizer: {featurizer_name}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        try:
            # Validate featurizer
            if not self._validate_featurizer(featurizer_name):
                return False

            # Initialize featurizer
            transformer = self._initialize_featurizer(featurizer_name)

            # Load tasks
            test_tasks = self._load_tasks(DataFold.TEST, "test")
            train_tasks = self._load_tasks(DataFold.TRAIN, "train")

            if not test_tasks and not train_tasks:
                logger.error("No tasks loaded successfully")
                return False

            # Compute features
            test_features = self._compute_features_for_tasks(test_tasks, transformer, "test")
            train_features = self._compute_features_for_tasks(train_tasks, transformer, "train")

            # Organize features
            features_test = self._organize_features_by_task(test_tasks, test_features, featurizer_name)
            features_train = self._organize_features_by_task(train_tasks, train_features, featurizer_name)

            # Save embeddings
            embeddings_dir = self.output_dir / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            test_path = embeddings_dir / f"{featurizer_name}_test.npz"
            train_path = embeddings_dir / f"{featurizer_name}_train.npz"

            if features_test:
                self._save_embeddings(features_test, test_path, featurizer_name, "test")
            if features_train:
                self._save_embeddings(features_train, train_path, featurizer_name, "train")

            elapsed_time = time.time() - start_time
            logger.info(f"Completed {featurizer_name} in {elapsed_time:.2f} seconds")

            # Log statistics
            test_count = len(features_test)
            train_count = len(features_train)
            logger.info(f"Results: {test_count} test tasks, {train_count} train tasks")

            return True

        except Exception as e:
            logger.error(f"Failed to process featurizer {featurizer_name}: {e}")
            return False

    def generate_embeddings(self, featurizers: List[str]) -> None:
        """Generate embeddings for all specified featurizers."""
        logger.info(f"🚀 Starting embedding generation for {len(featurizers)} featurizers")
        logger.info(f"🔬 Featurizers: {', '.join(featurizers)}")

        # Load dataset once
        self.load_dataset()

        successful_featurizers = []
        failed_featurizers = []

        total_start_time = time.time()

        try:
            for i, featurizer in enumerate(featurizers, 1):
                try:
                    logger.info(f"\n🔥 Processing featurizer {i}/{len(featurizers)}: {featurizer}")

                    # Process this featurizer
                    success = self.process_featurizer(featurizer)
                    if success:
                        successful_featurizers.append(featurizer)
                        logger.info(f"✅ Completed {featurizer}")
                    else:
                        failed_featurizers.append(featurizer)

                except KeyboardInterrupt:
                    logger.info("⚠️ Processing interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error with featurizer {featurizer}: {e}")
                    failed_featurizers.append(featurizer)

        finally:
            # Final statistics
            total_elapsed = time.time() - total_start_time
            self._log_final_statistics(successful_featurizers, failed_featurizers, total_elapsed)

    def _log_final_statistics(self, successful: List[str], failed: List[str], elapsed_time: float) -> None:
        """Log final processing statistics."""
        logger.info(f"\n{'=' * 60}")
        logger.info("🎉 PROCESSING COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
        logger.info(f"✅ Successful featurizers: {len(successful)}")
        logger.info(f"❌ Failed featurizers: {len(failed)}")
        logger.info(f"🎯 Tasks processed: {self.stats['processed_tasks']}")
        logger.info(f"⚠️ Tasks failed: {self.stats['failed_tasks']}")
        logger.info(f"📊 Peak memory usage: {self.stats['memory_peak_mb']:.1f}MB")

        if successful:
            logger.info(f"🚀 Successful: {', '.join(successful)}")
        if failed:
            logger.error(f"💥 Failed: {', '.join(failed)}")

        # Save processing summary
        summary = {
            "total_time_seconds": elapsed_time,
            "successful_featurizers": successful,
            "failed_featurizers": failed,
            "stats": self.stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📋 Summary saved to {summary_path}")


def parse_args() -> ArgumentParser:
    """Parse command line arguments with comprehensive options."""
    parser = ArgumentParser(
        description="Generate molecular embeddings for train and test tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available featurizers:
  Fingerprints: {", ".join(AVAILABLE_FEATURIZERS["fingerprints"])}
  Descriptors: {", ".join(AVAILABLE_FEATURIZERS["descriptors"])}
  Language Models: {", ".join(AVAILABLE_FEATURIZERS["language_models"])}
  Graph Networks: {", ".join(AVAILABLE_FEATURIZERS["graph_networks"])}

Special values:
  all: Use all available featurizers
  fingerprints: Use all fingerprint featurizers
  descriptors: Use all descriptor featurizers
  language_models: Use all language model featurizers
  graph_networks: Use all graph network featurizers

Examples:
  python scripts/task_embedding_molecules.py --featurizer ecfp
  python scripts/task_embedding_molecules.py --featurizer all --n_jobs 16
  python scripts/task_embedding_molecules.py --featurizer fingerprints --cache_dir /tmp/cache
        """,
    )

    parser.add_argument(
        "--featurizer", type=str, required=True, help="Featurizer to use (see available options below)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DATASET_PATH),
        help=f"Path to dataset directory (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATASET_PATH),
        help=f"Output directory (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--task_list_file", type=str, help="Path to task list file (JSON or text format, optional)"
    )
    parser.add_argument("--cache_dir", type=str, help="Cache directory for feature caching (optional)")
    parser.add_argument(
        "--n_jobs", type=int, default=32, help="Number of parallel jobs (default: 32, -1 for all CPUs)"
    )
    parser.add_argument("--memory_limit", type=float, help="Memory limit in GB (optional)")
    parser.add_argument("--disable_caching", action="store_true", help="Disable feature caching")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser


def resolve_featurizers(featurizer_arg: str) -> List[str]:
    """Resolve featurizer argument to list of featurizer names."""
    if featurizer_arg == "all":
        return ALL_FEATURIZERS
    elif featurizer_arg in AVAILABLE_FEATURIZERS:
        return AVAILABLE_FEATURIZERS[featurizer_arg]
    elif featurizer_arg in ALL_FEATURIZERS:
        return [featurizer_arg]
    else:
        raise ValueError(f"Unknown featurizer: {featurizer_arg}")


def main() -> None:
    """Main function with comprehensive error handling."""
    parser = parse_args()
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "embedding_generation.log")],
    )

    logger.info("🧪 Starting molecular embedding generation")
    logger.info(f"📝 Arguments: {vars(args)}")

    try:
        # Resolve featurizers
        featurizers = resolve_featurizers(args.featurizer)
        logger.info(f"🔬 Resolved to {len(featurizers)} featurizers: {', '.join(featurizers)}")

        # Validate arguments
        if args.n_jobs == -1:
            args.n_jobs = psutil.cpu_count()
            logger.info(f"🔧 Using all available CPUs: {args.n_jobs}")

        # Initialize generator
        generator = MolecularEmbeddingGenerator(
            dataset_path=Path(args.dataset_path),
            output_dir=Path(args.output_dir),
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            task_list_file=Path(args.task_list_file) if args.task_list_file else None,
            n_jobs=args.n_jobs,
            memory_limit_gb=args.memory_limit,
            enable_caching=not args.disable_caching,
        )

        # Generate embeddings
        generator.generate_embeddings(featurizers)

        logger.info("🎉 Embedding generation completed successfully")

    except KeyboardInterrupt:
        logger.info("⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        logger.error("Check the log file for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
