"""
Main pipeline orchestrator for THEMAP.

This module provides the high-level Pipeline class that orchestrates:
1. Data loading from directory structure
2. Feature computation (molecules, proteins)
3. Distance computation
4. Result output to CSV/JSON

Example:
    >>> from themap.config import PipelineConfig
    >>> from themap.pipeline import Pipeline
    >>> config = PipelineConfig.from_yaml("config.yaml")
    >>> pipeline = Pipeline(config)
    >>> results = pipeline.run()
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..config import PipelineConfig
from ..data.loader import DatasetLoader
from ..data.molecule_dataset import MoleculeDataset
from ..distance import (
    combine_distance_matrices,
    compute_dataset_distance_matrix,
    compute_metadata_distance_matrix,
)
from ..features.cache import FeatureCache
from ..features.molecule import MoleculeFeaturizer
from ..features.protein import ProteinFeaturizer
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Type aliases
DistanceMatrix = Dict[str, Dict[str, float]]


class Pipeline:
    """Main pipeline orchestrator for THEMAP distance computation.

    Orchestrates the complete workflow:
    1. Load datasets from train/test directories
    2. Compute molecule features (with caching)
    3. Compute protein features (with caching, if enabled)
    4. Compute distance matrices
    5. Combine matrices (if needed)
    6. Save results to output directory

    Attributes:
        config: Pipeline configuration
        loader: Dataset loader
        cache: Feature cache (if save_features is enabled)
        mol_featurizer: Molecule featurizer (if molecule distance enabled)
        prot_featurizer: Protein featurizer (if protein distance enabled)

    Examples:
        >>> config = PipelineConfig.from_yaml("config.yaml")
        >>> pipeline = Pipeline(config)
        >>> results = pipeline.run()
        >>> print(results["molecule"])  # molecule distance matrix
    """

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config

        # Initialize data loader
        self.loader = DatasetLoader(
            config.data.directory,
            config.data.task_list,
        )

        # Initialize feature cache if saving features
        self.cache: Optional[FeatureCache] = None
        if config.output.save_features:
            cache_dir = config.output.directory / "features"
            self.cache = FeatureCache(cache_dir)

        # Initialize featurizers (lazy)
        self._mol_featurizer: Optional[MoleculeFeaturizer] = None
        self._prot_featurizer: Optional[ProteinFeaturizer] = None

    @property
    def mol_featurizer(self) -> MoleculeFeaturizer:
        """Get molecule featurizer (lazy initialization)."""
        if self._mol_featurizer is None:
            self._mol_featurizer = MoleculeFeaturizer(
                self.config.molecule.featurizer,
                n_jobs=self.config.compute.n_jobs,
                device=self.config.compute.device,
            )
        return self._mol_featurizer

    @property
    def prot_featurizer(self) -> ProteinFeaturizer:
        """Get protein featurizer (lazy initialization)."""
        if self._prot_featurizer is None:
            self._prot_featurizer = ProteinFeaturizer(
                self.config.protein.featurizer,
                layer=self.config.protein.layer,
                device=self.config.compute.device,
            )
        return self._prot_featurizer

    def run(self) -> Dict[str, DistanceMatrix]:
        """Run the complete pipeline.

        Returns:
            Dictionary mapping distance type to distance matrix:
            - "molecule": molecule distance matrix (if enabled)
            - "protein": protein distance matrix (if enabled)
            - "combined": combined matrix (if combination != "separate")

        Raises:
            ValueError: If no distance type is enabled
        """
        logger.info("Starting THEMAP pipeline")

        # Validate configuration
        issues = self.config.validate()
        for issue in issues:
            logger.warning(f"Config issue: {issue}")

        # Create output directory
        self.config.output.directory.mkdir(parents=True, exist_ok=True)

        # Load datasets
        logger.info("Loading datasets...")
        train_datasets = self.loader.load_datasets("train")
        test_datasets = self.loader.load_datasets("test")

        if not train_datasets:
            raise ValueError("No training datasets found")
        if not test_datasets:
            raise ValueError("No test datasets found")

        train_ids = list(train_datasets.keys())
        test_ids = list(test_datasets.keys())

        logger.info(f"Loaded {len(train_ids)} train and {len(test_ids)} test datasets")

        results: Dict[str, DistanceMatrix] = {}

        # Compute molecule distances
        if self.config.molecule.enabled:
            logger.info("Computing molecule distances...")
            results["molecule"] = self._compute_molecule_distances(
                train_datasets, test_datasets, train_ids, test_ids
            )

        # Compute protein distances
        if self.config.protein.enabled:
            logger.info("Computing protein distances...")
            results["protein"] = self._compute_protein_distances(train_ids, test_ids)

        # Combine distances
        if len(results) > 1 and self.config.combination.strategy != "separate":
            logger.info("Combining distance matrices...")
            results["combined"] = combine_distance_matrices(
                results,
                weights=self.config.combination.weights,
                combination=self.config.combination.strategy,
            )

        # Save results
        self._save_results(results)

        logger.info("Pipeline completed successfully")
        return results

    def _compute_molecule_distances(
        self,
        train_datasets: Dict[str, MoleculeDataset],
        test_datasets: Dict[str, MoleculeDataset],
        train_ids: List[str],
        test_ids: List[str],
    ) -> DistanceMatrix:
        """Compute molecule distance matrix."""
        featurizer_name = self.config.molecule.featurizer

        # Try to load from cache
        train_features, train_labels, missing_train = self._load_or_compute_molecule_features(
            train_datasets, train_ids, featurizer_name
        )
        test_features, test_labels, missing_test = self._load_or_compute_molecule_features(
            test_datasets, test_ids, featurizer_name
        )

        # Convert to lists for distance computation
        train_features_list = [train_features[tid] for tid in train_ids]
        train_labels_list = [train_labels[tid] for tid in train_ids]
        test_features_list = [test_features[tid] for tid in test_ids]
        test_labels_list = [test_labels[tid] for tid in test_ids]

        # Compute distances
        return compute_dataset_distance_matrix(
            source_features=train_features_list,
            source_labels=train_labels_list,
            target_features=test_features_list,
            target_labels=test_labels_list,
            source_ids=train_ids,
            target_ids=test_ids,
            method=self.config.molecule.method,
            n_jobs=self.config.compute.n_jobs,
        )

    def _load_or_compute_molecule_features(
        self,
        datasets: Dict[str, MoleculeDataset],
        task_ids: List[str],
        featurizer_name: str,
    ) -> Tuple[
        Dict[str, NDArray[np.float32]],
        Dict[str, NDArray[np.int32]],
        List[str],
    ]:
        """Load molecule features from cache or compute them."""
        features: Dict[str, NDArray[np.float32]] = {}
        labels: Dict[str, NDArray[np.int32]] = {}
        missing: List[str] = []

        # Try to load from cache
        if self.cache:
            for tid in task_ids:
                cached_feat, cached_lab = self.cache.load_molecule_features(tid, featurizer_name)
                if cached_feat is not None and cached_lab is not None:
                    features[tid] = cached_feat
                    labels[tid] = cached_lab
                else:
                    missing.append(tid)
        else:
            missing = task_ids.copy()

        # Compute features for missing datasets
        if missing:
            logger.info(f"Computing features for {len(missing)} datasets")
            missing_datasets = {tid: datasets[tid] for tid in missing}

            # Use deduplicated featurization
            computed = self.mol_featurizer.featurize_datasets(missing_datasets, deduplicate=True)

            for tid in missing:
                features[tid] = computed[tid]
                labels[tid] = datasets[tid].labels

                # Save to cache
                if self.cache:
                    self.cache.save_molecule_features(tid, featurizer_name, features[tid], labels[tid])

        return features, labels, missing

    def _compute_protein_distances(
        self,
        train_ids: List[str],
        test_ids: List[str],
    ) -> DistanceMatrix:
        """Compute protein distance matrix."""
        featurizer_name = self.config.protein.featurizer

        # Load protein sequences
        sequences = self.loader.load_protein_sequences()
        if not sequences:
            logger.warning("No protein sequences found, skipping protein distances")
            return {}

        # Filter to only tasks with proteins
        train_with_protein = [tid for tid in train_ids if tid in sequences]
        test_with_protein = [tid for tid in test_ids if tid in sequences]

        if not train_with_protein or not test_with_protein:
            logger.warning("No matching protein sequences for tasks")
            return {}

        # Load or compute protein features
        train_features = self._load_or_compute_protein_features(
            {tid: sequences[tid] for tid in train_with_protein},
            featurizer_name,
        )
        test_features = self._load_or_compute_protein_features(
            {tid: sequences[tid] for tid in test_with_protein},
            featurizer_name,
        )

        # Stack into arrays
        train_vectors = np.vstack([train_features[tid] for tid in train_with_protein])
        test_vectors = np.vstack([test_features[tid] for tid in test_with_protein])

        # Compute distances
        return compute_metadata_distance_matrix(
            source_vectors=train_vectors,
            target_vectors=test_vectors,
            source_ids=train_with_protein,
            target_ids=test_with_protein,
            method=self.config.protein.method,
        )

    def _load_or_compute_protein_features(
        self,
        sequences: Dict[str, str],
        featurizer_name: str,
    ) -> Dict[str, NDArray[np.float32]]:
        """Load protein features from cache or compute them."""
        features: Dict[str, NDArray[np.float32]] = {}
        missing: Dict[str, str] = {}

        # Try to load from cache
        if self.cache:
            for tid, seq in sequences.items():
                cached = self.cache.load_protein_features(tid, featurizer_name)
                if cached is not None:
                    features[tid] = cached
                else:
                    missing[tid] = seq
        else:
            missing = sequences

        # Compute features for missing proteins
        if missing:
            logger.info(f"Computing protein features for {len(missing)} proteins")
            computed_features = self.prot_featurizer.featurize(missing)

            # Map back to task IDs
            task_ids = list(missing.keys())
            for i, tid in enumerate(task_ids):
                features[tid] = computed_features[i]

                # Save to cache
                if self.cache:
                    self.cache.save_protein_features(tid, featurizer_name, features[tid])

        return features

    def _save_results(self, results: Dict[str, DistanceMatrix]) -> None:
        """Save distance matrices to output directory."""
        output_dir = self.config.output.directory
        fmt = self.config.output.format

        for name, matrix in results.items():
            if not matrix:
                continue

            if fmt == "csv":
                self._save_matrix_csv(matrix, output_dir / f"{name}_distances.csv")
            elif fmt == "json":
                self._save_matrix_json(matrix, output_dir / f"{name}_distances.json")
            elif fmt == "npz":
                self._save_matrix_npz(matrix, output_dir / f"{name}_distances.npz")

        logger.info(f"Results saved to {output_dir}")

    def _save_matrix_csv(self, matrix: DistanceMatrix, path: Path) -> None:
        """Save distance matrix to CSV (wide format)."""
        # Convert to DataFrame
        df = pd.DataFrame(matrix).T  # Transpose so rows are source, cols are target

        df.to_csv(path)
        logger.info(f"Saved {path}")

    def _save_matrix_json(self, matrix: DistanceMatrix, path: Path) -> None:
        """Save distance matrix to JSON."""
        import json

        with open(path, "w") as f:
            json.dump(matrix, f, indent=2)
        logger.info(f"Saved {path}")

    def _save_matrix_npz(self, matrix: DistanceMatrix, path: Path) -> None:
        """Save distance matrix to NPZ."""
        # Convert to array
        target_ids = list(matrix.keys())
        source_ids = list(matrix[target_ids[0]].keys())

        arr = np.array([[matrix[tid][sid] for sid in source_ids] for tid in target_ids])

        np.savez(
            path,
            distances=arr,
            source_ids=source_ids,
            target_ids=target_ids,
        )
        logger.info(f"Saved {path}")


def run_pipeline(config_path: str) -> Dict[str, DistanceMatrix]:
    """Convenience function to run pipeline from config file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary of distance matrices.
    """
    config = PipelineConfig.from_yaml(config_path)
    pipeline = Pipeline(config)
    return pipeline.run()


def quick_distance(
    data_dir: str,
    output_dir: str = "output",
    molecule_featurizer: str = "ecfp",
    molecule_method: str = "euclidean",
    n_jobs: int = 8,
) -> Dict[str, DistanceMatrix]:
    """Quick distance computation with minimal configuration.

    Args:
        data_dir: Path to data directory with train/test folders.
        output_dir: Path to output directory.
        molecule_featurizer: Molecule featurizer name.
        molecule_method: Distance method.
        n_jobs: Number of parallel jobs.

    Returns:
        Dictionary of distance matrices.
    """
    from ..config import (
        ComputeConfig,
        DataConfig,
        MoleculeDistanceConfig,
        OutputConfig,
        PipelineConfig,
    )

    config = PipelineConfig(
        data=DataConfig(directory=Path(data_dir)),
        molecule=MoleculeDistanceConfig(
            enabled=True,
            featurizer=molecule_featurizer,
            method=molecule_method,
        ),
        output=OutputConfig(directory=Path(output_dir)),
        compute=ComputeConfig(n_jobs=n_jobs),
    )

    pipeline = Pipeline(config)
    return pipeline.run()
