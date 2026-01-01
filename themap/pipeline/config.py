"""
Configuration schema and validation for benchmarking pipelines.

This module defines the configuration structure for distance computation pipelines,
including dataset specifications, featurizers, distance methods, and output formats.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    source_fold: str  # 'TRAIN', 'TEST', 'VALIDATION'
    target_folds: List[str]  # ['TEST'], ['VALIDATION', 'TEST'], etc.
    path: Optional[str] = None  # Override default path if needed


@dataclass
class DirectoryConfig:
    """Configuration for directory-based dataset discovery."""

    root_path: str
    task_list_file: str
    load_molecules: bool = True
    load_proteins: bool = True
    load_metadata: bool = True
    metadata_types: Optional[List[str]] = None


@dataclass
class MoleculeConfig:
    """Configuration for molecular data processing."""

    datasets: Optional[List[DatasetConfig]] = None
    directory: Optional[DirectoryConfig] = None
    featurizers: List[str] = field(default_factory=lambda: ["ecfp"])
    distance_methods: List[str] = field(default_factory=lambda: ["euclidean"])

    def __post_init__(self):
        """Validate molecular configuration."""
        from ..distance.molecule_distance import MOLECULE_DISTANCE_METHODS
        from ..utils.featurizer_utils import AVAILABLE_FEATURIZERS

        # Ensure exactly one of datasets or directory is specified
        if self.datasets is not None and self.directory is not None:
            raise ValueError(
                "Cannot specify both 'datasets' and 'directory' for molecules. Choose one approach."
            )
        if self.datasets is None and self.directory is None:
            raise ValueError("Must specify either 'datasets' or 'directory' for molecules.")

        for featurizer in self.featurizers:
            if featurizer not in AVAILABLE_FEATURIZERS:
                raise ValueError(f"Unknown featurizer: {featurizer}. Available: {AVAILABLE_FEATURIZERS}")

        for method in self.distance_methods:
            if method not in MOLECULE_DISTANCE_METHODS:
                raise ValueError(f"Unknown distance method: {method}. Available: {MOLECULE_DISTANCE_METHODS}")


@dataclass
class ProteinConfig:
    """Configuration for protein data processing."""

    datasets: Optional[List[DatasetConfig]] = None
    directory: Optional[DirectoryConfig] = None
    featurizers: List[str] = field(default_factory=lambda: ["esm"])
    distance_methods: List[str] = field(default_factory=lambda: ["euclidean"])

    def __post_init__(self):
        """Validate protein configuration."""
        from ..distance.protein_distance import PROTEIN_DISTANCE_METHODS

        # Ensure exactly one of datasets or directory is specified
        if self.datasets is not None and self.directory is not None:
            raise ValueError(
                "Cannot specify both 'datasets' and 'directory' for proteins. Choose one approach."
            )
        if self.datasets is None and self.directory is None:
            raise ValueError("Must specify either 'datasets' or 'directory' for proteins.")

        # Available protein featurizers (ESM embeddings)
        available_protein_featurizers = ["esm", "esm2"]

        for featurizer in self.featurizers:
            if featurizer not in available_protein_featurizers:
                raise ValueError(
                    f"Unknown protein featurizer: {featurizer}. Available: {available_protein_featurizers}"
                )

        for method in self.distance_methods:
            if method not in PROTEIN_DISTANCE_METHODS:
                raise ValueError(f"Unknown distance method: {method}. Available: {PROTEIN_DISTANCE_METHODS}")


@dataclass
class MetadataConfig:
    """Configuration for metadata processing."""

    datasets: Optional[List[DatasetConfig]] = None
    directory: Optional[DirectoryConfig] = None
    features: List[str] = field(default_factory=list)  # Custom metadata features
    distance_methods: List[str] = field(default_factory=lambda: ["euclidean"])

    def __post_init__(self):
        """Validate metadata configuration."""
        # For metadata, both datasets and directory can be None (optional component)
        if self.datasets is not None and self.directory is not None:
            raise ValueError(
                "Cannot specify both 'datasets' and 'directory' for metadata. Choose one approach."
            )


@dataclass
class TaskDistanceConfig:
    """Configuration for combined task distance computation."""

    combination_strategy: str = "weighted_average"  # 'weighted_average', 'concatenation'
    weights: Dict[str, float] = field(
        default_factory=lambda: {"molecule": 1.0, "protein": 1.0, "metadata": 0.0}
    )

    def __post_init__(self):
        """Validate task distance configuration."""
        valid_strategies = ["weighted_average", "concatenation"]
        if self.combination_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown combination strategy: {self.combination_strategy}. Available: {valid_strategies}"
            )


@dataclass
class OutputConfig:
    """Configuration for output formats and locations."""

    directory: str = "pipeline_results"
    formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    save_intermediate: bool = True
    save_matrices: bool = False  # Save full distance matrices

    def __post_init__(self):
        """Validate output configuration."""
        valid_formats = ["json", "csv", "parquet", "pickle"]
        for fmt in self.formats:
            if fmt not in valid_formats:
                raise ValueError(f"Unknown output format: {fmt}. Available: {valid_formats}")


@dataclass
class ComputeConfig:
    """Configuration for computation parameters."""

    max_workers: int = 4
    cache_features: bool = True
    gpu_if_available: bool = True
    sample_size: Optional[int] = None  # Sample datasets for faster computation
    seed: int = 42


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    name: str
    description: str = ""
    molecule: Optional[MoleculeConfig] = None
    protein: Optional[ProteinConfig] = None
    metadata: Optional[MetadataConfig] = None
    task_distance: Optional[TaskDistanceConfig] = None
    output: OutputConfig = field(default_factory=OutputConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def __post_init__(self):
        """Validate pipeline configuration."""
        if not any([self.molecule, self.protein, self.metadata]):
            raise ValueError("At least one of molecule, protein, or metadata must be specified")

        # Set default task distance config if multiple modalities
        if sum(x is not None for x in [self.molecule, self.protein, self.metadata]) > 1:
            if self.task_distance is None:
                self.task_distance = TaskDistanceConfig()

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        # Convert nested dictionaries to dataclass instances
        config_data = data.copy()

        # Handle molecule config
        if "molecule" in config_data and config_data["molecule"]:
            molecule_data = config_data["molecule"]
            datasets = None
            directory = None

            if "datasets" in molecule_data and molecule_data["datasets"]:
                datasets = [DatasetConfig(**ds) for ds in molecule_data["datasets"]]

            if "directory" in molecule_data and molecule_data["directory"] is not None:
                directory = DirectoryConfig(**molecule_data["directory"])

            config_data["molecule"] = MoleculeConfig(
                datasets=datasets,
                directory=directory,
                featurizers=molecule_data.get("featurizers", ["ecfp"]),
                distance_methods=molecule_data.get("distance_methods", ["euclidean"]),
            )

        # Handle protein config
        if "protein" in config_data and config_data["protein"]:
            protein_data = config_data["protein"]
            datasets = None
            directory = None

            if "datasets" in protein_data and protein_data["datasets"]:
                datasets = [DatasetConfig(**ds) for ds in protein_data["datasets"]]

            if "directory" in protein_data and protein_data["directory"] is not None:
                directory = DirectoryConfig(**protein_data["directory"])

            config_data["protein"] = ProteinConfig(
                datasets=datasets,
                directory=directory,
                featurizers=protein_data.get("featurizers", ["esm"]),
                distance_methods=protein_data.get("distance_methods", ["euclidean"]),
            )

        # Handle metadata config
        if "metadata" in config_data and config_data["metadata"]:
            metadata_data = config_data["metadata"]
            datasets = None
            directory = None

            if "datasets" in metadata_data:
                datasets = [DatasetConfig(**ds) for ds in metadata_data["datasets"]]

            if "directory" in metadata_data:
                directory = DirectoryConfig(**metadata_data["directory"])

            config_data["metadata"] = MetadataConfig(
                datasets=datasets,
                directory=directory,
                features=metadata_data.get("features", []),
                distance_methods=metadata_data.get("distance_methods", ["euclidean"]),
            )

        # Handle task distance config
        if "task_distance" in config_data and config_data["task_distance"]:
            config_data["task_distance"] = TaskDistanceConfig(**config_data["task_distance"])

        # Handle output config
        if "output" in config_data:
            config_data["output"] = OutputConfig(**config_data["output"])

        # Handle compute config
        if "compute" in config_data:
            config_data["compute"] = ComputeConfig(**config_data["compute"])

        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        data = self.to_dict()

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def validate_datasets(self, base_data_path: Optional[str] = None) -> None:
        """Validate that all specified datasets exist."""
        from pathlib import Path

        if base_data_path is None:
            base_data_path = "datasets"

        base_path = Path(base_data_path)

        def check_datasets(datasets: List[DatasetConfig], data_type: str):
            for dataset in datasets:
                # Check molecule files
                if data_type == "molecule":
                    file_path = base_path / dataset.source_fold.lower() / f"{dataset.name}.jsonl.gz"
                    if not file_path.exists():
                        raise FileNotFoundError(f"Molecule dataset not found: {file_path}")

                # Check protein files
                elif data_type == "protein":
                    file_path = base_path / dataset.source_fold.lower() / f"{dataset.name}.fasta"
                    if not file_path.exists():
                        raise FileNotFoundError(f"Protein dataset not found: {file_path}")

        def check_directory_config(directory_config: DirectoryConfig):
            """Validate directory-based configuration."""
            # Resolve root path relative to base_data_path if it's relative
            root_path = Path(directory_config.root_path)
            if not root_path.is_absolute():
                root_path = base_path / root_path

            if not root_path.exists():
                raise FileNotFoundError(f"Dataset root directory not found: {root_path}")

            # Resolve task list file path
            task_list_path = Path(directory_config.task_list_file)
            if not task_list_path.is_absolute():
                task_list_path = root_path / task_list_path

            if not task_list_path.exists():
                raise FileNotFoundError(f"Task list file not found: {task_list_path}")

        # Validate molecule datasets
        if self.molecule:
            if self.molecule.datasets:
                check_datasets(self.molecule.datasets, "molecule")
            elif self.molecule.directory:
                check_directory_config(self.molecule.directory)

        # Validate protein datasets
        if self.protein:
            if self.protein.datasets:
                check_datasets(self.protein.datasets, "protein")
            elif self.protein.directory:
                check_directory_config(self.protein.directory)

        # Validate metadata datasets
        if self.metadata:
            if self.metadata.datasets:
                check_datasets(self.metadata.datasets, "metadata")
            elif self.metadata.directory:
                check_directory_config(self.metadata.directory)
