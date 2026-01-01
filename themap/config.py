"""
Configuration system for THEMAP pipeline.

This module provides YAML-based configuration for the distance computation pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .utils.logging import get_logger

logger = get_logger(__name__)

# Available featurizers
MOLECULE_FEATURIZERS = [
    # Fingerprints (fast)
    "ecfp",
    "fcfp",
    "maccs",
    "avalon",
    "topological",
    "atompair",
    # Descriptors (medium)
    "desc2D",
    "desc3D",
    "mordred",
    # Neural embeddings (slow, requires GPU)
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "MolT5",
    "gin_supervised_infomax",
    "gin_supervised_contextpred",
]

PROTEIN_FEATURIZERS = [
    "esm2_t12_35M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]

DISTANCE_METHODS = ["euclidean", "cosine", "manhattan"]
MOLECULE_DISTANCE_METHODS = DISTANCE_METHODS + ["otdd"]  # OTDD only for molecules

COMBINATION_STRATEGIES = ["average", "weighted_average", "separate"]


@dataclass
class MoleculeDistanceConfig:
    """Configuration for molecule-based distance computation."""

    enabled: bool = True
    featurizer: str = "ecfp"
    method: str = "euclidean"

    def __post_init__(self):
        if self.featurizer not in MOLECULE_FEATURIZERS:
            logger.warning(
                f"Featurizer '{self.featurizer}' not in known list. Available: {MOLECULE_FEATURIZERS}"
            )
        if self.method not in MOLECULE_DISTANCE_METHODS:
            raise ValueError(
                f"Unknown distance method '{self.method}'. Available: {MOLECULE_DISTANCE_METHODS}"
            )


@dataclass
class ProteinDistanceConfig:
    """Configuration for protein-based distance computation."""

    enabled: bool = False
    featurizer: str = "esm2_t33_650M_UR50D"
    method: str = "cosine"
    layer: Optional[int] = None  # Auto-detect based on model

    def __post_init__(self):
        if self.featurizer not in PROTEIN_FEATURIZERS:
            logger.warning(
                f"Protein featurizer '{self.featurizer}' not in known list. Available: {PROTEIN_FEATURIZERS}"
            )
        if self.method not in DISTANCE_METHODS:
            raise ValueError(f"Unknown distance method '{self.method}'. Available: {DISTANCE_METHODS}")


@dataclass
class CombinationConfig:
    """Configuration for combining multiple distance matrices."""

    strategy: str = "weighted_average"
    weights: Dict[str, float] = field(default_factory=lambda: {"molecule": 0.5, "protein": 0.5})

    def __post_init__(self):
        if self.strategy not in COMBINATION_STRATEGIES:
            raise ValueError(
                f"Unknown combination strategy '{self.strategy}'. Available: {COMBINATION_STRATEGIES}"
            )
        # Normalize weights
        if self.strategy == "weighted_average" and self.weights:
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}


@dataclass
class OutputConfig:
    """Configuration for output files."""

    directory: Path = field(default_factory=lambda: Path("output"))
    format: str = "csv"  # csv, json, npz
    save_features: bool = True

    def __post_init__(self):
        if isinstance(self.directory, str):
            self.directory = Path(self.directory)
        if self.format not in ["csv", "json", "npz"]:
            raise ValueError(f"Unknown output format '{self.format}'")


@dataclass
class ComputeConfig:
    """Configuration for computation settings."""

    n_jobs: int = 8
    batch_size: int = 1000
    device: str = "auto"  # auto, cpu, cuda

    def __post_init__(self):
        if self.n_jobs < 1:
            self.n_jobs = 1
        if self.batch_size < 1:
            self.batch_size = 1
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Unknown device '{self.device}'")


@dataclass
class DataConfig:
    """Configuration for data loading."""

    directory: Path = field(default_factory=lambda: Path("datasets"))
    task_list: Optional[str] = None  # If None, auto-discover all files

    def __post_init__(self):
        if isinstance(self.directory, str):
            self.directory = Path(self.directory)


@dataclass
class PipelineConfig:
    """Main configuration for the THEMAP pipeline.

    Example YAML:
    ```yaml
    data:
      directory: "datasets/TDC"
      task_list: "tasks.json"  # Optional

    distances:
      molecule:
        enabled: true
        featurizer: "ecfp"
        method: "euclidean"
      protein:
        enabled: false
        featurizer: "esm2_t33_650M_UR50D"
        method: "cosine"

    combination:
      strategy: "weighted_average"
      weights:
        molecule: 0.7
        protein: 0.3

    output:
      directory: "output/"
      save_features: true

    compute:
      n_jobs: 8
      batch_size: 1000
    ```
    """

    data: DataConfig = field(default_factory=DataConfig)
    molecule: MoleculeDistanceConfig = field(default_factory=MoleculeDistanceConfig)
    protein: ProteinDistanceConfig = field(default_factory=ProteinDistanceConfig)
    combination: CombinationConfig = field(default_factory=CombinationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            PipelineConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            PipelineConfig instance.
        """
        # Parse nested configs
        data_config = DataConfig(**config_dict.get("data", {}))

        # Handle nested 'distances' key
        distances = config_dict.get("distances", {})
        molecule_config = MoleculeDistanceConfig(**distances.get("molecule", {}))
        protein_config = ProteinDistanceConfig(**distances.get("protein", {}))

        combination_config = CombinationConfig(**config_dict.get("combination", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))
        compute_config = ComputeConfig(**config_dict.get("compute", {}))

        return cls(
            data=data_config,
            molecule=molecule_config,
            protein=protein_config,
            combination=combination_config,
            output=output_config,
            compute=compute_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": {
                "directory": str(self.data.directory),
                "task_list": self.data.task_list,
            },
            "distances": {
                "molecule": {
                    "enabled": self.molecule.enabled,
                    "featurizer": self.molecule.featurizer,
                    "method": self.molecule.method,
                },
                "protein": {
                    "enabled": self.protein.enabled,
                    "featurizer": self.protein.featurizer,
                    "method": self.protein.method,
                    "layer": self.protein.layer,
                },
            },
            "combination": {
                "strategy": self.combination.strategy,
                "weights": self.combination.weights,
            },
            "output": {
                "directory": str(self.output.directory),
                "format": self.output.format,
                "save_features": self.output.save_features,
            },
            "compute": {
                "n_jobs": self.compute.n_jobs,
                "batch_size": self.compute.batch_size,
                "device": self.compute.device,
            },
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors.

        Returns:
            List of warning/error messages.
        """
        issues = []

        # Check data directory
        if not self.data.directory.exists():
            issues.append(f"Data directory does not exist: {self.data.directory}")

        # Check train/test directories
        train_dir = self.data.directory / "train"
        test_dir = self.data.directory / "test"

        if not train_dir.exists():
            issues.append(f"Train directory does not exist: {train_dir}")
        if not test_dir.exists():
            issues.append(f"Test directory does not exist: {test_dir}")

        # Check task list file if specified
        if self.data.task_list:
            task_list_path = self.data.directory / self.data.task_list
            if not task_list_path.exists():
                issues.append(
                    f"Task list file does not exist: {task_list_path}. Will auto-discover all files."
                )

        # Check if at least one distance type is enabled
        if not self.molecule.enabled and not self.protein.enabled:
            issues.append("At least one distance type (molecule or protein) must be enabled")

        # Check protein directory if protein is enabled
        if self.protein.enabled:
            proteins_dir = self.data.directory / "proteins"
            if not proteins_dir.exists():
                issues.append(
                    f"Protein directory does not exist: {proteins_dir}. "
                    "Protein distance computation may fail."
                )

        return issues


def create_default_config(output_path: Union[str, Path] = "config.yaml") -> PipelineConfig:
    """Create and save a default configuration file.

    Args:
        output_path: Path to save the configuration file.

    Returns:
        Default PipelineConfig instance.
    """
    config = PipelineConfig()
    config.to_yaml(output_path)
    return config
