"""
Tests for pipeline configuration system.

This module tests the configuration schema, validation, and parsing functionality
for the benchmarking pipeline system.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from themap.pipeline.config import (
    DatasetConfig,
    MoleculeConfig,
    OutputConfig,
    PipelineConfig,
    ProteinConfig,
    TaskDistanceConfig,
)


@pytest.mark.unit
class TestDatasetConfig:
    """Test DatasetConfig class."""

    def test_dataset_config_creation(self):
        """Test basic dataset config creation."""
        config = DatasetConfig(name="CHEMBL123456", source_fold="TRAIN", target_folds=["TEST"])

        assert config.name == "CHEMBL123456"
        assert config.source_fold == "TRAIN"
        assert config.target_folds == ["TEST"]
        assert config.path is None

    def test_dataset_config_with_path(self):
        """Test dataset config with custom path."""
        config = DatasetConfig(
            name="CHEMBL123456",
            source_fold="TRAIN",
            target_folds=["TEST", "VALIDATION"],
            path="/custom/path/data.jsonl.gz",
        )

        assert config.path == "/custom/path/data.jsonl.gz"
        assert config.target_folds == ["TEST", "VALIDATION"]


@pytest.mark.unit
class TestMoleculeConfig:
    """Test MoleculeConfig class."""

    def test_molecule_config_creation(self):
        """Test basic molecule config creation."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        config = MoleculeConfig(datasets=datasets, featurizers=["ecfp"], distance_methods=["euclidean"])

        assert len(config.datasets) == 1
        assert config.featurizers == ["ecfp"]
        assert config.distance_methods == ["euclidean"]

    def test_molecule_config_defaults(self):
        """Test molecule config with default values."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        config = MoleculeConfig(datasets=datasets)

        assert config.featurizers == ["ecfp"]
        assert config.distance_methods == ["euclidean"]

    def test_molecule_config_invalid_featurizer(self):
        """Test molecule config with invalid featurizer."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        with pytest.raises(ValueError, match="Unknown featurizer"):
            MoleculeConfig(datasets=datasets, featurizers=["invalid_featurizer"])

    def test_molecule_config_invalid_distance_method(self):
        """Test molecule config with invalid distance method."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        with pytest.raises(ValueError, match="Unknown distance method"):
            MoleculeConfig(datasets=datasets, distance_methods=["invalid_method"])


@pytest.mark.unit
class TestProteinConfig:
    """Test ProteinConfig class."""

    def test_protein_config_creation(self):
        """Test basic protein config creation."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        config = ProteinConfig(datasets=datasets, featurizers=["esm"], distance_methods=["euclidean"])

        assert len(config.datasets) == 1
        assert config.featurizers == ["esm"]
        assert config.distance_methods == ["euclidean"]

    def test_protein_config_defaults(self):
        """Test protein config with default values."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        config = ProteinConfig(datasets=datasets)

        assert config.featurizers == ["esm"]
        assert config.distance_methods == ["euclidean"]

    def test_protein_config_invalid_featurizer(self):
        """Test protein config with invalid featurizer."""
        datasets = [DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])]

        with pytest.raises(ValueError, match="Unknown protein featurizer"):
            ProteinConfig(datasets=datasets, featurizers=["invalid_featurizer"])


@pytest.mark.unit
class TestTaskDistanceConfig:
    """Test TaskDistanceConfig class."""

    def test_task_distance_config_defaults(self):
        """Test task distance config with defaults."""
        config = TaskDistanceConfig()

        assert config.combination_strategy == "weighted_average"
        assert config.weights["molecule"] == 1.0
        assert config.weights["protein"] == 1.0
        assert config.weights["metadata"] == 0.0

    def test_task_distance_config_custom(self):
        """Test task distance config with custom values."""
        config = TaskDistanceConfig(
            combination_strategy="concatenation", weights={"molecule": 0.6, "protein": 0.4}
        )

        assert config.combination_strategy == "concatenation"
        assert config.weights["molecule"] == 0.6
        assert config.weights["protein"] == 0.4

    def test_task_distance_config_invalid_strategy(self):
        """Test task distance config with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown combination strategy"):
            TaskDistanceConfig(combination_strategy="invalid_strategy")


@pytest.mark.unit
class TestOutputConfig:
    """Test OutputConfig class."""

    def test_output_config_defaults(self):
        """Test output config with defaults."""
        config = OutputConfig()

        assert config.directory == "pipeline_results"
        assert config.formats == ["json", "csv"]
        assert config.save_intermediate is True
        assert config.save_matrices is False

    def test_output_config_custom(self):
        """Test output config with custom values."""
        config = OutputConfig(
            directory="custom_results",
            formats=["json", "parquet"],
            save_intermediate=False,
            save_matrices=True,
        )

        assert config.directory == "custom_results"
        assert config.formats == ["json", "parquet"]
        assert config.save_intermediate is False
        assert config.save_matrices is True

    def test_output_config_invalid_format(self):
        """Test output config with invalid format."""
        with pytest.raises(ValueError, match="Unknown output format"):
            OutputConfig(formats=["invalid_format"])


@pytest.mark.unit
class TestPipelineConfig:
    """Test main PipelineConfig class."""

    def test_pipeline_config_minimal(self):
        """Test minimal pipeline config."""
        molecule_config = MoleculeConfig(datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])])

        config = PipelineConfig(name="test_pipeline", molecule=molecule_config)

        assert config.name == "test_pipeline"
        assert config.molecule is not None
        assert config.protein is None
        assert config.task_distance is None  # No task distance for single modality

    def test_pipeline_config_multimodal(self):
        """Test multimodal pipeline config."""
        molecule_config = MoleculeConfig(datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])])
        protein_config = ProteinConfig(datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])])

        config = PipelineConfig(name="test_pipeline", molecule=molecule_config, protein=protein_config)

        assert config.task_distance is not None  # Auto-created for multiple modalities
        assert config.task_distance.combination_strategy == "weighted_average"

    def test_pipeline_config_no_modalities(self):
        """Test pipeline config with no modalities."""
        with pytest.raises(
            ValueError, match="At least one of molecule, protein, or metadata must be specified"
        ):
            PipelineConfig(name="test_pipeline")

    def test_pipeline_config_from_dict(self):
        """Test creating pipeline config from dictionary."""
        config_dict = {
            "name": "test_pipeline",
            "description": "Test pipeline",
            "molecule": {
                "datasets": [{"name": "CHEMBL123456", "source_fold": "TRAIN", "target_folds": ["TEST"]}],
                "featurizers": ["ecfp"],
                "distance_methods": ["euclidean"],
            },
            "output": {"directory": "test_results", "formats": ["json"]},
        }

        config = PipelineConfig.from_dict(config_dict)

        assert config.name == "test_pipeline"
        assert config.description == "Test pipeline"
        assert config.molecule is not None
        assert len(config.molecule.datasets) == 1
        assert config.output.directory == "test_results"

    def test_pipeline_config_to_dict(self):
        """Test converting pipeline config to dictionary."""
        molecule_config = MoleculeConfig(datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])])

        config = PipelineConfig(name="test_pipeline", description="Test pipeline", molecule=molecule_config)

        config_dict = config.to_dict()

        assert config_dict["name"] == "test_pipeline"
        assert config_dict["description"] == "Test pipeline"
        assert "molecule" in config_dict
        assert config_dict["molecule"]["datasets"][0]["name"] == "CHEMBL123456"


@pytest.mark.unit
class TestConfigFileOperations:
    """Test file operations for pipeline configs."""

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        molecule_config = MoleculeConfig(datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])])

        config = PipelineConfig(name="test_pipeline", description="Test pipeline", molecule=molecule_config)

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            # Save config
            config.save(config_path)

            # Load config
            loaded_config = PipelineConfig.from_file(config_path)

            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
            assert loaded_config.molecule is not None
            assert len(loaded_config.molecule.datasets) == 1
            assert loaded_config.molecule.datasets[0].name == "CHEMBL123456"

        finally:
            config_path.unlink()

    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        molecule_config = MoleculeConfig(datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])])

        config = PipelineConfig(name="test_pipeline", molecule=molecule_config)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config_path = Path(f.name)

        try:
            # Save config
            config.save(config_path)

            # Load config
            loaded_config = PipelineConfig.from_file(config_path)

            assert loaded_config.name == config.name
            assert loaded_config.molecule is not None

        finally:
            config_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_file("nonexistent.yaml")

    def test_load_invalid_format(self):
        """Test loading config with invalid format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            config_path = Path(f.name)
            f.write(b"invalid content")

        try:
            with pytest.raises(ValueError, match="Unsupported config format"):
                PipelineConfig.from_file(config_path)
        finally:
            config_path.unlink()

    @pytest.fixture
    def sample_config_dict(self) -> Dict[str, Any]:
        """Sample configuration dictionary for testing."""
        return {
            "name": "test_pipeline",
            "description": "Test configuration",
            "molecule": {
                "datasets": [
                    {"name": "CHEMBL123456", "source_fold": "TRAIN", "target_folds": ["TEST"]},
                    {"name": "CHEMBL789012", "source_fold": "TRAIN", "target_folds": ["TEST", "VALIDATION"]},
                ],
                "featurizers": ["ecfp", "maccs"],
                "distance_methods": ["euclidean", "cosine"],
            },
            "protein": {
                "datasets": [{"name": "CHEMBL123456", "source_fold": "TRAIN", "target_folds": ["TEST"]}],
                "featurizers": ["esm"],
                "distance_methods": ["euclidean"],
            },
            "task_distance": {
                "combination_strategy": "weighted_average",
                "weights": {"molecule": 0.7, "protein": 0.3},
            },
            "output": {
                "directory": "test_results",
                "formats": ["json", "csv"],
                "save_intermediate": True,
                "save_matrices": False,
            },
            "compute": {"max_workers": 4, "cache_features": True, "sample_size": 1000, "seed": 42},
        }

    def test_complex_config_from_dict(self, sample_config_dict):
        """Test creating complex config from dictionary."""
        config = PipelineConfig.from_dict(sample_config_dict)

        # Test basic properties
        assert config.name == "test_pipeline"
        assert config.description == "Test configuration"

        # Test molecule config
        assert config.molecule is not None
        assert len(config.molecule.datasets) == 2
        assert config.molecule.featurizers == ["ecfp", "maccs"]
        assert config.molecule.distance_methods == ["euclidean", "cosine"]

        # Test protein config
        assert config.protein is not None
        assert len(config.protein.datasets) == 1
        assert config.protein.featurizers == ["esm"]

        # Test task distance config
        assert config.task_distance is not None
        assert config.task_distance.combination_strategy == "weighted_average"
        assert config.task_distance.weights["molecule"] == 0.7
        assert config.task_distance.weights["protein"] == 0.3

        # Test output config
        assert config.output.directory == "test_results"
        assert config.output.formats == ["json", "csv"]

        # Test compute config
        assert config.compute.max_workers == 4
        assert config.compute.sample_size == 1000
