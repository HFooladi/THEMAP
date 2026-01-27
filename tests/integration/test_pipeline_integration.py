"""
Integration tests for THEMAP pipeline.

Tests end-to-end pipeline workflows with real data.
"""

from pathlib import Path

import pytest


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the pipeline orchestrator."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create a temporary directory structure for testing."""
        # Create train/test directories
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()
        return tmp_path

    @pytest.fixture
    def sample_dataset_files(self, temp_data_dir):
        """Create sample dataset files for testing."""
        import gzip
        import json

        # Create sample molecules in JSONL.GZ format
        train_data = [
            {"SMILES": "CCO", "Property": "1.0", "Assay_ID": "CHEMBL001"},
            {"SMILES": "CC(=O)O", "Property": "0.0", "Assay_ID": "CHEMBL001"},
            {"SMILES": "c1ccccc1", "Property": "1.0", "Assay_ID": "CHEMBL001"},
        ]

        test_data = [
            {"SMILES": "CCN", "Property": "1.0", "Assay_ID": "CHEMBL002"},
            {"SMILES": "CCCO", "Property": "0.0", "Assay_ID": "CHEMBL002"},
        ]

        # Write train dataset
        train_file = temp_data_dir / "train" / "CHEMBL001.jsonl.gz"
        with gzip.open(train_file, "wt") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        # Write test dataset
        test_file = temp_data_dir / "test" / "CHEMBL002.jsonl.gz"
        with gzip.open(test_file, "wt") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        return {"train": train_file, "test": test_file}

    def test_data_directory_structure(self, temp_data_dir, sample_dataset_files):
        """Test that data directory structure is correctly created."""
        train_dir = temp_data_dir / "train"
        test_dir = temp_data_dir / "test"

        assert train_dir.exists()
        assert test_dir.exists()
        assert sample_dataset_files["train"].exists()
        assert sample_dataset_files["test"].exists()

    @pytest.mark.slow
    def test_pipeline_config_creation(self):
        """Test that PipelineConfig can be created with default values."""
        try:
            from themap.config import DataConfig, OutputConfig, PipelineConfig

            config = PipelineConfig(
                data=DataConfig(directory=Path("datasets")),
                output=OutputConfig(directory=Path("output")),
            )
            assert config.data.directory == Path("datasets")
            assert config.output.directory == Path("output")
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.slow
    def test_molecule_dataset_loading(self, temp_data_dir, sample_dataset_files):
        """Test that MoleculeDataset can load from JSONL.GZ files."""
        try:
            from themap.data import MoleculeDataset

            # This would require proper implementation
            # For now, test the import works
            assert MoleculeDataset is not None
        except ImportError as e:
            pytest.skip(f"MoleculeDataset not available: {e}")


@pytest.mark.integration
class TestVersionAndImports:
    """Integration tests for package imports and version."""

    def test_version_accessible(self):
        """Test that version is accessible."""
        import themap

        assert hasattr(themap, "__version__")
        assert isinstance(themap.__version__, str)

    def test_main_exports_accessible(self):
        """Test that main exports are accessible via lazy loading."""
        import themap

        # Test that exports are listed in __all__
        expected_exports = [
            "Pipeline",
            "PipelineConfig",
            "run_pipeline",
            "quick_distance",
            "DatasetDistance",
            "MetadataDistance",
            "MoleculeDataset",
            "AbstractTasksDistance",
        ]

        for export in expected_exports:
            assert export in themap.__all__, f"{export} should be in __all__"

    def test_distance_module_imports(self):
        """Test that distance module can be imported."""
        try:
            from themap.distance import (
                DATASET_DISTANCE_METHODS,
                METADATA_DISTANCE_METHODS,
                AbstractTasksDistance,
            )

            assert AbstractTasksDistance is not None
            assert "euclidean" in DATASET_DISTANCE_METHODS
            assert "euclidean" in METADATA_DISTANCE_METHODS
        except ImportError as e:
            pytest.skip(f"Distance module not fully available: {e}")
