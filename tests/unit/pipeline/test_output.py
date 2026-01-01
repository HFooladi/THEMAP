"""
Tests for pipeline output management.

This module tests the output manager functionality for saving pipeline results
in various formats.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from themap.pipeline.config import OutputConfig
from themap.pipeline.output import OutputManager


@pytest.mark.unit
class TestOutputManager:
    """Test OutputManager class."""

    @pytest.fixture
    def output_config(self):
        """Create test output configuration."""
        return OutputConfig(
            directory="test_results", formats=["json", "csv"], save_intermediate=True, save_matrices=True
        )

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def output_manager(self, output_config, temp_dir):
        """Create OutputManager instance for testing."""
        # Override directory with temp directory
        output_config.directory = str(temp_dir / "test_results")
        return OutputManager(output_config)

    @pytest.fixture
    def sample_results(self):
        """Sample pipeline results for testing."""
        return {
            "config": {"name": "test_pipeline", "description": "Test pipeline"},
            "distance_results": [
                {
                    "source_dataset": "CHEMBL123456",
                    "target_dataset": "CHEMBL789012",
                    "modality": "molecule",
                    "method": "euclidean",
                    "distance": 0.75,
                    "computation_time": 1.23,
                },
                {
                    "source_dataset": "CHEMBL123456",
                    "target_dataset": "CHEMBL345678",
                    "modality": "protein",
                    "method": "cosine",
                    "distance": 0.42,
                    "computation_time": 2.34,
                },
            ],
            "runtime_seconds": 45.67,
            "errors": [],
        }

    def test_output_manager_initialization(self, output_manager, temp_dir):
        """Test OutputManager initialization."""
        assert output_manager.output_dir.exists()
        assert str(output_manager.output_dir).endswith("test_results")
        assert len(output_manager.generated_files) == 0

    def test_save_json_results(self, output_manager, sample_results):
        """Test saving results in JSON format."""
        saved_files = output_manager.save_results(sample_results, "test_results")

        assert "json" in saved_files
        json_file = saved_files["json"]
        assert json_file.exists()
        assert json_file.suffix == ".json"

        # Verify content
        with open(json_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["config"]["name"] == "test_pipeline"
        assert len(loaded_data["distance_results"]) == 2
        assert loaded_data["runtime_seconds"] == 45.67

    def test_save_csv_results(self, output_manager, sample_results):
        """Test saving results in CSV format."""
        saved_files = output_manager.save_results(sample_results, "test_results")

        assert "csv" in saved_files
        csv_file = saved_files["csv"]
        assert csv_file.exists()
        assert csv_file.suffix == ".csv"

        # Verify content
        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert "source_dataset" in df.columns
        assert "target_dataset" in df.columns
        assert "distance" in df.columns

    def test_save_pickle_results(self, output_manager, sample_results):
        """Test saving results in pickle format."""
        # Add pickle to formats
        output_manager.config.formats.append("pickle")

        saved_files = output_manager.save_results(sample_results, "test_results")

        assert "pickle" in saved_files
        pickle_file = saved_files["pickle"]
        assert pickle_file.exists()
        assert pickle_file.suffix == ".pickle"  # save_results uses .pickle extension

        # Verify content
        with open(pickle_file, "rb") as f:
            loaded_data = pickle.load(f)

        assert loaded_data["config"]["name"] == "test_pipeline"
        assert len(loaded_data["distance_results"]) == 2

    def test_save_distance_matrix(self, output_manager):
        """Test saving distance matrix."""
        matrix = np.array([[0.0, 0.5, 0.8], [0.5, 0.0, 0.3], [0.8, 0.3, 0.0]])
        row_labels = ["A", "B", "C"]
        col_labels = ["A", "B", "C"]

        saved_files = output_manager.save_distance_matrix(matrix, row_labels, col_labels, "test_matrix")

        assert "json" in saved_files
        assert "csv" in saved_files

        # Verify JSON content
        json_file = saved_files["json"]
        with open(json_file, "r") as f:
            data = json.load(f)

        assert data["shape"] == [3, 3]
        assert data["row_labels"] == row_labels
        assert data["col_labels"] == col_labels

        # Verify CSV content
        csv_file = saved_files["csv"]
        df = pd.read_csv(csv_file, index_col=0)
        assert df.shape == (3, 3)
        assert list(df.index) == row_labels
        assert list(df.columns) == col_labels

    def test_save_distance_matrix_disabled(self, output_manager):
        """Test distance matrix saving when disabled."""
        output_manager.config.save_matrices = False

        matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
        row_labels = ["A", "B"]
        col_labels = ["A", "B"]

        saved_files = output_manager.save_distance_matrix(matrix, row_labels, col_labels, "test_matrix")

        assert len(saved_files) == 0

    def test_save_intermediate_results(self, output_manager):
        """Test saving intermediate results."""
        intermediate_data = {"stage": "molecule_distance", "dataset": "CHEMBL123456", "progress": 0.5}

        file_path = output_manager.save_intermediate_results(intermediate_data, "test_stage", "dataset123")

        assert file_path is not None
        assert file_path.exists()
        assert "intermediate_test_stage_dataset123" in file_path.name

        # Verify content
        with open(file_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["stage"] == "molecule_distance"
        assert loaded_data["dataset"] == "CHEMBL123456"

    def test_save_intermediate_results_disabled(self, output_manager):
        """Test intermediate results saving when disabled."""
        output_manager.config.save_intermediate = False

        intermediate_data = {"test": "data"}

        file_path = output_manager.save_intermediate_results(intermediate_data, "test_stage")

        assert file_path is None

    def test_create_summary_report(self, output_manager, sample_results):
        """Test creating summary report."""
        # Add datasets_info to sample results
        sample_results["datasets_info"] = [
            {"name": "CHEMBL123456", "modalities": ["molecule", "protein"], "size": 1000},
            {"name": "CHEMBL789012", "modalities": ["molecule"], "size": 500},
        ]

        summary_file = output_manager.create_summary_report(sample_results)

        assert summary_file.exists()
        assert "pipeline_summary" in summary_file.name

        # Verify content
        with open(summary_file, "r") as f:
            summary = json.load(f)

        assert "pipeline_name" in summary
        assert "execution_timestamp" in summary
        assert "datasets_processed" in summary
        assert "distance_computations" in summary
        assert summary["datasets_processed"]["total_datasets"] == 2
        assert summary["distance_computations"]["total_computations"] == 2

    def test_json_serializer_numpy_arrays(self, output_manager):
        """Test JSON serialization of numpy arrays."""
        data_with_numpy = {
            "array": np.array([1, 2, 3]),
            "float": np.float64(3.14),
            "int": np.int32(42),
            "path": Path("/test/path"),
        }

        saved_files = output_manager.save_results(data_with_numpy, "numpy_test")
        json_file = saved_files["json"]

        # Verify it can be loaded back
        with open(json_file, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["array"] == [1, 2, 3]
        assert loaded_data["float"] == 3.14
        assert loaded_data["int"] == 42
        assert loaded_data["path"] == "/test/path"

    def test_csv_with_complex_data(self, output_manager):
        """Test CSV with complex nested data uses json_normalize."""
        # Create nested data - json_normalize can handle this
        complex_data = {"nested": {"deeply": {"nested": "data"}}, "list_of_dicts": [{"a": 1}, {"b": 2}]}

        saved_files = output_manager.save_results(complex_data, "complex_test")

        # json_normalize successfully handles nested dicts, so CSV is created
        assert "csv" in saved_files
        csv_file = saved_files["csv"]
        assert csv_file.suffix == ".csv"  # json_normalize handles this

    def test_extract_dataset_summary(self, output_manager):
        """Test dataset summary extraction."""
        results = {
            "datasets_info": [
                {"modalities": ["molecule"]},
                {"modalities": ["protein"]},
                {"modalities": ["molecule", "protein"]},
                {"modalities": ["metadata"]},
            ]
        }

        summary = output_manager._extract_dataset_summary(results)

        assert summary["total_datasets"] == 4
        assert summary["molecule_datasets"] == 2
        assert summary["protein_datasets"] == 2
        assert summary["metadata_datasets"] == 1

    def test_extract_distance_summary(self, output_manager):
        """Test distance computation summary extraction."""
        results = {
            "distance_results": [
                {"modality": "molecule", "method": "euclidean", "computation_time": 1.0},
                {"modality": "molecule", "method": "cosine", "computation_time": 2.0},
                {"modality": "protein", "method": "euclidean", "computation_time": 3.0},
            ]
        }

        summary = output_manager._extract_distance_summary(results)

        assert summary["total_computations"] == 3
        assert summary["by_modality"]["molecule"] == 2
        assert summary["by_modality"]["protein"] == 1
        assert summary["by_method"]["euclidean"] == 2
        assert summary["by_method"]["cosine"] == 1
        assert summary["average_computation_time"] == 2.0

    def test_generated_files_tracking(self, output_manager, sample_results):
        """Test that generated files are tracked."""
        initial_count = len(output_manager.generated_files)

        saved_files = output_manager.save_results(sample_results, "test")

        # Should track all generated files
        assert len(output_manager.generated_files) > initial_count

        for file_path in saved_files.values():
            assert file_path in output_manager.generated_files

    def test_unsupported_format_error(self, output_config, temp_dir):
        """Test error handling for unsupported formats."""
        output_config.directory = str(temp_dir)
        output_config.formats = ["unsupported_format"]

        output_manager = OutputManager(output_config)

        with pytest.raises(ValueError, match="Unsupported format"):
            output_manager.save_results({"test": "data"}, "test")
