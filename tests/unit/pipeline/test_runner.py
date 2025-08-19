"""
Tests for pipeline runner functionality.

This module tests the main pipeline execution engine with mocked dependencies
to ensure proper workflow coordination.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from themap.pipeline.config import DatasetConfig, MoleculeConfig, PipelineConfig, ProteinConfig
from themap.pipeline.runner import PipelineRunner


@pytest.mark.unit
class TestPipelineRunner:
    """Test PipelineRunner class."""

    @pytest.fixture
    def molecule_config(self):
        """Create molecule configuration for testing."""
        return MoleculeConfig(
            datasets=[
                DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"]),
                DatasetConfig("CHEMBL789012", "TRAIN", ["TEST"]),
            ],
            featurizers=["ecfp"],
            distance_methods=["euclidean"],
        )

    @pytest.fixture
    def protein_config(self):
        """Create protein configuration for testing."""
        return ProteinConfig(
            datasets=[
                DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"]),
                DatasetConfig("CHEMBL789012", "TRAIN", ["TEST"]),
            ],
            featurizers=["esm"],
            distance_methods=["euclidean"],
        )

    @pytest.fixture
    def pipeline_config(self, molecule_config):
        """Create pipeline configuration for testing."""
        return PipelineConfig(name="test_pipeline", description="Test pipeline", molecule=molecule_config)

    @pytest.fixture
    def multimodal_config(self, molecule_config, protein_config):
        """Create multimodal pipeline configuration."""
        return PipelineConfig(name="multimodal_test", molecule=molecule_config, protein=protein_config)

    @pytest.fixture
    def runner(self, pipeline_config):
        """Create PipelineRunner instance for testing."""
        return PipelineRunner(pipeline_config)

    @pytest.fixture
    def temp_datasets_dir(self):
        """Create temporary datasets directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_dir = Path(temp_dir) / "datasets"
            train_dir = datasets_dir / "train"
            test_dir = datasets_dir / "test"

            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            # Create dummy dataset files
            (train_dir / "CHEMBL123456.jsonl.gz").touch()
            (train_dir / "CHEMBL789012.jsonl.gz").touch()
            (train_dir / "CHEMBL123456.fasta").touch()
            (train_dir / "CHEMBL789012.fasta").touch()

            yield str(datasets_dir)

    def test_pipeline_runner_initialization(self, runner, pipeline_config):
        """Test PipelineRunner initialization."""
        assert runner.config == pipeline_config
        assert runner.logger is not None
        assert runner.output_manager is not None
        assert runner.start_time is None
        assert len(runner.results) == 0
        assert len(runner.errors) == 0

    def test_setup_logger(self, runner):
        """Test logger setup."""
        logger = runner._setup_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_validate_pipeline_success(self, runner, temp_datasets_dir):
        """Test successful pipeline validation."""
        # Mock validate_datasets to pass
        with patch.object(runner.config, "validate_datasets"):
            runner._validate_pipeline(temp_datasets_dir)
            # Should not raise any exceptions

    def test_validate_pipeline_failure(self, runner, temp_datasets_dir):
        """Test pipeline validation failure."""
        # Mock validate_datasets to raise exception
        with patch.object(
            runner.config, "validate_datasets", side_effect=FileNotFoundError("Dataset not found")
        ):
            with pytest.raises(FileNotFoundError):
                runner._validate_pipeline(temp_datasets_dir)

    @patch("themap.pipeline.runner.load_molecule_dataset_from_jsonl")
    def test_load_molecule_dataset(self, mock_load_molecule, runner, temp_datasets_dir):
        """Test loading molecule dataset."""
        # Mock the dataset loader
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_molecule.return_value = mock_dataset

        dataset_config = DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])
        base_path = Path(temp_datasets_dir)

        dataset_info = runner._load_molecule_dataset(base_path, dataset_config)

        assert dataset_info["name"] == "CHEMBL123456"
        assert dataset_info["type"] == "molecule"
        assert dataset_info["size"] == 100
        assert dataset_info["dataset"] == mock_dataset

        # Verify the loader was called correctly
        mock_load_molecule.assert_called_once()
        call_args = mock_load_molecule.call_args
        assert "CHEMBL123456.jsonl.gz" in call_args[0][0]

    @patch("themap.pipeline.runner.load_protein_dataset_from_fasta")
    def test_load_protein_dataset(self, mock_load_protein, runner, temp_datasets_dir):
        """Test loading protein dataset."""
        # Mock the dataset loader
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_load_protein.return_value = mock_dataset

        dataset_config = DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])
        base_path = Path(temp_datasets_dir)

        dataset_info = runner._load_protein_dataset(base_path, dataset_config)

        assert dataset_info["name"] == "CHEMBL123456"
        assert dataset_info["type"] == "protein"
        assert dataset_info["size"] == 50
        assert dataset_info["dataset"] == mock_dataset

        # Verify the loader was called correctly
        mock_load_protein.assert_called_once()
        call_args = mock_load_protein.call_args
        assert "CHEMBL123456.fasta" in call_args[0][0]

    @patch("themap.pipeline.runner.load_molecule_dataset_from_jsonl")
    def test_load_datasets_with_errors(self, mock_load_molecule, runner, temp_datasets_dir):
        """Test loading datasets with some failures."""
        # Mock one successful load and one failure
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        def side_effect(path, **kwargs):
            if "CHEMBL123456" in path:
                return mock_dataset
            else:
                raise ValueError("Failed to load dataset")

        mock_load_molecule.side_effect = side_effect

        datasets_info = runner._load_datasets(temp_datasets_dir)

        # Should have one successful load
        assert len(datasets_info) == 1
        assert datasets_info[0]["name"] == "CHEMBL123456"

        # Should have one error recorded
        assert len(runner.errors) == 1
        assert runner.errors[0]["stage"] == "dataset_loading"
        assert runner.errors[0]["dataset"] == "CHEMBL789012"

    @patch("themap.pipeline.runner.MoleculeDatasetDistance")
    def test_compute_molecule_distances(self, mock_distance_class, runner):
        """Test computing molecule distances."""
        # Mock distance computer
        mock_distance_computer = Mock()
        mock_distance_computer.compute_distance.return_value = 0.75
        mock_distance_class.return_value = mock_distance_computer

        # Mock datasets info
        datasets_info = [
            {"name": "CHEMBL123456", "type": "molecule", "source_fold": "TRAIN", "dataset": Mock()},
            {"name": "CHEMBL789012", "type": "molecule", "source_fold": "TRAIN", "dataset": Mock()},
        ]

        results = runner._compute_molecule_distances(datasets_info)

        # Should compute one distance (avoid duplicate for i <= j)
        assert len(results) == 1
        assert results[0]["source_dataset"] == "CHEMBL789012"
        assert results[0]["target_dataset"] == "CHEMBL123456"
        assert results[0]["distance"] == 0.75
        assert results[0]["modality"] == "molecule"
        assert results[0]["method"] == "euclidean"

        # Verify distance computer was created and used
        mock_distance_class.assert_called_once()
        mock_distance_computer.compute_distance.assert_called_once()

    @patch("themap.pipeline.runner.ProteinDatasetDistance")
    def test_compute_protein_distances(self, mock_distance_class, runner):
        """Test computing protein distances."""
        # Mock distance computer
        mock_distance_computer = Mock()
        mock_distance_computer.compute_distance.return_value = 0.42
        mock_distance_class.return_value = mock_distance_computer

        # Mock datasets info
        datasets_info = [
            {"name": "CHEMBL123456", "type": "protein", "source_fold": "TRAIN", "dataset": Mock()},
            {"name": "CHEMBL789012", "type": "protein", "source_fold": "TRAIN", "dataset": Mock()},
        ]

        # Set protein config for runner
        runner.config.protein = ProteinConfig(
            datasets=[DatasetConfig("CHEMBL123456", "TRAIN", ["TEST"])],
            featurizers=["esm"],
            distance_methods=["euclidean"],
        )

        results = runner._compute_protein_distances(datasets_info)

        assert len(results) == 1
        assert results[0]["modality"] == "protein"
        assert results[0]["distance"] == 0.42

    @patch("themap.pipeline.runner.TaskDistance")
    @patch("themap.pipeline.runner.Task")
    @patch("themap.pipeline.runner.Tasks")
    def test_compute_task_distances(
        self, mock_tasks_class, mock_task_class, mock_task_distance_class, multimodal_config
    ):
        """Test computing combined task distances."""
        runner = PipelineRunner(multimodal_config)

        # Mock task distance computer
        mock_task_distance = Mock()
        mock_task_distance.compute_distance.return_value = 0.85
        mock_task_distance_class.return_value = mock_task_distance

        # Mock Task and Tasks classes
        mock_task1 = Mock()
        mock_task1.task_id = "CHEMBL123456"
        mock_task2 = Mock()
        mock_task2.task_id = "CHEMBL789012"
        mock_task_class.side_effect = [mock_task1, mock_task2]

        mock_tasks_instance = Mock()
        mock_tasks_class.return_value = mock_tasks_instance

        # Mock datasets info with multiple modalities
        datasets_info = [
            {"name": "CHEMBL123456", "type": "molecule", "dataset": Mock()},
            {"name": "CHEMBL123456", "type": "protein", "dataset": Mock()},
            {"name": "CHEMBL789012", "type": "molecule", "dataset": Mock()},
            {"name": "CHEMBL789012", "type": "protein", "dataset": Mock()},
        ]

        results = runner._compute_task_distances(datasets_info)

        assert len(results) == 1
        assert results[0]["modality"] == "combined_task"
        assert results[0]["distance"] == 0.85
        assert results[0]["source_dataset"] == "CHEMBL789012"
        assert results[0]["target_dataset"] == "CHEMBL123456"

    @patch("themap.pipeline.runner.load_molecule_dataset_from_jsonl")
    @patch("themap.pipeline.runner.MoleculeDatasetDistance")
    def test_full_pipeline_run(self, mock_distance_class, mock_load_molecule, runner, temp_datasets_dir):
        """Test full pipeline execution."""
        # Mock dataset loading
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_molecule.return_value = mock_dataset

        # Mock distance computation
        mock_distance_computer = Mock()
        mock_distance_computer.compute_distance.return_value = 0.75
        mock_distance_class.return_value = mock_distance_computer

        # Mock output manager
        with (
            patch.object(runner.output_manager, "save_results") as mock_save_results,
            patch.object(runner.output_manager, "create_summary_report") as mock_create_summary,
        ):
            mock_save_results.return_value = {"json": Path("results.json")}
            mock_create_summary.return_value = Path("summary.json")

            results = runner.run(temp_datasets_dir)

        # Verify results structure
        assert "config" in results
        assert "datasets_info" in results
        assert "distance_results" in results
        assert "runtime_seconds" in results
        assert "errors" in results

        # Verify runtime tracking
        assert results["runtime_seconds"] > 0

        # Verify output saving was called
        mock_save_results.assert_called_once()
        mock_create_summary.assert_called_once()

    def test_pipeline_run_with_errors(self, runner, temp_datasets_dir):
        """Test pipeline run with validation errors."""
        # Mock validation to fail
        with patch.object(
            runner.config, "validate_datasets", side_effect=FileNotFoundError("Dataset not found")
        ):
            with pytest.raises(FileNotFoundError):
                runner.run(temp_datasets_dir)

        # Should have recorded the error
        assert len(runner.errors) == 1

    def test_compute_distances_no_datasets(self, runner):
        """Test distance computation with no datasets."""
        results = runner._compute_distances([])
        assert len(results) == 0

    def test_intermediate_results_saving(self, runner):
        """Test intermediate results are saved when configured."""
        runner.config.output.save_intermediate = True

        with patch.object(runner.output_manager, "save_intermediate_results") as mock_save_intermediate:
            # Mock datasets info
            datasets_info = [
                {"name": "CHEMBL123456", "type": "molecule", "source_fold": "TRAIN", "dataset": Mock()},
                {"name": "CHEMBL789012", "type": "molecule", "source_fold": "TRAIN", "dataset": Mock()},
            ]

            with patch("themap.pipeline.runner.MoleculeDatasetDistance") as mock_distance_class:
                mock_distance_computer = Mock()
                mock_distance_computer.compute_distance.return_value = 0.75
                mock_distance_class.return_value = mock_distance_computer

                runner._compute_molecule_distances(datasets_info)

            # Should save intermediate results
            mock_save_intermediate.assert_called_once()

    def test_error_handling_in_distance_computation(self, runner):
        """Test error handling during distance computation."""
        # Mock datasets info
        datasets_info = [
            {"name": "CHEMBL123456", "type": "molecule", "source_fold": "TRAIN", "dataset": Mock()},
            {"name": "CHEMBL789012", "type": "molecule", "source_fold": "TRAIN", "dataset": Mock()},
        ]

        with patch("themap.pipeline.runner.MoleculeDatasetDistance") as mock_distance_class:
            # Mock distance computation to fail
            mock_distance_computer = Mock()
            mock_distance_computer.compute_distance.side_effect = RuntimeError("Distance computation failed")
            mock_distance_class.return_value = mock_distance_computer

            results = runner._compute_molecule_distances(datasets_info)

        # Should return empty results but record error
        assert len(results) == 0
        assert len(runner.errors) == 1
        assert runner.errors[0]["stage"] == "distance_computation"
        assert "Distance computation failed" in runner.errors[0]["error"]
