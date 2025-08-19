"""
Comprehensive tests for the tasks_distance module.

This test suite covers:
- Utility functions and validation
- MoleculeDatasetDistance class
- ProteinDatasetDistance class
- TaskDistance class
- Error handling and edge cases
- Integration scenarios
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from themap.data.molecule_dataset import MoleculeDataset
from themap.data.protein_datasets import ProteinMetadataDataset
from themap.data.tasks import Task, Tasks
from themap.distance import (
    MOLECULE_DISTANCE_METHODS,
    PROTEIN_DISTANCE_METHODS,
    AbstractTasksDistance,
    DataValidationError,
    DistanceComputationError,
    MoleculeDatasetDistance,
    ProteinDatasetDistance,
    TaskDistance,
)
from themap.distance.base import (
    _get_dataset_distance,
    _validate_and_extract_task_id,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_molecule_dataset():
    """Create a mock molecule dataset for testing."""
    mock_dataset = Mock(spec=MoleculeDataset)
    mock_dataset.task_id = "CHEMBL123456"
    mock_dataset.data = [Mock() for _ in range(10)]
    return mock_dataset


@pytest.fixture
def sample_protein_dataset():
    """Create a mock protein dataset for testing."""
    mock_dataset = Mock(spec=ProteinMetadataDataset)
    mock_dataset.task_id = "PROTEIN123"
    mock_dataset.proteins = {"protein1": "MLSDEDFKAV", "protein2": "QLKEKGLF"}
    return mock_dataset


@pytest.fixture
def sample_task(sample_molecule_dataset, sample_protein_dataset):
    """Create a mock task with both molecule and protein data."""
    task = Mock(spec=Task)
    task.task_id = "CHEMBL123456"
    task.molecule_dataset = sample_molecule_dataset
    task.protein_dataset = sample_protein_dataset
    task.metadata_datasets = None
    return task


@pytest.fixture
def sample_tasks(sample_task):
    """Create a mock Tasks collection."""
    tasks = Mock(spec=Tasks)
    tasks.get_tasks.return_value = [sample_task]
    tasks.get_num_fold_tasks.return_value = 1
    tasks.get_distance_computation_ready_features.return_value = (
        [np.random.rand(128)],  # source_features
        [np.random.rand(128)],  # target_features
        ["train_CHEMBL123456"],  # source_names
        ["test_CHEMBL123456"],  # target_names
    )
    return tasks


@pytest.fixture
def sample_features():
    """Create sample feature arrays for testing."""
    source_features = [np.random.rand(128) for _ in range(3)]
    target_features = [np.random.rand(128) for _ in range(2)]
    source_names = ["train_CHEMBL001", "train_CHEMBL002", "train_CHEMBL003"]
    target_names = ["test_CHEMBL004", "test_CHEMBL005"]
    return source_features, target_features, source_names, target_names


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions and validation."""

    def test_validate_and_extract_task_id_valid(self):
        """Test successful task ID extraction."""
        assert _validate_and_extract_task_id("train_CHEMBL123456") == "CHEMBL123456"
        assert _validate_and_extract_task_id("test_PROTEIN789") == "PROTEIN789"
        assert _validate_and_extract_task_id("valid_TASK001") == "TASK001"

    def test_validate_and_extract_task_id_no_underscore(self):
        """Test task ID extraction when no underscore present."""
        # Should log warning and return the original string
        result = _validate_and_extract_task_id("CHEMBL123456")
        assert result == "CHEMBL123456"

    def test_validate_and_extract_task_id_invalid_type(self):
        """Test validation with invalid input type."""
        with pytest.raises(DataValidationError):
            _validate_and_extract_task_id(123)

        with pytest.raises(DataValidationError):
            _validate_and_extract_task_id(None)

    @patch("themap.distance.base.logger")
    def test_validate_and_extract_task_id_logging(self, mock_logger):
        """Test that warning is logged for malformed task names."""
        _validate_and_extract_task_id("malformed")
        mock_logger.warning.assert_called_once()

    def test_get_dataset_distance_success(self):
        """Test successful lazy import of DatasetDistance."""
        mock_distance_class = Mock()
        mock_module = Mock()
        mock_module.DatasetDistance = mock_distance_class

        with patch("builtins.__import__") as mock_import:
            mock_import.return_value = mock_module
            result = _get_dataset_distance()
            assert result == mock_distance_class

    def test_get_dataset_distance_import_error(self):
        """Test handling of import errors."""
        with patch("builtins.__import__", side_effect=ImportError("OTDD not available")):
            with pytest.raises(ImportError) as exc_info:
                _get_dataset_distance()
            assert "OTDD dependencies not available" in str(exc_info.value)


# ============================================================================
# Abstract Base Class Tests
# ============================================================================


class TestAbstractTasksDistance:
    """Test AbstractTasksDistance base class."""

    def test_initialization_with_tasks(self, sample_tasks):
        """Test initialization with Tasks object."""
        distance = AbstractTasksDistance(tasks=sample_tasks)
        assert distance.tasks == sample_tasks
        assert distance.molecule_method == "euclidean"
        assert distance.protein_method == "euclidean"

    def test_initialization_with_global_method(self, sample_tasks):
        """Test initialization with global method override."""
        distance = AbstractTasksDistance(tasks=sample_tasks, method="cosine")
        assert distance.molecule_method == "cosine"
        assert distance.protein_method == "cosine"
        assert distance.metadata_method == "cosine"

    def test_initialization_with_specific_methods(self, sample_tasks):
        """Test initialization with specific methods for each data type."""
        distance = AbstractTasksDistance(
            tasks=sample_tasks, molecule_method="otdd", protein_method="cosine", metadata_method="jaccard"
        )
        assert distance.molecule_method == "otdd"
        assert distance.protein_method == "cosine"
        # protein_method overrides metadata_method due to backward compatibility
        assert distance.metadata_method == "cosine"

    def test_setup_source_target_no_tasks(self):
        """Test setup when no tasks are provided."""
        distance = AbstractTasksDistance(tasks=None)
        assert distance.source is None
        assert distance.target is None
        assert distance.symmetric_tasks is True

    def test_abstract_methods_not_implemented(self, sample_tasks):
        """Test that abstract methods raise NotImplementedError."""
        distance = AbstractTasksDistance(tasks=sample_tasks)

        with pytest.raises(NotImplementedError):
            distance.get_distance()

        with pytest.raises(NotImplementedError):
            distance.get_hopts()

        with pytest.raises(NotImplementedError):
            distance.get_supported_methods("molecule")


# ============================================================================
# MoleculeDatasetDistance Tests
# ============================================================================


class TestMoleculeDatasetDistance:
    """Test MoleculeDatasetDistance class."""

    def test_initialization_valid_method(self, sample_tasks):
        """Test initialization with valid molecule method."""
        for method in MOLECULE_DISTANCE_METHODS:
            distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method=method)
            assert distance.molecule_method == method

    def test_initialization_invalid_method(self, sample_tasks):
        """Test initialization with invalid molecule method."""
        with pytest.raises(ValueError) as exc_info:
            MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="invalid")
        assert "not supported for molecule datasets" in str(exc_info.value)

    def test_extract_molecule_datasets_none_tasks(self):
        """Test extraction when no tasks are provided."""
        distance = MoleculeDatasetDistance(tasks=None)
        assert distance.source_molecule_datasets == []
        assert distance.target_molecule_datasets == []
        assert distance.source_task_ids == []
        assert distance.target_task_ids == []

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful euclidean distance computation."""
        mock_compute_features.return_value = sample_features

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")
        result = distance.euclidean_distance()

        assert isinstance(result, dict)
        assert "CHEMBL004" in result
        assert "CHEMBL005" in result
        assert "CHEMBL001" in result["CHEMBL004"]

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_no_features(self, mock_compute_features, sample_tasks):
        """Test euclidean distance computation with no features."""
        mock_compute_features.return_value = ([], [], [], [])

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")
        result = distance.euclidean_distance()

        assert result == {}

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_mismatched_features(self, mock_compute_features, sample_tasks):
        """Test euclidean distance with mismatched feature dimensions."""
        # Features with different dimensions
        source_features = [np.random.rand(64)]  # 64 dimensions
        target_features = [np.random.rand(128)]  # 128 dimensions
        source_names = ["train_CHEMBL001"]
        target_names = ["test_CHEMBL002"]

        mock_compute_features.return_value = (source_features, target_features, source_names, target_names)

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")

        with pytest.raises(DistanceComputationError) as exc_info:
            distance.euclidean_distance()
        assert "Feature dimension mismatch" in str(exc_info.value)

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_euclidean_distance_invalid_values(self, mock_compute_features, sample_tasks):
        """Test euclidean distance computation with NaN/inf values."""
        # Create features that will produce NaN distances
        source_features = [np.array([np.nan, 1.0, 2.0])]
        target_features = [np.array([1.0, np.inf, 2.0])]
        source_names = ["train_CHEMBL001"]
        target_names = ["test_CHEMBL002"]

        mock_compute_features.return_value = (source_features, target_features, source_names, target_names)

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="euclidean")

        with patch("scipy.spatial.distance.cdist", return_value=np.array([[np.nan]])):
            result = distance.euclidean_distance()
            # Should handle NaN by replacing with default value
            assert result["CHEMBL002"]["CHEMBL001"] == 1.0

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    @patch("themap.distance.molecule_distance._get_dataset_distance")
    @patch("themap.distance.molecule_distance.MoleculeDataloader")
    def test_otdd_distance_success(
        self, mock_dataloader, mock_get_dataset_distance, mock_compute_features, sample_tasks, sample_features
    ):
        """Test successful OTDD distance computation."""
        mock_compute_features.return_value = sample_features

        # Mock OTDD computation
        mock_distance_class = Mock()
        mock_distance_instance = Mock()
        mock_distance_tensor = Mock()
        mock_distance_tensor.cpu.return_value.item.return_value = 0.5
        mock_distance_instance.distance.return_value = mock_distance_tensor
        mock_distance_class.return_value = mock_distance_instance
        mock_get_dataset_distance.return_value = mock_distance_class

        # Mock MoleculeDataloader
        mock_dataloader.return_value = Mock()

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="otdd")
        distance.source_molecule_datasets = [
            Mock(spec=MoleculeDataset),
            Mock(spec=MoleculeDataset),
            Mock(spec=MoleculeDataset),
        ]
        distance.target_molecule_datasets = [Mock(spec=MoleculeDataset), Mock(spec=MoleculeDataset)]
        distance.source_task_ids = ["CHEMBL001", "CHEMBL002", "CHEMBL003"]
        distance.target_task_ids = ["CHEMBL004", "CHEMBL005"]

        result = distance.otdd_distance()

        assert isinstance(result, dict)
        assert len(result) == 2  # Two target tasks

    @patch.object(MoleculeDatasetDistance, "_compute_features")
    @patch("themap.distance.molecule_distance._get_dataset_distance")
    @patch("themap.distance.molecule_distance.MoleculeDataloader")
    def test_otdd_distance_error_handling(
        self, mock_dataloader, mock_get_dataset_distance, mock_compute_features, sample_tasks, sample_features
    ):
        """Test OTDD distance computation error handling."""
        mock_compute_features.return_value = sample_features
        mock_get_dataset_distance.side_effect = ImportError("OTDD not available")

        # Mock MoleculeDataloader
        mock_dataloader.return_value = Mock()

        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="otdd")
        distance.source_molecule_datasets = [Mock(spec=MoleculeDataset)]
        distance.target_molecule_datasets = [Mock(spec=MoleculeDataset)]
        distance.source_task_ids = ["CHEMBL001"]
        distance.target_task_ids = ["CHEMBL002"]

        result = distance.otdd_distance()

        # Should handle error gracefully and return fallback distance
        assert result["CHEMBL002"]["CHEMBL001"] == 1.0

    def test_get_supported_methods(self, sample_tasks):
        """Test getting supported methods for different data types."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        mol_methods = distance.get_supported_methods("molecule")
        assert set(mol_methods) == set(MOLECULE_DISTANCE_METHODS)

        prot_methods = distance.get_supported_methods("protein")
        assert set(prot_methods) == set(PROTEIN_DISTANCE_METHODS)

        meta_methods = distance.get_supported_methods("metadata")
        assert "euclidean" in meta_methods
        assert "cosine" in meta_methods

    def test_get_supported_methods_invalid_type(self, sample_tasks):
        """Test getting supported methods with invalid data type."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        with pytest.raises(ValueError) as exc_info:
            distance.get_supported_methods("invalid")
        assert "Unknown data type" in str(exc_info.value)

    def test_load_distance_success(self, sample_tasks):
        """Test successful loading of distance data."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        # Create temporary file with valid distance data
        distance_data = {"task1": {"task2": 0.5, "task3": 0.8}, "task2": {"task1": 0.5, "task3": 0.3}}

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump(distance_data, f)
            temp_path = f.name

        try:
            distance.load_distance(temp_path)
            assert distance.distance == distance_data
        finally:
            Path(temp_path).unlink()

    def test_load_distance_file_not_found(self, sample_tasks):
        """Test loading distance from non-existent file."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        with pytest.raises(FileNotFoundError):
            distance.load_distance("/non/existent/file.pkl")

    def test_load_distance_invalid_format(self, sample_tasks):
        """Test loading distance from invalid file format."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        # Create temporary file with invalid data
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump("invalid_data", f)  # Not a dictionary
            temp_path = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                distance.load_distance(temp_path)
            assert "Distance file must contain a dictionary" in str(exc_info.value)
        finally:
            Path(temp_path).unlink()

    def test_to_pandas(self, sample_tasks):
        """Test conversion to pandas DataFrame."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)
        distance.distance = {"task1": {"task2": 0.5}, "task2": {"task1": 0.5}}

        df = distance.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_repr(self, sample_tasks):
        """Test string representation."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="cosine")
        distance.source_molecule_datasets = [Mock(), Mock()]
        distance.target_molecule_datasets = [Mock()]

        repr_str = repr(distance)
        assert "MoleculeDatasetDistance" in repr_str
        assert "cosine" in repr_str
        assert "source_tasks=2" in repr_str
        assert "target_tasks=1" in repr_str


# ============================================================================
# ProteinDatasetDistance Tests
# ============================================================================


class TestProteinDatasetDistance:
    """Test ProteinDatasetDistance class."""

    def test_initialization_valid_method(self, sample_tasks):
        """Test initialization with valid protein method."""
        for method in PROTEIN_DISTANCE_METHODS:
            distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method=method)
            assert distance.protein_method == method

    def test_initialization_invalid_method(self, sample_tasks):
        """Test initialization with invalid protein method."""
        with pytest.raises(ValueError) as exc_info:
            ProteinDatasetDistance(tasks=sample_tasks, protein_method="invalid")
        assert "not supported for protein datasets" in str(exc_info.value)

    def test_sequence_identity_distance_not_implemented(self, sample_tasks):
        """Test that sequence identity distance raises NotImplementedError."""
        distance = ProteinDatasetDistance(tasks=sample_tasks)

        with pytest.raises(NotImplementedError) as exc_info:
            distance.sequence_identity_distance()
        assert "not yet implemented" in str(exc_info.value)

    @patch.object(ProteinDatasetDistance, "_compute_features")
    def test_euclidean_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful euclidean distance computation for proteins."""
        mock_compute_features.return_value = sample_features

        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")
        result = distance.euclidean_distance()

        assert isinstance(result, dict)
        assert "CHEMBL004" in result
        assert "CHEMBL005" in result

    @patch.object(ProteinDatasetDistance, "_compute_features")
    def test_cosine_distance_success(self, mock_compute_features, sample_tasks, sample_features):
        """Test successful cosine distance computation for proteins."""
        mock_compute_features.return_value = sample_features

        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="cosine")
        result = distance.cosine_distance()

        assert isinstance(result, dict)
        assert "CHEMBL004" in result
        assert "CHEMBL005" in result

    def test_get_distance_euclidean(self, sample_tasks):
        """Test get_distance method with euclidean method."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")

        with patch.object(distance, "euclidean_distance", return_value={"test": "result"}) as mock_euclidean:
            result = distance.get_distance()
            mock_euclidean.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_cosine(self, sample_tasks):
        """Test get_distance method with cosine method."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="cosine")

        with patch.object(distance, "cosine_distance", return_value={"test": "result"}) as mock_cosine:
            result = distance.get_distance()
            mock_cosine.assert_called_once()
            assert result == {"test": "result"}

    def test_get_distance_unknown_method(self, sample_tasks):
        """Test get_distance method with unknown method."""
        distance = ProteinDatasetDistance(tasks=sample_tasks, protein_method="euclidean")
        distance.protein_method = "unknown"  # Manually set invalid method

        with pytest.raises(ValueError) as exc_info:
            distance.get_distance()
        assert "Unknown protein method" in str(exc_info.value)


# ============================================================================
# TaskDistance Tests
# ============================================================================


class TestTaskDistance:
    """Test TaskDistance class."""

    def test_initialization_with_tasks(self, sample_tasks):
        """Test initialization with Tasks object."""
        distance = TaskDistance(tasks=sample_tasks)
        assert distance.tasks == sample_tasks
        assert distance.molecule_method == "euclidean"
        assert distance.protein_method == "euclidean"

    def test_initialization_legacy_mode(self):
        """Test initialization in legacy mode with task IDs."""
        source_ids = ["task1", "task2"]
        target_ids = ["task3", "task4"]
        chem_space = np.random.rand(2, 2)

        distance = TaskDistance(
            tasks=None,
            source_task_ids=source_ids,
            target_task_ids=target_ids,
            external_chemical_space=chem_space,
        )

        assert distance.source_task_ids == source_ids
        assert distance.target_task_ids == target_ids
        assert np.array_equal(distance.external_chemical_space, chem_space)

    def test_shape_property(self):
        """Test shape property."""
        distance = TaskDistance(tasks=None, source_task_ids=["t1", "t2", "t3"], target_task_ids=["t4", "t5"])
        assert distance.shape == (3, 2)

    @patch.object(TaskDistance, "compute_molecule_distance")
    def test_compute_molecule_distance(self, mock_compute_mol, sample_tasks):
        """Test compute_molecule_distance method."""
        mock_compute_mol.return_value = {"test": "result"}

        distance = TaskDistance(tasks=sample_tasks)
        result = distance.compute_molecule_distance(method="cosine", molecule_featurizer="morgan")

        mock_compute_mol.assert_called_once_with(method="cosine", molecule_featurizer="morgan")
        assert result == {"test": "result"}

    @patch.object(TaskDistance, "compute_protein_distance")
    def test_compute_protein_distance(self, mock_compute_prot, sample_tasks):
        """Test compute_protein_distance method."""
        mock_compute_prot.return_value = {"test": "result"}

        distance = TaskDistance(tasks=sample_tasks)
        result = distance.compute_protein_distance(method="cosine", protein_featurizer="esm2")

        mock_compute_prot.assert_called_once_with(method="cosine", protein_featurizer="esm2")
        assert result == {"test": "result"}

    def test_compute_molecule_distance_no_tasks(self):
        """Test compute_molecule_distance without tasks."""
        distance = TaskDistance(tasks=None)

        with pytest.raises(ValueError) as exc_info:
            distance.compute_molecule_distance()
        assert "Tasks collection required" in str(exc_info.value)

    def test_compute_protein_distance_no_tasks(self):
        """Test compute_protein_distance without tasks."""
        distance = TaskDistance(tasks=None)

        with pytest.raises(ValueError) as exc_info:
            distance.compute_protein_distance()
        assert "Tasks collection required" in str(exc_info.value)

    def test_compute_combined_distance_success(self, sample_tasks):
        """Test successful combined distance computation."""
        distance = TaskDistance(tasks=sample_tasks)

        # Mock the individual distance computations
        mol_distances = {"task1": {"task2": 0.5}}
        prot_distances = {"task1": {"task2": 0.3}}

        distance.molecule_distances = mol_distances
        distance.protein_distances = prot_distances

        result = distance.compute_combined_distance(combination_strategy="average")

        assert "task1" in result
        assert "task2" in result["task1"]
        assert result["task1"]["task2"] == 0.4  # (0.5 + 0.3) / 2

    def test_compute_combined_distance_weighted_average(self, sample_tasks):
        """Test combined distance with weighted average."""
        distance = TaskDistance(tasks=sample_tasks)

        mol_distances = {"task1": {"task2": 0.6}}
        prot_distances = {"task1": {"task2": 0.4}}

        distance.molecule_distances = mol_distances
        distance.protein_distances = prot_distances

        result = distance.compute_combined_distance(
            combination_strategy="weighted_average", molecule_weight=0.7, protein_weight=0.3
        )

        expected = (0.6 * 0.7 + 0.4 * 0.3) / (0.7 + 0.3)
        assert abs(result["task1"]["task2"] - expected) < 1e-6

    def test_compute_combined_distance_min_max(self, sample_tasks):
        """Test combined distance with min and max strategies."""
        distance = TaskDistance(tasks=sample_tasks)

        mol_distances = {"task1": {"task2": 0.6}}
        prot_distances = {"task1": {"task2": 0.4}}

        distance.molecule_distances = mol_distances
        distance.protein_distances = prot_distances

        # Test min strategy
        result_min = distance.compute_combined_distance(combination_strategy="min")
        assert result_min["task1"]["task2"] == 0.4

        # Test max strategy
        result_max = distance.compute_combined_distance(combination_strategy="max")
        assert result_max["task1"]["task2"] == 0.6

    def test_compute_combined_distance_invalid_strategy(self, sample_tasks):
        """Test combined distance with invalid strategy."""
        distance = TaskDistance(tasks=sample_tasks)

        distance.molecule_distances = {"task1": {"task2": 0.5}}
        distance.protein_distances = {"task1": {"task2": 0.3}}

        with pytest.raises(ValueError) as exc_info:
            distance.compute_combined_distance(combination_strategy="invalid")
        assert "Unknown combination strategy" in str(exc_info.value)

    def test_compute_ext_chem_distance_success(self):
        """Test compute_ext_chem_distance with external matrix."""
        chem_space = np.array([[0.1, 0.2], [0.3, 0.4]])
        distance = TaskDistance(
            tasks=None,
            source_task_ids=["s1", "s2"],
            target_task_ids=["t1", "t2"],
            external_chemical_space=chem_space,
        )

        result = distance.compute_ext_chem_distance("dummy_method")

        assert "t1" in result
        assert "t2" in result
        assert result["t1"]["s1"] == 0.1
        assert result["t2"]["s2"] == 0.4

    def test_compute_ext_chem_distance_no_matrix(self):
        """Test compute_ext_chem_distance without external matrix."""
        distance = TaskDistance(tasks=None)

        with pytest.raises(NotImplementedError) as exc_info:
            distance.compute_ext_chem_distance("dummy_method")
        assert "External chemical space matrix not provided" in str(exc_info.value)

    def test_compute_ext_prot_distance_success(self):
        """Test compute_ext_prot_distance with external matrix."""
        prot_space = np.array([[0.5, 0.6], [0.7, 0.8]])
        distance = TaskDistance(
            tasks=None,
            source_task_ids=["s1", "s2"],
            target_task_ids=["t1", "t2"],
            external_protein_space=prot_space,
        )

        result = distance.compute_ext_prot_distance("dummy_method")

        assert "t1" in result
        assert "t2" in result
        assert result["t1"]["s1"] == 0.5
        assert result["t2"]["s2"] == 0.8

    def test_compute_ext_prot_distance_no_matrix(self):
        """Test compute_ext_prot_distance without external matrix."""
        distance = TaskDistance(tasks=None)

        with pytest.raises(NotImplementedError) as exc_info:
            distance.compute_ext_prot_distance("dummy_method")
        assert "External protein space matrix not provided" in str(exc_info.value)

    def test_get_computed_distance(self, sample_tasks):
        """Test get_computed_distance method."""
        distance = TaskDistance(tasks=sample_tasks)

        mol_distances = {"test": "mol"}
        prot_distances = {"test": "prot"}
        combined_distances = {"test": "combined"}

        distance.molecule_distances = mol_distances
        distance.protein_distances = prot_distances
        distance.combined_distances = combined_distances

        assert distance.get_computed_distance("molecule") == mol_distances
        assert distance.get_computed_distance("protein") == prot_distances
        assert distance.get_computed_distance("combined") == combined_distances

    def test_get_computed_distance_invalid_type(self, sample_tasks):
        """Test get_computed_distance with invalid type."""
        distance = TaskDistance(tasks=sample_tasks)

        with pytest.raises(ValueError) as exc_info:
            distance.get_computed_distance("invalid")
        assert "Unknown distance type" in str(exc_info.value)

    def test_to_pandas_external_chemical(self):
        """Test to_pandas with external chemical space."""
        chem_space = np.array([[0.1, 0.2], [0.3, 0.4]])
        distance = TaskDistance(
            tasks=None,
            source_task_ids=["s1", "s2"],
            target_task_ids=["t1", "t2"],
            external_chemical_space=chem_space,
        )

        df = distance.to_pandas("external_chemical")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert df.index.tolist() == ["s1", "s2"]
        assert df.columns.tolist() == ["t1", "t2"]

    def test_to_pandas_no_distances(self, sample_tasks):
        """Test to_pandas when no distances are available."""
        distance = TaskDistance(tasks=sample_tasks)

        with pytest.raises(ValueError) as exc_info:
            distance.to_pandas("molecule")
        assert "No molecule distances available" in str(exc_info.value)

    def test_repr(self, sample_tasks):
        """Test string representation."""
        distance = TaskDistance(tasks=sample_tasks)
        distance.source_task_ids = ["s1", "s2"]
        distance.target_task_ids = ["t1"]
        distance.molecule_distances = {"test": "mol"}

        repr_str = repr(distance)
        assert "TaskDistance" in repr_str
        assert "mode=tasks" in repr_str
        assert "source_tasks=2" in repr_str
        assert "target_tasks=1" in repr_str
        assert "computed_distances=1" in repr_str

    @patch("pickle.load")
    def test_load_ext_chem_distance(self, mock_pickle_load):
        """Test loading external chemical distance."""
        mock_data = {
            "train_chembl_ids": ["t1", "t2"],
            "test_chembl_ids": ["t3", "t4"],
            "distance_matrices": np.array([[0.1, 0.2], [0.3, 0.4]]),
        }
        mock_pickle_load.return_value = mock_data

        with patch("builtins.open", create=True):
            result = TaskDistance.load_ext_chem_distance("dummy_path")

            assert result.source_task_ids == ["t1", "t2"]
            assert result.target_task_ids == ["t3", "t4"]
            assert np.array_equal(result.external_chemical_space, mock_data["distance_matrices"])

    @patch("pickle.load")
    def test_load_ext_chem_distance_missing_keys(self, mock_pickle_load):
        """Test loading external chemical distance with missing keys."""
        mock_data = {"invalid": "data"}
        mock_pickle_load.return_value = mock_data

        with patch("builtins.open", create=True):
            with pytest.raises(ValueError) as exc_info:
                TaskDistance.load_ext_chem_distance("dummy_path")
            assert "No source task IDs found" in str(exc_info.value)


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_distance_computation_error_inheritance(self):
        """Test DistanceComputationError inheritance."""
        error = DistanceComputationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_data_validation_error_inheritance(self):
        """Test DataValidationError inheritance."""
        error = DataValidationError("Test validation error")
        assert isinstance(error, Exception)
        assert str(error) == "Test validation error"

    def test_empty_feature_arrays(self, sample_tasks):
        """Test handling of empty feature arrays."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        with patch.object(distance, "_compute_features", return_value=([], [], [], [])):
            result = distance.euclidean_distance()
            assert result == {}

    def test_malformed_task_names(self, sample_tasks):
        """Test handling of malformed task names."""
        # Test with various malformed names
        test_cases = [
            "",  # Empty string
            "_",  # Just underscore
            "task_",  # Ends with underscore
            "_task",  # Starts with underscore
        ]

        for task_name in test_cases:
            result = _validate_and_extract_task_id(task_name)
            # Should handle gracefully
            assert isinstance(result, str)

    @patch("themap.distance.molecule_distance.logger")
    def test_logging_integration(self, mock_logger, sample_tasks):
        """Test that logging is properly integrated."""
        distance = MoleculeDatasetDistance(tasks=sample_tasks)

        # Test logging during error conditions
        with patch.object(distance, "_compute_features", side_effect=Exception("Test error")):
            with pytest.raises(DistanceComputationError):
                distance.euclidean_distance()

            # Verify error was logged
            mock_logger.error.assert_called()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("themap.distance.molecule_distance._get_dataset_distance")
    @patch("themap.distance.molecule_distance.MoleculeDataloader")
    @patch.object(MoleculeDatasetDistance, "_compute_features")
    def test_full_molecule_distance_workflow(
        self, mock_compute_features, mock_dataloader, mock_get_dataset_distance, sample_tasks, sample_features
    ):
        """Test complete molecule distance computation workflow."""
        # Setup mocks
        mock_compute_features.return_value = sample_features

        # Mock MoleculeDataloader
        mock_dataloader.return_value = Mock()

        mock_distance_class = Mock()
        mock_distance_instance = Mock()
        mock_distance_tensor = torch.tensor(0.75)
        mock_distance_instance.distance.return_value = mock_distance_tensor
        mock_distance_class.return_value = mock_distance_instance
        mock_get_dataset_distance.return_value = mock_distance_class

        # Create distance calculator
        distance = MoleculeDatasetDistance(tasks=sample_tasks, molecule_method="otdd")
        distance.source_molecule_datasets = [Mock(spec=MoleculeDataset) for _ in range(3)]
        distance.target_molecule_datasets = [Mock(spec=MoleculeDataset) for _ in range(2)]
        distance.source_task_ids = ["CHEMBL001", "CHEMBL002", "CHEMBL003"]
        distance.target_task_ids = ["CHEMBL004", "CHEMBL005"]

        # Test OTDD computation
        otdd_result = distance.otdd_distance()
        assert len(otdd_result) == 2
        assert len(otdd_result["CHEMBL004"]) == 3

        # Test euclidean computation
        euclidean_result = distance.euclidean_distance()
        assert len(euclidean_result) == 2
        assert len(euclidean_result["CHEMBL004"]) == 3

        # Test cosine computation
        cosine_result = distance.cosine_distance()
        assert len(cosine_result) == 2
        assert len(cosine_result["CHEMBL004"]) == 3

    def test_task_distance_complete_workflow(self, sample_tasks):
        """Test complete TaskDistance workflow."""
        distance = TaskDistance(tasks=sample_tasks)

        # Mock individual distance computations
        mol_distances = {"task1": {"task2": 0.5, "task3": 0.7}, "task4": {"task2": 0.3, "task3": 0.9}}
        prot_distances = {"task1": {"task2": 0.4, "task3": 0.6}, "task4": {"task2": 0.2, "task3": 0.8}}

        def mock_mol_compute(method=None, molecule_featurizer="ecfp"):
            distance.molecule_distances = mol_distances
            return mol_distances

        def mock_prot_compute(method=None, protein_featurizer="esm2_t33_650M_UR50D"):
            distance.protein_distances = prot_distances
            return prot_distances

        with patch.object(distance, "compute_molecule_distance", side_effect=mock_mol_compute):
            with patch.object(distance, "compute_protein_distance", side_effect=mock_prot_compute):
                # Test computing all distances
                all_results = distance.compute_all_distances()

                assert "molecule" in all_results
                assert "protein" in all_results
                assert "combined" in all_results

                # Verify combined distances are computed correctly
                combined = all_results["combined"]
                assert combined["task1"]["task2"] == 0.45  # (0.5 + 0.4) / 2

    def test_error_recovery_workflow(self, sample_tasks):
        """Test error recovery in complex workflows."""
        distance = TaskDistance(tasks=sample_tasks)

        # Simulate molecule distance failing but protein distance succeeding
        prot_distances = {"task1": {"task2": 0.5}}

        with patch.object(distance, "compute_molecule_distance", side_effect=Exception("Molecule failed")):
            with patch.object(distance, "compute_protein_distance", return_value=prot_distances):
                # Should gracefully handle partial failures
                all_results = distance.compute_all_distances()

                assert all_results["molecule"] == {}  # Failed gracefully
                assert all_results["protein"] == prot_distances  # Succeeded
                assert all_results["combined"] == {}  # Can't combine without both


if __name__ == "__main__":
    pytest.main([__file__])
