"""
Tests for the base distance computation module.

This test suite covers:
- Utility functions and validation
- AbstractTasksDistance base class
- Error handling and edge cases
"""

from unittest.mock import Mock, patch

import pytest

from themap.data.tasks import Tasks
from themap.distance.base import (
    MOLECULE_DISTANCE_METHODS,
    PROTEIN_DISTANCE_METHODS,
    AbstractTasksDistance,
    _get_dataset_distance,
    _validate_and_extract_task_id,
)
from themap.distance.exceptions import DataValidationError

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

        with patch("themap.models.otdd.src.distance.DatasetDistance", mock_distance_class):
            result = _get_dataset_distance()
            assert result == mock_distance_class

    def test_get_dataset_distance_import_error(self):
        """Test handling of import errors."""
        # Mock the import statement in the function to raise ImportError
        with patch.dict("sys.modules", {"themap.models.otdd.src.distance": None}):
            with pytest.raises(ImportError) as exc_info:
                _get_dataset_distance()
            assert "OTDD dependencies not available" in str(exc_info.value)


# ============================================================================
# Abstract Base Class Tests
# ============================================================================


class TestAbstractTasksDistance:
    """Test AbstractTasksDistance base class."""

    @pytest.fixture
    def sample_tasks(self):
        """Create a mock Tasks collection."""
        tasks = Mock(spec=Tasks)
        tasks.get_tasks.return_value = [Mock()]
        tasks.get_num_fold_tasks.return_value = 1
        return tasks

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

    def test_get_num_tasks(self, sample_tasks):
        """Test getting number of tasks."""
        distance = AbstractTasksDistance(tasks=sample_tasks)
        # Mock the source and target setup
        distance.source = [Mock(), Mock()]
        distance.target = [Mock()]

        source_num, target_num = distance.get_num_tasks()
        assert source_num == 2
        assert target_num == 1

    def test_call_method(self, sample_tasks):
        """Test that __call__ delegates to get_distance."""
        distance = AbstractTasksDistance(tasks=sample_tasks)

        with patch.object(distance, "get_distance", return_value={"test": "result"}) as mock_get_distance:
            result = distance()
            mock_get_distance.assert_called_once()
            assert result == {"test": "result"}

    def test_constants_available(self):
        """Test that distance method constants are available."""
        assert "euclidean" in MOLECULE_DISTANCE_METHODS
        assert "cosine" in MOLECULE_DISTANCE_METHODS
        assert "otdd" in MOLECULE_DISTANCE_METHODS

        assert "euclidean" in PROTEIN_DISTANCE_METHODS
        assert "cosine" in PROTEIN_DISTANCE_METHODS


if __name__ == "__main__":
    pytest.main([__file__])
