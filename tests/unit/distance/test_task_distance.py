"""
Tests for the combined task distance computation module.

This test suite covers:
- TaskDistance class
- Combined distance computation with multiple strategies
- External distance matrix handling
- Error handling and edge cases
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from themap.data.tasks import Tasks
from themap.distance.task_distance import TaskDistance

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_tasks():
    """Create a mock Tasks collection."""
    tasks = Mock(spec=Tasks)
    tasks.get_tasks.return_value = [Mock()]
    tasks.get_num_fold_tasks.return_value = 1
    return tasks


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

    def test_initialization_with_methods(self, sample_tasks):
        """Test initialization with specific methods."""
        distance = TaskDistance(
            tasks=sample_tasks, molecule_method="otdd", protein_method="cosine", metadata_method="jaccard"
        )
        assert distance.molecule_method == "otdd"
        assert distance.protein_method == "cosine"
        # protein_method overrides metadata_method due to backward compatibility
        assert distance.metadata_method == "cosine"

    def test_shape_property(self):
        """Test shape property."""
        distance = TaskDistance(tasks=None, source_task_ids=["t1", "t2", "t3"], target_task_ids=["t4", "t5"])
        assert distance.shape == (3, 2)

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

    @patch("themap.distance.task_distance.MoleculeDatasetDistance")
    def test_compute_molecule_distance_success(self, mock_mol_distance_class, sample_tasks):
        """Test successful molecule distance computation."""
        mock_mol_distance = Mock()
        mock_mol_distance.get_distance.return_value = {"task1": {"task2": 0.5}}
        mock_mol_distance_class.return_value = mock_mol_distance

        distance = TaskDistance(tasks=sample_tasks)
        result = distance.compute_molecule_distance(method="cosine", molecule_featurizer="morgan")

        mock_mol_distance_class.assert_called_once_with(tasks=sample_tasks, molecule_method="cosine")
        assert result == {"task1": {"task2": 0.5}}
        assert distance.molecule_distances == result

    @patch("themap.distance.task_distance.ProteinDatasetDistance")
    def test_compute_protein_distance_success(self, mock_prot_distance_class, sample_tasks):
        """Test successful protein distance computation."""
        mock_prot_distance = Mock()
        mock_prot_distance.get_distance.return_value = {"task1": {"task2": 0.3}}
        mock_prot_distance_class.return_value = mock_prot_distance

        distance = TaskDistance(tasks=sample_tasks)
        result = distance.compute_protein_distance(method="cosine", protein_featurizer="esm2")

        mock_prot_distance_class.assert_called_once_with(tasks=sample_tasks, protein_method="cosine")
        assert result == {"task1": {"task2": 0.3}}
        assert distance.protein_distances == result

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

    def test_compute_combined_distance_missing_distances(self, sample_tasks):
        """Test combined distance when individual distances are missing."""
        distance = TaskDistance(tasks=sample_tasks)

        # Only protein distances available initially
        distance.molecule_distances = None
        distance.protein_distances = {"task1": {"task2": 0.3}}

        def mock_compute_mol(method=None, molecule_featurizer="ecfp"):
            # Simulate what the actual method does - set instance variable AND return
            distance.molecule_distances = {"task1": {"task2": 0.5}}
            return distance.molecule_distances

        with patch.object(distance, "compute_molecule_distance", side_effect=mock_compute_mol):
            result = distance.compute_combined_distance()

            assert result["task1"]["task2"] == 0.4  # (0.5 + 0.3) / 2

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

    def test_get_distance_fallback_logic(self, sample_tasks):
        """Test get_distance method fallback logic."""
        distance = TaskDistance(tasks=sample_tasks)

        def mock_mol_compute(method=None, molecule_featurizer="ecfp"):
            distance.molecule_distances = {"mol": "result"}
            return distance.molecule_distances

        def mock_prot_compute(method=None, protein_featurizer="esm2_t33_650M_UR50D"):
            distance.protein_distances = {"prot": "result"}
            return distance.protein_distances

        # Test combined distance fallback to molecule distance
        with patch.object(distance, "compute_combined_distance", side_effect=Exception("Failed")):
            with patch.object(distance, "compute_molecule_distance", side_effect=mock_mol_compute):
                result = distance.get_distance()
                assert result == {"mol": "result"}

        # Reset for next test
        distance.molecule_distances = None

        # Test fallback to protein distance
        with patch.object(distance, "compute_combined_distance", side_effect=Exception("Failed")):
            with patch.object(distance, "compute_molecule_distance", side_effect=Exception("Failed")):
                with patch.object(distance, "compute_protein_distance", side_effect=mock_prot_compute):
                    result = distance.get_distance()
                    assert result == {"prot": "result"}

    def test_compute_all_distances_success(self, sample_tasks):
        """Test compute_all_distances method."""
        distance = TaskDistance(tasks=sample_tasks)

        mol_distances = {"task1": {"task2": 0.5}}
        prot_distances = {"task1": {"task2": 0.3}}

        def mock_mol_compute(method=None, molecule_featurizer="ecfp"):
            distance.molecule_distances = mol_distances
            return mol_distances

        def mock_prot_compute(method=None, protein_featurizer="esm2_t33_650M_UR50D"):
            distance.protein_distances = prot_distances
            return prot_distances

        with patch.object(distance, "compute_molecule_distance", side_effect=mock_mol_compute):
            with patch.object(distance, "compute_protein_distance", side_effect=mock_prot_compute):
                results = distance.compute_all_distances(combination_strategy="average")

                assert "molecule" in results
                assert "protein" in results
                assert "combined" in results
                assert results["molecule"] == mol_distances
                assert results["protein"] == prot_distances
                assert results["combined"]["task1"]["task2"] == 0.4

    def test_compute_all_distances_with_errors(self, sample_tasks):
        """Test compute_all_distances with some methods failing."""
        distance = TaskDistance(tasks=sample_tasks)

        prot_distances = {"task1": {"task2": 0.3}}

        with patch.object(distance, "compute_molecule_distance", side_effect=Exception("Molecule failed")):
            with patch.object(distance, "compute_protein_distance", return_value=prot_distances):
                results = distance.compute_all_distances()

                assert results["molecule"] == {}
                assert results["protein"] == prot_distances
                assert results["combined"] == {}

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

    def test_to_pandas_external_protein(self):
        """Test to_pandas with external protein space."""
        prot_space = np.array([[0.5, 0.6], [0.7, 0.8]])
        distance = TaskDistance(
            tasks=None,
            source_task_ids=["s1", "s2"],
            target_task_ids=["t1", "t2"],
            external_protein_space=prot_space,
        )

        df = distance.to_pandas("external_protein")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert df.index.tolist() == ["s1", "s2"]
        assert df.columns.tolist() == ["t1", "t2"]

    def test_to_pandas_computed_distances(self, sample_tasks):
        """Test to_pandas with computed distances."""
        distance = TaskDistance(tasks=sample_tasks)
        distance.molecule_distances = {"task1": {"task2": 0.5, "task3": 0.8}}

        df = distance.to_pandas("molecule")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 1)

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
    def test_load_ext_prot_distance(self, mock_pickle_load):
        """Test loading external protein distance."""
        mock_data = {
            "train_pubchem_ids": ["p1", "p2"],
            "test_pubchem_ids": ["p3", "p4"],
            "distance_matrices": np.array([[0.5, 0.6], [0.7, 0.8]]),
        }
        mock_pickle_load.return_value = mock_data

        with patch("builtins.open", create=True):
            result = TaskDistance.load_ext_prot_distance("dummy_path")

            assert result.source_task_ids == ["p1", "p2"]
            assert result.target_task_ids == ["p3", "p4"]
            assert np.array_equal(result.external_protein_space, mock_data["distance_matrices"])

    @patch("pickle.load")
    def test_load_ext_distance_missing_keys(self, mock_pickle_load):
        """Test loading external distance with missing keys."""
        mock_data = {"invalid": "data"}
        mock_pickle_load.return_value = mock_data

        with patch("builtins.open", create=True):
            with pytest.raises(ValueError) as exc_info:
                TaskDistance.load_ext_chem_distance("dummy_path")
            assert "No source task IDs found" in str(exc_info.value)

    def test_get_hopts(self, sample_tasks):
        """Test get_hopts method."""
        distance = TaskDistance(tasks=sample_tasks)

        with patch("themap.distance.task_distance.get_configure") as mock_get_configure:
            mock_get_configure.return_value = {"test": "config"}

            result = distance.get_hopts("molecule")
            mock_get_configure.assert_called_once_with("euclidean")
            assert result == {"test": "config"}

    def test_get_hopts_invalid_type(self, sample_tasks):
        """Test get_hopts with invalid data type."""
        distance = TaskDistance(tasks=sample_tasks)

        with pytest.raises(ValueError) as exc_info:
            distance.get_hopts("invalid")
        assert "Unknown data type" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
