from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from themap.data.exceptions import FeaturizationError, InvalidSMILESError
from themap.data.molecule_datapoint import MoleculeDatapoint


class TestMoleculeDatapointInitialization:
    """Test initialization and validation of MoleculeDatapoint."""

    def test_valid_initialization_with_all_params(self):
        """Test valid initialization with all parameters."""
        datapoint = MoleculeDatapoint(task_id="test_task", smiles="CCO", bool_label=True, numeric_label=0.8)
        assert datapoint.task_id == "test_task"
        assert datapoint.smiles == "CCO"
        assert datapoint.bool_label is True
        assert datapoint.numeric_label == 0.8
        assert datapoint._rdkit_mol is None

    def test_valid_initialization_without_numeric_label(self):
        """Test valid initialization without numeric_label."""
        datapoint = MoleculeDatapoint(task_id="classification_task", smiles="c1ccccc1", bool_label=False)
        assert datapoint.task_id == "classification_task"
        assert datapoint.smiles == "c1ccccc1"
        assert datapoint.bool_label is False
        assert datapoint.numeric_label is None

    def test_invalid_task_id_type(self):
        """Test that non-string task_id raises TypeError."""
        with pytest.raises(TypeError, match="task_id must be a string"):
            MoleculeDatapoint(task_id=123, smiles="CCO", bool_label=True)

    def test_invalid_smiles_type(self):
        """Test that non-string smiles raises TypeError."""
        with pytest.raises(TypeError, match="smiles must be a string"):
            MoleculeDatapoint(task_id="test", smiles=123, bool_label=True)

    def test_invalid_bool_label_type(self):
        """Test that non-boolean bool_label raises TypeError."""
        with pytest.raises(TypeError, match="bool_label must be a boolean"):
            MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=1)

    def test_invalid_numeric_label_type(self):
        """Test that invalid numeric_label type raises TypeError."""
        with pytest.raises(TypeError, match="numeric_label must be a number or None"):
            MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True, numeric_label="invalid")

    def test_empty_smiles_string(self):
        """Test that empty SMILES string raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            MoleculeDatapoint(task_id="test", smiles="", bool_label=True)

    def test_whitespace_only_smiles(self):
        """Test that whitespace-only SMILES raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            MoleculeDatapoint(task_id="test", smiles="   ", bool_label=True)

    def test_invalid_smiles_format(self):
        """Test that invalid SMILES format raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            MoleculeDatapoint(task_id="test", smiles="invalid_smiles_xyz", bool_label=True)

    def test_numeric_label_accepts_int(self):
        """Test that numeric_label accepts integer values."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True, numeric_label=5)
        assert datapoint.numeric_label == 5

    def test_numeric_label_accepts_float(self):
        """Test that numeric_label accepts float values."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True, numeric_label=5.5)
        assert datapoint.numeric_label == 5.5


class TestMoleculeDatapointRepresentation:
    """Test string representation of MoleculeDatapoint."""

    def test_repr_with_numeric_label(self):
        """Test __repr__ method with numeric_label."""
        datapoint = MoleculeDatapoint(
            task_id="task_id", smiles="c1ccccc1", bool_label=True, numeric_label=1.0
        )
        expected = "MoleculeDatapoint(task_id=task_id, smiles=c1ccccc1, bool_label=True, numeric_label=1.0)"
        assert str(datapoint) == expected

    def test_repr_without_numeric_label(self):
        """Test __repr__ method without numeric_label."""
        datapoint = MoleculeDatapoint(task_id="classification", smiles="CCO", bool_label=False)
        expected = (
            "MoleculeDatapoint(task_id=classification, smiles=CCO, bool_label=False, numeric_label=None)"
        )
        assert str(datapoint) == expected


class TestMoleculeDatapointSerialization:
    """Test serialization methods of MoleculeDatapoint."""

    def test_to_dict_with_all_fields(self):
        """Test to_dict method with all fields."""
        datapoint = MoleculeDatapoint(task_id="test_task", smiles="CCO", bool_label=True, numeric_label=0.8)
        result = datapoint.to_dict()
        expected = {"task_id": "test_task", "smiles": "CCO", "bool_label": True, "numeric_label": 0.8}
        assert result == expected

    def test_to_dict_without_numeric_label(self):
        """Test to_dict method without numeric_label."""
        datapoint = MoleculeDatapoint(task_id="test_task", smiles="CCO", bool_label=False)
        result = datapoint.to_dict()
        expected = {"task_id": "test_task", "smiles": "CCO", "bool_label": False, "numeric_label": None}
        assert result == expected

    def test_from_dict_with_all_fields(self):
        """Test from_dict method with all fields."""
        data = {"task_id": "test_task", "smiles": "CCO", "bool_label": True, "numeric_label": 0.8}
        datapoint = MoleculeDatapoint.from_dict(data)
        assert datapoint.task_id == "test_task"
        assert datapoint.smiles == "CCO"
        assert datapoint.bool_label is True
        assert datapoint.numeric_label == 0.8

    def test_from_dict_without_numeric_label(self):
        """Test from_dict method without numeric_label."""
        data = {"task_id": "test_task", "smiles": "CCO", "bool_label": False}
        datapoint = MoleculeDatapoint.from_dict(data)
        assert datapoint.task_id == "test_task"
        assert datapoint.smiles == "CCO"
        assert datapoint.bool_label is False
        assert datapoint.numeric_label is None

    def test_roundtrip_serialization(self):
        """Test that serialization roundtrip preserves data."""
        original = MoleculeDatapoint(
            task_id="roundtrip_test", smiles="c1ccccc1", bool_label=True, numeric_label=2.5
        )
        data = original.to_dict()
        restored = MoleculeDatapoint.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.smiles == original.smiles
        assert restored.bool_label == original.bool_label
        assert restored.numeric_label == original.numeric_label


class TestMoleculeDatapointProperties:
    """Test molecular properties of MoleculeDatapoint."""

    def test_rdkit_mol_property_lazy_loading(self):
        """Test that rdkit_mol property is lazily loaded."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)
        assert datapoint._rdkit_mol is None
        mol = datapoint.rdkit_mol
        assert mol is not None
        assert datapoint._rdkit_mol is not None
        # Second access should return cached molecule
        assert datapoint.rdkit_mol is mol

    def test_number_of_atoms_ethanol(self):
        """Test number_of_atoms for ethanol (CCO)."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)
        assert datapoint.number_of_atoms == 3  # 2 carbons + 1 oxygen

    def test_number_of_atoms_benzene(self):
        """Test number_of_atoms for benzene."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="c1ccccc1", bool_label=True)
        assert datapoint.number_of_atoms == 6  # 6 carbons

    def test_number_of_bonds_ethanol(self):
        """Test number_of_bonds for ethanol."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)
        assert datapoint.number_of_bonds == 2  # C-C and C-O

    def test_number_of_bonds_benzene(self):
        """Test number_of_bonds for benzene."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="c1ccccc1", bool_label=True)
        assert datapoint.number_of_bonds == 6  # 6 aromatic bonds

    def test_molecular_weight_ethanol(self):
        """Test molecular_weight for ethanol."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)
        # Ethanol: C2H6O, MW ≈ 46.07
        assert abs(datapoint.molecular_weight - 46.07) < 0.1

    def test_molecular_weight_benzene(self):
        """Test molecular_weight for benzene."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="c1ccccc1", bool_label=True)
        # Benzene: C6H6, MW ≈ 78.11
        assert abs(datapoint.molecular_weight - 78.11) < 0.1

    def test_logp_calculation(self):
        """Test LogP calculation."""
        datapoint = MoleculeDatapoint(
            task_id="test",
            smiles="CCO",  # Ethanol has negative LogP
            bool_label=True,
        )
        logp = datapoint.logp
        assert isinstance(logp, float)
        assert logp < 0  # Ethanol is hydrophilic

    def test_num_rotatable_bonds_ethanol(self):
        """Test num_rotatable_bonds for ethanol."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)
        assert datapoint.num_rotatable_bonds == 0  # C-O bond is not considered rotatable by RDKit

    def test_num_rotatable_bonds_butanol(self):
        """Test num_rotatable_bonds for butanol."""
        datapoint = MoleculeDatapoint(
            task_id="test",
            smiles="CCCCO",  # 1-butanol
            bool_label=True,
        )
        assert datapoint.num_rotatable_bonds == 2  # C-C-C-C chain has 2 rotatable bonds

    def test_smiles_canonical_normalization(self):
        """Test that canonical SMILES normalizes input."""
        datapoint = MoleculeDatapoint(
            task_id="test",
            smiles="C1=CC=CC=C1",  # Non-canonical benzene
            bool_label=True,
        )
        canonical = datapoint.smiles_canonical
        assert canonical in ["c1ccccc1", "C1=CC=CC=C1"]  # Both are valid canonical forms

    def test_property_failure_handling(self):
        """Test that properties handle molecule creation failure."""
        # Create a valid datapoint first
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        # Mock rdkit_mol property to return None
        with patch.object(type(datapoint), "rdkit_mol", new_callable=lambda: property(lambda self: None)):
            with pytest.raises(ValueError, match="Failed to create RDKit molecule"):
                _ = datapoint.number_of_atoms

            with pytest.raises(ValueError, match="Failed to create RDKit molecule"):
                _ = datapoint.number_of_bonds

            with pytest.raises(ValueError, match="Failed to create RDKit molecule"):
                _ = datapoint.molecular_weight

            with pytest.raises(ValueError, match="Failed to create RDKit molecule"):
                _ = datapoint.logp

            with pytest.raises(ValueError, match="Failed to create RDKit molecule"):
                _ = datapoint.num_rotatable_bonds

            with pytest.raises(ValueError, match="Failed to create RDKit molecule"):
                _ = datapoint.smiles_canonical


class TestMoleculeDatapointFingerprint:
    """Test fingerprint functionality of MoleculeDatapoint."""

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_fingerprint_success(self, mock_get_featurizer, mock_get_cache):
        """Test successful fingerprint generation."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_get_cache.return_value = mock_cache

        mock_featurizer = MagicMock()
        fake_fingerprint = np.array([1, 0, 1, 0])
        mock_featurizer.return_value = [fake_fingerprint]
        mock_get_featurizer.return_value = mock_featurizer

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_fingerprint()

        assert np.array_equal(result, fake_fingerprint)
        mock_get_featurizer.assert_called_once_with("ecfp")
        mock_featurizer.assert_called_once_with("CCO")
        mock_cache.store.assert_called_once()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    def test_get_fingerprint_cache_hit(self, mock_get_cache):
        """Test fingerprint retrieval from cache."""
        cached_fingerprint = np.array([0, 1, 0, 1])
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_fingerprint
        mock_get_cache.return_value = mock_cache

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_fingerprint()

        assert np.array_equal(result, cached_fingerprint)
        mock_cache.store.assert_not_called()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_fingerprint_force_recompute(self, mock_get_featurizer, mock_get_cache):
        """Test fingerprint force recompute bypasses cache."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = np.array([1, 1, 1, 1])  # Cached value
        mock_get_cache.return_value = mock_cache

        mock_featurizer = MagicMock()
        fresh_fingerprint = np.array([0, 0, 0, 0])
        mock_featurizer.return_value = [fresh_fingerprint]
        mock_get_featurizer.return_value = mock_featurizer

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_fingerprint(force_recompute=True)

        assert np.array_equal(result, fresh_fingerprint)
        mock_get_featurizer.assert_called_once_with("ecfp")
        mock_cache.store.assert_called_once()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_fingerprint_featurizer_returns_none(self, mock_get_featurizer, mock_get_cache):
        """Test fingerprint when featurizer returns None."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_featurizer = MagicMock()
        mock_featurizer.return_value = None
        mock_get_featurizer.return_value = mock_featurizer

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_fingerprint()

        assert result is None
        mock_cache.store.assert_not_called()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_fingerprint_exception(self, mock_get_featurizer, mock_get_cache):
        """Test fingerprint generation exception handling."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_get_featurizer.side_effect = Exception("Featurizer error")

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        with pytest.raises(FeaturizationError):
            datapoint.get_fingerprint()


class TestMoleculeDatapointFeatures:
    """Test features functionality of MoleculeDatapoint."""

    def test_get_features_none_featurizer(self):
        """Test get_features with None featurizer_name."""
        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_features(featurizer_name=None)
        assert result is None

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_features_success(self, mock_get_featurizer, mock_get_cache):
        """Test successful feature generation."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_get_cache.return_value = mock_cache

        mock_featurizer = MagicMock()
        fake_features = np.array([0.1, 0.2, 0.3])
        mock_featurizer.return_value = [fake_features]
        mock_get_featurizer.return_value = mock_featurizer

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_features(featurizer_name="custom_featurizer")

        assert np.array_equal(result, fake_features)
        mock_get_featurizer.assert_called_once_with("custom_featurizer")
        mock_featurizer.assert_called_once_with("CCO")
        mock_cache.store.assert_called_once()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    def test_get_features_cache_hit(self, mock_get_cache):
        """Test feature retrieval from cache."""
        cached_features = np.array([0.5, 0.6, 0.7])
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_features
        mock_get_cache.return_value = mock_cache

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_features(featurizer_name="cached_featurizer")

        assert np.array_equal(result, cached_features)
        mock_cache.store.assert_not_called()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_features_force_recompute(self, mock_get_featurizer, mock_get_cache):
        """Test features force recompute bypasses cache."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = np.array([1.0, 1.0, 1.0])  # Cached value
        mock_get_cache.return_value = mock_cache

        mock_featurizer = MagicMock()
        fresh_features = np.array([2.0, 2.0, 2.0])
        mock_featurizer.return_value = [fresh_features]
        mock_get_featurizer.return_value = mock_featurizer

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_features(featurizer_name="test_featurizer", force_recompute=True)

        assert np.array_equal(result, fresh_features)
        mock_get_featurizer.assert_called_once_with("test_featurizer")
        mock_cache.store.assert_called_once()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_features_featurizer_not_found(self, mock_get_featurizer, mock_get_cache):
        """Test features when featurizer is not found."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_get_featurizer.return_value = None

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_features(featurizer_name="nonexistent_featurizer")

        assert result is None
        mock_cache.store.assert_not_called()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_features_featurizer_returns_none(self, mock_get_featurizer, mock_get_cache):
        """Test features when featurizer returns None."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_featurizer = MagicMock()
        mock_featurizer.return_value = None
        mock_get_featurizer.return_value = mock_featurizer

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        result = datapoint.get_features(featurizer_name="failing_featurizer")

        assert result is None
        mock_cache.store.assert_not_called()

    @patch("themap.data.molecule_datapoint.get_global_feature_cache")
    @patch("themap.data.molecule_datapoint.get_featurizer")
    def test_get_features_exception(self, mock_get_featurizer, mock_get_cache):
        """Test features generation exception handling."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache

        mock_get_featurizer.side_effect = Exception("Feature generation error")

        datapoint = MoleculeDatapoint(task_id="test", smiles="CCO", bool_label=True)

        with pytest.raises(FeaturizationError):
            datapoint.get_features(featurizer_name="error_featurizer")


class TestMoleculeDatapointEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_different_smiles_representations_same_molecule(self):
        """Test that different SMILES for same molecule give consistent properties."""
        # Different representations of benzene
        datapoint1 = MoleculeDatapoint(task_id="test", smiles="c1ccccc1", bool_label=True)
        datapoint2 = MoleculeDatapoint(task_id="test", smiles="C1=CC=CC=C1", bool_label=True)

        # Should have same molecular properties
        assert datapoint1.number_of_atoms == datapoint2.number_of_atoms
        assert datapoint1.number_of_bonds == datapoint2.number_of_bonds
        assert abs(datapoint1.molecular_weight - datapoint2.molecular_weight) < 0.001

    def test_stereochemistry_handling(self):
        """Test handling of stereochemistry in SMILES."""
        # Test chiral center
        datapoint = MoleculeDatapoint(
            task_id="test",
            smiles="C[C@H](O)C",  # S-2-propanol
            bool_label=True,
        )

        # Should create valid molecule
        assert datapoint.rdkit_mol is not None
        assert datapoint.number_of_atoms == 4  # 3 carbons + 1 oxygen

    def test_large_molecule(self):
        """Test with a larger, more complex molecule."""
        # Caffeine
        caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        datapoint = MoleculeDatapoint(task_id="test", smiles=caffeine_smiles, bool_label=True)

        # Should handle complex molecule correctly
        assert datapoint.rdkit_mol is not None
        assert datapoint.number_of_atoms > 10
        assert datapoint.molecular_weight > 100

    def test_aromatic_vs_aliphatic(self):
        """Test distinction between aromatic and aliphatic systems."""
        # Benzene (aromatic)
        aromatic = MoleculeDatapoint(task_id="test", smiles="c1ccccc1", bool_label=True)

        # Cyclohexane (aliphatic)
        aliphatic = MoleculeDatapoint(task_id="test", smiles="C1CCCCC1", bool_label=True)

        # Both should have 6 carbons but different properties
        assert aromatic.number_of_atoms == aliphatic.number_of_atoms == 6
        # Aromatic should have lower molecular weight due to unsaturation (fewer hydrogens)
        assert aromatic.molecular_weight < aliphatic.molecular_weight
        # Difference should be at least 6 (benzene C6H6 vs cyclohexane C6H12)
        assert abs(aromatic.molecular_weight - aliphatic.molecular_weight) > 5


# Legacy tests for compatibility
def test_MoleculeDatapoint(datapoint_molecule):
    """Test the MoleculeDatapoint class functionality."""
    # Test the __repr__ method
    assert (
        str(datapoint_molecule)
        == "MoleculeDatapoint(task_id=task_id, smiles=c1ccccc1, bool_label=True, numeric_label=1.0)"
    )

    # Test the number_of_atoms method
    assert datapoint_molecule.number_of_atoms == 6

    # Test the number_of_bonds method
    assert datapoint_molecule.number_of_bonds == 6

    # Test the molecular_weight method
    assert round(datapoint_molecule.molecular_weight) == 78


def test_MoleculeDatapoint_validation():
    """Test input validation in MoleculeDatapoint."""
    # Test valid initialization
    datapoint = MoleculeDatapoint(task_id="test_task", smiles="c1ccccc1", bool_label=True, numeric_label=1.0)
    assert datapoint.task_id == "test_task"
    assert datapoint.smiles == "c1ccccc1"
    assert datapoint.bool_label is True
    assert datapoint.numeric_label == 1.0

    # Test invalid task_id
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id=123,  # Should be string
            smiles="c1ccccc1",
            bool_label=True,
        )

    # Test invalid smiles
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id="test_task",
            smiles=123,  # Should be string
            bool_label=True,
        )

    # Test invalid bool_label
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id="test_task",
            smiles="c1ccccc1",
            bool_label=1,  # Should be bool
        )

    # Test invalid numeric_label
    with pytest.raises(TypeError):
        MoleculeDatapoint(
            task_id="test_task",
            smiles="c1ccccc1",
            bool_label=True,
            numeric_label="invalid",  # Should be number or None
        )
