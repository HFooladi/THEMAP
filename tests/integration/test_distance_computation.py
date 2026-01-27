"""
Integration tests for distance computation.

Tests distance matrix computation workflows.
"""

import numpy as np
import pytest


@pytest.mark.integration
class TestDistanceComputation:
    """Integration tests for distance computation functionality."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature arrays for testing."""
        np.random.seed(42)
        return {
            "source": [np.random.randn(128).astype(np.float32) for _ in range(3)],
            "target": [np.random.randn(128).astype(np.float32) for _ in range(2)],
            "source_ids": ["train_CHEMBL001", "train_CHEMBL002", "train_CHEMBL003"],
            "target_ids": ["test_CHEMBL004", "test_CHEMBL005"],
        }

    @pytest.fixture
    def sample_labels(self):
        """Create sample labels for testing."""
        return {
            "source": [np.array([0, 1, 0, 1]), np.array([1, 1, 0]), np.array([0, 0, 1, 1, 0])],
            "target": [np.array([1, 0, 1]), np.array([0, 1])],
        }

    def test_euclidean_distance_computation(self, sample_features):
        """Test Euclidean distance computation between feature vectors."""
        from scipy.spatial.distance import cdist

        source_features = np.vstack(sample_features["source"])
        target_features = np.vstack(sample_features["target"])

        # Compute pairwise distances
        distances = cdist(target_features, source_features, metric="euclidean")

        # Check shape
        assert distances.shape == (2, 3), f"Expected (2, 3), got {distances.shape}"

        # Check all distances are non-negative
        assert np.all(distances >= 0), "All distances should be non-negative"

        # Check diagonal when comparing same vectors
        same_distances = cdist(source_features, source_features, metric="euclidean")
        assert np.allclose(np.diag(same_distances), 0), "Self-distance should be zero"

    def test_cosine_distance_computation(self, sample_features):
        """Test Cosine distance computation between feature vectors."""
        from scipy.spatial.distance import cdist

        source_features = np.vstack(sample_features["source"])
        target_features = np.vstack(sample_features["target"])

        # Compute pairwise cosine distances
        distances = cdist(target_features, source_features, metric="cosine")

        # Check shape
        assert distances.shape == (2, 3)

        # Check distance range (cosine distance is in [0, 2])
        assert np.all(distances >= 0), "Cosine distances should be non-negative"
        assert np.all(distances <= 2), "Cosine distances should be <= 2"

    def test_distance_matrix_symmetry(self, sample_features):
        """Test that distance matrices are symmetric when source == target."""
        from scipy.spatial.distance import cdist

        features = np.vstack(sample_features["source"])
        distances = cdist(features, features, metric="euclidean")

        # Check symmetry
        assert np.allclose(distances, distances.T), "Distance matrix should be symmetric"

    @pytest.mark.slow
    def test_dataset_distance_class(self, sample_features, sample_labels):
        """Test DatasetDistance class if available."""
        try:
            from themap.distance import DatasetDistance

            # Create distance calculator
            calculator = DatasetDistance(method="euclidean")

            # Test that the class exists and can be instantiated
            assert calculator is not None
            assert calculator.method == "euclidean"
        except ImportError as e:
            pytest.skip(f"DatasetDistance not available: {e}")

    @pytest.mark.slow
    def test_metadata_distance_class(self):
        """Test MetadataDistance class if available."""
        try:
            from themap.distance import MetadataDistance

            # Test that the class exists
            calculator = MetadataDistance(method="euclidean")
            assert calculator is not None
        except ImportError as e:
            pytest.skip(f"MetadataDistance not available: {e}")

    def test_distance_methods_constants(self):
        """Test that distance method constants are properly defined."""
        try:
            from themap.distance import (
                DATASET_DISTANCE_METHODS,
                METADATA_DISTANCE_METHODS,
            )

            # Check expected methods are present
            assert "euclidean" in DATASET_DISTANCE_METHODS
            assert "cosine" in DATASET_DISTANCE_METHODS
            assert "otdd" in DATASET_DISTANCE_METHODS

            assert "euclidean" in METADATA_DISTANCE_METHODS
            assert "cosine" in METADATA_DISTANCE_METHODS
            assert "manhattan" in METADATA_DISTANCE_METHODS
        except ImportError as e:
            pytest.skip(f"Distance constants not available: {e}")


@pytest.mark.integration
class TestPrototypeDistanceComputation:
    """Integration tests for prototype-based distance computation."""

    @pytest.fixture
    def prototype_data(self):
        """Create sample prototype data for testing."""
        np.random.seed(42)

        # Create prototypes (mean features per class)
        positive_prototype = np.random.randn(128).astype(np.float32)
        negative_prototype = np.random.randn(128).astype(np.float32)

        return {
            "positive": positive_prototype,
            "negative": negative_prototype,
        }

    def test_prototype_distance(self, prototype_data):
        """Test distance between class prototypes."""
        from scipy.spatial.distance import cosine, euclidean

        pos = prototype_data["positive"]
        neg = prototype_data["negative"]

        # Compute distances
        euc_dist = euclidean(pos, neg)
        cos_dist = cosine(pos, neg)

        # Check that distances are computed (can be float or numpy scalar)
        assert isinstance(euc_dist, (float, np.floating))
        assert isinstance(cos_dist, (float, np.floating))
        assert float(euc_dist) >= 0
        assert 0 <= float(cos_dist) <= 2

    def test_prototype_from_features(self):
        """Test computing prototypes from feature arrays."""
        np.random.seed(42)

        # Create sample features with labels
        features = np.random.randn(10, 128).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])

        # Compute prototypes
        positive_features = features[labels == 1]
        negative_features = features[labels == 0]

        positive_prototype = np.mean(positive_features, axis=0)
        negative_prototype = np.mean(negative_features, axis=0)

        # Check shapes
        assert positive_prototype.shape == (128,)
        assert negative_prototype.shape == (128,)

        # Check that prototypes are different
        assert not np.allclose(positive_prototype, negative_prototype)
