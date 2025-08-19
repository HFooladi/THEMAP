"""
Tests for PrototypicalNetwork model.
"""

import pytest
import torch
import torch.nn as nn

from themap.metalearning.models.prototypical_network import PrototypicalNetwork, PrototypicalNetworkConfig


class TestPrototypicalNetwork:
    """Test suite for PrototypicalNetwork."""

    @pytest.fixture
    def model_config(self):
        """Basic model configuration."""
        return PrototypicalNetworkConfig(
            input_dim=100,
            hidden_dims=[64, 32],
            output_dim=16,
            dropout_prob=0.1,
            activation="relu",
            distance_metric="euclidean",
        )

    @pytest.fixture
    def model(self, model_config):
        """Initialize model."""
        return PrototypicalNetwork(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.hidden_dims,
            output_dim=model_config.output_dim,
            dropout_prob=model_config.dropout_prob,
            activation=model_config.activation,
            distance_metric=model_config.distance_metric,
        )

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        batch_size = 10
        input_dim = 100
        n_classes = 2
        n_support = 5
        n_query = 15

        support_features = torch.randn(n_classes * n_support, input_dim)
        support_labels = torch.tensor([i for i in range(n_classes) for _ in range(n_support)])
        query_features = torch.randn(n_classes * n_query, input_dim)
        query_labels = torch.tensor([i for i in range(n_classes) for _ in range(n_query)])

        return {
            "support_features": support_features,
            "support_labels": support_labels,
            "query_features": query_features,
            "query_labels": query_labels,
        }

    def test_model_initialization(self, model_config):
        """Test model initialization."""
        model = PrototypicalNetwork(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.hidden_dims,
            output_dim=model_config.output_dim,
        )

        assert isinstance(model.encoder, nn.Sequential)
        assert model.input_dim == model_config.input_dim
        assert model.output_dim == model_config.output_dim
        assert model.distance_metric == "euclidean"

    def test_forward_pass(self, model, sample_data):
        """Test forward pass through encoder."""
        features = sample_data["support_features"]
        embeddings = model(features)

        assert embeddings.shape == (features.shape[0], model.output_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_compute_prototypes(self, model, sample_data):
        """Test prototype computation."""
        embeddings = model(sample_data["support_features"])
        labels = sample_data["support_labels"]

        prototypes = model.compute_prototypes(embeddings, labels)

        unique_labels = torch.unique(labels)
        assert prototypes.shape == (len(unique_labels), model.output_dim)
        assert not torch.isnan(prototypes).any()

    def test_compute_distances_euclidean(self, model, sample_data):
        """Test Euclidean distance computation."""
        model.distance_metric = "euclidean"

        support_embeddings = model(sample_data["support_features"])
        query_embeddings = model(sample_data["query_features"])
        prototypes = model.compute_prototypes(support_embeddings, sample_data["support_labels"])

        distances = model.compute_distances(query_embeddings, prototypes)

        assert distances.shape == (query_embeddings.shape[0], prototypes.shape[0])
        assert (distances >= 0).all()  # Euclidean distances are non-negative
        assert not torch.isnan(distances).any()

    def test_compute_distances_cosine(self, model, sample_data):
        """Test cosine distance computation."""
        model.distance_metric = "cosine"

        support_embeddings = model(sample_data["support_features"])
        query_embeddings = model(sample_data["query_features"])
        prototypes = model.compute_prototypes(support_embeddings, sample_data["support_labels"])

        distances = model.compute_distances(query_embeddings, prototypes)

        assert distances.shape == (query_embeddings.shape[0], prototypes.shape[0])
        assert (distances >= 0).all() and (distances <= 2).all()  # Cosine distances in [0, 2]
        assert not torch.isnan(distances).any()

    def test_predict_proba(self, model, sample_data):
        """Test probability prediction."""
        probabilities = model.predict_proba(
            sample_data["query_features"],
            sample_data["support_features"],
            sample_data["support_labels"],
        )

        n_classes = len(torch.unique(sample_data["support_labels"]))
        assert probabilities.shape == (sample_data["query_features"].shape[0], n_classes)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.shape[0]))
        assert (probabilities >= 0).all() and (probabilities <= 1).all()

    def test_episodic_loss(self, model, sample_data):
        """Test episodic loss computation."""
        loss, metrics = model.episodic_loss(
            sample_data["support_features"],
            sample_data["support_labels"],
            sample_data["query_features"],
            sample_data["query_labels"],
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert not torch.isnan(loss)
        assert loss.item() >= 0

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_different_activations(self, model_config):
        """Test different activation functions."""
        activations = ["relu", "leaky_relu", "gelu", "tanh"]

        for activation in activations:
            model = PrototypicalNetwork(
                input_dim=model_config.input_dim,
                hidden_dims=model_config.hidden_dims,
                output_dim=model_config.output_dim,
                activation=activation,
            )

            # Test that model can process data
            features = torch.randn(10, model_config.input_dim)
            embeddings = model(features)
            assert embeddings.shape == (10, model_config.output_dim)

    def test_invalid_activation(self, model_config):
        """Test invalid activation function."""
        with pytest.raises(ValueError, match="Unknown activation"):
            PrototypicalNetwork(
                input_dim=model_config.input_dim,
                hidden_dims=model_config.hidden_dims,
                output_dim=model_config.output_dim,
                activation="invalid_activation",
            )

    def test_invalid_distance_metric(self, model, sample_data):
        """Test invalid distance metric."""
        model.distance_metric = "invalid_metric"

        support_embeddings = model(sample_data["support_features"])
        query_embeddings = model(sample_data["query_features"])
        prototypes = model.compute_prototypes(support_embeddings, sample_data["support_labels"])

        with pytest.raises(ValueError, match="Unknown distance metric"):
            model.compute_distances(query_embeddings, prototypes)

    def test_gradient_flow(self, model, sample_data):
        """Test gradient flow through the model."""
        model.train()

        loss, _ = model.episodic_loss(
            sample_data["support_features"],
            sample_data["support_labels"],
            sample_data["query_features"],
            sample_data["query_labels"],
        )

        loss.backward()

        # Check that gradients exist and are not zero
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found"

    def test_config_to_dict(self, model_config):
        """Test configuration serialization."""
        config_dict = model_config.to_dict()

        expected_keys = [
            "input_dim",
            "hidden_dims",
            "output_dim",
            "dropout_prob",
            "activation",
            "distance_metric",
            "temperature",
            "learning_rate",
            "weight_decay",
        ]

        for key in expected_keys:
            assert key in config_dict

        assert config_dict["input_dim"] == model_config.input_dim
        assert config_dict["hidden_dims"] == model_config.hidden_dims
