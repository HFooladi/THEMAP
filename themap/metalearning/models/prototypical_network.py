"""
Prototypical Networks for Meta-Learning on Molecular Tasks.

This module implements Prototypical Networks as described in:
"Prototypical Networks for Few-shot Learning" by Snell et al. (2017)

The implementation is adapted for molecular property prediction tasks with
support for different molecular representations (ECFP, descriptors, neural embeddings).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...utils.logging import get_logger

logger = get_logger(__name__)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot molecular property prediction.

    The network learns to map molecular features to an embedding space where
    classification can be performed by computing distances to class prototypes.

    Args:
        input_dim (int): Dimension of input molecular features
        hidden_dims (List[int]): List of hidden layer dimensions
        output_dim (int): Dimension of the embedding space
        dropout_prob (float): Dropout probability
        activation (str): Activation function ('relu', 'leaky_relu', 'gelu')
        distance_metric (str): Distance metric for prototype comparison ('euclidean', 'cosine')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 64,
        dropout_prob: float = 0.1,
        activation: str = "relu",
        distance_metric: str = "euclidean",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.distance_metric = distance_metric

        # Build the encoder network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout_prob),
                ]
            )
            prev_dim = hidden_dim

        # Final embedding layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"Initialized PrototypicalNetwork with input_dim={input_dim}, "
            f"hidden_dims={hidden_dims}, output_dim={output_dim}"
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the encoder network.

        Args:
            x (Tensor): Input molecular features of shape (batch_size, input_dim)

        Returns:
            Tensor: Embedded features of shape (batch_size, output_dim)
        """
        return self.encoder(x)

    def compute_prototypes(self, support_embeddings: Tensor, support_labels: Tensor) -> Tensor:
        """
        Compute class prototypes from support set embeddings.

        Args:
            support_embeddings (Tensor): Support set embeddings (N_support, output_dim)
            support_labels (Tensor): Support set labels (N_support,)

        Returns:
            Tensor: Class prototypes (N_classes, output_dim)
        """
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)

        prototypes = torch.zeros(n_classes, self.output_dim, device=support_embeddings.device)

        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[i] = support_embeddings[mask].mean(dim=0)

        return prototypes

    def compute_distances(self, query_embeddings: Tensor, prototypes: Tensor) -> Tensor:
        """
        Compute distances between query embeddings and prototypes.

        Args:
            query_embeddings (Tensor): Query embeddings (N_query, output_dim)
            prototypes (Tensor): Class prototypes (N_classes, output_dim)

        Returns:
            Tensor: Distances (N_query, N_classes)
        """
        if self.distance_metric == "euclidean":
            # Compute squared Euclidean distances
            distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        elif self.distance_metric == "cosine":
            # Compute cosine distances (1 - cosine similarity)
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def predict_proba(
        self,
        query_features: Tensor,
        support_features: Tensor,
        support_labels: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Predict class probabilities for query samples.

        Args:
            query_features (Tensor): Query molecular features (N_query, input_dim)
            support_features (Tensor): Support molecular features (N_support, input_dim)
            support_labels (Tensor): Support labels (N_support,)
            temperature (float): Temperature parameter for softmax

        Returns:
            Tensor: Class probabilities (N_query, N_classes)
        """
        # Encode features
        query_embeddings = self.forward(query_features)
        support_embeddings = self.forward(support_features)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels)

        # Compute distances
        distances = self.compute_distances(query_embeddings, prototypes)

        # Convert distances to probabilities
        logits = -distances / temperature
        probabilities = F.softmax(logits, dim=1)

        return probabilities

    def episodic_loss(
        self,
        support_features: Tensor,
        support_labels: Tensor,
        query_features: Tensor,
        query_labels: Tensor,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute episodic loss for meta-learning.

        Args:
            support_features (Tensor): Support set features (N_support, input_dim)
            support_labels (Tensor): Support set labels (N_support,)
            query_features (Tensor): Query set features (N_query, input_dim)
            query_labels (Tensor): Query set labels (N_query,)
            temperature (float): Temperature parameter for softmax

        Returns:
            Tuple[Tensor, Dict[str, float]]: Loss tensor and metrics dictionary
        """
        # Get predictions
        probabilities = self.predict_proba(query_features, support_features, support_labels, temperature)

        # Compute cross-entropy loss
        loss = F.cross_entropy(torch.log(probabilities + 1e-8), query_labels.long())

        # Compute accuracy
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = (predictions == query_labels).float().mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

        return loss, metrics


class PrototypicalNetworkConfig:
    """Configuration class for PrototypicalNetwork."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 64,
        dropout_prob: float = 0.1,
        activation: str = "relu",
        distance_metric: str = "euclidean",
        temperature: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.activation = activation
        self.distance_metric = distance_metric
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "dropout_prob": self.dropout_prob,
            "activation": self.activation,
            "distance_metric": self.distance_metric,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
