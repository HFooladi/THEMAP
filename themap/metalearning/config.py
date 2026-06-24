"""Configuration dataclasses for the meta-learning subsystem.

These are pure-stdlib dataclasses (no torch) so they can be imported by the CLI
and the distance-selection utilities without pulling in heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

Algorithm = Literal["proto", "maml"]


@dataclass
class EncoderConfig:
    """MLP encoder (shared trunk) configuration.

    Attributes:
        hidden_dims: Sizes of the hidden layers.
        embed_dim: Output embedding dimensionality.
        dropout: Dropout probability between layers.
        activation: Activation function name (``"relu"`` or ``"gelu"``).
    """

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    embed_dim: int = 128
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "relu"


@dataclass
class ProtoConfig:
    """Prototypical Network configuration.

    Attributes:
        distance_metric: ``"euclidean"`` (squared) or ``"cosine"``.
        temperature: Softmax temperature applied to negative distances.
    """

    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    temperature: float = 1.0


@dataclass
class MAMLConfig:
    """Model-Agnostic Meta-Learning configuration.

    Attributes:
        inner_lr: Inner-loop SGD learning rate.
        inner_steps: Number of inner-loop adaptation steps during meta-training.
        eval_inner_steps: Inner-loop steps used at evaluation/adaptation time.
        first_order: If True, use first-order MAML (FOMAML); cheaper on CPU.
    """

    inner_lr: float = 0.01
    inner_steps: int = 5
    eval_inner_steps: int = 16
    first_order: bool = True


@dataclass
class TrainConfig:
    """Meta-training loop configuration.

    Attributes:
        n_support: Support examples per episode (total across classes).
        n_query: Query examples per episode (total across classes).
        num_epochs: Number of meta-training epochs.
        episodes_per_epoch: Meta-training steps per epoch.
        meta_batch_size: Episodes aggregated per outer optimizer step.
        outer_lr: Outer-loop (meta) Adam learning rate.
        weight_decay: Outer-loop weight decay.
        grad_clip: Max gradient norm for the outer update (0 disables).
        val_episodes: Episodes used per validation pass (0 disables validation).
        patience: Early-stopping patience in epochs (0 disables).
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.
        seed: Random seed.
    """

    n_support: int = 10
    n_query: int = 15
    num_epochs: int = 50
    episodes_per_epoch: int = 100
    meta_batch_size: int = 8
    outer_lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    val_episodes: int = 50
    patience: int = 10
    device: str = "auto"
    seed: int = 42


@dataclass
class ExperimentConfig:
    """End-to-end experiment configuration for :class:`MetaLearnExperiment`.

    Attributes:
        data_dir: Directory containing ``train``/``valid``/``test`` folds.
        distance_file: Path to a saved distance file (JSON/CSV/NPZ).
        target_id: Target task id to evaluate.
        k: Number of nearest source datasets to meta-train on.
        algorithm: ``"proto"`` or ``"maml"``.
        featurizer: Molecular featurizer name (e.g. ``"ecfp"``).
        support_sizes: Low-data support set sizes to sweep on the target.
        train_shot_mode: ``"match"`` meta-trains a fresh model per support size with a
            training shot that tracks the eval ``N`` (capped to source feasibility) —
            THEMAP's low-data default. ``"fixed"`` is the FS-Mol single-model protocol:
            train one model once at ``TrainConfig.n_support`` (set ≈64 for FS-Mol parity)
            and evaluate it across all support sizes.
        query_fraction: Fraction of the target held out as a fixed query set, shared
            across all support sizes so AUROC is comparable.
        seeds: Number of repeated seeds per support size.
        n_jobs: Parallel jobs for featurization.
        output_dir: Directory to write results to.
        source_fold: Fold the source datasets live in.
        target_fold: Fold the target dataset lives in.
        encoder: Encoder configuration.
        proto: Prototypical-network configuration.
        maml: MAML configuration.
        train: Meta-training configuration.
    """

    data_dir: str
    distance_file: str
    target_id: str
    k: int = 5
    algorithm: Algorithm = "proto"
    featurizer: str = "ecfp"
    support_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    train_shot_mode: Literal["match", "fixed"] = "match"
    query_fraction: float = 0.5
    seeds: int = 5
    n_jobs: int = 8
    output_dir: Optional[str] = None
    source_fold: str = "train"
    target_fold: str = "test"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    proto: ProtoConfig = field(default_factory=ProtoConfig)
    maml: MAMLConfig = field(default_factory=MAMLConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
