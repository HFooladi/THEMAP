#!/usr/bin/env python3
"""
Example script demonstrating prototypical networks for meta-learning on molecular tasks.

NOTE: This is a placeholder example showing the intended API for the meta-learning module.
Some features shown here are still under development and may not work yet.
See themap/metalearning/__init__.py for currently available exports.

This script shows how to:
1. Load tasks and create meta-learning splits
2. Configure and train a prototypical network
3. Evaluate the trained model on test tasks
4. Analyze results
"""

import logging
from pathlib import Path

# Import THEMAP modules
# Note: Some of these imports are placeholders for features under development
try:
    from themap.data.tasks import Tasks
    from themap.metalearning import (
        EvaluationConfig,
        MetaLearningEvaluator,
        MetaLearningTrainer,
        PrototypicalNetwork,
        TrainingConfig,
        create_meta_splits,
    )

    # These are placeholder imports for future functionality
    PrototypicalNetworkConfig = None  # Placeholder - use PrototypicalNetwork directly
    create_task_folders = None  # Placeholder - not yet implemented
    load_tasks_from_directory = None  # Placeholder - use Tasks.from_directory instead
except ImportError as e:
    raise ImportError(
        f"Meta-learning module not fully available: {e}. "
        "Some features in this example are still under development."
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting prototypical networks meta-learning example")

    # Configuration
    data_dir = Path("datasets")
    output_dir = Path("metalearning_example_output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load tasks
    logger.info("Loading tasks from datasets")
    train_tasks = load_tasks_from_directory(data_dir / "train")
    valid_tasks = load_tasks_from_directory(data_dir / "valid")
    test_tasks = load_tasks_from_directory(data_dir / "test")

    all_tasks = train_tasks + valid_tasks + test_tasks
    logger.info(f"Loaded {len(all_tasks)} total tasks")

    # Step 2: Create meta-learning splits
    logger.info("Creating meta-learning task splits")
    task_splits = create_meta_splits(
        tasks=all_tasks,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
        min_samples_per_class=10,  # Ensure tasks have enough samples
    )

    logger.info(f"Task splits: {task_splits.summary()}")

    # Create folder structure
    _meta_folders = create_task_folders(
        task_splits=task_splits,
        base_dir=output_dir / "task_splits",
    )

    # Step 3: Determine input dimensions
    # Sample a task to get feature dimensions
    sample_task = task_splits.train_tasks[0]
    sample_featurizer_name = "ecfp"

    # Get feature dimensions from a sample molecule
    # datapoints returns list of dicts, not objects
    if sample_task.molecule_dataset and sample_task.molecule_dataset.datapoints:
        sample_datapoint = sample_task.molecule_dataset.datapoints[0]
        from themap.utils.featurizer_utils import get_featurizer

        featurizer = get_featurizer(sample_featurizer_name)
        sample_features = featurizer(sample_datapoint["smiles"])
        input_dim = len(sample_features)
        logger.info(f"Using input dimension: {input_dim}")
    else:
        input_dim = 2048  # Default ECFP dimension
        logger.info(f"Using default input dimension: {input_dim}")

    # Step 4: Configure model
    model_config = PrototypicalNetworkConfig(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=64,
        dropout_prob=0.1,
        activation="relu",
        distance_metric="euclidean",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    # Step 5: Configure training
    training_config = TrainingConfig(
        model_config=model_config,
        # Episode configuration
        n_way=2,  # Binary classification
        n_support=5,  # 5-shot learning
        n_query=15,  # 15 query samples
        featurizer_name=sample_featurizer_name,
        # Training parameters
        num_epochs=50,
        episodes_per_epoch=100,
        val_episodes_per_epoch=50,
        batch_size=4,
        # Optimization
        learning_rate=1e-3,
        lr_scheduler="step",
        lr_step_size=15,
        lr_gamma=0.5,
        gradient_clip=1.0,
        # Validation and early stopping
        val_frequency=5,
        early_stopping_patience=10,
        # Output
        output_dir=str(output_dir / "training"),
        experiment_name="prototypical_network_example",
        random_seed=42,
    )

    # Step 6: Train model
    logger.info("Starting meta-learning training")
    trainer = MetaLearningTrainer(
        config=training_config,
        task_splits=task_splits,
    )

    training_history = trainer.train()
    logger.info("Training completed successfully")

    # Step 7: Evaluate model
    logger.info("Starting evaluation on test tasks")

    evaluation_config = EvaluationConfig(
        n_way=training_config.n_way,
        n_support=training_config.n_support,
        n_query=training_config.n_query,
        featurizer_name=training_config.featurizer_name,
        num_episodes_per_task=100,
        per_task_analysis=True,
        task_difficulty_analysis=True,
        output_dir=str(output_dir / "evaluation"),
        random_seed=42,
    )

    evaluator = MetaLearningEvaluator(
        model=trainer.model,
        config=evaluation_config,
        task_splits=task_splits,
    )

    evaluation_results = evaluator.evaluate()

    # Step 8: Print results
    print("\n" + "=" * 60)
    print("PROTOTYPICAL NETWORKS META-LEARNING RESULTS")
    print("=" * 60)

    print("\nDataset Summary:")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Train tasks: {len(task_splits.train_tasks)}")
    print(f"  Validation tasks: {len(task_splits.val_tasks)}")
    print(f"  Test tasks: {len(task_splits.test_tasks)}")

    print("\nExperiment Configuration:")
    print(f"  {training_config.n_way}-way {training_config.n_support}-shot learning")
    print(f"  Feature type: {training_config.featurizer_name}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Model output dimension: {model_config.output_dim}")
    print(f"  Distance metric: {model_config.distance_metric}")

    print("\nTraining Results:")
    if training_history:
        final_train_acc = training_history["train_accuracy"][-1] if training_history["train_accuracy"] else 0
        final_val_acc = training_history["val_accuracy"][-1] if training_history["val_accuracy"] else 0
        print(f"  Final training accuracy: {final_train_acc:.4f}")
        print(f"  Final validation accuracy: {final_val_acc:.4f}")
        print(f"  Best validation accuracy: {trainer.best_val_accuracy:.4f}")

    print("\nTest Results:")
    overall_results = evaluation_results["overall"]
    print(f"  Test accuracy: {overall_results['accuracy']:.4f} ± {overall_results['accuracy_ci']:.4f}")
    print(f"  Test loss: {overall_results['loss']:.4f}")
    print(f"  Number of test episodes: {overall_results['num_episodes']}")

    # Task difficulty analysis
    if "difficulty_analysis" in evaluation_results and evaluation_results["difficulty_analysis"]:
        difficulty_analysis = evaluation_results["difficulty_analysis"]
        if difficulty_analysis.get("correlation") is not None:
            print("\nTask Difficulty Analysis:")
            print(f"  Correlation between hardness and accuracy: {difficulty_analysis['correlation']:.4f}")
            print(f"  P-value: {difficulty_analysis['correlation_pvalue']:.4f}")

    print(f"\nOutput Directory: {output_dir}")
    print(f"  Training logs: {output_dir / 'training'}")
    print(f"  Evaluation results: {output_dir / 'evaluation'}")
    print(f"  Task splits: {output_dir / 'task_splits'}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
