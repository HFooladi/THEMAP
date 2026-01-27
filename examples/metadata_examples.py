"""
Example functions for creating specialized metadata datasets.

This module demonstrates how to create different types of metadata datasets
using the THEMAP metadata framework.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from dpu_utils.utils import RichPath
from numpy.typing import NDArray

from themap.data.metadata import (
    BaseMetadataDataset,
    CategoricalMetadataDataset,
    MetadataDatasets,
    NumericalMetadataDataset,
    TextMetadataDataset,
)


def create_assay_description_datasets(
    directory: Union[str, RichPath],
    task_list_file: Optional[Union[str, RichPath]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> MetadataDatasets:
    """Create metadata datasets for assay descriptions.

    This demonstrates how to create a specific metadata type using the blueprint.

    Args:
        directory: Directory containing train/valid/test subdirectories with metadata files
        task_list_file: Optional file containing list of tasks to include
        cache_dir: Optional directory for persistent caching

    Returns:
        MetadataDatasets instance configured for assay descriptions
    """
    return MetadataDatasets.from_directory(
        directory=directory,
        metadata_type="assay_description",
        dataset_class=TextMetadataDataset,
        task_list_file=task_list_file,
        cache_dir=cache_dir,
        file_pattern="*.json",
    )


def create_bioactivity_datasets(
    directory: Union[str, RichPath],
    task_list_file: Optional[Union[str, RichPath]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> MetadataDatasets:
    """Create metadata datasets for bioactivity values (numerical).

    Args:
        directory: Directory containing train/valid/test subdirectories with metadata files
        task_list_file: Optional file containing list of tasks to include
        cache_dir: Optional directory for persistent caching

    Returns:
        MetadataDatasets instance configured for bioactivity data
    """
    return MetadataDatasets.from_directory(
        directory=directory,
        metadata_type="bioactivity",
        dataset_class=NumericalMetadataDataset,
        task_list_file=task_list_file,
        cache_dir=cache_dir,
        file_pattern="*.csv",
    )


def create_target_organism_datasets(
    directory: Union[str, RichPath],
    task_list_file: Optional[Union[str, RichPath]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> MetadataDatasets:
    """Create metadata datasets for target organisms (categorical).

    Args:
        directory: Directory containing train/valid/test subdirectories with metadata files
        task_list_file: Optional file containing list of tasks to include
        cache_dir: Optional directory for persistent caching

    Returns:
        MetadataDatasets instance configured for target organism data
    """
    return MetadataDatasets.from_directory(
        directory=directory,
        metadata_type="target_organism",
        dataset_class=CategoricalMetadataDataset,
        task_list_file=task_list_file,
        cache_dir=cache_dir,
        file_pattern="*.json",
    )


@dataclass
class CustomMetadataDataset(BaseMetadataDataset):
    """Example of how to create a custom metadata dataset type.

    This shows how you can extend the system for specialized metadata types.
    Implement preprocess_data() and get_features() for your specific use case.

    Example:
        >>> dataset = CustomMetadataDataset(
        ...     task_id="CHEMBL123",
        ...     metadata_type="custom",
        ...     raw_data={"key": "value"}
        ... )
        >>> features = dataset.get_features("custom_featurizer")
    """

    def preprocess_data(self) -> Any:
        """Custom preprocessing logic."""
        return self.raw_data

    def get_features(self, featurizer_name: str, **kwargs: Any) -> NDArray[np.float32]:
        """Custom featurization logic.

        Args:
            featurizer_name: Name of the featurizer to use
            **kwargs: Additional featurizer arguments

        Returns:
            Feature vector as numpy array

        Raises:
            ValueError: If featurizer_name is not supported
        """
        self.preprocess_data()

        if featurizer_name == "custom_featurizer":
            # Implement your custom feature computation here
            # This is a placeholder that returns random features
            self.features = np.random.randn(128).astype(np.float32)
        else:
            raise ValueError(f"Unknown featurizer for custom metadata: {featurizer_name}")

        return self.features


if __name__ == "__main__":
    # Example usage
    print("Metadata Examples Module")
    print("=" * 50)
    print("\nAvailable factory functions:")
    print("  - create_assay_description_datasets()")
    print("  - create_bioactivity_datasets()")
    print("  - create_target_organism_datasets()")
    print("\nCustom dataset class:")
    print("  - CustomMetadataDataset")
