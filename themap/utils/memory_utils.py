import mmap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from themap.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryEfficientFeatureStorage:
    """Memory-efficient feature storage using memory mapping and compression.

    This class provides efficient storage and retrieval of molecular features with:
    - Memory mapping for large datasets that don't fit in RAM
    - Automatic compression for storage space optimization
    - Lazy loading to reduce memory footprint
    - Efficient batch operations
    """

    def __init__(
        self,
        storage_dir: Union[str, Path],
        use_memory_mapping: bool = True,
        compression_level: int = 6,
        max_memory_cache_mb: int = 1024,
    ):
        """Initialize memory-efficient feature storage.

        Args:
            storage_dir: Directory for storing feature files
            use_memory_mapping: Whether to use memory mapping for large files
            compression_level: Compression level (0-9, 0=no compression, 9=max compression)
            max_memory_cache_mb: Maximum memory cache size in MB
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.use_memory_mapping = use_memory_mapping
        self.compression_level = compression_level
        self.max_memory_cache_bytes = max_memory_cache_mb * 1024 * 1024

        # Memory management
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._memory_usage = 0
        self._access_order: List[str] = []

        # Memory-mapped file handles
        self._mmap_handles: Dict[str, mmap.mmap] = {}
        self._file_handles: Dict[str, Any] = {}

        logger.info(f"Initialized MemoryEfficientFeatureStorage at {self.storage_dir}")

    def store_features(
        self, identifier: str, features: np.ndarray, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store features with efficient compression and organization.

        Args:
            identifier: Unique identifier for the features
            features: Feature array to store
            metadata: Optional metadata to store with features

        Returns:
            Path to stored file
        """
        try:
            # Create filename
            filename = f"{identifier}.npz"
            filepath = self.storage_dir / filename

            # Prepare data for storage
            save_data: Dict[str, np.ndarray] = {"features": features}
            if metadata:
                # Convert metadata to numpy array if it's not already
                if isinstance(metadata, dict):
                    metadata_array = np.array([metadata], dtype=object)
                elif not isinstance(metadata, np.ndarray):  # type: ignore
                    metadata_array = np.asarray(metadata)
                else:
                    metadata_array = metadata
                save_data["metadata"] = metadata_array

            # Save with compression
            if self.compression_level > 0:
                if "metadata" in save_data:
                    np.savez_compressed(
                        filepath, features=save_data["features"], metadata=save_data["metadata"]
                    )
                else:
                    np.savez_compressed(filepath, features=save_data["features"])
            else:
                if "metadata" in save_data:
                    np.savez(filepath, features=save_data["features"], metadata=save_data["metadata"])
                else:
                    np.savez(filepath, features=save_data["features"])

            # Update memory cache if within limits
            feature_size = features.nbytes
            if feature_size <= self.max_memory_cache_bytes:
                self._add_to_memory_cache(identifier, features)

            logger.debug(f"Stored features {identifier} ({feature_size / 1024 / 1024:.2f} MB)")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to store features {identifier}: {e}")
            return str(filepath)  # Return filepath even if error occurs

    def load_features(
        self, identifier: str, use_memory_mapping: Optional[bool] = None
    ) -> Optional[np.ndarray]:
        """Load features with optional memory mapping.

        Args:
            identifier: Identifier for the features to load
            use_memory_mapping: Override the default memory mapping setting

        Returns:
            Loaded features array or None if not found
        """
        # Check memory cache first
        if identifier in self._memory_cache:
            self._update_access_order(identifier)
            logger.debug(f"Memory cache hit for {identifier}")
            return self._memory_cache[identifier]

        # Try to load from disk
        filename = f"{identifier}.npz"
        filepath = self.storage_dir / filename

        if not filepath.exists():
            logger.debug(f"Features file not found: {filepath}")
            return None

        try:
            # Determine whether to use memory mapping
            use_mmap = use_memory_mapping if use_memory_mapping is not None else self.use_memory_mapping

            if use_mmap:
                return self._load_with_memory_mapping(identifier, filepath)
            else:
                return self._load_into_memory(identifier, filepath)

        except Exception as e:
            logger.error(f"Failed to load features {identifier}: {e}")
            return None

    def _load_with_memory_mapping(self, identifier: str, filepath: Path) -> np.ndarray:
        """Load features using memory mapping for large files.

        Args:
            identifier: Feature identifier
            filepath: Path to the feature file

        Returns:
            Memory-mapped feature array
        """
        if identifier in self._mmap_handles:
            # Reuse existing memory map
            logger.debug(f"Reusing memory map for {identifier}")
            data = np.load(self._file_handles[identifier])
            features: np.ndarray = data["features"]
            return features

        # Create new memory map
        try:
            # Open file and create memory map
            file_handle = open(filepath, "rb")
            data = np.load(file_handle, mmap_mode="r")
            features = data["features"]

            # Store handles for cleanup
            self._file_handles[identifier] = file_handle

            logger.debug(f"Created memory map for {identifier}")
            return features

        except Exception as e:
            logger.error(f"Failed to create memory map for {identifier}: {e}")
            raise

    def _load_into_memory(self, identifier: str, filepath: Path) -> np.ndarray:
        """Load features directly into memory.

        Args:
            identifier: Feature identifier
            filepath: Path to the feature file

        Returns:
            Feature array loaded into memory
        """
        data = np.load(filepath)
        features: np.ndarray = data["features"]

        # Add to memory cache if space allows
        if features.nbytes <= self.max_memory_cache_bytes:
            self._add_to_memory_cache(identifier, features)

        logger.debug(f"Loaded {identifier} into memory ({features.nbytes / 1024 / 1024:.2f} MB)")
        return features

    def _add_to_memory_cache(self, identifier: str, features: np.ndarray) -> None:
        """Add features to memory cache with LRU eviction.

        Args:
            identifier: Feature identifier
            features: Features to cache
        """
        feature_size = features.nbytes

        # Check if we need to evict old entries
        while (self._memory_usage + feature_size) > self.max_memory_cache_bytes and self._access_order:
            # Evict least recently used
            oldest_id = self._access_order.pop(0)
            if oldest_id in self._memory_cache:
                evicted_size = self._memory_cache[oldest_id].nbytes
                del self._memory_cache[oldest_id]
                self._memory_usage -= evicted_size
                logger.debug(f"Evicted {oldest_id} from memory cache")

        # Add new entry
        if feature_size <= self.max_memory_cache_bytes:
            self._memory_cache[identifier] = features
            self._memory_usage += feature_size
            self._update_access_order(identifier)
            logger.debug(f"Added {identifier} to memory cache")

    def _update_access_order(self, identifier: str) -> None:
        """Update access order for LRU tracking.

        Args:
            identifier: Feature identifier that was accessed
        """
        if identifier in self._access_order:
            self._access_order.remove(identifier)
        self._access_order.append(identifier)

    def batch_store_features(
        self,
        identifiers: List[str],
        features_list: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Store multiple feature arrays efficiently in batch.

        Args:
            identifiers: List of identifiers for each feature array
            features_list: List of feature arrays to store
            metadata_list: Optional list of metadata dictionaries

        Returns:
            List of file paths where features were stored
        """
        if len(identifiers) != len(features_list):
            raise ValueError("Number of identifiers must match number of feature arrays")

        if metadata_list and len(metadata_list) != len(features_list):
            raise ValueError("Number of metadata entries must match number of feature arrays")

        paths = []
        for i, (identifier, features) in enumerate(zip(identifiers, features_list)):
            metadata = metadata_list[i] if metadata_list else None
            path = self.store_features(identifier, features, metadata)
            paths.append(path)

        logger.info(f"Batch stored {len(identifiers)} feature arrays")
        return paths

    def batch_load_features(
        self, identifiers: List[str], use_memory_mapping: Optional[bool] = None
    ) -> Dict[str, Optional[np.ndarray]]:
        """Load multiple feature arrays efficiently in batch.

        Args:
            identifiers: List of identifiers to load
            use_memory_mapping: Override memory mapping setting for this batch

        Returns:
            Dictionary mapping identifiers to loaded features (None if not found)
        """
        results = {}
        for identifier in identifiers:
            features = self.load_features(identifier, use_memory_mapping)
            results[identifier] = features

        loaded_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch loaded {loaded_count}/{len(identifiers)} feature arrays")
        return results

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about storage usage and performance.

        Returns:
            Dictionary with storage statistics
        """
        # Calculate disk usage
        total_disk_usage = 0
        file_count = 0

        for file_path in self.storage_dir.glob("*.npz"):
            total_disk_usage += file_path.stat().st_size
            file_count += 1

        # Memory cache stats
        memory_cache_count = len(self._memory_cache)
        memory_cache_usage = self._memory_usage

        # Memory map stats
        memory_map_count = len(self._mmap_handles)

        return {
            "disk_usage_bytes": total_disk_usage,
            "disk_usage_mb": total_disk_usage / (1024 * 1024),
            "stored_files": file_count,
            "memory_cache_items": memory_cache_count,
            "memory_cache_usage_bytes": memory_cache_usage,
            "memory_cache_usage_mb": memory_cache_usage / (1024 * 1024),
            "memory_mapped_files": memory_map_count,
            "cache_hit_potential": memory_cache_count / max(file_count, 1),
        }

    def cleanup(self) -> None:
        """Clean up memory maps and file handles."""
        # Close memory mapped files
        for identifier in list(self._mmap_handles.keys()):
            try:
                if identifier in self._file_handles:
                    self._file_handles[identifier].close()
                    del self._file_handles[identifier]
                if identifier in self._mmap_handles:
                    del self._mmap_handles[identifier]
            except Exception as e:
                logger.warning(f"Error cleaning up memory map for {identifier}: {e}")

        # Clear memory cache
        self._memory_cache.clear()
        self._access_order.clear()
        self._memory_usage = 0

        logger.info("Cleaned up memory storage")

    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage by reorganizing files and compressing if needed.

        Returns:
            Statistics about the optimization process
        """
        logger.info("Starting storage optimization")

        original_size = 0
        optimized_size = 0
        processed_files = 0

        for file_path in self.storage_dir.glob("*.npz"):
            try:
                original_size += file_path.stat().st_size

                # Load and recompress if compression level changed
                data = np.load(file_path)
                features = data["features"]
                metadata = data.get("metadata", None)

                # Create temporary file with new compression
                temp_path = file_path.with_suffix(".tmp.npz")
                save_data = {"features": features}
                if metadata is not None:
                    save_data["metadata"] = metadata

                if self.compression_level > 0:
                    np.savez_compressed(temp_path, **save_data)
                else:
                    np.savez(temp_path, **save_data)

                # Replace original with optimized version
                temp_path.replace(file_path)
                optimized_size += file_path.stat().st_size
                processed_files += 1

            except Exception as e:
                logger.warning(f"Failed to optimize {file_path}: {e}")

        savings_bytes = original_size - optimized_size
        savings_percent = (savings_bytes / original_size * 100) if original_size > 0 else 0

        optimization_stats = {
            "processed_files": processed_files,
            "original_size_mb": original_size / (1024 * 1024),
            "optimized_size_mb": optimized_size / (1024 * 1024),
            "savings_mb": savings_bytes / (1024 * 1024),
            "savings_percent": savings_percent,
        }

        logger.info(f"Storage optimization complete: {savings_percent:.1f}% size reduction")
        return optimization_stats

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup
