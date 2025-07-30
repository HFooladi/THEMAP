import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from .logging import get_logger

if TYPE_CHECKING:
    from .memory_utils import MemoryEfficientFeatureStorage


logger = get_logger(__name__)


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key for molecular features.

    Args:
        smiles (str): SMILES string of the molecule.
        featurizer_name (str): Name of the featurizer used.
    """

    smiles: str
    featurizer_name: str

    def __post_init__(self):
        if not self.smiles.strip():
            raise ValueError("SMILES cannot be empty")
        if not self.featurizer_name.strip():
            raise ValueError("Featurizer name cannot be empty")


class GlobalFeatureCache:
    """Thread-safe global cache for molecular features."""

    def __init__(self):
        self._features: Dict[CacheKey, np.ndarray] = {}
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "stores": 0}

    def get(self, cache_key: CacheKey) -> Optional[np.ndarray]:
        """Get features from cache."""
        with self._lock:
            if cache_key in self._features:
                self._stats["hits"] += 1
                return self._features[cache_key].copy()
            else:
                self._stats["misses"] += 1
                return None

    def store(self, cache_key: CacheKey, features: np.ndarray) -> None:
        """Store features in cache."""
        if features is None:
            raise ValueError("Cannot store None features")

        with self._lock:
            self._features[cache_key] = features.copy()
            self._stats["stores"] += 1

    def evict(self, cache_key: CacheKey) -> bool:
        """Remove features from cache."""
        with self._lock:
            if cache_key in self._features:
                del self._features[cache_key]
                return True
            return False

    def batch_get(self, cache_keys: List[CacheKey]) -> List[Optional[np.ndarray]]:
        """Get multiple features in a single call for better performance."""
        with self._lock:
            results = []
            for cache_key in cache_keys:
                if cache_key in self._features:
                    self._stats["hits"] += 1
                    results.append(self._features[cache_key].copy())
                else:
                    self._stats["misses"] += 1
                    results.append(None)
            return results

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            return {**self._stats.copy(), "hit_rate": hit_rate, "cache_size": len(self._features)}


# Global cache instance
_global_cache = GlobalFeatureCache()


def get_global_feature_cache() -> GlobalFeatureCache:
    """Get the global feature cache instance."""
    return _global_cache


class PersistentFeatureCache:
    """Persistent cache for molecular features with disk storage.

    This class provides efficient caching of computed molecular features with:
    - Content-based cache keys for reliable cache hits
    - Disk persistence using numpy's efficient binary format
    - Memory management with LRU-style eviction
    - Cache statistics and monitoring
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_memory_cache_size: int = 100,
        enable_compression: bool = True,
        use_memory_efficient_storage: bool = True,
    ):
        """Initialize the persistent feature cache.

        Args:
            cache_dir: Directory for storing cached features
            max_memory_cache_size: Maximum number of items in memory cache
            enable_compression: Whether to compress cached data
            use_memory_efficient_storage: Whether to use memory-efficient storage backend
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_cache_size = max_memory_cache_size
        self.enable_compression = enable_compression
        self.use_memory_efficient_storage = use_memory_efficient_storage

        # Initialize memory-efficient storage if requested
        if use_memory_efficient_storage:
            from .memory_utils import MemoryEfficientFeatureStorage

            compression_level = 6 if enable_compression else 0
            self._memory_storage: Optional["MemoryEfficientFeatureStorage"] = MemoryEfficientFeatureStorage(
                storage_dir=cache_dir,
                compression_level=compression_level,
                max_memory_cache_mb=1024,  # 1GB default
            )
        else:
            self._memory_storage = None

        # In-memory cache for frequently accessed features (only used when not using memory storage)
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._access_times: Dict[str, float] = {}

        # Cache statistics
        self._stats = {"hits": 0, "misses": 0, "memory_hits": 0, "disk_hits": 0, "stores": 0}

        logger.info(
            f"Initialized PersistentFeatureCache at {self.cache_dir} (memory_efficient={use_memory_efficient_storage})"
        )

    def generate_cache_key(
        self, smiles_list: List[str], featurizer_name: str, additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique cache key based on content.

        Args:
            smiles_list: List of SMILES strings
            featurizer_name: Name of the featurizer used
            additional_params: Additional parameters that affect feature computation

        Returns:
            Unique cache key string
        """
        # Sort SMILES to ensure consistent ordering
        sorted_smiles = sorted(smiles_list)

        # Create content string
        content_parts = [
            f"featurizer:{featurizer_name}",
            f"smiles_count:{len(sorted_smiles)}",
            f"smiles_hash:{hashlib.md5('|'.join(sorted_smiles).encode()).hexdigest()}",
        ]

        # Add additional parameters if provided
        if additional_params:
            sorted_params = sorted(additional_params.items())
            params_str = "|".join(f"{k}:{v}" for k, v in sorted_params)
            content_parts.append(f"params:{hashlib.md5(params_str.encode()).hexdigest()}")

        # Generate final hash
        content = "|".join(content_parts)
        cache_key = hashlib.sha256(content.encode()).hexdigest()[:16]

        logger.debug(f"Generated cache key {cache_key} for {len(smiles_list)} SMILES with {featurizer_name}")
        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key.

        Args:
            cache_key: Cache key string

        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{cache_key}.npz"

    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve features from cache.

        Args:
            cache_key: Cache key to retrieve

        Returns:
            Cached features if found, None otherwise
        """
        if self._memory_storage:
            # Use memory-efficient storage
            features = self._memory_storage.load_features(cache_key)
            if features is not None:
                self._stats["hits"] += 1
                self._stats["memory_hits"] += 1
                logger.debug(f"Memory-efficient storage hit for key {cache_key}")
                return features
        else:
            # Use legacy caching approach
            # Check memory cache first
            if cache_key in self._memory_cache:
                self._access_times[cache_key] = time.time()
                self._stats["hits"] += 1
                self._stats["memory_hits"] += 1
                logger.debug(f"Memory cache hit for key {cache_key}")
                return self._memory_cache[cache_key]

            # Check disk cache
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    if self.enable_compression:
                        data = np.load(cache_path)
                        features = data["features"]
                    else:
                        loaded_data = np.load(cache_path)
                        features = (
                            loaded_data if isinstance(loaded_data, np.ndarray) else loaded_data["arr_0"]
                        )

                    # Add to memory cache
                    self._add_to_memory_cache(cache_key, features)

                    self._stats["hits"] += 1
                    self._stats["disk_hits"] += 1
                    logger.debug(f"Disk cache hit for key {cache_key}")
                    return features

                except Exception as e:
                    logger.warning(f"Failed to load cached features from {cache_path}: {e}")
                    # Remove corrupted cache file
                    cache_path.unlink(missing_ok=True)

        # Cache miss
        self._stats["misses"] += 1
        logger.debug(f"Cache miss for key {cache_key}")
        return None

    def store(self, cache_key: str, features: np.ndarray) -> None:
        """Store features in cache.

        Args:
            cache_key: Cache key to store under
            features: Features to cache
        """
        try:
            if self._memory_storage:
                # Use memory-efficient storage
                self._memory_storage.store_features(cache_key, features)
            else:
                # Use legacy storage approach
                # Store to disk
                cache_path = self.get_cache_path(cache_key)

                if self.enable_compression:
                    np.savez_compressed(cache_path, features=features)
                else:
                    np.save(cache_path, features)

                # Add to memory cache
                self._add_to_memory_cache(cache_key, features)

            self._stats["stores"] += 1
            logger.debug(f"Stored features to cache with key {cache_key}")

        except Exception as e:
            logger.error(f"Failed to store features to cache: {e}")

    def _add_to_memory_cache(self, cache_key: str, features: np.ndarray) -> None:
        """Add features to memory cache with LRU eviction.

        Args:
            cache_key: Cache key
            features: Features to cache
        """
        # Evict old entries if necessary
        while len(self._memory_cache) >= self.max_memory_cache_size:
            # Find least recently used entry
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._memory_cache[oldest_key]
            del self._access_times[oldest_key]

        # Add new entry
        self._memory_cache[cache_key] = features
        self._access_times[cache_key] = time.time()

    def clear_memory_cache(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()
        self._access_times.clear()
        logger.info("Cleared memory cache")

    def clear_disk_cache(self) -> None:
        """Clear the disk cache."""
        for cache_file in self.cache_dir.glob("*.npz"):
            cache_file.unlink()
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        logger.info("Cleared disk cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.npz")) + list(self.cache_dir.glob("*.npy"))),
        }

    def get_cache_size_info(self) -> Dict[str, Union[int, float]]:
        """Get information about cache sizes.

        Returns:
            Dictionary with cache size information
        """
        if self._memory_storage:
            # Get stats from memory-efficient storage
            return self._memory_storage.get_storage_stats()
        else:
            # Calculate disk usage for legacy storage
            disk_usage = sum(f.stat().st_size for f in self.cache_dir.iterdir() if f.is_file())

            # Estimate memory usage
            memory_usage = sum(arr.nbytes for arr in self._memory_cache.values())

            return {
                "disk_usage_bytes": disk_usage,
                "disk_usage_mb": disk_usage / (1024 * 1024),
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "memory_cache_items": len(self._memory_cache),
                "disk_cache_files": len(
                    list(self.cache_dir.glob("*.npz")) + list(self.cache_dir.glob("*.npy"))
                ),
            }


class GlobalMoleculeCache:
    """Global cache for managing molecular features across multiple datasets.

    This class coordinates feature computation and caching across multiple datasets,
    implementing global SMILES deduplication and efficient batch processing.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize global molecule cache.

        Args:
            cache_dir: Directory for persistent cache storage
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.persistent_cache = PersistentFeatureCache(cache_dir) if cache_dir else None

        # Global SMILES tracking
        self._global_smiles_registry: Dict[str, str] = {}  # SMILES -> canonical hash
        self._features_registry: Dict[str, Dict[str, np.ndarray]] = {}  # hash -> {featurizer: features}

        logger.info(f"Initialized GlobalMoleculeCache with cache_dir={cache_dir}")

    def get_unique_smiles_across_datasets(self, datasets: List[Any]) -> Dict[str, List[tuple]]:
        """Get all unique SMILES across datasets with their positions.

        Args:
            datasets: List of datasets to analyze

        Returns:
            Dictionary mapping unique SMILES to list of (dataset_idx, molecule_idx) tuples
        """
        smiles_map: Dict[str, List[tuple]] = {}
        total_molecules = 0

        for dataset_idx, dataset in enumerate(datasets):
            for mol_idx, mol in enumerate(dataset.data):
                canonical_smiles = self._canonicalize_smiles(mol.smiles)
                if canonical_smiles not in smiles_map:
                    smiles_map[canonical_smiles] = []
                smiles_map[canonical_smiles].append((dataset_idx, mol_idx))
                total_molecules += 1

        unique_count = len(smiles_map)
        duplicate_count = total_molecules - unique_count

        logger.info(
            f"Found {unique_count} unique SMILES across {len(datasets)} datasets "
            f"({duplicate_count} duplicates, {duplicate_count / total_molecules * 100:.1f}% deduplication)"
        )

        return smiles_map

    def _canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES string for consistent hashing.

        Args:
            smiles: Input SMILES string

        Returns:
            Canonical SMILES string
        """
        # For now, just return the input SMILES
        # In a production system, you might want to use RDKit to canonicalize
        return smiles.strip()

    def batch_compute_features(
        self, unique_smiles: List[str], featurizer_name: str, batch_size: int = 1000, n_jobs: int = -1
    ) -> Dict[str, np.ndarray]:
        """Efficiently compute features for unique SMILES in batches.

        Args:
            unique_smiles: List of unique SMILES to compute features for
            featurizer_name: Name of featurizer to use
            batch_size: Batch size for processing
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary mapping SMILES to computed features
        """
        from .featurizer_utils import get_featurizer

        # Check cache first
        cache_key = None
        if self.persistent_cache:
            cache_key = self.persistent_cache.generate_cache_key(unique_smiles, featurizer_name)
            cached_features = self.persistent_cache.get(cache_key)
            if cached_features is not None:
                logger.info(f"Using cached features for {len(unique_smiles)} unique SMILES")
                return dict(zip(unique_smiles, cached_features))

        logger.info(f"Computing features for {len(unique_smiles)} unique SMILES using {featurizer_name}")
        start_time = time.time()

        # Get featurizer
        featurizer = get_featurizer(featurizer_name, n_jobs=n_jobs)

        # Process in batches for memory efficiency
        all_features = []

        for i in range(0, len(unique_smiles), batch_size):
            batch_smiles = unique_smiles[i : i + batch_size]
            logger.debug(
                f"Processing batch {i // batch_size + 1}/{(len(unique_smiles) + batch_size - 1) // batch_size}"
            )

            # Preprocess and transform batch
            try:
                processed_smiles, _ = featurizer.preprocess(batch_smiles)

                if hasattr(featurizer, "transform"):
                    batch_features = featurizer.transform(processed_smiles)
                else:
                    batch_features = featurizer(processed_smiles)

                all_features.append(batch_features)

            except Exception as e:
                logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                # Create zero features for failed batch
                feature_dim = 2048  # Default dimension, should be parameterized
                batch_features = np.zeros((len(batch_smiles), feature_dim))
                all_features.append(batch_features)

        # Combine all batches
        if all_features:
            combined_features = np.vstack(all_features)
        else:
            # Fallback if no features computed
            feature_dim = 2048
            combined_features = np.zeros((len(unique_smiles), feature_dim))

        # Store in cache
        if self.persistent_cache and cache_key:
            self.persistent_cache.store(cache_key, combined_features)

        elapsed_time = time.time() - start_time
        logger.info(f"Computed features for {len(unique_smiles)} SMILES in {elapsed_time:.2f} seconds")

        # Create mapping
        return dict(zip(unique_smiles, combined_features))
