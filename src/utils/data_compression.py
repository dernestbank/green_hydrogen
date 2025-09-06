"""
Advanced Data Compression for Large Datasets

Provides intelligent compression strategies for API responses and cached data,
optimized for different data types (DataFrames, dictionaries, lists, etc.).
Supports multiple compression algorithms with automatic selection based on
data characteristics and performance trade-offs.
"""

import gzip
import bz2
import lzma
import pickle
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from time import time
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    compression_time: float
    decompression_time: Optional[float] = None
    data_type: str = "unknown"

    def __post_init__(self):
        """Calculate compression ratio if not provided."""
        if self.original_size > 0:
            self.compression_ratio = self.original_size / max(self.compressed_size, 1)
        else:
            self.compression_ratio = 1.0


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    total_operations: int = 0
    total_original_size: int = 0
    total_compressed_size: int = 0
    average_ratio: float = 0.0
    best_algorithm: Optional[str] = None
    compression_times: List[float] = field(default_factory=list)
    algorithm_usage: Dict[str, int] = field(default_factory=dict)

    def add_result(self, result: CompressionResult):
        """Add a compression result to statistics."""
        self.total_operations += 1
        self.total_original_size += result.original_size
        self.total_compressed_size += result.compressed_size
        self.compression_times.append(result.compression_time)
        self.algorithm_usage[result.algorithm] = self.algorithm_usage.get(result.algorithm, 0) + 1

        # Update average ratio
        if self.total_compressed_size > 0:
            self.average_ratio = self.total_original_size / max(self.total_compressed_size, 1)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_operations': self.total_operations,
            'total_original_mb': self.total_original_size / (1024**2),
            'total_compressed_mb': self.total_compressed_size / (1024**2),
            'average_ratio': self.average_ratio,
            'space_savings_percent': round((1 - self.total_compressed_size / max(self.total_original_size, 1)) * 100, 2),
            'best_algorithm': self.best_algorithm,
            'avg_compression_time_ms': round(np.mean(self.compression_times) * 1000, 2) if self.compression_times else 0,
            'algorithm_usage': self.algorithm_usage
        }


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compress(self, data: Any) -> bytes:
        """Compress data and return bytes."""
        pass

    @abstractmethod
    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress data and return original."""
        pass

    def get_compression_info(self) -> Dict[str, Any]:
        """Get information about this compression strategy."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'description': f"Compression using {self.name}"
        }


class GZipCompression(CompressionStrategy):
    """GZip compression strategy - fast and good general purpose."""

    def __init__(self):
        super().__init__("gzip")

    def compress(self, data: Any) -> bytes:
        """Compress data using gzip."""
        pickled_data = pickle.dumps(data)
        return gzip.compress(pickled_data, compresslevel=6)

    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress gzip data."""
        decompressed = gzip.decompress(compressed_data)
        return pickle.loads(decompressed)


class BZ2Compression(CompressionStrategy):
    """BZ2 compression strategy - better compression, slower."""

    def __init__(self):
        super().__init__("bz2")

    def compress(self, data: Any) -> bytes:
        """Compress data using bz2."""
        pickled_data = pickle.dumps(data)
        return bz2.compress(pickled_data, compresslevel=9)

    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress bz2 data."""
        decompressed = bz2.decompress(compressed_data)
        return pickle.loads(decompressed)


class LZMACompression(CompressionStrategy):
    """LZMA compression strategy - best compression, slowest."""

    def __init__(self):
        super().__init__("lzma")

    def compress(self, data: Any) -> bytes:
        """Compress data using lzma."""
        pickled_data = pickle.dumps(data)
        return lzma.compress(pickled_data, preset=6)

    def decompress(self, compressed_data: bytes) -> Any:
        """Decompress lzma data."""
        decompressed = lzma.decompress(compressed_data)
        return pickle.loads(decompressed)


class DataFrameCompression(CompressionStrategy):
    """Optimized compression for pandas DataFrames."""

    def __init__(self):
        super().__init__("dataframe")
        self.fallback_strategy = GZipCompression()

    def compress(self, data: pd.DataFrame) -> bytes:
        """Compress DataFrame with type-aware compression."""
        try:
            # Convert DataFrame to HDF5 format in memory for better compression
            import io
            buffer = io.BytesIO()

            # Use feather format for DataFrames (good compression + fast)
            data.to_feather(buffer)
            compressed_data = buffer.getvalue()
            return self.fallback_strategy.compress({
                'format': 'feather',
                'data': compressed_data,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            })
        except Exception as e:
            logger.warning(f"DataFrame compression failed, using fallback: {e}")
            return self.fallback_strategy.compress(data)

    def decompress(self, compressed_data: bytes) -> pd.DataFrame:
        """Decompress DataFrame."""
        try:
            decompressed_dict = self.fallback_strategy.decompress(compressed_data)

            if isinstance(decompressed_dict, dict) and decompressed_dict.get('format') == 'feather':
                # Reconstruct DataFrame from feather
                import io
                buffer = io.BytesIO(decompressed_dict['data'])
                df = pd.read_feather(buffer)

                # Restore dtypes if needed
                if 'dtypes' in decompressed_dict:
                    dtype_mapping = {}
                    for col, dtype_str in decompressed_dict['dtypes'].items():
                        if col in df.columns:
                            # Try to convert back to original dtype
                            try:
                                if dtype_str.startswith('int'):
                                    dtype_mapping[col] = 'Int64'  # Nullable integer
                                elif dtype_str.startswith('float'):
                                    dtype_mapping[col] = 'Float64'  # Nullable float
                            except:
                                pass
                    if dtype_mapping:
                        df = df.astype(dtype_mapping)

                return df
            else:
                return pd.DataFrame(decompressed_dict)
        except Exception as e:
            logger.warning(f"DataFrame decompression failed, using fallback: {e}")
            return self.fallback_strategy.decompress(compressed_data)


class CompressionManager:
    """
    Intelligent compression manager that selects optimal compression
    based on data characteristics and performance requirements.
    """

    def __init__(self, enable_compression: bool = True,
                 compression_threshold_kb: int = 100,
                 benchmark_compression: bool = True):
        """
        Initialize compression manager.

        Args:
            enable_compression: Whether to enable compression
            compression_threshold_kb: Only compress data larger than this threshold
            benchmark_compression: Whether to benchmark compression algorithms
        """
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold_kb * 1024
        self.benchmark_compression = benchmark_compression

        # Available compression strategies
        self.strategies = {
            'gzip': GZipCompression(),
            'bz2': BZ2Compression(),
            'lzma': LZMACompression(),
            'dataframe': DataFrameCompression()
        }

        # Default strategy preferences by data type
        self.type_preferences = {
            'dataframe': 'dataframe',  # Use optimized DataFrame compression
            'dict': 'gzip',           # Good for JSON-like data
            'list': 'gzip',           # Good for arrays
            'numpy.ndarray': 'lzma',   # Best for numerical data
            'default': 'gzip'         # Fallback
        }

        # Statistics tracking
        self.stats = CompressionStats()
        self.operation_stats = {}

        logger.info(f"Compression manager initialized (threshold: {compression_threshold_kb}KB)")

    def should_compress(self, data: Any, data_size_bytes: Optional[int] = None) -> bool:
        """
        Determine if data should be compressed based on size and settings.

        Args:
            data: Data to potentially compress
            data_size_bytes: Pre-calculated data size

        Returns:
            True if data should be compressed
        """
        if not self.enable_compression:
            return False

        if data_size_bytes is None:
            data_size_bytes = self._estimate_data_size(data)

        return data_size_bytes > self.compression_threshold

    def _estimate_data_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            if hasattr(data, 'memory_usage'):
                # DataFrame/Series
                return data.memory_usage(deep=True).sum()
            elif hasattr(data, 'nbytes'):
                # NumPy array
                return data.nbytes
            elif isinstance(data, (dict, list)):
                # Collections - estimate based on pickled size
                return len(pickle.dumps(data))
            elif isinstance(data, bytes):
                return len(data)
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            else:
                # Fallback to pickled size
                return len(pickle.dumps(data))
        except Exception:
            # Ultimate fallback
            return 1024  # Assume 1KB if estimation fails

    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data for compression strategy selection."""
        if isinstance(data, pd.DataFrame):
            return 'dataframe'
        elif isinstance(data, pd.Series):
            return 'series'
        elif isinstance(data, np.ndarray):
            return 'numpy.ndarray'
        elif isinstance(data, dict):
            return 'dict'
        elif isinstance(data, list):
            return 'list'
        elif isinstance(data, tuple):
            return 'tuple'
        elif isinstance(data, set):
            return 'set'
        else:
            return 'default'

    def _select_best_strategy(self, data: Any, data_type: str) -> CompressionStrategy:
        """
        Select the best compression strategy for the given data.

        Args:
            data: The data to compress
            data_type: Detected data type

        Returns:
            Selected compression strategy
        """
        # Use type-specific preference
        preferred_algorithm = self.type_preferences.get(data_type, 'gzip')

        if self.benchmark_compression and data_type in ['dataframe', 'dict', 'list']:
            # Benchmark different strategies for large datasets
            data_size = self._estimate_data_size(data)
            if data_size > 500 * 1024:  # Only benchmark > 500KB
                return self._benchmark_strategies(data, data_type)

        return self.strategies.get(preferred_algorithm, self.strategies['gzip'])

    def _benchmark_strategies(self, data: Any, data_type: str) -> CompressionStrategy:
        """
        Benchmark different compression strategies and select the best.

        Args:
            data: Data to benchmark
            data_type: Data type for optimization

        Returns:
            Best performing strategy
        """
        results = {}
        data_size = self._estimate_data_size(data)

        # Test candidates based on data type
        candidates = []
        if data_type == 'dataframe':
            candidates = ['dataframe', 'gzip', 'bz2']
        else:
            candidates = ['gzip', 'bz2']

        for algorithm in candidates:
            try:
                strategy = self.strategies[algorithm]
                start_time = time()

                compressed = strategy.compress(data)
                compressed_size = len(compressed)
                compression_time = time() - start_time

                # Calculate compression efficiency
                ratio = data_size / max(compressed_size, 1)
                compression_speed = compressed_size / max(compression_time, 0.001)  # bytes/second

                results[algorithm] = {
                    'ratio': ratio,
                    'speed': compression_speed,
                    'size': compressed_size
                }

                logger.debug(f"Compression {algorithm}: {ratio:.2f}x ratio, {compression_speed/1024:.1f}KB/s")

            except Exception as e:
                logger.warning(f"Benchmark failed for {algorithm}: {e}")
                continue

        # Select best strategy based on compression ratio and speed
        if results:
            best_algorithm = max(results.keys(),
                               key=lambda x: results[x]['ratio'] * min(1.0, results[x]['speed'] / 1000000))
            logger.debug(f"Selected {best_algorithm} for {data_type}")
            return self.strategies[best_algorithm]

        return self.strategies['gzip']

    def compress_data(self, data: Any, force_compress: bool = False) -> Tuple[Any, Optional[CompressionResult]]:
        """
        Compress data if appropriate, otherwise return original.

        Args:
            data: Data to compress
            force_compress: Force compression regardless of threshold

        Returns:
            Tuple of (processed_data, compression_result_or_none)
        """
        data_type = self._detect_data_type(data)
        data_size = self._estimate_data_size(data)

        if not (force_compress or self.should_compress(data, data_size)):
            return data, None

        # Select compression strategy
        strategy = self._select_best_strategy(data, data_type)

        # Compress data
        start_time = time()
        compressed_data = strategy.compress(data)
        compression_time = time() - start_time

        # Create compression result
        compressed_size = len(compressed_data)
        result = CompressionResult(
            original_size=data_size,
            compressed_size=compressed_size,
            compression_ratio=data_size / max(compressed_size, 1),
            algorithm=strategy.name,
            compression_time=compression_time,
            data_type=data_type
        )

        # Update statistics
        self.stats.add_result(result)

        #Store operation result
        op_key = f"{strategy.name}_{data_type}"
        if op_key not in self.operation_stats:
            self.operation_stats[op_key] = []
        self.operation_stats[op_key].append(result)

        # Mark data as compressed for storage
        processed_data = {
            '__compressed__': True,
            'algorithm': strategy.name,
            'original_type': data_type,
            'data': compressed_data
        }

        logger.debug(f"Compressed {data_type} ({data_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB, "
                    f"{result.compression_ratio:.2f}x, {compression_time*1000:.1f}ms)")

        return processed_data, result

    def decompress_data(self, data: Any) -> Any:
        """
        Decompress data if compressed.

        Args:
            data: Data to decompress

        Returns:
            Decompressed data
        """
        if not self._is_compressed(data):
            return data

        # Extract compression info
        algorithm = data.get('algorithm', 'gzip')
        original_type = data.get('original_type', 'unknown')
        compressed_data = data.get('data')

        if compressed_data is None:
            raise ValueError("Compressed data is missing")

        # Get decompression strategy
        strategy = self.strategies.get(algorithm)
        if strategy is None:
            raise ValueError(f"Unknown compression algorithm: {algorithm}")

        # Decompress
        start_time = time()
        decompressed_data = strategy.decompress(compressed_data)
        decompression_time = time() - start_time

        # Update decompression stats
        if hasattr(decompressed_data, '__class__'):
            class_name = decompressed_data.__class__.__name__
            if hasattr(self.stats, 'total_operations'):
                # This is a simple way to track decompression time
                pass

        logger.debug(f"Decompressed {original_type} using {algorithm} ({decompression_time*1000:.1f}ms)")

        return decompressed_data

    def _is_compressed(self, data: Any) -> bool:
        """Check if data is compressed."""
        return (isinstance(data, dict) and
                data.get('__compressed__') is True)

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        base_stats = self.stats.get_summary()
        base_stats.update({
            'enabled': self.enable_compression,
            'threshold_kb': self.compression_threshold / 1024,
            'benchmark_enabled': self.benchmark_compression,
            'available_strategies': list(self.strategies.keys()),
            'operation_breakdown': self.operation_stats
        })
        return base_stats

    def reset_stats(self):
        """Reset compression statistics."""
        self.stats = CompressionStats()
        self.operation_stats = {}

    def optimize_for_data_type(self, data_type: str, algorithm: str):
        """
        Set preferred compression algorithm for a data type.

        Args:
            data_type: Type of data (e.g., 'dataframe', 'dict')
            algorithm: Preferred compression algorithm
        """
        if algorithm in self.strategies:
            self.type_preferences[data_type] = algorithm
            logger.info(f"Optimized {data_type} compression to use {algorithm}")
        else:
            logger.warning(f"Unknown compression algorithm: {algorithm}")


# Convenience functions
def compress_dataframe(df: pd.DataFrame, compression_level: int = 6) -> Tuple[bytes, CompressionResult]:
    """
    Convenience function to compress a pandas DataFrame.

    Args:
        df: DataFrame to compress
        compression_level: Compression level (1-9)

    Returns:
        Tuple of (compressed_data, compression_result)
    """
    compressor = DataFrameCompression()
    start_time = time()
    compressed = compressor.compress(df)
    compression_time = time() - start_time

    result = CompressionResult(
        original_size=df.memory_usage(deep=True).sum(),
        compressed_size=len(compressed),
        compression_ratio=df.memory_usage(deep=True).sum() / max(len(compressed), 1),
        algorithm='dataframe',
        compression_time=compression_time,
        data_type='dataframe'
    )

    return compressed, result


def benchmark_compression(data: Any, algorithms: Optional[List[str]] = None) -> Dict[str, CompressionResult]:
    """
    Benchmark different compression algorithms on data.

    Args:
        data: Data to benchmark compression on
        algorithms: List of algorithms to test (default: all)

    Returns:
        Dictionary mapping algorithm names to results
    """
    if algorithms is None:
        algorithms = ['gzip', 'bz2', 'lzma']

    compressor = CompressionManager(enable_compression=True, benchmark_compression=False)
    results = {}

    for algorithm in algorithms:
        try:
            strategy = compressor.strategies[algorithm]
            start_time = time()
            compressed = strategy.compress(data)
            compression_time = time() - start_time

            data_size = compressor._estimate_data_size(data)
            result = CompressionResult(
                original_size=data_size,
                compressed_size=len(compressed),
                compression_ratio=data_size / max(len(compressed), 1),
                algorithm=algorithm,
                compression_time=compression_time,
                data_type=compressor._detect_data_type(data)
            )
            results[algorithm] = result
        except Exception as e:
            logger.error(f"Benchmark failed for {algorithm}: {e}")

    return results


if __name__ == "__main__":
    # Example usage and benchmarking
    import numpy as np

    # Create sample data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=8760, freq='H'),
        'value1': np.random.randn(8760),
        'value2': np.random.randn(8760),
        'category': np.random.choice(['A', 'B', 'C'], 8760)
    })

    # Benchmark compression
    results = benchmark_compression(df)

    print("Compression Benchmark Results:")
    for algorithm, result in results.items():
        print("3.1f")
        print(".0f")
        print("2.2f")

    # Test compression manager
    manager = CompressionManager()
    compressed, result = manager.compress_data(df)

    if result:
        print(f"\nCompression Manager Result:")
        print(f"Original size: {result.original_size / 1024:.1f} KB")
        print(f"Compressed size: {result.compressed_size / 1024:.1f} KB")
        print(f"Compression ratio: {result.compression_ratio:.2f}x")
        print(f"Algorithm: {result.algorithm}")
        print(f"Compression time: {result.compression_time * 1000:.1f} ms")

        # Test decompression
        decompressed = manager.decompress_data(compressed)
        print(f"Decompression successful: {isinstance(decompressed, pd.DataFrame)}")
    else:
        print("\nNo compression applied (data too small or compression disabled)")

    # Print statistics
    print(f"\nCompression Statistics:")
    stats = manager.get_compression_stats()
    print(f"Total operations: {stats['total_operations']}")
    print(f"Space savings: {stats['space_savings_percent']:.1f}%")