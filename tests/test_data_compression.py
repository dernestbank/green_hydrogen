"""
Tests for Data Compression functionality

Test intelligent compression strategies for different data types,
compression algorithms, and performance benchmarking.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.utils.data_compression import (
    CompressionManager,
    GZipCompression,
    BZ2Compression,
    LZMACompression,
    DataFrameCompression,
    CompressionResult,
    benchmark_compression,
    compress_dataframe
)


class TestCompressionStrategies:
    """Test individual compression strategies."""

    def test_gzip_compression(self):
        """Test GZip compression strategy."""
        strategy = GZipCompression()

        test_data = {"message": "Hello World", "numbers": [1, 2, 3, 4, 5] * 100}

        compressed = strategy.compress(test_data)
        decompressed = strategy.decompress(compressed)

        assert test_data == decompressed
        assert len(compressed) < len(str(test_data).encode())

    def test_bz2_compression(self):
        """Test BZ2 compression strategy."""
        strategy = BZ2Compression()

        test_string = "This is a test string that should compress well when repeated. " * 50

        compressed = strategy.compress(test_string)
        decompressed = strategy.decompress(compressed)

        assert test_string == decompressed

    def test_lzma_compression(self):
        """Test LZMA compression strategy."""
        strategy = LZMACompression()

        test_data = np.random.randn(1000).tobytes()

        compressed = strategy.compress(test_data)
        decompressed = strategy.decompress(compressed)

        assert test_data == decompressed

    def test_dataframe_compression(self):
        """Test DataFrame-optimized compression."""
        strategy = DataFrameCompression()

        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='H'),
            'value1': np.random.randn(100),
            'value2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })

        compressed = strategy.compress(df)
        decompressed = strategy.decompress(compressed)

        assert isinstance(decompressed, pd.DataFrame)
        pd.testing.assert_frame_equal(df, decompressed)


class TestCompressionManager:
    """Test compression manager with automatic strategy selection."""

    @pytest.fixture
    def manager(self):
        """Create compression manager."""
        return CompressionManager(
            enable_compression=True,
            compression_threshold_kb=1,  # Low threshold for testing
            benchmark_compression=False
        )

    def test_compression_manager_init(self, manager):
        """Test compression manager initialization."""
        assert manager.enable_compression is True
        assert manager.compression_threshold == 1024  # 1KB
        assert len(manager.strategies) == 4  # gzip, bz2, lzma, dataframe

    def test_should_compress(self, manager):
        """Test compression threshold logic."""
        small_data = {"key": "value"}
        large_data = {"data": [1] * 1000}

        assert not manager.should_compress(small_data)
        assert manager.should_compress(large_data, 2048)  # Explicit large size

    def test_data_type_detection(self, manager):
        """Test automatic data type detection."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        series = pd.Series([1, 2, 3])
        array = np.array([1, 2, 3])
        dictionary = {"key": "value"}
        list_data = [1, 2, 3]
        tuple_data = (1, 2, 3)
        text = "hello world"

        assert manager._detect_data_type(df) == 'dataframe'
        assert manager._detect_data_type(series) == 'series'
        assert manager._detect_data_type(array) == 'numpy.ndarray'
        assert manager._detect_data_type(dictionary) == 'dict'
        assert manager._detect_data_type(list_data) == 'list'
        assert manager._detect_data_type(tuple_data) == 'tuple'
        assert manager._detect_data_type(text) == 'default'

    def test_compress_decompress_cycle(self, manager):
        """Test full compression and decompression cycle."""
        original_data = {"data": [1, 2, 3] * 200}  # Make it large enough to compress

        compressed, result = manager.compress_data(original_data)
        decompressed = manager.decompress_data(compressed)

        assert isinstance(result, CompressionResult)
        assert result.original_size > 0
        assert result.compressed_size > 0
        assert result.compression_ratio >= 1.0
        assert result.algorithm in ['gzip', 'bz2', 'lzma', 'dataframe']
        assert original_data == decompressed

    def test_dataframe_compression_cycle(self, manager):
        """Test DataFrame specific compression."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=1000, freq='H'),
            'value': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })

        compressed, result = manager.compress_data(df)
        decompressed = manager.decompress_data(compressed)

        assert isinstance(decompressed, pd.DataFrame)
        assert len(decompressed) == len(df)
        assert list(decompressed.columns) == list(df.columns)
        assert result.algorithm == 'dataframe'

    def test_small_data_no_compression(self, manager):
        """Test that small data bypasses compression."""
        small_data = {"key": "value"}

        compressed, result = manager.compress_data(small_data)

        assert result is None
        assert compressed == small_data

    def test_force_compression(self, manager):
        """Test forced compression regardless of size."""
        small_data = {"key": "value"}

        compressed, result = manager.compress_data(small_data, force_compress=True)

        assert result is not None
        assert isinstance(compressed, dict)
        assert '__compressed__' in compressed

    def test_compression_disabled(self):
        """Test behavior when compression is disabled."""
        manager = CompressionManager(enable_compression=False, compression_threshold_kb=0)

        large_data = {"data": [1] * 1000}

        compressed, result = manager.compress_data(large_data)

        assert result is None
        assert compressed == large_data

    def test_compression_stats(self, manager):
        """Test compression statistics collection."""
        manager.compress_data({"data": [1] * 200}, force_compress=True)
        manager.compress_data({"data": [2] * 200}, force_compress=True)

        stats = manager.get_compression_stats()

        assert stats['enabled'] is True
        assert stats['total_operations'] == 2
        assert 'average_ratio' in stats
        assert len(stats['operation_breakdown']) == 2

    def test_reset_stats(self, manager):
        """Test statistics reset functionality."""
        manager.compress_data({"data": [1] * 200}, force_compress=True)
        assert manager.stats.total_operations == 1

        manager.reset_stats()
        assert manager.stats.total_operations == 0

    def test_optimize_for_data_type(self, manager):
        """Test data type optimization."""
        manager.optimize_for_data_type('dict', 'bz2')

        assert manager.type_preferences['dict'] == 'bz2'


class TestBenchmarkCompression:
    """Test compression benchmarking functionality."""

    def test_benchmark_compression(self):
        """Test compression benchmarking across algorithms."""
        data = {"payload": [1, 2, 3] * 500}

        results = benchmark_compression(data, algorithms=['gzip', 'bz2'])

        assert 'gzip' in results
        assert 'bz2' in results

        gzip_result = results['gzip']
        bz2_result = results['bz2']

        assert isinstance(gzip_result, CompressionResult)
        assert isinstance(bz2_result, CompressionResult)
        assert gzip_result.original_size > 0
        assert gzip_result.compressed_size > 0
        assert gzip_result.algorithm == 'gzip'
        assert bz2_result.algorithm == 'bz2'

    def test_dataframe_compression_function(self):
        """Test standalone DataFrame compression function."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5] * 100,
            'b': ['hello'] * 500
        })

        compressed, result = compress_dataframe(df)

        assert isinstance(compressed, bytes)
        assert isinstance(result, CompressionResult)
        assert result.data_type == 'dataframe'
        assert result.original_size > 0

        # Test decompression manually
        strategy = DataFrameCompression()
        decompressed = strategy.decompress(compressed)

        assert isinstance(decompressed, pd.DataFrame)
        assert len(decompressed) == len(df)


class TestCompressionResult:
    """Test compression result data structure."""

    def test_compression_result_creation(self):
        """Test compression result creation and properties."""
        result = CompressionResult(
            original_size=1000,
            compressed_size=500,
            compression_ratio=2.0,
            algorithm='gzip',
            compression_time=0.1,
            data_type='dict'
        )

        assert result.original_size == 1000
        assert result.compressed_size == 500
        assert result.compression_ratio == 2.0
        assert result.algorithm == 'gzip'
        assert result.compression_time == 0.1
        assert result.data_type == 'dict'

    def test_compression_result_auto_ratio(self):
        """Test automatic compression ratio calculation."""
        result = CompressionResult(
            original_size=2048,
            compressed_size=1024,
            algorithm='gzip',
            compression_time=0.2
        )

        assert result.compression_ratio == 2.0


class TestIntegrationScenarios:
    """Test compression in realistic scenarios."""

    def test_large_dataframe_compression(self):
        """Test compression of large DataFrame typical for API responses."""
        np.random.seed(42)

        # Simulate a year of hourly renewable energy data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=8760, freq='H'),
            'solar_radiation': np.maximum(0, np.sin(np.arange(8760) * 2 * np.pi / 24) + np.random.normal(0, 0.3, 8760)),
            'wind_speed': np.maximum(0, 5 + 3 * np.random.normal(0, 0.8, 8760)),
            'capacity_factor': np.random.beta(2, 5, 8760),
            'temperature': 15 + 10 * np.sin(np.arange(8760) * 2 * np.pi / 8760) + np.random.normal(0, 2, 8760)
        })

        manager = CompressionManager()
        compressed, result = manager.compress_data(df)

        assert result is not None
        assert result.data_type == 'dataframe'
        assert result.compression_ratio > 1.0

        # Verify data integrity
        decompressed = manager.decompress_data(compressed)
        pd.testing.assert_frame_equal(df, decompressed)

        # Log compression stats
        logger = manager.compression_manager.logger if hasattr(manager, 'compression_manager') else manager.logger
        logger.info(f"YEAR")

    def test_api_response_compression(self):
        """Test compression of typical API response structure."""
        api_response = {
            "data": {
                "solar_cf": [0.23, 0.45, 0.67, 0.12, 0.89] * 200,  # Simulate many hourly values
                "wind_cf": [0.45, 0.67, 0.12, 0.89, 0.23] * 200,
                "metadata": {
                    "location": "Test Site",
                    "capacity_mw": 100,
                    "turbine_type": "Vestas V150-4.2MW"
                },
                "timestamps": [datetime.now() + timedelta(hours=i) for i in range(1000)]
            },
            "status": "success",
            "request_id": "abc123"
        }

        manager = CompressionManager()
        compressed, result = manager.compress_data(api_response)

        assert result is not None
        assert result.original_size > len(str(api_response))
        assert result.compression_ratio > 1.0

        # Verify round-trip integrity
        decompressed = manager.decompress_data(compressed)
        assert api_response == decompressed


if __name__ == "__main__":
    pytest.main([__file__])