"""
Tests for Performance Benchmarking System

Test the API performance benchmarking against cached data,
measuring improvement ratios and response times.
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

from api_performance_benchmark import (
    BenchmarkResult,
    benchmark_scenario,
    MockRenewablesNinjaAPI,
    ComparativeBenchmark
)


class TestBenchmarkResult:
    """Test benchmark result data structure."""

    def test_benchmark_result_creation(self):
        """Test benchmark result creation."""
        response_times = [0.1, 0.2, 0.15, 0.18, 0.12]

        result = BenchmarkResult(
            scenario="test_scenario",
            sample_size=5,
            total_time=0.75,
            avg_response_time=0.15,
            median_response_time=0.15,
            min_response_time=0.1,
            max_response_time=0.2,
            throughput=6.67,  # 5 requests / 0.75 seconds
            response_times=response_times,
            metadata={"test": True}
        )

        assert result.scenario == "test_scenario"
        assert result.sample_size == 5
        assert result.avg_response_time == 0.15
        assert result.throughput == pytest.approx(6.67, abs=0.1)
        assert result.metadata["test"] is True


class TestMockAPI:
    """Test mock Renewables Ninja API."""

    def test_mock_api_creation(self):
        """Test mock API initialization."""
        mock_api = MockRenewablesNinjaAPI(base_latency_ms=100, jitter_ms=20)

        assert mock_api.base_latency_ms == 100
        assert mock_api.jitter_ms == 20

    def test_mock_solar_data_generation(self):
        """Test solar data generation."""
        mock_api = MockRenewablesNinjaAPI()

        df = mock_api._generate_solar_data(40.7128, -74.0060)

        assert len(df) == 8760  # 365 days * 24 hours
        assert 'timestamp' in df.columns
        assert 'capacity_factor' in df.columns
        assert df['capacity_factor'].min() >= 0
        assert df['capacity_factor'].max() <= 1

    def test_mock_api_latency(self):
        """Test API response latency."""
        mock_api = MockRenewablesNinjaAPI(base_latency_ms=50, jitter_ms=0)

        start_time = time.time()
        result = mock_api.fetch_solar_data(40.7128, -74.0060)
        end_time = time.time()

        # Should have simulated latency
        elapsed = end_time - start_time
        assert elapsed >= 0.045  # At least 45ms due to base latency
        assert len(result) == 8760  # Should still return correct data

    def test_mock_cache_consistency(self):
        """Test that mock data is consistent across calls."""
        mock_api = MockRenewablesNinjaAPI()

        df1 = mock_api.fetch_solar_data(40.7128, -74.0060)
        df2 = mock_api.fetch_solar_data(40.7128, -74.0060)

        # Should return identical data (cached in mock)
        assert df1.equals(df2)


class TestBenchmarkScenario:
    """Test benchmark scenario running."""

    @pytest.fixture
    def mock_api(self):
        """Create mock API with minimal latency."""
        return MockRenewablesNinjaAPI(base_latency_ms=1, jitter_ms=0)

    def test_benchmark_scenario_sequential(self, mock_api):
        """Test benchmark scenario with sequential execution."""
        locations = [(40.7128, -74.0060)]

        start_time = time.time()
        result = benchmark_scenario("test_seq", mock_api, locations, iterations=2)
        end_time = time.time()

        assert result.scenario == "test_seq"
        assert result.sample_size == 4  # 2 iterations * 2 API calls (solar + wind)
        assert result.avg_response_time > 0
        assert result.throughput > 0
        assert len(result.response_times) == 4

    def test_benchmark_scenario_concurrent(self, mock_api):
        """Test benchmark scenario with concurrent execution."""
        locations = [(40.7128, -74.0060), (34.0522, -118.2437)]

        result = benchmark_scenario("test_conc", mock_api, locations, iterations=1, concurrent=True)

        assert result.scenario == "test_conc"
        assert result.sample_size == 4  # 2 locations * 2 API calls
        assert result.metadata.get('concurrent') is True
        assert result.metadata.get('locations_count') == 2

    def test_benchmark_with_cache(self, mock_api):
        """Test benchmark with cached API."""
        from utils.api_cache import APICache, CachedRenewablesNinjaAPI

        locations = [(40.7128, -74.0060)]

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = APICache(cache_dir=cache_dir, default_ttl=3600)
            cached_api = CachedRenewablesNinjaAPI(mock_api, cache_ttl=3600, cache_dir=cache_dir)

            result = benchmark_scenario("cached_test", cached_api, locations, iterations=2)

            assert result.scenario == "cached_test"
            assert result.sample_size > 0
            assert result.avg_response_time > 0

            # Second run should be faster (cached)
            result2 = benchmark_scenario("cached_test2", cached_api, locations, iterations=1)
            # Should be faster than first run due to cache hits
            assert result2.avg_response_time < result.avg_response_time

            cache.close()


class TestComparativeBenchmark:
    """Test comparative benchmark analysis."""

    def test_comparative_benchmark_creation(self):
        """Test comparative benchmark creation."""
        result1 = BenchmarkResult("api", 10, 1.0, 0.1, 0.1, 0.05, 0.15, 10.0)
        result2 = BenchmarkResult("cache_cold", 10, 1.0, 0.08, 0.08, 0.04, 0.12, 12.5)
        result3 = BenchmarkResult("cache_hot", 10, 1.0, 0.02, 0.02, 0.01, 0.03, 50.0)

        comp = ComparativeBenchmark(
            api_direct=result1,
            cache_cold=result2,
            cache_hot=result3
        )

        assert comp.api_direct == result1
        assert comp.cache_hot == result3
        assert comp.cache_compressed is None

        # Test performance improvements
        improvements = comp.get_performance_improvement()

        hot_improvement = improvements.get('hot_cache_vs_api', 0)
        assert hot_improvement == result1.avg_response_time / result3.avg_response_time
        assert hot_improvement == 5.0  # 0.1 / 0.02

    def test_performance_improvement_calculation(self):
        """Test performance improvement calculations."""
        # Create mock results with known performance differences
        api_result = BenchmarkResult("api", 100, 50.0, 0.5, 0.5, 0.3, 0.7, 200.0)
        cache_result = BenchmarkResult("cache", 100, 49.0, 0.05, 0.05, 0.03, 0.07, 2000.0)

        comp = ComparativeBenchmark(
            api_direct=api_result,
            cache_cold=None,
            cache_hot=cache_result
        )

        improvements = comp.get_performance_improvement()

        # Hot cache should be 10x faster (0.5 / 0.05 = 10)
        assert 'hot_cache_vs_api' in improvements
        assert improvements['hot_cache_vs_api'] == 10.0

    def test_edge_cases(self):
        """Test edge cases in benchmarking."""
        # Test with empty results
        comp = ComparativeBenchmark(api_direct=None, cache_cold=None, cache_hot=None)

        improvements = comp.get_performance_improvement()
        assert improvements == {}


class TestPerformanceMetrics:
    """Test performance metrics calculations."""

    def test_throughput_calculation(self):
        """Test throughput calculation accuracy."""
        # 10 requests in 2 seconds = 5 req/sec
        result = BenchmarkResult(
            scenario="throughput_test",
            sample_size=10,
            total_time=2.0,
            avg_response_time=0.15,
            median_response_time=0.15,
            min_response_time=0.1,
            max_response_time=0.2,
            throughput=5.0
        )

        assert result.throughput == 5.0
        assert result.avg_response_time == 0.15

    def test_response_time_statistics(self):
        """Test response time statistical calculations."""
        response_times = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Manually calculate expected values
        expected_avg = sum(response_times) / len(response_times)
        expected_median = sorted(response_times)[len(response_times) // 2]
        expected_min = min(response_times)
        expected_max = max(response_times)

        result = BenchmarkResult(
            scenario="stats_test",
            sample_size=len(response_times),
            total_time=sum(response_times),
            avg_response_time=expected_avg,
            median_response_time=expected_median,
            min_response_time=expected_min,
            max_response_time=expected_max,
            throughput=len(response_times) / sum(response_times),
            response_times=response_times
        )

        assert result.avg_response_time == expected_avg
        assert result.median_response_time == expected_median
        assert result.min_response_time == expected_min
        assert result.max_response_time == expected_max


class TestIntegration:
    """Integration tests for benchmarking system."""

    def test_full_benchmark_workflow(self):
        """Test complete benchmark workflow."""
        from api_performance_benchmark import run_comprehensive_benchmark

        # Mock the benchmark results to avoid long execution time
        with patch('api_performance_benchmark.benchmark_scenario') as mock_benchmark, \
             patch('api_performance_benchmark.run_memory_efficiency_test'):

            # Set up mock results
            result1 = BenchmarkResult("Direct API", 25, 5.0, 0.2, 0.19, 0.15, 0.25, 5.0)
            result2 = BenchmarkResult("Cache Cold", 25, 2.5, 0.1, 0.09, 0.08, 0.12, 10.0)
            result3 = BenchmarkResult("Cache Hot", 25, 1.8, 0.07, 0.06, 0.05, 0.09, 13.9)
            result4 = BenchmarkResult("Cache Compressed", 25, 1.9, 0.075, 0.07, 0.06, 0.10, 13.2)

            mock_benchmark.side_effect = [result1, result2, result3, result4]

            # Should not raise exceptions
            results = run_comprehensive_benchmark()

            assert isinstance(results, dict)
            assert len(results) == 4

            # Verify results are BenchmarkResult instances
            assert all(isinstance(r, BenchmarkResult) for r in results.values())


if __name__ == "__main__":
    pytest.main([__file__])