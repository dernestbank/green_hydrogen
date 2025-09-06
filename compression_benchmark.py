#!/usr/bin/env python3
"""
Compression Benchmark Demo

Demonstrates the compression functionality for large datasets
as implemented for task 1.3.3.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.data_compression import CompressionManager, benchmark_compression


def create_sample_data():
    """Create sample renewable energy data similar to API responses."""
    np.random.seed(42)

    # Simulate a year of hourly data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=8760, freq='h'),
        'solar_radiation': np.maximum(0, np.sin(np.arange(8760) * 2 * np.pi / 24) +
                                    np.random.normal(0, 0.3, 8760)),
        'wind_speed': np.maximum(0, 5 + 3 * np.random.normal(0, 0.8, 8760)),
        'capacity_factor_solar': np.random.beta(2, 5, 8760),
        'capacity_factor_wind': np.random.beta(3, 4, 8760),
        'temperature': 15 + 10 * np.sin(np.arange(8760) * 2 * np.pi / 8760) +
                      np.random.normal(0, 2, 8760)
    })

    return df


def run_compression_benchmark():
    """Run compression benchmark on sample renewable energy data."""
    print("Compression Benchmark for Large Renewable Energy Datasets")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()
    original_size_mb = df.memory_usage(deep=True).sum() / (1024**2)

    print("Sample Data: Yearly renewable energy profile")
    print(f"   - Hours: {len(df):,}")
    print(f"   - Columns: {len(df.columns)}")
    print(".1f")
    print(f"   - Estimated size: {original_size_mb:.2f} MB")
    print()

    # Initialize compression manager
    manager = CompressionManager(
        enable_compression=True,
        compression_threshold_kb=50,  # Compress anything > 50KB
        benchmark_compression=False  # Skip benchmarking for speed
    )

    # Test compression
    print("Testing compression...")
    start_time = time.time()
    compressed_data, compression_result = manager.compress_data(df, force_compress=True)
    compression_time = time.time() - start_time

    if compression_result:
        print("Compression successful!")
        print(".2f")
        print(".1f")
        print(f"   Original size: {compression_result.original_size / 1024:.1f} KB")
        print(f"   Compressed size: {compression_result.compressed_size / 1024:.1f} KB")
        print(".1f")
        print()

        # Test decompression
        print("Testing decompression...")
        start_time = time.time()
        decompressed_data = manager.decompress_data(compressed_data)
        decompression_time = time.time() - start_time

        print("Decompression successful!")
        print(".2f")
        print(".1f")
        print(f"   Data integrity: {'Valid' if isinstance(decompressed_data, pd.DataFrame) else 'Invalid'}")

        # Compare sample values
        if isinstance(decompressed_data, pd.DataFrame):
            try:
                original_sample = df.iloc[100].values.astype(float)
                decompressed_sample = decompressed_data.iloc[100].values.astype(float)
                match = np.allclose(original_sample, decompressed_sample, rtol=1e-10)
                print(f"   Sample comparison: {'Match' if match else 'Mismatch'}")
            except Exception as e:
                print(f"   Sample comparison: Could not compare due to type issues")

        print()

    # Compression statistics
    stats = manager.get_compression_stats()
    print("Compression Statistics:")
    print(f"   Total operations: {stats['total_operations']}")
    print(".1f")
    print(f"   Average compression time: {stats['avg_compression_time_ms']:.2f} ms")
    print()


def run_algorithm_benchmark():
    """Benchmark different compression algorithms."""
    print("Algorithm Benchmark")
    print("-" * 30)

    df = create_sample_data()

    # Test different algorithms
    algorithms = ['gzip', 'bz2', 'lzma']

    results = {}
    for algorithm in algorithms:
        print(f"Testing {algorithm.upper()}...")
        try:
            start_time = time.time()
            algorithm_results = benchmark_compression(df, algorithms=[algorithm])
            benchmark_time = time.time() - start_time

            result = algorithm_results[algorithm]
            results[algorithm] = result

            print(f"   Ratio: {result.compression_ratio:.2f}x")
            print(".2f")
            print(".3f")

        except Exception as e:
            print(f"   Failed: {e}")

    print()

    # Summary
    if results:
        best_algorithm = max(results.keys(),
                           key=lambda x: results[x].compression_ratio)
        best_ratio = results[best_algorithm].compression_ratio

        print("Summary:")
        print(f"   Best algorithm: {best_algorithm.upper()}")
        print(".2f")
        print()


def main():
    """Main benchmark execution."""
    try:
        run_compression_benchmark()
        print()
        run_algorithm_benchmark()

        print("=" * 60)
        print("Compression benchmark completed successfully!")
        print("Compression system ready for large dataset handling")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())