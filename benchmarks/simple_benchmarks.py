"""
Simple Performance Benchmarks

Basic benchmarking script to measure execution performance
and provide key performance metrics.
"""

import time
import psutil
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List

from src.models.hydrogen_model import HydrogenModel


def run_simple_benchmark():
    """Run a simple performance benchmark suite."""

    print("🧪 Running Performance Benchmarks...\n")

    # System information
    print("System Information:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(".1f")

    # Configuration
    config = {
        'elec_max_load': 100,
        'elec_reference_capacity': 10,
        'elec_efficiency': 83,
        'h2_vol_to_mass': 0.089,
        'elec_min_load': 10,
        'elec_cost_reduction': 1.0,
        'solar_capex': 1120,
        'solar_opex': 16990,
        'wind_capex': 1942,
        'wind_opex': 25000,
        'battery_capex': {0: 0, 1: 827, 2: 542, 4: 446, 8: 421},
        'battery_opex': {0: 0, 1: 4833, 2: 9717, 4: 19239, 8: 39314},
        'battery_replacement': 100,
        'battery_efficiency': 85,
        'battery_min': 0,
        'battery_lifetime': 10,
        'powerplant_reference_capacity': 1,
        'powerplant_cost_reduction': 1.0,
        'powerplant_equip': 1.0,
        'powerplant_install': 0.0,
        'powerplant_land': 0.0,
        'elec_equip': 1.0,
        'elec_install': 0.0,
        'elec_land': 0.0,
        'electrolyser_stack_cost': 40,
        'water_cost': 5,
        'discount_rate': 4,
        'project_life': 20,
        'ae': {
            'elec_min_load': 10,
            'elec_overload': 100,
            'elec_overload_recharge': 0,
            'spec_consumption': 4.7,
            'stack_lifetime': 60000,
            'electrolyser_capex': 1000,
            'electrolyser_om': 4,
            'water_needs': 10
        },
        'pem': {
            'elec_min_load': 10,
            'elec_overload': 100,
            'elec_overload_recharge': 0,
            'spec_consumption': 4.4,
            'stack_lifetime': 50000,
            'electrolyser_capex': 1200,
            'electrolyser_om': 3,
            'water_needs': 9
        }
    }

    # Create config file
    config_path = Path("temp_benchmark_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Generate test data
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')
    solar_data = pd.DataFrame({
        'US.CA': np.clip(np.random.beta(2, 3, 8760), 0, 1)
    }, index=dates)
    wind_data = pd.DataFrame({
        'US.CA': np.clip(np.random.beta(2.5, 2.5, 8760), 0, 1)
    }, index=dates)

    # Test scenarios
    scenarios = {
        'Small System': {'capacity': 10, 'solar': 20.0, 'wind': 0.0},
        'Medium System': {'capacity': 25, 'solar': 50.0, 'wind': 25.0, 'battery': True},
        'Large System': {'capacity': 100, 'solar': 200.0, 'wind': 100.0, 'battery': True}
    }

    results = {}

    print("\nBenchmarking different system sizes:")
    print("-" * 50)

    for scenario_name, params in scenarios.items():
        print(f"\nTesting: {scenario_name}")

        # Initialize model
        init_start = time.time()
        model = HydrogenModel(
            config_path=str(config_path),
            elec_type='AE',
            elec_capacity=params['capacity'],
            solar_capacity=params['solar'],
            wind_capacity=params['wind'],
            battery_power=int(params['capacity'] * 0.1) if params.get('battery') else 0,
            battery_hours=4 if params.get('battery') else 0,
            solardata=solar_data,
            winddata=wind_data
        )
        init_time = time.time() - init_start

        # Operating calculations
        calc_start = time.time()
        operating_outputs = model.calculate_electrolyser_output()
        calc_time = time.time() - calc_start

        # Cost calculations
        cost_start = time.time()
        lcoh_fixed = model.calculate_costs('fixed')
        cost_time = time.time() - cost_start

        # Summary
        summary_start = time.time()
        summary = model.get_results_summary()
        summary_time = time.time() - summary_start

        total_time = init_time + calc_time + cost_time + summary_time
        h2_production = operating_outputs['Hydrogen Output for Fixed Operation [t/yr]']

        results[scenario_name] = {
            'total_time': total_time,
            'init_time': init_time,
            'calc_time': calc_time,
            'cost_time': cost_time,
            'h2_production': h2_production,
            'lcoh': lcoh_fixed,
            'capacity_factor': operating_outputs['Achieved Electrolyser Capacity Factor']
        }

        print(f"   System size: {params['capacity']} MW")
        print(".1f"        print(f"   H2 production: {h2_production:.0f} t/yr")
        print(".1%"        print(".3f"        print(".2f"

    # Comparative analysis
    print("\n" + "="*50)
    print("COMPARATIVE ANALYSIS:")
    print("="*50)

    sizes = list(results.keys())
    times = [r['total_time'] for r in results.values()]

    print("Time comparison:")
    for scenario, result in results.items():
        print(".2f"
    # Performance assessment
    assessment = {}
    for scenario, result in results.items():
        assessment[scenario] = "PASS" if result['total_time'] < 10.0 else "REVIEW"

    print("\nPerformance Assessment:")
    for scenario, status in assessment.items():
        print(f"   {scenario}: {status}")

    # Summary statistics
    total_execution_time = sum([r['total_time'] for r in results.values()])
    avg_lcoh = sum([r['lcoh'] for r in results.values()]) / len(results)

    print("
Summary Statistics:"    print(".2f"    print(".2f"    print(".1f"
    # Cleanup
    config_path.unlink(missing_ok=True)

    print("\n✅ Benchmark completed successfully")

    return results


def benchmark_memory_usage():
    """Basic memory usage benchmark."""
    print("\n🧠 Running memory usage assessment...")

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Simple model creation
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')
    solar_data = pd.DataFrame({'US.CA': [0.5] * 8760}, index=dates)
    wind_data = pd.DataFrame({'US.CA': [0.3] * 8760}, index=dates)

    model_memory = process.memory_info().rss / 1024 / 1024

    model = HydrogenModel(
        elec_type='AE',
        elec_capacity=25,
        solar_capacity=50.0,
        wind_capacity=25.0,
        battery_power=5,
        battery_hours=4,
        solardata=solar_data,
        winddata=wind_data
    )

    with_model_memory = process.memory_info().rss / 1024 / 1024

    # Run calculations
    operating_outputs = model.calculate_electrolyser_output()
    lcoh = model.calculate_costs('fixed')

    final_memory = process.memory_info().rss / 1024 / 1024

    print(".1f")
    print(".1f")
    print(".1f")

    # Cleanup
    del model, solar_data, wind_data

    return {
        'initial_memory': initial_memory,
        'model_memory': model_memory,
        'with_model_memory': with_model_memory,
        'final_memory': final_memory
    }


def main():
    """Main benchmark function."""
    print("🚀 Hydrogen Production Framework - Performance Benchmarks")
    print("=" * 60)

    # Run main benchmark
    performance_results = run_simple_benchmark()

    # Run memory benchmark
    memory_results = benchmark_memory_usage()

    print("\n" + "=" * 60)
    print("✓ Benchmark suite completed successfully")

    # Save results
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'performance_results': performance_results,
        'memory_results': memory_results
    }

    output_file = Path("benchmarks/benchmark_results.json")
    output_file.parent.mkdir(exist_ok=True)

    import json
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"✅ Results saved to {output_file}")

    return output


if __name__ == "__main__":
    main()