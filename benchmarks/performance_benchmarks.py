"""
Performance Benchmarks

Comprehensive benchmarking suite to measure and analyze system performance
across various scenarios, dataset sizes, and operating conditions.
"""

import time
import memory_profiler
import psutil
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import cProfile
import pstats
from contextlib import contextmanager

from src.models.hydrogen_model import HydrogenModel


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite.

    Measures execution times, memory usage, scalability, and provides
    detailed performance analysis reports.
    """

    def __init__(self):
        self.results = {}
        self.baseline_config = {
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

    @contextmanager
    def track_resources(self, operation_name: str):
        """
        Context manager to track CPU and memory usage for an operation.

        Args:
            operation_name: Name of the operation being tracked
        """
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        try:
            yield
        finally:
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_time = time.time()

            memory_used = end_memory - start_memory
            execution_time = end_time - start_time

            self.results[f"{operation_name}_memory_mb"] = memory_used
            self.results[f"{operation_name}_time_seconds"] = execution_time

    def benchmark_initialization(self, system_sizes: List[int]) -> Dict[str, Any]:
        """
        Benchmark model initialization times for different system sizes.

        Args:
            system_sizes: List of system sizes (MW) to test

        Returns:
            Benchmark results for initialization
        """
        print("\n🧪 Benchmarking model initialization...")

        results = {}
        base_data_dates = pd.date_range('2020-01-01', periods=8760, freq='h')
        solar_data = pd.DataFrame({
            'US.CA': np.clip(np.random.beta(2, 3, 8760), 0, 1)
        }, index=base_data_dates)
        wind_data = pd.DataFrame({
            'US.CA': np.clip(np.random.beta(2.5, 2.5, 8760), 0, 1)
        }, index=base_data_dates)

        for size in system_sizes:
            print(f"   Testing {size} MW system...")

            start_time = time.time()
            with self.track_resources(f"init_{size}mw"):
                model = HydrogenModel(
                    elec_type='AE',
                    elec_capacity=size,
                    solar_capacity=size * 1.5,
                    wind_capacity=size * 0.5,
                    battery_power=int(size * 0.1),
                    battery_hours=4,
                    solardata=solar_data,
                    winddata=wind_data
                )
            end_time = time.time()

            results[size] = {
                'initialization_time': end_time - start_time,
                'memory_used_mb': self.results.get(f"init_{size}mw_memory_mb", 0)
            }

        return results

    def benchmark_calculations(self, test_scenarios: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Benchmark calculation performance for different scenarios.

        Args:
            test_scenarios: Dictionary of scenario names to configurations

        Returns:
            Benchmark results for calculations
        """
        print("\n🧮 Benchmarking calculations...")

        results = {}
        base_data_dates = pd.date_range('2020-01-01', periods=8760, freq='h')
        solar_data = pd.DataFrame({
            'US.CA': np.clip(np.random.beta(2, 3, 8760), 0, 1)
        }, index=base_data_dates)
        wind_data = pd.DataFrame({
            'US.CA': np.clip(np.random.beta(2.5, 2.5, 8760), 0, 1)
        }, index=base_data_dates)

        for scenario_name, config in test_scenarios.items():
            print(f"   Testing scenario: {scenario_name}")

            model = HydrogenModel(
                elec_type=config.get('electrolyser', 'AE'),
                elec_capacity=config.get('capacity', 20),
                solar_capacity=config.get('solar', 40.0),
                wind_capacity=config.get('wind', 0.0),
                battery_power=config.get('battery_power', 0),
                battery_hours=config.get('battery_hours', 0),
                solardata=solar_data,
                winddata=wind_data
            )

            # Benchmark each major calculation step
            step_times = {}
            total_memory = 0

            # Operating outputs calculation
            with self.track_resources(f"calc_{scenario_name}_operating"):
                start_time = time.time()
                operating_outputs = model.calculate_electrolyser_output()
                end_time = time.time()
            step_times['operating_outputs'] = end_time - start_time
            total_memory += self.results.get(f"calc_{scenario_name}_operating_memory_mb", 0)

            # Fixed cost calculation
            with self.track_resources(f"calc_{scenario_name}_fixed_cost"):
                start_time = time.time()
                lcoh_fixed = model.calculate_costs('fixed')
                end_time = time.time()
            step_times['cost_calculation_fixed'] = end_time - start_time
            total_memory += self.results.get(f"calc_{scenario_name}_fixed_cost_memory_mb", 0)

            # Variable cost calculation
            with self.track_resources(f"calc_{scenario_name}_variable_cost"):
                start_time = time.time()
                lcoh_variable = model.calculate_costs('variable')
                end_time = time.time()
            step_times['cost_calculation_variable'] = end_time - start_time
            total_memory += self.results.get(f"calc_{scenario_name}_variable_cost_memory_mb", 0)

            # Summary calculation
            with self.track_resources(f"calc_{scenario_name}_summary"):
                start_time = time.time()
                summary = model.get_results_summary()
                end_time = time.time()
            step_times['summary_calculation'] = end_time - start_time
            total_memory += self.results.get(f"calc_{scenario_name}_summary_memory_mb", 0)

            results[scenario_name] = {
                'step_times': step_times,
                'total_time': sum(step_times.values()),
                'total_memory_mb': total_memory,
                'performance_metrics': {
                    'h2_production': operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 0),
                    'capacity_factor': operating_outputs.get('Achieved Electrolyser Capacity Factor', 0),
                    'lcoh_fixed': lcoh_fixed,
                    'lcoh_variable': lcoh_variable
                }
            }

        return results

    def benchmark_scalability(self, dataset_sizes: List[int]) -> Dict[str, Any]:
        """
        Benchmark system scalability with different dataset sizes.

        Args:
            dataset_sizes: List of dataset sizes (hours) to test

        Returns:
            Scalability benchmark results
        """
        print("\n📊 Benchmarking scalability...")

        results = {}

        for hours in dataset_sizes:
            print(f"   Testing with {hours} hours of data...")

            # Create dataset of specified size
            dates = pd.date_range('2020-01-01', periods=hours, freq='h')
            solar_data = pd.DataFrame({
                'US.CA': np.clip(np.random.beta(2, 3, hours), 0, 1)
            }, index=dates)

            if hours > 8760:  # Add wind data for larger datasets
                wind_data = pd.DataFrame({
                    'US.CA': np.clip(np.random.beta(2.5, 2.5, hours), 0, 1)
                }, index=dates)
                wind_capacity = 20.0
            else:
                wind_data = pd.DataFrame({'US.CA': [0.0] * hours}, index=dates)
                wind_capacity = 0.0

            with self.track_resources(f"scale_{hours}h"):
                start_time = time.time()
                model = HydrogenModel(
                    elec_type='AE',
                    elec_capacity=25,
                    solar_capacity=50.0,
                    wind_capacity=wind_capacity,
                    battery_power=5,
                    battery_hours=4,
                    solardata=solar_data,
                    winddata=wind_data
                )

                operating_outputs = model.calculate_electrolyser_output()
                lcoh = model.calculate_costs('fixed')
                end_time = time.time()

            results[hours] = {
                'data_points': hours,
                'execution_time': end_time - start_time,
                'memory_used_mb': self.results.get(f"scale_{hours}h_memory_mb", 0),
                'h2_production': operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 0),
                'processing_rate': hours / (end_time - start_time) if (end_time - start_time) > 0 else 0
            }

        return results

    def benchmark_optimized_configurations(self) -> Dict[str, Any]:
        """
        Benchmark performance with different optimization configurations.

        Returns:
            Optimized configuration benchmark results
        """
        print("\n⚡ Benchmarking optimized configurations...")

        optimizations = {
            'baseline': {},
            'no_battery': {'battery_power': 0, 'battery_hours': 0},
            'large_battery': {'battery_power': 10, 'battery_hours': 8},
            'high_efficiency': {'elec_type': 'PEM', 'elec_efficiency': 90},
            'cost_optimized': {'solar_capex': 800, 'wind_capex': 1500}
        }

        results = {}
        base_data_dates = pd.date_range('2020-01-01', periods=8760, freq='h')

        for config_name, config_override in optimizations.items():
            print(f"   Testing configuration: {config_name}")

            # Merge configurations
            config = self.baseline_config.copy()
            config.update(config_override)

            # Create config file
            config_path = Path(f"temp_config_{config_name}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            solar_data = pd.DataFrame({
                'US.CA': np.clip(np.random.beta(2, 3, 8760), 0, 1)
            }, index=base_data_dates)
            wind_data = pd.DataFrame({
                'US.CA': np.clip(np.random.beta(2.5, 2.5, 8760), 0, 1)
            }, index=base_data_dates)

            with self.track_resources(f"opt_{config_name}"):
                start_time = time.time()
                model = HydrogenModel(
                    config_path=str(config_path),
                    elec_type=config.get('elec_type', 'AE'),
                    elec_capacity=25,
                    solar_capacity=50.0,
                    wind_capacity=config.get('wind_capex', 1942) / 1942 * 25,  # Scale based on cost
                    battery_power=config.get('battery_power', 5),
                    battery_hours=config.get('battery_hours', 4),
                    solardata=solar_data,
                    winddata=wind_data
                )

                operating_outputs = model.calculate_electrolyser_output()
                lcoh = model.calculate_costs('fixed')
                end_time = time.time()

            # Cleanup
            config_path.unlink(missing_ok=True)

            results[config_name] = {
                'execution_time': end_time - start_time,
                'memory_used_mb': self.results.get(f"opt_{config_name}_memory_mb", 0),
                'h2_production': operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 0),
                'lcoh': lcoh,
                'capacity_factor': operating_outputs.get('Achieved Electrolyser Capacity Factor', 0)
            }

        return results

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite and generate comprehensive report.

        Returns:
            Complete benchmark results
        """
        print("🚀 Starting comprehensive performance benchmark suite...\n")

        print("System Information:")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")

        # Define test scenarios
        test_scenarios = {
            'small_system': {'capacity': 10, 'solar': 20.0, 'wind': 0.0},
            'medium_system': {'capacity': 25, 'solar': 50.0, 'wind': 25.0, 'battery_power': 5},
            'large_system': {'capacity': 100, 'solar': 200.0, 'wind': 50.0, 'battery_power': 20, 'battery_hours': 8}
        }

        # Run all benchmarks
        benchmark_results = {
            'initialization': self.benchmark_initialization([5, 10, 25, 50, 100]),
            'calculations': self.benchmark_calculations(test_scenarios),
            'scalability': self.benchmark_scalability([8760, 17520, 35040]),  # 1, 2, 4 years
            'optimizations': self.benchmark_optimized_configurations(),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        return benchmark_results

    def generate_performance_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive performance report.

        Args:
            results: Benchmark results
            output_file: Optional file path to save report

        Returns:
            Formatted performance report
        """
        report = []
        report.append("╔════════════════════════════════════════════════════════════════╗")
        report.append("║                    PERFORMANCE BENCHMARK REPORT                ║")
        report.append("╚════════════════════════════════════════════════════════════════╝\n")

        # Summary statistics
        report.append("📊 SUMMARY STATISTICS:")
        report.append("-" * 50)

        if 'initialization' in results:
            init_times = [v['initialization_time'] for v in results['initialization'].values()]
            report.append(".2f"
        if 'scalability' in results:
            scale_rates = [v['processing_rate'] for v in results['scalability'].values()]
            report.append(".0f"
        report.append("")

        # Detailed results
        report.append("🔍 DETAILED RESULTS:")
        report.append("-" * 50)

        for category, data in results.items():
            if category.startswith(('initialization', 'calculations', 'scalability', 'optimizations')):
                report.append(f"\n{category.upper()} BENCHMARKS:")
                report.append("-" * 30)

                if isinstance(data, dict):
                    for item, metrics in data.items():
                        report.append(f"• {item}:")

                        if 'execution_time' in metrics:
                            report.append(".3f")
                        if 'memory_used_mb' in metrics:
                            report.append(".2f")
                        if 'processing_rate' in metrics:
                            report.append(".0f")
                        if 'total_time' in metrics:
                            report.append(".3f")

                        # Add performance metrics if available
                        if 'performance_metrics' in metrics:
                            perf = metrics['performance_metrics']
                            report.append(".0f")
                            report.append(".3f")
                            report.append(".3f")

                        report.append("")

        # Performance recommendations
        report.append("💡 PERFORMANCE RECOMMENDATIONS:")
        report.append("-" * 35)

        # Basic analysis
        if 'scalability' in results:
            scale_results = results['scalability']
            hours_list = list(scale_results.keys())
            times_list = [scale_results[h]['execution_time'] for h in hours_list]

            if len(hours_list) > 1:
                # Check for performance degradation
                degradation = (times_list[-1] / times_list[0]) / (hours_list[-1] / hours_list[0])
                if degradation > 2.0:
                    report.append("⚠️  Performance degrades significantly with larger datasets")
                    report.append("   Consider data partitioning or streaming processing")
                else:
                    report.append("✅ Scalability performance is good")

        if 'optimizations' in results:
            opt_results = results['optimizations']
            baseline_lcoh = opt_results.get('baseline', {}).get('lcoh', 100)
            baseline_time = opt_results.get('baseline', {}).get('execution_time', 1)

            best_lcoh_config = min(opt_results.items(), key=lambda x: x[1].get('lcoh', 100))
            best_time_config = min(opt_results.items(), key=lambda x: x[1].get('execution_time', 100))

            if best_lcoh_config[0] != 'baseline':
                improvement = (baseline_lcoh - best_lcoh_config[1]['lcoh']) / baseline_lcoh
                report.append(".1f")

            if best_time_config[0] != 'baseline':
                speedup = baseline_time / best_time_config[1]['execution_time']
                report.append(".1f")

        report.append("")
        report.append("Report generated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

        final_report = "\n".join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(final_report)
            print(f"\nReport saved to: {output_file}")

        return final_report


def main():
    """Run complete benchmark suite."""
    print("🚀 Hydrogen Production Framework - Performance Benchmarks\n")

    # Initialize benchmark suite
    benchmark = PerformanceBenchmark()

    # Run comprehensive benchmarks
    results = benchmark.run_full_benchmark_suite()

    # Generate and display report
    report = benchmark.generate_performance_report(
        results,
        output_file="reports/performance_benchmark_report.txt"
    )

    print("\n" + "="*70)
    print("BENCHMARK REPORT SUMMARY:")
    print("="*70)

    # Quick summary
    if 'scalability' in results:
        scale_results = results['scalability']
        max_rate = max([r['processing_rate'] for r in scale_results.values()])
        print(".0f")

    if 'optimizations' in results:
        opt_results = results['optimizations']
        best_lcoh = min([r.get('lcoh', 100) for r in opt_results.values()])
        print(".2f")

    print("\nDetailed report saved to: reports/performance_benchmark_report.txt")

    return results


if __name__ == "__main__":
    main()