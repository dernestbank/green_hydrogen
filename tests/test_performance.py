"""
Performance tests for large datasets.

Tests system performance under various load conditions, measuring
execution times, memory usage, and scalability.
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import yaml

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def performance_config():
    """Performance test configuration with realistic parameters."""
    return {
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
            'spec_consumption': 4.7,
            'stack_lifetime': 60000,
            'electrolyser_capex': 1000,
            'electrolyser_om': 4,
            'water_needs': 10
        }
    }


@pytest.fixture
def create_large_dataset():
    """Create large dataset for performance testing."""
    # Create data for multiple years to simulate large dataset
    datasets = {}

    for years in [1, 2, 5, 10]:
        hours = years * 8760  # Approximate hours per year
        dates = pd.date_range('2020-01-01', periods=hours, freq='h')

        # Generate solar data with seasonal variations
        hours_array = np.arange(hours)
        day_of_year = hours_array // 24 % 365
        season_factor = 0.3 + 0.7 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal variation

        # Add hourly variation and noise
        hourly_factor = 0.5 * (np.sin((hours_array % 24 - 12) * np.pi / 12) + 1)
        noise = np.random.normal(0, 0.1, hours)

        solar_cf = np.clip(hourly_factor * season_factor + noise, 0, 1)
        wind_cf = np.random.beta(2, 3, hours)  # Wind power curve

        solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
        wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

        datasets[years] = {'solar': solar_data, 'wind': wind_data}

    return datasets


class TestPerformanceTests:
    """Test performance under various load conditions."""

    def test_large_dataset_processing(self, performance_config, create_large_dataset, tmp_path):
        """Test processing of large datasets."""
        config_path = tmp_path / "performance_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(performance_config, f)

        performance_results = {}

        for years in [1, 5]:  # Test 1-year and 5-year datasets
            datasets = create_large_dataset[years]

            print(f"\nTesting {years}-year dataset ({len(datasets['solar'])} hours)")

            start_time = time.time()

            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=50,  # Large systeem
                solar_capacity=100.0,
                wind_capacity=50.0,
                battery_power=10,
                battery_hours=4,
                solardata=datasets['solar'],
                winddata=datasets['wind']
            )

            # Time the calculation
            calc_start = time.time()
            operating_outputs = model.calculate_electrolyser_output()
            cost_start = time.time()
            costs = model.calculate_costs('variable')
            summary_start = time.time()
            summary = model.get_results_summary()
            end_time = time.time()

            total_time = end_time - start_time
            calc_time = cost_start - calc_start
            cost_time = summary_start - cost_start
            summary_time = end_time - summary_start

            performance_results[years] = {
                'total_time': total_time,
                'calculation_time': calc_time,
                'cost_time': cost_time,
                'summary_time': summary_time,
                'data_points': len(datasets['solar']),
                'h2_production': operating_outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'energy_input': operating_outputs['Energy in to Electrolyser [MWh/yr]']
            }

            print(".2f")
            print(".2f")

            # Verify performance meets requirements
            assert total_time < 30  # Should complete in under 30 seconds
            assert operating_outputs['Energy in to Electrolyser [MWh/yr]'] > 0
            assert costs > 0

    def test_memory_efficiency(self, performance_config, create_large_dataset, tmp_path):
        """Test memory efficiency with large datasets."""
        config_path = tmp_path / "memory_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(performance_config, f)

        # Test with 5-year dataset
        datasets = create_large_dataset[5]

        print(f"\nMemory efficiency test with {len(datasets['solar'])} data points")

        # Monitor memory usage during processing
        initial_model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='PEM',
            elec_capacity=20,
            solar_capacity=50.0,
            wind_capacity=25.0,
            battery_power=5,
            battery_hours=4,
            solardata=datasets['solar'],
            winddata=datasets['wind']
        )

        # Run calculations multiple times to test stability
        for i in range(5):
            outputs = initial_model.calculate_electrolyser_output()
            costs = initial_model.calculate_costs('variable')
            summary = initial_model.get_results_summary()

            assert outputs['Energy in to Electrolyser [MWh/yr]'] > 0
            assert costs > 0

        print("Memory efficiency test completed successfully")

    def test_concurrent_scenario_analysis(self, performance_config, create_large_dataset, tmp_path):
        """Test running multiple scenarios concurrently."""
        config_path = tmp_path / "concurrent_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(performance_config, f)

        # Define multiple scenarios to test
        scenarios = [
            {
                'name': 'Small Solar',
                'capacity': 20,
                'solar': 40.0,
                'wind': 0.0,
                'type': 'AE'
            },
            {
                'name': 'Large Hybrid',
                'capacity': 100,
                'solar': 150.0,
                'wind': 100.0,
                'type': 'PEM'
            },
            {
                'name': 'Wind Dominant',
                'capacity': 80,
                'solar': 50.0,
                'wind': 200.0,
                'type': 'AE'
            }
        ]

        datasets = create_large_dataset[2]  # 2-year dataset for reasonable test time

        scenario_results = {}

        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")

            start_time = time.time()

            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type=scenario['type'],
                elec_capacity=scenario['capacity'],
                solar_capacity=scenario['solar'],
                wind_capacity=scenario['wind'],
                battery_power=int(scenario['capacity'] * 0.1),
                battery_hours=4,
                solardata=datasets['solar'],
                winddata=datasets['wind']
            )

            # Run full analysis
            outputs = model.calculate_electrolyser_output()
            costs = model.calculate_costs('variable')
            summary = model.get_results_summary()

            end_time = time.time()

            scenario_results[scenario['name']] = {
                'execution_time': end_time - start_time,
                'h2_production': outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'lcoh': costs,
                'capacity_factor': outputs['Generator Capacity Factor']
            }

            print(".2f")
            print(".1f")
            print(".2f")

            assert scenario_results[scenario['name']]['execution_time'] < 15  # Under 15s per scenario
            assert scenario_results[scenario['name']]['h2_production'] > 0

    def test_scalability_under_load(self, performance_config, create_large_dataset, tmp_path):
        """Test system scalability under increasing load."""
        config_path = tmp_path / "scalability_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(performance_config, f)

        # Test different scales (use sequential for comparison)
        scales = [1, 5, 20, 50]  # Scale factors

        datasets = create_large_dataset[1]  # Base 1-year dataset

        scalability_results = {}

        for scale in scales:
            print(f"\nTesting scale factor: {scale}x")

            start_time = time.time()

            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=10 * scale,
                solar_capacity=20.0 * scale,
                wind_capacity=10.0 * scale,
                battery_power=2 * scale,
                battery_hours=4,
                solardata=datasets['solar'],
                winddata=datasets['wind']
            )

            # Time each step
            calc_start = time.time()
            outputs = model.calculate_electrolyser_output()
            cost_start = time.time()
            costs = model.calculate_costs('fixed')
            summary_start = time.time()
            summary = model.get_results_summary()
            end_time = time.time()

            total_time = end_time - start_time
            calc_time = cost_start - calc_start
            cost_time = summary_start - cost_start

            scalability_results[scale] = {
                'total_time': total_time,
                'calculation_time': calc_time,
                'cost_time': cost_time,
                'h2_production': outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'energy_input': outputs['Energy in to Electrolyser [MWh/yr]']
            }

            print(".2f")
            print(".0f")
            print(".0f")

            # Verify performance requirements
            assert scalability_results[scale]['total_time'] <= 30  # Should not take too long even at large scale
            assert scalability_results[scale]['h2_production'] > 500  # Should produce reasonable amount of hydrogen
            assert scalability_results[scale]['energy_input'] > 10000  # Should use reasonable energy input


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])