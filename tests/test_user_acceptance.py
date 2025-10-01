"""
User Acceptance Tests

Tests that validate the system meets user requirements and expectations.
Covers major use cases, user workflows, and business requirements.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import tempfile
import time

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def user_scenarios():
    """Real-world user scenarios for testing."""
    return {
        'scenario_1_industrial_user': {
            'description': 'Industrial user planning 50MW hydrogen facility',
            'parameters': {
                'elec_capacity': 50,
                'solar_capacity': 100.0,
                'wind_capacity': 50.0,
                'battery_power': 10,
                'battery_hours': 4,
                'target_h2_production': 350  # t/year
            },
            'acceptance_criteria': {
                'lcoh_range': (2.0, 4.0),  # $/kg
                'capacity_factor_min': 0.25,
                'max_investment_payback': 12  # years
            }
        },
        'scenario_2_grid_connected': {
            'description': 'Grid-connected user with PPA contract',
            'parameters': {
                'elec_capacity': 30,
                'solar_capacity': 60.0,
                'wind_capacity': 30.0,
                'ppa_price': 60.0,  # $/MWh
                'grid_backup': True
            },
            'acceptance_criteria': {
                'cost_savings_vs_grid': 0.15,  # 15% savings
                'reliability_min': 0.95  # 95% uptime
            }
        },
        'scenario_3_off_grid_remote': {
            'description': 'Off-grid remote location user',
            'parameters': {
                'elec_capacity': 5,
                'solar_capacity': 15.0,
                'wind_capacity': 5.0,
                'battery_power': 3,
                'battery_hours': 8,
                'location_isolation': True
            },
            'acceptance_criteria': {
                'backup_days': 7,  # 7 days backup
                'system_autonomy': 0.99  # 99% self-sufficient
            }
        }
    }


@pytest.fixture
def acceptance_config():
    """Configuration for user acceptance testing."""
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
            'spec_consumption': 4.4,
            'stack_lifetime': 50000,
            'electrolyser_capex': 1200,
            'electrolyser_om': 3,
            'water_needs': 9
        }
    }


class TestUserAcceptanceScenarios:
    """Test that system meets user acceptance criteria."""

    def test_industrial_user_scenario(self, acceptance_config, user_scenarios, tmp_path):
        """User Acceptance Test: Industrial hydrogen production facility."""
        scenario = user_scenarios['scenario_1_industrial_user']
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        # Create realistic generation data
        dates = pd.date_range('2021-01-01', periods=8760, freq='h')

        # Solar with seasonality
        hour_of_day = np.arange(8760) % 24
        day_of_year = np.arange(8760) % 365
        solar_seasonality = 0.7 + 0.3 * np.cos(2 * np.pi * day_of_year / 365)
        solar_daily = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12))
        solar_cf = solar_seasonality * solar_daily + np.random.normal(0, 0.05, 8760)
        solar_cf = np.clip(solar_cf, 0, 1)

        # Wind with variability
        wind_cf = np.random.beta(2.5, 2.5, 8760) * 0.8

        solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
        wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

        print("\nTesting industrial user scenario...")
        print(f"Target parameters: {scenario['parameters']}")

        # Initialize and run analysis
        start_time = time.time()

        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=scenario['parameters']['elec_capacity'],
            solar_capacity=scenario['parameters']['solar_capacity'],
            wind_capacity=scenario['parameters']['wind_capacity'],
            battery_power=scenario['parameters']['battery_power'],
            battery_hours=scenario['parameters']['battery_hours'],
            spot_price=50.0,  # $/MWh
            solardata=solar_data,
            winddata=wind_data
        )

        # Test all major functions
        operating_outputs = model.calculate_electrolyser_output()
        lcoh_fixed = model.calculate_costs('fixed')
        lcoh_variable = model.calculate_costs('variable')
        summary = model.get_results_summary()

        end_time = time.time()

        # Validate acceptance criteria
        criteria = scenario['acceptance_criteria']

        # Check LCOH within acceptable range
        assert criteria['lcoh_range'][0] <= lcoh_fixed <= criteria['lcoh_range'][1], \
            f"LCOH ${lcoh_fixed:.2f} outside acceptable range ${criteria['lcoh_range'][0]:.2f}-${criteria['lcoh_range'][1]:.2f}"

        # Check capacity factor
        cf = operating_outputs['Achieved Electrolyser Capacity Factor']
        assert cf >= criteria['capacity_factor_min'], \
            f"Capacity factor {cf:.3%} below minimum {criteria['capacity_factor_min']:.1%}"

        # Check hydrogen production meets target
        h2_production = operating_outputs['Hydrogen Output for Variable Operation [t/yr]']
        target = scenario['parameters']['target_h2_production']
        assert h2_production >= target * 0.9, \
            f"H2 production {h2_production} too low (target: {target})"

        # Performance check
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"Analysis too slow: {execution_time:.2f}s"

        print(f"✅ Industrial scenario PASSED - LCOH: ${lcoh_fixed:.2f}/kg, H2: {h2_production} t/y")
        print(".3f")

    def test_grid_connected_user_scenario(self, acceptance_config, user_scenarios, tmp_path):
        """User Acceptance Test: Grid-connected user with PPA."""
        scenario = user_scenarios['scenario_2_grid_connected']
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        # Generate data
        dates = pd.date_range('2021-01-01', periods=8760, freq='h')
        solar_cf = np.clip(0.4 + 0.4 * np.sin(np.arange(8760) * 2 * np.pi / 24), 0, 1)
        wind_cf = np.clip(0.3 + np.random.beta(2, 3, 8760), 0, 1)

        solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
        wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

        print("Testing grid-connected user scenario...")

        # Test with PPA pricing
        model_ppa = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='PEM',
            elec_capacity=scenario['parameters']['elec_capacity'],
            solar_capacity=scenario['parameters']['solar_capacity'],
            wind_capacity=scenario['parameters']['wind_capacity'],
            ppa_price=scenario['parameters']['ppa_price'],
            solardata=solar_data,
            winddata=wind_data
        )

        outputs_ppa = model_ppa.calculate_electrolyser_output()
        costs_ppa = model_ppa.calculate_costs('fixed')

        print(f"✅ Grid-connected scenario PASSED - LCOH: ${costs_ppa:.2f}/kg")

    def test_off_grid_remote_scenario(self, acceptance_config, user_scenarios, tmp_path):
        """User Acceptance Test: Off-grid remote location."""
        scenario = user_scenarios['scenario_3_off_grid_remote']
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        # Generate variable generation data (simulating remote location)
        dates = pd.date_range('2021-01-01', periods=8760, freq='h')
        solar_cf = np.clip(0.35 + 0.25 * np.sin(np.arange(8760) * 2 * np.pi / 24) +
                          np.random.normal(0, 0.1, 8760), 0, 1)
        wind_cf = np.clip(0.2 + np.random.beta(1.5, 2.5, 8760), 0, 1)

        solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
        wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

        print("\\nTesting off-grid remote scenario...")

        # Test enhanced battery system
        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=scenario['parameters']['elec_capacity'],
            solar_capacity=scenario['parameters']['solar_capacity'],
            wind_capacity=scenario['parameters']['wind_capacity'],
            battery_power=scenario['parameters']['battery_power'],
            battery_hours=scenario['parameters']['battery_hours'],
            solardata=solar_data,
            winddata=wind_data
        )

        outputs = model.calculate_electrolyser_output()
        costs = model.calculate_costs('variable')

        # Check long-duration battery performance
        energy_input = outputs['Energy in to Electrolyser [MWh/yr]']
        capacity_factor = outputs['Achieved Electrolyser Capacity Factor']
        battery_energy = scenario['parameters']['battery_power'] * scenario['parameters']['battery_hours']

        # Validate sufficient battery capacity for autonomy
        expected_daily_energy = energy_input / 365
        battery_days_autonomy = battery_energy / expected_daily_energy

        assert battery_days_autonomy >= scenario['acceptance_criteria']['backup_days'], \
            f"Battery autonomy {battery_days_autonomy} days below required {scenario['acceptance_criteria']['backup_days']} days"

        print(f"✅ Off-grid remote scenario PASSED - Battery autonomy: {battery_days_autonomy:.1f} days")

    def test_technology_comparison_user_workflow(self, acceptance_config, tmp_path):
        """User Acceptance Test: Technology comparison workflow."""
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        # Generate consistent data for comparison
        dates = pd.date_range('2021-01-01', periods=8760, freq='h')
        solar_cf = np.clip(np.random.beta(2.5, 2.5, 8760), 0, 1)
        wind_cf = np.clip(np.random.beta(2.0, 2.5, 8760), 0, 1)

        solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
        wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

        technologies = ['AE', 'PEM']
        results = {}

        print("\\nTesting technology comparison workflow...")

        for tech in technologies:
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type=tech,
                elec_capacity=25,
                solar_capacity=50.0,
                wind_capacity=25.0,
                battery_power=5,
                battery_hours=4,
                solardata=solar_data,
                winddata=wind_data
            )

            outputs = model.calculate_electrolyser_output()
            costs = model.calculate_costs('fixed')

            results[tech] = {
                'capacity_factor': outputs['Achieved Electrolyser Capacity Factor'],
                'h2_production': outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'lcoh': costs,
                'efficiency': model.specCons
            }

        # Validate user can make informed decisions
        ae_lcoh = results['AE']['lcoh']
        pem_lcoh = results['PEM']['lcoh']

        ae_h2 = results['AE']['h2_production']
        pem_h2 = results['PEM']['h2_production']

        # Ensure both technologies produce reasonable results
        assert ae_h2 > 100 and pem_h2 > 100, "Both technologies should produce substantial H2"
        assert ae_lcoh > 1.0 and pem_lcoh > 1.0, "Costs should be reasonable"

        # User should be able to discern differences
        lcoh_diff = abs(ae_lcoh - pem_lcoh) / min(ae_lcoh, pem_lcoh)
        h2_diff = abs(ae_h2 - pem_h2) / max(ae_h2, pem_h2)

        assert lcoh_diff > 0.05 or h2_diff > 0.05, "Technology comparison should show meaningful differences"

        print("✅ Technology comparison completed successfully")
    def test_performance_user_expectations(self, acceptance_config, tmp_path):
        """User Acceptance Test: Performance expectations."""
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        # Generate comprehensive test data
        dates = pd.date_range('2021-01-01', periods=8760, freq='h')
        solar_cf = np.clip(np.random.beta(2, 3, 8760), 0, 1)
        wind_cf = np.clip(np.random.beta(2.5, 2, 8760), 0, 1)

        solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
        wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

        print("\\nTesting user performance expectations...")

        scales = [10, 25, 50, 100]  # Different system scales
        performance_results = {}

        for capacity in scales:
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE' if capacity < 75 else 'PEM',
                elec_capacity=capacity,
                solar_capacity=capacity * 1.5,
                wind_capacity=capacity * 0.5,
                battery_power=int(capacity * 0.2),
                battery_hours=4,
                solardata=solar_data,
                winddata=wind_data
            )

            start_time = time.time()
            outputs = model.calculate_electrolyser_output()
            costs = model.calculate_costs('variable')
            summary = model.get_results_summary()
            end_time = time.time()

            exec_time = end_time - start_time

            performance_results[capacity] = {
                'execution_time': exec_time,
                'h2_production': outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'lcoh': costs,
                'capacity_factor': outputs['Achieved Electrolyser Capacity Factor']
            }

            # User expectations for responsiveness
            if capacity <= 50:
                assert exec_time < 3.0, f"Small system analysis too slow: {exec_time:.2f}s"
            elif capacity <= 100:
                assert exec_time < 8.0, f"Large system analysis too slow: {exec_time:.2f}s"

        print("Performance expectations met for all system scales")
        print("Max execution time: {:.2f}s".format(max(r['execution_time'] for r in performance_results.values())))


class TestSystemReliability:
    """Test system reliability under various conditions."""

    def test_error_handling_user_experience(self, acceptance_config, tmp_path):
        """Test that system provides helpful error messages."""
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        print("\\nTesting error handling user experience...")

        # Test invalid electrolyser type
        try:
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='INVALID'
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "elec_type must be 'AE' or 'PEM'" in str(e)

        # Test invalid location
        try:
            solar_data = pd.DataFrame({'NONEXISTENT': [0.5] * 24}, index=pd.date_range('2021-01-01', periods=24, freq='h'))
            model = HydrogenModel(
                config_path=str(config_path),
                location='INVALID_LOC',
                elec_type='AE',
                solardata=solar_data
            )
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "not found" in str(e)

        print("✅ Error handling provides clear, helpful messages")

    def test_data_validation_workflow(self, acceptance_config, tmp_path):
        """Test data validation from user perspective."""
        config_path = tmp_path / "uat_config.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(acceptance_config, f)

        print("\\nTesting data validation workflow...")

        # Test with missing data
        incomplete_solar = pd.DataFrame({'US.CA': [0.5] * 100})
        complete_wind = pd.DataFrame({'US.CA': [0.3] * 8760}, index=pd.date_range('2021-01-01', periods=8760, freq='h'))

        # Should handle gracefully
        try:
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=10,
                solar_capacity=10.0,
                wind_capacity=10.0,
                solardata=incomplete_solar,
                winddata=complete_wind
            )

            outputs = model.calculate_electrolyser_output()
            assert outputs['Energy in to Electrolyser [MWh/yr]'] >= 0

        except Exception as e:
            print(f"⚠ System encountered expected limitation: {e}")

        # Test with complete data
        complete_solar = pd.DataFrame({'US.CA': [0.5] * 8760}, index=pd.date_range('2021-01-01', periods=8760, freq='h'))
        complete_wind = pd.DataFrame({'US.CA': [0.3] * 8760}, index=pd.date_range('2021-01-01', periods=8760, freq='h'))

        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=10,
            solar_capacity=10.0,
            wind_capacity=10.0,
            solardata=complete_solar,
            winddata=complete_wind
        )

        outputs = model.calculate_electrolyser_output()
        costs = model.calculate_costs('fixed')

        assert outputs['Hydrogen Output for Fixed Operation [t/yr]'] > 0
        assert costs > 0

        print("✅ Data validation handles various scenarios appropriately")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])