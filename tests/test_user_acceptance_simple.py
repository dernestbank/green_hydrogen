"""
Simple User Acceptance Tests

Basic validation that the system meets core user requirements.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import time

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def basic_config():
    """Basic configuration for quick testing."""
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


@pytest.fixture
def sample_data():
    """Generate sample solar and wind data."""
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')
    solar_cf = np.clip(0.4 + 0.3 * np.random.beta(2, 3, 8760), 0, 1)
    wind_cf = np.clip(0.3 + 0.2 * np.random.beta(2.5, 2, 8760), 0, 1)

    solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
    wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

    return {'solar': solar_data, 'wind': wind_data}


class TestUserAcceptance:
    """Basic user acceptance tests."""

    def test_core_functionality(self, basic_config, sample_data, tmp_path):
        """Test that core functionality works as expected."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(basic_config, f)

        print("Running core functionality test...")

        # Initialize model
        start_time = time.time()
        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=20,
            solar_capacity=40.0,
            wind_capacity=20.0,
            battery_power=4,
            battery_hours=4,
            solardata=sample_data['solar'],
            winddata=sample_data['wind']
        )

        # Test all major functions
        outputs = model.calculate_electrolyser_output()
        lcoh = model.calculate_costs('fixed')
        summary = model.get_results_summary()
        end_time = time.time()

        # Validate results
        assert outputs['Hydrogen Output for Fixed Operation [t/yr]'] > 0
        assert outputs['Achieved Electrolyser Capacity Factor'] > 0
        assert lcoh > 0
        assert lcoh < 10  # Reasonable range
        assert end_time - start_time < 5.0  # Performance requirement

        print(f"✅ Simple scenario PASSED - LCOH: ${lcoh:.2f}/kg")
        print(f"   Execution time: {end_time - start_time:.1f}s")
        print(f"   H2 production: {outputs['Hydrogen Output for Fixed Operation [t/yr]']:.0f} t/yr")
    def test_error_handling(self, basic_config, sample_data, tmp_path):
        """Test user-friendly error handling."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(basic_config, f)

        print("\\nTesting error handling...")

        # Test invalid electrolyser type
        try:
            model = HydrogenModel(
                config_path=str(config_path),
                elec_type='INVALID',
                solardata=sample_data['solar']
            )
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "elec_type must be 'AE' or 'PEM'" in str(e)
            print("✓ Clear error message for invalid type")

        # Test invalid location
        try:
            model = HydrogenModel(
                config_path=str(config_path),
                location='INVALID',
                solardata=sample_data['solar']
            )
            assert False, "Should raise KeyError"
        except (KeyError, ValueError):  # Accept either
            print("✓ Clear error message for invalid location")

    def test_different_system_sizes(self, basic_config, sample_data, tmp_path):
        """Test scalability across different system sizes."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(basic_config, f)

        sizes = [5, 10, 25, 50]  # MW capacities
        results = {}

        print("\\nTesting different system sizes...")

        for size in sizes:
            start_time = time.time()
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=size,
                solar_capacity=size * 1.5,
                wind_capacity=size * 0.5,
                solardata=sample_data['solar'],
                winddata=sample_data['wind']
            )

            outputs = model.calculate_electrolyser_output()
            lcoh = model.calculate_costs('fixed')
            end_time = time.time()

            results[size] = {
                'h2_production': outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'lcoh': lcoh,
                'execution_time': end_time - start_time,
                'capacity_factor': outputs['Achieved Electrolyser Capacity Factor']
            }

            # Performance check
            assert end_time - start_time < 3.0, f"System size {size}MW too slow"

        # Validate scaling makes sense
        for i in range(1, len(sizes)):
            old_size, new_size = sizes[i-1], sizes[i]
            old_h2, new_h2 = results[old_size]['h2_production'], results[new_size]['h2_production']

            # Should scale roughly linearly
            ratio = new_h2 / old_h2 if old_h2 > 0 else 0
            expected_ratio = new_size / old_size
            assert abs(ratio - expected_ratio) / expected_ratio < 0.5, \
                f"Bad scaling: size ratio {expected_ratio}, H2 ratio {ratio}"

        print("✓ All system sizes perform appropriately")
        max_time = max(r['execution_time'] for r in results.values())
        print(f"   Maximum execution time: {max_time:.1f}s")

    def test_system_reliability(self, basic_config, sample_data, tmp_path):
        """Test system reliability and consistency."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(basic_config, f)

        print("\\nTesting system reliability...")

        # Run same analysis multiple times
        results = []
        for i in range(5):
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=15,
                solar_capacity=30.0,
                wind_capacity=15.0,
                solardata=sample_data['solar'],
                winddata=sample_data['wind']
            )

            outputs = model.calculate_electrolyser_output()
            lcoh = model.calculate_costs('fixed')

            results.append({
                'h2_production': outputs['Hydrogen Output for Fixed Operation [t/yr]'],
                'lcoh': lcoh
            })

        # Check consistency
        h2_values = [r['h2_production'] for r in results]
        h2_cv = np.std(h2_values) / np.mean(h2_values) if np.mean(h2_values) > 0 else 0

        lcoh_values = [r['lcoh'] for r in results]
        lcoh_cv = np.std(lcoh_values) / np.mean(lcoh_values) if np.mean(lcoh_values) > 0 else 0

        # Coefficient of variation should be low (results should be consistent)
        assert h2_cv < 0.01, f"H2 production not consistent: CV = {h2_cv:.3f}"
        assert lcoh_cv < 0.05, f"LCOH not consistent: CV = {lcoh_cv:.3f}"

        print(f"✅ System reliability PASSED - H2 CV: {h2_cv:.3f}")
        print(f"   LCOH CV: {lcoh_cv:.3f}")
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])