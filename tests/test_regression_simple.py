"""
Regression tests against known results.

Basic tests to ensure model produces consistent results.
"""

import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def regression_config():
    """Configuration for regression testing."""
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
def baseline_data():
    """Create consistent baseline renewable energy data."""
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')  # 1 year of hourly data

    # Consistent solar profile
    solar_data = pd.DataFrame({
        'US.CA': np.clip(0.3 + 0.5 * np.sin(np.arange(8760) * 2 * np.pi / 24), 0, 1)
    }, index=dates)

    # Consistent wind profile
    np.random.seed(42)  # For reproducible results
    wind_data = pd.DataFrame({
        'US.CA': np.clip(np.random.beta(2, 3, 8760), 0, 1)
    }, index=dates)

    return {'solar': solar_data, 'wind': wind_data}


class TestRegressionTests:
    """Basic regression tests for model consistency."""

    def test_known_baseline_results(self, regression_config, baseline_data, tmp_path):
        """Test against known baseline results for consistency."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(regression_config, f)

        # Test configuration
        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=10,
            solar_capacity=20.0,
            wind_capacity=10.0,
            battery_power=0,
            battery_hours=0,
            solardata=baseline_data['solar'],
            winddata=baseline_data['wind']
        )

        # Calculate results
        outputs = model.calculate_electrolyser_output()
        lcoh = model.calculate_costs('fixed')

        # Basic consistency checks
        assert outputs['Generator Capacity Factor'] > 0.1
        assert outputs['Generator Capacity Factor'] < 1.0
        assert outputs['Hydrogen Output for Fixed Operation [t/yr]'] > 0
        assert lcoh > 0
        assert lcoh < 10  # Reasonable cost range

        print("Regression Test Results:")
        print(f"Generator CF: {outputs['Generator Capacity Factor']:.3f}")
        print(f"H2 Production: {outputs['Hydrogen Output for Fixed Operation [t/yr]']:.0f} t/y")
        print(f"LCOH: ${lcoh:.2f}/kg")

    def test_reproducibility(self, regression_config, baseline_data, tmp_path):
        """Test that results are reproducible with identical inputs."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(regression_config, f)

        # Run the same calculation multiple times
        results = []
        for i in range(3):
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=10,
                solar_capacity=20.0,
                wind_capacity=10.0,
                battery_power=0,
                battery_hours=0,
                solardata=baseline_data['solar'],
                winddata=baseline_data['wind']
            )

            outputs = model.calculate_electrolyser_output()
            results.append(outputs)

        # Check reproducibility
        for i in range(1, len(results)):
            assert abs(results[0]['Generator Capacity Factor'] - results[i]['Generator Capacity Factor']) < 0.001
            assert abs(results[0]['Hydrogen Output for Fixed Operation [t/yr]'] -
                      results[i]['Hydrogen Output for Fixed Operation [t/yr]']) < 1.0

        print("Reproducibility test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])