"""
User Acceptance Tests - Minimal

Validates core user requirements are met.
"""

import pytest
import pandas as pd
import numpy as np

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def config():
    """Basic config for testing."""
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
        }
    }


@pytest.fixture
def test_data():
    """Generate test data."""
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')
    solar_cf = np.clip(0.4 + 0.3 * np.random.beta(2, 3, 8760), 0, 1)
    wind_cf = np.clip(0.3 + 0.2 * np.random.beta(2.5, 2, 8760), 0, 1)

    solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
    wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

    return {'solar': solar_data, 'wind': wind_data}


class TestUserAcceptance:
    """User acceptance tests."""

    def test_industrial_user_scenario(self, config, test_data, tmp_path):
        """Test industrial hydrogen production meets user criteria."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=30,  # Larger system
            solar_capacity=60.0,
            wind_capacity=30.0,
            battery_power=6,
            battery_hours=2,
            solardata=test_data['solar'],
            winddata=test_data['wind']
        )

        outputs = model.calculate_electrolyser_output()
        lcoh = model.calculate_costs('fixed')

        # User acceptance criteria
        assert outputs['Hydrogen Output for Fixed Operation [t/yr]'] >= 100, "Insufficient H2 production"
        assert outputs['Achieved Electrolyser Capacity Factor'] >= 0.20, "Low capacity factor"
        assert 1.5 <= lcoh <= 8.0, f"LCOH ${lcoh} outside acceptable range"

    def test_error_handling(self, config, test_data, tmp_path):
        """Test system provides helpful error messages."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Test invalid electrolyser type
        try:
            model = HydrogenModel(
                config_path=str(config_path),
                elec_type='INVALID',
                solardata=test_data['solar']
            )
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "elec_type must be" in str(e), "Unclear error message"

        # Test invalid location
        try:
            model = HydrogenModel(
                config_path=str(config_path),
                location='INVALID',
                solardata=test_data['solar']
            )
            assert False, "Should raise error"
        except (KeyError, ValueError):
            pass  # Accept any appropriate error

    def test_scalability(self, config, test_data, tmp_path):
        """Test system scales appropriately with capacity."""
        config_path = tmp_path / "config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        results = {}
        for capacity in [10, 20, 40]:
            model = HydrogenModel(
                config_path=str(config_path),
                location='US.CA',
                elec_type='AE',
                elec_capacity=capacity,
                solar_capacity=capacity * 1.5,
                wind_capacity=capacity * 0.5,
                solardata=test_data['solar'],
                winddata=test_data['wind']
            )

            outputs = model.calculate_electrolyser_output()
            results[capacity] = outputs['Hydrogen Output for Fixed Operation [t/yr]']

        # Check scaling makes sense
        small, medium, large = results[10], results[20], results[40]
        assert medium > small, "Medium system should produce more than small"
        assert large > medium, "Large system should produce more than medium"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])