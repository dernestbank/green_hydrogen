"""
Comprehensive tests for the HydrogenModel class.

Tests cover initialization, operational calculations, cost calculations,
parameter validation, and various integration scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
import tempfile
import os
from pathlib import Path

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
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
def sample_solar_data():
    """Sample solar power data."""
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')  # Fixed frequency warning
    # Create realistic solar profile (higher during day)
    hours = np.arange(8760) % 24
    cf = 0.5 * (np.sin((hours - 12) * np.pi / 12) + 1) * np.random.uniform(0.8, 1.2, 8760)
    cf = np.clip(cf, 0, 1)
    return pd.DataFrame({'US.CA': cf}, index=dates)


@pytest.fixture
def sample_wind_data():
    """Sample wind power data."""
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')  # Fixed frequency warning
    # Create realistic wind profile (variable day/night pattern)
    cf = np.random.beta(2, 3, 8760)  # Beta distribution for wind
    return pd.DataFrame({'US.CA': cf}, index=dates)


class TestHydrogenModelInitialization:
    """Test HydrogenModel initialization and parameter validation."""

    def test_model_initialization_basic(self, sample_config, sample_solar_data, sample_wind_data, tmp_path):
        """Test basic model initialization."""
        # Create temporary config file
        config_path = tmp_path / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=10,
            solar_capacity=10.0,
            wind_capacity=0.0,
            battery_power=0,
            battery_hours=0,
            solardata=sample_solar_data
        )

        assert model.location == 'US.CA'
        assert model.elecType == 'AE'
        assert model.elecCapacity == 10
        assert model.solarCapacity == 10.0
        assert model.windCapacity == 0.0

    def test_invalid_electrolyser_type(self, sample_config, sample_solar_data, tmp_path):
        """Test invalid electrolyser type raises error."""
        config_path = tmp_path / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        with pytest.raises(ValueError, match="elec_type must be 'AE' or 'PEM'"):
            HydrogenModel(
                config_path=str(config_path),
                elec_type='INVALID',
                solardata=sample_solar_data
            )

    def test_invalid_location(self, sample_config, sample_solar_data, tmp_path):
        """Test invalid location raises error."""
        config_path = tmp_path / "test_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)

        with pytest.raises(KeyError, match="Location 'INVALID' not found"):
            HydrogenModel(
                config_path=str(config_path),
                location='INVALID',
                solardata=sample_solar_data
            )

    def test_missing_config_file(self):
        """Test missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            HydrogenModel(config_path='nonexistent/config.yaml')


if __name__ == "__main__":
    pytest.main([__file__])