"""Pytest configuration file for Hydrogen Production Framework tests."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from pathlib import Path


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = {
        'elecMaxLoad': 100,
        'elecReferenceCapacity': 10,
        'elecCostReduction': 1.0,
        'elecEquip': 1.0,
        'elecInstall': 0.0,
        'elecLand': 0.0,
        'AE': {
            'elecMinLoad': 20,
            'elecOverload': 100,
            'elecOverloadRecharge': 0,
            'specCons': 4.5,
            'stackLifetime': 80000,
            'electrolyserCapex': 1000,
            'electrolyserOandM': 2.5,
            'waterNeeds': 10
        },
        'PEM': {
            'elecMinLoad': 10,
            'elecOverload': 120,
            'elecOverloadRecharge': 4,
            'specCons': 4.7,
            'stackLifetime': 60000,
            'electrolyserCapex': 1000,
            'electrolyserOandM': 4,
            'waterNeeds': 10
        },
        'H2VoltoMass': 0.089,
        'elecEff': 83,
        'batteryEfficiency': 85,
        'battMin': 0.0,
        'battLifetime': 10,
        'discountRate': 4,
        'projectLife': 20
    }
    return config


@pytest.fixture
def sample_location_data():
    """Create sample renewable energy data for testing."""
    # Create hourly data for 24 hours
    hours = range(24)

    # Solar data: higher during day, lower at night
    solar_cf = [
        0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6, 0.8,  # 00:00 - 07:00
        0.9, 1.0, 0.95, 0.9, 0.85, 0.7, 0.5, 0.3,  # 08:00 - 15:00
        0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0     # 16:00 - 23:00
    ]

    # Wind data: variable throughout the day
    wind_cf = [
        0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,  # 00:00 - 07:00
        0.8, 0.85, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65,  # 08:00 - 15:00
        0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25    # 16:00 - 23:00
    ]

    return {
        'solar_cf': solar_cf,
        'wind_cf': wind_cf,
        'hours': list(hours)
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def mock_api_response():
    """Mock API response for testing."""
    return {
        'data': [
            {'hour': h, 'solar_cf': 0.5 + 0.3 * np.sin(h / 24 * 2 * np.pi),
             'wind_cf': 0.5 + 0.4 * np.cos(h / 24 * 2 * np.pi)}
            for h in range(24)
        ],
        'metadata': {
            'location': 'Test_City',
            'latitude': -34.0522,
            'longitude': 150.2447
        }
    }


@pytest.fixture
def sample_user_inputs():
    """Sample user inputs for testing the hydrogen model."""
    return {
        'nominal_electrolyser_capacity': 10.0,  # MW
        'nominal_solar_farm_capacity': 10.0,    # MW
        'nominal_wind_farm_capacity': 5.0,     # MW
        'battery_rated_power': 5.0,            # MW
        'duration_of_storage_hours': 2,         # hours
        'generator_type': 'Hybrid',
        'electrolyser_choice': 'PEM',
        'sec_at_nominal_load': 50.0,
        'electrolyser_min_load': 0.1,
        'electrolyser_max_load': 1.0,
        'max_overload_duration': 0,
        'time_between_overload': 0
    }


@pytest.fixture
def sample_calculation_results():
    """Sample calculation results for testing visualization functions."""
    return {
        'annual_hydrogen_production': 10000,  # tonnes/year
        'levelized_cost_hydrogen': 4.5,      # AUD/kg
        'capacity_factor': 0.75,
        'total_energy_consumed': 180000,     # MWh/year
        'energy_surplus': 15000,            # MWh/year
        'stack_replacement_schedule': [5, 8, 11, 14, 17]
    }

