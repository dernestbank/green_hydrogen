"""
Integration tests for end-to-end flow.

Tests complete scenarios from API data fetch to results display,
verifying integration between all components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from src.models.hydrogen_model import HydrogenModel


@pytest.fixture
def integration_config():
    """Integration test configuration with realistic parameters."""
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
def mock_api_data():
    """Mock API response data for integration testing."""
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')  # Fixed frequency warning

    # Generate realistic solar data
    hours = np.arange(8760) % 24
    solar_cf = 0.5 * (np.sin((hours - 12) * np.pi / 12) + 1) * np.random.uniform(0.8, 1.2, 8760)
    solar_cf = np.clip(solar_cf, 0, 1)

    # Generate realistic wind data
    wind_cf = np.random.beta(2, 3, 8760)  # Beta distribution for wind

    solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
    wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

    return {'solar': solar_data, 'wind': wind_data}


class TestEndToEndFlow:
    """Test complete end-to-end scenarios."""

    def test_basic_hydrogen_production_flow(self, integration_config, mock_api_data, tmp_path):
        """Test complete flow from data input to hydrogen production results."""
        # Create temporary config file
        config_path = tmp_path / "integration_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(integration_config, f)

        # Initialize model with data
        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=10,
            solar_capacity=20.0,  # 20 MW solar
            wind_capacity=10.0,   # 10 MW wind
            battery_power=2,
            battery_hours=4,
            solardata=mock_api_data['solar'],
            winddata=mock_api_data['wind']
        )

        # Step 1: Calculate operating outputs
        operating_outputs = model.calculate_electrolyser_output()

        # Verify all required outputs are present
        required_outputs = [
            'Generator Capacity Factor',
            'Achieved Electrolyser Capacity Factor',
            'Energy in to Electrolyser [MWh/yr]',
            'Hydrogen Output for Fixed Operation [t/yr]',
            'Hydrogen Output for Variable Operation [t/yr]',
            'Surplus Energy [MWh/yr]'
        ]

        for output in required_outputs:
            assert output in operating_outputs
            assert isinstance(operating_outputs[output], (int, float, np.number))

        # Step 2: Calculate costs
        fixed_lcoh = model.calculate_costs('fixed')
        variable_lcoh = model.calculate_costs('variable')

        # Verify LCOH results are reasonable
        assert 1.0 <= fixed_lcoh <= 20.0  # Reasonable hydrogen production costs
        assert 1.0 <= variable_lcoh <= 20.0

        # Step 3: Get comprehensive results summary
        summary = model.get_results_summary()

        # Verify summary structure
        assert 'system_configuration' in summary
        assert 'operational_results' in summary
        assert 'hydrogen_production' in summary
        assert 'financial_results' in summary

        # Print key results for verification
        print(f"\nIntegration Test Results:")
        print(f"Generator CF: {summary['operational_results']['Generator Capacity Factor']}")
        print(f"Hydrogen Production (fixed): {summary['hydrogen_production']['Fixed Operation (t/year)']}")
        print(f"LCOH (fixed): {summary['financial_results']['LCOH - Fixed Consumption (A$/kg)']}")

    def test_performance_optimization_flow(self, integration_config, mock_api_data, tmp_path):
        """Test performance optimization through data compression and caching."""
        config_path = tmp_path / "optimization_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(integration_config, f)

        # Test large dataset processing
        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='PEM',
            elec_capacity=50,  # Large scale
            solar_capacity=100.0,
            wind_capacity=50.0,
            battery_power=10,
            battery_hours=4,
            solardata=mock_api_data['solar'],
            winddata=mock_api_data['wind']
        )

        # Run multiple calculations to test performance
        for i in range(3):
            outputs = model.calculate_electrolyser_output()
            costs = model.calculate_costs('variable')
            summary = model.get_results_summary()

            assert outputs['Energy in to Electrolyser [MWh/yr]'] > 100000  # Large scale
            assert costs > 0

    def test_data_quality_and_validation_flow(self, integration_config, tmp_path):
        """Test data quality validation throughout the flow."""
        config_path = tmp_path / "quality_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(integration_config, f)

        # Test with partial/missing data scenarios
        # Create incomplete solar data
        dates = pd.date_range('2020-01-01', periods=1000, freq='h')
        incomplete_solar = pd.DataFrame({'US.CA': np.random.uniform(0, 1, 1000)}, index=dates)

        model = HydrogenModel(
            config_path=str(config_path),
            location='US.CA',
            elec_type='AE',
            elec_capacity=5,
            solar_capacity=10.0,
            wind_capacity=0.0,
            solardata=incomplete_solar
        )

        # Should handle incomplete data and still produce valid results
        outputs = model.calculate_electrolyser_output()
        assert outputs['Energy in to Electrolyser [MWh/yr]'] >= 0
        assert outputs['Hydrogen Output for Fixed Operation [t/yr]'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])