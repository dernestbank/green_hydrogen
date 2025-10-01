"""
Basic Usage Example for Hydrogen Production Model

This example demonstrates the fundamental usage of the hydrogen production
model for renewable energy-based green hydrogen production costing.

Author: Hydrogen Production Framework Team
Date: 2025
"""

import pandas as pd
import yaml
from pathlib import Path

from src.models.hydrogen_model import HydrogenModel


def main():
    """Main example function demonstrating basic model usage."""

    print("=== Hydrogen Production Model - Basic Usage Example ===\n")

    # 1. Load configuration
    print("1. Loading configuration...")
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
    except FileNotFoundError:
        print("⚠ Using default configuration")
        config = create_default_config()

    # 2. Create sample data (typically from API or files)
    print("\n2. Preparing renewable energy data...")
    solar_data, wind_data = create_sample_renewable_data()
    print("✓ Sample solar and wind data created")

    # 3. Initialize the hydrogen model
    print("\n3. Initializing hydrogen model...")
    model = HydrogenModel(
        config_path='config/config.yaml',
        location='US.CA',  # Example location
        elec_type='AE',    # Alkaline Electrolyser
        elec_capacity=50,  # 50 MW electrolyser
        solar_capacity=100.0,  # 100 MW solar farm
        wind_capacity=50.0,    # 50 MW wind farm
        battery_power=10.0,     # 10 MW battery
        battery_hours=4,        # 4 hours storage
        spot_price=50.0,        # 50 $/MWh
        solardata=solar_data,
        winddata=wind_data
    )

    print("✓ Model initialized with parameters:")
    print(f"   - Electrolyser: {model.elecCapacity} MW {model.elecType}")
    print(".1f")
    print(f"   - Battery: {model.batteryPower} MW × {model.batteryHours} h")

    # 4. Calculate operating outputs
    print("\n4. Calculating operational outputs...")
    operating_outputs = model.calculate_electrolyser_output()

    print("✓ Operational results:")
    print(".3f")
    print(".1f")
    print(".1f")
    print(".2f")

    # 5. Calculate Levelized Cost of Hydrogen (LCOH)
    print("\n5. Calculating Levelized Cost of Hydrogen...")

    # Calculate both fixed and variable SEC
    lcoh_fixed = model.calculate_costs('fixed')  # Fixed specific consumption
    lcoh_variable = model.calculate_costs('variable')  # Variable specific consumption

    print("✓ Cost analysis results:")
    print(".2f")
    print(".2f")
    print(".3f")
    print(f"   - Project lifespan: {model.projectLife} years")

    # 6. Get comprehensive results summary
    print("\n6. Generating results summary...")
    summary = model.get_results_summary()

    print("✓ Results summary generated for dashboard display")
    print(f"   - System configuration: {summary['system_configuration']['Electrolyser Capacity (MW)']} MW electrolyser")
    print(".1%")
    print(".1%")
    print(".2f")

    # 7. Display key KPIs
    print("\n7. Key Performance Indicators:")
    print(f"   ┌─────────────────────────────────────────────────┐")
    print(f"   │ Capacity Factor:     {operating_outputs['Achieved Electrolyser Capacity Factor']:.1%} │")
    print(".1f")
    print(".1f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"   └─────────────────────────────────────────────────┘")

    # 8. Summary statistics
    print("\n8. Summary Statistics:")
    stats = get_summary_statistics(operating_outputs)
    for key, value in stats.items():
        print(f"   - {key}: {value}")

    print("\n=== Example completed successfully! ===")
    print("The hydrogen production model is ready for your analysis.")
    return model, operating_outputs, summary


def create_sample_renewable_data():
    """Create sample renewable energy data for demonstration."""
    import numpy as np

    # Create 1 year of hourly data (8760 hours)
    dates = pd.date_range('2020-01-01', periods=8760, freq='h')

    # Solar generation profile (peak during day, varying by season)
    hours = np.arange(8760) % 24
    year_progress = (np.arange(8760) % 8760) / 8760

    # Seasonal solar variation (summer > winter)
    season_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (year_progress - 0.25))

    # Daily solar pattern with noise
    daily_pattern = np.maximum(0, 0.5 * np.sin(np.pi * (hours - 6) / 12))
    solar_cf = np.clip(season_factor * daily_pattern + np.random.normal(0, 0.05, 8760), 0, 1)

    # Wind generation profile (variable, with some diurnal pattern)
    wind_cf = np.clip(0.3 + 0.4 * np.random.beta(2, 3, 8760), 0, 1)

    solar_data = pd.DataFrame({'US.CA': solar_cf}, index=dates)
    wind_data = pd.DataFrame({'US.CA': wind_cf}, index=dates)

    return solar_data, wind_data


def create_default_config():
    """Create a minimal default configuration if config file not found."""
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
            'spec_consumption': 4.7,
            'stack_lifetime': 60000,
            'electrolyser_capex': 1200,
            'electrolyser_om': 3,
            'water_needs': 9
        }
    }
    return config


def get_summary_statistics(operating_outputs):
    """Calculate summary statistics from operating outputs."""
    return {
        'Average Load Factor': '.1%',
        'Peak Power Usage': '.1f',
        'Annual Capacity Factor': '.1%',
        'Operating Hours': int(operating_outputs['Total Time Electrolyser is Operating'] * 8760),
        'Efficiency': '.1%'
    }


if __name__ == "__main__":
    try:
        model, outputs, summary = main()
    except Exception as e:
        print(f"✗ Example failed with error: {e}")
        import traceback
        traceback.print_exc()