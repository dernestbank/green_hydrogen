#!/usr/bin/env python3
"""
Test script to verify location-based data loading and sensitivity
"""

import sys
import os
sys.path.append('src')

from models.hydrogen_model import HydrogenModel
import pandas as pd
import numpy as np

def test_location_sensitivity():
    """Test that different locations produce different results"""

    print("=== TESTING LOCATION SENSITIVITY ===")

    # Test different locations
    locations = ['US.CA', 'US.TX', 'US.NM', 'US.MT']

    results = {}

    for location in locations:
        try:
            print(f"\n--- Testing location: {location} ---")

            # Create model with solar only for simplicity
            model = HydrogenModel(
                location=location,
                elec_type='PEM',
                elec_capacity=10,
                solar_capacity=10,
                wind_capacity=0,
                battery_power=0
            )

            # Calculate results
            operating_results = model.calculate_electrolyser_output()

            results[location] = {
                'generator_cf': operating_results['Generator Capacity Factor'],
                'electrolyser_cf': operating_results['Achieved Electrolyser Capacity Factor'],
                'annual_energy': operating_results['Energy in to Electrolyser [MWh/yr]'],
                'hydrogen_production': operating_results['Hydrogen Output for Fixed Operation [t/yr]']
            }

            print(f"Generator CF: {results[location]['generator_cf']:.3f}")
            print(f"Electrolyser CF: {results[location]['electrolyser_cf']:.3f}")
            print(f"Annual Energy: {results[location]['annual_energy']:,.0f}")
            print(f"Hydrogen Production: {results[location]['hydrogen_production']:.1f} t/yr")

        except Exception as e:
            print(f"Error testing location {location}: {e}")
            results[location] = None

    # Analyze results
    print("\n=== RESULTS ANALYSIS ===")

    if len(results) > 1:
        cfs = [r['generator_cf'] for r in results.values() if r is not None]
        if len(cfs) > 1:
            cf_range = max(cfs) - min(cfs)
            print(f"Generator CF range: {cf_range:.3f} ({min(cfs):.3f} to {max(cfs):.3f})")

            if cf_range > 0.01:  # Significant difference
                print("✅ Locations show different renewable profiles")
            else:
                print("❌ Locations show identical renewable profiles")
        else:
            print("❌ Could not calculate CF range")

        # Check if results are actually different
        unique_cfs = set()
        for loc, result in results.items():
            if result:
                unique_cfs.add(round(result['generator_cf'], 4))

        if len(unique_cfs) > 1:
            print(f"✅ Found {len(unique_cfs)} unique generator capacity factors")
        else:
            print("❌ All locations have identical capacity factors")

    return results

def test_data_loading():
    """Test that CSV data is loaded correctly"""

    print("\n=== TESTING DATA LOADING ===")

    try:
        # Load CSV files
        solar_df = pd.read_csv('Data/solar-traces.csv', header=[0], skiprows=[1], index_col=0)
        wind_df = pd.read_csv('Data/wind-traces.csv', header=[0], skiprows=[1], index_col=0)

        print(f"Solar data shape: {solar_df.shape}")
        print(f"Wind data shape: {wind_df.shape}")

        # Check available columns
        print(f"Solar columns: {list(solar_df.columns)}")
        print(f"Wind columns: {list(wind_df.columns)}")

        # Test specific location data
        test_locations = ['US.CA', 'US.TX', 'US.NM', 'US.MT']

        print("\nSolar data statistics by location:")
        for loc in test_locations:
            if loc in solar_df.columns:
                data = solar_df[loc]
                print(f"  {loc}: mean={data.mean():.4f}, std={data.std():.4f}, non_zero_hours={data[data > 0].count()}")

        print("\nWind data statistics by location:")
        for loc in test_locations:
            if loc in wind_df.columns:
                data = wind_df[loc]
                print(f"  {loc}: mean={data.mean():.4f}, std={data.std():.4f}, non_zero_hours={data[data > 0].count()}")

        return True

    except Exception as e:
        print(f"Error loading data: {e}")
        return False

if __name__ == "__main__":
    # Test data loading first
    data_loaded = test_data_loading()

    if data_loaded:
        # Test location sensitivity
        results = test_location_sensitivity()

        print("\n=== SUMMARY ===")
        if results:
            print("Location sensitivity test completed")
        else:
            print("Location sensitivity test failed")
    else:
        print("Data loading test failed")
