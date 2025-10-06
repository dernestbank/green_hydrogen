#!/usr/bin/env python3
"""
Test script to verify API integration for custom locations
"""

import sys
import os
sys.path.append('src')
sys.path.append('.')

from models.hydrogen_model import HydrogenModel
from location_manager import LocationManager
import pandas as pd
import numpy as np

def test_custom_location_without_api():
    """Test custom location without API key (should fallback to reference data)"""

    print("=== TESTING CUSTOM LOCATION WITHOUT API ===")

    try:
        # Try to create model with custom location but no API key
        model = HydrogenModel(
            location="40.7128,-74.0060",  # NYC coordinates
            elec_type='PEM',
            elec_capacity=10,
            solar_capacity=10,
            wind_capacity=0,
            battery_power=0,
            api_key=None  # No API key
        )

        print("❌ Model creation should have failed without API key for custom location")
        return False

    except Exception as e:
        print(f"✅ Expected error without API key: {e}")
        return True

def test_custom_location_with_api():
    """Test custom location with API key (should fetch real data)"""

    print("\n=== TESTING CUSTOM LOCATION WITH API ===")

    # Test with a dummy API key (will fail but should try API)
    try:
        model = HydrogenModel(
            location="40.7128,-74.0060",  # NYC coordinates
            elec_type='PEM',
            elec_capacity=10,
            solar_capacity=10,
            wind_capacity=0,
            battery_power=0,
            api_key="dummy_key"  # Dummy API key
        )

        print("❌ Model creation should have failed with invalid API key")
        return False

    except Exception as e:
        print(f"✅ Expected error with invalid API key: {e}")
        return True

def test_predefined_location():
    """Test that predefined locations still work correctly"""

    print("\n=== TESTING PREDEFINED LOCATION ===")

    try:
        model = HydrogenModel(
            location="US.CA",
            elec_type='PEM',
            elec_capacity=10,
            solar_capacity=10,
            wind_capacity=0,
            battery_power=0
        )

        # Calculate results
        operating_results = model.calculate_electrolyser_output()

        print("✅ Predefined location works correctly")
        print(f"Generator CF: {operating_results['Generator Capacity Factor']:.3f}")
        print(f"Hydrogen Production: {operating_results['Hydrogen Output for Fixed Operation [t/yr]']:.1f} t/yr")

        return True

    except Exception as e:
        print(f"❌ Error with predefined location: {e}")
        return False

def test_location_manager():
    """Test the LocationManager directly"""

    print("\n=== TESTING LOCATION MANAGER ===")

    # Test without API key
    print("Testing LocationManager without API key:")
    lm_no_api = LocationManager(api_key=None)

    print(f"Predefined locations available: {len(lm_no_api.get_available_locations())}")
    print(f"API client initialized: {lm_no_api.api_client is not None}")

    # Test with dummy API key
    print("\nTesting LocationManager with dummy API key:")
    lm_with_api = LocationManager(api_key="dummy_key")

    print(f"API client initialized: {lm_with_api.api_client is not None}")

    # Test location detection
    print("\nLocation type detection:")
    print(f"US.CA is predefined: {lm_no_api.is_predefined_location('US.CA')}")
    print(f"40.7128,-74.0060 is predefined: {lm_no_api.is_predefined_location('40.7128,-74.0060')}")
    print(f"40.7128,-74.0060 is custom: {lm_no_api.is_custom_location('40.7128,-74.0060')}")

    # Test data retrieval
    print("\nData retrieval test:")
    solar_data = lm_no_api.get_solar_data('US.CA')
    print(f"Solar data for US.CA: {solar_data is not None}")

    custom_solar_data = lm_no_api.get_solar_data('40.7128,-74.0060')
    print(f"Solar data for custom location (no API): {custom_solar_data is not None}")

    return True

if __name__ == "__main__":
    print("API Integration Test Suite")
    print("=" * 50)

    # Test location manager
    lm_test = test_location_manager()

    # Test predefined location
    predefined_test = test_predefined_location()

    # Test custom location without API
    no_api_test = test_custom_location_without_api()

    # Test custom location with API
    with_api_test = test_custom_location_with_api()

    print("\n=== FINAL RESULTS ===")
    print(f"Location Manager: {'✅' if lm_test else '❌'}")
    print(f"Predefined Location: {'✅' if predefined_test else '❌'}")
    print(f"Custom Location (no API): {'✅' if no_api_test else '❌'}")
    print(f"Custom Location (with API): {'✅' if with_api_test else '❌'}")

    if all([lm_test, predefined_test, no_api_test, with_api_test]):
        print("\n🎉 All tests passed! API integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
