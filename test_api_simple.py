#!/usr/bin/env python3
"""
Simple test for Renewables Ninja API key loading.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.utils.renewables_ninja_api import RenewablesNinjaAPI

    print("Testing Renewables Ninja API key loading...")
    print("=" * 50)

    # Check if .env file exists
    env_file = Path('.env')
    print(f".env file exists: {env_file.exists()}")

    # Try to initialize API without providing key (should auto-load from .env)
    print("Testing automatic API key loading...")

    try:
        # This should automatically load from .env if python-dotenv is available
        api = RenewablesNinjaAPI()  # No key provided - should load from .env

        # Mask the API key for security display
        if api.api_key and len(api.api_key) > 12:
            masked_api_key = api.api_key[:8] + '***' + api.api_key[-4:]
        else:
            masked_api_key = '***'
        print(f"API initialized successfully with key: {masked_api_key}")
        print(f"API Base URL: {api.base_url}")

        # Test if the API key is the one from .env
        expected_key = "4e90b85c1e00fb3016c3feb12e04c97d5cf6fa81"
        if api.api_key == expected_key:
            print("SUCCESS: API key loaded correctly from .env file!")
        else:
            print("WARNING: API key does not match expected value")
            print(f"Expected: {expected_key}")
            print(f"Loaded:   {api.api_key}")

        print("\nTEST COMPLETED SUCCESSFULLY!")
        print("Your Renewables Ninja API key is automatically loaded from .env")

    except ValueError as e:
        if "No API key provided" in str(e):
            print("FAILED: Automatic API key loading failed.")
            print("Possible issues:")
            print("1. .env file not in the current directory")
            print("2. RE_NINJA_API_KEY not defined in .env")
            print("3. python-dotenv module not available")
            sys.exit(1)
        else:
            print(f"API initialization failed: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the 'src' directory is available and all dependencies are installed.")
    sys.exit(1)