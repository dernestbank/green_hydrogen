#!/usr/bin/env python3
"""
Test script to verify that the Renewables Ninja API key is properly loaded from .env file.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.utils.renewables_ninja_api import RenewablesNinjaAPI

    print("🔍 Testing Renewables Ninja API key loading...")
    print("=" * 50)

    # Test 1: Check if .env file exists
    env_file = Path('.env')
    print(f"📄 .env file exists: {env_file.exists()}")

    if env_file.exists():
        # Read and display the API key from .env (masked for security)
        with open(env_file, 'r') as f:
            content = f.read().strip()

        found_key = False
        if 'RE_NINJA_API_KEY' in content:
            found_key = True
            # Extract the API key value (after the =)
            key_value = content.split('=')[1] if '=' in content else 'NOT_FOUND'
            # Mask most of the key for security
            masked_key = key_value[:8] + '...' + key_value[-4:] if len(key_value) > 12 else 'KEY_TOO_SHORT'
            print(f"🔑 API key in .env file: {masked_key}")

        if not found_key:
            print("❌ API key not found in .env file. Expected format: RE_NINJA_API_KEY=your_key_here")
            sys.exit(1)
    else:
        print("⚠️  No .env file found. Creating example...")
        with open('.env', 'w') as f:
            f.write("# Add your Renewables Ninja API key here\n")
            f.write("# Get your API key from: https://www.renewables.ninja/user/api\n")
            f.write("RE_NINJA_API_KEY=your_api_key_here\n")
        print("📝 Created .env file template. Please add your API key.")
        sys.exit(1)

    # Test 2: Try to initialize API without providing key (should auto-load from .env)
    print("\n🚀 Testing automatic API key loading...")

    try:
        # This should automatically load from .env if python-dotenv is available
        api = RenewablesNinjaAPI()  # No key provided - should load from .env

        # Mask the API key for security display
        masked_api_key = api.api_key[:8] + '***' + api.api_key[-4:] if len(api.api_key) > 12 else '***'
        print(f"✅ API initialized successfully with key: {masked_api_key}")
        print(f"🌐 API Base URL: {api.base_url}")

        # Test 3: Try a quick API call to verify the key works
        print("\n🧪 Testing API connectivity...")
        try:
            # Try to get a small amount of data (this may take a moment)
            test_data = api.get_solar_data(lat=-33.86, lon=151.21, date_from='2024-01-01', date_to='2024-01-02')
            if 'data' in test_data:
                data_points = len(test_data['data'])
                print(f"✅ API call successful! Retrieved {data_points} data points")
                if data_points > 0:
                    print("🎉 Your Renewables Ninja API key is fully functional!")
            else:
                print("⚠️  API call succeeded but no data returned (check date range)")
        except Exception as e:
            print(f"⚠️  API call failed: {str(e)[:100]}...")
            print("This could be due to network issues or rate limiting.")

    except ValueError as e:
        if "No API key provided" in str(e):
            print("❌ Automatic API key loading failed.")
            print("Possible issues:")
            print("1. python-dotenv module not installed (run: pip install python-dotenv)")
            print("2. .env file not in the current directory")
            print("3. API key format incorrect in .env file")
            print("4. Environment variable conflicts")
            sys.exit(1)
        else:
            print(f"❌ API initialization failed: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✅ API key loading test completed successfully!")
    print("Your Renewables Ninja API is ready to use!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the 'src' directory is available and all dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)