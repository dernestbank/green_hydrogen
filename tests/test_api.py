import pytest
from unittest.mock import Mock, patch
from src.utils.renewables_ninja_api import RenewablesNinjaAPI
import requests

# Mock API responses for testing
MOCK_SOLAR_RESPONSE = {
    "data": {
        "2024-01-01 12:00": {
            "electricity": 0.85,
            "irradiance": 750.5,
            "capacity_factor": 0.85
        },
        "2024-01-01 13:00": {
            "electricity": 0.75,
            "irradiance": 650.2,
            "capacity_factor": 0.75
        }
    },
    "metadata": {
        "points": 2,
        "recorded_period": "[2024-01-01 12:00, 2024-01-01 13:00]"
    }
}

MOCK_WIND_RESPONSE = {
    "data": {
        "2024-01-01 12:00": {
            "electricity": 0.65,
            "wind_speed": 8.5,
            "capacity_factor": 0.65
        },
        "2024-01-01 13:00": {
            "electricity": 0.72,
            "wind_speed": 9.2,
            "capacity_factor": 0.72
        }
    },
    "metadata": {
        "points": 2,
        "recorded_period": "[2024-01-01 12:00, 2024-01-01 13:00]"
    }
}

class TestRenewablesNinjaAPI:

    @pytest.fixture
    def api_client(self):
        """Create API client instance for testing"""
        return RenewablesNinjaAPI("test_token_for_testing")

    @patch('requests.Session.get')
    def test_get_solar_data_success(self, mock_get, api_client):
        """Test successful solar data fetching"""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MOCK_SOLAR_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the method
        result = api_client.get_solar_data(-33.8688, 151.2093, "2024-01-01", "2024-01-02")

        # Verify the call was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "https://www.renewables.ninja/api/data/pv" in args[0]
        assert kwargs['params']['lat'] == -33.8688
        assert kwargs['params']['lon'] == 151.2093
        assert kwargs['params']['date_from'] == "2024-01-01"
        assert kwargs['params']['date_to'] == "2024-01-02"
        assert 'Authorization' in kwargs['headers'] if 'headers' in kwargs else api_client.session.headers['Authorization'] == "Token test_token_for_testing"

        # Verify the result
        assert 'data' in result
        assert 'metadata' in result
        assert result['data']['2024-01-01 12:00']['capacity_factor'] == 0.85

    @patch('requests.Session.get')
    def test_get_wind_data_success(self, mock_get, api_client):
        """Test successful wind data fetching"""
        mock_response = Mock()
        mock_response.json.return_value = MOCK_WIND_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = api_client.get_wind_data(-33.8688, 151.2093, "2024-01-01", "2024-01-02", 100)

        mock_get.assert_called_once()
        assert 'data' in result
        assert 'metadata' in result
        assert result['data']['2024-01-01 12:00']['capacity_factor'] == 0.65

    @patch('requests.Session.get')
    def test_api_request_failure(self, mock_get, api_client):
        """Test API request failure handling"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="API request failed"):
            api_client.get_solar_data(-33.8688, 151.2093)

    def test_get_hybrid_data_aggregation(self, api_client):
        """Test hybrid data aggregation logic"""
        with patch.object(api_client, 'get_solar_data', return_value=MOCK_SOLAR_RESPONSE):
            with patch.object(api_client, 'get_wind_data', return_value=MOCK_WIND_RESPONSE):
                result = api_client.get_hybrid_data(-33.8688, 151.2093,
                                                  solar_capacity=1.5, wind_capacity=2.0,
                                                  date_from="2024-01-01", date_to="2024-01-02")

                assert 'solar_data' in result
                assert 'wind_data' in result
                assert 'hybrid_energy' in result
                assert 'total_energy_generated' in result
                assert 'metadata' in result

                # Check calculated total energy: (0.85*1.5 + 0.65*2.0) + (0.75*1.5 + 0.72*2.0)
                expected_total = (0.85 * 1.5 + 0.65 * 2.0) + (0.75 * 1.5 + 0.72 * 2.0)
                assert abs(result['total_energy_generated'] - expected_total) < 0.001

    def test_usa_location_profiles(self, api_client):
        """Test API integration with USA location profiles"""
        # USA locations
        locations = [
            {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
            {"name": "Houston", "lat": 29.7604, "lon": -95.3698}
        ]

        for location in locations:
            with patch.object(api_client, 'get_solar_data', return_value=MOCK_SOLAR_RESPONSE):
                with patch.object(api_client, 'get_wind_data', return_value=MOCK_WIND_RESPONSE):
                    try:
                        # Test solar data fetching for Australian locations
                        solar_result = api_client.get_solar_data(
                            location["lat"], location["lon"],
                            "2024-01-01", "2024-01-02"
                        )
                        assert 'data' in solar_result
                        assert location["lat"] >= 24.0 and location["lat"] <= 49.0  # USA latitude range
                        assert location["lon"] >= -125.0 and location["lon"] <= -67.0  # USA longitude range

                        # Test wind data fetching for Australian locations
                        wind_result = api_client.get_wind_data(
                            location["lat"], location["lon"],
                            "2024-01-01", "2024-01-02"
                        )
                        assert 'data' in wind_result

                        # Test hybrid system for Australian locations
                        hybrid_result = api_client.get_hybrid_data(
                            location["lat"], location["lon"],
                            solar_capacity=2.0, wind_capacity=1.5
                        )

                        assert 'solar_data' in hybrid_result
                        assert 'wind_data' in hybrid_result
                        assert 'hybrid_energy' in hybrid_result
                        assert 'metadata' in hybrid_result

                        # Validate location data in metadata
                        assert hybrid_result['metadata']['lat'] == location["lat"]
                        assert hybrid_result['metadata']['lon'] == location["lon"]

                    except Exception as e:
                        pytest.fail(f"Failed to process {location['name']}: {e}")