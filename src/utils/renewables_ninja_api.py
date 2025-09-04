import requests
from time import sleep
import random
from typing import Dict, Any, Optional

class RenewablesNinjaAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.renewables.ninja/api/"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        })
        # Rate limiting configuration
        self.max_retries = 5
        self.base_delay = 1  # seconds
        self.max_delay = 60  # seconds

    def _make_api_request_with_retry(self, url: str, params: Dict[str, Any], max_retries: Optional[int] = None) -> Dict[str, Any]:
        """Make API request with rate limit handling and exponential backoff"""
        retries = max_retries or self.max_retries

        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                remaining_header = response.headers.get('X-RateLimit-Remaining')
                if remaining_header and int(remaining_header) <= 1:
                    # Close to rate limit, add small delay
                    sleep(random.uniform(0.5, 2.0))

                return response.json()

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                if status_code == 429:  # Too Many Requests
                    if attempt < retries:
                        delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
                        sleep(delay)
                        continue
                    else:
                        raise Exception(f"Rate limit exceeded after {retries + 1} attempts: {e}")
                elif status_code == 503:  # Service Unavailable
                    if attempt < retries:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        sleep(delay)
                        continue
                    else:
                        raise Exception(f"Service unavailable after {retries + 1} attempts: {e}")
                else:
                    status_msg = f"with status {status_code}" if status_code else "unknown status"
                    raise Exception(f"API request failed {status_msg}: {e}")

            except requests.exceptions.Timeout:
                if attempt < retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    sleep(delay)
                    continue
                else:
                    raise Exception("API request timed out after retries")

            except requests.exceptions.ConnectionError:
                if attempt < retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    sleep(delay)
                    continue
                else:
                    raise Exception("API connection error after retries")

        return {}  # Should not reach here

    def get_solar_data(self, lat: float, lon: float, date_from: Optional[str] = None, date_to: Optional[str] = None) -> Dict[str, Any]:
        """Get solar PV data from Renewables Ninja API"""
        url = f"{self.base_url}data/pv"
        params = {
            'lat': lat,
            'lon': lon,
            'dataset': 'merra2',
            'capacity': 1,
            'system_loss': 0.1,
            'tracking': 0,
            'tilt': 35,
            'azim': 180,
            'format': 'json'
        }
        if date_from:
            params['date_from'] = date_from
        if date_to:
            params['date_to'] = date_to

        return self._make_api_request_with_retry(url, params)

    def get_wind_data(self, lat: float, lon: float, date_from: Optional[str] = None, date_to: Optional[str] = None, height: int = 100) -> Dict[str, Any]:
        """Get wind data from Renewables Ninja API"""
        url = f"{self.base_url}data/wind"
        params = {
            'lat': lat,
            'lon': lon,
            'dataset': 'merra2',
            'capacity': 1,
            'height': height,
            'turbine': 'Vestas V80 2000',
            'format': 'json'
        }
        if date_from:
            params['date_from'] = date_from
        if date_to:
            params['date_to'] = date_to

        return self._make_api_request_with_retry(url, params)

    def get_hybrid_data(self, lat: float, lon: float, solar_capacity: float = 1, wind_capacity: float = 1,
                        date_from: Optional[str] = None, date_to: Optional[str] = None, wind_height: int = 100) -> Dict[str, Any]:
        """Get and aggregate solar and wind data for hybrid system analysis"""
        solar_data = self.get_solar_data(lat, lon, date_from, date_to)
        wind_data = self.get_wind_data(lat, lon, date_from, date_to, wind_height)

        # Simple aggregation logic - combine total energy production
        total_energy = {}
        if 'data' in solar_data and 'data' in wind_data:
            solar_production = solar_data['data']
            wind_production = wind_data['data']
            # Assume both have same timestamps and hourly values
            for timestamp in solar_production.keys():
                if timestamp in wind_production:
                    solar_val = float(solar_production[timestamp].get('electricity', 0) if solar_production[timestamp] else 0)
                    wind_val = float(wind_production[timestamp].get('electricity', 0) if wind_production[timestamp] else 0)
                    total_energy[timestamp] = solar_val * solar_capacity + wind_val * wind_capacity

        return {
            'solar_data': solar_data,
            'wind_data': wind_data,
            'hybrid_energy': total_energy,
            'total_energy_generated': sum(total_energy.values()) if total_energy else 0,
            'metadata': {
                'solar_capacity': solar_capacity,
                'wind_capacity': wind_capacity,
                'wind_height': wind_height,
                'lat': lat,
                'lon': lon
            }
        }
