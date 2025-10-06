"""
Location Manager for handling predefined and custom locations with API integration
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import logging
from renewables_ninja_api import RenewablesNinjaAPI

logger = logging.getLogger(__name__)

class LocationManager:
    """
    Manages location data and API integration for renewable energy data.

    Handles both predefined locations (US states) and custom locations with
    optional API integration for real-time data fetching.
    """

    # Predefined US state locations with coordinates and resource factors
    PREDEFINED_LOCATIONS = {
        'US.CA': {
            'name': 'California',
            'lat': 36.7783,
            'lon': -119.4179,
            'solar_factor': 1.2,
            'wind_factor': 0.8,
            'description': 'High solar, moderate wind'
        },
        'US.TX': {
            'name': 'Texas',
            'lat': 31.9686,
            'lon': -99.9018,
            'solar_factor': 1.1,
            'wind_factor': 1.3,
            'description': 'Good solar, excellent wind'
        },
        'US.NM': {
            'name': 'New Mexico',
            'lat': 34.9727,
            'lon': -105.0324,
            'solar_factor': 1.4,
            'wind_factor': 1.1,
            'description': 'Excellent solar, good wind'
        },
        'US.MT': {
            'name': 'Montana',
            'lat': 47.0527,
            'lon': -110.6331,
            'solar_factor': 0.9,
            'wind_factor': 1.4,
            'description': 'Moderate solar, excellent wind'
        },
        'US.ND': {
            'name': 'North Dakota',
            'lat': 47.6506,
            'lon': -100.4370,
            'solar_factor': 0.8,
            'wind_factor': 1.5,
            'description': 'Lower solar, exceptional wind'
        },
        'US.NE': {
            'name': 'Nebraska',
            'lat': 41.4925,
            'lon': -99.9018,
            'solar_factor': 1.0,
            'wind_factor': 1.2,
            'description': 'Average solar, good wind'
        },
        'US.NY': {
            'name': 'New York',
            'lat': 42.1657,
            'lon': -74.9481,
            'solar_factor': 0.7,
            'wind_factor': 0.9,
            'description': 'Lower solar, moderate wind'
        },
        'US.AZ': {
            'name': 'Arizona',
            'lat': 33.7298,
            'lon': -111.4312,
            'solar_factor': 1.5,
            'wind_factor': 0.7,
            'description': 'Exceptional solar, lower wind'
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LocationManager.

        Args:
            api_key: Optional API key for renewables.ninja. If provided, enables API integration.
        """
        self.api_key = api_key
        self.api_client = None

        # Initialize API client if key provided
        if self.api_key:
            try:
                self.api_client = RenewablesNinjaAPI(self.api_key)
                logger.info("Renewables Ninja API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize API client: {e}")
                self.api_client = None

        # Load reference data
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference solar and wind data from CSV files."""
        try:
            solar_file = Path('Data/solar-traces.csv')
            wind_file = Path('Data/wind-traces.csv')

            if solar_file.exists():
                self.solar_df = pd.read_csv(solar_file, header=[0], skiprows=[1], index_col=0)
                logger.info(f"Loaded solar data: {self.solar_df.shape}")
            else:
                logger.warning(f"Solar data file not found: {solar_file}")
                self.solar_df = pd.DataFrame()

            if wind_file.exists():
                self.wind_df = pd.read_csv(wind_file, header=[0], skiprows=[1], index_col=0)
                logger.info(f"Loaded wind data: {self.wind_df.shape}")
            else:
                logger.warning(f"Wind data file not found: {wind_file}")
                self.wind_df = pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            self.solar_df = pd.DataFrame()
            self.wind_df = pd.DataFrame()

    def is_predefined_location(self, location: str) -> bool:
        """
        Check if location is predefined.

        Args:
            location: Location identifier

        Returns:
            True if location is predefined, False otherwise
        """
        return location in self.PREDEFINED_LOCATIONS

    def is_custom_location(self, location: str) -> bool:
        """
        Check if location is custom (not predefined).

        Args:
            location: Location identifier

        Returns:
            True if location is custom, False otherwise
        """
        return not self.is_predefined_location(location)

    def get_location_info(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Get location information.

        Args:
            location: Location identifier

        Returns:
            Location info dict or None if not found
        """
        return self.PREDEFINED_LOCATIONS.get(location)

    def get_solar_data(self, location: str, solar_capacity: float = 0) -> Optional[pd.Series]:
        """
        Get solar data for location.

        Args:
            location: Location identifier
            solar_capacity: Solar capacity in MW (for API scaling)

        Returns:
            Solar capacity factor series or None if not available
        """
        if self.is_predefined_location(location):
            # Use CSV data for predefined locations
            if location in self.solar_df.columns:
                return self.solar_df[location]
            else:
                logger.warning(f"No solar data for predefined location: {location}")
                return None

        elif self.is_custom_location(location) and self.api_client:
            # Try to get API data for custom locations
            try:
                # Parse location as "lat,lon" format
                if ',' in location:
                    lat, lon = map(float, location.split(','))
                    api_data = self.api_client.get_solar_data(lat, lon)

                    if 'data' in api_data and api_data['data']:
                        # Convert API response to capacity factor series
                        solar_series = pd.Series(api_data['data'])
                        # Scale by capacity if provided
                        if solar_capacity > 0:
                            solar_series = solar_series * solar_capacity
                        return solar_series
                    else:
                        logger.warning(f"No solar data in API response for location: {location}")
                        return None
                else:
                    logger.warning(f"Invalid custom location format: {location}. Expected 'lat,lon'")
                    return None

            except Exception as e:
                logger.error(f"Error fetching solar data from API: {e}")
                return None

        else:
            logger.warning(f"No solar data available for location: {location}")
            return None

    def get_wind_data(self, location: str, wind_capacity: float = 0) -> Optional[pd.Series]:
        """
        Get wind data for location.

        Args:
            location: Location identifier
            wind_capacity: Wind capacity in MW (for API scaling)

        Returns:
            Wind capacity factor series or None if not available
        """
        if self.is_predefined_location(location):
            # Use CSV data for predefined locations
            if location in self.wind_df.columns:
                return self.wind_df[location]
            else:
                logger.warning(f"No wind data for predefined location: {location}")
                return None

        elif self.is_custom_location(location) and self.api_client:
            # Try to get API data for custom locations
            try:
                # Parse location as "lat,lon" format
                if ',' in location:
                    lat, lon = map(float, location.split(','))
                    api_data = self.api_client.get_wind_data(lat, lon)

                    if 'data' in api_data and api_data['data']:
                        # Convert API response to capacity factor series
                        wind_series = pd.Series(api_data['data'])
                        # Scale by capacity if provided
                        if wind_capacity > 0:
                            wind_series = wind_series * wind_capacity
                        return wind_series
                    else:
                        logger.warning(f"No wind data in API response for location: {location}")
                        return None
                else:
                    logger.warning(f"Invalid custom location format: {location}. Expected 'lat,lon'")
                    return None

            except Exception as e:
                logger.error(f"Error fetching wind data from API: {e}")
                return None

        else:
            logger.warning(f"No wind data available for location: {location}")
            return None

    def get_hybrid_data(self, location: str, solar_capacity: float = 0, wind_capacity: float = 0) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Get both solar and wind data for location.

        Args:
            location: Location identifier
            solar_capacity: Solar capacity in MW
            wind_capacity: Wind capacity in MW

        Returns:
            Tuple of (solar_data, wind_data)
        """
        solar_data = self.get_solar_data(location, solar_capacity)
        wind_data = self.get_wind_data(location, wind_capacity)

        return solar_data, wind_data

    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Validate latitude and longitude coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            True if coordinates are valid
        """
        return -90 <= lat <= 90 and -180 <= lon <= 180

    def get_available_locations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available predefined locations.

        Returns:
            Dictionary of location info
        """
        return self.PREDEFINED_LOCATIONS.copy()

    def get_location_statistics(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for location data.

        Args:
            location: Location identifier

        Returns:
            Dictionary with statistics or None if not available
        """
        solar_data = self.get_solar_data(location)
        wind_data = self.get_wind_data(location)

        if solar_data is None and wind_data is None:
            return None

        stats = {'location': location}

        if solar_data is not None:
            stats['solar'] = {
                'mean': float(solar_data.mean()),
                'std': float(solar_data.std()),
                'min': float(solar_data.min()),
                'max': float(solar_data.max()),
                'non_zero_hours': int((solar_data > 0).sum())
            }

        if wind_data is not None:
            stats['wind'] = {
                'mean': float(wind_data.mean()),
                'std': float(wind_data.std()),
                'min': float(wind_data.min()),
                'max': float(wind_data.max()),
                'non_zero_hours': int((wind_data > 0).sum())
            }

        return stats
