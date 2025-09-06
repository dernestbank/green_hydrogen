import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataTransformers:
    """Data transformation utilities for converting between different formats"""

    def __init__(self):
        """Initialize data transformers"""
        self.ninja_energy_conversion_factor = 1000  # API returns kW, model expects MW
        self.model_time_resolution = 'H'  # Hourly by default
        self.efficiency_assumptions = {
            'solar_pv_to_dc': 0.90,      # Solar PV to DC efficiency
            'dc_to_ac_inverter': 0.95,   # DC to AC inverter efficiency
            'electrical_losses': 0.02,  # General electrical losses
            'electrolyser_part_load': 0.85  # Electrolyser part-load efficiency
        }

    def ninja_to_model_solar_format(self, ninja_response: Dict[str, Any],
                                   model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Renewables Ninja solar API format to hydrogen model format

        Args:
            ninja_response: Raw API response from Renewables Ninja
            model_config: Configuration parameters for the model

        Returns:
            Transformed data in model-expected format
        """
        transformed_data = {
            'data_type': 'solar_pv_generation',
            'location': {
                'latitude': model_config.get('latitude', 40.7128),
                'longitude': model_config.get('longitude', -74.0060),
                'capacity_mw': model_config.get('nominal_solar_farm_capacity', 10.0)
            },
            'time_series': {},
            'metadata': {}
        }

        try:
            # Extract time series data
            if 'data' in ninja_response:
                api_data = ninja_response['data']

                # Convert API data to expected format
                for timestamp, values in api_data.items():
                    # Handle different API response formats
                    if isinstance(values, dict):
                        # Get electricity generation in MW (convert from kW if needed)
                        electricity_kw = values.get('electricity', 0)
                        electricity_mw = electricity_kw / self.ninja_energy_conversion_factor

                        # Apply system losses from configuration
                        system_loss = model_config.get('solar_system_loss', 10.0) / 100
                        net_electricity = electricity_mw * (1 - system_loss)

                        # Apply efficiency factors
                        dc_output = net_electricity * self.efficiency_assumptions['dc_to_ac_inverter']

                        transformed_data['time_series'][timestamp] = {
                            'gross_generation_mw': electricity_mw,
                            'system_losses_pct': system_loss * 100,
                            'net_generation_mw': net_electricity,
                            'dc_output_mw': dc_output,
                            'irradiance_w_m2': values.get('irradiance', 0),
                            'capacity_factor': values.get('capacity_factor', 0)
                        }

            # Add metadata
            transformed_data['metadata'] = {
                'api_source': 'renewables_ninja',
                'solar_model': 'PVsyst',
                'tilt_degrees': model_config.get('solar_tilt', 35.0),
                'azimuth_degrees': model_config.get('solar_azimuth', 180.0),
                'tracking_type': model_config.get('solar_tracking', 'Fixed'),
                'system_losses_total_pct': model_config.get('solar_system_loss', 10.0),
                'dataset_used': model_config.get('solar_dataset', 'MERRA-2'),
                'capacity_factor_avg': self._calculate_average_capacity_factor(transformed_data),
                'total_generation_mwh': self._calculate_total_generation(transformed_data)
            }

        except Exception as e:
            logger.error(f"Error transforming Ninja solar data: {e}")
            raise ValueError(f"Failed to transform solar data: {e}")

        return transformed_data

    def ninja_to_model_wind_format(self, ninja_response: Dict[str, Any],
                                  model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Renewables Ninja wind API format to hydrogen model format

        Args:
            ninja_response: Raw API response from Renewables Ninja
            model_config: Configuration parameters for the model

        Returns:
            Transformed data in model-expected format
        """
        transformed_data = {
            'data_type': 'wind_power_generation',
            'location': {
                'latitude': model_config.get('latitude', 40.7128),
                'longitude': model_config.get('longitude', -74.0060),
                'capacity_mw': model_config.get('nominal_wind_farm_capacity', 10.0)
            },
            'time_series': {},
            'metadata': {}
        }

        try:
            # Extract time series data
            if 'data' in ninja_response:
                api_data = ninja_response['data']

                for timestamp, values in api_data.items():
                    if isinstance(values, dict):
                        # Get electricity generation in MW
                        electricity_kw = values.get('electricity', 0)
                        electricity_mw = electricity_kw / self.ninja_energy_conversion_factor

                        # Apply any system losses if specified
                        system_loss = model_config.get('wind_system_loss', 5.0) / 100  # Wind has lower losses typically
                        net_electricity = electricity_mw * (1 - system_loss)

                        # Apply efficiency factors specific to wind systems
                        ac_output = net_electricity * self.efficiency_assumptions['dc_to_ac_inverter']

                        transformed_data['time_series'][timestamp] = {
                            'wind_speed_m_s': values.get('wind_speed', 0),
                            'gross_generation_mw': electricity_mw,
                            'system_losses_pct': system_loss * 100,
                            'net_generation_mw': net_electricity,
                            'ac_output_mw': ac_output,
                            'capacity_factor': values.get('capacity_factor', 0),
                            'hub_height_m': model_config.get('wind_hub_height_m', 80.0)
                        }

            # Add metadata
            transformed_data['metadata'] = {
                'api_source': 'renewables_ninja',
                'turbine_model': model_config.get('wind_turbine_model', 'Vestas V90 2000'),
                'hub_height_m': model_config.get('wind_hub_height_m', 80.0),
                'rotor_diameter_m': model_config.get('wind_rotor_diameter_m', 90.0),
                'cut_in_speed_m_s': model_config.get('wind_cut_in_speed', 3.0),
                'rated_speed_m_s': model_config.get('wind_rated_speed', 12.0),
                'cut_out_speed_m_s': model_config.get('wind_cut_out_speed', 25.0),
                'dataset_used': model_config.get('wind_dataset', 'MERRA-2'),
                'capacity_factor_avg': self._calculate_average_capacity_factor(transformed_data),
                'total_generation_mwh': self._calculate_total_generation(transformed_data)
            }

        except Exception as e:
            logger.error(f"Error transforming Ninja wind data: {e}")
            raise ValueError(f"Failed to transform wind data: {e}")

        return transformed_data

    def create_hybrid_generation_profile(self, solar_data: Dict[str, Any],
                                       wind_data: Dict[str, Any],
                                       model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create hybrid generation profile by combining solar and wind data

        Args:
            solar_data: Transformed solar generation data
            wind_data: Transformed wind generation data
            model_config: Configuration parameters

        Returns:
            Combined hybrid generation profile
        """
        hybrid_data = {
            'data_type': 'hybrid_power_generation',
            'location': {
                'latitude': model_config.get('latitude', solar_data['location']['latitude']),
                'longitude': model_config.get('longitude', solar_data['location']['longitude']),
                'solar_capacity_mw': solar_data['location']['capacity_mw'],
                'wind_capacity_mw': wind_data['location']['capacity_mw'],
                'total_capacity_mw': (solar_data['location']['capacity_mw'] +
                                    wind_data['location']['capacity_mw'])
            },
            'time_series': {},
            'metadata': {}
        }

        # Combine time series data
        solar_ts = solar_data['time_series']
        wind_ts = wind_data['time_series']
        all_timestamps = set(solar_ts.keys()) | set(wind_ts.keys())

        for timestamp in all_timestamps:
            solar_values = solar_ts.get(timestamp, {})
            wind_values = wind_ts.get(timestamp, {})

            # Calculate hybrid values
            solar_generation = solar_values.get('ac_output_mw', 0)
            wind_generation = wind_values.get('ac_output_mw', 0)
            total_hybrid = solar_generation + wind_generation

            # Calculate weighted capacity factor
            total_capacity = hybrid_data['location']['total_capacity_mw']
            capacity_factor = total_hybrid / total_capacity if total_capacity > 0 else 0
            capacity_factor = min(capacity_factor, 1.0)  # Cap at 100%

            hybrid_data['time_series'][timestamp] = {
                'solar_generation_mw': solar_generation,
                'wind_generation_mw': wind_generation,
                'total_hybrid_mw': total_hybrid,
                'hybrid_capacity_factor': capacity_factor,
                'solar_capacity_factor': solar_values.get('capacity_factor', 0),
                'wind_capacity_factor': wind_values.get('capacity_factor', 0)
            }

        # Combine metadata
        hybrid_data['metadata'] = {
            'api_source': 'renewables_ninja_hybrid',
            'solar_capacity_mw': solar_data['location']['capacity_mw'],
            'wind_capacity_mw': wind_data['location']['capacity_mw'],
            'total_installed_capacity_mw': hybrid_data['location']['total_capacity_mw'],
            'capacity_mix_solar_pct': (solar_data['location']['capacity_mw'] /
                                     hybrid_data['location']['total_capacity_mw'] * 100),
            'capacity_mix_wind_pct': (wind_data['location']['capacity_mw'] /
                                    hybrid_data['location']['total_capacity_mw'] * 100),
            'average_hybrid_capacity_factor': self._calculate_average_capacity_factor(hybrid_data, 'hybrid_capacity_factor'),
            'total_generation_mwh': self._calculate_total_generation(hybrid_data, 'total_hybrid_mw')
        }

        return hybrid_data

    def standardize_electrolyser_load_profile(self, power_generation: Dict[str, Any],
                                             electrolyser_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize electrolyser load profile from power generation data

        Args:
            power_generation: Power generation time series data
            electrolyser_config: Electrolyser configuration parameters

        Returns:
            Standardized electrolyser load profile
        """
        electrolyser_profile = {
            'data_type': 'electrolyser_load_profile',
            'time_series': {},
            'metadata': {}
        }

        electrolyser_capacity = electrolyser_config.get('nominal_electrolyser_capacity', 10.0)
        min_load_pct = electrolyser_config.get('electrolyser_min_load', 10) / 100
        max_load_pct = electrolyser_config.get('electrolyser_max_load', 100) / 100

        # Support different data keys for different generation types
        power_keys = ['ac_output_mw', 'total_hybrid_mw', 'gross_generation_mw']
        power_source_key = None

        # Find which power key exists in the data
        sample_data = next(iter(power_generation['time_series'].values()), {})
        for key in power_keys:
            if key in sample_data:
                power_source_key = key
                break

        if not power_source_key:
            raise ValueError("Power generation data key not found")

        for timestamp, values in power_generation['time_series'].items():
            available_power = values.get(power_source_key, 0)

            # Calculate maximum electrolyser load possible
            max_possible_load = min(available_power, electrolyser_capacity)

            # Apply minimum and maximum load constraints
            if max_possible_load < electrolyser_capacity * min_load_pct:
                # Below minimum load - can't operate
                load_mw = 0
                load_pct = 0
            else:
                # Load within acceptable range
                load_mw = max_possible_load
                load_pct = (load_mw / electrolyser_capacity) * 100

                # Apply maximum load constraint
                if load_mw > electrolyser_capacity * max_load_pct:
                    load_mw = electrolyser_capacity * max_load_pct
                    load_pct = max_load_pct * 100

            # Apply efficiency penalties for part-load operation
            if load_pct > 0 and load_pct < 50:
                # Low load penalty
                efficiency_penalty = 0.8  # 20% efficiency reduction
                effective_load = load_mw * efficiency_penalty
            elif load_pct >= 50:
                effective_load = load_mw
            else:
                effective_load = 0

            electrolyser_profile['time_series'][timestamp] = {
                'available_power_mw': available_power,
                'electrolyser_load_mw': load_mw,
                'electrolyser_load_pct': load_pct,
                'effective_load_mw': effective_load,
                'operating_mode': 'producing' if load_mw > 0 else 'standby',
                'excess_power_mw': available_power - load_mw
            }

        electrolyser_profile['metadata'] = {
            'electrolyser_capacity_mw': electrolyser_capacity,
            'min_load_pct': min_load_pct * 100,
            'max_load_pct': max_load_pct * 100,
            'operating_hours': self._calculate_operating_hours(electrolyser_profile),
            'average_load_factor': self._calculate_average_load_factor(electrolyser_profile),
            'total_energy_consumed_mwh': self._calculate_total_energy_consumed(electrolyser_profile)
        }

        return electrolyser_profile

    def _calculate_average_capacity_factor(self, data: Dict[str, Any],
                                         capacity_factor_key: str = 'capacity_factor') -> float:
        """Calculate average capacity factor from time series data"""
        capacity_factors = []

        for values in data['time_series'].values():
            if isinstance(values, dict) and capacity_factor_key in values:
                cf = values[capacity_factor_key]
                if isinstance(cf, (int, float)) and cf >= 0:
                    capacity_factors.append(cf)

        return float(np.mean(capacity_factors)) if capacity_factors else 0.0

    def _calculate_total_generation(self, data: Dict[str, Any], power_key: str = 'net_generation_mw') -> float:
        """Calculate total generation in MWh from time series data"""
        total_generation = 0

        for values in data['time_series'].values():
            if isinstance(values, dict) and power_key in values:
                generation = values[power_key]
                if isinstance(generation, (int, float)):
                    total_generation += generation

        return total_generation

    def _calculate_operating_hours(self, electrolyser_profile: Dict[str, Any]) -> int:
        """Calculate total operating hours from electrolyser profile"""
        operating_hours = 0

        for values in electrolyser_profile['time_series'].values():
            if isinstance(values, dict) and values.get('operating_mode') == 'producing':
                operating_hours += 1

        return operating_hours

    def _calculate_average_load_factor(self, electrolyser_profile: Dict[str, Any]) -> float:
        """Calculate average load factor from electrolyser profile"""
        load_factors = []

        for values in electrolyser_profile['time_series'].values():
            if isinstance(values, dict):
                load_pct = values.get('electrolyser_load_pct', 0)
                if load_pct > 0:  # Only count operating periods
                    load_factors.append(load_pct / 100)

        return np.mean(load_factors) if load_factors else 0

    def _calculate_total_energy_consumed(self, electrolyser_profile: Dict[str, Any]) -> float:
        """Calculate total energy consumed by electrolyser in MWh"""
        total_energy = 0

        for values in electrolyser_profile['time_series'].values():
            if isinstance(values, dict):
                load_mw = values.get('effective_load_mw', 0)
                if load_mw > 0:
                    total_energy += load_mw

        return total_energy

    def convert_to_excel_format(self, transformed_data: Dict[str, Any],
                               excel_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert transformed data to Excel-compatible format

        Args:
            transformed_data: Transformed data in model format
            excel_config: Excel format configuration

        Returns:
            DataFrame in Excel-compatible format
        """
        excel_rows = []

        time_series = transformed_data.get('time_series', {})
        column_mapping = excel_config.get('column_mapping', {})

        for timestamp, values in time_series.items():
            row = {'timestamp': timestamp}

            for model_key, excel_header in column_mapping.items():
                if model_key in values:
                    row[excel_header] = values[model_key]
                else:
                    row[excel_header] = None

            excel_rows.append(row)

        df = pd.DataFrame(excel_rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        return df