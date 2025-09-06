import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing utility for hydrogen production modeling"""

    def __init__(self):
        self.units = {
            'energy': ['MWh', 'MW', 'W'],
            'power': ['MW', 'kW', 'W'],
            'hydrogen_volume': ['t/day', 'kg/day', 't/year', 'kg/year'],
            'hydrogen_energy': ['MWh/kg', 'MJ/kg'],
            'currency': ['USD', 'AUD', 'EUR'],
            'location': ['latitude', 'longitude']
        }

    def validate_input_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters from the user interface (1_Inputs.py)"""

        validated_inputs = {}
        errors = []

        # Validate location parameters
        validated_inputs.update(self._validate_location_parameters(inputs, errors))

        # Validate system sizing parameters
        validated_inputs.update(self._validate_sizing_parameters(inputs, errors))

        # Validate electrolyser parameters
        validated_inputs.update(self._validate_electrolyser_parameters(inputs, errors))

        # Validate power plant parameters
        validated_inputs.update(self._validate_power_plant_parameters(inputs, errors))

        # Validate financial parameters
        validated_inputs.update(self._validate_financial_parameters(inputs, errors))

        # Validate grid and battery parameters
        validated_inputs.update(self._validate_grid_battery_parameters(inputs, errors))

        # Validate API and operational parameters
        validated_inputs.update(self._validate_api_parameters(inputs, errors))

        if errors:
            raise ValueError("Input validation errors: " + "; ".join(errors))

        return validated_inputs

    def _validate_location_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate location-related parameters"""
        validated = {}

        # Site location (US states/cities)
        if 'site_location' in inputs:
            location = inputs['site_location']
            valid_locations = [
                "Custom", "US.CA", "US.TX", "US.NM", "US.AZ", "US.NV", "US.UT", "US.CO",
                "US.WY", "US.MT", "US.OR", "US.WA", "US.ID", "US.IA", "US.KS", "US.OK",
                "US.ND", "US.SD", "US.MN", "US.WI", "US.MI", "US.IN", "US.IL", "US.OH",
                "US.PA", "US.NY", "US.VT", "US.NH", "US.MA", "US.RI", "US.CT", "US.NJ",
                "US.DE", "US.MD", "US.VA", "US.WV", "US.KY", "US.TN", "US.VA", "US.AR",
                "US.LA", "US.MS", "US.AL", "US.GA", "US.FL", "US.SC", "US.NC", "US.HI", "US.AK"
            ]
            if location not in valid_locations:
                errors.append(f"Invalid site location: {location}. Must be one of: {valid_locations}")
            else:
                validated['site_location'] = location

        # Custom location coordinates (when site_location == "Custom")
        if inputs.get('site_location') == "Custom" or 'latitude' in inputs:
            # Latitude validation (-90 to 90 degrees)
            if 'latitude' in inputs:
                lat = inputs['latitude']
                if not isinstance(lat, (int, float)):
                    errors.append("Latitude must be a number")
                elif not (-90 <= lat <= 90):
                    errors.append("Latitude must be between -90 and 90 degrees")
                else:
                    validated['latitude'] = float(lat)

            # Longitude validation (-180 to 180 degrees)
            if 'longitude' in inputs:
                lon = inputs['longitude']
                if not isinstance(lon, (int, float)):
                    errors.append("Longitude must be a number")
                elif not (-180 <= lon <= 180):
                    errors.append("Longitude must be between -180 and 180 degrees")
                else:
                    validated['longitude'] = float(lon)

            # API key validation
            if 'api_key' in inputs:
                api_key = inputs['api_key']
                if not api_key or not isinstance(api_key, str):
                    errors.append("API key must be a non-empty string")
                elif len(api_key) < 10:
                    errors.append("API key seems too short (minimum 10 characters)")
                else:
                    validated['api_key'] = api_key

        return validated

    def _validate_sizing_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate system sizing parameters"""
        validated = {}

        # Power plant configuration
        if 'power_plant_configuration' in inputs:
            config = inputs['power_plant_configuration']
            valid_configs = [
                "C1. Standalone Solar PV Generator with Electrolyser",
                "C2. Standalone Solar PV Generator with Electrolyser and Battery",
                "C3. Grid Connected Solar PV Generator with Electrolyser",
                "C4. Grid Connected Solar PV Generator with Electrolyser with Surplus Retailed to Grid",
                "C5. Grid Connected Solar PV Generator with Electrolyser and Battery",
                "C6. Grid Connected Solar PV Generator with Electrolyser and Battery with Surplus Retailed to Grid",
                "C7. Solar PPA with Electrolyser",
                "C8. Solar PPA with Electrolyser and Battery",
                "C9. Standalone Wind Generator with Electrolyser",
                "C10. Standalone Wind Generator with Electrolyser and Battery",
                "C11. Grid Connected Wind Generator with Electrolyser",
                "C12. Grid Connected Wind Generator with Electrolyser with Surplus Retailed to Grid",
                "C13. Grid Connected Wind Generator with Electrolyser and Battery",
                "C14. Grid Connected Wind Generator with Electrolyser and Battery with Surplus Retail to Gird",
                "C15. Wind PPA with Electrolyser",
                "C16. Wind PPA with Electrolyser and Battery",
                "C17. Standalone Hybrid Generator with Electrolyser",
                "C18. Standalone Hybrid Generator with Electrolyser and Battery",
                "C19. Grid Connected Hybrid Generator with Electrolyser",
                "C20. Grid Connected Hybrid Generator with Electrolyser with Surplus Retailed to Grid",
                "C21. Grid Connected Hybrid Generator with Electrolyser and Battery",
                "C22. Grid Connected Hybrid Generator with Electrolyser and Battery with Surplus Retailed to Grid",
                "C23. Hybrid PPA with Electrolyser",
                "C24. Hybrid PPA with Electrolyser and Battery"
            ]
            if config not in valid_configs:
                errors.append(f"Invalid power plant configuration: {config}")
            else:
                validated['power_plant_configuration'] = config
                # Determine generator type from configuration
                if config.startswith(("C1.", "C2.", "C3.", "C4.", "C5.", "C6.", "C7.", "C8.")):
                    validated['generator_type'] = "Solar PV"
                elif config.startswith(("C9.", "C10.", "C11.", "C12.", "C13.", "C14.", "C15.", "C16.")):
                    validated['generator_type'] = "Wind"
                elif config.startswith(("C17.", "C18.", "C19.", "C20.", "C21.", "C22.", "C23.", "C24.")):
                    validated['generator_type'] = "Hybrid"

        # Electrolyser capacity
        if 'nominal_electrolyser_capacity' in inputs:
            capacity = inputs['nominal_electrolyser_capacity']
            if not isinstance(capacity, (int, float)):
                errors.append("Electrolyser capacity must be a number")
            elif capacity <= 0:
                errors.append("Electrolyser capacity must be positive")
            elif capacity > 1000:
                errors.append(f"Electrolyser capacity ({capacity}) is unusually high (>1000 MW)")
            else:
                validated['nominal_electrolyser_capacity'] = float(capacity)

        # Solar farm capacity (if applicable)
        if 'nominal_solar_farm_capacity' in inputs:
            capacity = inputs['nominal_solar_farm_capacity']
            if not isinstance(capacity, (int, float)):
                errors.append("Solar farm capacity must be a number")
            elif capacity < 0:
                errors.append("Solar farm capacity cannot be negative")
            elif capacity > 50000:
                errors.append(f"Solar farm capacity ({capacity}) is unusually high (>50,000 MW)")
            else:
                validated['nominal_solar_farm_capacity'] = float(capacity)

        # Wind farm capacity (if applicable)
        if 'nominal_wind_farm_capacity' in inputs:
            capacity = inputs['nominal_wind_farm_capacity']
            if not isinstance(capacity, (int, float)):
                errors.append("Wind farm capacity must be a number")
            elif capacity < 0:
                errors.append("Wind farm capacity cannot be negative")
            elif capacity > 50000:
                errors.append(f"Wind farm capacity ({capacity}) is unusually high (>50,000 MW)")
            else:
                validated['nominal_wind_farm_capacity'] = float(capacity)

        # Battery parameters (if applicable)
        battery_fields = {
            'battery_rated_power': (0.1, 1000, "Battery rated power"),
            'battery_indirect_costs_percent_of_capex': (0, 100, "Battery indirect costs"),
            'battery_replacement_cost_of_capex': (0, 200, "Battery replacement cost"),
            'battery_opex_a_mw_yr': (0, 100000, "Battery OPEX"),
            'round_trip_efficiency': (0.01, 1.0, "Battery round trip efficiency"),
            'minimum_state_of_charge': (0, 95, "Minimum state of charge"),
            'maximum_state_of_charge': (5, 100, "Maximum state of charge"),
            'duration_of_storage_hours': (1, 24, "Storage duration")
        }

        for field, (min_val, max_val, description) in battery_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        return validated

    def _validate_electrolyser_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate electrolyser design and performance parameters"""
        validated = {}

        # Electrolyser choice
        if 'electrolyser_choice' in inputs:
            choice = inputs['electrolyser_choice']
            if choice not in ["PEM", "ALK"]:
                errors.append(f"Invalid electrolyser choice: {choice}. Must be PEM or ALK")
            else:
                validated['electrolyser_choice'] = choice

        # Design parameters
        design_fields = {
            'active_cell_area_m2': (0.1, 1000, "Active cell area"),
            'current_density_amps_cm2': (0.1, 100, "Current density"),
            'anode_thickness_mm': (0.01, 10, "Anode thickness"),
            'cathode_thickness_mm': (0.01, 10, "Cathode thickness"),
            'membrane_thickness_mm': (0.01, 5, "Membrane thickness"),
            'operating_temperature_c': (0, 120, "Operating temperature"),
            'operating_pressure_bar': (0.1, 100, "Operating pressure"),
            'limiting_current_density_amps_cm2': (0.1, 200, "Limiting current density"),
            'exchange_current_density_amps_cm2': (0.001, 100, "Exchange current density")
        }

        for field, (min_val, max_val, description) in design_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        # Performance parameters
        performance_fields = {
            'sec_at_nominal_load': (20, 100, "SEC at nominal load"),
            'total_system_sec': (25, 120, "Total system SEC"),
            'electrolyser_min_load': (0, 100, "Minimum load percentage"),
            'electrolyser_max_load': (50, 200, "Maximum load percentage"),
            'max_overload_duration': (0, 100, "Maximum overload duration"),
            'time_between_overload': (0, 168, "Time between overload")  # hours
        }

        for field, (min_val, max_val, description) in performance_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val} (percentage for load)")
                else:
                    if "load" in field and field != "max_overload_duration":
                        validated[field] = float(value) / 100  # Convert percentage to decimal
                    else:
                        validated[field] = float(value)

        # Cost parameters
        cost_fields = {
            'reference_capacity': (1, 10000, "Reference capacity"),
            'reference_cost': (100, 5000, "Reference cost"),
            'scale_index': (0.5, 1.0, "End of Scale index"),
            'land_cost_percent': (0, 50, "Land cost percentage"),
            'installation_cost_percent': (0, 50, "Installation cost percentage"),
            'om_cost_percent': (0, 10, "O&M cost percentage"),
            'stack_replacement_percent': (10, 100, "Stack replacement percentage"),
            'water_cost': (0, 50, "Water cost")
        }

        for field, (min_val, max_val, description) in cost_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        return validated

    def _validate_power_plant_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate solar and wind power plant parameters"""
        validated = {}

        # Solar PV parameters
        solar_fields = {
            'solar_reference_capacity': (1, 10000, "Solar reference capacity"),
            'solar_reference_equipment_cost': (500, 3000, "Solar reference cost"),
            'solar_scale_index': (0.7, 1.0, "Solar scale index"),
            'solar_cost_reduction': (0, 50, "Solar cost reduction"),
            'solar_installation_costs': (0, 50, "Solar installation costs"),
            'solar_land_cost': (0, 50, "Solar land cost"),
            'solar_opex': (1000, 100000, "Solar OPEX"),
            'solar_degradation': (0, 5, "Solar degradation"),
            'solar_capacity_kw': (0.1, 1000000, "Solar capacity"),
            'solar_system_loss': (0, 50, "Solar system loss"),
            'solar_tilt': (0, 90, "Solar tilt"),
            'solar_azimuth': (-180, 180, "Solar azimuth")
        }

        for field, (min_val, max_val, description) in solar_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        # Wind power parameters
        wind_fields = {
            'wind_reference_capacity': (1, 10000, "Wind reference capacity"),
            'wind_reference_cost': (1000, 10000, "Wind reference cost"),
            'wind_scale_index': (0.7, 1.0, "Wind scale index"),
            'wind_cost_reduction': (0, 50, "Wind cost reduction"),
            'wind_installation_costs': (0, 50, "Wind installation costs"),
            'wind_land_cost': (0, 50, "Wind land cost"),
            'wind_opex': (5000, 200000, "Wind OPEX"),
            'wind_degradation': (0, 3, "Wind degradation"),
            'wind_capacity_kw': (0.1, 1000000, "Wind capacity"),
            'wind_hub_height_m': (10, 150, "Wind hub height")
        }

        for field, (min_val, max_val, description) in wind_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        # Dataset and turbine validations
        if 'solar_dataset' in inputs:
            if inputs['solar_dataset'] not in ["MERRA-2 (global)", "Other"]:
                errors.append("Invalid solar dataset choice")
            else:
                validated['solar_dataset'] = inputs['solar_dataset']

        if 'wind_dataset' in inputs:
            if inputs['wind_dataset'] not in ["MERRA-2 (global)", "Other"]:
                errors.append("Invalid wind dataset choice")
            else:
                validated['wind_dataset'] = inputs['wind_dataset']

        if 'wind_turbine_model' in inputs:
            if inputs['wind_turbine_model'] not in ["Vestas V90 2000", "Other"]:
                errors.append("Invalid wind turbine model")
            else:
                validated['wind_turbine_model'] = inputs['wind_turbine_model']

        if 'solar_tracking' in inputs:
            tracking_options = ["None", "Single-axis", "Dual-axis", "Other"]
            if inputs['solar_tracking'] not in tracking_options:
                errors.append("Invalid solar tracking option")
            else:
                validated['solar_tracking'] = inputs['solar_tracking']

        return validated

    def _validate_financial_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate financial and economic parameters"""
        validated = {}

        # Basic financial parameters
        financial_fields = {
            'plant_life_years': (1, 50, "Plant life"),
            'discount_rate': (0, 20, "Discount rate"),
            'inflation_rate': (0, 20, "Inflation rate"),
            'tax_rate': (0, 50, "Tax rate"),
            'financing_via_equity': (0, 100, "Equity financing"),
            'direct_equity_of_total_equity': (0, 100, "Direct equity"),
            'loan_term_years': (1, 30, "Loan term"),
            'interest_rate_on_loan_p_a': (0, 20, "Interest rate"),
            'salvage_costs_of_total_investments': (0, 50, "Salvage value"),
            'decommissioning_costs_of_total_investments': (0, 50, "Decommissioning costs"),
            'additional_upfront_costs_a': (0, 1000000000, "Additional upfront costs"),
            'additional_annual_costs_a_yr': (0, 100000000, "Additional annual costs"),
            'average_electricity_spot_price_a_mwh': (0, 500, "Electricity spot price"),
            'oxygen_retail_price_a_kg': (0, 5, "Oxygen retail price")
        }

        for field, (min_val, max_val, description) in financial_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        # Depreciation profile validation
        if 'depreciation_profile' in inputs:
            profile = inputs['depreciation_profile']
            valid_profiles = [
                "Straight Line",
                "Modified Accelerated Cost Recovery System (3 years)",
                "Modified Accelerated Cost Recovery System (5 years)",
                "Modified Accelerated Cost Recovery System (7 years)",
                "Modified Accelerated Cost Recovery System (10 years)",
                "Modified Accelerated Cost Recovery System (15 years)",
                "Modified Accelerated Cost Recovery System (20 years)"
            ]
            if profile not in valid_profiles:
                errors.append(f"Invalid depreciation profile: {profile}")
            else:
                validated['depreciation_profile'] = profile

        return validated

    def _validate_grid_battery_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate grid connection and battery system parameters"""
        validated = {}

        grid_fields = {
            'grid_connection_cost_percent': (0, 20, "Grid connection cost"),
            'grid_service_charge_percent': (0, 20, "Grid service charge"),
            'principal_ppa_cost_percent': (0, 50, "Principal PPA cost"),
            'transmission_connection_cost_percent': (0, 30, "Transmission connection cost")
        }

        for field, (min_val, max_val, description) in grid_fields.items():
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{description} must be a number")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{description} must be between {min_val} and {max_val}")
                else:
                    validated[field] = float(value)

        return validated

    def _validate_api_parameters(self, inputs: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """Validate API-related parameters"""
        validated = {}

        # Year selection validation
        for year_key in ['solar_year', 'wind_year']:
            if year_key in inputs:
                year = inputs[year_key]
                if not isinstance(year, int):
                    errors.append(f"{year_key} must be an integer")
                elif not (2000 <= year <= 2030):
                    errors.append(f"{year_key} must be between 2000 and 2030")
                else:
                    validated[year_key] = year

        # Boolean validation for include_raw_data fields
        boolean_fields = ['solar_include_raw_data', 'wind_include_raw_data']
        for field in boolean_fields:
            if field in inputs:
                value = inputs[field]
                if not isinstance(value, bool):
                    errors.append(f"{field} must be True or False")
                else:
                    validated[field] = value

        return validated

    def process_api_data(self, api_data: Dict[str, Any], data_type: str = 'electricity') -> pd.DataFrame:
        """Process API response data into a clean pandas DataFrame"""

        if 'data' not in api_data:
            raise ValueError("API response missing 'data' key")

        data = api_data['data']
        if not isinstance(data, dict):
            raise ValueError("API data must be a dictionary")

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')

        # Reset index to have timestamp as a column
        df = df.reset_index().rename(columns={'index': 'timestamp'})

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure timestamp is index
        df = df.set_index('timestamp')

        # Validate data quality
        quality_issues = self._validate_data_quality(df, data_type)

        if quality_issues:
            logger.warning(f"Data quality issues found: {quality_issues}")

        return df

    def _validate_data_quality(self, df: pd.DataFrame, data_type: str) -> List[str]:
        """Validate data quality and return list of issues"""

        issues = []

        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"{missing_count} missing values detected")

        # Check for outliers (using IQR method)
        if data_type == 'electricity':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    issues.append(f"{outliers} outliers detected in {col}")

        # Check for negative values where they shouldn't exist
        if data_type == 'electricity':
            negative_count = (df.select_dtypes(include=[np.number]) < 0).sum().sum()
            if negative_count > 0:
                issues.append(f"{negative_count} negative values detected in electricity data")

        # Check datetime continuity
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                inferred_freq = pd.infer_freq(df.index[:10])  # Check first 10 timestamps
                if inferred_freq is None:
                    consecutive_timestamps = df.index.to_series().diff().dropna()
                    if len(consecutive_timestamps) > 0:
                        mode_diff = consecutive_timestamps.mode()
                        if len(mode_diff) > 0:
                            expected_gap = mode_diff.iloc[0]
                            irregular_count = (
                                ((consecutive_timestamps != expected_gap) &
                                 (consecutive_timestamps != expected_gap))
                            ).sum()
                            if irregular_count > 0:
                                issues.append(f"{irregular_count} irregular timestamp gaps detected")
            except (TypeError, ValueError):
                pass  # Skip frequency inference if it fails

        return issues

    def interpolate_missing_data(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing values in time series data"""

        if method not in ['linear', 'nearest']:
            raise ValueError("Invalid interpolation method. Use: linear or nearest")

        if df.isnull().sum().sum() == 0:
            return df  # No missing values to interpolate

        # Use pandas interpolate method
        interpolated_df = df.copy()
        try:
            if method == 'linear':
                interpolated_df = df.interpolate(method='linear')
            else:
                interpolated_df = df.interpolate(method='nearest')
        except:
            # Fallback to simple interpolation
            interpolated_df = df.interpolate()

        # For any remaining missing values (at edges), use forward/backward fill
        interpolated_df = interpolated_df.ffill().bfill()

        return interpolated_df

    def aggregate_data(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """Aggregate data to specified frequency (D=daily, M=monthly, Y=yearly)"""

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex for aggregation")

        # Aggregate numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        aggregated = df[numeric_cols].resample(freq).sum()

        return aggregated

    def convert_units(self, value: float, from_unit: str, to_unit: str, conversion_factor: Optional[float] = None) -> float:
        """Convert between different units"""

        if from_unit == to_unit:
            return value  # No conversion needed

        # Energy units (MWh, GWh, TWh, kWh, W)
        energy_conversions = {
            ('MWh', 'kWh'): 1000,
            ('kWh', 'MWh'): 0.001,
            ('MWh', 'GWh'): 0.001,
            ('GWh', 'MWh'): 1000,
            ('MWh', 'TWh'): 0.000001,
            ('TWh', 'MWh'): 1000000,
            ('kWh', 'Wh'): 1000,
            ('Wh', 'kWh'): 0.001,
            ('GWh', 'TWh'): 0.001,
            ('TWh', 'GWh'): 1000
        }

        # Power units (MW, GW, TW, kW, W)
        power_conversions = {
            ('MW', 'kW'): 1000,
            ('kW', 'MW'): 0.001,
            ('MW', 'GW'): 0.001,
            ('GW', 'MW'): 1000,
            ('MW', 'TW'): 0.000001,
            ('TW', 'MW'): 1000000,
            ('kW', 'W'): 1000,
            ('W', 'kW'): 0.001,
            ('GW', 'TW'): 0.001,
            ('TW', 'GW'): 1000
        }

        # Hydrogen production units (t/day, kg/day, t/year, kg/year)
        hydrogen_conversions = {
            ('t/day', 'kg/day'): 1000,
            ('kg/day', 't/day'): 0.001,
            ('t/day', 't/year'): 365.25,
            ('t/year', 't/day'): 1/365.25,
            ('kg/day', 'kg/year'): 365.25,
            ('kg/year', 'kg/day'): 1/365.25,
            ('t/year', 'kg/year'): 1000,
            ('kg/year', 't/year'): 0.001
        }

        # Hydrogen energy content conversions
        hydrogen_energy_conversions = {
            ('MWh/kg', 'kWh/kg'): 1000,
            ('kWh/kg', 'MWh/kg'): 0.001,
            ('MJ/kg', 'kWh/kg'): 1/3.6,  # 1 MJ = 3.6 kWh
            ('kWh/kg', 'MJ/kg'): 3.6,
            ('MWh/kg', 'MJ/kg'): 3600,  # 1 MWh = 3600 MJ
            ('MJ/kg', 'MWh/kg'): 1/3600,
            ('Btu/lb', 'MWh/kg'): 0.002928,  # Approximate conversion
            ('MWh/kg', 'Btu/lb'): 341.214
        }

        # Currency conversions (approximate, latest rates)
        currency_conversions = {
            ('USD', 'EUR'): 0.88,    # 1 USD = 0.88 EUR (approximate)
            ('EUR', 'USD'): 1.14,    # 1 EUR = 1.14 USD
            ('USD', 'AUD'): 1.53,    # 1 USD = 1.53 AUD
            ('AUD', 'USD'): 0.65,    # 1 AUD = 0.65 USD
            ('USD', 'USD'): 1.0,     # 1 USD = 1 USD
            ('EUR', 'EUR'): 1.0,
            ('AUD', 'AUD'): 1.0
        }

        # Temperature conversions (Celsius, Fahrenheit, Kelvin)
        temperature_conversions = {
            ('C', 'F'): lambda x: (x * 9/5) + 32,
            ('F', 'C'): lambda x: (x - 32) * 5/9,
            ('C', 'K'): lambda x: x + 273.15,
            ('K', 'C'): lambda x: x - 273.15,
            ('F', 'K'): lambda x: (x - 32) * 5/9 + 273.15,
            ('K', 'F'): lambda x: (x - 273.15) * 9/5 + 32,
            ('C', 'C'): lambda x: x,
            ('F', 'F'): lambda x: x,
            ('K', 'K'): lambda x: x
        }

        # Time units conversions
        time_conversions = {
            ('year', 'day'): 365.25,
            ('day', 'year'): 1/365.25,
            ('year', 'hour'): 8766,  # 365.25 * 24
            ('hour', 'year'): 1/8766,
            ('day', 'hour'): 24,
            ('hour', 'day'): 1/24,
            ('hour', 'minute'): 60,
            ('minute', 'hour'): 1/60,
            ('day', 'minute'): 1440,
            ('minute', 'day'): 1/1440
        }

        # Length/distance units
        length_conversions = {
            ('km', 'm'): 1000,
            ('m', 'km'): 0.001,
            ('km', 'mile'): 0.621371,
            ('mile', 'km'): 1.60934,
            ('m', 'ft'): 3.28084,
            ('ft', 'm'): 0.3048,
            ('km', 'ft'): 3280.84,
            ('ft', 'km'): 0.0003048
        }

        # Area units
        area_conversions = {
            ('km²', 'm²'): 1000000,
            ('m²', 'km²'): 1/1000000,
            ('km²', 'acre'): 247.105,
            ('acre', 'km²'): 1/247.105,
            ('m²', 'ft²'): 10.7639,
            ('ft²', 'm²'): 1/10.7639,
            ('acre', 'ft²'): 43560,
            ('ft²', 'acre'): 1/43560
        }

        # All conversion dictionaries
        all_conversions = {
            **energy_conversions,
            **power_conversions,
            **hydrogen_conversions,
            **hydrogen_energy_conversions,
            **currency_conversions,
            **time_conversions,
            **length_conversions,
            **area_conversions
        }

        if conversion_factor is None:
            key = (from_unit, to_unit)

            # Check for special temperature conversions first
            if from_unit in ['C', 'F', 'K'] or to_unit in ['C', 'F', 'K']:
                if key in temperature_conversions:
                    conversion_func = temperature_conversions[key]
                    return conversion_func(value)
                else:
                    raise ValueError(f"Temperature conversion not supported: {from_unit} to {to_unit}")

            # Regular linear conversions
            if key in all_conversions:
                conversion_factor = all_conversions[key]
            else:
                raise ValueError(f"No conversion factor available for {from_unit} to {to_unit}. Available conversions include energy (MWh, GWh, kWh), power (MW, GW, kW), hydrogen (kg, t), currency (USD, EUR, AUD), temperature (C, F, K), time, and length units.")

        if conversion_factor is None:
            raise ValueError("Conversion factor cannot be None")

        # Handle both numeric and callable conversion factors (for temperature)
        if callable(conversion_factor):
            return conversion_factor(value)

        return value * conversion_factor

    def calculate_capacity_factor(self, actual_production: pd.Series, rated_capacity: float) -> pd.Series:
        """Calculate capacity factor from actual production and rated capacity"""

        if not isinstance(actual_production.index, pd.DatetimeIndex):
            raise ValueError("Production data must have DatetimeIndex")

        # Calculate hours in each period
        time_diff = actual_production.index.to_series().diff().fillna(pd.Timedelta(hours=1))
        hours = time_diff.dt.total_seconds() / 3600

        # Maximum possible production
        max_production = rated_capacity * hours

        # Capacity factor
        capacity_factor = actual_production / max_production

        # Clamp to [0, 1] range
        capacity_factor = capacity_factor.clip(0, 1)

        return capacity_factor

    def fill_missing_values_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple forward/backward fill for missing values"""
        return df.ffill().bfill()

    def detect_anomalies(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
        """Detect anomalies using modified z-score"""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Calculate mean and std
        mean_val = df[column].mean()
        std_val = df[column].std()

        if std_val == 0:
            return pd.Series(False, index=df.index)

        # Simple outlier detection (values outside +/- threshold * std)
        return (df[column] - mean_val).abs() > (threshold * std_val)