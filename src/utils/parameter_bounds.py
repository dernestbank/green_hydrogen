from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

class ParameterBounds:
    """Class for defining and checking parameter bounds"""

    def __init__(self):
        """Initialize parameter bounds definitions"""
        self.bounds = self._initialize_bounds()
        self.default_values = self._initialize_defaults()

    def _initialize_bounds(self) -> Dict[str, Dict[str, Tuple[float, float, str]]]:
        """Initialize parameter bounds for different categories

        Returns:
            Dictionary with categories containing parameter bounds
            Format: {'category': {'parameter': (min, max, description)}}
        """
        return {
            'location': {
                'latitude': (-90.0, 90.0, 'Latitude (degrees)'),
                'longitude': (-180.0, 180.0, 'Longitude (degrees)'),
                'altitude': (0, 5000, 'Altitude (meters)')
            },

            'capacity': {
                'nominal_electrolyser_capacity': (0.1, 1000.0, 'Electrolyser capacity (MW)'),
                'nominal_solar_farm_capacity': (0.1, 50000.0, 'Solar farm capacity (MW)'),
                'nominal_wind_farm_capacity': (0.1, 50000.0, 'Wind farm capacity (MW)'),
                'battery_rated_power': (0.1, 1000.0, 'Battery rated power (MW)'),
                'duration_of_storage_hours': (1, 24, 'Battery storage duration (hours)'),
                'solar_capacity_kw': (1, 10000000, 'Solar capacity for API (kW)'),
                'wind_capacity_kw': (1, 10000000, 'Wind capacity for API (kW)')
            },

            'efficiency': {
                'sec_at_nominal_load': (20, 120, 'SEC at nominal load (kWh/kg)'),
                'total_system_sec': (25, 160, 'Total system SEC (kWh/kg)'),
                'electrolyser_efficiency': (0.1, 1.0, 'Electrolyser efficiency (fraction)'),
                'round_trip_efficiency': (0.1, 1.0, 'Battery round trip efficiency (fraction)'),
                'solar_pv_degradation_rate_percent_year': (0.0, 5.0, 'Solar degradation rate (%/year)'),
                'wind_farm_degradation_rate_percent_year': (0.0, 5.0, 'Wind degradation rate (%/year)')
            },

            'electrical_design': {
                'active_cell_area_m2': (1, 1000, 'Active cell area (cm²)'),
                'current_density_amps_cm2': (0.1, 100, 'Current density (A/cm²)'),
                'anode_thickness_mm': (0.01, 10, 'Anode thickness (mm)'),
                'cathode_thickness_mm': (0.01, 10, 'Cathode thickness (mm)'),
                'membrane_thickness_mm': (0.01, 5, 'Membrane thickness (mm)'),
                'operating_temperature_c': (-40, 120, 'Operating temperature (°C)'),
                'operating_pressure_bar': (0.01, 100, 'Operating pressure (bar)'),
                'limiting_current_density_amps_cm2': (0.1, 200, 'Limiting current density (A/cm²)'),
                'exchange_current_density_amps_cm2': (0.001, 100, 'Exchange current density (A/cm²)')
            },

            'operational_limits': {
                'electrolyser_min_load': (0, 100, 'Minimum load percentage (%)'),
                'electrolyser_max_load': (50, 200, 'Maximum load percentage (%)'),
                'max_overload_duration': (0, 168, 'Maximum overload duration (hours)'),
                'time_between_overload': (0, 8760, 'Time between overload (hours)'),
                'minimum_state_of_charge': (0, 95, 'Minimum state of charge (%)'),
                'maximum_state_of_charge': (5, 100, 'Maximum state of charge (%)')
            },

            'financial': {
                'plant_life_years': (1, 50, 'Plant life (years)'),
                'discount_rate': (0, 20, 'Discount rate (%)'),
                'inflation_rate': (0, 20, 'Inflation rate (%)'),
                'tax_rate': (0, 50, 'Tax rate (%)'),
                'financing_via_equity': (0, 100, 'Equity financing (%)'),
                'direct_equity_of_total_equity': (0, 100, 'Direct equity (%)'),
                'loan_term_years': (1, 30, 'Loan term (years)'),
                'interest_rate_on_loan_p_a': (0, 20, 'Interest rate (%)'),
                'salvage_costs_of_total_investments': (0, 50, 'Salvage value (%)'),
                'decommissioning_costs_of_total_investments': (0, 50, 'Decommissioning costs (%)'),
                'additional_upfront_costs_a': (0, 1e9, 'Additional upfront costs ($)'),
                'additional_annual_costs_a_yr': (0, 1e8, 'Additional annual costs ($/year)'),
                'average_electricity_spot_price_a_mwh': (0, 500, 'Electricity spot price ($/MWh)'),
                'oxygen_retail_price_a_kg': (0, 5, 'Oxygen retail price ($/kg)')
            },

            'costs_electrolyser': {
                'reference_capacity': (1, 10000, 'Reference capacity (kW)'),
                'reference_cost': (100, 5000, 'Reference cost ($/kW)'),
                'scale_index': (0.5, 1.0, 'Scale index'),
                'land_cost_percent': (0, 50, 'Land cost (%)'),
                'installation_cost_percent': (0, 50, 'Installation cost (%)'),
                'om_cost_percent': (0, 10, 'O&M cost (%)'),
                'stack_replacement_percent': (10, 100, 'Stack replacement (%)'),
                'water_cost': (0, 50, 'Water cost ($/kL)'),
                'electrolyser_economies_of_scale': (0, 100, 'Economies of scale (%)')
            },

            'costs_solar': {
                'solar_reference_capacity': (1, 10000, 'Solar reference capacity (kW)'),
                'solar_reference_equipment_cost': (500, 5000, 'Solar reference cost ($/kW)'),
                'solar_scale_index': (0.7, 1.0, 'Solar scale index'),
                'solar_cost_reduction': (0, 50, 'Solar cost reduction (%)'),
                'solar_installation_costs': (0, 50, 'Solar installation costs (%)'),
                'solar_land_cost': (0, 50, 'Solar land cost (%)'),
                'solar_opex': (1000, 500000, 'Solar O&M ($/MW/year)'),
                'solar_degradation': (0, 5, 'Solar degradation (%)'),
                'solar_monthly_degradation': (0, 0.5, 'Solar monthly degradation (%)'),
                'solar_yearly_degradation': (0, 5, 'Solar yearly degradation (%)')
            },

            'costs_wind': {
                'wind_reference_capacity': (1, 10000, 'Wind reference capacity (kW)'),
                'wind_reference_cost': (1000, 10000, 'Wind reference cost ($/kW)'),
                'wind_scale_index': (0.7, 1.0, 'Wind scale index'),
                'wind_cost_reduction': (0, 50, 'Wind cost reduction (%)'),
                'wind_installation_costs': (0, 50, 'Wind installation costs (%)'),
                'wind_land_cost': (0, 50, 'Wind land cost (%)'),
                'wind_opex': (5000, 500000, 'Wind O&M ($/MW/year)'),
                'wind_degradation': (0, 5, 'Wind degradation (%)'),
                'wind_monthly_degradation': (0, 0.5, 'Wind monthly degradation (%)'),
                'wind_yearly_degradation': (0, 5, 'Wind yearly degradation (%)')
            },

            'costs_battery': {
                'battery_capex_a_kwh': (50, 1000, 'Battery CAPEX ($/kWh)'),
                'battery_indirect_costs_percent_of_capex': (0, 50, 'Battery indirect costs (%)'),
                'battery_replacement_cost_of_capex': (10, 150, 'Battery replacement cost (%)'),
                'battery_opex_a_mw_yr': (1, 1000, 'Battery O&M ($/MW/year)')
            },

            'grid_connection': {
                'grid_connection_cost_percent': (0, 30, 'Grid connection cost (%)'),
                'grid_service_charge_percent': (0, 20, 'Grid service charge (%)'),
                'principal_ppa_cost_percent': (0, 50, 'Principal PPA cost (%)'),
                'transmission_connection_cost_percent': (0, 30, 'Transmission connection cost (%)')
            },

            'api_configuration': {
                'solar_year': (2000, 2030, 'Solar data year'),
                'wind_year': (2000, 2030, 'Wind data year'),
                'wind_hub_height_m': (10, 150, 'Wind hub height (m)'),
                'solar_tilt': (0, 90, 'Solar tilt (°)'),
                'solar_azimuth': (-180, 180, 'Solar azimuth (°)'),
                'solar_system_loss': (0, 50, 'Solar system loss (%)')
            },

            'custom_curves': {
                'electrolyser_custom_c1': (-1000, 10000, 'Custom curve coefficient C1'),
                'electrolyser_custom_c2': (-100, 100, 'Custom curve coefficient C2'),
                'electrolyser_custom_c3': (-1000, 10000, 'Custom curve coefficient C3'),
                'electrolyser_custom_c4': (-10, 10, 'Custom curve coefficient C4')
            }
        }

    def _initialize_defaults(self) -> Dict[str, Any]:
        """Initialize default values for parameters"""
        return {
            # Location defaults
            'latitude': 40.7128,    # New York City
            'longitude': -74.0060,
            'altitude': 0,

            # Capacity defaults
            'nominal_electrolyser_capacity': 10.0,  # MW
            'nominal_solar_farm_capacity': 10.0,     # MW
            'nominal_wind_farm_capacity': 10.0,      # MW
            'battery_rated_power': 5.0,              # MW
            'duration_of_storage_hours': 4,          # hours

            # API capacity defaults
            'solar_capacity_kw': 1000.0,     # kW
            'wind_capacity_kw': 1000.0,      # kW

            # Efficiency defaults
            'sec_at_nominal_load': 55.0,     # kWh/kg
            'total_system_sec': 58.0,        # kWh/kg
            'electrolyser_efficiency': 0.75, # fraction
            'round_trip_efficiency': 0.90,   # fraction

            # Electrical design defaults
            'active_cell_area_m2': 10.0,     # cm²
            'current_density_amps_cm2': 10.0, # A/cm²
            'anode_thickness_mm': 0.1,       # mm
            'cathode_thickness_mm': 0.1,     # mm
            'membrane_thickness_mm': 0.1,    # mm
            'operating_temperature_c': 25.0, # °C
            'operating_pressure_bar': 1.0,   # bar

            # Current density defaults
            'limiting_current_density_amps_cm2': 10.0,  # A/cm²
            'exchange_current_density_amps_cm2': 10.0,   # A/cm²

            # Operational limits defaults
            'electrolyser_min_load': 10,     # %
            'electrolyser_max_load': 100,    # %
            'max_overload_duration': 0,      # hours
            'time_between_overload': 24,     # hours
            'minimum_state_of_charge': 10,   # %
            'maximum_state_of_charge': 90,   # %

            # Financial defaults
            'plant_life_years': 25,          # years
            'discount_rate': 5.0,            # %
            'inflation_rate': 2.0,           # %
            'tax_rate': 30.0,                # %
            'financing_via_equity': 30.0,    # %
            'direct_equity_of_total_equity': 100.0,  # %
            'loan_term_years': 10,           # years
            'interest_rate_on_loan_p_a': 5.0,  # %

            # Cost defaults
            'reference_capacity': 1000.0,    # kW
            'reference_cost': 1500.0,        # $/kW
            'scale_index': 0.9,              # fraction
            'land_cost_percent': 6.0,        # %
            'installation_cost_percent': 0.0,  # %
            'om_cost_percent': 2.5,          # %

            # Solar cost defaults
            'solar_reference_capacity': 1000.0,       # kW
            'solar_reference_equipment_cost': 1500.0, # $/kW
            'solar_scale_index': 0.9,                 # fraction
            'solar_installation_costs': 0.0,          # %
            'solar_land_cost': 8.0,                   # %
            'solar_opex': 17000.0,                    # $/MW/year
            'solar_degradation': 0.5,                 # %

            # Wind cost defaults
            'wind_reference_capacity': 1000.0,   # kW
            'wind_reference_cost': 3000.0,       # $/kW
            'wind_scale_index': 0.9,             # fraction
            'wind_installation_costs': 0.0,      # %
            'wind_land_cost': 8.0,               # %
            'wind_opex': 25000.0,                # $/MW/year
            'wind_degradation': 1.5,             # %

            # Battery cost defaults
            'battery_capex_a_kwh': 300.0,        # $/kWh
            'battery_indirect_costs_percent_of_capex': 10.0,  # %
            'battery_replacement_cost_of_capex': 50.0,        # %
            'battery_opex_a_mw_yr': 10.0,        # $/MW/year

            # Grid defaults
            'grid_connection_cost_percent': 5.0,   # %
            'grid_service_charge_percent': 2.0,    # %
            'principal_ppa_cost_percent': 0.0,     # %
            'transmission_connection_cost_percent': 5.0,  # %

            # API defaults
            'solar_year': 2023,
            'wind_year': 2023,
            'wind_hub_height_m': 80.0,     # m
            'solar_tilt': 35.0,            # degrees
            'solar_azimuth': 180.0,        # degrees
            'solar_system_loss': 10.0,     # %

            # Additional costs defaults
            'salvage_costs_of_total_investments': 5.0,     # %
            'decommissioning_costs_of_total_investments': 5.0,  # %
            'additional_upfront_costs_a': 0.0,           # $
            'additional_annual_costs_a_yr': 0.0,        # $/year
            'average_electricity_spot_price_a_mwh': 0.0,   # $/MWh
            'oxygen_retail_price_a_kg': 0.0,             # $/kg

            # Degradation defaults
            'solar_pv_degradation_rate_percent_year': 0.5,  # %
            'wind_farm_degradation_rate_percent_year': 1.5, # %
            'solar_monthly_degradation': 0.0417,        # %/month
            'wind_monthly_degradation': 0.125,         # %/month
            'solar_yearly_degradation': 0.5,            # %/year
            'wind_yearly_degradation': 1.5,             # %/year

            # Stack replacement
            'stack_replacement_percent': 40.0,   # %
            'water_cost': 5.0,                   # $/kL
            'electrolyser_economies_of_scale': 0.0,  # %

            # Custom curve defaults (example values)
            'electrolyser_custom_c1': 1046.93,
            'electrolyser_custom_c2': -3.479,
            'electrolyser_custom_c3': 61.567,
            'electrolyser_custom_c4': -0.261
        }

    def check_parameter_bounds(self, parameter: str, value: Any) -> Tuple[bool, str]:
        """
        Check if a parameter value is within defined bounds

        Args:
            parameter: Parameter name
            value: Parameter value to check

        Returns:
            Tuple of (is_valid, message)
        """
        # Handle None values
        if value is None:
            return True, "Parameter value is None (will use default)"

        # Find parameter in bounds
        for category, params in self.bounds.items():
            if parameter in params:
                min_val, max_val, description = params[parameter]

                # Try to convert to numeric if it's not already
                try:
                    if isinstance(value, str):
                        # Try to convert string numbers
                        if '.' in value or 'e' in value.lower():
                            numeric_value = float(value)
                        else:
                            numeric_value = int(value)
                    else:
                        numeric_value = value

                    if not isinstance(numeric_value, (int, float, np.number)):
                        return False, f"Parameter {parameter} must be numeric, got {type(value)}"

                    if not (min_val <= numeric_value <= max_val):
                        return False, f"Parameter {parameter} ({numeric_value}) is outside valid range [{min_val}, {max_val}] for {description}"

                    return True, f"Valid: {description} = {numeric_value}"

                except (ValueError, TypeError):
                    return False, f"Parameter {parameter} cannot be converted to number: {value}"

        # Parameter not found in bounds (might be valid but not constrained)
        return True, f"Parameter {parameter} not found in bounds definition (will accept value)"

    def get_parameter_bounds(self, parameter: str) -> Optional[Tuple[float, float, str]]:
        """
        Get bounds for a specific parameter

        Args:
            parameter: Parameter name

        Returns:
            Tuple of (min, max, description) or None if parameter not found
        """
        for category, params in self.bounds.items():
            if parameter in params:
                return params[parameter]
        return None

    def validate_parameters_batch(self, parameters: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate multiple parameters at once

        Args:
            parameters: Dictionary of parameter names and values

        Returns:
            Dictionary with 'valid' and 'invalid' lists containing validation results
        """
        results = {'valid': [], 'invalid': []}

        for param_name, value in parameters.items():
            is_valid, message = self.check_parameter_bounds(param_name, value)
            result = {
                'parameter': param_name,
                'value': value,
                'valid': is_valid,
                'message': message
            }

            if is_valid:
                results['valid'].append(result)
            else:
                results['invalid'].append(result)

        return results

    def apply_defaults_and_bounds(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values and bound checking to parameters

        Args:
            parameters: Dictionary of parameter names and values

        Returns:
            Dictionary with defaults applied and bounds checked
        """
        processed_params = {}

        # First, apply defaults for missing parameters
        for param_name, default_value in self.default_values.items():
            processed_params[param_name] = parameters.get(param_name, default_value)

        # Then validate bounds for all parameters
        for param_name, value in processed_params.items():
            bounds = self.get_parameter_bounds(param_name)
            if bounds:
                min_val, max_val, description = bounds
                if isinstance(value, (int, float, np.number)):
                    # Clamp to bounds
                    processed_params[param_name] = np.clip(value, min_val, max_val)

        return processed_params

    def get_parameter_suggestions(self, parameter: str, current_value: Any) -> Dict[str, Any]:
        """
        Get suggestions for parameter optimization if value is outside preferred ranges

        Args:
            parameter: Parameter name
            current_value: Current parameter value

        Returns:
            Dictionary containing suggestions and alternative values
        """
        bounds = self.get_parameter_bounds(parameter)
        if not bounds:
            return {'suggestions': [], 'reason': 'Parameter not found in bounds definition'}

        min_val, max_val, description = bounds

        suggestions = {
            'parameter': parameter,
            'current': current_value,
            'bounds': {'min': min_val, 'max': max_val, 'description': description},
            'suggestions': []
        }

        # Generate suggestions based on parameter type
        if parameter.endswith('_capacity') or 'power' in parameter:
            # Capacity suggestions
            typical_values = self._get_typical_capacity_values(parameter)
            suggestions['suggestions'] = typical_values

        elif 'efficiency' in parameter or 'degradation' in parameter:
            # Efficiency suggestions
            optimum_value = self._get_optimum_efficiency_value(parameter)
            if optimum_value:
                suggestions['suggestions'] = [optimum_value]

        elif 'cost' in parameter and '%' in parameter:
            # Cost percentage suggestions
            typical_percentages = self._get_typical_cost_percentages(parameter)
            suggestions['suggestions'] = typical_percentages

        return suggestions

    def _get_typical_capacity_values(self, parameter: str) -> List[float]:
        """Get typical capacity values for different parameter types"""
        if 'solar' in parameter:
            return [1, 5, 10, 50, 100, 500, 1000]  # MW
        elif 'wind' in parameter:
            return [2, 5, 10, 50, 100, 500, 1000]  # MW
        elif 'electrolyser' in parameter:
            return [0.5, 1, 2, 5, 10, 25, 50, 100]  # MW
        elif 'battery' in parameter:
            return [1, 2, 5, 10, 25, 50, 100]  # MW
        else:
            return [1, 5, 10, 50, 100]  # Default MW values

    def _get_optimum_efficiency_value(self, parameter: str) -> Optional[float]:
        """Get optimum efficiency/degradation values"""
        if 'electrolyser_efficiency' in parameter:
            return 0.75
        elif 'round_trip_efficiency' in parameter:
            return 0.90
        elif 'degradation' in parameter:
            if 'solar' in parameter:
                return 0.005  # 0.5%/year
            elif 'wind' in parameter:
                return 0.015  # 1.5%/year

        return None

    def _get_typical_cost_percentages(self, parameter: str) -> List[float]:
        """Get typical cost percentage values"""
        if 'land' in parameter:
            return [5, 8, 10, 15]
        elif 'installation' in parameter or 'grid' in parameter:
            return [0, 5, 10, 15, 20]
        elif 'o&m' in parameter.lower():
            return [2, 2.5, 3, 4, 5]
        elif 'replacement' in parameter:
            return [25, 40, 50, 75]
        else:
            return [0, 5, 10, 15, 20, 25]

    def get_bounds_summary(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of all parameter bounds

        Args:
            category: Specific category to get summary for, or None for all

        Returns:
            Dictionary containing bounds summary
        """
        if category and category in self.bounds:
            return {'category': category, 'parameters': self.bounds[category]}
        elif category:
            return {'error': f'Category {category} not found'}
        else:
            return {'categories': list(self.bounds.keys()), 'bounds': self.bounds}

    def export_bounds_to_csv(self, output_file: str) -> None:
        """
        Export parameter bounds to CSV file

        Args:
            output_file: Path to output CSV file
        """
        import csv

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Category', 'Parameter', 'Min Value', 'Max Value', 'Description'])

            for category, params in self.bounds.items():
                for param, (min_val, max_val, desc) in params.items():
                    writer.writerow([category, param, min_val, max_val, desc])