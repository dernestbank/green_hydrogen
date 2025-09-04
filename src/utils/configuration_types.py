from typing import Dict, List, Any, Optional, Set
from enum import Enum

class ConfigurationType(Enum):
    """Enumeration of power plant configuration types"""
    C1 = "C1. Standalone Solar PV Generator with Electrolyser"
    C2 = "C2. Standalone Solar PV Generator with Electrolyser and Battery"
    C3 = "C3. Grid Connected Solar PV Generator with Electrolyser"
    C4 = "C4. Grid Connected Solar PV Generator with Electrolyser with Surplus Retailed to Grid"
    C5 = "C5. Grid Connected Solar PV Generator with Electrolyser and Battery"
    C6 = "C6. Grid Connected Solar PV Generator with Electrolyser and Battery with Surplus Retailed to Grid"
    C7 = "C7. Solar PPA with Electrolyser"
    C8 = "C8. Solar PPA with Electrolyser and Battery"
    C9 = "C9. Standalone Wind Generator with Electrolyser"
    C10 = "C10. Standalone Wind Generator with Electrolyser and Battery"
    C11 = "C11. Grid Connected Wind Generator with Electrolyser"
    C12 = "C12. Grid Connected Wind Generator with Electrolyser with Surplus Retailed to Grid"
    C13 = "C13. Grid Connected Wind Generator with Electrolyser and Battery"
    C14 = "C14. Grid Connected Wind Generator with Electrolyser and Battery with Surplus Retail to Gird"
    C15 = "C15. Wind PPA with Electrolyser"
    C16 = "C16. Wind PPA with Electrolyser and Battery"
    C17 = "C17. Standalone Hybrid Generator with Electrolyser"
    C18 = "C18. Standalone Hybrid Generator with Electrolyser and Battery"
    C19 = "C19. Grid Connected Hybrid Generator with Electrolyser"
    C20 = "C20. Grid Connected Hybrid Generator with Electrolyser with Surplus Retailed to Grid"
    C21 = "C21. Grid Connected Hybrid Generator with Electrolyser and Battery"
    C22 = "C22. Grid Connected Hybrid Generator with Electrolyser and Battery with Surplus Retailed to Grid"
    C23 = "C23. Hybrid PPA with Electrolyser"
    C24 = "C24. Hybrid PPA with Electrolyser and Battery"

class ConfigurationManager:
    """Manager for handling different power plant configuration types (C1-C24)"""

    def __init__(self):
        """Initialize configuration manager with all configuration definitions"""
        self.config_definitions = self._initialize_config_definitions()

    def _initialize_config_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize definitions for all configuration types"""

        base_config = {
            "required_sizing": ["nominal_electrolyser_capacity"],
            "required_operational": [],
            "required_financial": [
                "plant_life_years", "discount_rate", "inflation_rate", "tax_rate",
                "financing_via_equity", "loan_term_years", "interest_rate_on_loan_p_a"
            ]
        }

        # Solar-based configurations (C1-C8)
        solar_configs = {
            f"C{i}": {
                **base_config,
                "generator_type": "Solar PV",
                "required_sizing": ["nominal_electrolyser_capacity", "nominal_solar_farm_capacity"],
                "has_battery": i in [2, 5, 6, 8],
                "has_grid": i in [3, 4, 5, 6],
                "has_ppa": i in [7, 8],
                "has_surplus_retail": i in [4, 6],
            } for i in range(1, 9)
        }

        # Wind-based configurations (C9-C16)
        wind_configs = {
            f"C{i}": {
                **base_config,
                "generator_type": "Wind",
                "required_sizing": ["nominal_electrolyser_capacity", "nominal_wind_farm_capacity"],
                "has_battery": i in [10, 13, 14, 16],
                "has_grid": i in [11, 12, 13, 14],
                "has_ppa": i in [15, 16],
                "has_surplus_retail": i in [12, 14],
            } for i in [9, 10, 11, 12, 13, 14, 15, 16]
        }

        # Hybrid configurations (C17-C24)
        hybrid_configs = {
            f"C{i}": {
                **base_config,
                "generator_type": "Hybrid",
                "required_sizing": [
                    "nominal_electrolyser_capacity",
                    "nominal_solar_farm_capacity",
                    "nominal_wind_farm_capacity"
                ],
                "has_battery": i in [18, 21, 22, 24],
                "has_grid": i in [19, 20, 21, 22],
                "has_ppa": i in [23, 24],
                "has_surplus_retail": i in [20, 22],
            } for i in [17, 18, 19, 20, 21, 22, 23, 24]
        }

        # Merge all configurations
        all_configs = {**solar_configs, **wind_configs, **hybrid_configs}
        return all_configs

    def get_config_definition(self, config_type: str) -> Dict[str, Any]:
        """Get definition for a specific configuration type"""
        config_type_clean = config_type.split()[0] if '.' in config_type else config_type
        return self.config_definitions.get(config_type_clean, {})

    def validate_configuration_type(self, config_type: str) -> bool:
        """Validate that a configuration type exists"""
        config_type_clean = config_type.split()[0] if '.' in config_type else config_type
        return config_type_clean in self.config_definitions

    def get_required_parameters(self, config_type: str) -> Dict[str, List[str]]:
        """Get all required parameters for a configuration type"""
        config = self.get_config_definition(config_type)

        required_params = {
            "sizing": config.get("required_sizing", []),
            "operational": config.get("required_operational", []),
            "financial": config.get("required_financial", []),
        }

        # Add battery-specific requirements if configuration has battery
        if config.get("has_battery", False):
            required_params["battery"] = [
                "battery_rated_power", "duration_of_storage_hours",
                "minimum_state_of_charge", "maximum_state_of_charge",
                "round_trip_efficiency"
            ]

        # Add grid-specific requirements if configuration has grid
        if config.get("has_grid", False):
            required_params["grid"] = [
                "grid_connection_cost_percent",
                "grid_service_charge_percent"
            ]

        # Add PPA-specific requirements if configuration has PPA
        if config.get("has_ppa", False):
            required_params["ppa"] = [
                "principal_ppa_cost_percent",
                "transmission_connection_cost_percent"
            ]

        return required_params

    def get_optional_parameters(self, config_type: str) -> Dict[str, List[str]]:
        """Get optional parameters for a configuration type"""
        config = self.get_config_definition(config_type)

        optional_params = {
            "electrolyser_advanced": [
                "active_cell_area_m2", "current_density_amps_cm2",
                "anode_thickness_mm", "cathode_thickness_mm", "membrane_thickness_mm",
                "operating_temperature_c", "operating_pressure_bar",
                "limiting_current_density_amps_cm2", "exchange_current_density_amps_cm2"
            ],
            "power_plant_advanced": [],
            "financial_advanced": [
                "salvage_costs_of_total_investments",
                "decommissioning_costs_of_total_investments",
                "additional_upfront_costs_a", "additional_annual_costs_a_yr",
                "average_electricity_spot_price_a_mwh", "oxygen_retail_price_a_kg"
            ],
            "environmental": []
        }

        # Add solar-specific parameters if solar configuration
        if config.get("generator_type") in ["Solar PV", "Hybrid"]:
            optional_params["power_plant_advanced"].extend([
                "solar_dataset", "solar_year", "solar_capacity_kw", "solar_system_loss",
                "solar_tracking", "solar_tilt", "solar_azimuth", "solar_include_raw_data"
            ])

        # Add wind-specific parameters if wind configuration
        if config.get("generator_type") in ["Wind", "Hybrid"]:
            optional_params["power_plant_advanced"].extend([
                "wind_dataset", "wind_year", "wind_capacity_kw",
                "wind_hub_height_m", "wind_turbine_model", "wind_include_raw_data"
            ])

        return optional_params

    def get_default_values(self, config_type: str) -> Dict[str, Any]:
        """Get default values for a configuration type"""
        config = self.get_config_definition(config_type)
        generator_type = config.get("generator_type", "Solar PV")

        # Base defaults
        defaults = {
            "nominal_electrolyser_capacity": 10.0,  # MW
            "generator_type": generator_type,
            "electrolyser_choice": "PEM",

            # Efficiency parameters
            "sec_at_nominal_load": 55.0,  # kWh/kg
            "total_system_sec": 58.0,     # kWh/kg
            "electrolyser_efficiency": 0.75,

            # Operational limits
            "electrolyser_min_load": 0.1,  # 10%
            "electrolyser_max_load": 1.0,  # 100%
            "max_overload_duration": 0,
            "time_between_overload": 24,  # hours

            # Financial parameters
            "plant_life_years": 25,
            "discount_rate": 5.0,         # %
            "inflation_rate": 2.0,        # %
            "tax_rate": 30.0,             # %
            "financing_via_equity": 30.0, # %
            "loan_term_years": 10,
            "interest_rate_on_loan_p_a": 5.0,  # %

            "land_cost_percent": 6.0,
            "installation_cost_percent": 0.0,
            "om_cost_percent": 2.5,
            "stack_replacement_percent": 40.0,
            "water_cost": 5.0,

            "salvage_costs_of_total_investments": 5.0,
            "decommissioning_costs_of_total_investments": 5.0,
            "additional_upfront_costs_a": 0.0,
            "additional_annual_costs_a_yr": 0.0,

            "depreciation_profile": "Modified Accelerated Cost Recovery System (7 years)",
        }

        # Generator-specific defaults
        if generator_type == "Solar PV":
            solar_defaults = self._get_solar_defaults(config)
            defaults.update(solar_defaults)
        elif generator_type == "Wind":
            wind_defaults = self._get_wind_defaults(config)
            defaults.update(wind_defaults)
        elif generator_type == "Hybrid":
            solar_defaults = self._get_solar_defaults(config)
            wind_defaults = self._get_wind_defaults(config)
            defaults.update(solar_defaults)
            defaults.update(wind_defaults)

        # Battery defaults (if applicable)
        if config.get("has_battery", False):
            battery_defaults = self._get_battery_defaults()
            defaults.update(battery_defaults)

        # Grid defaults (if applicable)
        if config.get("has_grid", False) or config.get("has_ppa", False):
            grid_defaults = self._get_grid_defaults()
            defaults.update(grid_defaults)

        return defaults

    def _get_solar_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get solar-specific default values"""
        return {
            "nominal_solar_farm_capacity": 10.0,  # MW
            "solar_dataset": "MERRA-2 (global)",
            "solar_year": 2023,
            "solar_capacity_kw": 1000.0,   # kW (used for API)
            "solar_system_loss": 10.0,     # %
            "solar_tracking": "None",
            "solar_tilt": 35.0,           # degrees
            "solar_azimuth": 180.0,       # degrees
            "solar_include_raw_data": False,

            "solar_reference_capacity": 1000.0,     # kW
            "solar_reference_equipment_cost": 1500.0,  # A$/kW
            "solar_scale_index": 0.9,
            "solar_cost_reduction": 0.0,    # %
            "solar_installation_costs": 0.0,  # %
            "solar_land_cost": 8.0,         # %
            "solar_opex": 17000.0,          # A$/MW/year
            "solar_degradation": 0.5,       # %
            "solar_pv_degradation_rate_percent_year": 0.5,  # %
        }

    def _get_wind_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get wind-specific default values"""
        return {
            "nominal_wind_farm_capacity": 10.0,  # MW
            "wind_dataset": "MERRA-2 (global)",
            "wind_year": 2023,
            "wind_capacity_kw": 1000.0,     # kW (used for API)
            "wind_hub_height_m": 80.0,      # meters
            "wind_turbine_model": "Vestas V90 2000",
            "wind_include_raw_data": False,

            "wind_reference_capacity": 1000.0,   # kW
            "wind_reference_cost": 3000.0,       # A$/kW
            "wind_scale_index": 0.9,
            "wind_cost_reduction": 0.0,     # %
            "wind_installation_costs": 0.0, # %
            "wind_land_cost": 8.0,          # %
            "wind_opex": 25000.0,           # A$/MW/year
            "wind_degradation": 1.5,        # %
            "wind_farm_degradation_rate_percent_year": 1.5,  # %
        }

    def _get_battery_defaults(self) -> Dict[str, Any]:
        """Get battery-specific default values"""
        return {
            "battery_rated_power": 5.0,           # MW
            "duration_of_storage_hours": 4,       # hours
            "round_trip_efficiency": 90.0,        # %
            "minimum_state_of_charge": 10.0,      # %
            "maximum_state_of_charge": 90.0,      # %
            "battery_capex_a_kwh": 300.0,         # A$/kWh
            "battery_indirect_costs_percent_of_capex": 10.0,    # %
            "battery_replacement_cost_of_capex": 50.0,          # %
            "battery_opex_a_mw_yr": 10.0,         # A$/MW/yr
        }

    def _get_grid_defaults(self) -> Dict[str, Any]:
        """Get grid-specific default values"""
        return {
            "grid_connection_cost_percent": 5.0,        # %
            "grid_service_charge_percent": 2.0,         # %
            "principal_ppa_cost_percent": 0.0,          # %
            "transmission_connection_cost_percent": 5.0, # %
            "average_electricity_spot_price_a_mwh": 0.0, # A$/MWh
        }

    def apply_configuration_presets(self, inputs: Dict[str, Any],
                                   config_type: str) -> Dict[str, Any]:
        """Apply configuration-specific presets to user inputs"""
        config = self.get_config_definition(config_type)
        defaults = self.get_default_values(config_type)

        # Start with system-wide defaults
        preset_inputs = {**defaults}

        # Apply or merge user inputs, preferring user provided values
        preset_inputs.update(inputs)

        return preset_inputs

    def validate_configuration_compatibility(self, inputs: Dict[str, Any],
                                          config_type: str) -> List[str]:
        """Validate configuration compatibility and logical consistency"""
        issues = []
        config = self.get_config_definition(config_type)

        generator_type = config.get("generator_type")

        # Check sizing compatibility
        if generator_type == "Solar PV" or generator_type == "Hybrid":
            solar_capacity = inputs.get("nominal_solar_farm_capacity", 0)
            if solar_capacity <= 0:
                issues.append("Solar capacity must be positive for this configuration")

        if generator_type == "Wind" or generator_type == "Hybrid":
            wind_capacity = inputs.get("nominal_wind_farm_capacity", 0)
            if wind_capacity <= 0:
                issues.append("Wind capacity must be positive for this configuration")

        # Battery configuration checks
        if config.get("has_battery", False):
            battery_power = inputs.get("battery_rated_power", 0)
            if battery_power <= 0:
                issues.append("Battery rated power must be positive for this configuration")
        elif inputs.get("battery_rated_power", 0) > 0:
            issues.append("Battery configuration detected but not required for this power plant type")

        # Grid configuration checks
        if config.get("has_grid", False) or config.get("has_ppa", False):
            # These configurations support grid interaction, which is handled in financial modeling
            pass
        else:
            grid_cost = inputs.get("grid_connection_cost_percent", 0)
            if grid_cost > 0:
                issues.append("Grid connection cost set but configuration does not support grid connection")

        # PPA configuration checks
        if config.get("has_ppa", False):
            ppa_cost = inputs.get("principal_ppa_cost_percent", 0)
            if ppa_cost == 0:
                issues.append("PPA configuration requires principal PPA cost to be set")
        elif inputs.get("principal_ppa_cost_percent", 0) > 0:
            issues.append("PPA cost set but configuration does not support power purchase agreements")

        return issues

    def get_configuration_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Get matrix of configuration features for display/comparison"""
        matrix = {}
        for config_type, definition in self.config_definitions.items():
            matrix[config_type] = {
                "has_solar": definition.get("generator_type", "") in ["Solar PV", "Hybrid"],
                "has_wind": definition.get("generator_type", "") in ["Wind", "Hybrid"],
                "has_battery": definition.get("has_battery", False),
                "has_grid": definition.get("has_grid", False),
                "has_ppa": definition.get("has_ppa", False),
                "has_surplus_retail": definition.get("has_surplus_retail", False),
            }
        return matrix

    def get_similar_configurations(self, config_type: str) -> List[str]:
        """Get list of similar configuration types"""
        config = self.get_config_definition(config_type)
        generator_type = config.get("generator_type", "")
        similar = []

        for other_config, other_definition in self.config_definitions.items():
            if (other_config != config_type.split()[0] and
                other_definition.get("generator_type") == generator_type and
                other_definition.get("has_battery") == config.get("has_battery", False)):
                similar.append(other_definition["C0"])  # Full name

        return similar