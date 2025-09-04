from typing import Dict, Any, List, Optional
from src.utils.parameter_bounds import ParameterBounds
from src.utils.configuration_types import ConfigurationManager
import json
import datetime

class PresetManager:
    """Manager for configuration presets (common scenarios)"""

    def __init__(self):
        """Initialize preset manager"""
        self.bounds = ParameterBounds()
        self.config_manager = ConfigurationManager()
        self.presets = self._initialize_presets()

    def _initialize_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined configuration presets"""

        presets = {
            'pilot_project': {
                'name': 'Pilot Project (Small Scale)',
                'description': 'Small-scale pilot project for technology demonstration',
                'category': 'Scale',
                'parameters': {
                    'nominal_electrolyser_capacity': 1.0,    # 1 MW
                    'nominal_solar_farm_capacity': 1.5,      # 1.5 MW
                    'battery_rated_power': 0.5,              # 0.5 MW
                    'duration_of_storage_hours': 2,
                    'plant_life_years': 15,
                    'location': 'Remote industrial area'
                }
            },

            'demonstration_plant': {
                'name': 'Demonstration Plant (Medium Scale)',
                'description': 'Medium-scale demonstration plant for commercial viability',
                'category': 'Scale',
                'parameters': {
                    'nominal_electrolyser_capacity': 10.0,   # 10 MW
                    'nominal_solar_farm_capacity': 15.0,     # 15 MW
                    'nominal_wind_farm_capacity': 5.0,       # 5 MW (if hybrid)
                    'battery_rated_power': 3.0,              # 3 MW
                    'duration_of_storage_hours': 4,
                    'plant_life_years': 20,
                    'location': 'Industrial zone with grid access'
                }
            },

            'commercial_scale': {
                'name': 'Commercial Scale Plant',
                'description': 'Full-scale commercial hydrogen production plant',
                'category': 'Scale',
                'parameters': {
                    'nominal_electrolyser_capacity': 100.0,  # 100 MW
                    'nominal_solar_farm_capacity': 150.0,    # 150 MW
                    'nominal_wind_farm_capacity': 75.0,      # 75 MW (if hybrid)
                    'battery_rated_power': 25.0,             # 25 MW
                    'duration_of_storage_hours': 6,
                    'plant_life_years': 25,
                    'location': 'Dedicated industrial hydrogen park'
                }
            },

            'urban_location': {
                'name': 'Urban Location Setup',
                'description': 'Configuration optimized for urban installation constraints',
                'category': 'Location',
                'parameters': {
                    'location_type': 'Urban',
                    'land_cost_percent': 25.0,               # Higher land costs
                    'grid_connection_cost_percent': 15.0,    # Higher grid costs
                    'installation_cost_percent': 20.0,       # Higher installation costs
                    'nominal_electrolyser_capacity': 5.0,    # Smaller due to space constraints
                    'nominal_solar_farm_capacity': 7.5,      # Rooftop/urban solar
                    'battery_rated_power': 2.0,              # Smaller battery
                    'round_trip_efficiency': 0.88,           # Possibly higher efficiency systems
                }
            },

            'remote_location': {
                'name': 'Remote Location Setup',
                'description': 'Configuration for remote/off-grid installations',
                'category': 'Location',
                'parameters': {
                    'location_type': 'Remote',
                    'land_cost_percent': 3.0,                # Lower land costs
                    'transmission_connection_cost_percent': 20.0,  # Higher transmission costs
                    'nominal_solar_farm_capacity': 20.0,     # Larger solar for reliability
                    'nominal_wind_farm_capacity': 10.0,      # Add wind for diversity
                    'battery_rated_power': 8.0,              # Larger battery for stability
                    'duration_of_storage_hours': 8,          # Longer storage
                    'electrolyser_min_load': 15,             # Higher minimum load due to energy variability
                }
            },

            'industrial_park': {
                'name': 'Industrial Park Integration',
                'description': 'Optimized for integration into existing industrial facilities',
                'category': 'Business Model',
                'parameters': {
                    'integration_type': 'Industrial facility integration',
                    'nominal_electrolyser_capacity': 25.0,   # Industrial scale
                    'grid_connection_cost_percent': 8.0,     # Existing infrastructure
                    'installation_cost_percent': 12.0,       # Some shared infrastructure
                    'land_cost_percent': 8.0,               # Existing land use
                    'oxygen_retail_price_a_kg': 0.15,       # Industrial oxygen pricing
                    'average_electricity_spot_price_a_mwh': 45.0,  # Industrial rate
                    'water_cost': 2.5,                      # Industrial water rate
                }
            },

            'cost_optimized': {
                'name': 'Cost-Optimized Configuration',
                'description': 'Emphasizes capital cost reduction and operational efficiency',
                'category': 'Business Model',
                'parameters': {
                    'optimization_focus': 'Capital cost minimization',
                    'scale_index': 0.75,                    # Favor larger scale for economies
                    'electrolyser_efficiency': 0.78,        # Higher efficiency systems
                    'om_cost_percent': 1.8,                  # Lower O&M costs
                    'land_cost_percent': 4.0,               # Minimize land use
                    'nominal_electrolyser_capacity': 50.0,  # Large scale
                    'nominal_solar_farm_capacity': 75.0,    # Large scale
                    'plant_life_years': 30,                 # Maximize project life
                }
            },

            'conservative_financial': {
                'name': 'Conservative Financial Model',
                'description': 'Conservative financial assumptions for risk-averse investors',
                'category': 'Financial',
                'parameters': {
                    'risk_profile': 'Conservative',
                    'discount_rate': 8.0,                   # Higher discount rate
                    'plant_life_years': 20,                 # Shorter project life
                    'nominal_electrolyser_capacity': 25.0,  # Medium scale
                    'nominal_solar_farm_capacity': 37.5,    # Medium scale
                    'battery_rated_power': 5.0,             # Moderate battery
                    'electrolyser_efficiency': 0.70,        # Conservative efficiency
                    'total_system_sec': 65.0,               # Higher SEC
                }
            },

            'optimistic_financial': {
                'name': 'Optimistic Financial Model',
                'description': 'Optimistic financial assumptions for high-growth scenarios',
                'category': 'Financial',
                'parameters': {
                    'risk_profile': 'Optimistic',
                    'discount_rate': 4.0,                   # Lower discount rate
                    'plant_life_years': 25,                 # Longer project life
                    'nominal_electrolyser_capacity': 75.0,  # Large scale
                    'nominal_solar_farm_capacity': 100.0,   # Large scale
                    'battery_rated_power': 15.0,            # Large battery
                    'electrolyser_efficiency': 0.80,        # High efficiency
                    'total_system_sec': 48.0,               # Lower SEC
                    'hydrogen_price_a_kg': 6.0,             # Higher H2 price (future projection)
                }
            },

            'research_demo': {
                'name': 'Research & Demonstration',
                'description': 'Configuration for R&D and technological demonstration',
                'category': 'Specialized',
                'parameters': {
                    'purpose': 'Research and demonstration',
                    'nominal_electrolyser_capacity': 0.5,   # Lab scale
                    'nominal_solar_farm_capacity': 1.0,     # Small research array
                    'nominal_wind_farm_capacity': 0.5,      # Small wind turbine
                    'battery_rated_power': 0.25,            # Small battery system
                    'electrolyser_min_load': 20,            # Allow wider operating range
                    'electrolyser_max_load': 120,           # Allow overload for testing
                    'data_logging_interval': 1,             # Higher frequency data logging
                    'include_raw_data': True,               # Store all raw data
                }
            },

            'emergency_backup': {
                'name': 'Emergency Backup System',
                'description': 'Configuration optimized for emergency power backup scenarios',
                'category': 'Specialized',
                'parameters': {
                    'purpose': 'Emergency backup',
                    'nominal_electrolyser_capacity': 5.0,   # Backup scale
                    'battery_rated_power': 8.0,             # Large battery for reliability
                    'duration_of_storage_hours': 12,        # Long duration storage
                    'minimum_state_of_charge': 20,          # Higher minimum charge
                    'maximum_state_of_charge': 95,          # Maximize usable capacity
                    'electrolyser_min_load': 5,             # Very low minimum load
                    'round_trip_efficiency': 0.85,          # Focus on reliability over efficiency
                }
            },

            'hydrogen_refueling': {
                'name': 'Hydrogen Refueling Station',
                'description': 'Configuration specialized for hydrogen fuel cell vehicle refueling',
                'category': 'End Use',
                'parameters': {
                    'end_use': 'Hydrogen refueling station',
                    'nominal_electrolyser_capacity': 3.5,   # Typical station scale (500 kg/day)
                    'nominal_solar_farm_capacity': 5.0,     # Size for daily production
                    'battery_rated_power': 2.0,             # Sufficient for load leveling
                    'hydrogen_price_a_kg': 8.0,             # Retail fueling price
                    'operating_pressure_bar': 35.0,         # High pressure for refueling
                    'include_raw_data': False,              # Minimal data logging for cost
                    'data_logging_interval': 60,            # Hourly summary only
                }
            },

            'industrial_feedstock': {
                'name': 'Industrial Feedstock',
                'description': 'Configuration optimized for industrial hydrogen consumption',
                'category': 'End Use',
                'parameters': {
                    'end_use': 'Industrial hydrogen feedstock',
                    'nominal_electrolyser_capacity': 50.0,  # Large industrial scale
                    'nominal_solar_farm_capacity': 75.0,    # Large scale generation
                    'nominal_wind_farm_capacity': 25.0,     # Hybrid addition
                    'hydrogen_price_a_kg': 2.5,             # Industrial bulk price
                    'operating_pressure_bar': 25.0,         # High pressure delivery
                    'round_trip_efficiency': 0.92,          # High efficiency for cost optimization
                    'total_system_sec': 50.0,               # Low SEC for competitiveness
                }
            }
        }

        return presets

    def get_available_presets(self, category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available configuration presets, optionally filtered by category

        Args:
            category: Category to filter presets by ('Scale', 'Location', 'Business Model', etc.)

        Returns:
            Dictionary of presets with their configurations
        """
        if category:
            filtered_presets = {
                key: preset for key, preset in self.presets.items()
                if preset.get('category') == category
            }
            return filtered_presets
        return self.presets

    def get_preset_info(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific preset

        Args:
            preset_name: Name of the preset

        Returns:
            Dictionary containing preset information or None if not found
        """
        return self.presets.get(preset_name)

    def apply_preset(self, preset_name: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply a preset configuration to a base configuration

        Args:
            preset_name: Name of the preset to apply
            base_config: Base configuration to merge with preset (optional)

        Returns:
            Complete configuration dictionary with preset applied
        """
        if preset_name not in self.presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        preset = self.presets[preset_name]

        # Start with system defaults
        config = dict(self.bounds.default_values)

        # Apply preset on top of defaults
        config.update(preset['parameters'])

        # Apply user-provided base configuration on top of preset
        if base_config:
            config.update(base_config)

        # Validate bounds
        config = self.bounds.apply_defaults_and_bounds(config)

        # Add metadata about the preset used
        config['_preset_applied'] = {
            'name': preset['name'],
            'description': preset['description'],
            'category': preset['category']
        }

        return config

    def create_custom_preset(self, name: str, description: str, category: str,
                           parameters: Dict[str, Any], save_to_file: bool = False) -> None:
        """
        Create a custom configuration preset

        Args:
            name: Unique name for the preset
            description: Description of the preset
            category: Category for organization
            parameters: Parameter values for the preset
            save_to_file: Whether to save the preset to persistent storage
        """
        # Validate parameters against bounds
        validation_result = self.bounds.validate_parameters_batch(parameters)

        if validation_result['invalid']:
            invalid_params = [item['parameter'] for item in validation_result['invalid']]
            raise ValueError(f"Invalid parameters in preset: {', '.join(invalid_params)}")

        # Create preset structure
        preset = {
            'name': name,
            'description': description,
            'category': category,
            'parameters': parameters,
            'custom': True,
            'created_at': datetime.datetime.now().isoformat()
        }

        # Add to presets dictionary
        self.presets[name.lower().replace(' ', '_')] = preset

        if save_to_file:
            self._save_preset_to_file(name.lower().replace(' ', '_'), preset)

    def compare_presets(self, preset_names: List[str],
                       metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple presets across specified metrics

        Args:
            preset_names: List of preset names to compare
            metrics: List of parameter names to compare (optional)

        Returns:
            Dictionary containing comparison data
        """
        if not metrics:
            metrics = [
                'nominal_electrolyser_capacity',
                'nominal_solar_farm_capacity',
                'nominal_wind_farm_capacity',
                'battery_rated_power',
                'total_system_sec',
                'plant_life_years'
            ]

        comparison = {}

        for preset_name in preset_names:
            preset = self.get_preset_info(preset_name)
            if not preset:
                continue

            preset_config = self.apply_preset(preset_name)
            comparison[preset_name] = {
                'name': preset['name'],
                'category': preset['category'],
                'parameters': {metric: preset_config.get(metric, 'N/A') for metric in metrics}
            }

        return comparison

    def suggest_presets(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest presets based on user requirements

        Args:
            requirements: Dictionary of user requirements (capacity, location, etc.)

        Returns:
            List of suggested presets with match scores
        """
        suggestions = []

        capacity_req = requirements.get('required_capacity', 0)
        has_battery = requirements.get('requires_battery', False)
        grid_connected = requirements.get('grid_connected', True)
        location_type = requirements.get('location_type', 'any')

        for preset_name, preset in self.presets.items():
            score = 0
            reasons = []

            # Capacity matching
            preset_capacity = preset['parameters'].get('nominal_electrolyser_capacity', 0)
            if abs(preset_capacity - capacity_req) / max(capacity_req, 1) < 0.3:
                score += 3
                reasons.append("Capacity matches requirements")

            # Battery matching
            preset_has_battery = preset['parameters'].get('battery_rated_power', 0) > 0
            if has_battery == preset_has_battery:
                score += 2
                reasons.append("Battery configuration matches")

            # Location type matching
            preset_location = preset['parameters'].get('location_type', '')
            if location_type == 'any' or preset_location.lower() == location_type.lower():
                score += 1
                reasons.append("Location type compatibility")

            if score > 1:  # Only suggest if there's some match
                suggestions.append({
                    'preset': preset_name,
                    'name': preset['name'],
                    'description': preset['description'],
                    'score': score,
                    'reasons': reasons,
                    'parameters': preset['parameters']
                })

        # Sort by score (highest first)
        suggestions.sort(key=lambda x: x['score'], reverse=True)

        return suggestions[:5]  # Return top 5 suggestions

    def export_preset(self, preset_name: str, format: str = 'json') -> str:
        """
        Export a preset configuration in the requested format

        Args:
            preset_name: Name of the preset to export
            format: Export format ('json', 'yaml', 'csv')

        Returns:
            String representation of the preset in requested format
        """
        preset = self.get_preset_info(preset_name)
        if not preset:
            raise ValueError(f"Preset '{preset_name}' not found")

        if format == 'json':
            return json.dumps(preset, indent=2, default=str)
        elif format == 'yaml':
            # Simple YAML-like output (could use yaml library if installed)
            yaml_output = f"""name: "{preset['name']}"
description: "{preset['description']}"
category: "{preset['category']}"
parameters:
"""
            for key, value in preset['parameters'].items():
                yaml_output += f"  {key}: {value}\n"
            return yaml_output
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _save_preset_to_file(self, preset_name: str, preset_data: Dict[str, Any]) -> None:
        """Save a custom preset to persistent storage"""
        import os
        from pathlib import Path

        presets_dir = Path("config/presets")
        presets_dir.mkdir(parents=True, exist_ok=True)

        preset_file = presets_dir / f"{preset_name}.json"

        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2, default=str)

    def load_custom_presets(self) -> None:
        """Load custom presets from persistent storage"""
        import os
        from pathlib import Path

        presets_dir = Path("config/presets")

        if presets_dir.exists():
            for preset_file in presets_dir.glob("*.json"):
                preset_name = preset_file.stem
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                    self.presets[preset_name] = preset_data

    def get_preset_categories(self) -> List[str]:
        """Get list of unique preset categories"""
        categories = set()
        for preset in self.presets.values():
            if 'category' in preset:
                categories.add(preset['category'])
        return sorted(list(categories))