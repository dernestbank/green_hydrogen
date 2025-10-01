"""
Configuration Manager for Hydrogen Cost Analysis Tool
Handles saving, loading, and managing user configurations.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class SystemConfiguration:
    """Data class for system configuration."""
    location: str = ""
    electrolyser_type: str = "PEM"
    nominal_electrolyser_capacity: float = 10.0
    nominal_solar_farm_capacity: float = 10.0
    nominal_wind_farm_capacity: float = 0.0
    battery_power_rating: float = 0.0
    battery_storage_duration: float = 0.0
    hourly_electricity_price: float = 40.0
    hydrogen_selling_price: float = 350.0
    discount_rate: float = 0.04
    project_life: int = 20

@dataclass
class ConfigurationMetadata:
    """Metadata for configuration files."""
    name: str
    description: str = ""
    created_date: str = ""
    modified_date: str = ""
    version: str = "2.0"
    author: str = ""

class ConfigurationManager:
    """
    Manages user configurations for the hydrogen cost analysis tool.

    Features:
    - Save/load configurations to/from disk
    - Configuration presets and templates
    - Configuration validation
    - Import/export configurations
    - Configuration versioning and metadata
    """

    def __init__(self, config_dir: str = "user_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.presets_dir = self.config_dir / "presets"
        self.presets_dir.mkdir(exist_ok=True)

        # Create default presets
        self._create_default_presets()

    def save_configuration(self, config_data: Dict[str, Any], name: str,
                         description: str = "", save_location: Optional[str] = None) -> str:
        """
        Save a configuration to disk.

        Args:
            config_data: Configuration data dictionary
            name: Configuration name
            description: Optional description
            save_location: Optional custom save path

        Returns:
            Path where configuration was saved
        """
        # Create metadata
        metadata = ConfigurationMetadata(
            name=name,
            description=description,
            created_date=datetime.now().isoformat(),
            modified_date=datetime.now().isoformat(),
            author="User"
        )

        # Combine metadata and configuration
        full_config = {
            "metadata": asdict(metadata),
            "configuration": config_data,
            "timestamp": datetime.now().isoformat()
        }

        # Determine save path
        if save_location:
            save_path = Path(save_location)
        else:
            # Create filename from name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name.replace(' ', '_')}_{timestamp}.yaml"
            save_path = self.config_dir / filename

        # Save as YAML for human readability
        with open(save_path, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {save_path}")
        return str(save_path)

    def load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load a configuration from disk.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration data dictionary
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    import json
                    data = json.load(f)
                else:
                    raise ValueError("Unsupported configuration file format. Use .yaml or .json")

            # Extract configuration data
            if isinstance(data, dict) and "configuration" in data:
                config_data = data["configuration"]
                # Add metadata if available
                if "metadata" in data:
                    config_data["_metadata"] = data["metadata"]
                return config_data
            else:
                # Legacy format compatibility
                return data

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def list_configurations(self) -> List[Dict[str, Any]]:
        """
        List all saved user configurations.

        Returns:
            List of configuration information
        """
        configs = []

        for config_file in self.config_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)

                metadata = data.get("metadata", {})

                config_info = {
                    "name": metadata.get("name", config_file.stem),
                    "description": metadata.get("description", ""),
                    "file_path": str(config_file),
                    "created_date": metadata.get("created_date", ""),
                    "modified_date": metadata.get("modified_date", ""),
                    "version": metadata.get("version", "1.0")
                }

                configs.append(config_info)

            except Exception as e:
                logger.warning(f"Error reading configuration {config_file}: {e}")

        return sorted(configs, key=lambda x: x["created_date"], reverse=True)

    def delete_configuration(self, config_path: str) -> bool:
        """
        Delete a configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            True if deleted successfully
        """
        config_file = Path(config_path)

        if config_file.exists() and config_file.parent == self.config_dir:
            config_file.unlink()
            logger.info(f"Configuration deleted: {config_path}")
            return True

        return False

    def _create_default_presets(self):
        """Create default configuration presets."""

        presets = {
            "Base Case": {
                "location": "US.CA",
                "electrolyser_type": "PEM",
                "nominal_electrolyser_capacity": 10.0,
                "nominal_solar_farm_capacity": 10.0,
                "nominal_wind_farm_capacity": 0.0,
                "battery_power_rating": 0.0,
                "battery_storage_duration": 0.0,
                "hourly_electricity_price": 40.0,
                "hydrogen_selling_price": 350.0,
                "discount_rate": 0.04,
                "project_life": 20
            },
            "Wind + Battery": {
                "location": "REZ-W1",
                "electrolyser_type": "PEM",
                "nominal_electrolyser_capacity": 10.0,
                "nominal_solar_farm_capacity": 0.0,
                "nominal_wind_farm_capacity": 15.0,
                "battery_power_rating": 2.0,
                "battery_storage_duration": 4.0,
                "hourly_electricity_price": 35.0,
                "hydrogen_selling_price": 320.0,
                "discount_rate": 0.04,
                "project_life": 20
            },
            "Hybrid System": {
                "location": "REZ-H1",
                "electrolyser_type": "PEM",
                "nominal_electrolyser_capacity": 20.0,
                "nominal_solar_farm_capacity": 25.0,
                "nominal_wind_farm_capacity": 10.0,
                "battery_power_rating": 5.0,
                "battery_storage_duration": 4.0,
                "hourly_electricity_price": 38.0,
                "hydrogen_selling_price": 360.0,
                "discount_rate": 0.04,
                "project_life": 25
            },
            "Large Scale": {
                "location": "REZ-L1",
                "electrolyser_type": "PEM",
                "nominal_electrolyser_capacity": 50.0,
                "nominal_solar_farm_capacity": 100.0,
                "nominal_wind_farm_capacity": 30.0,
                "battery_power_rating": 10.0,
                "battery_storage_duration": 6.0,
                "hourly_electricity_price": 42.0,
                "hydrogen_selling_price": 380.0,
                "discount_rate": 0.05,
                "project_life": 30
            },
            "Cost Optimized": {
                "location": "REZ-O1",
                "electrolyser_type": "AE",
                "nominal_electrolyser_capacity": 15.0,
                "nominal_solar_farm_capacity": 20.0,
                "nominal_wind_farm_capacity": 5.0,
                "battery_power_rating": 1.0,
                "battery_storage_duration": 2.0,
                "hourly_electricity_price": 25.0,
                "hydrogen_selling_price": 280.0,
                "discount_rate": 0.03,
                "project_life": 20
            }
        }

        for preset_name, config in presets.items():
            preset_file = self.presets_dir / f"{preset_name.replace(' ', '_').lower()}.yaml"

            if not preset_file.exists():
                metadata = ConfigurationMetadata(
                    name=preset_name,
                    description=f"Default preset: {preset_name}",
                    created_date=datetime.now().isoformat(),
                    modified_date=datetime.now().isoformat(),
                    author="System"
                )

                full_preset = {
                    "metadata": asdict(metadata),
                    "configuration": config,
                    "timestamp": datetime.now().isoformat()
                }

                with open(preset_file, 'w') as f:
                    yaml.dump(full_preset, f, default_flow_style=False, sort_keys=False)

    def get_preset_configurations(self) -> List[Dict[str, Any]]:
        """
        Get all preset configurations.

        Returns:
            List of preset configuration information
        """
        presets = []

        for preset_file in self.presets_dir.glob("*.yaml"):
            try:
                with open(preset_file, 'r') as f:
                    data = yaml.safe_load(f)

                metadata = data.get("metadata", {})
                config = data.get("configuration", {})

                preset_info = {
                    "name": metadata.get("name", preset_file.stem),
                    "description": metadata.get("description", ""),
                    "file_path": str(preset_file),
                    "configuration": config,
                    "is_preset": True
                }

                presets.append(preset_info)

            except Exception as e:
                logger.warning(f"Error reading preset {preset_file}: {e}")

        return sorted(presets, key=lambda x: x["name"])

    def get_preset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific preset by name.

        Args:
            name: Preset name

        Returns:
            Preset configuration or None if not found
        """
        presets = self.get_preset_configurations()

        for preset in presets:
            if preset["name"] == name:
                return preset

        return None

    def export_configuration(self, config_data: Dict[str, Any], export_path: str,
                           format_type: str = "yaml") -> str:
        """
        Export configuration to external file.

        Args:
            config_data: Configuration data to export
            export_path: Path to export to
            format_type: Export format ('yaml', 'json')

        Returns:
            Path where configuration was exported
        """
        if format_type.lower() == "yaml":
            with open(export_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        elif format_type.lower() == "json":
            import json
            with open(export_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
        else:
            raise ValueError("Unsupported export format. Use 'yaml' or 'json'")

        logger.info(f"Configuration exported to: {export_path}")
        return export_path

    def import_configuration(self, import_path: str) -> Dict[str, Any]:
        """
        Import configuration from external file.

        Args:
            import_path: Path to import from

        Returns:
            Imported configuration data
        """
        return self.load_configuration(import_path)

    def validate_configuration(self, config_data: Dict[str, Any]) -> List[str]:
        """
        Validate configuration data structure and values.

        Args:
            config_data: Configuration data to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        required_fields = [
            'location', 'electrolyser_type', 'nominal_electrolyser_capacity',
            'nominal_solar_farm_capacity', 'nominal_wind_farm_capacity'
        ]

        for field in required_fields:
            if field not in config_data:
                errors.append(f"Missing required field: {field}")

        # Validate value ranges
        if 'nominal_electrolyser_capacity' in config_data:
            capacity = config_data['nominal_electrolyser_capacity']
            if not isinstance(capacity, (int, float)) or capacity <= 0:
                errors.append("Electrolyser capacity must be positive number")

        if 'electrolyser_type' in config_data:
            elec_type = config_data['electrolyser_type']
            if elec_type not in ['AE', 'PEM']:
                errors.append("Electrolyser type must be 'AE' or 'PEM'")

        if 'nominal_solar_farm_capacity' in config_data:
            solar_cap = config_data['nominal_solar_farm_capacity']
            if not isinstance(solar_cap, (int, float)) or solar_cap < 0:
                errors.append("Solar capacity must be non-negative number")

        if 'nominal_wind_farm_capacity' in config_data:
            wind_cap = config_data['nominal_wind_farm_capacity']
            if not isinstance(wind_cap, (int, float)) or wind_cap < 0:
                errors.append("Wind capacity must be non-negative number")

        if 'discount_rate' in config_data:
            disc_rate = config_data['discount_rate']
            if not isinstance(disc_rate, (int, float)) or disc_rate < 0 or disc_rate > 1:
                errors.append("Discount rate must be between 0 and 1")

        return errors

# Factory function
def create_configuration_manager(config_dir: str = "user_configs") -> ConfigurationManager:
    """
    Factory function to create a ConfigurationManager instance.

    Args:
        config_dir: Directory to store configurations

    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager(config_dir)