import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
import os

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for loading and managing YAML-based configurations"""

    def __init__(self, config_dir: str = "config", default_config_file: str = "config.yaml"):
        """
        Initialize the configuration manager

        Args:
            config_dir: Directory containing configuration files
            default_config_file: Name of the default configuration file
        """
        self.config_dir = Path(config_dir)
        self.default_config_file = default_config_file
        self._config_cache: Dict[str, Any] = {}
        self._loaded_files: Dict[str, Path] = {}

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_name: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_name: Name of config file (without extension). If None, loads default
            reload: Force reload even if cached

        Returns:
            Configuration dictionary
        """
        config_file = self._get_config_path(config_name)

        # Check cache if not reloading
        if not reload and config_name in self._config_cache:
            return self._config_cache[config_name]

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {config_file}")

        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_file}. Using default configuration.")
            config = self._create_default_config(config_name)

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_file}: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")

        # Cache the loaded configuration
        self._config_cache[config_name or "default"] = config
        self._loaded_files[config_name or "default"] = config_file

        return config

    def save_config(self, config: Dict[str, Any], config_name: Optional[str] = None,
                   backup: bool = True) -> None:
        """
        Save configuration to YAML file

        Args:
            config: Configuration dictionary to save
            config_name: Name of config file (without extension). If None, uses default
            backup: Whether to create backup of existing file
        """
        config_file = self._get_config_path(config_name)

        if backup and config_file.exists():
            backup_file = config_file.with_suffix('.yaml.backup')
            config_file.replace(backup_file)
            logger.info(f"Created backup: {backup_file}")

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
                logger.info(f"Saved configuration to {config_file}")

        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            raise

    def get_value(self, key_path: str, default: Any = None,
                  config_name: Optional[str] = None) -> Any:
        """
        Get a configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'api.rate_limit')
            default: Default value if key not found
            config_name: Name of config file to load. If None, uses default

        Returns:
            Configuration value or default
        """
        config = self.load_config(config_name)

        keys = key_path.split('.')
        try:
            value = config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set_value(self, key_path: str, value: Any,
                  config_name: Optional[str] = None) -> None:
        """
        Set a configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
            config_name: Name of config file to modify
        """
        config = self.load_config(config_name)
        keys = key_path.split('.')

        # Navigate to the parent dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value

        # Save the modified configuration
        self.save_config(config, config_name)

    def merge_configs(self, base_config: Dict[str, Any],
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge override configuration into base configuration

        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if (key in result and isinstance(result[key], dict) and
                isinstance(value, dict)):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def validate_config(self, config: Dict[str, Any],
                       schema: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against a schema

        Args:
            config: Configuration dictionary to validate
            schema: Schema definition dictionary

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        for section, section_schema in schema.items():
            if section not in config:
                if section_schema.get('required', False):
                    errors.append(f"Missing required section: {section}")
                continue

            section_config = config[section]
            if isinstance(section_schema.get('fields'), dict):
                # Validate fields in this section
                for field, field_schema in section_schema['fields'].items():
                    if field not in section_config:
                        if field_schema.get('required', False):
                            errors.append(f"Missing required field: {section}.{field}")
                        continue

                    value = section_config[field]
                    expected_type = field_schema.get('type')

                    if expected_type and not isinstance(value, eval(expected_type)):
                        errors.append(f"Invalid type for {section}.{field}: expected {expected_type}, got {type(value)}")

                    if field_schema.get('min_value') is not None and value < field_schema['min_value']:
                        errors.append(f"Value for {section}.{field} below minimum: {field_schema['min_value']}")

                    if field_schema.get('max_value') is not None and value > field_schema['max_value']:
                        errors.append(f"Value for {section}.{field} above maximum: {field_schema['max_value']}")

        return errors

    def export_config_json(self, config_name: Optional[str] = None,
                          output_file: Optional[str] = None) -> str:
        """
        Export configuration as JSON string

        Args:
            config_name: Name of config file to export
            output_file: Optional file path to save JSON

        Returns:
            JSON string representation of configuration
        """
        config = self.load_config(config_name)
        json_str = json.dumps(config, indent=2, default=str)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)

        return json_str

    def _get_config_path(self, config_name: Optional[str] = None) -> Path:
        """Get full path to configuration file"""
        filename = f"{config_name}.yaml" if config_name else self.default_config_file
        return self.config_dir / filename

    def _create_default_config(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a default configuration if file doesn't exist"""
        if config_name == "user_preferences":
            return self._get_default_user_preferences()
        elif config_name == "electrolyser_configs":
            return self._get_default_electrolyser_configs()
        else:
            return self._get_default_main_config()

    def _get_default_main_config(self) -> Dict[str, Any]:
        """Get default main application configuration"""
        return {
            'application': {
                'name': 'Green Hydrogen Production Framework',
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'api': {
                'base_url': 'https://www.renewables.ninja/api',
                'rate_limit': 50,
                'timeout': 30
            },
            'defaults': {
                'latitude': 40.7128,
                'longitude': -74.0060,
                'solar_capacity': 5.0,
                'wind_capacity': 3.0,
                'electrolyser_capacity': 1.0
            },
            'visualization': {
                'theme': 'light',
                'plot_width': 800,
                'plot_height': 600
            }
        }

    def _get_default_user_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            'ui': {
                'language': 'en',
                'timezone': 'UTC',
                'currency': 'USD'
            },
            'defaults': {
                'auto_save': True,
                'confirm_actions': False,
                'number_format': 'scientific'
            }
        }

    def _get_default_electrolyser_configs(self) -> Dict[str, Any]:
        """Get default electrolyser configurations"""
        return {
            'C1': {
                'name': 'PEM Electrolyser - Small Scale',
                'capacity_min': 0.1,
                'capacity_max': 1.0,
                'efficiency': 0.65,
                'capital_cost': 800,  # USD/kW
                'operating_cost': 0.05,  # USD/kWh
                'lifetime': 15  # years
            },
            'C2': {
                'name': 'PEM Electrolyser - Medium Scale',
                'capacity_min': 1.0,
                'capacity_max': 10.0,
                'efficiency': 0.68,
                'capital_cost': 700,
                'operating_cost': 0.045,
                'lifetime': 20
            },
            'C3': {
                'name': 'PEM Electrolyser - Large Scale',
                'capacity_min': 10.0,
                'capacity_max': 100.0,
                'efficiency': 0.70,
                'capital_cost': 600,
                'operating_cost': 0.04,
                'lifetime': 25
            }
        }