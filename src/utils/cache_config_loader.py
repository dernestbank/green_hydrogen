"""
Cache Configuration Loader

Loads cache configuration from YAML files and applies settings
to cache managers and invalidation strategies.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .api_cache import APICache
from .cache_strategies import (
    CacheInvalidationManager,
    SmartCacheManager,
    InvalidationRule,
    CacheUpdatePolicy,
    InvalidationTrigger,
    CacheUpdateStrategy
)

logger = logging.getLogger(__name__)


class CacheConfigLoader:
    """
    Loads and applies cache configuration from YAML files.
    """

    def __init__(self, config_path: Optional[str] = None, environment: str = "production"):
        """
        Initialize cache configuration loader.

        Args:
            config_path: Path to cache configuration YAML file
            environment: Environment name (development, testing, production)
        """
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "cache_config.yaml")

        self.config_path = Path(config_path)
        """
        Initialize cache configuration loader.

        Args:
            config_path: Path to cache configuration YAML file
            environment: Environment name (development, testing, production)
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "cache_config.yaml"

        self.config_path = Path(config_path or "")
        self.environment = environment
        self.config = self._load_config()

        logger.info(f"Cache configuration loaded for environment: {environment}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Apply environment-specific overrides
            if 'environments' in config and self.environment in config['environments']:
                env_config = config['environments'][self.environment]
                config = self._merge_configs(config, env_config)

            return config

        except Exception as e:
            logger.error(f"Error loading cache config from {self.config_path}: {e}")
            return self._get_default_config()

    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """
        Recursively merge configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if file loading fails.

        Returns:
            Default configuration dictionary
        """
        return {
            'cache': {
                'default_ttl_seconds': 86400,
                'max_cache_size_mb': 1024,
                'cache_directory': 'cache/api_responses',
                'enable_compression': True,
                'invalidation': {
                    'cleanup_interval': 3600,
                    'data_ttl': {
                        'solar_data': 86400,
                        'wind_data': 86400,
                        'hybrid_data': 86400,
                        'location_data': 604800
                    },
                    'max_stale_age': 172800
                },
                'warming': {
                    'enabled': True,
                    'common_locations': [
                        [40.7128, -74.0060, "New York, NY"],
                        [34.0522, -118.2437, "Los Angeles, CA"],
                        [41.8781, -87.6298, "Chicago, IL"]
                    ],
                    'data_types': ['solar', 'wind'],
                    'strategy': 'common_locations',
                    'max_warm_entries': 50
                },
                'monitoring': {
                    'stats_enabled': True,
                    'log_level': 'INFO',
                    'performance_metrics': True
                }
            }
        }

    def create_api_cache(self) -> APICache:
        """
        Create APICache instance with loaded configuration.

        Returns:
            Configured APICache instance
        """
        cache_config = self.config.get('cache', {})

        return APICache(
            cache_dir=cache_config.get('cache_directory', 'cache/api_responses'),
            default_ttl=cache_config.get('default_ttl_seconds', 86400),
            max_size=cache_config.get('max_cache_size_mb', 1024) * 1024 * 1024,  # Convert MB to bytes
            enable_compression=cache_config.get('enable_compression', True)
        )

    def create_invalidation_manager(self, cache: APICache) -> CacheInvalidationManager:
        """
        Create CacheInvalidationManager with configured rules.

        Args:
            cache: APICache instance

        Returns:
            Configured CacheInvalidationManager
        """
        manager = CacheInvalidationManager(cache)

        # Add configured invalidation rules
        invalidation_config = self.config.get('cache', {}).get('invalidation', {})

        # Add TTL-based rules for different data types
        data_ttl = invalidation_config.get('data_ttl', {})
        for data_type, ttl in data_ttl.items():
            rule = InvalidationRule(
                trigger=InvalidationTrigger.TIME_BASED,
                pattern=f"{data_type}:*",
                max_age=ttl,
                description=f"TTL rule for {data_type}"
            )
            manager.add_invalidation_rule(rule)

        # Add pattern rules
        patterns = invalidation_config.get('patterns', [])
        for pattern_config in patterns:
            rule = InvalidationRule(
                trigger=InvalidationTrigger.TIME_BASED,
                pattern=pattern_config['pattern'],
                max_age=pattern_config['max_age'],
                description=pattern_config['reason']
            )
            manager.add_invalidation_rule(rule)

        return manager

    def create_smart_cache_manager(self) -> SmartCacheManager:
        """
        Create SmartCacheManager with full configuration.

        Returns:
            Configured SmartCacheManager
        """
        cache_config = self.config.get('cache', {})
        cache_dir = cache_config.get('cache_directory', 'cache/smart_api_responses')

        manager = SmartCacheManager(cache_dir=cache_dir)

        # Configure common locations from config
        warming_config = cache_config.get('warming', {})
        if warming_config.get('enabled', True):
            common_locations = warming_config.get('common_locations', [])
            # Convert to (lat, lon) tuples
            manager.common_locations = [(loc[0], loc[1]) for loc in common_locations]

        return manager

    def get_cache_warming_locations(self) -> List[Tuple[float, float, str]]:
        """
        Get cache warming locations from configuration.

        Returns:
            List of (lat, lon, description) tuples
        """
        warming_config = self.config.get('cache', {}).get('warming', {})
        return warming_config.get('common_locations', [])

    def get_cache_warming_settings(self) -> Dict[str, Any]:
        """
        Get cache warming settings.

        Returns:
            Cache warming configuration
        """
        return self.config.get('cache', {}).get('warming', {})

    def get_monitoring_settings(self) -> Dict[str, Any]:
        """
        Get cache monitoring settings.

        Returns:
            Monitoring configuration
        """
        return self.config.get('cache', {}).get('monitoring', {})

    def get_performance_settings(self) -> Dict[str, Any]:
        """
        Get cache performance settings.

        Returns:
            Performance configuration
        """
        return self.config.get('cache', {}).get('performance', {})

    def setup_logging(self):
        """Configure logging based on configuration."""
        monitoring_config = self.get_monitoring_settings()
        log_level = monitoring_config.get('log_level', 'INFO')

        # Configure cache-specific logger
        cache_logger = logging.getLogger('src.utils')
        cache_logger.setLevel(getattr(logging, log_level.upper()))

        # Add console handler if not already present
        if not cache_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            cache_logger.addHandler(handler)

        logger.info(f"Cache logging configured at {log_level} level")

    def export_current_config(self, output_path: Optional[str] = None) -> str:
        """
        Export current configuration to YAML format.

        Args:
            output_path: Optional path to save configuration

        Returns:
            Configuration as YAML string
        """
        yaml_content = yaml.dump(self.config, default_flow_style=False, indent=2)

        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(yaml_content)
                logger.info(f"Configuration exported to {output_path}")
            except Exception as e:
                logger.error(f"Error exporting configuration: {e}")

        return yaml_content

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate cache configuration for common issues.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            cache_config = self.config.get('cache', {})

            # Check required fields
            if not cache_config.get('cache_directory'):
                errors.append("cache_directory not specified")

            # Check TTL values are reasonable
            ttl = cache_config.get('default_ttl_seconds', 0)
            if ttl <= 0:
                errors.append("default_ttl_seconds must be positive")

            # Check cache size limits
            max_size = cache_config.get('max_cache_size_mb', 0)
            if max_size <= 0:
                errors.append("max_cache_size_mb must be positive")

            # Validate warming configuration
            warming_config = cache_config.get('warming', {})
            if warming_config.get('enabled'):
                locations = warming_config.get('common_locations', [])
                for i, location in enumerate(locations):
                    if not isinstance(location, list) or len(location) < 2:
                        errors.append(f"Invalid location format at index {i}")
                    else:
                        lat, lon = location[0], location[1]
                        if not (-90 <= lat <= 90):
                            errors.append(f"Invalid latitude {lat} at index {i}")
                        if not (-180 <= lon <= 180):
                            errors.append(f"Invalid longitude {lon} at index {i}")

            # Check directory permissions
            cache_dir = Path(cache_config.get('cache_directory', ''))
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                test_file = cache_dir / "write_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"Cache directory not writable: {e}")

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Cache configuration validation passed")
        else:
            logger.error(f"Cache configuration validation failed: {errors}")

        return is_valid, errors


def load_cache_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> CacheConfigLoader:
    """
    Convenience function to load cache configuration.

    Args:
        config_path: Path to configuration file
        environment: Environment name (auto-detected if None)

    Returns:
        CacheConfigLoader instance
    """
    if environment is None:
        # Auto-detect environment from environment variable
        environment = os.getenv('ENVIRONMENT', 'production').lower()

        # Common environment detection patterns
        if os.getenv('DEBUG') or os.getenv('FLASK_DEBUG'):
            environment = 'development'
        elif 'test' in os.getenv('PYTEST_CURRENT_TEST', '').lower():
            environment = 'testing'

    loader = CacheConfigLoader(config_path=config_path, environment=environment)
    loader.setup_logging()

    # Validate configuration
    is_valid, errors = loader.validate_configuration()
    if not is_valid:
        logger.warning(f"Cache configuration issues detected: {errors}")

    return loader