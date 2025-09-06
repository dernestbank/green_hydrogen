"""
Tests for Cache Configuration Loader

Test loading cache configuration from YAML files and applying
settings to cache managers.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from src.utils.cache_config_loader import CacheConfigLoader, load_cache_config


class TestCacheConfigLoader:
    """Test cases for CacheConfigLoader."""

    @pytest.fixture
    def sample_config(self):
        """Sample cache configuration."""
        return {
            'cache': {
                'default_ttl_seconds': 7200,
                'max_cache_size_mb': 512,
                'cache_directory': 'test_cache',
                'enable_compression': True,
                'invalidation': {
                    'cleanup_interval': 1800,
                    'data_ttl': {
                        'solar_data': 3600,
                        'wind_data': 3600
                    },
                    'max_stale_age': 86400,
                    'patterns': [
                        {
                            'pattern': '*test*',
                            'max_age': 1800,
                            'reason': 'test_data'
                        }
                    ]
                },
                'warming': {
                    'enabled': True,
                    'common_locations': [
                        [40.7, -74.0, "Test Location"],
                        [34.0, -118.2, "Another Location"]
                    ],
                    'data_types': ['solar', 'wind'],
                    'max_warm_entries': 25
                },
                'monitoring': {
                    'stats_enabled': True,
                    'log_level': 'DEBUG',
                    'performance_metrics': True
                }
            },
            'environments': {
                'testing': {
                    'cache': {
                        'default_ttl_seconds': 60,
                        'max_cache_size_mb': 10,
                        'warming': {
                            'enabled': False
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def config_file(self, sample_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            yield f.name

        Path(f.name).unlink()  # Cleanup

    def test_load_config_file(self, config_file):
        """Test loading configuration from file."""
        loader = CacheConfigLoader(config_path=config_file, environment='production')

        assert loader.config['cache']['default_ttl_seconds'] == 7200
        assert loader.config['cache']['max_cache_size_mb'] == 512
        assert loader.config['cache']['enable_compression'] is True

    def test_environment_override(self, config_file):
        """Test environment-specific configuration override."""
        # Load with testing environment
        loader = CacheConfigLoader(config_path=config_file, environment='testing')

        # Should have testing overrides applied
        assert loader.config['cache']['default_ttl_seconds'] == 60  # Overridden
        assert loader.config['cache']['max_cache_size_mb'] == 10    # Overridden
        assert loader.config['cache']['warming']['enabled'] is False # Overridden

        # Should retain base values where not overridden
        assert loader.config['cache']['enable_compression'] is True

    def test_default_config_fallback(self):
        """Test fallback to default config when file doesn't exist."""
        loader = CacheConfigLoader(config_path='nonexistent_file.yaml')

        # Should have default values
        assert 'cache' in loader.config
        assert 'default_ttl_seconds' in loader.config['cache']
        assert loader.config['cache']['default_ttl_seconds'] == 86400  # Default value

    def test_create_api_cache(self, config_file):
        """Test creating APICache from configuration."""
        loader = CacheConfigLoader(config_path=config_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override cache directory to temp directory
            loader.config['cache']['cache_directory'] = temp_dir

            cache = loader.create_api_cache()

            assert cache.default_ttl == 7200
            assert cache.enable_compression is True

            cache.close()

    def test_create_invalidation_manager(self, config_file):
        """Test creating invalidation manager from configuration."""
        loader = CacheConfigLoader(config_path=config_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            loader.config['cache']['cache_directory'] = temp_dir

            cache = loader.create_api_cache()
            manager = loader.create_invalidation_manager(cache)

            # Should have configured rules
            assert len(manager.invalidation_rules) > 0

            cache.close()

    def test_create_smart_cache_manager(self, config_file):
        """Test creating smart cache manager from configuration."""
        loader = CacheConfigLoader(config_path=config_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            loader.config['cache']['cache_directory'] = temp_dir

            smart_manager = loader.create_smart_cache_manager()

            # Should have configured common locations
            expected_locations = [(40.7, -74.0), (34.0, -118.2)]
            assert smart_manager.common_locations == expected_locations

            smart_manager.close()

    def test_get_cache_warming_locations(self, config_file):
        """Test getting cache warming locations."""
        loader = CacheConfigLoader(config_path=config_file)

        locations = loader.get_cache_warming_locations()

        assert len(locations) == 2
        assert locations[0] == [40.7, -74.0, "Test Location"]
        assert locations[1] == [34.0, -118.2, "Another Location"]

    def test_get_settings_methods(self, config_file):
        """Test various settings getter methods."""
        loader = CacheConfigLoader(config_path=config_file)

        # Test warming settings
        warming_settings = loader.get_cache_warming_settings()
        assert warming_settings['enabled'] is True
        assert warming_settings['max_warm_entries'] == 25

        # Test monitoring settings
        monitoring_settings = loader.get_monitoring_settings()
        assert monitoring_settings['stats_enabled'] is True
        assert monitoring_settings['log_level'] == 'DEBUG'

    def test_export_current_config(self, config_file):
        """Test exporting current configuration."""
        loader = CacheConfigLoader(config_path=config_file)

        # Export to string
        yaml_content = loader.export_current_config()
        assert 'cache:' in yaml_content
        assert 'default_ttl_seconds: 7200' in yaml_content

    def test_validate_configuration_success(self, config_file):
        """Test successful configuration validation."""
        loader = CacheConfigLoader(config_path=config_file)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set valid cache directory
            loader.config['cache']['cache_directory'] = temp_dir

            is_valid, errors = loader.validate_configuration()

            assert is_valid is True
            assert len(errors) == 0

    def test_validate_configuration_errors(self, config_file):
        """Test configuration validation with errors."""
        loader = CacheConfigLoader(config_path=config_file)

        # Introduce validation errors
        loader.config['cache']['default_ttl_seconds'] = -1  # Invalid TTL
        loader.config['cache']['max_cache_size_mb'] = 0     # Invalid size
        loader.config['cache']['warming']['common_locations'] = [
            [200, -300]  # Invalid coordinates
        ]

        is_valid, errors = loader.validate_configuration()

        assert is_valid is False
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])


if __name__ == "__main__":
    pytest.main([__file__])