"""
Tests for Cache Invalidation and Update Strategies

Test advanced cache management including invalidation rules,
update policies, and smart cache management strategies.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.utils.api_cache import APICache
from src.utils.cache_strategies import (
    CacheInvalidationManager,
    SmartCacheManager,
    InvalidationRule,
    CacheUpdatePolicy,
    InvalidationTrigger,
    CacheUpdateStrategy
)


class TestCacheInvalidationManager:
    """Test cases for CacheInvalidationManager."""

    @pytest.fixture
    def temp_cache_manager(self):
        """Create temporary cache and invalidation manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = APICache(
                cache_dir=str(Path(temp_dir) / "test_cache"),
                default_ttl=60,
                max_size=1024*1024  # 1MB
            )
            manager = CacheInvalidationManager(cache)
            yield manager
            cache.close()

    def test_manager_initialization(self, temp_cache_manager):
        """Test invalidation manager initialization."""
        manager = temp_cache_manager

        # Should have default rules registered
        assert len(manager.invalidation_rules) > 0

        # Should have empty policies and history initially
        assert len(manager.update_policies) == 0
        assert len(manager.invalidation_history) == 0

    def test_add_invalidation_rule(self, temp_cache_manager):
        """Test adding invalidation rules."""
        manager = temp_cache_manager
        initial_count = len(manager.invalidation_rules)

        # Add custom rule
        rule = InvalidationRule(
            trigger=InvalidationTrigger.USER_REQUEST,
            pattern="test:*",
            description="Test rule"
        )
        manager.add_invalidation_rule(rule)

        assert len(manager.invalidation_rules) == initial_count + 1
        assert rule in manager.invalidation_rules

    def test_remove_invalidation_rule(self, temp_cache_manager):
        """Test removing invalidation rules."""
        manager = temp_cache_manager

        # Add a rule to remove
        rule = InvalidationRule(
            trigger=InvalidationTrigger.USER_REQUEST,
            pattern="removable:*",
            description="Removable rule"
        )
        manager.add_invalidation_rule(rule)
        initial_count = len(manager.invalidation_rules)

        # Remove the rule
        removed = manager.remove_invalidation_rule("removable:*", InvalidationTrigger.USER_REQUEST)

        assert removed is True
        assert len(manager.invalidation_rules) == initial_count - 1

        # Try to remove non-existent rule
        removed = manager.remove_invalidation_rule("nonexistent:*", InvalidationTrigger.USER_REQUEST)
        assert removed is False

    def test_add_update_policy(self, temp_cache_manager):
        """Test adding update policies."""
        manager = temp_cache_manager

        policy = CacheUpdatePolicy(
            strategy=CacheUpdateStrategy.LAZY,
            priority=3
        )
        manager.add_update_policy("test_policy", policy)

        assert "test_policy" in manager.update_policies
        assert manager.update_policies["test_policy"].strategy == CacheUpdateStrategy.LAZY

    def test_invalidate_by_pattern(self, temp_cache_manager):
        """Test pattern-based invalidation."""
        manager = temp_cache_manager
        cache = manager.cache

        # Add test data
        test_data = {"test": "data"}
        cache.set("solar_pv", {"lat": 40, "lon": -74}, test_data)
        cache.set("solar_pv", {"lat": 41, "lon": -75}, test_data)
        cache.set("wind", {"lat": 40, "lon": -74}, test_data)

        # Verify data exists
        assert cache.get("solar_pv", {"lat": 40, "lon": -74}) == test_data

        # Invalidate solar pattern
        removed = manager.invalidate_by_pattern("solar*", "test invalidation")

        # Should have recorded the invalidation
        assert len(manager.invalidation_history) == 1

        # Verify invalidation was recorded
        timestamp, pattern, reason = manager.invalidation_history[0]
        assert pattern == "solar*"
        assert reason == "test invalidation"
        assert removed >= 0

    def test_invalidate_by_location(self, temp_cache_manager):
        """Test location-based invalidation."""
        manager = temp_cache_manager
        cache = manager.cache

        # Add location-based test data
        test_data = {"location": "test"}
        cache.set("solar_pv", {"lat": 40.7128, "lon": -74.0060}, test_data)
        cache.set("wind", {"lat": 40.7128, "lon": -74.0060}, test_data)
        cache.set("solar_pv", {"lat": 41.8781, "lon": -87.6298}, test_data)  # Different location

        # Invalidate specific location
        removed = manager.invalidate_by_location(40.7128, -74.0060)

        # Should have recorded the invalidation
        assert len(manager.invalidation_history) > 0

    def test_smart_invalidation(self, temp_cache_manager):
        """Test smart invalidation based on rules."""
        manager = temp_cache_manager
        cache = manager.cache

        # Add test data
        test_data = {"test": "data"}
        cache.set("solar_pv", {"lat": 40, "lon": -74}, test_data)

        # Add a custom rule with short max_age for testing
        rule = InvalidationRule(
            trigger=InvalidationTrigger.TIME_BASED,
            pattern="test:*",
            max_age=1,  # 1 second
            description="Fast expiry test"
        )
        manager.add_invalidation_rule(rule)

        # Execute smart invalidation
        results = manager.smart_invalidation()

        # Should return results dictionary
        assert isinstance(results, dict)

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client for testing."""
        client = Mock()
        import pandas as pd

        # Mock responses
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=24, freq='H'),
            'capacity_factor': [0.1 + 0.01 * i for i in range(24)]
        })

        client.fetch_solar_data.return_value = mock_df
        client.fetch_wind_data.return_value = mock_df
        client.fetch_hybrid_data.return_value = {"solar": mock_df, "wind": mock_df}

        return client

    def test_cache_warming_strategy(self, temp_cache_manager, mock_api_client):
        """Test cache warming strategy."""
        manager = temp_cache_manager

        locations = [(40.7128, -74.0060), (34.0522, -118.2437)]

        # Execute cache warming
        results = manager.cache_warming_strategy(
            locations=locations,
            api_client=mock_api_client,
            data_types=['solar', 'wind']
        )

        # Should return results with counts
        assert 'warmed_entries' in results
        assert 'errors' in results
        assert isinstance(results['warmed_entries'], int)
        assert isinstance(results['errors'], int)

        # API should have been called for locations
        assert mock_api_client.fetch_solar_data.called
        assert mock_api_client.fetch_wind_data.called

    def test_get_invalidation_statistics(self, temp_cache_manager):
        """Test invalidation statistics."""
        manager = temp_cache_manager

        # Add some invalidation history
        manager.invalidate_by_pattern("test*", "test_reason")
        manager.invalidate_by_pattern("another*", "test_reason")
        manager.invalidate_by_pattern("different*", "other_reason")

        stats = manager.get_invalidation_statistics()

        # Should contain expected keys
        assert 'total_invalidations' in stats
        assert 'recent_invalidations_24h' in stats
        assert 'invalidations_by_reason' in stats
        assert 'active_rules_count' in stats
        assert 'cache_stats' in stats

        # Should have correct counts
        assert stats['total_invalidations'] == 3
        assert stats['invalidations_by_reason']['test_reason'] == 2
        assert stats['invalidations_by_reason']['other_reason'] == 1

    def test_export_invalidation_report(self, temp_cache_manager):
        """Test invalidation report export."""
        manager = temp_cache_manager

        # Add some activity
        manager.invalidate_by_pattern("test*", "export_test")

        # Generate report
        report = manager.export_invalidation_report()

        # Should contain expected sections
        assert "Cache Invalidation Report" in report
        assert "Total Invalidations:" in report
        assert "Invalidations by Reason" in report
        assert "Cache Statistics" in report
        assert "Active Rules" in report


class TestSmartCacheManager:
    """Test cases for SmartCacheManager."""

    @pytest.fixture
    def temp_smart_manager(self):
        """Create temporary smart cache manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SmartCacheManager(
                cache_dir=str(Path(temp_dir) / "smart_cache")
            )
            yield manager
            manager.close()

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client."""
        client = Mock()
        import pandas as pd
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=24, freq='H'),
            'capacity_factor': [0.2 + 0.01 * i for i in range(24)]
        })

        client.fetch_solar_data.return_value = mock_df
        client.fetch_wind_data.return_value = mock_df
        client.fetch_hybrid_data.return_value = {"solar": mock_df, "wind": mock_df}

        return client

    def test_smart_manager_initialization(self, temp_smart_manager):
        """Test smart manager initialization."""
        manager = temp_smart_manager

        # Should have cache and invalidation manager
        assert manager.cache is not None
        assert manager.invalidation_manager is not None
        assert len(manager.common_locations) > 0

    def test_initialize_cache_warming(self, temp_smart_manager, mock_api_client):
        """Test cache warming initialization."""
        manager = temp_smart_manager

        results = manager.initialize_cache_warming(mock_api_client)

        # Should return results
        assert isinstance(results, dict)
        assert 'warmed_entries' in results
        assert 'errors' in results

        # API should have been called for common locations
        assert mock_api_client.fetch_solar_data.called
        assert mock_api_client.fetch_wind_data.called

    def test_get_comprehensive_stats(self, temp_smart_manager):
        """Test comprehensive statistics."""
        manager = temp_smart_manager

        stats = manager.get_comprehensive_stats()

        # Should contain all expected sections
        assert 'cache_stats' in stats
        assert 'invalidation_stats' in stats
        assert 'manager_info' in stats

        # Manager info should be populated
        assert stats['manager_info']['initialized'] is True
        assert 'cache_directory' in stats['manager_info']
        assert stats['manager_info']['common_locations_count'] > 0


    def test_context_manager(self, mock_api_client):
        """Test smart manager as context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with SmartCacheManager(cache_dir=str(Path(temp_dir) / "context_test")) as manager:
                # Should be able to use manager within context
                results = manager.initialize_cache_warming(mock_api_client)
                assert isinstance(results, dict)


class TestInvalidationRulesAndPolicies:
    """Test invalidation rules and policies."""

    def test_invalidation_rule_creation(self):
        """Test creating invalidation rules."""
        rule = InvalidationRule(
            trigger=InvalidationTrigger.TIME_BASED,
            pattern="solar:*",
            max_age=3600,
            description="Solar data TTL"
        )

        assert rule.trigger == InvalidationTrigger.TIME_BASED
        assert rule.pattern == "solar:*"
        assert rule.max_age == 3600
        assert rule.description == "Solar data TTL"

    def test_cache_update_policy_creation(self):
        """Test creating update policies."""
        policy = CacheUpdatePolicy(
            strategy=CacheUpdateStrategy.SCHEDULED,
            schedule="0 2 * * *",  # Daily at 2 AM
            priority=3
        )

        assert policy.strategy == CacheUpdateStrategy.SCHEDULED
        assert policy.schedule == "0 2 * * *"
        assert policy.priority == 3
        assert policy.batch_size == 10  # default

    def test_rule_with_condition(self):
        """Test invalidation rule with custom condition."""
        def custom_condition():
            return True

        rule = InvalidationRule(
            trigger=InvalidationTrigger.DATA_CHANGE,
            pattern="overnight:*",
            condition=custom_condition,
            description="Overnight data changes"
        )

        assert rule.condition is not None
        assert callable(rule.condition)
        assert rule.condition() is True


if __name__ == "__main__":
    pytest.main([__file__])