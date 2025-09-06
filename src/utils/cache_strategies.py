"""
Advanced Cache Invalidation and Update Strategies

Provides intelligent cache management strategies including:
- Time-based invalidation policies
- Data freshness validation
- Smart cache warming
- Conditional updates
- Cache coherence management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .api_cache import APICache

logger = logging.getLogger(__name__)


class InvalidationTrigger(Enum):
    """Cache invalidation trigger types."""
    TIME_BASED = "time_based"
    DATA_CHANGE = "data_change"
    USER_REQUEST = "user_request"
    SYSTEM_EVENT = "system_event"
    ERROR_RECOVERY = "error_recovery"


class CacheUpdateStrategy(Enum):
    """Cache update strategy types."""
    EAGER = "eager"          # Update immediately when data changes
    LAZY = "lazy"            # Update only when requested and stale
    SCHEDULED = "scheduled"  # Update on a schedule
    CONDITIONAL = "conditional"  # Update based on conditions


@dataclass
class InvalidationRule:
    """Defines cache invalidation rules."""
    trigger: InvalidationTrigger
    pattern: str
    condition: Optional[Callable] = None
    max_age: Optional[int] = None  # seconds
    description: str = ""


@dataclass
class CacheUpdatePolicy:
    """Defines cache update policies."""
    strategy: CacheUpdateStrategy
    schedule: Optional[str] = None  # cron-like schedule
    condition_func: Optional[Callable] = None
    priority: int = 1  # 1=low, 5=high
    batch_size: int = 10


class CacheInvalidationManager:
    """
    Advanced cache invalidation and update strategy manager.

    Handles complex cache invalidation scenarios, update policies,
    and maintains cache coherence across the system.
    """

    def __init__(self, cache: APICache):
        """
        Initialize invalidation manager.

        Args:
            cache: APICache instance to manage
        """
        self.cache = cache
        self.invalidation_rules: List[InvalidationRule] = []
        self.update_policies: Dict[str, CacheUpdatePolicy] = {}
        self.invalidation_history: List[Tuple[datetime, str, str]] = []
        self.max_history = 1000

        # Register default invalidation rules
        self._register_default_rules()

        logger.info("Cache invalidation manager initialized")

    def _register_default_rules(self):
        """Register default invalidation rules."""

        # Time-based rules
        self.add_invalidation_rule(InvalidationRule(
            trigger=InvalidationTrigger.TIME_BASED,
            pattern="solar:*",
            max_age=86400,  # 24 hours
            description="Solar data older than 24 hours"
        ))

        self.add_invalidation_rule(InvalidationRule(
            trigger=InvalidationTrigger.TIME_BASED,
            pattern="wind:*",
            max_age=86400,  # 24 hours
            description="Wind data older than 24 hours"
        ))

        self.add_invalidation_rule(InvalidationRule(
            trigger=InvalidationTrigger.TIME_BASED,
            pattern="hybrid:*",
            max_age=86400,  # 24 hours
            description="Hybrid data older than 24 hours"
        ))

        # Error recovery rule
        self.add_invalidation_rule(InvalidationRule(
            trigger=InvalidationTrigger.ERROR_RECOVERY,
            pattern="*",
            description="Invalidate potentially corrupted data"
        ))

    def add_invalidation_rule(self, rule: InvalidationRule):
        """
        Add a cache invalidation rule.

        Args:
            rule: InvalidationRule to add
        """
        self.invalidation_rules.append(rule)
        logger.debug(f"Added invalidation rule: {rule.description}")

    def remove_invalidation_rule(self, pattern: str, trigger: InvalidationTrigger) -> bool:
        """
        Remove invalidation rule by pattern and trigger.

        Args:
            pattern: Cache key pattern
            trigger: Invalidation trigger type

        Returns:
            True if rule was removed
        """
        initial_count = len(self.invalidation_rules)
        self.invalidation_rules = [
            rule for rule in self.invalidation_rules
            if not (rule.pattern == pattern and rule.trigger == trigger)
        ]

        removed = len(self.invalidation_rules) < initial_count
        if removed:
            logger.debug(f"Removed invalidation rule for pattern: {pattern}")

        return removed

    def add_update_policy(self, name: str, policy: CacheUpdatePolicy):
        """
        Add cache update policy.

        Args:
            name: Policy name
            policy: CacheUpdatePolicy definition
        """
        self.update_policies[name] = policy
        logger.debug(f"Added update policy: {name}")

    def invalidate_by_pattern(self, pattern: str, reason: str = "manual") -> int:
        """
        Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match cache keys
            reason: Reason for invalidation

        Returns:
            Number of entries invalidated
        """
        removed_count = self.cache.invalidate_pattern(pattern)

        # Record invalidation
        self._record_invalidation(pattern, reason)

        logger.info(f"Invalidated {removed_count} entries matching '{pattern}' - {reason}")
        return removed_count

    def invalidate_stale_data(self, max_age_seconds: int = 86400) -> int:
        """
        Invalidate stale cache entries based on age.

        Args:
            max_age_seconds: Maximum age in seconds (default: 24 hours)

        Returns:
            Number of entries invalidated
        """
        removed_count = 0
        cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)

        try:
            # Get all cache keys and check timestamps
            for key in list(self.cache.cache):
                # Check if entry is expired (diskcache handles this internally)
                # Force cleanup of expired entries
                self.cache.cleanup_expired()
                removed_count += 1

            self._record_invalidation(f"stale_data_older_than_{max_age_seconds}s", "time_based")

        except Exception as e:
            logger.error(f"Error during stale data invalidation: {e}")

        logger.info(f"Invalidated {removed_count} stale cache entries")
        return removed_count

    def invalidate_by_location(self, lat: float, lon: float, radius_km: float = 0) -> int:
        """
        Invalidate cache entries for specific location or radius.

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius in kilometers (0 for exact match)

        Returns:
            Number of entries invalidated
        """
        removed_count = 0

        if radius_km == 0:
            # Exact location match
            location_pattern = f"lat_{lat}_lon_{lon}"
            removed_count = self.invalidate_by_pattern(f"*{location_pattern}*", f"location_{lat}_{lon}")
        else:
            # Radius-based invalidation
            logger.warning(f"Radius-based invalidation not fully implemented - using exact match for {lat}, {lon}")
            removed_count = self.invalidate_by_location(lat, lon, 0)

        return removed_count

    def invalidate_by_date_range(self, start_date: str, end_date: str) -> int:
        """
        Invalidate cache entries for specific date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Number of entries invalidated
        """
        removed_count = 0

        # Pattern matching for date ranges in cache keys
        date_patterns = [
            f"*{start_date}*",
            f"*{end_date}*",
            f"*year_{start_date[:4]}*",
            f"*year_{end_date[:4]}*"
        ]

        for pattern in date_patterns:
            removed_count += self.invalidate_by_pattern(pattern, f"date_range_{start_date}_to_{end_date}")

        return removed_count

    def smart_invalidation(self) -> Dict[str, int]:
        """
        Execute smart invalidation based on registered rules.

        Returns:
            Dictionary with invalidation results by rule type
        """
        results = {}

        for rule in self.invalidation_rules:
            try:
                if rule.trigger == InvalidationTrigger.TIME_BASED and rule.max_age:
                    count = self.invalidate_stale_data(rule.max_age)
                    results[f"time_based_{rule.pattern}"] = count

                elif rule.trigger == InvalidationTrigger.DATA_CHANGE:
                    # Check for data changes (implementation depends on data sources)
                    if rule.condition and rule.condition():
                        count = self.invalidate_by_pattern(rule.pattern, "data_change")
                        results[f"data_change_{rule.pattern}"] = count

                elif rule.trigger == InvalidationTrigger.ERROR_RECOVERY:
                    # This would be triggered by external error detection
                    pass

            except Exception as e:
                logger.error(f"Error executing invalidation rule {rule.description}: {e}")

        return results

    def cache_warming_strategy(self, locations: List[Tuple[float, float]],
                             api_client, data_types: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Implement cache warming strategy for common locations.

        Args:
            locations: List of (lat, lon) tuples to warm
            api_client: API client to fetch data
            data_types: Types of data to warm ('solar', 'wind', 'hybrid')

        Returns:
            Dictionary with warming results
        """
        if data_types is None:
            data_types = ['solar', 'wind']

        results = {'warmed_entries': 0, 'errors': 0}

        for lat, lon in locations:
            for data_type in data_types:
                try:
                    # Basic warming strategy - fetch recent year data
                    current_year = datetime.now().year
                    params = {
                        'lat': lat,
                        'lon': lon,
                        'year': current_year,
                        'dataset': 'merra2'
                    }

                    # Check if data is already cached
                    cache_key_params = {k: str(v) for k, v in params.items()}
                    cached = self.cache.get(data_type, cache_key_params, prefix=data_type)

                    if cached is None:
                        logger.info(f"Warming cache for {data_type} data at ({lat}, {lon})")

                        # Fetch data based on type
                        if data_type == 'solar' and hasattr(api_client, 'fetch_solar_data'):
                            fresh_data = api_client.fetch_solar_data(**params)
                            self.cache.set(data_type, cache_key_params, fresh_data, prefix=data_type)
                            results['warmed_entries'] += 1

                        elif data_type == 'wind' and hasattr(api_client, 'fetch_wind_data'):
                            fresh_data = api_client.fetch_wind_data(**params)
                            self.cache.set(data_type, cache_key_params, fresh_data, prefix=data_type)
                            results['warmed_entries'] += 1

                        elif data_type == 'hybrid' and hasattr(api_client, 'fetch_hybrid_data'):
                            fresh_data = api_client.fetch_hybrid_data(**params)
                            self.cache.set(data_type, cache_key_params, fresh_data, prefix=data_type)
                            results['warmed_entries'] += 1

                except Exception as e:
                    logger.error(f"Error warming cache for {data_type} at ({lat}, {lon}): {e}")
                    results['errors'] += 1

        logger.info(f"Cache warming completed: {results['warmed_entries']} entries warmed, {results['errors']} errors")
        return results

    def conditional_update(self, condition_func: Callable, pattern: str,
                          api_client, update_params: Dict) -> bool:
        """
        Execute conditional cache update based on custom condition.

        Args:
            condition_func: Function returning True if update needed
            pattern: Cache key pattern to update
            api_client: API client for fresh data
            update_params: Parameters for fetching fresh data

        Returns:
            True if update was performed
        """
        try:
            if condition_func():
                logger.info(f"Conditional update triggered for pattern: {pattern}")

                # Invalidate matching entries
                invalidated = self.invalidate_by_pattern(pattern, "conditional_update")

                # Optionally warm cache with fresh data
                # This would depend on the specific update_params structure

                self._record_invalidation(pattern, "conditional_update")
                return True

        except Exception as e:
            logger.error(f"Error during conditional update for {pattern}: {e}")

        return False

    def get_invalidation_statistics(self) -> Dict[str, Any]:
        """
        Get cache invalidation statistics.

        Returns:
            Dictionary with invalidation statistics
        """
        # Count invalidations by reason
        reason_counts = {}
        for _, pattern, reason in self.invalidation_history:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_invalidations = [
            (ts, pattern, reason) for ts, pattern, reason in self.invalidation_history
            if ts > recent_cutoff
        ]

        stats = {
            'total_invalidations': len(self.invalidation_history),
            'recent_invalidations_24h': len(recent_invalidations),
            'invalidations_by_reason': reason_counts,
            'active_rules_count': len(self.invalidation_rules),
            'active_policies_count': len(self.update_policies),
            'cache_stats': self.cache.get_stats()
        }

        return stats

    def _record_invalidation(self, pattern: str, reason: str):
        """Record invalidation event in history."""
        self.invalidation_history.append((datetime.now(), pattern, reason))

        # Trim history if it gets too long
        if len(self.invalidation_history) > self.max_history:
            self.invalidation_history = self.invalidation_history[-self.max_history:]

    def export_invalidation_report(self, filepath: Optional[Path] = None) -> str:
        """
        Export invalidation activity report.

        Args:
            filepath: Optional file path to save report

        Returns:
            Report content as string
        """
        stats = self.get_invalidation_statistics()

        report = f"""
Cache Invalidation Report
Generated: {datetime.now().isoformat()}

=== Summary ===
Total Invalidations: {stats['total_invalidations']}
Recent Invalidations (24h): {stats['recent_invalidations_24h']}
Active Rules: {stats['active_rules_count']}
Active Policies: {stats['active_policies_count']}

=== Invalidations by Reason ===
"""

        for reason, count in stats['invalidations_by_reason'].items():
            report += f"{reason}: {count}\n"

        report += f"""
=== Cache Statistics ===
Total Entries: {stats['cache_stats'].get('total_entries', 'N/A')}
Cache Size: {stats['cache_stats'].get('cache_size_mb', 'N/A')} MB
Cache Directory: {stats['cache_stats'].get('cache_directory', 'N/A')}

=== Active Rules ===
"""

        for rule in self.invalidation_rules:
            report += f"- {rule.trigger.value}: {rule.pattern} ({rule.description})\n"

        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(report)
                logger.info(f"Invalidation report exported to {filepath}")
            except Exception as e:
                logger.error(f"Error exporting report: {e}")

        return report


class SmartCacheManager:
    """
    High-level smart cache management combining caching and invalidation strategies.
    """

    def __init__(self, cache_dir: str = "cache/smart_cache"):
        """
        Initialize smart cache manager.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache = APICache(cache_dir=cache_dir)
        self.invalidation_manager = CacheInvalidationManager(self.cache)

        # Common location coordinates for cache warming
        self.common_locations = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437), # Los Angeles
            (41.8781, -87.6298),  # Chicago
            (29.7604, -95.3698),  # Houston
            (39.2904, -76.6122),  # Baltimore
            (25.7617, -80.1918),  # Miami
        ]

        logger.info("Smart cache manager initialized")

    def initialize_cache_warming(self, api_client):
        """
        Initialize cache with common locations and recent data.

        Args:
            api_client: API client for data fetching
        """
        logger.info("Starting intelligent cache warming...")
        results = self.invalidation_manager.cache_warming_strategy(
            locations=self.common_locations,
            api_client=api_client,
            data_types=['solar', 'wind']
        )

        logger.info(f"Cache warming completed: {results}")
        return results

    def daily_maintenance(self, api_client) -> Dict[str, Any]:
        """
        Perform daily cache maintenance operations.

        Args:
            api_client: API client for fresh data

        Returns:
            Maintenance operation results
        """
        logger.info("Starting daily cache maintenance...")

        results = {
            'invalidated_stale': 0,
            'warmed_entries': 0,
            'cleanup_expired': 0,
            'errors': []
        }

        try:
            # Clean up expired entries
            results['cleanup_expired'] = self.cache.cleanup_expired()

            # Invalidate stale data (older than 24 hours)
            results['invalidated_stale'] = self.invalidation_manager.invalidate_stale_data(86400)

            # Smart invalidation based on rules
            smart_results = self.invalidation_manager.smart_invalidation()
            results['smart_invalidation'] = smart_results

            # Warm cache for common locations
            warming_results = self.invalidation_manager.cache_warming_strategy(
                locations=self.common_locations[:3],  # Top 3 locations only
                api_client=api_client
            )
            results['warmed_entries'] = warming_results.get('warmed_entries', 0)

        except Exception as e:
            error_msg = f"Error during daily maintenance: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        logger.info(f"Daily maintenance completed: {results}")
        return results

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache and invalidation statistics."""
        return {
            'cache_stats': self.cache.get_stats(),
            'invalidation_stats': self.invalidation_manager.get_invalidation_statistics(),
            'manager_info': {
                'cache_directory': str(self.cache.cache_dir),
                'common_locations_count': len(self.common_locations),
                'initialized': True
            }
        }

    def close(self):
        """Close cache connections."""
        self.cache.close()
        logger.info("Smart cache manager closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()