"""
API Response Caching System

Provides file-based caching for API responses with configurable TTL,
cache invalidation, and compression support for large datasets.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from diskcache import Cache

logger = logging.getLogger(__name__)


class APICache:
    """
    File-based cache for API responses with TTL and intelligent compression support.

    Uses diskcache for persistent storage with automatic expiration,
    smart compression for large datasets, and cache statistics.
    """

    def __init__(self,
                 cache_dir: str = "cache/api_responses",
                 default_ttl: int = 3600,
                 max_size: int = 1024**3,  # 1GB
                 enable_compression: bool = True,
                 compression_threshold_kb: int = 100):
        """
        Initialize the API cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds (1 hour default)
            max_size: Maximum cache size in bytes (1GB default)
            enable_compression: Enable compression for large responses
            compression_threshold_kb: Only compress data larger than this threshold
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = Cache(
            directory=str(self.cache_dir),
            size_limit=max_size,
            eviction_policy='least-recently-used'
        )

        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.compression_threshold_kb = compression_threshold_kb

        # Initialize compression manager
        from .data_compression import CompressionManager
        self.compression_manager = CompressionManager(
            enable_compression=enable_compression,
            compression_threshold_kb=compression_threshold_kb,
            benchmark_compression=False  # Disable benchmarks for speed in cache context
        )

        logger.info(f"API Cache initialized at {self.cache_dir} with {max_size/1024**2:.1f}MB limit")
        logger.info(f"Compression: {'enabled' if enable_compression else 'disabled'} "
                   f"(threshold: {compression_threshold_kb}KB)")
    
    def _generate_key(self, 
                     endpoint: str, 
                     params: Dict[str, Any],
                     prefix: str = "api") -> str:
        """
        Generate a unique cache key from endpoint and parameters.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            prefix: Key prefix for organization
            
        Returns:
            Unique cache key
        """
        # Create a deterministic hash of endpoint + params
        params_str = json.dumps(params, sort_keys=True)
        key_content = f"{endpoint}_{params_str}"
        
        # Generate SHA256 hash for unique key
        key_hash = hashlib.sha256(key_content.encode()).hexdigest()
        
        return f"{prefix}:{endpoint.replace('/', '_')}:{key_hash[:16]}"
    
    def get(self,
            endpoint: str,
            params: Dict[str, Any],
            prefix: str = "api") -> Optional[Any]:
        """
        Retrieve cached response if available and not expired, with decompression support.

        Args:
            endpoint: API endpoint
            params: Request parameters
            prefix: Cache key prefix

        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(endpoint, params, prefix)

        try:
            cached_data = self.cache.get(key)
            if cached_data is not None:
                logger.debug(f"Cache hit for key: {key}")

                # Check if data was compressed during storage
                if isinstance(cached_data, dict) and cached_data.get('_compression_used'):
                    # Need to decompress
                    logger.debug(f"Decompressing cached data for {key}")
                    try:
                        original_data = self.compression_manager.decompress_data(cached_data['data'])
                        return original_data
                    except Exception as e:
                        logger.warning(f"Failed to decompress cached data for {key}: {e}")
                        return None
                else:
                    # No compression used
                    return cached_data
            else:
                logger.debug(f"Cache miss for key: {key}")
                return None

        except Exception as e:
            logger.warning(f"Cache retrieval error for {key}: {e}")
            return None
    
    def set(self,
            endpoint: str,
            params: Dict[str, Any],
            response_data: Any,
            ttl: Optional[int] = None,
            prefix: str = "api") -> bool:
        """
        Store API response in cache with TTL and intelligent compression.

        Args:
            endpoint: API endpoint
            params: Request parameters
            response_data: Response data to cache
            ttl: Time-to-live in seconds (uses default if None)
            prefix: Cache key prefix

        Returns:
            True if successfully cached, False otherwise
        """
        key = self._generate_key(endpoint, params, prefix)
        ttl = ttl or self.default_ttl

        try:
            # Compress data if appropriate
            processed_data, compression_result = self.compression_manager.compress_data(
                response_data,
                force_compress=False  # Let compression manager decide
            )

            # Store compression metadata if compression was used
            if compression_result:
                compressed_package = {
                    '_compression_used': True,
                    'original_size': compression_result.original_size,
                    'compressed_size': compression_result.compressed_size,
                    'compression_ratio': compression_result.compression_ratio,
                    'algorithm': compression_result.algorithm,
                    'data': processed_data
                }
                # Store compression result in stats
                logger.debug(f"Compressed {compression_result.data_type} data for {key}: "
                            f"{compression_result.original_size/1024:.1f}KB -> "
                            f"{compression_result.compressed_size/1024:.1f}KB "
                            f"({compression_result.compression_ratio:.2f}x)")

                success = self.cache.set(key, compressed_package, expire=ttl)
            else:
                # No compression - store original data
                success = self.cache.set(key, processed_data, expire=ttl)

            if success:
                logger.debug(f"Cached response for {key} (TTL: {ttl}s)")
            else:
                logger.warning(f"Failed to cache response for {key}")

            return success

        except Exception as e:
            logger.error(f"Cache storage error for {key}: {e}")
            return False
    
    def invalidate(self, 
                   endpoint: str, 
                   params: Dict[str, Any],
                   prefix: str = "api") -> bool:
        """
        Remove specific cached entry.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            prefix: Cache key prefix
            
        Returns:
            True if entry was removed, False if not found
        """
        key = self._generate_key(endpoint, params, prefix)
        
        try:
            existed = self.cache.pop(key) is not None
            if existed:
                logger.info(f"Invalidated cache for {key}")
            else:
                logger.debug(f"No cache entry to invalidate for {key}")
            return existed
            
        except Exception as e:
            logger.error(f"Cache invalidation error for {key}: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Remove all cached entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "renewables_ninja:*")
            
        Returns:
            Number of entries removed
        """
        removed_count = 0
        
        try:
            # Get all keys matching pattern
            matching_keys = [key for key in self.cache if pattern in key]
            
            for key in matching_keys:
                if self.cache.pop(key) is not None:
                    removed_count += 1
                    
            logger.info(f"Invalidated {removed_count} cache entries matching '{pattern}'")
            return removed_count
            
        except Exception as e:
            logger.error(f"Pattern invalidation error for '{pattern}': {e}")
            return 0
    
    def clear_all(self) -> bool:
        """
        Clear all cached entries.
        
        Returns:
            True if cache was cleared successfully
        """
        try:
            self.cache.clear()
            logger.info("Cleared all cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and metadata.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            stats = {
                'total_entries': len(self.cache),
                'cache_size_bytes': self.cache.volume(),
                'cache_size_mb': round(self.cache.volume() / 1024**2, 2),
                'max_size_mb': round(self.cache.size_limit / 1024**2, 2),
                'cache_directory': str(self.cache_dir),
                'eviction_policy': 'least-recently-used',
                'default_ttl_seconds': self.default_ttl,
                'compression_enabled': self.enable_compression
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def cleanup_expired(self) -> int:
        """
        Manually clean up expired entries (usually handled automatically).
        
        Returns:
            Number of entries removed
        """
        try:
            initial_count = len(self.cache)
            self.cache.expire()
            final_count = len(self.cache)
            
            removed_count = initial_count - final_count
            logger.info(f"Cleaned up {removed_count} expired cache entries")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0
    
    def close(self):
        """Close the cache connection."""
        try:
            self.cache.close()
            logger.debug("Cache connection closed")
        except Exception as e:
            logger.error(f"Error closing cache: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class CachedRenewablesNinjaAPI:
    """
    Wrapper for RenewablesNinjaAPI with caching support.
    
    Provides transparent caching of API responses with configurable TTL
    and automatic cache invalidation strategies.
    """
    
    def __init__(self, 
                 api_client,
                 cache_ttl: int = 86400,  # 24 hours
                 cache_dir: str = "cache/renewables_ninja"):
        """
        Initialize cached API wrapper.
        
        Args:
            api_client: Instance of RenewablesNinjaAPI
            cache_ttl: Cache time-to-live in seconds (24 hours default)
            cache_dir: Cache directory path
        """
        self.api_client = api_client
        self.cache = APICache(cache_dir=cache_dir, default_ttl=cache_ttl)
        
        logger.info(f"Initialized cached Renewables Ninja API with {cache_ttl}s TTL")
    
    def fetch_solar_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch solar data with caching.
        
        Args:
            **kwargs: Solar data parameters
            
        Returns:
            Solar capacity factor DataFrame
        """
        endpoint = "solar_pv"
        cache_key_params = {k: str(v) for k, v in kwargs.items()}
        
        # Try cache first
        cached_data = self.cache.get(endpoint, cache_key_params, prefix="solar")
        
        if cached_data is not None:
            logger.info("Using cached solar data")
            return cached_data
        
        # Fetch from API if not cached
        logger.info("Fetching fresh solar data from API")
        fresh_data = self.api_client.fetch_solar_data(**kwargs)
        
        # Cache the response
        self.cache.set(endpoint, cache_key_params, fresh_data, prefix="solar")
        
        return fresh_data
    
    def fetch_wind_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch wind data with caching.
        
        Args:
            **kwargs: Wind data parameters
            
        Returns:
            Wind capacity factor DataFrame
        """
        endpoint = "wind"
        cache_key_params = {k: str(v) for k, v in kwargs.items()}
        
        # Try cache first
        cached_data = self.cache.get(endpoint, cache_key_params, prefix="wind")
        
        if cached_data is not None:
            logger.info("Using cached wind data")
            return cached_data
        
        # Fetch from API if not cached
        logger.info("Fetching fresh wind data from API")
        fresh_data = self.api_client.fetch_wind_data(**kwargs)
        
        # Cache the response
        self.cache.set(endpoint, cache_key_params, fresh_data, prefix="wind")
        
        return fresh_data
    
    def fetch_hybrid_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch hybrid system data with caching.
        
        Args:
            **kwargs: Hybrid system parameters
            
        Returns:
            Dictionary with solar and wind DataFrames
        """
        endpoint = "hybrid"
        cache_key_params = {k: str(v) for k, v in kwargs.items()}
        
        # Try cache first
        cached_data = self.cache.get(endpoint, cache_key_params, prefix="hybrid")
        
        if cached_data is not None:
            logger.info("Using cached hybrid data")
            return cached_data
        
        # Fetch from API if not cached
        logger.info("Fetching fresh hybrid data from API")
        fresh_data = self.api_client.fetch_hybrid_data(**kwargs)
        
        # Cache the response
        self.cache.set(endpoint, cache_key_params, fresh_data, prefix="hybrid")
        
        return fresh_data
    
    def invalidate_location_cache(self, lat: float, lon: float):
        """
        Invalidate all cached data for a specific location.
        
        Args:
            lat: Latitude
            lon: Longitude
        """
        location_pattern = f"lat_{lat}_lon_{lon}"
        
        solar_removed = self.cache.invalidate_pattern(f"solar:*{location_pattern}*")
        wind_removed = self.cache.invalidate_pattern(f"wind:*{location_pattern}*") 
        hybrid_removed = self.cache.invalidate_pattern(f"hybrid:*{location_pattern}*")
        
        total_removed = solar_removed + wind_removed + hybrid_removed
        logger.info(f"Invalidated {total_removed} cache entries for location ({lat}, {lon})")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear_all()
    
    def close(self):
        """Close cache connection."""
        self.cache.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Integration with cache strategies
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cache_strategies import SmartCacheManager


def create_smart_cached_api(api_client, cache_dir: str = "cache/smart_renewables_ninja"):
    """
    Factory function to create a smart cached API client.
    
    Args:
        api_client: RenewablesNinjaAPI instance
        cache_dir: Cache directory path
        
    Returns:
        SmartCacheManager instance with integrated API client
    """
    from .cache_strategies import SmartCacheManager
    
    smart_manager = SmartCacheManager(cache_dir=cache_dir)
    
    # Create cached API wrapper
    cached_api = CachedRenewablesNinjaAPI(
        api_client=api_client,
        cache_ttl=86400,  # 24 hours
        cache_dir=cache_dir
    )
    
    # Initialize cache warming for common locations
    smart_manager.initialize_cache_warming(api_client)
    
    # Return both smart manager and cached API
    return smart_manager, cached_api