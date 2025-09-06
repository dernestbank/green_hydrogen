import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import interp1d, griddata
import logging
try:
    from geopy.distance import geodesic
except ImportError:
    # Fallback to haversine distance if geopy not available
    def geodesic(point1, point2):
        """Simple haversine distance calculation as fallback"""
        lat1, lon1 = point1
        lat2, lon2 = point2

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371  # Earth radius in km
import math

logger = logging.getLogger(__name__)

class LocationInterpolation:
    """Location interpolation for renewable energy data using various interpolation methods"""

    def __init__(self, interpolation_method: str = 'idw'):
        """
        Initialize location interpolation

        Args:
            interpolation_method: Default interpolation method ('idw', 'linear', 'cubic', 'nearest')
        """
        self.default_method = interpolation_method
        self.supported_methods = ['idw', 'linear', 'cubic', 'nearest', 'bilinear']

        # Earth's radius for great circle distance calculations
        self.earth_radius_km = 6371

    def interpolate_single_location(self, target_location: Tuple[float, float],
                                  reference_locations: List[Tuple[float, float, float]],
                                  values: List[float],
                                  method: Optional[str] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Interpolate value for a single target location

        Args:
            target_location: (latitude, longitude) of target point
            reference_locations: List of (lat, lon, value) tuples for reference points
            values: Values at reference locations (alternative to including in reference_locations)
            method: Interpolation method to use
            **kwargs: Additional parameters for interpolation methods

        Returns:
            Dictionary with interpolated value and metadata
        """
        method = method or self.default_method

        if method not in self.supported_methods:
            raise ValueError(f"Unsupported interpolation method: {method}")

        # Extract coordinates and values from reference locations
        coords = []
        ref_values = []

        for i, ref_loc in enumerate(reference_locations):
            if len(ref_loc) == 3:
                lat, lon, val = ref_loc
            elif len(ref_loc) == 2:
                lat, lon = ref_loc
                val = values[i] if i < len(values) else 0
            else:
                raise ValueError("Reference location must be (lat, lon, value) or (lat, lon) with separate values")

            coords.append([lat, lon])
            ref_values.append(val)

        coords = np.array(coords)
        ref_values = np.array(ref_values)

        if len(coords) < 3 and method in ['linear', 'cubic']:
            logger.warning(f"Using nearest neighbor: insufficient points ({len(coords)}) for {method} interpolation")
            method = 'nearest'

        interpolated_value = None  # Initialize to avoid unbound variable

        try:
            if method == 'idw':
                interpolated_value = self._idw_interpolation(target_location, coords, ref_values,
                                                          **kwargs)
            elif method == 'linear':
                interpolated_value = self._linear_interpolation(target_location, coords, ref_values)
            elif method == 'cubic':
                interpolated_value = self._cubic_interpolation(target_location, coords, ref_values)
            elif method == 'nearest':
                interpolated_value = self._nearest_neighbor(target_location, coords, ref_values)
            elif method == 'bilinear':
                interpolated_value = self._bilinear_interpolation(target_location, coords, ref_values)

            # Calculate quality metrics
            quality = self._calculate_interpolation_quality(target_location, coords, ref_values)

            return {
                'interpolated_value': interpolated_value,
                'method': method,
                'quality_metrics': quality,
                'target_location': target_location,
                'reference_points_count': len(coords),
                'nearest_distance_km': self._calculate_nearest_distance(target_location, coords)
            }

        except Exception as e:
            logger.error(f"Interpolation failed for method {method}: {e}")
            raise ValueError(f"Interpolation failed: {e}")

    def interpolate_grid(self, target_grid: List[Tuple[float, float]],
                        reference_locations: List[Tuple[float, float, float]],
                        method: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Interpolate values for multiple target locations (grid)

        Args:
            target_grid: List of (lat, lon) tuples for interpolation grid
            reference_locations: List of (lat, lon, value) tuples for reference points
            method: Interpolation method
            **kwargs: Additional interpolation parameters

        Returns:
            Dictionary with interpolated grid and quality metrics
        """
        method = method or self.default_method
        results = []

        for target_location in target_grid:
            try:
                result = self.interpolate_single_location(target_location, reference_locations,
                                                        [], method, **kwargs)
                results.append({
                    'location': target_location,
                    'value': result['interpolated_value'],
                    'quality': result['quality_metrics']
                })
            except Exception as e:
                logger.warning(f"Failed to interpolate point {target_location}: {e}")
                results.append({
                    'location': target_location,
                    'value': None,
                    'quality': {'success': False, 'error': str(e)}
                })

        return {
            'method': method,
            'results': results,
            'grid_size': len(target_grid),
            'successful_interpolations': sum(1 for r in results if r['value'] is not None)
        }

    def _idw_interpolation(self, target: Tuple[float, float],
                          coords: np.ndarray,
                          values: np.ndarray,
                          power: float = 2) -> float:
        """Inverse Distance Weighting (IDW) interpolation"""
        distances = self._calculate_distances(target, coords)

        # Add small epsilon to avoid division by zero
        distances = np.where(distances == 0, 1e-10, distances)

        # Calculate weights
        weights = 1.0 / distances ** power
        weights_sum = np.sum(weights)

        # Normalize weights
        weights = weights / weights_sum

        # Calculate weighted average
        interpolated_value = np.sum(weights * values)

        return float(interpolated_value)

    def _linear_interpolation(self, target: Tuple[float, float],
                            coords: np.ndarray,
                            values: np.ndarray) -> float:
        """Linear interpolation using Delaunay triangulation"""
        try:
            # Create Delaunay triangulation
            tri = Delaunay(coords[:, :2])  # Use lat, lon only

            # Find which triangle contains the target point
            simplex = tri.find_simplex(target)

            if simplex != -1:
                # Target is inside triangulation, use barycentric coordinates
                vertices = tri.simplices[simplex]
                triangle_coords = coords[vertices]
                triangle_values = values[vertices]

                # Calculate barycentric coordinates
                barycentric = self._calculate_barycentric_coordinates(target, triangle_coords)

                # Interpolate using barycentric coordinates
                interpolated_value = np.sum(barycentric * triangle_values)
                return float(interpolated_value)
            else:
                # Target is outside triangulation, fall back to nearest neighbor
                return self._nearest_neighbor(target, coords, values)

        except Exception as e:
            logger.warning(f"Delaunay triangulation failed, using nearest neighbor: {e}")
            return self._nearest_neighbor(target, coords, values)

    def _cubic_interpolation(self, target: Tuple[float, float],
                           coords: np.ndarray,
                           values: np.ndarray) -> float:
        """Cubic interpolation (simplified as bi-cubic)"""
        # For simplicity, use IDW with cubic distance weighting
        return self._idw_interpolation(target, coords, values, power=3)

    def _nearest_neighbor(self, target: Tuple[float, float],
                         coords: np.ndarray,
                         values: np.ndarray) -> float:
        """Nearest neighbor interpolation"""
        distances = self._calculate_distances(target, coords)
        nearest_idx = np.argmin(distances)

        return float(values[nearest_idx])

    def _bilinear_interpolation(self, target: Tuple[float, float],
                               coords: np.ndarray,
                               values: np.ndarray) -> float:
        """
        Bilinear interpolation for structured grid data
        Note: This assumes a regular grid arrangement of reference points
        """
        try:
            # Try to detect grid structure
            unique_lats, unique_lons = np.unique(coords[:, 0]), np.unique(coords[:, 1])

            if len(unique_lats) > 1 and len(unique_lons) > 1:
                # Structured grid case
                interp_func = interp1d(unique_lons, values.reshape((len(unique_lats), -1)),
                                      kind='linear', bounds_error=False, fill_value='extrapolate')
                return float(interp_func(target[1]))
            else:
                # Not a structured grid, fall back to linear interpolation
                return self._linear_interpolation(target, coords, values)

        except Exception as e:
            logger.warning(f"Bilinear interpolation failed: {e}")
            return self._nearest_neighbor(target, coords, values)

    def _calculate_barycentric_coordinates(self, point: Tuple[float, float],
                                         triangle: np.ndarray) -> np.ndarray:
        """Calculate barycentric coordinates of a point within a triangle"""
        # Extract triangle vertices
        p1, p2, p3 = triangle[:3, :2]

        # Calculate area of the triangle
        area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

        # Avoid division by zero
        if abs(area) < 1e-10:
            return np.array([1.0, 0.0, 0.0])  # Degenerate triangle

        # Calculate barycentric coordinates
        lambda1 = ((p2[1] - p3[1]) * (point[0] - p3[0]) + (p3[0] - p2[0]) * (point[1] - p3[1])) / \
                 (2 * area)
        lambda2 = ((p3[1] - p1[1]) * (point[0] - p3[0]) + (p1[0] - p3[0]) * (point[1] - p3[1])) / \
                 (2 * area)
        lambda3 = 1 - lambda1 - lambda2

        return np.array([lambda1, lambda2, lambda3])

    def _calculate_distances(self, target: Tuple[float, float], coords: np.ndarray) -> np.ndarray:
        """Calculate great circle distances between target and reference points"""
        distances = []

        for coord in coords:
            # Use geodesic distance for accurate calculation
            distance = geodesic(target, (coord[0], coord[1])).km
            distances.append(distance)

        return np.array(distances)

    def _calculate_nearest_distance(self, target: Tuple[float, float], coords: np.ndarray) -> float:
        """Calculate distance to nearest reference point"""
        distances = self._calculate_distances(target, coords)
        return float(np.min(distances))

    def _calculate_interpolation_quality(self, target: Tuple[float, float],
                                       coords: np.ndarray,
                                       values: np.ndarray) -> Dict[str, Any]:
        """Calculate quality metrics for interpolation"""
        # Number of reference points
        n_points = len(coords)

        # Distance to nearest point
        nearest_distance = self._calculate_nearest_distance(target, coords)

        # Average distance to all points
        avg_distance = np.mean(self._calculate_distances(target, coords))

        # Spatial distribution quality (coefficient of variation of distances)
        distances = self._calculate_distances(target, coords)
        cv_distances = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0

        # Triangle quality if sufficient points
        triangle_quality = None
        if n_points >= 3:
            try:
                # Calculate quality of triangulation around target point
                tri = Delaunay(coords[:, :2])
                simplex = tri.find_simplex(target)

                if simplex != -1:
                    vertices = tri.simplices[simplex]
                    triangle_coords = coords[vertices]

                    # Calculate triangle area
                    p1, p2, p3 = triangle_coords[:3, :2]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                                    (p3[0] - p1[0]) * (p2[1] - p1[1]))

                    # Calculate quality as area / perimeter ratio
                    perimeter = (self._distance(p1, p2) + self._distance(p2, p3) +
                               self._distance(p3, p1))
                    triangle_quality = area / perimeter if perimeter > 0 else 0
                else:
                    triangle_quality = 0
            except Exception:
                triangle_quality = 0

        # Overall quality score (0-1, higher is better)
        quality_score = 0
        quality_score += min(n_points / 10, 1) * 0.3          # Prefer more points
        quality_score += (1 - min(cv_distances, 1)) * 0.3     # Prefer even distribution
        quality_score += (1 - min(nearest_distance / 10, 1)) * 0.3  # Prefer closer points
        quality_score += (triangle_quality or 0) * 0.1         # Prefer good triangulation

        return {
            'nearest_distance_km': nearest_distance,
            'average_distance_km': avg_distance,
            'coefficient_of_variation': cv_distances,
            'reference_points_count': n_points,
            'triangle_quality': triangle_quality,
            'overall_quality_score': quality_score,
            'confidence_level': self._get_confidence_level(quality_score)
        }

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points (for quality metrics)"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _get_confidence_level(self, quality_score: float) -> str:
        """Convert quality score to confidence level"""
        if quality_score >= 0.8:
            return "Very High"
        elif quality_score >= 0.6:
            return "High"
        elif quality_score >= 0.4:
            return "Medium"
        elif quality_score >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def optimize_interpolation_method(self, target: Tuple[float, float],
                                    coords: np.ndarray,
                                    values: np.ndarray,
                                    test_sample_size: int = 100) -> Dict[str, Any]:
        """
        Optimize interpolation method by testing accuracy on a sample

        Args:
            target: Target location
            coords: Reference coordinates
            values: Reference values
            test_sample_size: Number of reference points to use for optimization

        Returns:
            Best interpolation method and its estimated accuracy
        """
        if len(coords) < test_sample_size:
            test_sample_size = max(len(coords) // 2, 3)

        # Leave one out cross-validation
        errors = {method: [] for method in self.supported_methods}

        for i in range(len(coords)):
            # Remove one point for testing
            test_point = coords[i]
            test_value = values[i]
            train_coords = np.delete(coords, i, axis=0)
            train_values = np.delete(values, i, axis=0)

            # Test each method
            for method in self.supported_methods:
                try:
                    predicted_value = self.interpolate_single_location(
                        (test_point[0], test_point[1]),
                        [(c[0], c[1], v) for c, v in zip(train_coords, train_values)],
                        [],
                        method=method
                    )['interpolated_value']

                    error = abs(predicted_value - test_value)
                    errors[method].append(error)

                except Exception as e:
                    logger.debug(f"Method {method} failed for point {i}: {e}")
                    errors[method].append(float('inf'))

        # Calculate mean errors and select best method
        method_scores = {}
        for method, method_errors in errors.items():
            if method_errors:
                finite_errors = [e for e in method_errors if e != float('inf')]
                if finite_errors:
                    method_scores[method] = np.mean(finite_errors)
                else:
                    method_scores[method] = float('inf')
            else:
                method_scores[method] = float('inf')

        # Find best method
        best_method = min(method_scores, key=method_scores.get)
        best_score = method_scores[best_method]

        return {
            'recommended_method': best_method,
            'estimated_error': best_score,
            'method_scores': method_scores,
            'sample_size': test_sample_size,
            'confidence': len([e for e in errors[best_method] if e != float('inf')]) / len(errors[best_method])
        }

    def batch_interpolate_locations(self, locations: List[Tuple[float, float]],
                                  data_source: Any,
                                  method: Optional[str] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Batch interpolate multiple locations

        Args:
            locations: List of (lat, lon) tuples to interpolate
            data_source: Source of reference data (could be database, file, etc.)
            method: Interpolation method
            **kwargs: Additional parameters

        Returns:
            Batch interpolation results
        """
        # This would integrate with the data source to get reference locations
        # For now, return a placeholder structure

        return {
            'method': method or self.default_method,
            'locations_count': len(locations),
            'status': 'not_implemented',
            'message': 'Batch interpolation requires integration with data source'
        }