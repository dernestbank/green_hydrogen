"""
PEM Electrolyser Model

Updated PEM electrolyser implementation with advanced features
including variable SEC modeling and stack lifecycle management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import yaml

from .electrolyser_pem_advanced import AdvancedPEMModel

logger = logging.getLogger(__name__)


class ElectrolyserPEM:
    """
    PEM Electrolyser Model - High-level interface

    This class provides a simplified interface to the advanced PEM model,
    maintaining backward compatibility with existing code.
    """

    def __init__(self, **kwargs):
        """
        Initialize PEM electrolyser with configuration.

        Args:
            **kwargs: Configuration parameters (backward compatibility)
                     or 'config_path' to load from YAML
        """
        self.config = kwargs.copy()

        # Load configuration if specified
        if 'config_path' in kwargs:
            config_path = Path(kwargs['config_path'])
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    self.config.update(loaded_config)
            else:
                logger.warning(f"Config file not found: {config_path}")

        # Create advanced PEM model
        self.advanced_model = AdvancedPEMModel(self.config)

        # Legacy interface parameters
        self.electrolyser_type = "PEM"
        self.min_load = self.advanced_model.min_load_percent
        self.max_load = self.advanced_model.max_load_percent

    def calculate_efficiency(self, power_kw: Optional[float] = None) -> float:
        """
        Calculate current efficiency.

        Args:
            power_kw: Power input (optional)

        Returns:
            Efficiency as percentage
        """
        if power_kw is None:
            return self.advanced_model.metrics.efficiency_achieved * 100

        # Calculate efficiency for given power
        load_fraction = power_kw / (self.advanced_model.ref_capacity_mw * 1000)
        sec = self.advanced_model._calculate_sec_variable(load_fraction)
        _, efficiency = self.advanced_model._calculate_hydrogen_production(power_kw, sec)

        return efficiency * 100

    def calculate_hydrogen_production(self, power_kw: float, time_hours: float = 1.0) -> float:
        """
        Calculate hydrogen production for given power input.

        Args:
            power_kw: Power input in kW
            time_hours: Operating time in hours

        Returns:
            Hydrogen production in kg
        """
        # Calculate load fraction
        nominal_power = self.advanced_model.ref_capacity_mw * 1000
        load_fraction = min(power_kw / nominal_power, self.advanced_model.max_load_percent)

        # Use advanced model to calculate
        result = self.advanced_model.operate_at_load(load_fraction, time_hours)

        if result['success']:
            return result['hydrogen_produced_kg']
        else:
            logger.warning(f"Operation failed: {result.get('error', 'Unknown error')}")
            return 0.0

    def set_operating_temperature(self, temperature_celsius: float):
        """
        Set operating temperature.

        Args:
            temperature_celsius: Temperature in Celsius
        """
        self.advanced_model.state.temperature_k = temperature_celsius + 273.15

    def get_status(self) -> Dict[str, Union[float, int, str]]:
        """
        Get current electrolyser status.

        Returns:
            Status dictionary
        """
        return {
            'type': 'PEM',
            'operating_hours': self.advanced_model.state.operating_hours,
            'stack_cycles': self.advanced_model.state.stack_cycles,
            'efficiency': self.advanced_model.metrics.efficiency_achieved * 100,
            'degradation_rate': self.advanced_model.state.degradation_rate,
            'last_overload': self.advanced_model.state.last_overload_time.isoformat()
                           if self.advanced_model.state.last_overload_time else None,
            'total_hydrogen_produced_kg': self.advanced_model.metrics.hydrogen_produced_kg,
            'total_electricity_consumed_kwh': self.advanced_model.metrics.electricity_consumed_kwh,
            'total_water_consumed_liters': self.advanced_model.metrics.water_consumed_liters
        }

    def reset(self):
        """Reset electrolyser state."""
        self.advanced_model.reset_state()
        logger.info("PEM electrolyser reset")

    # Advanced features exposed through simple interface
    def enable_overload_mode(self, enabled: bool = True):
        """
        Enable/disable overload operation mode.

        Args:
            enabled: Whether to enable overload mode
        """
        if enabled:
            self.advanced_model.max_load_percent = self.advanced_model.overload_capacity_percent
            logger.info("PEM overload mode enabled")
        else:
            self.advanced_model.max_load_percent = 1.0
            logger.info("PEM overload mode disabled")

    def get_optimal_load_profile(self, power_profile: pd.Series) -> pd.DataFrame:
        """
        Calculate optimal operating profile for given power availability.

        Args:
            power_profile: Available power time series

        Returns:
            Optimal operating schedule
        """
        return self.advanced_model.calculate_optimal_operating_points(power_profile)

    def get_performance_metrics(self) -> Dict[str, Union[float, int]]:
        """Get detailed performance metrics."""
        return self.advanced_model.get_performance_metrics()

    # Backward compatibility methods
    def calculate_sec(self, load_fraction: float) -> float:
        """
        Calculate SEC at given load fraction.

        Args:
            load_fraction: Load as fraction of nominal capacity

        Returns:
            Specific energy consumption in kWh/Nm³
        """
        return self.advanced_model._calculate_sec_variable(load_fraction)

    def check_operational_limits(self, load_fraction: float) -> Tuple[bool, Optional[str]]:
        """
        Check if operation is within limits.

        Args:
            load_fraction: Load fraction to check

        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        return self.advanced_model._can_operate_at_load(load_fraction)


# Factory function
def create_pem_electrolyser(config_params: Optional[Dict] = None) -> ElectrolyserPEM:
    """
    Factory function to create PEM electrolyser with default configuration.

    Args:
        config_params: Optional configuration overrides

    Returns:
        Configured PEM electrolyser
    """
    # Load default configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            config_params = config_params or {}
            config.update(config_params)
    else:
        # Fallback configuration
        config = config_params or {}
        config.setdefault('pem', {})
        config['pem'].update({
            'elec_min_load': 10,
            'elec_overload': 120,
            'elec_overload_recharge': 4,
            'spec_consumption': 4.7,
            'stack_lifetime': 60000,
            'electrolyser_capex': 1000,
            'electrolyser_om': 4,
            'water_needs': 10
        })
        config.setdefault('elec_reference_capacity', 10)

    return ElectrolyserPEM(**config)
