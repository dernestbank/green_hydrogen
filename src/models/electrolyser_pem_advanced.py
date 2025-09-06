"""
Advanced PEM Electrolyser Model

Comprehensive PEM electrolyser model with electrochemical physics,
variable SEC modeling, overload handling, and stack lifecycle management.
Based on the legacy model but extended with modern engineering practices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ElectrolyserState:
    """Current operating state of the electrolyser."""
    operating_hours: float = 0.0
    stack_cycles: int = 0
    degradation_rate: float = 0.0
    current_efficiency: float = 1.0
    temperature_k: float = 333.15
    pressure_anode_atm: float = 1.0
    pressure_cathode_atm: float = 13.6
    last_overload_time: Optional[datetime] = None


@dataclass
class ElectrolyserMetrics:
    """Performance metrics for electrolyser operation."""
    hydrogen_produced_kg: float = 0.0
    electricity_consumed_kwh: float = 0.0
    water_consumed_liters: float = 0.0
    operating_time_hours: float = 0.0
    efficiency_achieved: float = 0.0
    stack_replacements: int = 0


@dataclass
class SECPolynomial:
    """Polynomial coefficients for Specific Energy Consumption."""
    coefficients: List[float]  # [a3, a2, a1, a0] for ax^3 + bx^2 + cx + d
    load_min: float = 0.1
    load_max: float = 1.2
    valid_temperature_range_k: Tuple[float, float] = (313.15, 353.15)


class AdvancedPEMModel:
    """
    Advanced PEM Electrolyser Model

    Features:
    - Electrochemical physics-based voltage calculations
    - Variable SEC with polynomial curves
    - Overload handling with recovery modeling
    - Stack lifecycle management
    - Operating point optimization
    - Degradation modeling
    - Temperature and pressure effects
    """

    # Physical constants
    FARADAY = 96485.0    # C/mol
    GAS_CONSTANT = 8.314  # J/mol/K
    IDEAL_GAS_CONSTANT = 0.0821  # L*atm/mol/K

    def __init__(self, config_params):
        """
        Initialize PEM electrolyser model.

        Args:
            config_params: Configuration parameters from YAML
        """
        # Extract PEM-specific parameters
        pem_config = config_params.get('pem', {})

        # Basic parameters
        self.min_load_percent = pem_config.get('elec_min_load', 10) / 100
        self.max_load_percent = 1.0  # Normal operating range (overload disabled by default)
        self.overload_capacity_percent = pem_config.get('elec_overload', 120) / 100
        self.overload_recovery_hours = pem_config.get('elec_overload_recharge', 4)
        self.nominal_sec_kwh_per_nm3 = pem_config.get('spec_consumption', 4.7)
        self.electrolyser_capex_usd_per_kw = pem_config.get('electrolyser_capex', 1000)
        self.electrolyser_om_percent = pem_config.get('electrolyser_om', 4) / 100
        self.stack_lifetime_hours = pem_config.get('stack_lifetime', 60000)
        self.stack_replacement_cost_percent = pem_config.get('stack_cost_percent', 40) / 100
        self.ref_capacity_mw = config_params.get('elec_reference_capacity', 10)
        self.water_needs_liters_per_kg_h2 = pem_config.get('water_needs', 10)

        # Electrochemical parameters (from legacy model)
        self.electrochemical_params = self._get_default_electrochemical_params()

        # Operating state
        self.state = ElectrolyserState()

        # Variable SEC polynomial (will be initialized)
        self.sec_polynomial = None

        # Metrics tracking
        self.metrics = ElectrolyserMetrics()

        # Initialize variable SEC model
        self._initialize_sec_polynomial()

        logger.info(".1f")

    def _get_default_electrochemical_params(self) -> Dict[str, float]:
        """Get default electrochemical parameters for PEM model."""
        return {
            'T': 333.15,           # K (60°C)
            'Pa': 1.0,             # atm Anode pressure
            'Pc': 13.6,            # atm Cathode pressure
            'j0_a': 1.65e-8,       # A/cm² Exchange current density Anode
            'j0_c': 9e-2,          # A/cm² Exchange current density Cathode
            'alpha_a': 2.0,        # Charge transfer coefficient anode
            'alpha_c': 0.5,        # Charge transfer coefficient cathode
            'd_m': 178e-4,         # cm Membrane thickness (178 µm)
            'sigma_m_coeff': (0.005139*22 - 0.00326),  # Pre-exponential for membrane
            'j_lim': 3.0,          # A/cm² Limiting current density
            'd_e': 200e-4,         # cm Electrode thickness
            'rho_e_a': 5e-3,       # Ω·cm Anode electrode resistivity
            'rho_e_c': 8e-2,       # Ω·cm Cathode electrode resistivity
        }

    def _initialize_sec_polynomial(self):
        """
        Initialize variable SEC polynomial based on operating points.

        The polynomial models SEC as a function of load fraction:
        SEC(load) = a3 * load^3 + a2 * load^2 + a1 * load + a0

        Where:
        - a0 is the no-load SEC (theoretical minimum)
        - Load affects efficiency through various losses
        """
        # Define polynomial based on typical PEM performance characteristics
        # SEC typically increases at low loads due to fixed losses and decreases slightly at high loads
        min_load = self.min_load_percent
        nominal_load = 1.0
        overload_load = self.overload_capacity_percent

        # Base SEC at different operating points
        sec_nominal = self.nominal_sec_kwh_per_nm3
        sec_min_load = sec_nominal * 1.8  # Higher at low load due to parasitic losses
        sec_overload = sec_nominal * 0.95  # Slightly better at overload due to efficiency gains

        # Fit polynomial: SEC(load) = a3*x^3 + a2*x^2 + a1*x + a0
        # Using points: (min_load, sec_min_load), (1.0, sec_nominal), (overload_load, sec_overload)

        # Solve for coefficients using three points
        A = np.array([
            [min_load**3, min_load**2, min_load, 1],
            [nominal_load**3, nominal_load**2, nominal_load, 1],
            [overload_load**3, overload_load**2, overload_load, 1]
        ])
        b = np.array([sec_min_load, sec_nominal, sec_overload])

        try:
            coeffs = np.linalg.solve(A, b)
            self.sec_polynomial = SECPolynomial(
                coefficients=coeffs.tolist(),
                load_min=min_load,
                load_max=overload_load
            )
            logger.info(f"Variable SEC polynomial initialized: {coeffs}")
        except np.linalg.LinAlgError:
            # Fallback to constant SEC
            logger.warning("Could not fit SEC polynomial, using constant SEC")
            self.sec_polynomial = None

    def _calculate_sec_variable(self, load_fraction: float) -> float:
        """
        Calculate variable SEC using polynomial model.

        Args:
            load_fraction: Load as fraction of nominal capacity (0-1.2)

        Returns:
            Specific energy consumption in kWh/Nm³
        """
        if self.sec_polynomial is None:
            return self.nominal_sec_kwh_per_nm3

        # Clip load to valid range
        load_clipped = np.clip(load_fraction,
                              self.sec_polynomial.load_min,
                              self.sec_polynomial.load_max)

        # Evaluate polynomial
        coeffs = self.sec_polynomial.coefficients
        sec = (coeffs[0] * load_clipped**3 +
               coeffs[1] * load_clipped**2 +
               coeffs[2] * load_clipped +
               coeffs[3])

        # Apply degradation factor
        degradation_factor = 1.0 + (self.state.degradation_rate * self.state.operating_hours / 8760)

        return sec * degradation_factor

    def calculate_electrochemical_voltage(self, current_density: float,
                                        temperature_k: Optional[float] = None) -> float:
        """
        Calculate cell voltage using electrochemical model.

        Args:
            current_density: Current density in A/cm²
            temperature_k: Operating temperature in Kelvin

        Returns:
            Cell voltage in Volts
        """
        params = self.electrochemical_params.copy()
        if temperature_k:
            params['T'] = temperature_k

        # Reversible voltage (Nernst equation)
        V_rev = self._calculate_reversible_voltage(params)

        # Activation overpotentials
        V_act = self._calculate_activation_overpotential(current_density, params)

        # Diffusion overpotential
        V_diff = self._calculate_diffusion_overpotential(current_density, params)

        # Ohmic overpotential
        V_ohm = self._calculate_ohmic_overpotential(current_density, params)

        total_voltage = V_rev + V_act + V_diff + V_ohm

        # Apply degradation factor
        degradation_penalty = self.state.degradation_rate * 0.001  # Small voltage increase
        total_voltage *= (1.0 + degradation_penalty)

        return total_voltage

    def _calculate_reversible_voltage(self, params: Dict[str, float]) -> float:
        """Calculate theoretical reversible voltage."""
        T = params['T']
        V0 = 1.229 - 0.0009 * (T - 298.15)  # Temperature correction

        # Nernst equation correction (simplified - assumes ideal activities)
        # For simplicity, we use the temperature-corrected value
        return V0

    def _calculate_activation_overpotential(self, j: float, params: Dict[str, float]) -> float:
        """Calculate activation overpotential."""
        # Anode activation
        T = params['T']
        V_act_a = (self.GAS_CONSTANT * T) / (params['alpha_a'] * self.FARADAY) * \
                 np.arcsinh(j / (2 * params['j0_a']))

        # Cathode activation
        V_act_c = (self.GAS_CONSTANT * T) / (params['alpha_c'] * self.FARADAY) * \
                 np.arcsinh(j / (2 * params['j0_c']))

        return V_act_a + V_act_c

    def _calculate_diffusion_overpotential(self, j: float, params: Dict[str, float]) -> float:
        """Calculate diffusion overpotential."""
        return - (self.GAS_CONSTANT * params['T']) / (2 * self.FARADAY) * \
               np.log(1 - j / params['j_lim'])

    def _calculate_ohmic_overpotential(self, j: float, params: Dict[str, float]) -> float:
        """Calculate ohmic overpotential."""
        # Membrane ionic conductivity
        sigma_m = self._calculate_membrane_conductivity(params['T'])

        # Total resistance
        R_total = params['d_m'] / sigma_m  # Membrane resistance

        # Electrode resistances
        R_electrode_a = params['rho_e_a'] * params['d_e']
        R_electrode_c = params['rho_e_c'] * params['d_e']

        total_resistance = R_total + R_electrode_a + R_electrode_c

        return j * total_resistance

    def _calculate_membrane_conductivity(self, temperature_k: float) -> float:
        """Calculate membrane conductivity based on temperature."""
        # Arrhenius relationship for membrane conductivity
        coeff = self.electrochemical_params['sigma_m_coeff']
        return coeff * np.exp(1268 * (1/303 - 1/temperature_k))

    def _calculate_current_density_from_power(self, power_kw: float, voltage_per_cell: float) -> float:
        """
        Calculate current density from power input.

        Args:
            power_kw: Power in kW
            voltage_per_cell: Operating voltage per cell

        Returns:
            Current density in A/cm²
        """
        # Assumes 100 cm² active area per cell (typical for PEM)
        active_area_cm2 = 100.0

        # Convert power to current
        power_watts = power_kw * 1000
        current_per_cell_a = power_watts / voltage_per_cell

        # Current density
        current_density = current_per_cell_a / active_area_cm2

        return current_density

    def _can_operate_at_load(self, load_fraction: float) -> Tuple[bool, Optional[str]]:
        """
        Check if electrolyser can operate at given load.

        Args:
            load_fraction: Load as fraction of nominal capacity

        Returns:
            Tuple of (can_operate, reason_if_not)
        """
        if load_fraction < self.min_load_percent:
            return False, f"Load {load_fraction:.1%} below minimum {self.min_load_percent:.1%}"

        if load_fraction > self.max_load_percent:
            return False, f"Load {load_fraction:.1%} exceeds maximum {self.max_load_percent:.1%}"

        # Check overload recovery time
        if (self.state.last_overload_time and
            load_fraction > 1.0 and
            (datetime.now() - self.state.last_overload_time).total_seconds() / 3600 <
            self.overload_recovery_hours):
            return False, f"Overload recovery required ({self.overload_recovery_hours}h)"

        return True, None

    def _calculate_hydrogen_production(self, power_kw: float, sec: float) -> Tuple[float, float]:
        """
        Calculate hydrogen production from power input.

        Args:
            power_kw: Power input in kW
            sec: Specific energy consumption in kWh/Nm³

        Returns:
            Tuple of (hydrogen_kg, electricity_efficiency)
        """
        # Power in kWh
        power_kwh = power_kw

        # Volume of hydrogen produced (Nm³)
        hydrogen_nm3 = power_kwh / sec

        # Convert to mass (kg) - 1 Nm³ H2 = 0.089 kg at STP
        hydrogen_kg = hydrogen_nm3 * 0.089

        # Efficiency (higher heating value of H2 is ~3.55 kWh/Nm³)
        theoretical_power = hydrogen_nm3 * 3.55  # Theoretical power for produced H2
        efficiency = theoretical_power / power_kwh

        return hydrogen_kg, efficiency

    def operate_at_load(self, load_fraction: float, time_hours: float = 1.0) -> Dict[str, Union[float, str]]:
        """
        Operate electrolyser at specified load for time period.

        Args:
            load_fraction: Load as fraction of nominal capacity (0-1.2)
            time_hours: Operating time in hours

        Returns:
            Operation results dictionary
        """
        # Check if operation is allowed
        can_operate, reason = self._can_operate_at_load(load_fraction)
        if not can_operate:
            return {
                'success': False,
                'error': reason if reason else "Unknown error",
                'hydrogen_produced_kg': 0.0,
                'power_consumed_kw': 0.0,
                'efficiency': 0.0
            }

        # Calculate SEC at this load
        sec = self._calculate_sec_variable(load_fraction)

        # Calculate power consumption
        nominal_power_kw = self.ref_capacity_mw * 1000
        power_kw = nominal_power_kw * load_fraction

        # Calculate hydrogen production
        hydrogen_kg, efficiency = self._calculate_hydrogen_production(power_kw, sec)

        # Scale for time period
        hydrogen_total = hydrogen_kg * time_hours
        power_total = power_kw * time_hours
        water_consumed = hydrogen_total * self.water_needs_liters_per_kg_h2

        # Update state
        self.state.operating_hours += time_hours
        self.metrics.hydrogen_produced_kg += hydrogen_total
        self.metrics.electricity_consumed_kwh += power_total
        self.metrics.water_consumed_liters += water_consumed
        self.metrics.operating_time_hours += time_hours

        # Check for overload
        if load_fraction > 1.0:
            self.state.last_overload_time = datetime.now()

        # Update efficiency metric
        self.metrics.efficiency_achieved = (
            (self.metrics.hydrogen_produced_kg * 3.55) / self.metrics.electricity_consumed_kwh
            if self.metrics.electricity_consumed_kwh > 0 else 0
        )

        # Check for stack replacement
        if (self.state.operating_hours - self.state.stack_cycles * self.stack_lifetime_hours) >= self.stack_lifetime_hours:
            self.state.stack_cycles += 1
            self.metrics.stack_replacements += 1
            logger.info(f"Stack replacement {self.state.stack_cycles} after {self.state.operating_hours:.0f} hours")

        # Update degradation (simplified model)
        age_factor = self.state.operating_hours / self.stack_lifetime_hours
        self.state.degradation_rate = min(0.2, age_factor * 0.05)  # Max 5% degradation per 20 stack lifetimes

        result = {
            'success': True,
            'load_fraction': load_fraction,
            'hydrogen_produced_kg': hydrogen_total,
            'power_consumed_kw': power_total,
            'sec_kwh_per_nm3': sec,
            'efficiency': efficiency,
            'water_consumed_liters': water_consumed,
            'operating_hours': self.state.operating_hours,
            'degradation_rate': self.state.degradation_rate,
            'stack_cycles': self.state.stack_cycles
        }

        logger.debug(".3f")
        return result

    def calculate_optimal_operating_points(self, power_profile: pd.Series) -> pd.DataFrame:
        """
        Calculate optimal operating points for given power profile.

        Args:
            power_profile: Time series of available power in kW

        Returns:
            DataFrame with optimal operating schedule
        """
        results = []

        for timestamp, power_kw in power_profile.items():
            # Calculate optimal load based on efficiency curves
            optimal_load = self._find_optimal_load_fraction(power_kw / (self.ref_capacity_mw * 1000))

            result = self.operate_at_load(optimal_load, time_hours=1.0)
            result['timestamp'] = timestamp
            result['available_power_kw'] = power_kw
            results.append(result)

        return pd.DataFrame(results)

    def _find_optimal_load_fraction(self, available_load_fraction: float) -> float:
        """
        Find optimal load fraction for efficiency.

        This is a simplified optimization - in practice, this would
        consider efficiency curves, operational constraints, etc.
        """
        # Clip to operational range
        min_load = self.min_load_percent
        available_load_clipped = np.clip(available_load_fraction, min_load, self.max_load_percent)

        # For now, return the available load (could be optimized further)
        return available_load_clipped

    def get_performance_metrics(self) -> Dict[str, Union[float, int]]:
        """Get current performance metrics."""
        return {
            'hydrogen_produced_total_kg': self.metrics.hydrogen_produced_kg,
            'electricity_consumed_total_kwh': self.metrics.electricity_consumed_kwh,
            'water_consumed_total_liters': self.metrics.water_consumed_liters,
            'operating_time_total_hours': self.metrics.operating_time_hours,
            'average_efficiency': self.metrics.efficiency_achieved,
            'current_degradation_rate': self.state.degradation_rate,
            'stack_cycles': self.state.stack_cycles,
            'stack_replacements': self.metrics.stack_replacements,
            'capacity_factor': (
                self.metrics.operating_time_hours *
                self.metrics.electricity_consumed_kwh /
                max(self.metrics.operating_time_hours, 1) /
                (self.ref_capacity_mw * 1000)
            ) if self.metrics.operating_time_hours > 0 else 0
        }

    def reset_state(self):
        """Reset electrolyser state and metrics."""
        self.state = ElectrolyserState()
        self.metrics = ElectrolyserMetrics()
        logger.info("PEM electrolyser state reset")


# Factory function for creating PEM model from config
def create_pem_model(config_loader) -> AdvancedPEMModel:
    """
    Create PEM electrolyser model from configuration.

    Args:
        config_loader: Configuration loader instance

    Returns:
        Configured PEM electrolyser model
    """
    config = config_loader.get_config()
    return AdvancedPEMModel(config)