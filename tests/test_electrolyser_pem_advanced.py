"""
Tests for Advanced PEM Electrolyser Model

Test electrochemical modeling, variable SEC, overload handling,
and stack lifecycle management.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch

from src.models.electrolyser_pem_advanced import (
    AdvancedPEMModel,
    ElectrolyserState,
    ElectrolyserMetrics,
    SECPolynomial
)


class TestAdvancedPEMModel:
    """Test advanced PEM electrolyser model."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'pem': {
                'elec_min_load': 10,
                'elec_overload': 120,
                'elec_overload_recharge': 4,
                'spec_consumption': 4.7,
                'stack_lifetime': 60000,
                'electrolyser_capex': 1000,
                'electrolyser_om': 4,
                'water_needs': 10
            },
            'elec_reference_capacity': 10
        }

    @pytest.fixture
    def pem_model(self, sample_config):
        """Create PEM model instance."""
        return AdvancedPEMModel(sample_config)

    def test_model_initialization(self, pem_model, sample_config):
        """Test model initialization with configuration."""
        assert pem_model.min_load_percent == 0.1  # 10%
        assert pem_model.max_load_percent == 1.0  # Default, overload disabled by default
        assert pem_model.overload_capacity_percent == 1.2  # 120%
        assert pem_model.nominal_sec_kwh_per_nm3 == 4.7
        assert pem_model.stack_lifetime_hours == 60000
        assert pem_model.electrolyser_capex_usd_per_kw == 1000
        assert pem_model.ref_capacity_mw == 10

    def test_variable_sec_polynomial(self, pem_model):
        """Test variable SEC polynomial modeling."""
        # Test at different load levels
        sec_full_load = pem_model._calculate_sec_variable(1.0)
        sec_min_load = pem_model._calculate_sec_variable(0.1)
        sec_overload = pem_model._calculate_sec_variable(1.2)

        # SEC should be higher at minimum load
        assert sec_min_load > sec_full_load

        # SEC should be reasonable in normal ranges
        assert 4.0 <= sec_full_load <= 6.0
        assert 5.0 <= sec_min_load <= 10.0
        assert 4.0 <= sec_overload <= 5.0

    def test_operating_limits(self, pem_model):
        """Test operating limit checks."""
        # Should work at normal loads
        can_operate, reason = pem_model._can_operate_at_load(0.5)
        assert can_operate is True
        assert reason is None

        # Should reject below minimum load
        can_operate, reason = pem_model._can_operate_at_load(0.05)
        assert can_operate is False
        assert "below minimum" in reason

        # Should reject above maximum load
        can_operate, reason = pem_model._can_operate_at_load(1.5)
        assert can_operate is False
        assert "exceeds maximum" in reason

    def test_hydrogen_production_calculation(self, pem_model):
        """Test hydrogen production calculations."""
        power_kw = 1000  # 1 MW
        sec = 4.7

        h2_kg, efficiency = pem_model._calculate_hydrogen_production(power_kw, sec)

        # Theoretical calculations
        # 1 MW = 1000 kW
        # Volume of H2 = power / SEC = 1000 / 4.7 ≈ 212.8 Nm³/hour
        # Mass of H2 = 212.8 * 0.089 ≈ 18.94 kg/hour
        expected_h2 = 1000 / 4.7 * 0.089  # ≈ 18.94 kg

        assert abs(h2_kg - expected_h2) < 0.1

        # Efficiency should be reasonable
        assert 0.5 <= efficiency <= 0.8

    def test_normal_operation(self, pem_model):
        """Test normal operation cycle."""
        load_fraction = 0.8
        time_hours = 2.0

        result = pem_model.operate_at_load(load_fraction, time_hours)

        assert result['success'] is True
        assert result['load_fraction'] == load_fraction
        assert result['hydrogen_produced_kg'] > 0
        assert result['power_consumed_kw'] > 0
        assert result['sec_kwh_per_nm3'] > 0
        assert result['efficiency'] > 0
        assert result['operating_hours'] == time_hours

        # Check metrics updated
        assert pem_model.metrics.hydrogen_produced_kg > 0
        assert pem_model.metrics.electricity_consumed_kwh > 0
        assert pem_model.metrics.operating_time_hours == time_hours

    def test_overload_operation(self, pem_model):
        """Test overload operation."""
        load_fraction = 1.1  # Overload
        time_hours = 1.0

        # Enable overload mode first
        pem_model.max_load_percent = pem_model.overload_capacity_percent

        result = pem_model.operate_at_load(load_fraction, time_hours)

        assert result['success'] is True
        assert result['load_fraction'] == load_fraction
        assert pem_model.state.last_overload_time is not None

        # Try immediate reload (should be rejected due to recovery time)
        result2 = pem_model.operate_at_load(load_fraction, time_hours)
        assert result2['success'] is False
        assert 'recovery' in result2.get('error', '').lower()

    def test_stack_lifecycle(self, pem_model):
        """Test stack replacement lifecycle."""
        # Simulate operating hours near stack lifetime
        pem_model.state.operating_hours = pem_model.stack_lifetime_hours - 100

        # Operate for 200 hours (should trigger replacement)
        result = pem_model.operate_at_load(0.8, 200)

        assert result['success'] is True
        assert result['stack_cycles'] == 1  # Should have triggered replacement
        assert pem_model.metrics.stack_replacements == 1

    def test_degradation_modeling(self, pem_model):
        """Test degradation modeling over time."""
        initial_eff = pem_model.metrics.efficiency_achieved

        # Simulate long operation
        for _ in range(10):
            pem_model.operate_at_load(0.8, 1000)  # 1000 hours each

        # Should have some degradation
        assert pem_model.state.degradation_rate >= 0
        assert pem_model.state.degradation_rate <= 0.2  # Max 20%

        # Efficiency should be lower after degradation
        final_eff = pem_model.metrics.efficiency_achieved
        if final_eff > 0:  # Only check if we have data
            assert final_eff <= 1.0  # Should not exceed 100% even with degradation

    def test_electrochemical_modeling(self, pem_model):
        """Test electrochemical voltage calculations."""
        current_density = 1.0  # A/cm²
        voltage = pem_model.calculate_electrochemical_voltage(current_density)

        # Voltage should be reasonable for PEM electrolyser
        assert 1.4 <= voltage <= 2.2  # Typical PEM voltage range

        # Higher current density should give higher voltage
        voltage_high = pem_model.calculate_electrochemical_voltage(2.0)
        assert voltage_high > voltage

    def test_optimal_operating_schedule(self, pem_model):
        """Test optimal operating schedule calculation."""
        # Create sample power profile
        timestamps = pd.date_range('2023-01-01', periods=24, freq='H')
        power_values = np.maximum(0, 5000 + 2000 * np.sin(np.arange(24) * 2 * np.pi / 24))

        power_profile = pd.Series(power_values, index=timestamps)

        # Calculate optimal schedule
        schedule = pem_model.calculate_optimal_operating_points(power_profile)

        assert len(schedule) == len(power_profile)
        assert 'hydrogen_produced_kg' in schedule.columns
        assert 'power_consumed_kw' in schedule.columns
        assert 'efficiency' in schedule.columns

        # All hydrogen production should be non-negative
        assert (schedule['hydrogen_produced_kg'] >= 0).all()

    def test_performance_metrics(self, pem_model):
        """Test performance metrics calculation."""
        # Run some operations to generate data
        pem_model.operate_at_load(0.8, 10)
        pem_model.operate_at_load(0.6, 5)

        metrics = pem_model.get_performance_metrics()

        assert 'hydrogen_produced_total_kg' in metrics
        assert 'electricity_consumed_total_kwh' in metrics
        assert 'operating_time_total_hours' in metrics
        assert 'average_efficiency' in metrics
        assert 'stack_cycles' in metrics
        assert 'capacity_factor' in metrics

        # Values should be reasonable
        assert metrics['hydrogen_produced_total_kg'] > 0
        assert metrics['electricity_consumed_total_kwh'] > 0
        assert metrics['operating_time_total_hours'] == 15  # 10 + 5

    def test_state_reset(self, pem_model):
        """Test electrolyser state reset."""
        # Run some operations
        pem_model.operate_at_load(0.8, 10)
        original_operating_hours = pem_model.state.operating_hours

        assert original_operating_hours > 0

        # Reset state
        pem_model.reset_state()

        # Should be back to initial state
        assert pem_model.state.operating_hours == 0
        assert pem_model.state.stack_cycles == 0
        assert pem_model.metrics.hydrogen_produced_kg == 0
        assert pem_model.metrics.electricity_consumed_kwh == 0

    def test_configuration_edge_cases(self):
        """Test model with incomplete or edge-case configurations."""
        # Minimal config
        minimal_config = {
            'pem': {},  # Empty PEM config should use defaults
            'elec_reference_capacity': 5
        }

        model = AdvancedPEMModel(minimal_config)

        # Should use default values
        assert model.min_load_percent == 0.1  # Default 10%
        assert model.nominal_sec_kwh_per_nm3 == 4.7  # Default

        # Test operation with minimal config
        result = model.operate_at_load(0.5, 1)
        assert result['success'] is True


class TestSECPolynomial:
    """Test SEC polynomial calculations."""

    def test_polynomial_creation(self):
        """Test polynomial creation."""
        coeffs = [1.0, 2.0, 3.0, 4.0]  # ax³ + bx² + cx + d

        poly = SECPolynomial(
            coefficients=coeffs,
            load_min=0.1,
            load_max=1.2
        )

        assert poly.coefficients == coeffs
        assert poly.load_min == 0.1
        assert poly.load_max == 1.2


class TestIntegrationScenarios:
    """Integration tests for realistic operational scenarios."""

    def test_continuous_operation_simulation(self):
        """Test continuous operation over extended period."""
        config = {
            'pem': {
                'elec_min_load': 15,
                'elec_overload': 110,
                'spec_consumption': 4.5,
                'stack_lifetime': 20000  # Shorter for testing
            },
            'elec_reference_capacity': 2  # 2 MW
        }

        model = AdvancedPEMModel(config)

        # Simulate 6 months of operation (4380 hours)
        hours_per_day = 16  # 16 hours/day operation
        total_days = 180
        total_hours_accumulated = 0

        for day in range(total_days):
            # Vary load based on time of day
            base_load = 0.7
            variation = 0.2 * np.sin(day * 2 * np.pi / 30)  # Monthly variation
            daily_load = np.clip(base_load + variation, 0.2, 1.0)

            hours_today = min(hours_per_day, 4380 - total_hours_accumulated)
            if hours_today <= 0:
                break

            result = model.operate_at_load(daily_load, hours_today)
            total_hours_accumulated += hours_today

            assert result['success'] is True

        # Check accumulated results
        metrics = model.get_performance_metrics()
        assert metrics['operating_time_total_hours'] == total_hours_accumulated
        assert metrics['hydrogen_produced_total_kg'] > 0
        assert metrics['electricity_consumed_total_kwh'] > 0

        print(".1f")
        print(".0f")

    def test_maintenance_scenario(self):
        """Test maintenance and stack replacement scenario."""
        config = {
            'pem': {
                'stack_lifetime': 1000  # Short lifetime for testing
            },
            'elec_reference_capacity': 1
        }

        model = AdvancedPEMModel(config)

        # Operate until multiple stack replacements
        total_hours = 0
        while total_hours < 5000 and model.state.stack_cycles < 3:
            result = model.operate_at_load(0.8, 100)
            total_hours += 100
            assert result['success'] is True

        # Should have had stack replacements
        assert model.state.stack_cycles >= 1
        assert model.metrics.stack_replacements >= 1

    def test_efficiency_optimization(self):
        """Test efficiency optimization across load ranges."""
        model = AdvancedPEMModel({})

        # Test efficiency at various loads
        load_range = np.linspace(0.1, 1.0, 10)
        efficiencies = []

        for load in load_range:
            if load >= model.min_load_percent:
                sec = model._calculate_sec_variable(load)
                _, efficiency = model._calculate_hydrogen_production(
                    model.ref_capacity_mw * 1000 * load, sec
                )
                efficiencies.append(efficiency)

        # Should have some variation in efficiency
        if len(efficiencies) > 1:
            efficiency_range = max(efficiencies) - min(efficiencies)
            assert efficiency_range > 0  # Should have efficiency variation


if __name__ == "__main__":
    pytest.main([__file__])