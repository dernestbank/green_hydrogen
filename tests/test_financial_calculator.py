"""
Tests for FinancialCalculator module.
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.financial_calculator import FinancialCalculator, create_financial_calculator


class TestFinancialCalculator:
    """Test suite for FinancialCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a FinancialCalculator instance for testing."""
        return FinancialCalculator(discount_rate=0.04, project_life=20)

    @pytest.fixture
    def sample_operating_outputs(self):
        """Sample operational outputs from hydrogen model."""
        return {
            "Generator Capacity Factor": 0.25,
            "Time Electrolyser is at its Rated Capacity": 0.15,
            "Total Time Electrolyser is Operating": 0.85,
            "Achieved Electrolyser Capacity Factor": 0.30,
            "Energy in to Electrolyser [MWh/yr]": 52560,
            "Surplus Energy [MWh/yr]": 14640,
            "Hydrogen Output for Fixed Operation [t/yr]": 1000,
            "Hydrogen Output for Variable Operation [t/yr]": 1050
        }

    @pytest.fixture
    def sample_cost_params(self):
        """Sample cost parameters."""
        return {
            'gen_capex_per_mw': 1000,
            'electrolyser_capex': 1000,
            'elec_capacity': 10,
            'solar_capacity': 10,
            'wind_capacity': 0,
            'electrolyser_capacity': 10,
            'gen_opex_per_mw': 15000,
            'electrolyser_om': 45,
            'water_usage': 50,
            'water_cost': 5,
            'battery_capacity': 2,
            'battery_duration': 4,
            'battery_capex_per_mwh': 400,
            'battery_om_per_mw': 10000,
            'stack_lifetime': 60000,
            'stack_replacement_cost': 320,
            'ppa_price': 0,
            'spot_price': 40
        }

    def test_initialization(self, calculator):
        """Test FinancialCalculator initialization."""
        assert calculator.discount_rate == 0.04
        assert calculator.project_life == 20
        assert calculator.cash_flows == {}

    def test_create_factory_function(self):
        """Test factory function for creating FinancialCalculator."""
        calc = create_financial_calculator(discount_rate=0.05, project_life=25)
        assert calc.discount_rate == 0.05
        assert calc.project_life == 25

    def test_calculate_lcoh(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test LCOH calculation."""
        lcoh = calculator.calculate_lcoh(sample_operating_outputs, sample_cost_params, 'fixed')
        assert isinstance(lcoh, float)
        assert lcoh > 0  # Should be positive

    def test_calculate_npv(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test NPV calculation."""
        npv = calculator.calculate_npv(sample_operating_outputs, sample_cost_params, hydrogen_price=350)
        assert isinstance(npv, (int, float))
        # NPV can be positive or negative depending on project economics

    def test_calculate_payback_period(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test payback period calculation."""
        payback = calculator.calculate_payback_period(sample_operating_outputs, sample_cost_params,
                                                    hydrogen_price=350)
        assert payback is None or (isinstance(payback, (int, float)) and payback > 0)

    def test_calculate_roi(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test ROI calculation."""
        roi = calculator.calculate_roi(sample_operating_outputs, sample_cost_params, hydrogen_price=350)
        assert isinstance(roi, (int, float))

    def test_get_cash_flow_summary_no_data(self, calculator):
        """Test cash flow summary when no data is available."""
        summary = calculator.get_cash_flow_summary()
        assert isinstance(summary, dict)
        assert len(summary) == 0  # Should be empty

    def test_get_cash_flow_summary_with_data(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test cash flow summary with available data."""
        # Calculate NPV first to populate cash flows
        calculator.calculate_npv(sample_operating_outputs, sample_cost_params, hydrogen_price=350)

        summary = calculator.get_cash_flow_summary()
        assert isinstance(summary, dict)
        assert 'total_nominal_cash_flow' in summary
        assert 'average_annual_cash_flow' in summary
        assert 'project_life' in summary
        assert 'discount_rate' in summary

    def test_perform_sensitivity_analysis(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test sensitivity analysis."""
        base_case = {'lcoh': 5.0}

        parameter_ranges = {
            'hydrogen_price': [300, 350, 400, 450, 500],
            'gen_capex_per_mw': [800, 900, 1000, 1100, 1200]
        }

        results = calculator.perform_sensitivity_analysis(
            base_case, parameter_ranges, sample_operating_outputs, sample_cost_params
        )

        assert isinstance(results, dict)
        assert 'hydrogen_price' in results
        assert 'gen_capex_per_mw' in results

        # Check structure of results
        for param_results in results.values():
            assert isinstance(param_results, list)
            assert len(param_results) > 0
            assert 'parameter_value' in param_results[0]
            assert 'lcoh' in param_results[0]

    def test_export_cash_flows_no_data(self, calculator, tmp_path):
        """Test export when no data is available."""
        filepath = tmp_path / "test_cash_flows.csv"
        calculator.export_cash_flows(str(filepath))
        # Should not raise an error and no file should be created
        assert not filepath.exists()

    def test_generate_project_cash_flows(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test project cash flow generation."""
        annual_hydrogen = sample_operating_outputs["Hydrogen Output for Variable Operation [t/yr]"]

        cash_flows = calculator._generate_project_cash_flows(
            sample_operating_outputs, sample_cost_params, annual_hydrogen
        )

        assert isinstance(cash_flows, pd.DataFrame)
        assert 'Year' in cash_flows.columns
        assert 'Total' in cash_flows.columns
        assert len(cash_flows) == calculator.project_life + 1  # Years 0-20

        # Check that Year 0 has negative cash flow (CAPEX)
        assert cash_flows['Total'][0] < 0

    def test_calculate_discounted_cash_flows(self, calculator, sample_operating_outputs, sample_cost_params):
        """Test discounted cash flow calculation."""
        annual_hydrogen = sample_operating_outputs["Hydrogen Output for Variable Operation [t/yr]"]

        # Generate cash flows first
        cash_flows = calculator._generate_project_cash_flows(
            sample_operating_outputs, sample_cost_params, annual_hydrogen
        )

        # Calculate discounted flows
        discounted_results = calculator._calculate_discounted_cash_flows(cash_flows, annual_hydrogen)

        assert isinstance(discounted_results, dict)
        assert 'discounted_costs' in discounted_results
        assert 'discounted_hydrogen' in discounted_results
        assert len(discounted_results['discounted_costs']) == calculator.project_life + 1
        assert len(discounted_results['discounted_hydrogen']) == calculator.project_life + 1

        # Year 0 should have no hydrogen production (discounted)
        assert discounted_results['discounted_hydrogen'][0] == 0


if __name__ == "__main__":
    pytest.main([__file__])