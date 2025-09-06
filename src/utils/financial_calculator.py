"""
Financial Calculator Module
Provides comprehensive financial analysis capabilities for renewable hydrogen projects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class FinancialCalculator:
    """
    Comprehensive financial calculator for renewable hydrogen production projects.
    Handles LCOH, NPV, cash flow analysis, sensitivity analysis, and profitability metrics.
    """

    def __init__(self, discount_rate: float = 0.04, project_life: int = 20):
        """
        Initialize the financial calculator.

        Args:
            discount_rate: Discount rate for NPV calculations (default 4%)
            project_life: Project lifetime in years (default 20)
        """
        self.discount_rate = discount_rate
        self.project_life = project_life
        self.cash_flows = {}

    def calculate_lcoh(self, operating_outputs: Dict, cost_params: Dict,
                      specific_consumption_type: str = 'fixed') -> float:
        """
        Calculate Levelized Cost of Hydrogen (LCOH) using discounted cash flow analysis.

        Args:
            operating_outputs: Dictionary with operational results from hydrogen model
            cost_params: Dictionary with cost parameters
            specific_consumption_type: 'fixed' or 'variable' consumption model

        Returns:
            LCOH in $/kg
        """
        # Extract hydrogen production
        if specific_consumption_type == "variable":
            annual_hydrogen = operating_outputs["Hydrogen Output for Variable Operation [t/yr]"]
        else:
            annual_hydrogen = operating_outputs["Hydrogen Output for Fixed Operation [t/yr]"]

        if annual_hydrogen <= 0:
            logger.warning("Annual hydrogen production is zero or negative")
            return 0.0

        # Generate cash flows
        cash_flows = self._generate_project_cash_flows(operating_outputs, cost_params, annual_hydrogen)
        self.cash_flows = cash_flows

        # Calculate discounted cash flows
        discounted_flows = self._calculate_discounted_cash_flows(cash_flows, annual_hydrogen)

        # Check for valid calculations
        total_discounted_costs = discounted_flows['discounted_costs'].sum()
        total_discounted_hydrogen = discounted_flows['discounted_hydrogen'].sum()

        if total_discounted_hydrogen <= 0:
            logger.warning("Total discounted hydrogen production is zero or negative")
            return 0.0

        # Calculate LCOH - ensure positive result
        lcoh = abs(total_discounted_costs) / total_discounted_hydrogen
        return round(lcoh, 2)

    def calculate_npv(self, operating_outputs: Dict, cost_params: Dict,
                     hydrogen_price: float = 350) -> float:
        """
        Calculate Net Present Value (NPV) of the project.

        Args:
            operating_outputs: Operational results from hydrogen model
            cost_params: Cost parameters
            hydrogen_price: Price per tonne of hydrogen ($/t)

        Returns:
            NPV in dollars
        """
        # Use variable operation hydrogen production
        annual_hydrogen = operating_outputs["Hydrogen Output for Variable Operation [t/yr]"]

        # Generate cash flows
        cash_flows = self._generate_project_cash_flows(operating_outputs, cost_params, annual_hydrogen)

        # Add revenue from hydrogen sales (for years 1-20, year 0 has no revenue)
        cash_flows['hydrogen_revenue'] = [0] + [annual_hydrogen * hydrogen_price / 1000] * self.project_life  # $/year

        # Add revenue to total cash flow
        cash_flows['Total'] = (cash_flows['Total'] +
                             cash_flows['hydrogen_revenue'])

        # Calculate NPV (exclude Year 0 for NPV calculation)
        npv = 0
        for year in range(1, self.project_life + 1):
            npv += cash_flows['Total'][year] / ((1 + self.discount_rate) ** year)

        self.cash_flows = cash_flows
        return round(npv, 2)

    def calculate_payback_period(self, operating_outputs: Dict, cost_params: Dict,
                               hydrogen_price: float = 350) -> Optional[float]:
        """
        Calculate payback period in years.

        Args:
            operating_outputs: Operational results from hydrogen model
            cost_params: Cost parameters
            hydrogen_price: Price per tonne of hydrogen ($/t)

        Returns:
            Payback period in years (or None if not achieved)
        """
        # Calculate NPV first to get cash flows
        self.calculate_npv(operating_outputs, cost_params, hydrogen_price)

        # Calculate cumulative discounted cash flows
        cumulative_cf = 0
        for year in range(1, self.project_life + 1):
            cumulative_cf += self.cash_flows['Total'][year] / ((1 + self.discount_rate) ** year)
            if cumulative_cf >= 0:
                return year

        return None  # Payback not achieved within project life

    def calculate_roi(self, operating_outputs: Dict, cost_params: Dict,
                     hydrogen_price: float = 350) -> float:
        """
        Calculate Return on Investment (ROI).

        Args:
            operating_outputs: Operational results from hydrogen model
            cost_params: Cost parameters
            hydrogen_price: Price per tonne of hydrogen ($/t)

        Returns:
            ROI as a percentage
        """
        npv = self.calculate_npv(operating_outputs, cost_params, hydrogen_price)
        initial_investment = abs(self.cash_flows['total'][0])  # Year 0 CAPEX (negative)

        if initial_investment == 0:
            return 0

        roi = (npv / initial_investment) * 100
        return round(roi, 2)

    def perform_sensitivity_analysis(self, base_case: Dict, parameter_ranges: Dict,
                                   operating_outputs: Dict, cost_params: Dict) -> Dict:
        """
        Perform sensitivity analysis on key parameters.

        Args:
            base_case: Dictionary with base case results
            parameter_ranges: Dictionary with parameter ranges to test
            operating_outputs: Operational results
            cost_params: Base cost parameters

        Returns:
            Dictionary with sensitivity results
        """
        results = {}

        for param_name, param_range in parameter_ranges.items():
            param_results = []

            for variation in param_range:
                # Modify the parameter
                modified_cost_params = cost_params.copy()

                if param_name in modified_cost_params:
                    modified_cost_params[param_name] = variation
                elif param_name == 'hydrogen_price':
                    # Special case for hydrogen price
                    lcoh = self.calculate_lcoh(operating_outputs, modified_cost_params)
                    param_results.append({
                        'parameter_value': variation,
                        'lcoh': lcoh,
                        'npv': self.calculate_npv(operating_outputs, modified_cost_params, variation),
                        'margin': lcoh / variation if variation > 0 else 0
                    })
                else:
                    # For other parameters, calculate impact on LCOH
                    lcoh = self.calculate_lcoh(operating_outputs, modified_cost_params)
                    param_results.append({
                        'parameter_value': variation,
                        'lcoh': lcoh,
                        'change_percent': ((lcoh - base_case.get('lcoh', 0)) / base_case.get('lcoh', 1)) * 100
                    })

            results[param_name] = param_results

        return results

    def get_cash_flow_summary(self) -> Dict:
        """
        Get cash flow summary statistics.

        Returns:
            Dictionary with cash flow summary
        """
        if self.cash_flows is None or (isinstance(self.cash_flows, dict) and not self.cash_flows):
            logger.warning("No cash flows calculated yet")
            return {}

        total_cash_flow = sum(self.cash_flows.get('Total', []))

        summary = {
            'total_nominal_cash_flow': total_cash_flow,
            'average_annual_cash_flow': total_cash_flow / self.project_life if self.project_life > 0 else 0,
            'project_life': self.project_life,
            'discount_rate': self.discount_rate
        }

        # Analyze cash flows by component
        if 'Total' in self.cash_flows:
            positive_years = sum(1 for cf in self.cash_flows['Total'][1:] if cf > 0)
            negative_years = sum(1 for cf in self.cash_flows['Total'][1:] if cf < 0)

            summary.update({
                'profitable_years': positive_years,
                'loss_years': negative_years,
                'break_even_ratio': positive_years / self.project_life if self.project_life > 0 else 0
            })

        return summary

    def export_cash_flows(self, filepath: str) -> None:
        """
        Export cash flow data to CSV.

        Args:
            filepath: Path to save the CSV file
        """
        if self.cash_flows is None or (isinstance(self.cash_flows, dict) and not self.cash_flows):
            logger.warning("No cash flows to export")
            return

        df = pd.DataFrame(self.cash_flows)
        df.to_csv(filepath, index=False)
        logger.info(f"Cash flows exported to {filepath}")

    def _generate_project_cash_flows(self, operating_outputs: Dict, cost_params: Dict,
                                   annual_hydrogen: float) -> pd.DataFrame:
        """
        Generate project cash flows based on operational and cost parameters.

        Args:
            operating_outputs: Operational results
            cost_params: Cost parameters
            annual_hydrogen: Annual hydrogen production in tonnes/year

        Returns:
            DataFrame with cash flows by year
        """
        cash_flow_df = pd.DataFrame(index=range(self.project_life + 1))

        # Initialize cash flow columns
        columns = ['Year', 'Gen_CAPEX', 'Elec_CAPEX', 'Gen_OPEX', 'Elec_OandM',
                  'Power_cost', 'Stack_replacement', 'Water_cost', 'Battery_cost', 'Total']
        for col in columns:
            cash_flow_df[col] = 0.0

        cash_flow_df['Year'] = range(self.project_life + 1)

        # Year 0: Capital expenditures (negative cash outflows)
        gen_capex = (cost_params.get('gen_capex_per_mw', 0) *
                    (cost_params.get('solar_capacity', 0) + cost_params.get('wind_capacity', 0)))
        cash_flow_df.at[0, 'Gen_CAPEX'] = -gen_capex  # Negative for outflow

        elec_capex = cost_params.get('electrolyser_capex', 0) * cost_params.get('electrolyser_capacity', 0)
        cash_flow_df.at[0, 'Elec_CAPEX'] = -elec_capex  # Negative for outflow

        # Battery CAPEX
        battery_energy = cost_params.get('battery_capacity', 0) * cost_params.get('battery_duration', 0)
        battery_capex_rate = cost_params.get('battery_capex_per_mwh', 400)
        cash_flow_df.at[0, 'Battery_cost'] = -battery_capex_rate * battery_energy  # Negative for outflow

        # Operating years
        gen_opex_annual = (cost_params.get('gen_opex_per_mw', 0) *
                          (cost_params.get('solar_capacity', 0) + cost_params.get('wind_capacity', 0)))
        cash_flow_df.loc[1:, 'Gen_OPEX'] = gen_opex_annual

        elec_om_annual = cost_params.get('electrolyser_om', 0) * cost_params.get('electrolyser_capacity', 0)
        cash_flow_df.loc[1:, 'Elec_OandM'] = elec_om_annual

        # Power costs (if in PPA mode)
        if cost_params.get('ppa_price', 0) > 0:
            power_cost = operating_outputs.get("Energy in to Electrolyser [MWh/yr]", 0) * cost_params['ppa_price']
            cash_flow_df.loc[1:, 'Power_cost'] = power_cost
        else:
            # Surplus energy revenue
            surplus_revenue = (operating_outputs.get("Surplus Energy [MWh/yr]", 0) *
                             cost_params.get('spot_price', 0))
            cash_flow_df.loc[1:, 'Power_cost'] = -surplus_revenue

        # Stack replacement costs
        stack_lifetime_hours = cost_params.get('stack_lifetime', 60000)
        operating_hours_per_year = operating_outputs.get("Total Time Electrolyser is Operating", 0) * 8760
        stack_replacement_years = []

        for year in range(1, self.project_life):
            cumulative_hours = operating_hours_per_year * year
            if cumulative_hours > 0 and cumulative_hours % stack_lifetime_hours < operating_hours_per_year:
                stack_replacement_years.append(year)

        stack_replacement_cost = cost_params.get('stack_replacement_cost', 0) * cost_params.get('electrolyser_capacity', 0)
        for year in stack_replacement_years:
            cash_flow_df.at[year, 'Stack_replacement'] = stack_replacement_cost

        # Water costs
        water_usage_per_tonne = cost_params.get('water_usage', 0)
        water_cost_per_unit = cost_params.get('water_cost', 0)
        annual_water_cost = annual_hydrogen * water_usage_per_tonne * water_cost_per_unit
        cash_flow_df.loc[1:, 'Water_cost'] = annual_water_cost

        # Battery replacement and O&M
        battery_lifetime = cost_params.get('battery_lifetime', 10)
        if battery_lifetime > 0 and battery_lifetime <= self.project_life:
            replacement_year = battery_lifetime
            battery_replacement_rate = cost_params.get('battery_replacement_rate', 100) / 100
            battery_initial_capex = abs(cash_flow_df.at[0, 'Battery_cost'])  # Get positive value from negative CAPEX
            cash_flow_df.at[replacement_year, 'Battery_cost'] -= battery_initial_capex * battery_replacement_rate

        # Battery O&M costs
        battery_om_cost = cost_params.get('battery_om_per_mw', 0) * cost_params.get('battery_capacity', 0)
        cash_flow_df.loc[1:, 'Battery_cost'] += battery_om_cost

        # Calculate total cash flow
        numeric_columns = [col for col in cash_flow_df.columns if col not in ['Year']]
        cash_flow_df['Total'] = cash_flow_df[numeric_columns].sum(axis=1)

        return cash_flow_df

    def _calculate_discounted_cash_flows(self, cash_flows: pd.DataFrame, annual_hydrogen: float) -> Dict:
        """
        Calculate discounted cash flows and hydrogen production.

        Args:
            cash_flows: Undiscounted cash flows
            annual_hydrogen: Annual hydrogen production in tonnes/year

        Returns:
            Dictionary with discounted costs and hydrogen
        """
        discounted_costs = []
        discounted_hydrogen = []
        kg_per_tonne = 1000

        for year in range(self.project_life + 1):
            if year == 0:
                # Year 0 costs (no hydrogen production yet)
                discounted_cost = cash_flows['Total'][year]
                hydrogen_kg = 0
            else:
                # Operating years
                discounted_cost = cash_flows['Total'][year] / ((1 + self.discount_rate) ** year)
                hydrogen_kg = annual_hydrogen * kg_per_tonne / ((1 + self.discount_rate) ** year)

            discounted_costs.append(discounted_cost)
            discounted_hydrogen.append(hydrogen_kg)

        return {
            'discounted_costs': np.array(discounted_costs),
            'discounted_hydrogen': np.array(discounted_hydrogen)
        }


# Factory function for easy instantiation
def create_financial_calculator(discount_rate: float = 0.04, project_life: int = 20) -> FinancialCalculator:
    """
    Factory function to create a FinancialCalculator instance.

    Args:
        discount_rate: Discount rate for NPV calculations
        project_life: Project lifetime in years

    Returns:
        FinancialCalculator instance
    """
    return FinancialCalculator(discount_rate=discount_rate, project_life=project_life)