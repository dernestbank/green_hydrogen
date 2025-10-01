"""Visualization utilities for hydrogen production framework.

This module provides data formatting utilities for charts and visualizations
used in the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ChartDataFormatter:
    """Utility class for formatting data for various chart types."""

    @staticmethod
    def format_duration_curve_data(hourly_data: pd.Series,
                                   title: str = "Duration Curve",
                                   color: str = "#1f77b4") -> Dict[str, Any]:
        """
        Format hourly data into duration curve format for Plotly.

        Args:
            hourly_data: Series with hourly values
            title: Title for the chart
            color: Color for the line

        Returns:
            Dict formatted for Plotly line chart
        """
        sorted_data = hourly_data.sort_values(ascending=False)
        time_percent = np.arange(len(sorted_data)) / len(sorted_data) * 100

        return {
            'data': [{
                'x': time_percent,
                'y': sorted_data.values,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Capacity Factor',
                'line': {'color': color, 'width': 2}
            }],
            'layout': {
                'title': title,
                'xaxis': {
                    'title': 'Time (%)',
                    'tickformat': ',.0f'
                },
                'yaxis': {
                    'title': 'Capacity Factor',
                    'tickformat': ',.2f'
                },
                'showlegend': False
            }
        }

    @staticmethod
    def format_cost_breakdown_data(cost_data: Dict[str, float],
                                   title: str = "Cost Breakdown") -> Dict[str, Any]:
        """
        Format cost data for pie chart visualization.

        Args:
            cost_data: Dictionary of cost categories with values
            title: Chart title

        Returns:
            Dict formatted for Plotly pie chart
        """
        labels = list(cost_data.keys())
        values = list(cost_data.values())

        # Calculate percentages
        total = sum(values)
        percentages = ['{:.1f}%'.format(v/total*100) for v in values]

        return {
            'data': [{
                'values': values,
                'labels': labels,
                'type': 'pie',
                'text': percentages,
                'textposition': 'inside',
                'hoverinfo': 'label+percent+value',
                'name': ''
            }],
            'layout': {
                'title': title,
                'showlegend': True
            }
        }

    @staticmethod
    def format_hourly_profile_data(hourly_data: pd.DataFrame,
                                   columns: List[str],
                                   title: str = "Hourly Profile",
                                   y_label: str = "Value") -> Dict[str, Any]:
        """
        Format hourly data for line/scatter plot.

        Args:
            hourly_data: DataFrame with hourly data
            columns: Columns to plot
            title: Chart title
            y_label: Y-axis label

        Returns:
            Dict formatted for Plotly chart
        """
        plot_data = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, col in enumerate(columns):
            if col in hourly_data.columns:
                plot_data.append({
                    'x': hourly_data.index.tolist(),
                    'y': hourly_data[col].tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': col,
                    'line': {'color': colors[i % len(colors)]}
                })

        return {
            'data': plot_data,
            'layout': {
                'title': title,
                'xaxis': {
                    'title': 'Hour of Year',
                    'tickmode': 'array',
                    'tickvals': [0, 8760//4, 8760//2, 3*8760//4, 8760],
                    'ticktext': ['Jan', 'Apr', 'Jul', 'Oct', 'Dec']
                },
                'yaxis': {
                    'title': y_label,
                    'tickformat': ',.2f'
                },
                'showlegend': True
            }
        }

    @staticmethod
    def format_bullet_chart_data(current_value: float,
                                target_value: float,
                                ranges: List[float],
                                title: str = "Metric Indicator") -> Dict[str, Any]:
        """
        Format data for bullet chart visualization.

        Args:
            current_value: Current value to display
            target_value: Target/measure value
            ranges: List of quantitative range values [poor, satisfactory, good]
            title: Chart title

        Returns:
            Dict formatted for Plotly bullet chart
        """
        # This is a conceptual structure - actual implementation would depend on specific bullet chart library

        return {
            'current': current_value,
            'target': target_value,
            'ranges': ranges,
            'qualitative_ranges': ['Poor', 'Satisfactory', 'Good'],
            'title': title
        }

    @staticmethod
    def format_comparison_chart_data(scenarios: List[Dict[str, Any]],
                                    metrics: List[str],
                                    title: str = "Scenario Comparison") -> Dict[str, Any]:
        """
        Format scenario comparison data for bar chart.

        Args:
            scenarios: List of scenario dictionaries
            metrics: List of metric keys to compare
            title: Chart title

        Returns:
            Dict formatted for Plotly bar chart
        """
        plot_data = []

        for i, metric in enumerate(metrics):
            x_values = []
            y_values = []

            for scenario in scenarios:
                scenario_name = scenario.get('name', f'Scenario {i+1}')
                if metric in scenario:
                    x_values.append(scenario_name)
                    y_values.append(scenario[metric])

            if x_values and y_values:
                plot_data.append({
                    'x': x_values,
                    'y': y_values,
                    'type': 'bar',
                    'name': metric
                })

        return {
            'data': plot_data,
            'layout': {
                'title': title,
                'barmode': 'group',
                'xaxis': {'title': 'Scenarios'},
                'yaxis': {'title': 'Values', 'tickformat': ',.2f'},
                'showlegend': True
            }
        }

    @staticmethod
    def format_waterfall_chart_data(cost_components: Dict[str, float],
                                   final_value: float,
                                   title: str = "Cost Breakdown") -> Dict[str, Any]:
        """
        Format cost data for waterfall chart.

        Args:
            cost_components: Dictionary of cost components
            final_value: Total cost value
            title: Chart title

        Returns:
            Dict formatted for Plotly waterfall chart
        """
        measures = []
        values = []
        labels = []

        # Add base value
        measures.append("total")
        values.append(0)
        labels.append("Base")

        # Add cost components
        for component, cost in cost_components.items():
            measures.append("relative")
            values.append(cost)
            labels.append(component)

        # Make final total
        measures.append("total")
        values.append(final_value)
        labels.append("Total")

        return {
            'data': [{
                'type': 'waterfall',
                'name': 'Cost Components',
                'orientation': 'v',
                'measure': measures,
                'x': labels,
                'y': values,
                'textposition': 'outside'
            }],
            'layout': {
                'title': title,
                'showlegend': False
            }
        }

    @staticmethod
    def format_summary_cards_data(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format metrics data for Streamlit metric cards.

        Args:
            metrics: Dictionary of metric names and values

        Returns:
            List of dictionaries with title, value, delta for metric cards
        """
        cards = []

        for name, data in metrics.items():
            if isinstance(data, dict):
                value = data.get('value', 0)
                delta = data.get('delta', None)
                format_str = data.get('format', ',.2f')

                cards.append({
                    'title': name,
                    'value': f"{value:{format_str}}",
                    'delta': delta
                })
            else:
                cards.append({
                    'title': name,
                    'value': data,
                    'delta': None
                })

        return cards

# Convenience functions for common chart types
def get_duration_curve_data(model_results: Dict[str, Any], chart_type: str = 'generator') -> Dict[str, Any]:
    """
    Extract and format duration curve data from model results.

    Args:
        model_results: Results dictionary from hydrogen model
        chart_type: 'generator' or 'electrolyser'

    Returns:
        Formatted chart data
    """
    # This would typically call the model's get_duration_curve_data method
    # Placeholder for now
    formatter = ChartDataFormatter()

    if chart_type == 'generator':
        # Placeholder data - in real implementation, this would come from model
        sample_data = pd.Series(np.random.uniform(0, 1, 8760))
        return formatter.format_duration_curve_data(sample_data,
                                                   title="Generator Duration Curve")

    return {}

def get_cost_waterfall_data(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and format cost breakdown for waterfall chart.

    Args:
        model_results: Results dictionary from hydrogen model

    Returns:
        Formatted waterfall chart data
    """
    # Extract cost components from model results
    cost_components = {}

    if 'financial_results' in model_results:
        # Add relevant costs
        cost_components['CAPEX'] = model_results['financial_results'].get('capex', 0)
        cost_components['OPEX'] = model_results['financial_results'].get('opex', 0)
        cost_components['Other]'] = model_results['financial_results'].get('other_costs', 0)

    formatter = ChartDataFormatter()
    total_cost = model_results.get('financial_results', {}).get('total_cost', 0)

    return formatter.format_waterfall_chart_data(cost_components, total_cost,
                                                title="Cost Breakdown")

def get_hourly_production_data(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format hourly production data for visualization.

    Args:
        model_results: Results dictionary from hydrogen model

    Returns:
        Formatted chart data
    """
    # Extract hourly data from model results
    hourly_df = model_results.get('hourly_data', pd.DataFrame())

    if hourly_df.empty:
        # Create sample data if not available
        hourly_df = pd.DataFrame({
            'hydrogen_production': np.random.uniform(0, 100, 8760),
            'power_generation': np.random.uniform(0, 1, 8760)
        }, index=range(8760))

    formatter = ChartDataFormatter()
    return formatter.format_hourly_profile_data(hourly_df,
                                               ['hydrogen_production', 'power_generation'],
                                               title="Hourly Production Profile",
                                               y_label="Production (kg/hr) / CF")


class KPICalculator:
    """Utility class for calculating Key Performance Indicators."""

    @staticmethod
    def calculate_capacity_factor(hourly_generation: pd.Series,
                                  installed_capacity_kwh: float,
                                  hours_per_year: int = 8760) -> float:
        """
        Calculate capacity factor from hourly generation data.

        Args:
            hourly_generation: Series with hourly generation (MWh)
            installed_capacity_kwh: Installed capacity (kW)
            hours_per_year: Number of hours in year

        Returns:
            Capacity factor (0-1)
        """
        if installed_capacity_kwh <= 0 or len(hourly_generation) == 0:
            return 0.0

        # Calculate capacity factor
        # Annual energy generated / (installed capacity * hours per year)
        total_energy_mwh = hourly_generation.sum()
        max_energy_mwh = installed_capacity_kwh * 1000 * hours_per_year / 1000  # Convert kWh to MWh scaling

        capacity_factor = total_energy_mwh / max_energy_mwh if max_energy_mwh > 0 else 0.0

        return min(max(capacity_factor, 0.0), 1.0)

    @staticmethod
    def calculate_lcoh(annual_h2_production_t: float,
                       capex_total: float,
                       opex_total: float,
                       discount_rate: float = 0.04,
                       project_life_years: int = 20) -> float:
        """
        Calculate Levelized Cost of Hydrogen.

        Args:
            annual_h2_production_t: Annual hydrogen production (tonnes)
            capex_total: Total capital expenditures
            opex_total: Annual operational expenditures
            discount_rate: Discount rate (0-1)
            project_life_years: Project lifetime (years)

        Returns:
            LCOH in $/kg
        """
        if annual_h2_production_t <= 0:
            return float('inf')

        # Convert tonnes to kg
        annual_h2_production_kg = annual_h2_production_t * 1000

        # Calculate NPV of costs
        npv_capex = capex_total
        npv_opex = opex_total * ((1 - (1 + discount_rate) ** -project_life_years) / discount_rate)

        total_discounted_cost = npv_capex + npv_opex

        lcoh = total_discounted_cost / (annual_h2_production_kg * project_life_years)

        return lcoh

    @staticmethod
    def calculate_system_efficiency(annual_h2_production_kg: float,
                                    annual_energy_input_mwh: float,
                                    h2_energy_content_kwh_per_kg: float = 39.4) -> float:
        """
        Calculate system efficiency (electrical to hydrogen energy efficiency).

        Args:
            annual_h2_production_kg: Annual hydrogen production (kg)
            annual_energy_input_mwh: Annual energy input (MWh)
            h2_energy_content_kwh_per_kg: Energy content of hydrogen (kWh/kg)

        Returns:
            System efficiency (0-1)
        """
        if annual_h2_production_kg <= 0 or annual_energy_input_mwh <= 0:
            return 0.0

        theoretical_energy_output_kwh = annual_h2_production_kg * h2_energy_content_kwh_per_kg
        input_energy_kwh = annual_energy_input_mwh * 1000

        efficiency = theoretical_energy_output_kwh / input_energy_kwh

        return min(max(efficiency, 0.0), 1.0)

    @staticmethod
    def calculate_payback_period(initial_investment: float,
                                annual_cashflow: float,
                                discount_rate: float = 0.04) -> Tuple[float, bool]:
        """
        Calculate payback period in years.

        Args:
            initial_investment: Initial capital expenditure
            annual_cashflow: Annual net cash flow
            discount_rate: Discount rate (0-1)

        Returns:
            Tuple of (payback_period, payed_back) where payed_back is True if project breaks even
        """
        if annual_cashflow <= 0:
            return float('inf'), False

        # Simple payback (no discounting)
        if discount_rate == 0:
            payback = initial_investment / annual_cashflow
            return payback, payback <= 100  # Assume 100 year max

        # Discounted payback period
        npv_cumulative = 0
        year = 0

        while npv_cumulative < initial_investment and year < 100:
            year += 1
            npv_cumulative += annual_cashflow / (1 + discount_rate) ** year

            if npv_cumulative >= initial_investment:
                # Interpolate for more precision
                remaining = initial_investment - (npv_cumulative - annual_cashflow / (1 + discount_rate) ** year)
                fraction = remaining / (annual_cashflow / (1 + discount_rate) ** year)
                return year - 1 + fraction, True

        return float('inf'), False

    @staticmethod
    def calculate_roi(initial_investment: float,
                      annual_profit: float,
                      discount_rate: float = 0.04,
                      project_life_years: int = 20) -> float:
        """
        Calculate Return on Investment over project life.

        Args:
            initial_investment: Initial capital expenditure
            annual_profit: Annual profit
            discount_rate: Discount rate (0-1)
            project_life_years: Project lifetime (years)

        Returns:
            ROI as a decimal (e.g., 0.15 for 15%)
        """
        if initial_investment <= 0:
            return float('-inf')

        total_discounted_profit = sum(annual_profit / (1 + discount_rate) ** year
                                     for year in range(1, project_life_years + 1))

        roi = total_discounted_profit / initial_investment

        return roi

    @staticmethod
    def calculate_cost_breakdown(capex_components: Dict[str, float],
                                opex_possible: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate cost breakdown percentages.

        Args:
            capex_components: Capital cost breakdown
            opex_possible: Operational cost breakdown

        Returns:
            Dictionary with percentages and totals
        """
        capex_total = sum(capex_components.values())
        opex_total = sum(opex_possible.values())

        breakdown = {
            'capex_total': capex_total,
            'opex_total': opex_total,
            'capex_breakdown': {k: v / capex_total if capex_total > 0 else 0
                               for k, v in capex_components.items()},
            'opex_breakdown': {k: v / opex_total if opex_total > 0 else 0
                              for k, v in opex_possible.items()},
        }

        return breakdown

    @staticmethod
    def calculate_emissions_intensity(h2_production_kg: float,
                                    grid_emissions_percent: float = 0.4,
                                    electrolysis_efficiency: float = 0.7) -> float:
        """
        Calculate carbon emissions intensity of hydrogen production.

        Args:
            h2_production_kg: Hydrogen production (kg)
            grid_emissions_percent: Grid emission factor (kg CO2/kWh)
            electrolysis_efficiency: Electrolyser efficiency

        Returns:
            CO2 emissions per kg hydrogen (kg CO2/kg H2)
        """
        # Estimate electrical energy input per kg H2
        energy_per_kg_kwh = 39.4 / electrolysis_efficiency  # ~56 kWh/kg for 70% efficiency

        emissions_intensity = (energy_per_kg_kwh * grid_emissions_percent) / h2_production_kg

        return emissions_intensity

    def calculate_kpis_from_model_results(self, model_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all KPIs from model results.

        Args:
            model_results: Comprehensive model results dictionary

        Returns:
            Dictionary of KPI categories with formatted values
        """
        kpis = {
            'system_performance': {},
            'financial_metrics': {},
            'energy_efficiency': {},
            'environmental_impact': {}
        }

        # System Performance KPIs
        if 'operating_outputs' in model_results:
            outputs = model_results['operating_outputs']
            kpis['system_performance'] = {
                'generator_capacity_factor': outputs.get('Generator Capacity Factor', 0),
                'electrolyser_capacity_factor': outputs.get('Achieved Electrolyser Capacity Factor', 0),
                'electrolyser_load_factor': outputs.get('Time Electrolyser is at its Rated Capacity', 0),
                'annual_energy_input_mwh': outputs.get('Energy in to Electrolyser [MWh/yr]', 0),
                'annual_surplus_energy_mwh': outputs.get('Surplus Energy [MWh/yr]', 0),
                'annual_h2_production_t': outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 0),
            }

        # Financial Metrics
        if 'financial_results' in model_results:
            finance = model_results['financial_results']
            kpis['financial_metrics'] = {
                'lcoh_fixed': finance.get('LCOH - Fixed Consumption ($/kg)', 0),
                'lcoh_variable': finance.get('LCOH - Variable Consumption ($/kg)', 0),
                'payback_period': self.calculate_payback_period(
                    finance.get('capex', 0),
                    finance.get('annual_profit', 0),
                    finance.get('discount_rate', 0.04)
                )[0],
                'roi': self.calculate_roi(
                    finance.get('capex', 0),
                    finance.get('annual_profit', 0),
                    finance.get('discount_rate', 0.04),
                    finance.get('project_life', 20)
                ),
            }

        # Energy Efficiency KPIs
        if 'technical_parameters' in model_results:
            tech = model_results['technical_parameters']
            kpis['energy_efficiency'] = {
                'electrolyser_efficiency': tech.get('Electrolyser Efficiency', 0.7),
                'system_efficiency': self.calculate_system_efficiency(
                    kpis['system_performance'].get('annual_h2_production_t', 0) * 1000,  # Convert to kg
                    kpis['system_performance'].get('annual_energy_input_mwh', 0)
                ),
                'h2_production_rate_kg_day': kpis['system_performance'].get('annual_h2_production_t', 0) * 1000 / 365,
            }

        # Environmental Impact KPIs
        if 'anniversary_h2_production' in model_results:
            h2_kg = model_results['anniversary_h2_production'] * 1000  # Convert to kg
            kpis['environmental_impact'] = {
                'emissions_intensity_kg_co2_per_kg_h2': self.calculate_emissions_intensity(
                    h2_kg,
                    grid_emissions_percent=0.4,  # Adapt for renewable mix
                    electrolysis_efficiency=kpis['energy_efficiency'].get('electrolyser_efficiency', 0.7)
                ),
                'well_to_wheels_emissions': 0,  # Placeholder for future calculation
            }

        return kpis


class DataAggregator:
    """Utility class for aggregating time series data into different periods."""

    @staticmethod
    def create_datetime_index(start_date: str = '2023-01-01',
                            hours: int = 8760,
                            freq: str = 'H') -> pd.DatetimeIndex:
        """
        Create datetime index for time series data.

        Args:
            start_date: Start date string
            hours: Number of hours
            freq: Frequency string

        Returns:
            DatetimeIndex
        """
        return pd.date_range(start_date, periods=hours, freq=freq)

    @staticmethod
    def aggregate_hourly_to_period(data: pd.DataFrame,
                                  period: str = 'M',
                                  agg_funcs: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Aggregate hourly data to specified period (M, Q, Y).

        Args:
            data: DataFrame with datetime index
            period: Period string ('M' for monthly, 'Q' for quarterly, 'A' for annual)
            agg_funcs: Dictionary mapping columns to aggregation functions

        Returns:
            Aggregated DataFrame
        """
        if agg_funcs is None:
            agg_funcs = {
                'default_mean': 'mean',
                'sum_cols': 'sum'
            }

        # Determine aggregation functions for columns
        grouping = {}

        for col in data.columns:
            if 'production' in col.lower() or 'consumption' in col.lower() or 'energy' in col.lower():
                grouping[col] = 'sum'
            elif 'capacity' in col.lower() or 'factor' in col.lower() or 'efficiency' in col.lower():
                grouping[col] = 'mean'
            else:
                grouping[col] = 'mean'  # Default to mean

        if not data.index.is_datetime64_any_dtype:
            data.index = DataAggregator.create_datetime_index()

        aggregated = data.groupby(pd.Grouper(freq=period)).agg(grouping)

        return aggregated

    @staticmethod
    def calculate_seasonal_averages(hourly_data: pd.Series, seasons: Optional[Dict[str, List[int]]] = None) -> Dict[str, float]:
        """
        Calculate seasonal averages based on month groupings.

        Args:
            hourly_data: Hourly time series
            seasons: Dict mapping season names to list of months (1-12)

        Returns:
            Dictionary of seasonal averages
        """
        if seasons is None:
            seasons = {
                'Winter': [12, 1, 2],
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Autumn': [9, 10, 11]
            }

        if not hourly_data.index.is_datetime64_any_dtype:
            hourly_data.index = DataAggregator.create_datetime_index()

        seasonal_avg = {}

        for season, months in seasons.items():
            season_data = hourly_data[hourly_data.index.month.isin(months)]
            seasonal_avg[season] = season_data.mean()

        return seasonal_avg

    @staticmethod
    def create_summary_statistics(hourly_data: pd.Series,
                                 periods: List[str] = ['D', 'W', 'M', 'Q', 'A']) -> pd.DataFrame:
        """
        Create summary statistics for different time periods.

        Args:
            hourly_data: Hourly time series
            periods: List of periods to calculate (D=daily, W=weekly, M=monthly, Q=quarterly, A=annual)

        Returns:
            DataFrame with statistics for each period
        """
        if not hourly_data.index.is_datetime64_any_dtype:
            hourly_data.index = DataAggregator.create_datetime_index()

        stats = []
        period_names = []

        for period in periods:
            if period == 'D':
                grouped = hourly_data.groupby(hourly_data.index.date)
                period_name = 'Daily'
            elif period == 'W':
                grouped = hourly_data.groupby(pd.Grouper(freq='W-MON'))
                period_name = 'Weekly'
            elif period == 'M':
                grouped = hourly_data.groupby(pd.Grouper(freq='M'))
                period_name = 'Monthly'
            elif period == 'Q':
                grouped = hourly_data.groupby(pd.Grouper(freq='Q'))
                period_name = 'Quarterly'
            elif period == 'A':
                grouped = hourly_data.groupby(pd.Grouper(freq='A'))
                period_name = 'Annual'
            else:
                continue

            period_stats = grouped.agg(['mean', 'sum', 'std', 'min', 'max']).head(10)  # First 10 periods

            for stat in ['mean', 'sum', 'std', 'min', 'max']:
                col_name = f"{period_name}_{stat}"
                stats.append(getattr(period_stats[stat], stat.lower()))
                period_names.append(col_name)

        summary_df = pd.DataFrame(stats).T
        summary_df.columns = period_names[:len(summary_df.columns)]

        return summary_df.head()

    @staticmethod
    def create_moving_averages(data: pd.Series,
                              windows: List[int] = [24, 168, 720],
                              center: bool = False) -> Dict[str, pd.Series]:
        """
        Calculate moving averages for different window sizes.

        Args:
            data: Time series data
            windows: List of window sizes in hours
            center: Whether to center the window

        Returns:
            Dictionary of moving averages
        """
        moving_avgs = {}

        for window in windows:
            moving_avgs[f'ma_{window}h'] = data.rolling(window=window, center=center).mean()

        return moving_avgs

    @staticmethod
    def identify_peak_periods(hourly_data: pd.Series,
                             method: str = 'percentile',
                             threshold: float = 0.95) -> pd.DataFrame:
        """
        Identify peak and low periods in the data.

        Args:
            hourly_data: Hourly time series
            method: Method for peak identification ('percentile', 'std')
            threshold: Threshold for peak identification

        Returns:
            DataFrame with peak period classifications
        """
        if method == 'percentile':
            peak_threshold = hourly_data.quantile(threshold)
            low_threshold = hourly_data.quantile(1 - threshold)
        elif method == 'std':
            peak_threshold = hourly_data.mean() + threshold * hourly_data.std()
            low_threshold = hourly_data.mean() - threshold * hourly_data.std()
        else:
            raise ValueError("Method must be 'percentile' or 'std'")

        peak_periods = hourly_data >= peak_threshold
        low_periods = hourly_data <= low_threshold
        normal_periods = ~(peak_periods | low_periods)

        result_df = pd.DataFrame({
            'value': hourly_data,
            'is_peak': peak_periods,
            'is_low': low_periods,
            'is_normal': normal_periods,
            'peak_threshold': peak_threshold,
            'low_threshold': low_threshold
        })

        return result_df

    @staticmethod
    def aggregate_financial_data(cashflows_df: pd.DataFrame,
                                period: str = 'A') -> pd.DataFrame:
        """
        Aggregate financial data (cash flows, costs) by period.

        Args:
            cashflows_df: DataFrame with yearly cash flow data
            period: Period for aggregation

        Returns:
            Aggregated financial DataFrame
        """
        # Assuming cashflows_df has columns like Year, Gen_CAPEX, Elec_CAPEX, OPEX, etc.
        numeric_cols = cashflows_df.select_dtypes(include=[np.number]).columns
        # For annual aggregation, group by year
        financial_agg = cashflows_df.set_index('Year').groupby(
            lambda x: x.year if hasattr(x, 'year') else x
        )[numeric_cols].sum()

        return financial_agg

    def create_comprehensive_data_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create comprehensive data summary with multiple aggregations.

        Args:
            model_results: Dictionary containing model results with time series data

        Returns:
            Dictionary with various aggregated summaries
        """
        summary = {
            'seasonal_averages': {},
            'period_statistics': {},
            'moving_averages': {},
            'peak_analysis': {},
            'financial_summary': {}
        }

        # Extract time series data
        if 'hourly_data' in model_results:
            hourly_df = model_results['hourly_data']

            # Seasonal averages for key metrics
            for col in ['Generator_CF', 'Electrolyser_CF', 'Hydrogen_prod_fixed']:
                if col in hourly_df.columns:
                    summary['seasonal_averages'][col] = self.calculate_seasonal_averages(hourly_df[col])

            # Summary statistics for different periods
            summary['period_statistics']['generator_cf'] = self.create_summary_statistics(hourly_df.get('Generator_CF', pd.Series()))

            # Moving averages
            if 'Generator_CF' in hourly_df.columns:
                summary['moving_averages']['generator_cf'] = self.create_moving_averages(hourly_df['Generator_CF'])

            # Peak analysis
            if 'Electrolyser_CF' in hourly_df.columns:
                summary['peak_analysis']['electrolyser_cf'] = self.identify_peak_periods(hourly_df['Electrolyser_CF'])

        # Financial aggregations
        if 'cash_flow_df' in model_results:
            cf_df = model_results['cash_flow_df']
            summary['financial_summary']['annual'] = self.aggregate_financial_data(cf_df, 'A')

        return summary


class UnitConverter:
    """Utility class for unit conversions commonly used in energy modeling."""

    @staticmethod
    def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert energy values between different units.

        Supported units: 'kWh', 'MWh', 'GWh', 'TWh', 'J', 'MJ', 'GJ'
        """
        # Base unit: kWh
        conversion_to_kwh = {
            'kWh': 1.0,
            'MWh': 1000.0,
            'GWh': 1000000.0,
            'TWh': 1000000000.0,
            'J': 1/3600000.0,  # Joules to kWh
            'MJ': 1000/3600000.0,
            'GJ': 1000000/3600000.0
        }

        conversion_from_kwh = {v: k for k, v in conversion_to_kwh.items()}

        if from_unit not in conversion_to_kwh or to_unit not in conversion_from_kwh:
            raise ValueError(f"Unsupported units: {from_unit} to {to_unit}")

        base_value = value * conversion_to_kwh[from_unit]
        converted_value = base_value * conversion_from_kwh[to_unit]

        return converted_value

    @staticmethod
    def convert_mass(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert mass values between different units.

        Supported units: 'g', 'kg', 't', 'ton', 'tonne', 'lb', 'kg/day', 't/year'
        """
        conversion_to_kg = {
            'g': 0.001,
            'kg': 1.0,
            't': 1000.0,
            'ton': 907.185,  # US short ton
            'tonne': 1000.0,
            'lb': 0.453592
        }

        # Handle time-based units
        if 'kg/day' == from_unit:
            base_value = value * conversion_to_kg['kg']
        elif 't/year' == from_unit:
            base_value = value * conversion_to_kg['t']
        elif 'kg/day' == to_unit:
            raise ValueError("Cannot convert to time-based units directly")
        elif 't/year' == to_unit:
            raise ValueError("Cannot convert to time-based units directly")
        else:
            base_value = value * conversion_to_kg.get(from_unit, 1.0)

        if to_unit in ['g', 'kg', 't', 'ton', 'tonne', 'lb']:
            converted_value = base_value / conversion_to_kg[to_unit]
        else:
            converted_value = base_value

        return converted_value

    @staticmethod
    def convert_power(value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert power values between different units.

        Supported units: 'W', 'kW', 'MW', 'GW'
        """
        conversion_to_watts = {
            'W': 1.0,
            'kW': 1000.0,
            'MW': 1000000.0,
            'GW': 1000000000.0
        }

        if from_unit not in conversion_to_watts or to_unit not in conversion_to_watts:
            raise ValueError(f"Unsupported power units: {from_unit} to {to_unit}")

        base_value = value * conversion_to_watts[from_unit]
        converted_value = base_value / conversion_to_watts[to_unit]

        return converted_value

    @staticmethod
    def convert_currency(value: float, from_currency: str = 'USD', to_currency: str = 'AUD',
                        exchange_rate: float = 1.0) -> float:
        """
        Convert currency values using exchange rate.

        Note: Uses static exchange rate - update as needed for real-time rates.
        """
        if from_currency == to_currency:
            return value

        # Simple conversion using exchange rate
        if from_currency == 'USD' and to_currency == 'AUD':
            return value * exchange_rate
        elif from_currency == 'AUD' and to_currency == 'USD':
            return value / exchange_rate
        else:
            raise ValueError(f"Unsupported currency conversion: {from_currency} to {to_currency}")

        return value

    @staticmethod
    def convert_unit_to_base(unit_str: str) -> Tuple[float, str]:
        """
        Parse unit strings with multipliers (e.g., '100kW' -> 100, 'kW').
        """
        import re
        match = re.match(r'(\d*\.?\d+)([a-zA-Z]+)', unit_str)
        if match:
            value, unit = match.groups()
            return float(value), unit
        return float(unit_str), ''


class FormatHelpers:
    """Utility class for formatting numbers and strings for display."""

    @staticmethod
    def format_number(value: Union[float, int], decimals: int = 2,
                     use_scientific: bool = False) -> str:
        """
        Format numbers with specified decimal places or scientific notation.
        """
        if use_scientific or abs(value) >= 1000000 or abs(value) < 0.001:
            return f"{value:.2e}"
        else:
            return f"{value:.{decimals}f}"

    @staticmethod
    def format_with_unit(value: Union[float, int], unit: str,
                        compact: bool = True, decimals: int = 2) -> str:
        """
        Format values with appropriate SI prefixes for readability.

        Args:
            value: Numeric value
            unit: Unit string (e.g., 'W', '$')
            compact: Whether to use K/M/B prefixes
            decimals: Number of decimal places
        """
        if not compact:
            return f"{FormatHelpers.format_number(value, decimals)} {unit}"

        abs_value = abs(value)

        if abs_value >= 1e12:
            formatted_value = value / 1e12
            prefix = 'T'
        elif abs_value >= 1e9:
            formatted_value = value / 1e9
            prefix = 'B'
        elif abs_value >= 1e6:
            formatted_value = value / 1e6
            prefix = 'M'
        elif abs_value >= 1e3:
            formatted_value = value / 1e3
            prefix = 'K'
        else:
            formatted_value = value
            prefix = ''

        formatted_number = FormatHelpers.format_number(formatted_value, decimals)
        return f"{formatted_number}{prefix} {unit}"

    @staticmethod
    def format_energy(value_mwh: float, compact: bool = True) -> str:
        """Format energy values with appropriate units."""
        return FormatHelpers.format_with_unit(value_mwh, 'MWh', compact)

    @staticmethod
    def format_mass(value_kg: float, compact: bool = True) -> str:
        """Format mass values with appropriate units."""
        if value_kg >= 1000:
            return FormatHelpers.format_with_unit(value_kg / 1000, 't', compact)
        else:
            return FormatHelpers.format_with_unit(value_kg, 'kg', compact)

    @staticmethod
    def format_cost(value: float, currency: str = '$', compact: bool = True) -> str:
        """Format cost values with currency symbol."""
        return FormatHelpers.format_with_unit(value, currency, compact)

    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """Format values as percentages."""
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_efficiency(value: float, decimals: int = 1) -> str:
        """Format efficiency values as percentages."""
        return f"{value * 100:.{decimals}f}%"

    @staticmethod
    def create_metric_display(value: Union[float, int], label: str,
                             unit: str = '', delta: Optional[Union[float, int]] = None,
                             delta_color: str = 'normal') -> Dict[str, Any]:
        """
        Create a metric display dictionary for Streamlit metric components.

        Args:
            value: The main value to display
            label: Label for the metric
            unit: Unit string
            delta: Optional delta value
            delta_color: Color for delta ('normal', 'inverse', 'off')

        Returns:
            Dictionary suitable for metric display
        """
        formatted_value = FormatHelpers.format_with_unit(value, unit) if unit else str(value)

        result = {
            'label': label,
            'value': formatted_value
        }

        if delta is not None:
            if isinstance(delta, (int, float)):
                delta_str = f"{delta:+.2f}"
                if unit:
                    delta_str += f" {unit}"
            else:
                delta_str = str(delta)

            result['delta'] = delta_str
            result['delta_color'] = delta_color

        return result

    @staticmethod
    def get_si_prefix(value: float) -> Tuple[float, str]:
        """
        Get SI prefix for a value.

        Returns:
            Tuple of (scaled_value, prefix)
        """
        abs_value = abs(value)

        if abs_value >= 1e12:
            return value / 1e12, 'T'
        elif abs_value >= 1e9:
            return value / 1e9, 'G'
        elif abs_value >= 1e6:
            return value / 1e6, 'M'
        elif abs_value >= 1e3:
            return value / 1e3, 'K'
        elif abs_value >= 1:
            return value, ''
        elif abs_value >= 1e-3:
            return value * 1e3, 'm'
        elif abs_value >= 1e-6:
            return value * 1e6, 'μ'
        else:
            return value * 1e9, 'n'

    @staticmethod
    def auto_scale_value(value: float, unit: str) -> str:
        """
        Automatically scale value with SI prefix.

        Args:
            value: Numeric value
            unit: Base unit (e.g., 'W', 'Hz')

        Returns:
            Formatted string with SI prefix
        """
        scaled_value, prefix = FormatHelpers.get_si_prefix(value)
        return f"{scaled_value:.2f} {prefix}{unit}"

# Convenience functions for common conversions
def convert_energy_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convenience function for energy conversion."""
    return UnitConverter.convert_energy(value, from_unit, to_unit)

def convert_mass_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convenience function for mass conversion."""
    return UnitConverter.convert_mass(value, from_unit, to_unit)

def format_large_number(value: float, unit: str = '') -> str:
    """Convenience function for formatting large numbers."""
    return FormatHelpers.format_with_unit(value, unit)

def format_metric(value: float, label: str, unit: str = '') -> Dict[str, Any]:
    """Convenience function for creating metric displays."""
    return FormatHelpers.create_metric_display(value, label, unit)