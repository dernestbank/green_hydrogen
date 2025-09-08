"""
Results Comparison Tool for Hydrogen Cost Analysis Tool
Enables side-by-side comparison of different analysis scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ResultsComparisonTool:
    """
    Comprehensive results comparison tool for hydrogen production scenarios.

    Features:
    - Side-by-side comparison of multiple scenarios
    - Difference analysis and percentage changes
    - Ranking and prioritization of scenarios
    - Visualization-ready comparison data
    - Export comparison results
    """

    def __init__(self):
        self.scenarios = {}
        self.comparison_results = {}
        self.baseline_scenario = None

    def add_scenario(self, name: str, scenario_data: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a scenario to the comparison tool.

        Args:
            name: Unique scenario name
            scenario_data: Complete scenario results data
            metadata: Optional metadata for the scenario

        Returns:
            True if added successfully
        """
        try:
            scenario = {
                "name": name,
                "data": scenario_data,
                "metadata": metadata or {},
                "added_date": datetime.now().isoformat()
            }

            # Extract key metrics for quick comparison
            scenario["key_metrics"] = self._extract_key_metrics(scenario_data)

            self.scenarios[name] = scenario

            # Set first scenario as baseline if none exists
            if self.baseline_scenario is None:
                self.baseline_scenario = name

            logger.info(f"Scenario '{name}' added to comparison")
            return True

        except Exception as e:
            logger.error(f"Error adding scenario '{name}': {e}")
            return False

    def set_baseline_scenario(self, scenario_name: str) -> bool:
        """
        Set the baseline scenario for percentage calculations.

        Args:
            scenario_name: Name of scenario to use as baseline

        Returns:
            True if set successfully
        """
        if scenario_name in self.scenarios:
            self.baseline_scenario = scenario_name
            logger.info(f"Baseline scenario set to: {scenario_name}")
            return True
        else:
            logger.error(f"Scenario '{scenario_name}' not found")
            return False

    def compare_scenarios(self, metric_groups: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare all scenarios across specified metric groups.

        Args:
            metric_groups: Specific metric groups to compare (optional)

        Returns:
            Comparison results dictionary
        """
        if len(self.scenarios) < 2:
            logger.warning("Need at least 2 scenarios to compare")
            return {}

        comparison_results = {}
        baseline_data = self.scenarios[self.baseline_scenario]["key_metrics"]

        for scenario_name, scenario in self.scenarios.items():
            if scenario_name == self.baseline_scenario:
                continue

            comparison_results[scenario_name] = self._compare_with_baseline(
                baseline_data, scenario["key_metrics"]
            )

        self.comparison_results = comparison_results
        return self._format_comparison_results(comparison_results)

    def _extract_key_metrics(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from scenario data."""
        metrics = {}

        # Extract from operating outputs
        if "operating_outputs" in scenario_data:
            op_out = scenario_data["operating_outputs"]
            metrics.update({
                "annual_h2_production_t": op_out.get("Hydrogen Output for Fixed Operation [t/yr]", 0),
                "generator_capacity_factor": op_out.get("Generator Capacity Factor", 0),
                "electrolyser_capacity_factor": op_out.get("Achieved Electrolyser Capacity Factor", 0),
                "time_rated_capacity": op_out.get("Time Electrolyser is at its Rated Capacity", 0),
                "energy_electrolyser": op_out.get("Energy in to Electrolyser [MWh/yr]", 0),
                "surplus_energy": op_out.get("Surplus Energy [MWh/yr]", 0)
            })

        # Extract financial metrics
        if "lcoh" in scenario_data:
            lcoh_data = scenario_data["lcoh"]
            metrics["lcoh_fixed"] = lcoh_data.get("fixed", 0)
            metrics["lcoh_variable"] = lcoh_data.get("variable", 0)

        if "npv" in scenario_data:
            metrics["npv"] = scenario_data["npv"]

        # Extract input configuration
        if "inputs_summary" in scenario_data:
            inputs = scenario_data["inputs_summary"]
            metrics["electrolyser_capacity"] = inputs.get("nominal_electrolyser_capacity", 0)
            metrics["solar_capacity"] = inputs.get("nominal_solar_farm_capacity", 0)
            metrics["wind_capacity"] = inputs.get("nominal_wind_farm_capacity", 0)
            metrics["battery_power"] = inputs.get("battery_power_rating", 0)

        return metrics

    def _compare_with_baseline(self, baseline_metrics: Dict[str, Any],
                             scenario_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare scenario metrics with baseline."""
        comparison = {}

        for key in set(baseline_metrics.keys()) | set(scenario_metrics.keys()):
            baseline_value = baseline_metrics.get(key, 0)
            scenario_value = scenario_metrics.get(key, 0)

            comparison[key] = {
                "baseline": baseline_value,
                "scenario": scenario_value,
                "difference": scenario_value - baseline_value,
                "percentage_change": ((scenario_value - baseline_value) / baseline_value * 100
                                    if baseline_value != 0 else 0 if scenario_value == 0 else float('inf'))
            }

        return comparison

    def _format_comparison_results(self, raw_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Format comparison results for presentation."""
        formatted_results = {
            "summary": {},
            "detailed_comparison": raw_comparison,
            "ranking": self._rank_scenarios(raw_comparison),
            "recommendations": self._generate_recommendations(raw_comparison)
        }

        # Summary statistics
        if raw_comparison:
            metrics_summary = {}
            for metric in ["annual_h2_production_t", "lcoh_fixed", "npv"]:
                values = [comp.get(metric, {}).get("percentage_change", 0)
                         for comp in raw_comparison.values()]
                metrics_summary[metric] = {
                    "avg_change": np.mean(values),
                    "min_change": min(values),
                    "max_change": max(values)
                }

            formatted_results["summary"] = metrics_summary

        return formatted_results

    def _rank_scenarios(self, comparison_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank scenarios based on performance metrics."""
        scenario_scores = []

        for scenario_name, metrics in comparison_data.items():
            score = 0

            # Positive factors (higher is better)
            h2_change = metrics.get("annual_h2_production_t", {}).get("percentage_change", 0)
            score += h2_change * 0.3  # 30% weight for H2 production

            npv_change = metrics.get("npv", {}).get("percentage_change", 0)
            score += npv_change * 0.4  # 40% weight for NPV

            # Negative factors (lower is better - so we use negative change)
            lcoh_change = metrics.get("lcoh_fixed", {}).get("percentage_change", 0)
            score -= lcoh_change * 0.3  # 30% weight for LCOH (negative because lower is better)

            scenario_scores.append({
                "scenario": scenario_name,
                "score": score,
                "rank": 0,  # Will be set below
                "key_metrics": {
                    "h2_production_change": h2_change,
                    "npv_change": npv_change,
                    "lcoh_change": lcoh_change
                }
            })

        # Sort by score and assign ranks
        scenario_scores.sort(key=lambda x: x["score"], reverse=True)

        for i, scenario in enumerate(scenario_scores):
            scenario["rank"] = i + 1

        return scenario_scores

    def _generate_recommendations(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        if not comparison_data:
            return recommendations

        # Check for cost-effective scenarios
        cost_improvements = []
        for scenario_name, metrics in comparison_data.items():
            lcoh_change = metrics.get("lcoh_fixed", {}).get("percentage_change", 0)
            if lcoh_change < -5:  # 5% reduction in LCOH
                cost_improvements.append(scenario_name)

        if cost_improvements:
            recommendations.append(f"Consider implementing scenarios: {', '.join(cost_improvements)} "
                                "which show significant LCOH reductions.")

        # Check for high production scenarios
        high_production = []
        for scenario_name, metrics in comparison_data.items():
            h2_change = metrics.get("annual_h2_production_t", {}).get("percentage_change", 0)
            if h2_change > 10:  # 10% increase in production
                high_production.append(scenario_name)

        if high_production:
            recommendations.append(f"Scenarios {', '.join(high_production)} offer substantial "
                                "production increases.")

        # Check for NPV improvements
        npv_improvements = []
        for scenario_name, metrics in comparison_data.items():
            npv_change = metrics.get("npv", {}).get("percentage_change", 0)
            if npv_change > 15:  # 15% NPV improvement
                npv_improvements.append(scenario_name)

        if npv_improvements:
            recommendations.append(f"Strong financial performance from: {', '.join(npv_improvements)} "
                                "with significant NPV improvements.")

        if not recommendations:
            recommendations.append("No significant differences detected between scenarios. "
                                "Consider adjusting scenario parameters for more meaningful comparisons.")

        return recommendations

    def get_comparison_dataframe(self, metric: str) -> pd.DataFrame:
        """
        Get comparison data as a pandas DataFrame for a specific metric.

        Args:
            metric: Metric name to compare

        Returns:
            DataFrame with comparison data
        """
        if not self.comparison_results:
            self.compare_scenarios()

        data = []

        for scenario_name, metrics in self.comparison_results.items():
            if metric in metrics:
                metric_data = metrics[metric]
                data.append({
                    "Scenario": scenario_name,
                    "Baseline": metric_data["baseline"],
                    "Compared": metric_data["scenario"],
                    "Difference": metric_data["difference"],
                    "Change (%)": metric_data["percentage_change"]
                })

        return pd.DataFrame(data)

    def export_comparison_results(self, export_path: str, format_type: str = "excel") -> str:
        """
        Export comparison results to file.

        Args:
            export_path: Path to export file
            format_type: Export format ('excel', 'csv', 'json')

        Returns:
            Path where file was saved
        """
        if format_type.lower() == "excel":
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # Summary
                summary_df = pd.DataFrame([self.comparison_results.get("summary", {})])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Rankings
                if "ranking" in self.comparison_results:
                    ranking_df = pd.DataFrame(self.comparison_results["ranking"])
                    ranking_df.to_excel(writer, sheet_name='Rankings', index=False)

                # Detailed comparisons for each scenario
                for scenario_name, metrics in self.comparison_results.get("detailed_comparison", {}).items():
                    df_data = []
                    for metric_name, metric_data in metrics.items():
                        df_data.append({
                            "Metric": metric_name,
                            "Baseline": metric_data.get("baseline"),
                            "Scenario": metric_data.get("scenario"),
                            "Difference": metric_data.get("difference"),
                            "Change (%)": metric_data.get("percentage_change")
                        })

                    if df_data:
                        metric_df = pd.DataFrame(df_data)
                        safe_sheet_name = scenario_name.replace(" ", "_")[:30]
                        metric_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)

        elif format_type.lower() == "csv":
            # Export as CSV
            comparison_df = self.get_comparison_dataframe("annual_h2_production_t")  # Default metric
            comparison_df.to_csv(export_path, index=False)

        elif format_type.lower() == "json":
            import json
            with open(export_path, 'w') as f:
                json.dump(self.comparison_results, f, indent=2, default=str)

        else:
            raise ValueError("Unsupported export format. Use 'excel', 'csv', or 'json'")

        logger.info(f"Comparison results exported to: {export_path}")
        return export_path

    def clear_scenarios(self):
        """Clear all scenarios from the comparison tool."""
        self.scenarios = {}
        self.comparison_results = {}
        self.baseline_scenario = None
        logger.info("All scenarios cleared from comparison tool")

    def get_scenario_summary(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information for a specific scenario.

        Args:
            scenario_name: Name of scenario

        Returns:
            Scenario summary or None if not found
        """
        if scenario_name in self.scenarios:
            scenario = self.scenarios[scenario_name]
            return {
                "name": scenario["name"],
                "metadata": scenario["metadata"],
                "key_metrics": scenario["key_metrics"],
                "added_date": scenario["added_date"]
            }
        return None

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """
        List all scenarios in the comparison tool.

        Returns:
            List of scenario information
        """
        return [{
            "name": scenario["name"],
            "description": scenario["metadata"].get("description", ""),
            "added_date": scenario["added_date"],
            "metric_count": len(scenario["key_metrics"])
        } for scenario in self.scenarios.values()]

# Factory function
def create_comparison_tool() -> ResultsComparisonTool:
    """
    Factory function to create a ResultsComparisonTool instance.

    Returns:
        ResultsComparisonTool instance
    """
    return ResultsComparisonTool()

def compare_multiple_scenarios(scenarios: Dict[str, Dict[str, Any]],
                             baseline: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare multiple scenarios in a single call.

    Args:
        scenarios: Dictionary of scenario data
        baseline: Optional baseline scenario name

    Returns:
        Comparison results
    """
    tool = ResultsComparisonTool()

    # Add all scenarios
    for name, data in scenarios.items():
        tool.add_scenario(name, data)

    # Set baseline if specified
    if baseline and baseline in scenarios:
        tool.set_baseline_scenario(baseline)

    # Run comparison
    return tool.compare_scenarios()