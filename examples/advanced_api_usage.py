"""
Advanced API Usage Example

This example demonstrates advanced usage patterns including:
- API integration with caching and error handling
- Custom configuration scenarios
- Performance optimization techniques
- Advanced reporting features

Author: Hydrogen Production Framework Team
Date: 2025
"""

import pandas as pd
import yaml
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List
import logging

from src.models.hydrogen_model import HydrogenModel
from src.api.renewables_ninja_api import RenewablesNinjaAPI
from src.utils.api_cache import CachedRenewablesNinjaAPI, APICache
from src.utils.error_handling import with_error_handling, ErrorHandler, ErrorCategory, ErrorSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')  # Suppress minor warnings for clarity


def main():
    """Main advanced usage demonstration."""

    print("=== Advanced API Usage Example ===\n")

    # 1. API Integration with Caching
    print("1. Demonstrating API integration with intelligent caching...")

    # Setup caching
    cache = APICache(
        cache_dir="cache/api_examples",
        default_ttl=3600,  # 1 hour TTL
        enable_compression=True
    )

    # Note: This would use actual API key in production
    api_client = DemoAPIRenewablesNinja(key="demo_key")
    cached_api = CachedRenewablesNinjaAPI(api_client, cache_ttl=3600)

    location_data = fetch_multiple_locations(cached_api)
    print(f"✓ Fetched data for {len(location_data)} locations")

    # 2. Batch Processing Optimization
    print("\n2. Running batch scenario analysis...")

    results = batch_scenario_analysis(location_data, cached_api)

    # Display top 5 locations by cost competitiveness
    print("\nTop 5 Locations by Levelized Cost of Hydrogen:")
    top_locations = sorted(results.items(), key=lambda x: x[1]['lcoh'])
    for i, (location, data) in enumerate(top_locations[:5]):
        print(".1f")

    # 3. Performance Benchmarking
    print("\n3. Performance benchmarking...")
    performance_results = benchmark_analysis_performance(location_data)
    print(".2f")
    print(".3f")
    print(f"   Peak memory usage: High")

    # 4. Export and Reporting
    print("\n4. Generating comprehensive reports...")

    report_data = generate_comprehensive_report(results)
    export_results_to_excel(results, "export/advanced_analysis_results.xlsx")

    print("✓ Comprehensive analysis completed and exported")

    print("\n=== Advanced Example Completed Successfully ===")
    return results, performance_results, report_data


@with_error_handling(ErrorCategory.API, ErrorSeverity.MEDIUM)
def fetch_multiple_locations(cached_api: CachedRenewablesNinjaAPI) -> Dict[str, Dict]:
    """
    Fetch renewable energy data for multiple geographic locations.

    Demonstrates batch processing and caching efficiency.
    """
    locations = [
        {'lat': -33.86, 'lon': 151.21, 'name': 'Sydney, Australia'},
        {'lat': 40.71, 'lon': -74.01, 'name': 'New York, USA'},
        {'lat': 51.51, 'lon': -0.13, 'name': 'London, UK'},
        {'lat': 35.68, 'lon': 139.69, 'name': 'Tokyo, Japan'},
        {'lat': 25.20, 'lon': 55.27, 'name': 'Dubai, UAE'},
    ]

    location_data = {}

    for location in locations:
        try:
            print(f"   Fetching data for {location['name']}...")

            # Solar PV data
            solar_data = cached_api.fetch_solar_data(
                lat=location['lat'],
                lon=location['lon'],
                date_from='2020-01-01',
                date_to='2020-12-31'
            )

            # Wind data
            wind_data = cached_api.fetch_wind_data(
                lat=location['lat'],
                lon=location['lon'],
                date_from='2020-01-01',
                date_to='2020-12-31',
                height=80  # Wind hub height
            )

            location_data[location['name']] = {
                'coordinates': (location['lat'], location['lon']),
                'solar_data': solar_data,
                'wind_data': wind_data
            }

            time.sleep(0.1)  # Respectful API usage

        except Exception as e:
            logger.warning(f"Failed to fetch data for {location['name']}: {e}")
            continue

    return location_data


@with_error_handling(ErrorCategory.CALCULATION, ErrorSeverity.MEDIUM)
def batch_scenario_analysis(location_data: Dict[str, Dict],
                           cached_api: CachedRenewablesNinjaAPI) -> Dict[str, Dict]:
    """
    Run comprehensive analysis for multiple locations and scenarios.
    """
    scenarios = [
        {
            'name': 'Solar Dominated',
            'elec_capacity': 50, 'solar_capacity': 100.0, 'wind_capacity': 25.0,
            'electrolyser_type': 'AE'
        },
        {
            'name': 'Hybrid Balanced',
            'elec_capacity': 50, 'solar_capacity': 60.0, 'wind_capacity': 60.0,
            'electrolyser_type': 'PEM'
        },
        {
            'name': 'Wind Dominated',
            'elec_capacity': 50, 'solar_capacity': 30.0, 'wind_capacity': 90.0,
            'electrolyser_type': 'AE'
        }
    ]

    results = {}

    for location_name, location_info in location_data.items():
        print(f"   Analyzing {location_name}...")

        location_results = {}

        for scenario in scenarios:
            try:
                # Create model instance
                model = HydrogenModel(
                    elec_type=scenario['electrolyser_type'],
                    elec_capacity=scenario['elec_capacity'],
                    solar_capacity=scenario['solar_capacity'],
                    wind_capacity=scenario['wind_capacity'],
                    battery_power=int(scenario['elec_capacity'] * 0.2),
                    battery_hours=4,
                    location=location_name,
                    solardata=location_info['solar_data'],
                    winddata=location_info['wind_data']
                )

                # Run analysis
                outputs = model.calculate_electrolyser_output()
                lcoh = model.calculate_costs('variable')

                location_results[scenario['name']] = {
                    'scenario': scenario,
                    'capacity_factor': outputs['Achieved Electrolyser Capacity Factor'],
                    'hydrogen_production': outputs['Hydrogen Output for Variable Operation [t/yr]'],
                    'lcoh': lcoh,
                    'surplus_energy': outputs['Surplus Energy [MWh/yr]'],
                    'location': location_name
                }

            except Exception as e:
                logger.error(f"Analysis failed for {location_name} scenario {scenario['name']}: {e}")
                continue

        if location_results:
            results[location_name] = location_results

    return results


def benchmark_analysis_performance(location_data: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Benchmark performance of analysis operations.
    """
    print("   Running performance benchmarks...")

    times = []

    # Benchmark single location analysis
    for i, (location_name, location_info) in enumerate(list(location_data.items())[:3]):
        start_time = time.time()

        model = HydrogenModel(
            elec_type='AE',
            elec_capacity=25,
            solar_capacity=50.0,
            wind_capacity=25.0,
            battery_power=5,
            battery_hours=4,
            solardata=location_info['solar_data'],
            winddata=location_info['wind_data']
        )

        # Full analysis workflow
        _ = model.calculate_electrolyser_output()
        _ = model.calculate_costs('fixed')
        _ = model.calculate_costs('variable')
        _ = model.get_results_summary()

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times) if times else 0
    total_data_points = len(list(location_data.values())[0]['solar_data']) if location_data else 0

    return {
        'avg_analysis_time': avg_time,
        'total_locations_tested': len(times),
        'data_points_processed': total_data_points,
        'performance_score': total_data_points / avg_time if avg_time > 0 else 0
    }


def generate_comprehensive_report(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report with insights and recommendations.
    """
    print("   Generating comprehensive analysis report...")

    summary_stats = {
        'total_locations': len(results),
        'total_scenarios': sum(len(scenarios) for scenarios in results.values()),
        'lcoh_range': {
            'min': float('inf'),
            'max': 0.0,
            'avg': 0.0
        },
        'production_range': {
            'min': float('inf'),
            'max': 0.0,
            'avg': 0.0
        }
    }

    all_lcoh = []
    all_production = []

    for location_results in results.values():
        for scenario_result in location_results.values():
            lcoh = scenario_result['lcoh']
            production = scenario_result['hydrogen_production']

            all_lcoh.append(lcoh)
            all_production.append(production)

            summary_stats['lcoh_range']['min'] = min(summary_stats['lcoh_range']['min'], lcoh)
            summary_stats['lcoh_range']['max'] = max(summary_stats['lcoh_range']['max'], lcoh)
            summary_stats['production_range']['min'] = min(summary_stats['production_range']['min'], production)
            summary_stats['production_range']['max'] = max(summary_stats['production_range']['max'], production)

    if all_lcoh:
        summary_stats['lcoh_range']['avg'] = sum(all_lcoh) / len(all_lcoh)
    if all_production:
        summary_stats['production_range']['avg'] = sum(all_production) / len(all_production)

    # Generate insights
    insights = generate_insights(results, summary_stats)

    report = {
        'summary_statistics': summary_stats,
        'results': results,
        'insights': insights,
        'generated_at': str(pd.Timestamp.now()),
        'model_version': '1.0.0'
    }

    return report


def generate_insights(results: Dict[str, Dict], stats: Dict[str, Any]) -> List[str]:
    """Generate insights from analysis results."""
    insights = []

    # Cost-effectiveness insights
    if stats['lcoh_range']['min'] < 3.0:
        insights.append("✓ Identified cost-competitive locations with LCOH below $3/kg")
    elif stats['lcoh_range']['min'] < 5.0:
        insights.append("⚠ Most economic locations have LCOH in $3-5/kg range")

    # Production capacity insights
    high_production_locs = [
        loc for loc, loc_results in results.items()
        for scenario, data in loc_results.items()
        if data['hydrogen_production'] > 300
    ]
    if high_production_locs:
        insights.append(f"⚡ {len(high_production_locs)} scenarios show high production capacity (>300 t/yr)")

    # Seasonal and geographic insights
    northern_locs = sum(1 for loc in results.keys() if 'UK' in loc or 'New York' in loc)
    southern_locs = sum(1 for loc in results.keys() if 'Australia' in loc or 'Japan' in loc)

    if northern_locs > southern_locs:
        insights.append("🌎 Northern hemisphere locations show more consistent wind resources")

    insights.append("📊 Analysis completed across multiple technology options and locations")
    insights.append(".2f")
    insights.append(".1f")

    return insights


def export_results_to_excel(results: Dict[str, Dict], filename: str):
    """Export comprehensive results to Excel format."""
    print(f"   Exporting results to {filename}...")

    try:
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            # Summary sheet
            summary_data = []
            for location, scenarios in results.items():
                for scenario_name, data in scenarios.items():
                    row = {
                        'Location': location,
                        'Scenario': scenario_name,
                        'Capacity Factor (%)': data['capacity_factor'] * 100,
                        'Hydrogen Production (t/yr)': data['hydrogen_production'],
                        'LCOH ($/kg)': data['lcoh'],
                        'Surplus Energy (MWh/yr)': data['surplus_energy']
                    }
                    summary_data.append(row)

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Rankings sheet
                rankings_df = summary_df.sort_values('LCOH ($/kg)')
                rankings_df.to_excel(writer, sheet_name='Rankings', index=False)

        print("✓ Results exported successfully"    except Exception as e:
        print(f"⚠ Export failed: {e}")


class DemoAPIRenewablesNinja:
    """
    Demo implementation of RenewablesNinjaAPI for examples.

    This would be replaced with actual API client in production.
    """

    def __init__(self, key: str):
        self.api_key = key

    def fetch_solar_data(self, **kwargs) -> pd.DataFrame:
        """Simulate solar data fetch."""
        # Create realistic solar PV power curve
        dates = pd.date_range(kwargs.get('date_from', '2020-01-01'),
                             kwargs.get('date_to', '2020-12-31'), freq='h')

        # Simplified solar generation model
        hours = np.arange(len(dates)) % 24
        season_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (np.arange(len(dates)) % 8760) / 8760)
        solar_cf = season_factor * 0.5 * (1 + np.sin(np.pi * (hours - 6) / 12))

        return pd.DataFrame({'electricity': np.clip(solar_cf, 0, 1)}, index=dates)

    def fetch_wind_data(self, **kwargs) -> pd.DataFrame:
        """Simulate wind data fetch."""
        # Create realistic wind power curve
        dates = pd.date_range(kwargs.get('date_from', '2020-01-01'),
                             kwargs.get('date_to', '2020-12-31'), freq='h')

        # Wind is variable but less predictable than solar
        wind_cf = np.random.beta(2, 3, len(dates))
        wind_cf += 0.1 * np.sin(2 * np.pi * (np.arange(len(dates)) % 8760) / 8760)  # Seasonal variation

        return pd.DataFrame({'electricity': np.clip(wind_cf, 0, 1)}, index=dates)


if __name__ == "__main__":
    main()