# Hydrogen Production Framework - Usage Examples

This directory contains comprehensive examples demonstrating various usage patterns and capabilities of the Hydrogen Production Framework.

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [API Integration](#api-integration)
- [Configuration Management](#configuration-management)
- [Performance Optimization](#performance-optimization)
- [Advanced Scenarios](#advanced-scenarios)
- [Error Handling](#error-handling)
- [Export and Reporting](#export-and-reporting)

## Setup

### Install Dependencies
```bash
pip install -r requirement.txt
```

### API Key Configuration
1. Get your Renewables Ninja API key from https://www.renewables.ninja/user/api
2. Create a `.env` file in the project root:
```bash
# Create .env file
touch .env
```

3. Add your API key to the `.env` file:
```bash
# Content of .env file
RE_NINJA_API_KEY=your_actual_api_key_here
```

4. Test the API key loading:
```bash
python test_api_simple.py
```

### Automatic API Key Loading
The framework automatically loads your Renewables Ninja API key from the `.env` file:

```python
from src.utils.renewables_ninja_api import RenewablesNinjaAPI

# API key automatically loaded from .env - no manual setup needed!
api = RenewablesNinjaAPI()

# Ready to use
solar_data = api.get_solar_data(lat=-33.86, lon=151.21)
```

**API Key Priority Order:**
1. Direct parameter: `RenewablesNinjaAPI(api_key="your_key")`
2. Environment variable: `export RE_NINJA_API_KEY=your_key`
3. `.env` file: `RE_NINJA_API_KEY=your_key`
4. Error if none found

## Quick Start

```python
from src.models.hydrogen_model import HydrogenModel

# Initialize model with basic configuration
model = HydrogenModel(
    elec_type='AE',
    elec_capacity=50,      # 50 MW electrolyser
    solar_capacity=100.0,  # 100 MW solar
    wind_capacity=50.0,    # 50 MW wind
)

# Run basic analysis
outputs = model.calculate_electrolyser_output()
lcoh = model.calculate_costs('fixed')
summary = model.get_results_summary()
```

## Basic Usage

See [`basic_usage.py`](basic_usage.py) for a comprehensive introduction to:

- Loading configuration files
- Creating renewable energy data
- Running operational analysis
- Calculating Levelized Cost of Hydrogen (LCOH)
- Interpreting results and KPIs

**Key Features Demonstrated:**
- Model initialization with custom parameters
- Basic operational analysis
- Cost calculation methods
- Results interpretation

## API Integration

See [`advanced_api_usage.py`](advanced_api_usage.py) for advanced API usage patterns:

### Caching API Responses
```python
from src.utils.api_cache import APICache, CachedRenewablesNinjaAPI

# Create cached API client (API key automatically loaded from .env)
cache = APICache(cache_dir="cache/api_responses", default_ttl=3600)
api_client = RenewablesNinjaAPI()
cached_api = CachedRenewablesNinjaAPI(api_client, cache_ttl=3600)

# Fetch data (automatically cached)
solar_data = cached_api.fetch_solar_data(lat=-33.86, lon=151.21)
```

**Note:** The API key is automatically loaded from your `.env` file (see [Setup](#setup) section above).

### Batch Processing
```python
# Fetch data for multiple locations
locations = [
    {'name': 'Sydney', 'lat': -33.86, 'lon': 151.21},
    {'name': 'London', 'lat': 51.51, 'lon': -0.13}
]

for location in locations:
    solar_data = cached_api.fetch_solar_data(**location)
    wind_data = cached_api.fetch_wind_data(**location)
```

## Configuration Management

### Using YAML Configuration Files
```python
from src.models.hydrogen_model import HydrogenModel

# Model automatically loads config/config.yaml
model = HydrogenModel()

# Override specific parameters
model = HydrogenModel(
    elec_capacity=100,
    solar_capacity=200.0,
    # Other parameters use config file defaults
)
```

### Configuration Schema
The configuration file supports:

- **Electrolyser parameters**: AE and PEM specific settings
- **Cost parameters**: CAPEX and OPEX by component
- **Battery parameters**: Storage capacity and costs
- **Financial parameters**: Discount rates, project life
- **Operational bounds**: Minimum/maximum loads, efficiency

## Performance Optimization

### Intelligent Caching
```python
from src.utils.cache_config_loader import CacheConfigLoader
from src.utils.cache_strategies import SmartCacheManager

# Load optimized cache configuration
config_loader = CacheConfigLoader()
smart_manager = config_loader.create_smart_cache_manager()
```

### Data Compression
```python
from src.utils.data_compression import CompressionManager

compressor = CompressionManager(enable_compression=True)
compressed_data, result = compressor.compress_data(large_dataframe)

print(".1f")
```

### Batch Processing
```python
# Process multiple scenarios efficiently
scenarios = [
    {'solar': 50, 'wind': 50, 'capacity': 40},
    {'solar': 80, 'wind': 30, 'capacity': 40},
    {'solar': 30, 'wind': 80, 'capacity': 40}
]

results = []
for scenario in scenarios:
    model = HydrogenModel(
        solar_capacity=scenario['solar'],
        wind_capacity=scenario['wind'],
        elec_capacity=scenario['capacity']
    )

    # Process in batch
    outputs = model.calculate_electrolyser_output()
    results.append(outputs)
```

## Advanced Scenarios

### Hybrid Renewable Systems
```python
# Optimal mix of solar and wind
optimal_model = HydrogenModel(
    solar_capacity=120.0,  # 120 MW solar
    wind_capacity=80.0,    # 80 MW wind
    elec_capacity=60,      # 60 MW electrolyser
    battery_power=15,      # 15 MW battery
    battery_hours=6        # 6 hours storage
)
```

### Demand-Response Optimization
```python
# Model with variable electricity prices
smart_model = HydrogenModel(
    spot_price=75.0,       # $75/MWh spot price
    battery_power=10,
    battery_hours=4
)

# Optimize for electricity prices
outputs = smart_model.calculate_electrolyser_output()
cost_savings = outputs['Surplus Energy [MWh/yr]'] * smart_model.spotPrice
```

### Sensitivity Analysis
```python
import numpy as np

# Test different electrolyser capacities
capacities = [10, 25, 50, 100]
results = {}

for capacity in capacities:
    model = HydrogenModel(elec_capacity=capacity, solar_capacity=capacity*2)
    outputs = model.calculate_electrolyser_output()
    costs = model.calculate_costs('variable')

    results[capacity] = {
        'capacity_factor': outputs['Achieved Electrolyser Capacity Factor'],
        'lcoh': costs,
        'production': outputs['Hydrogen Output for Variable Operation [t/yr]']
    }
```

## Error Handling

### Using Error Handler
```python
from src.utils.error_handling import with_error_handling, ErrorCategory, ErrorSeverity

@with_error_handling(ErrorCategory.CALCULATION, ErrorSeverity.MEDIUM)
def run_sensitivity_analysis():
    # Your analysis code here
    pass

try:
    run_sensitivity_analysis()
except Exception as e:
    print(f"Analysis failed: {e}")
```

### Validation
```python
from src.utils.error_handling import validate_numeric_input

# Validate user inputs
capacity = validate_numeric_input(50, min_val=1, max_val=1000, value_name="electrolyser capacity")
solar_capacity = validate_numeric_input(100.0, min_val=0, value_name="solar capacity")
```

## Export and Reporting

### PDF Reports
```python
from src.utils.pdf_report_generator import create_pdf_report

# Generate comprehensive PDF report
pdf_data = create_pdf_report(model_results, charts_data)
with open('reports/hydrogen_analysis.pdf', 'wb') as f:
    f.write(pdf_data)
```

### Excel Exports
```python
from src.utils.data_export_manager import create_complete_data_export

# Export all analysis data
excel_data = create_complete_data_export(session_data, format_type="excel")
with open('reports/complete_analysis.xlsx', 'wb') as f:
    f.write(excel_data)
```

### JSON Export
```python
# Export scenario comparisons
import json

with open('reports/scenario_comparison.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
```

## Best Practices

### Model Initialization
1. Always validate configuration files
2. Use appropriate battery storage durations
3. Consider geographical variations in renewable resources

### Performance Tips
1. Enable caching for repeated API calls
2. Use appropriate data formats for large datasets
3. Batch process similar scenarios together

### Error Prevention
1. Validate all input parameters
2. Handle API rate limits appropriately
3. Monitor memory usage for large analyses

## Configuration Examples

### Australia (High Solar Potential)
```python
australia_config = {
    'location': 'US.CA',
    'solar_capacity': 150.0,  # High solar irradiance
    'wind_capacity': 50.0,    # Moderate wind
    'elec_type': 'PEM',       # PEM preferred for solar synergy
    'grid_connection': True
}
```

### Northern Europe (High Wind Potential)
```python
nordic_config = {
    'location': 'Oslo',
    'solar_capacity': 30.0,   # Lower solar irradiance
    'wind_capacity': 200.0,   # High wind potential
    'elec_type': 'AE',        # AE more economical for wind
    'battery_hours': 8        # Longer battery storage
}
```

### Grid-Connected Systems
```python
grid_config = {
    'ppa_price': 65.0,        # Fixed purchase price
    'solar_capacity': 50.0,   # Smaller solar for balancing
    'wind_capacity': 50.0,    # Smaller wind for balancing
    'elec_capacity': 30,
    'grid_backup': True
}
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce dataset size or enable compression
2. **API Rate Limits**: Use cached API clients with appropriate TTL
3. **Configuration Errors**: Validate configuration files with schema
4. **Type Errors**: Check parameter types match expected values

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
model = HydrogenModel(debug=True)
```

## Related Resources

- [API Documentation](../docs/api_reference.md)
- [Configuration Guide](../docs/configuration.md)
- [Performance Guide](../docs/performance_optimization.md)
- [Mathematical Models](../docs/mathematical_models.md)

## License

This example code is provided under the same license as the Hydrogen Production Framework.