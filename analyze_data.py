import pandas as pd
import numpy as np

# Load the CSV files
solar_df = pd.read_csv('Data/solar-traces.csv', header=[0], skiprows=[1], index_col=0)
wind_df = pd.read_csv('Data/wind-traces.csv', header=[0], skiprows=[1], index_col=0)

print('SOLAR DATA ANALYSIS')
print('Columns:', list(solar_df.columns))
print('Sample data for different locations:')
for col in ['US.CA', 'US.TX', 'US.NM', 'US.MT']:
    if col in solar_df.columns:
        print(f'{col}: mean={solar_df[col].mean():.6f}, std={solar_df[col].std():.6f}, min={solar_df[col].min():.6f}, max={solar_df[col].max():.6f}')

print()
print('WIND DATA ANALYSIS')
print('Columns:', list(wind_df.columns))
for col in ['US.CA', 'US.TX', 'US.NM', 'US.MT']:
    if col in wind_df.columns:
        print(f'{col}: mean={wind_df[col].mean():.6f}, std={wind_df[col].std():.6f}, min={wind_df[col].min():.6f}, max={wind_df[col].max():.6f}')

print()
print('CORRELATION ANALYSIS')
solar_cols = [col for col in solar_df.columns if col.startswith('US.')]
wind_cols = [col for col in wind_df.columns if col.startswith('US.')]

if len(solar_cols) > 1:
    solar_corr = solar_df[solar_cols].corr()
    print('Solar correlation matrix:')
    print(solar_corr)

if len(wind_cols) > 1:
    wind_corr = wind_df[wind_cols].corr()
    print('Wind correlation matrix:')
    print(wind_corr)
