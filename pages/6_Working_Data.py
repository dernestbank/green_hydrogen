import streamlit as st
from utils import add_s2d2_footer
from src.models.hydrogen_model import HydrogenModel
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Working Data")

st.header("Hourly Electrolyser Operation Profiles")

# Sidebar filters
with st.sidebar:
    st.header("Operation Filters")

    time_range = st.sidebar.selectbox(
        "Time Period to Show",
        ["24 Hours", "7 Days", "30 Days", "Full Year"],
        index=1,
        help="Select the time period for detailed analysis"
    )

    data_view = st.sidebar.selectbox(
        "Primary Data View",
        ["Operation Profiles", "Energy Flows", "Production Rates"],
        index=0,
        help="Choose the main focus of the analysis"
    )

    show_battery = st.sidebar.checkbox("Show Battery Data", value=True)
    show_electrolyser = st.sidebar.checkbox("Show Electrolyser Data", value=True)
    show_generators = st.sidebar.checkbox("Show Generator Data", value=True)

if st.session_state.model_results:
    results = st.session_state.model_results
    inputs_summary = results.get('inputs_summary', {})
    operating_outputs = results.get('operating_outputs', {})

    # Try to get hourly operation data from session state first
    hourly_data = None

    # Check if we have stored hourly data in session state
    if hasattr(st.session_state, 'hourly_operation_data'):
        hourly_data = st.session_state.hourly_operation_data
    else:
        # Create model instance and generate hourly data
        try:
            model = HydrogenModel(
                location=inputs_summary.get('location', 'REZ-N1'),
                elec_type='PEM' if inputs_summary.get('electrolyser_type', 'PEM') == 'PEM' else 'AE',
                elec_capacity=inputs_summary.get('nominal_electrolyser_capacity', 10),
                solar_capacity=inputs_summary.get('nominal_solar_farm_capacity', 10),
                wind_capacity=inputs_summary.get('nominal_wind_farm_capacity', 0),
                battery_power=inputs_summary.get('battery_power_rating', 0),
                battery_hours=inputs_summary.get('battery_storage_duration', 0),
                spot_price=inputs_summary.get('hourly_electricity_price', 40)
            )
            hourly_data = model._calculate_hourly_operation()

            # Store in session state for future use
            st.session_state.hourly_operation_data = hourly_data
        except Exception as e:
            st.warning(f"Could not generate hourly operation data: {str(e)}")
            hourly_data = pd.DataFrame()

    # Ensure we have the required data
    if not hourly_data.empty:
        # Add timestamp column for visualization if not present
        if 'timestamp' not in hourly_data.columns:
            hourly_data['timestamp'] = pd.date_range(
                start='2023-01-01 00:00:00',
                periods=len(hourly_data),
                freq='H'
            )

        hourly_data['hour'] = range(len(hourly_data))

        # Generate battery SOC if battery is enabled but SOC is missing
        if (inputs_summary.get('battery_power_rating', 0) > 0 and
            'Battery_SOC' not in hourly_data.columns):

            # Simple battery SOC model for visualization
            battery_energy = (inputs_summary.get('battery_power_rating', 0) *
                            inputs_summary.get('battery_storage_duration', 4))
            max_soc = 0.95
            min_soc = 0.05

            # Mock battery state - in real model this would be calculated
            hourly_data['Battery_SOC'] = np.clip(
                np.sin(hourly_data['hour'] * 2 * np.pi / 8760) * 0.4 + 0.5,
                min_soc, max_soc
            )

            # Calculate battery net charge (simplified)
            battery_eff = 0.85
            surplus_power = (hourly_data['Generator_CF'] * inputs_summary.get('nominal_solar_farm_capacity', 10) -
                           hourly_data['Electrolyser_CF'] * inputs_summary.get('nominal_electrolyser_capacity', 10))

            hourly_data['Battery_Net_Charge'] = np.where(
                surplus_power > 0,
                surplus_power * battery_eff,
                surplus_power / battery_eff
            )
    else:
        # Create empty dataframe with required columns
        hourly_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=8760, freq='H'),
            'Generator_CF': [0.5] * 8760,
            'Electrolyser_CF': [0.3] * 8760,
            'Hydrogen_prod_fixed': [0.025] * 8760,
            'Hydrogen_prod_variable': [0.026] * 8760,
            'Battery_SOC': [0.5] * 8760 if inputs_summary.get('battery_power_rating', 0) > 0 else None
        }).dropna(axis=1, how='all')
        hourly_data['hour'] = range(len(hourly_data))

    # Filter data based on selected time range
    if time_range == "24 Hours":
        display_data = hourly_data.head(24).copy()
    elif time_range == "7 Days":
        display_data = hourly_data.head(24*7).copy()
    elif time_range == "30 Days":
        display_data = hourly_data.head(24*30).copy()
    else:
        display_data = hourly_data.copy()

    # Main content area
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Average Electrolyser Load Factor",
            value=f"{display_data['Electrolyser_CF'].mean():.1%}"
        )

    with col2:
        st.metric(
            label="Operating Hours",
            value=f"{(display_data['Electrolyser_CF'] > 0).sum()}"
        )

    with col3:
        avg_h2 = display_data['Hydrogen_prod_fixed'].mean() * inputs_summary.get('nominal_electrolyser_capacity', 10) * 1000
        st.metric(
            label="Avg Hourly H₂ Production",
            value=f"{avg_h2:.1f} kg"
        )

    # Primary data view
    st.subheader(f"📊 {data_view}")

    if data_view == "Operation Profiles":
        # Create subplot for operation profiles
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Generator Availability',
                'Electrolyser Load Factor',
                'Battery State of Charge' if show_battery else 'Surplus/Deficit Power'
            ],
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        # Generator availability
        if show_generators:
            fig.add_trace(
                go.Scatter(
                    x=display_data['timestamp'],
                    y=display_data['Generator_CF'],
                    mode='lines',
                    name='Generator CF',
                    line=dict(color='#1f77b4', width=1)
                ),
                row=1, col=1
            )

        # Electrolyser operation
        if show_electrolyser:
            fig.add_trace(
                go.Scatter(
                    x=display_data['timestamp'],
                    y=display_data['Electrolyser_CF'],
                    mode='lines',
                    name='Electrolyser Load',
                    line=dict(color='#ff7f0e', width=2)
                ),
                row=2, col=1
            )

            # Add min/max load lines
            fig.add_hline(y=0.1, line_dash="dash", line_color="red",
                          annotation_text="Min Load", row="all", col="all")
            fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                          annotation_text="Max Load", row="all", col="all")

        # Battery or power balance
        if show_battery and 'Battery_SOC' in display_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=display_data['timestamp'],
                    y=display_data['Battery_SOC'],
                    mode='lines',
                    name='Battery SOC',
                    line=dict(color='#2ca02c', width=2),
                    fill='tozeroy'
                ),
                row=3, col=1
            )
        else:
            # Power surplus/deficit
            generator_power = display_data['Generator_CF'] * inputs_summary.get('nominal_solar_farm_capacity', 10) + \
                            display_data['Generator_CF'] * inputs_summary.get('nominal_wind_farm_capacity', 0)
            electrolyser_power = display_data['Electrolyser_CF'] * inputs_summary.get('nominal_electrolyser_capacity', 10)

            surplus_power = generator_power - electrolyser_power

            fig.add_trace(
                go.Scatter(
                    x=display_data['timestamp'],
                    y=surplus_power,
                    mode='lines',
                    name='Power Surplus/Deficit',
                    line=dict(color='#d62728', width=2)
                ),
                row=3, col=1
            )

        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Capacity Factor", row=1, col=1)
        fig.update_yaxes(title_text="Load Factor", row=2, col=1)
        if show_battery and 'Battery_SOC' in display_data.columns:
            fig.update_yaxes(title_text="State of Charge", row=3, col=1)
        else:
            fig.update_yaxes(title_text="Power (MW)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

    elif data_view == "Energy Flows":
        st.subheader("⚡ Energy Flow Analysis")

        # Calculate energy flows
        gen_capacity = inputs_summary.get('nominal_solar_farm_capacity', 10) + inputs_summary.get('nominal_wind_farm_capacity', 0)
        elec_capacity = inputs_summary.get('nominal_electrolyser_capacity', 10)

        energy_flows = display_data.copy()
        energy_flows['generation_mw'] = energy_flows['Generator_CF'] * gen_capacity
        energy_flows['electrolyser_mw'] = energy_flows['Electrolyser_CF'] * elec_capacity
        energy_flows['surplus_mw'] = energy_flows['generation_mw'] - energy_flows['electrolyser_mw']

        col1, col2 = st.columns(2)

        with col1:
            # Energy generation profile
            fig_gen = go.Figure()
            fig_gen.add_trace(go.Scatter(
                x=energy_flows['timestamp'],
                y=energy_flows['generation_mw'],
                mode='lines',
                name='Generation',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_gen.add_trace(go.Scatter(
                x=energy_flows['timestamp'],
                y=energy_flows['electrolyser_mw'],
                mode='lines',
                name='Electrolyser Demand',
                line=dict(color='#ff7f0e', width=2)
            ))

            fig_gen.update_layout(
                title="Power Generation vs Demand",
                xaxis_title="Time",
                yaxis_title="Power (MW)",
                height=400
            )
            st.plotly_chart(fig_gen, use_container_width=True)

        with col2:
            # Energy balance
            avg_gen = energy_flows['generation_mw'].mean()
            avg_demand = energy_flows['electrolyser_mw'].mean()
            avg_surplus = energy_flows['surplus_mw'].mean()

            fig_balance = go.Figure()

            fig_balance.add_trace(go.Bar(
                x=['Generation', 'Electrolyser Demand', 'Surplus'],
                y=[avg_gen, avg_demand, max(0, avg_surplus)],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            ))

            if avg_surplus < 0:
                fig_balance.add_trace(go.Bar(
                    x=['Deficit'],
                    y=[abs(avg_surplus)],
                    marker_color=['#d62728']
                ))

            fig_balance.update_layout(
                title="Average Energy Balance",
                yaxis_title="Power (MW)",
                height=400
            )
            st.plotly_chart(fig_balance, use_container_width=True)

    elif data_view == "Production Rates":
        st.subheader("🧪 Hydrogen Production Analysis")

        # Calculate production rates
        production_data = display_data.copy()
        production_data['h2_production_kg_fixed'] = production_data['Hydrogen_prod_fixed'] * \
                                                   inputs_summary.get('nominal_electrolyser_capacity', 10) * 1000
        production_data['h2_production_kg_variable'] = production_data['Hydrogen_prod_variable'] * \
                                                      inputs_summary.get('nominal_electrolyser_capacity', 10) * 1000

        # Enhanced H2 Production Time Series Visualization
        st.subheader("🔬 Enhanced Hydrogen Production Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Production volatility analysis
            fixed_volatility = production_data['h2_production_kg_fixed'].std()
            var_volatility = production_data['h2_production_kg_variable'].std()

            st.metric("Fixed Production Volatility", f"{fixed_volatility:,.0f} kg/h")
            st.metric("Variable Production Volatility", f"{var_volatility:,.0f} kg/h")
            st.metric("Volatility Reduction",
                     f"{((fixed_volatility - var_volatility) / fixed_volatility * 100):.1f}%"
                     if fixed_volatility > 0 else "N/A")

        with col2:
            # Peak production analysis
            fixed_peak = production_data['h2_production_kg_fixed'].max()
            var_peak = production_data['h2_production_kg_variable'].max()

            st.metric("Peak Production (Fixed)", f"{fixed_peak:,.0f} kg/h")
            st.metric("Peak Production (Variable)", f"{var_peak:,.0f} kg/h")
            st.metric("Peak Increase", f"{((var_peak - fixed_peak) / fixed_peak * 100):.1f}%")

        # Production rate visualization
        fig_prod = go.Figure()

        fig_prod.add_trace(go.Scatter(
            x=production_data['timestamp'],
            y=production_data['h2_production_kg_fixed'],
            mode='lines',
            name='Fixed Efficiency',
            line=dict(color='#1f77b4', width=2)
        ))

        fig_prod.add_trace(go.Scatter(
            x=production_data['timestamp'],
            y=production_data['h2_production_kg_variable'],
            mode='lines',
            name='Variable Efficiency',
            line=dict(color='#ff7f0e', width=2)
        ))

        fig_prod.update_layout(
            title="Hourly Hydrogen Production Rate",
            xaxis_title="Time",
            yaxis_title="Production Rate (kg/hour)",
            height=500
        )

        st.plotly_chart(fig_prod, use_container_width=True)



    # Detailed hourly data table
    with st.expander("📋 Detailed Hourly Data", expanded=False):
        # Format data for display
        table_data = display_data.copy()
        table_data = table_data[['timestamp', 'Generator_CF', 'Electrolyser_CF',
                               'Hydrogen_prod_fixed', 'Hydrogen_prod_variable']]

        # Convert capacity factors to percentages for display
        table_data['Generator_CF'] = (table_data['Generator_CF'] * 100).round(1)
        table_data['Electrolyser_CF'] = (table_data['Electrolyser_CF'] * 100).round(1)
        table_data['Hydrogen_prod_fixed'] *= inputs_summary.get('nominal_electrolyser_capacity', 10) * 1000
        table_data['Hydrogen_prod_variable'] *= inputs_summary.get('nominal_electrolyser_capacity', 10) * 1000

        # Format timestamps
        table_data['timestamp'] = table_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

        table_data.columns = ['Timestamp', 'Generator CF (%)', 'Electrolyser Load (%)',
                            'H₂ Production Fixed (kg)', 'H₂ Production Variable (kg)']

        st.dataframe(table_data, use_container_width=True)

        # Export functionality
        if st.button("Export Hourly Data"):
            csv = table_data.to_csv(index=False)
            st.download_button(
                label="Download Hourly Data CSV",
                data=csv,
                file_name="hourly_operation_data.csv",
                mime="text/csv",
                key="download_hourly_csv"
            )

    # Operation summary
    st.subheader("📈 Operation Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            label="Electrolyser Utilization",
            value=f"{(display_data['Electrolyser_CF'] > 0).mean():.1%}"
        )

    with summary_col2:
        st.metric(
            label="Peak Generation Capacity Factor",
            value=f"{display_data['Generator_CF'].max():.1%}"
        )

    with summary_col3:
        if 'Battery_SOC' in display_data.columns and len(display_data) > 0:
            battery_range = display_data['Battery_SOC'].max() - display_data['Battery_SOC'].min()
            st.metric(
                label="Battery SOC Range",
                value=f"{battery_range:.1%}"
            )

else:
    st.info("Please go to the 'Inputs' page and run the model calculation first.")

# Add S2D2 Lab footer
add_s2d2_footer()
