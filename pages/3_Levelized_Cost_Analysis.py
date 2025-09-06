import streamlit as st
from utils import add_s2d2_footer
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Levelized Cost Analysis")

st.header("Capital Expenditure (CAPEX) Analysis")

if st.session_state.model_results:
    results = st.session_state.model_results
    inputs_summary = results.get('inputs_summary', {})
    operating_outputs = results.get('operating_outputs', {})

    # Calculate CAPEX breakdown (sample data based on typical renewable hydrogen projects)
    solar_capacity = inputs_summary.get('nominal_solar_farm_capacity', 10)
    wind_capacity = inputs_summary.get('nominal_wind_farm_capacity', 0)
    battery_power = inputs_summary.get('battery_rated_power', 2)

    # CAPEX components (A$/kW) - scaled appropriately
    capex_breakdown = {
        'Solar PV System': solar_capacity * 1100,  # A$/kW
        'Wind Turbine System': wind_capacity * 1600,  # A$/kW
        'Battery Storage': battery_power * 400,  # A$/kW (4-hour system)
        'Electrolyser System': inputs_summary.get('nominal_electrolyser_capacity', 10) * 800,  # A$/kW
        'Electrical Infrastructure': (solar_capacity + wind_capacity) * 300,  # A$/kW
        'General Facilities': (solar_capacity + wind_capacity + inputs_summary.get('nominal_electrolyser_capacity', 10)) * 200,  # A$/kW
        'Engineering & Supervision': (solar_capacity + wind_capacity + inputs_summary.get('nominal_electrolyser_capacity', 10)) * 150,  # A$/kW
        'Other Costs': (solar_capacity + wind_capacity + inputs_summary.get('nominal_electrolyser_capacity', 10)) * 100,  # A$/kW
        'Owner\'s Costs': (solar_capacity + wind_capacity + inputs_summary.get('nominal_electrolyser_capacity', 10)) * 150   # A$/kW
    }

    # Remove zero values
    capex_breakdown = {k: v for k, v in capex_breakdown.items() if v > 0}
    total_capex = sum(capex_breakdown.values())

    # Create two columns for visualizations
    col1, col2 = st.columns([1, 1])

    with col1:
        # CAPEX Breakdown Pie Chart
        st.subheader("CAPEX Component Breakdown")

        labels = list(capex_breakdown.keys())
        values = list(capex_breakdown.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hoverinfo='label+percent+value',
            textinfo='percent',
            marker_colors=colors[:len(labels)],
            pull=[0.1 if label == 'Electrolyser System' else 0 for label in labels]  # Highlight electrolyser
        )])

        fig_pie.update_layout(
            title=f"CAPEX Breakdown (Total: ${total_capex:,.0f})",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # CAPEX Summary Metrics
        st.subheader("CAPEX Summary")

        capex_per_kw = total_capex / (solar_capacity + wind_capacity + inputs_summary.get('nominal_electrolyser_capacity', 10))
        capex_per_t_h2 = total_capex / operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 1)

        col2.metric(
            label="Total CAPEX",
            value=f"${total_capex:,.0f}"
        )

        col2.metric(
            label="CAPEX per kW Capacity",
            value=f"${capex_per_kw:.0f}/kW"
        )

        col2.metric(
            label="CAPEX per tonne H₂",
            value=f"${capex_per_t_h2:.0f}/t"
        )

        # CAPEX vs OPEX Comparison
        st.subheader("CAPEX vs OPEX Comparison")

        opex_breakdown = {
            'Operations & Maintenance': total_capex * 0.02,  # Assuming 2% of capex yearly
            'Water Costs': operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 1) * 50,
            'Insurance': total_capex * 0.005,
            'Other Expenses': total_capex * 0.015
        }

        total_opex = sum(opex_breakdown.values())

        # Create OPEX bar chart
        opex_labels = list(opex_breakdown.keys())
        opex_values = list(opex_breakdown.values())

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=opex_labels,
            y=opex_values,
            name='OPEX',
            marker_color='rgba(219, 64, 82, 0.7)'
        ))

        # Add CAPEX comparison line
        fig_bar.add_trace(go.Scatter(
            x=opex_labels,
            y=[total_capex] * len(opex_labels),
            mode='lines',
            name='Total CAPEX',
            line=dict(color='rgba(55, 128, 191, 1)', width=3)
        ))

        fig_bar.update_layout(
            title="CAPEX vs Annual OPEX Components",
            xaxis_title="",
            yaxis_title="Cost ($)",
            barmode='group',
            height=400
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Breakdown Table
    st.subheader("Detailed CAPEX Breakdown Table")

    capex_df = pd.DataFrame({
        'Component': list(capex_breakdown.keys()),
        'Cost ($)': list(capex_breakdown.values()),
        'Percentage': [f"{v/total_capex:.1%}" for v in capex_breakdown.values()]
    })

    st.dataframe(capex_df, use_container_width=True)

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

# Keep placeholders for future development
st.header("Future Development Areas")
with st.expander("Annual Operational Profile", expanded=False):
    st.write("Table/chart for annual operational profile - to be implemented")

with st.expander("Discounted and Non-Discounted Cash Flows", expanded=False):
    st.write("Table/chart for cash flows - to be implemented")

# Add S2D2 Lab footer
add_s2d2_footer()
