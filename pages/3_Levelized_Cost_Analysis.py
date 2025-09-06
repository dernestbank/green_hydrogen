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

st.header("Operating Expenditure (OPEX) Timeline Analysis")

if st.session_state.model_results:
    results = st.session_state.model_results
    inputs_summary = results.get('inputs_summary', {})
    operating_outputs = results.get('operating_outputs', {})

    project_life = 20  # Assumed 20-year project life
    years = list(range(1, project_life + 1))

    # Calculate OPEX components over time
    base_electrolyser_capacity = inputs_summary.get('nominal_electrolyser_capacity', 10)
    annual_h2 = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 100)

    # OPEX components (per year)
    opex_components = {
        'Operations & Maintenance': [base_electrolyser_capacity * 25 * (1.02 ** (year_idx)) for year_idx in range(project_life)],  # 2% escalation
        'Water Costs': [annual_h2 * 50] * project_life,
        'Insurance': [base_electrolyser_capacity * 8] * project_life,
        'Other Expenses': [base_electrolyser_capacity * 12] * project_life,
        'Stack Replacement': [0] * project_life
    }

    # Add stack replacement costs (every 5 years, cost of 40% of electrolyser capex)
    stack_cost = base_electrolyser_capacity * 800 * 0.4  # 40% of electrolyser CAPEX
    for year in range(5, project_life + 1, 5):  # Years 5, 10, 15, 20
        opex_components['Stack Replacement'][year - 1] = stack_cost

    # Calculate total OPEX per year
    total_opex = [sum([opex_components[comp][i] for comp in opex_components.keys()]) for i in range(project_life)]
    cumulative_opex = np.cumsum(total_opex)

    # Create two columns for OPEX visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OPEX Component Breakdown Over Time")

        # Stacked area chart for OPEX components
        fig_area = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (component, values) in enumerate(opex_components.items()):
            fig_area.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines',
                name=component,
                stackgroup='one',
                line=dict(color=colors[i % len(colors)])
            ))

        fig_area.update_layout(
            title="OPEX Components Over Project Life",
            xaxis_title="Year",
            yaxis_title="Annual OPEX ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig_area, use_container_width=True)

        # OPEX Summary Metrics
        avg_annual_opex = np.mean(total_opex)
        total_opex_20year = sum(total_opex)
        opex_percentage_of_capex = total_opex_20year / (inputs_summary.get('nominal_electrolyser_capacity', 10) * 1100 + inputs_summary.get('nominal_electrolyser_capacity', 10) * 800) * 100

        st.metric(
            label="Average Annual OPEX",
            value=f"${avg_annual_opex:,.0f}"
        )

        st.metric(
            label="Total OPEX (20 years)",
            value=f"${total_opex_20year:,.0f}"
        )

        st.metric(
            label="OPEX/CAPEX Ratio",
            value=f"{opex_percentage_of_capex:.1f}%"
        )

    with col2:
        st.subheader("Cumulative OPEX & Cash Flow Analysis")

        # Combined chart: cumulative OPEX and revenue
        fig_combined = go.Figure()

        # Cumulative OPEX
        fig_combined.add_trace(go.Scatter(
            x=years,
            y=cumulative_opex,
            mode='lines+markers',
            name='Cumulative OPEX',
            line=dict(color='#d62728', width=3)
        ))

        # Revenue (simplified: hydrogen selling price * production)
        h2_price_per_tonne = 350  # AUD per tonne
        annual_revenue = [annual_h2 * h2_price_per_tonne] * project_life
        cumulative_revenue = np.cumsum(annual_revenue)

        fig_combined.add_trace(go.Scatter(
            x=years,
            y=cumulative_revenue,
            mode='lines+markers',
            name='Cumulative Revenue',
            line=dict(color='#2ca02c', width=3)
        ))

        # Net cash position
        net_cumulative = cumulative_revenue - cumulative_opex

        fig_combined.add_trace(go.Scatter(
            x=years,
            y=net_cumulative,
            mode='lines+markers',
            name='Net Cumulative Cash Flow',
            line=dict(color='#1f77b4', width=4)
        ))

        fig_combined.update_layout(
            title="Cumulative Cash Flow Analysis",
            xaxis_title="Year",
            yaxis_title="Cumulative Value ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig_combined, use_container_width=True)

        # Net Present Value calculation
        discount_rate = 0.04
        discounted_net_cf = [net_cumulative[i] / ((1 + discount_rate) ** (i + 1)) for i in range(project_life)]
        total_discounted_cf = sum(discounted_net_cf)

        st.metric(
            label="Present Value of 20yr Net Cash Flow",
            value=f"${total_discounted_cf:,.0f}"
        )

    # Detailed OPEX breakdown table
    st.subheader("Detailed OPEX Breakdown by Year")

    opex_df = pd.DataFrame({
        'Year': years,
        **opex_components,
        'Total OPEX': total_opex
    })

    # Format monetary columns
    monetary_cols = [col for col in opex_df.columns if col != 'Year']
    opex_df[monetary_cols] = opex_df[monetary_cols].round(0)

    st.dataframe(opex_df.style.format({col: "${:,.0f}" for col in monetary_cols}), use_container_width=True)

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

# Keep placeholders for future development
st.header("Future Development Areas")
with st.expander("Annual Operational Profile", expanded=False):
    st.write("Table/chart for annual operational profile - to be implemented")

with st.expander("Legacy S3 Levelised Cost Analysis", expanded=False):
    st.write("Detailed discounted cash flow analysis to match Excel S3 - to be implemented")

# Add S2D2 Lab footer
add_s2d2_footer()
