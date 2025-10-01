import streamlit as st
from utils import add_s2d2_footer
from src.utils.financial_calculator import FinancialCalculator
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Cash Flow Analysis")

# Create FinancialCalculator instance
calculator = FinancialCalculator(discount_rate=0.04, project_life=20)

def create_payback_period_analysis(cash_flows, discount_rate, project_life):
    """Create payback period analysis visualization"""
    years = cash_flows['Year'].values

    # Calculate discounted cumulative cash flows
    discounted_cumulative = []
    cumulative_sum = 0

    for year in range(len(cash_flows)):
        if year == 0:
            discounted_cf = cash_flows['Total'][year]
        else:
            discounted_cf = cash_flows['Total'][year] / ((1 + discount_rate) ** year)
        cumulative_sum += discounted_cf
        discounted_cumulative.append(cumulative_sum)

    # Create figure
    fig = go.Figure()

    # Add cumulative discounted cash flow line
    fig.add_trace(go.Scatter(
        x=years,
        y=discounted_cumulative,
        mode='lines+markers',
        name='Cumulative Discounted Cash Flow',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Add investment line (negative initial investment)
    initial_investment = abs(cash_flows['Total'][0])
    fig.add_trace(go.Scatter(
        x=years,
        y=[initial_investment * -1] * len(years),
        mode='lines',
        name='Break-even Target',
        line=dict(color='red', width=2, dash='dot')
    ))

    # Find payback period
    payback_year = None
    for i, cum_cf in enumerate(discounted_cumulative):
        if cum_cf >= 0:
            payback_year = i
            break

    # Add payback annotation if found
    if payback_year is not None:
        fig.add_annotation(
            x=payback_year,
            y=discounted_cumulative[payback_year],
            text=f"Payback Year: {payback_year}",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=-50
        )

        # Add vertical line at payback year
        fig.add_vline(x=payback_year, line_width=2, line_dash="dash", line_color="green")

    fig.update_layout(
        title="Payback Period Analysis",
        xaxis_title="Year",
        yaxis_title="Cumulative Discounted Cash Flow ($)",
        height=500,
        showlegend=True
    )

    return fig

def create_stacked_cash_flow_waterfall(cash_flows):
    """Create stacked cash flow waterfall chart showing component contributions"""
    years = list(range(len(cash_flows)))

    # Initialize cumulative flows
    cumulative_capex = 0
    cumulative_opex = 0
    cumulative_revenue = 0
    cumulative_other = 0

    capex_flows = []
    opex_flows = []
    revenue_flows = []
    other_flows = []

    for year in years:
        # CAPEX (Year 0 only, negative outflow)
        capex = cash_flows['Gen_CAPEX'][year] + cash_flows['Elec_CAPEX'][year] + cash_flows['Battery_cost'][year]
        capex_flows.append(capex)
        cumulative_capex += capex

        # OPEX (Years 1+, positive outflow)
        opex = (cash_flows['Gen_OPEX'][year] + cash_flows['Elec_OandM'][year] +
                cash_flows['Power_cost'][year] + cash_flows['Water_cost'][year] +
                cash_flows['Stack_replacement'][year])
        opex_flows.append(opex)
        cumulative_opex += opex

        # Revenue (Years 1+, negative inflow as it's already included in Total)
        if 'hydrogen_revenue' in cash_flows.columns and year > 0:
            revenue = cash_flows['hydrogen_revenue'][year]
        else:
            revenue = 0
        revenue_flows.append(revenue)
        cumulative_revenue += revenue

        # Other components
        other = cash_flows['Total'][year] - capex - opex - revenue
        other_flows.append(other)
        cumulative_other += other

    # Create waterfall chart
    fig = go.Figure()

    # Add traces for each component
    fig.add_trace(go.Waterfall(
        name="CAPEX",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(capex_flows) - 1),
        x=years,
        y=capex_flows,
        connector=dict(mode="between", line=dict(width=2, color="rgb(0, 0, 0)")),
        decreasing=dict(marker=dict(color="red")),
        increasing=dict(marker=dict(color="red")),
        totals=dict(marker=dict(color="red"))
    ))

    fig.add_trace(go.Waterfall(
        name="OPEX",
        orientation="v",
        measure=["relative"] * len(opex_flows),
        x=years,
        y=opex_flows,
        connector=dict(mode="between", line=dict(width=2, color="rgb(0, 0, 0)")),
        decreasing=dict(marker=dict(color="orange")),
        increasing=dict(marker=dict(color="orange")),
        totals=dict(marker=dict(color="orange"))
    ))

    fig.add_trace(go.Waterfall(
        name="Revenue",
        orientation="v",
        measure=["relative"] * len(revenue_flows),
        x=years,
        y=revenue_flows,
        connector=dict(mode="between", line=dict(width=2, color="rgb(0, 0, 0)")),
        decreasing=dict(marker=dict(color="green")),
        increasing=dict(marker=dict(color="green")),
        totals=dict(marker=dict(color="green"))
    ))

    if any(abs(x) > 1 for x in other_flows):
        fig.add_trace(go.Waterfall(
            name="Other",
            orientation="v",
            measure=["relative"] * len(other_flows),
            x=years,
            y=other_flows,
            connector=dict(mode="between", line=dict(width=2, color="rgb(0, 0, 0)")),
            decreasing=dict(marker=dict(color="blue")),
            increasing=dict(marker=dict(color="blue")),
            totals=dict(marker=dict(color="blue"))
        ))

    fig.update_layout(
        title="Cash Flow Waterfall by Component",
        xaxis_title="Year",
        yaxis_title="Cash Flow ($)",
        waterfallgap=0.3,
        showlegend=True,
        height=500
    )

    return fig

def create_cash_flow_visualizations(cash_flows, hydrogen_price):
    """Create comprehensive cash flow visualizations"""
    years = cash_flows['Year'].values

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Annual Cash Flows',
            'Cumulative Cash Flows',
            'Revenue vs Costs Breakdown',
            'Discounted Cash Flows'
        ),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )

    # Annual Cash Flows
    fig.add_trace(
        go.Bar(
            x=years,
            y=cash_flows['Total'],
            name='Annual Cash Flow',
            marker_color=['red' if x < 0 else 'green' for x in cash_flows['Total']],
            showlegend=False
        ),
        row=1, col=1
    )

    # Cumulative Cash Flows
    cumulative = np.cumsum(cash_flows['Total'])
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='#1f77b4', width=3),
            showlegend=False
        ),
        row=1, col=2
    )

    # Revenue vs Costs (stacked)
    if 'hydrogen_revenue' in cash_flows.columns:
        revenues = cash_flows['hydrogen_revenue'].copy()
        costs = cash_flows['Total'] - cash_flows['hydrogen_revenue']

        fig.add_trace(
            go.Bar(
                x=years,
                y=costs,
                name='Total Costs',
                marker_color='red',
                offsetgroup=0
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                x=years,
                y=revenues,
                name='Hydrogen Revenue',
                marker_color='green',
                offsetgroup=0
            ),
            row=2, col=1
        )

    # Discounted Cash Flows
    discounted = [cf / ((1 + calculator.discount_rate) ** year) for year, cf in enumerate(cash_flows['Total'])]
    cumulative_discounted = np.cumsum(discounted)

    fig.add_trace(
        go.Scatter(
            x=years,
            y=cumulative_discounted,
            mode='lines+markers',
            name='Cumulative Discounted',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            showlegend=False
        ),
        row=2, col=2
    )

    # Add horizontal line at y=0 for cumulative charts
    fig.add_hline(y=0, line_dash="dash", line_color="black", row="all", col="all")

    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Cash Flow ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Cash Flow ($)", row=1, col=2)
    fig.update_yaxes(title_text="Cash Flow ($)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Discounted Cash Flow ($)", row=2, col=2)

    return fig

def calculate_additional_metrics(cash_flows, initial_investment, hydrogen_price):
    """Calculate additional profitability metrics"""
    # Simple Payback Period (without discounting)
    cumulative_cf = np.cumsum(cash_flows['Total'])
    payback_year = None
    for i, cum_cf in enumerate(cumulative_cf):
        if cum_cf >= 0:
            payback_year = i
            break

    # Annual profit margin
    annual_profits = cash_flows['Total'][1:]  # Exclude year 0
    profitable_years = sum(1 for profit in annual_profits if profit > 0)
    avg_annual_profit = np.mean(annual_profits)

    return {
        'simple_payback_years': payback_year,
        'profitable_years': profitable_years,
        'total_years': len(annual_profits),
        'avg_annual_profit': avg_annual_profit,
        'profit_margin_percentage': (avg_annual_profit / (hydrogen_price * 100)) * 100 if hydrogen_price > 0 else 0
    }

st.header("Annual Revenue/Cost Projections")

# Sidebar controls
with st.sidebar:
    st.header("Analysis Parameters")

    hydrogen_price = st.slider(
        "Hydrogen Selling Price ($/kg)",
        min_value=200,
        max_value=800,
        value=350,
        step=10,
        help="Price at which hydrogen is sold"
    )

    discount_rate = st.slider(
        "Discount Rate (%)",
        min_value=0,
        max_value=15,
        value=4,
        step=1,
        help="Annual discount rate for NPV calculations"
    )

    show_details = st.checkbox("Show Detailed Breakdown", value=False)
    show_sensitivity = st.checkbox("Show Sensitivity Analysis", value=False)

# Update calculator with user inputs
calculator.discount_rate = discount_rate / 100

if st.session_state.model_results:
    results = st.session_state.model_results
    inputs_summary = results.get('inputs_summary', {})
    operating_outputs = results.get('operating_outputs', {})

    # Create cost parameters from the current model
    cost_params = {
        'gen_capex_per_mw': 1000,  # Approximate values - could be refined
        'electrolyser_capex': 1000,
        'elec_capacity': inputs_summary.get('nominal_electrolyser_capacity', 10),
        'solar_capacity': inputs_summary.get('nominal_solar_farm_capacity', 10),
        'wind_capacity': inputs_summary.get('nominal_wind_farm_capacity', 0),
        'electrolyser_capacity': inputs_summary.get('nominal_electrolyser_capacity', 10),
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

    # Calculate NPV and get cash flows
    npv = calculator.calculate_npv(operating_outputs, cost_params, hydrogen_price)
    cash_flows = calculator.cash_flows

    # Key Financial Metrics
    col1, col2, col3, col4 = st.columns(4)

    # Calculate initial investment
    initial_investment = abs(cash_flows['Total'][0]) if isinstance(cash_flows, pd.DataFrame) and len(cash_flows) > 0 else 0

    with col1:
        st.metric(
            label="Net Present Value (NPV)",
            value=f"${npv:,.0f}",
            delta=f"{npv/initial_investment:.1%}" if initial_investment > 0 else None
        )

    roi = calculator.calculate_roi(operating_outputs, cost_params, hydrogen_price)
    with col2:
        st.metric(
            label="Return on Investment (ROI)",
            value=f"{roi:.1f}%"
        )

    payback = calculator.calculate_payback_period(operating_outputs, cost_params, hydrogen_price)
    with col3:
        st.metric(
            label="Discounted Payback Period",
            value=f"{payback:.1f} years" if payback else "Not achieved"
        )

    lcoh = calculator.calculate_lcoh(operating_outputs, cost_params)
    with col4:
        st.metric(
            label="Levelized Cost of Hydrogen",
            value=f"${lcoh:.2f}/kg"
        )

    # Cash Flow Visualizations
    st.subheader("Cash Flow Analysis")

    # Create and display cash flow chart
    if isinstance(cash_flows, pd.DataFrame) and not cash_flows.empty:
        fig = create_cash_flow_visualizations(cash_flows, hydrogen_price)
        st.plotly_chart(fig, use_container_width=True)

        # Payback Period Analysis Charts
        st.subheader("Payback Period Analysis")

        # Show payback period chart
        payback_fig = create_payback_period_analysis(cash_flows, discount_rate/100, 20)
        st.plotly_chart(payback_fig, use_container_width=True)

        # Payback period insights
        discounted_cumulative = []
        cumulative_sum = 0

        for year in range(len(cash_flows)):
            if year == 0:
                discounted_cf = cash_flows['Total'][year]
            else:
                discounted_cf = cash_flows['Total'][year] / ((1 + discount_rate/100) ** year)
            cumulative_sum += discounted_cf
            discounted_cumulative.append(cumulative_sum)

        payback_year = None
        for i, cum_cf in enumerate(discounted_cumulative):
            if cum_cf >= 0:
                payback_year = i
                break

        if payback_year:
            st.info(f"💡 **Payback Insight**: The project recovers its initial investment by Year {payback_year}.")

            # Calculate payback fraction if not exactly at year boundary
            if payback_year < len(discounted_cumulative) - 1:
                prev_cum = discounted_cumulative[payback_year - 1]
                current_cum = discounted_cumulative[payback_year]
                if current_cum != 0:
                    fraction = prev_cum / current_cum
                    exact_payback = payback_year - 1 + fraction
                    st.write(f"Exact payback period: {exact_payback:.2f} years")
        else:
            st.warning("⚠️ **Payback Warning**: The project does not reach break-even within the 20-year timeframe based on current cash flows.")

        # Stacked Cash Flow Waterfall
        st.subheader("Cash Flow Component Breakdown")

        try:
            waterfall_fig = create_stacked_cash_flow_waterfall(cash_flows)
            st.plotly_chart(waterfall_fig, use_container_width=True)

            # Waterfall insights
            st.info("💡 **Waterfall Analysis**: This chart shows how each financial component contributes to the overall cash flow. Red bars represent costs (negative cash flows), green bars show revenues (positive cash flows), allowing you to track major cost drivers and revenue streams throughout the project lifecycle.")

        except Exception as e:
            st.warning(f"Could not generate waterfall chart: {str(e)}")

        # Detailed Cash Flow Table
        if show_details and isinstance(cash_flows, pd.DataFrame):
            st.subheader("Detailed Cash Flow Projections")

            # Format the table for display
            display_df = cash_flows.copy()
            numeric_cols = [col for col in display_df.columns if col != 'Year' and display_df[col].dtype in ['int64', 'float64']]

            # Round monetary values
            for col in numeric_cols:
                display_df[col] = display_df[col].round(0)

            st.dataframe(display_df.style.format({col: "${:,.0f}" for col in numeric_cols}), use_container_width=True)

        # Additional Metrics
        st.subheader("Additional Financial Metrics")

        initial_investment = abs(cash_flows['Total'][0]) if len(cash_flows) > 0 else 0
        additional_metrics = calculate_additional_metrics(cash_flows, initial_investment, hydrogen_price)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Simple Payback Period",
                value=f"{additional_metrics['simple_payback_years']} years" if additional_metrics['simple_payback_years'] else "Not achieved"
            )

        with col2:
            st.metric(
                label="Profitable Years",
                value=f"{additional_metrics['profitable_years']}/{additional_metrics['total_years']}"
            )

        with col3:
            st.metric(
                label="Average Annual Profit",
                value=f"${additional_metrics['avg_annual_profit']:,.0f}"
            )

        # Sensitivity Analysis
        if show_sensitivity:
            st.subheader("Sensitivity Analysis")

            # Price sensitivity
            price_range = np.linspace(hydrogen_price * 0.8, hydrogen_price * 1.2, 9)
            sensitivity_results = []

            for price in price_range:
                npv_sens = calculator.calculate_npv(operating_outputs, cost_params, price)
                sensitivity_results.append({
                    'Hydrogen Price ($/kg)': price,
                    'NPV': npv_sens
                })

            sensitivity_df = pd.DataFrame(sensitivity_results)

            # Create sensitivity plot
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=sensitivity_df['Hydrogen Price ($/kg)'],
                y=sensitivity_df['NPV'],
                mode='lines+markers',
                name='NPV vs Price',
                line=dict(color='#1f77b4', width=3)
            ))

            fig_sens.update_layout(
                title="NPV Sensitivity to Hydrogen Price",
                xaxis_title="Hydrogen Price ($/kg)",
                yaxis_title="Net Present Value ($)",
                height=400
            )

            st.plotly_chart(fig_sens, use_container_width=True)

            # Export functionality
            if st.button("Export Cash Flow Data") and isinstance(cash_flows, pd.DataFrame):
                csv = cash_flows.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="cash_flow_analysis.csv",
                    mime="text/csv",
                    key="download_csv"
                )

else:
    st.info("Please go to the 'Inputs' page and run the model calculation first.")

# Add S2D2 Lab footer
add_s2d2_footer()
