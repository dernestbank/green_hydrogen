import streamlit as st
import numpy as np
from utils import add_s2d2_footer
from src.utils.visualization import KPICalculator, FormatHelpers, ChartDataFormatter
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Results")

st.header("Summary of Key Inputs and Results")

if st.session_state.model_results:
    results = st.session_state.model_results
    inputs_summary = results['inputs_summary']
    operating_outputs = results['operating_outputs']
    lcoh = results['lcoh']
    business_case = results['business_case']

    # Inputs section (collapsed by default)
    with st.expander("View Key Inputs", expanded=False):
        st.markdown(f"""
        - **Location:** {inputs_summary.get('site_location', 'N/A')}
        - **Configuration:** {inputs_summary.get('power_plant_configuration', 'N/A')}
        - **Electrolyser Capacity:** {inputs_summary.get('nominal_electrolyser_capacity', 'N/A')} MW
        - **Power Plant Capacity:** Solar: {inputs_summary.get('nominal_solar_farm_capacity', 'N/A')} MW, Wind: {inputs_summary.get('nominal_wind_farm_capacity', 'N/A')} MW
        - **Battery Capacity:** {inputs_summary.get('battery_rated_power', 'N/A')} MW ({inputs_summary.get('duration_of_storage_hours', 'N/A')} hours)
        """)

    # Key Metrics Cards
    st.subheader("📊 Key Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cf_gen = operating_outputs.get('Generator Capacity Factor', 0)
        st.metric(
            label="Generator Capacity Factor",
            value=f"{cf_gen:.1%}",
            help="Average capacity factor of renewable energy generation"
        )

    with col2:
        cf_elec = operating_outputs.get('Achieved Electrolyser Capacity Factor', 0)
        st.metric(
            label="Electrolyser Capacity Factor",
            value=f"{cf_elec:.1%}",
            help="Average capacity factor of electrolyser operation"
        )

    with col3:
        h2_prod = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 0)
        formatted_h2 = FormatHelpers.format_with_unit(h2_prod, 't/year')
        st.metric(
            label="Hydrogen Production",
            value=formatted_h2,
            help="Annual hydrogen production (fixed specific energy consumption)"
        )

    with col4:
        formatted_lcoh = FormatHelpers.format_cost(lcoh, '$/kg')
        st.metric(
            label="Levelized Cost of Hydrogen",
            value=formatted_lcoh,
            help="LCOH - Fixed consumption case"
        )

    # Additional metrics in next row
    col5, col6 = st.columns(2)

    with col5:
        energy_elec = operating_outputs.get('Energy in to Electrolyser [MWh/yr]', 0)
        formatted_energy = FormatHelpers.format_energy(energy_elec)
        st.metric(
            label="Annual Energy to Electrolyser",
            value=formatted_energy
        )

    with col6:
        surplus_energy = operating_outputs.get('Surplus Energy [MWh/yr]', 0)
        formatted_surplus = FormatHelpers.format_energy(surplus_energy)
        st.metric(
            label="Annual Surplus Energy",
            value=formatted_surplus,
            help="Energy not utilised by electrolyser"
        )

    # Energy Balance Pie Chart
    st.subheader("Energy Balance")

    # Calculate energy balance from operating outputs
    energy_elec = operating_outputs.get('Energy in to Electrolyser [MWh/yr]', 1000)
    surplus_energy = operating_outputs.get('Surplus Energy [MWh/yr]', 500)

    # Create energy balance data
    energy_labels = ['Electrolyser Consumption', 'Surplus Energy']
    energy_values = [energy_elec, surplus_energy]
    energy_colors = ['#1f77b4', '#ff7f0e']

    # Calculate percentages
    total_energy = sum(energy_values)
    energy_percentages = [f"{val/total_energy:.1%}" for val in energy_values]

    fig_pie = go.Figure(data=[go.Pie(
        labels=energy_labels,
        values=energy_values,
        hoverinfo='label+percent+value',
        textinfo='label+percent',
        marker_colors=energy_colors,
        title="Annual Energy Distribution"
    )])

    fig_pie.update_layout(
        title="Energy Balance (MWh/year)",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

st.header("Visualizations")

# Create two columns for the visualizations
col1, col2 = st.columns(2)

# Template for displaying images with loading state
def display_image_placeholder(container, title, description):
    container.subheader(title)
    with container.container():
        # Create a box with a border and centered text
        container.markdown(
            f"""
            <div style="
                border: 2px dashed #cccccc;
                border-radius: 5px;
                padding: 20px;
                text-align: center;
                margin: 10px 0;
                background-color: #f5f5f5;
            ">
                <p style="color: #666666;">🖼️ {description}</p>
                <p style="color: #999999; font-size: 12px;">Click to load image</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Left column visualizations
with col1:
    st.subheader("Annual Duration Curves")

    if st.session_state.model_results:
        # Create duration curve data
        # This would typically use model.results.get_duration_curve_data()
        # For now, create placeholder data matching Excel S2 format

        hours = list(range(8760))  # 8760 hours in a year
        gen_cf = [0.7 + 0.3 * (x / 8760) + 0.1 * (x % 24) / 24 for x in hours]  # Sample generator CF
        elec_cf = [0.6 + 0.4 * (x / 8760) * 0.8 for x in hours]  # Sample electrolyser CF

        # Sort in descending order for duration curve
        gen_sorted = sorted(gen_cf, reverse=True)
        elec_sorted = sorted(elec_cf, reverse=True)

        # Create Plotly figure
        fig = go.Figure()

        # Add generator trace
        fig.add_trace(go.Scatter(
            x=list(range(1, len(gen_sorted) + 1)),
            y=gen_sorted,
            mode='lines',
            name='Generator CF',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add electrolyser trace
        fig.add_trace(go.Scatter(
            x=list(range(1, len(elec_sorted) + 1)),
            y=elec_sorted,
            mode='lines',
            name='Electrolyser CF',
            line=dict(color='#ff7f0e', width=2)
        ))

        # Update layout to match Excel S2
        fig.update_layout(
            title="Annual Duration Curves (Excel S2)",
            xaxis_title="Hours of operation per year",
            yaxis_title="Capacity Factor",
            yaxis_tickformat=".1%",
            xaxis=dict(tickmode='array',
                      tickvals=[0, 876, 1752, 2628, 3504, 4380, 5256, 6132, 7008, 7884, 8760],
                      ticktext=['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']),
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        display_image_placeholder(col1, "Annual Duration Curves",
                                "Shows duration curves for power plant and electrolyser")
    
    st.subheader("Capacity Factor Distributions")

    if st.session_state.model_results:
        # Create capacity factor distributions
        # This would typically use actual hourly data from the model

        # Sample data for generator capacity factors (8760 hours)
        gen_cf = np.random.beta(2, 5, 8760)  # Beta distribution around 0.25-0.35
        elec_cf = np.random.uniform(0, 1, 8760) * np.random.beta(2, 3, 8760)  # More variable electrolyser

        # Create histogram data
        fig_hist = go.Figure()

        # Add generator histogram
        fig_hist.add_trace(go.Histogram(
            x=gen_cf,
            name='Generator CF',
            opacity=0.7,
            xbins=dict(start=0, end=1, size=0.05),
            marker_color='#1f77b4'
        ))

        # Add electrolyser histogram
        fig_hist.add_trace(go.Histogram(
            x=elec_cf,
            name='Electrolyser CF',
            opacity=0.7,
            xbins=dict(start=0, end=1, size=0.05),
            marker_color='#ff7f0e'
        ))

        fig_hist.update_layout(
            title="Capacity Factor Distributions",
            xaxis_title="Capacity Factor",
            yaxis_title="Frequency (hours)",
            barmode='overlay',
            showlegend=True,
            height=400
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        display_image_placeholder(col1, "Interactive Hourly Capacity Factors",
                                "Interactive plot of hourly capacity factors")

    # Energy Balance Pie Chart
    st.subheader("Energy Balance")

    if st.session_state.model_results:
        # Calculate energy balance from operating outputs
        energy_elec = operating_outputs.get('Energy in to Electrolyser [MWh/yr]', 1000)
        surplus_energy = operating_outputs.get('Surplus Energy [MWh/yr]', 500)

        # Create energy balance data
        energy_labels = ['Electrolyser Consumption', 'Surplus Energy']
        energy_values = [energy_elec, surplus_energy]
        energy_colors = ['#1f77b4', '#ff7f0e']

        # Calculate percentages
        total_energy = sum(energy_values)
        energy_percentages = [f"{val/total_energy:.1%}" for val in energy_values]

        fig_pie = go.Figure(data=[go.Pie(
            labels=energy_labels,
            values=energy_values,
            hoverinfo='label+percent+value',
            textinfo='label+percent',
            marker_colors=energy_colors,
            title="Annual Energy Distribution"
        )])

        fig_pie.update_layout(
            title="Energy Balance (GWh/year)",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        display_image_placeholder(col1, "Energy Balance",
                                "Pie chart showing energy distribution")

    st.subheader("LCOH Cost Breakdown")

    if st.session_state.model_results:
        # Calculate LCOH components (CAPEX, OPEX, etc.)
        # Using the business_case or financial results if available
        business_case = results.get('business_case', {})

        # Sample cost breakdown - in real implementation, calculate from model results
        cost_components = {
            'CAPEX': business_case.get('capex', 2000),
            'OPEX': business_case.get('opex', 300),
            'Stack Replacement': business_case.get('stack_replacement', 150),
            'Water Costs': business_case.get('water_costs', 50)
        }

        # Calculate total costs
        total_cost = sum(cost_components.values())

        # Create waterfall chart using our ChartDataFormatter
        formatter = ChartDataFormatter()
        fig_waterfall = go.Figure()

        # Create cumulative x values and y values
        components = ['Base'] + list(cost_components.keys()) + ['Total']
        values = [0] + list(cost_components.values()) + [0]
        cumulative = [0]

        for i, val in enumerate(values[1:], 1):
            cumulative.append(cumulative[-1] + val)

        # Add waterfall bars
        for i in range(len(values)):
            if i == 0:
                # Base
                fig_waterfall.add_trace(go.Bar(
                    x=[components[i]],
                    y=[values[i]],
                    name='Base',
                    marker_color='rgba(55, 128, 191, 0.7)',
                    showlegend=False
                ))
            elif i == len(values) - 1:
                # Total
                fig_waterfall.add_trace(go.Bar(
                    x=[components[i]],
                    y=[cumulative[-1]],
                    name='Total',
                    marker_color='rgba(50, 171, 96, 0.7)',
                    showlegend=False
                ))
            else:
                # Components
                fig_waterfall.add_trace(go.Bar(
                    x=[components[i]],
                    y=[values[i]],
                    name=components[i],
                    marker_color='rgba(219, 64, 82, 0.7)' if values[i] > 0 else 'rgba(50, 171, 96, 0.7)',
                    showlegend=False
                ))

        fig_waterfall.update_layout(
            title="LCOH Cost Components (A$/kg)",
            xaxis_title="",
            yaxis_title="Cumulative Cost (A$/kg)",
            showlegend=False,
            height=400
        )

        # Add annotations for values
        for i, val in enumerate(cumulative):
            if i < len(components) and val != 0:
                fig_waterfall.add_annotation(
                    x=components[i],
                    y=val + 0.1,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(size=10)
                )

        st.plotly_chart(fig_waterfall, use_container_width=True)
    else:
        display_image_placeholder(col1, "LCOH Cost Breakdown",
                                "Waterfall chart showing LCOH cost components")

    display_image_placeholder(col1, "Annual Operating Cost Analysis",
                            "Visualization of operating costs")

# Right column visualizations
with col2:
    display_image_placeholder(col2, "Capital and Indirect Costs", 
                            "Breakdown of capital and indirect costs")
    
    display_image_placeholder(col2, "Annual Sales Analysis", 
                            "Visualization of annual sales data")
    
    display_image_placeholder(col2, "Operating Cost Breakdown", 
                            "Detailed operating cost components")
    
    display_image_placeholder(col2, "Cumulative Cash Flow", 
                            "Project cash flow over time")

# Add S2D2 Lab footer
add_s2d2_footer()
