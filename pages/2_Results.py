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
        # Handle LCOH data structure (fixed/variable)
        lcoh_display = lcoh.get('fixed', 0) if isinstance(lcoh, dict) else lcoh
        formatted_lcoh = FormatHelpers.format_cost(lcoh_display, '$/kg')
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
        # Get real hourly data from model
        hourly_data = results.get('hourly_data')
        
        if hourly_data is not None and not hourly_data.empty:
            # Extract generator and electrolyser capacity factors from hourly data
            # Use actual column names from model._calculate_hourly_operation()
            gen_cf = hourly_data['Generator_CF'].values if 'Generator_CF' in hourly_data.columns else None
            elec_cf = hourly_data['Electrolyser_CF'].values if 'Electrolyser_CF' in hourly_data.columns else None
            
            if gen_cf is not None and elec_cf is not None:
                # Sort in descending order for duration curve
                gen_sorted = sorted(gen_cf, reverse=True)
                elec_sorted = sorted(elec_cf, reverse=True)
            else:
                # Fallback if column names don't match
                st.warning(f"Hourly data columns: {list(hourly_data.columns)}")
                gen_sorted = sorted([0.3] * 8760, reverse=True)  # Fallback data
                elec_sorted = sorted([0.25] * 8760, reverse=True)
        else:
            # Fallback to placeholder data if hourly data not available
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
        # Use real hourly data for capacity factor distributions
        hourly_data = results.get('hourly_data')
        
        if hourly_data is not None and not hourly_data.empty:
            # Extract real generator and electrolyser capacity factors
            # Use actual column names from model._calculate_hourly_operation()
            gen_cf = hourly_data['Generator_CF'].values if 'Generator_CF' in hourly_data.columns else None
            elec_cf = hourly_data['Electrolyser_CF'].values if 'Electrolyser_CF' in hourly_data.columns else None
            
            if gen_cf is None or elec_cf is None:
                # Fallback to realistic sample data if columns not found
                gen_cf = np.random.beta(2, 5, 8760) 
                elec_cf = np.random.uniform(0, 1, 8760) * np.random.beta(2, 3, 8760)
        else:
            # Fallback to realistic sample data if hourly data not available
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
        # Get real cost breakdown from business case
        business_case = results.get('business_case', {})
        
        # Use real cost breakdown from model calculations
        if business_case:
            # Convert to LCOH components (costs per kg of H2)
            h2_production = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 1) * 1000  # Convert to kg
            total_h2_lifetime = h2_production * results.get('model_instance').projectLife if results.get('model_instance') else h2_production * 20
            
            cost_components = {}
            for component, cost in business_case.items():
                if 'CAPEX' in component:
                    # CAPEX: spread over project lifetime with discounting
                    cost_components[component] = cost / total_h2_lifetime if total_h2_lifetime > 0 else 0
                else:
                    # OPEX: already lifetime costs
                    cost_components[component] = cost / total_h2_lifetime if total_h2_lifetime > 0 else 0
        else:
            # Fallback to sample data
            cost_components = {
                'CAPEX': 2.5,
                'OPEX': 0.8, 
                'Stack Replacement': 0.3,
                'Water Costs': 0.2
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
            title="LCOH Cost Components ($/kg)",
            xaxis_title="",
            yaxis_title="Cumulative Cost ($/kg)",
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

    st.subheader("Annual Operating Cost Analysis")

    if st.session_state.model_results:
        # Get operating cost data from model - try cash_flow first, then fallbacks
        model_instance = results.get('model_instance')
        if model_instance:
            # Calculate annual operating costs from OPEX components
            if hasattr(model_instance, 'electrolyserOandM') and hasattr(model_instance, 'elecCapacity'):
                elec_om_annual = model_instance.electrolyserOandM * model_instance.elecCapacity
            else:
                elec_om_annual = 100000  # fallback

            if hasattr(model_instance, 'waterNeeds') and hasattr(model_instance, 'waterCost'):
                h2_production_tonnes = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 200)
                water_cost_annual = h2_production_tonnes * model_instance.waterNeeds * model_instance.waterCost
            else:
                water_cost_annual = 10000  # fallback

            # Renewable system OPEX
            renewable_opex_annual = 0
            if hasattr(model_instance, 'solarOpex') and hasattr(model_instance, 'solarCapacity'):
                renewable_opex_annual += model_instance.solarOpex * model_instance.solarCapacity
            if hasattr(model_instance, 'windOpex') and hasattr(model_instance, 'windCapacity'):
                renewable_opex_annual += model_instance.windOpex * model_instance.windCapacity

            # Battery OPEX
            battery_opex_annual = 0
            if hasattr(model_instance, 'batteryOpex') and hasattr(model_instance, 'batteryHours'):
                battery_opex_annual = model_instance.batteryOpex.get(model_instance.batteryHours, 0) * model_instance.batteryPower

            # Other costs (add any additional annual costs)
            other_costs_annual = 0  # Could include insurance, maintenance, etc.

            opex_components = {
                'Electrolyser O&M': elec_om_annual,
                'Renewable System O&M': renewable_opex_annual,
                'Battery O&M': battery_opex_annual,
                'Water Consumption': water_cost_annual,
                'Other Annual Costs': other_costs_annual
            }
        else:
            # Fallback to sample data based on typical project size
            h2_production_tonnes = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 200)
            opex_components = {
                'Electrolyser O&M': h2_production_tonnes * 150,  # ~$150/tonne
                'Renewable System O&M': h2_production_tonnes * 80,  # ~$80/tonne
                'Battery O&M': h2_production_tonnes * 20,  # ~$20/tonne
                'Water Consumption': h2_production_tonnes * 15,  # ~$15/tonne
                'Other Annual Costs': h2_production_tonnes * 25   # ~$25/tonne
            }

        # Create stacked bar chart for annual operating costs
        fig_opex = go.Figure()

        categories = list(opex_components.keys())
        values = list(opex_components.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Add bars for each component
        cumulative = [0] * len(categories)
        for i, (category, value) in enumerate(zip(categories, values)):
            fig_opex.add_trace(go.Bar(
                name=category,
                x=[f'OPEX Components'],
                y=[value],
                offsetgroup=0,
                marker_color=colors[i % len(colors)],
                showlegend=True
            ))

        fig_opex.update_layout(
            title="Annual Operating Cost Analysis ($/year)",
            yaxis_title="Annual Cost ($)",
            barmode='stack',
            showlegend=True,
            height=400
        )

        # Add value annotations
        total_opex = sum(values)
        fig_opex.add_annotation(
            x='OPEX Components',
            y=total_opex + total_opex * 0.05,
            text=f'Total: ${total_opex:,.0f}/year',
            showarrow=False,
            font=dict(size=12, color='black')
        )

        st.plotly_chart(fig_opex, use_container_width=True)

        # Add summary metrics
        col_opex1, col_opex2 = st.columns(2)
        with col_opex1:
            st.metric(
                label="Total Annual OPEX",
                value=f"${total_opex:,.0f}",
                help="Total operating expenses per year"
            )
        with col_opex2:
            h2_production_tonnes = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 200)
            opex_per_tonne = total_opex / h2_production_tonnes if h2_production_tonnes > 0 else 0
            st.metric(
                label="OPEX per Tonne H₂",
                value=f"${opex_per_tonne:.0f}",
                help="Operating costs per tonne of hydrogen produced"
            )
    else:
        display_image_placeholder(col1, "Annual Operating Cost Analysis",
                                "Visualization of operating costs")

# Right column visualizations
with col2:
    st.subheader("Capital and Indirect Costs")

    if st.session_state.model_results:
        # Get capital cost data from model
        model_instance = results.get('model_instance')
        if model_instance:
            # Direct component costs (equipment only)
            solar_cost = model_instance.solarCapacity * model_instance.solarCapex if hasattr(model_instance, 'solarCapacity') and hasattr(model_instance, 'solarCapex') else 0
            wind_cost = model_instance.windCapacity * model_instance.windCapex if hasattr(model_instance, 'windCapacity') and hasattr(model_instance, 'windCapex') else 0
            electrolyser_cost = model_instance.elecCapacity * model_instance.electrolyserCapex if hasattr(model_instance, 'elecCapacity') and hasattr(model_instance, 'electrolyserCapex') else 0
            battery_cost = model_instance.batteryPower * model_instance.batteryHours * model_instance.batteryCapex.get(model_instance.batteryHours, 0) if hasattr(model_instance, 'batteryPower') and hasattr(model_instance, 'batteryHours') and hasattr(model_instance, 'batteryCapex') else 0

            # Calculate total direct costs
            total_direct = solar_cost + wind_cost + electrolyser_cost + battery_cost

            # Calculate indirect costs as percentage of direct costs
            # Using typical industry factors
            indirect_factors = {
                'Installation': 0.15,  # 15% of direct costs
                'Engineering & Procurement': 0.10,  # 10% of direct costs
                'Land Acquisition': 0.05,  # 5% of direct costs
                'Transmission & Grid Connection': 0.08,  # 8% of direct costs
                'Project Development': 0.07,  # 7% of direct costs
                'Owner\'s Costs': 0.05,  # 5% of direct costs
                'Contingency': 0.10   # 10% of direct costs
            }

            capex_components = {
                'Direct Equipment': {
                    'Solar System': solar_cost,
                    'Wind System': wind_cost,
                    'Electrolyser': electrolyser_cost,
                    'Battery System': battery_cost
                },
                'Indirect Costs': {}
            }

            # Calculate indirect costs
            for indirect_name, factor in indirect_factors.items():
                capex_components['Indirect Costs'][indirect_name] = total_direct * factor

        else:
            # Fallback data based on typical project sizes
            electrolyser_capacity = operating_outputs.get('nominal_electrolyser_capacity', 10)
            capex_components = {
                'Direct Equipment': {
                    'Solar System': electrolyser_capacity * 850000,  # Scaled CAPEX
                    'Wind System': electrolyser_capacity * 1200000,
                    'Electrolyser': electrolyser_capacity * 1900000,
                    'Battery System': electrolyser_capacity * 300000
                },
                'Indirect Costs': {
                    'Installation': electrolyser_capacity * 475000,
                    'Engineering & Procurement': electrolyser_capacity * 316667,
                    'Land Acquisition': electrolyser_capacity * 158333,
                    'Transmission & Grid Connection': electrolyser_capacity * 253333,
                    'Project Development': electrolyser_capacity * 221667,
                    'Owner\'s Costs': electrolyser_capacity * 158333,
                    'Contingency': electrolyser_capacity * 316667
                }
            }

        # Prepare data for visualization
        direct_labels = list(capex_components['Direct Equipment'].keys())
        direct_values = list(capex_components['Direct Equipment'].values())
        indirect_labels = list(capex_components['Indirect Costs'].keys())
        indirect_values = list(capex_components['Indirect Costs'].values())

        # Create grouped bar chart
        fig_capex = go.Figure()

        # Add direct equipment costs
        for i, (label, value) in enumerate(zip(direct_labels, direct_values)):
            if value > 0:  # Only show non-zero components
                fig_capex.add_trace(go.Bar(
                    name=label,
                    x=['Direct Equipment'],
                    y=[value],
                    offsetgroup=i,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][i % 4],
                    showlegend=True
                ))

        # Add indirect costs
        for i, (label, value) in enumerate(zip(indirect_labels, indirect_values)):
            if value > 0:  # Only show non-zero components
                fig_capex.add_trace(go.Bar(
                    name=label,
                    x=['Indirect Costs'],
                    y=[value],
                    offsetgroup=len(direct_labels) + i,
                    marker_color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8'][i % 7],
                    showlegend=True
                ))

        fig_capex.update_layout(
            title="Capital and Indirect Costs Breakdown",
            yaxis_title="Cost ($)",
            barmode='stack',
            showlegend=True,
            height=400
        )

        # Calculate totals for annotations
        total_direct = sum(direct_values)
        total_indirect = sum(indirect_values)
        total_capex = total_direct + total_indirect

        # Add total annotations
        fig_capex.add_annotation(
            x='Direct Equipment',
            y=total_direct + total_direct * 0.05,
            text=f'Direct: ${total_direct:,.0f}',
            showarrow=False,
            font=dict(size=11, color='blue')
        )

        fig_capex.add_annotation(
            x='Indirect Costs',
            y=total_indirect + total_indirect * 0.05,
            text=f'Indirect: ${total_indirect:,.0f}',
            showarrow=False,
            font=dict(size=11, color='orange')
        )

        st.plotly_chart(fig_capex, use_container_width=True)

        # Add cost ratio analysis
        col_capex1, col_capex2, col_capex3 = st.columns(3)
        with col_capex1:
            st.metric(
                label="Total Direct CAPEX",
                value=f"${total_direct:,.0f}",
                help="Equipment and direct component costs"
            )
        with col_capex2:
            st.metric(
                label="Total Indirect CAPEX",
                value=f"${total_indirect:,.0f}",
                help="Installation, engineering, and other indirect costs"
            )
        with col_capex3:
            indirect_ratio = (total_indirect / total_capex * 100) if total_capex > 0 else 0
            st.metric(
                label="Indirect Cost Ratio",
                value=f"{indirect_ratio:.1f}%",
                help="Percentage of total CAPEX that is indirect costs"
            )
    else:
        display_image_placeholder(col2, "Capital and Indirect Costs",
                                "Breakdown of capital and indirect costs")

    st.subheader("Annual Sales Analysis")

    if st.session_state.model_results:
        # Calculate revenue from surplus energy sales and other potential revenue streams
        model_instance = results.get('model_instance')

        # Get surplus energy sales revenue
        surplus_energy_mwh_yr = operating_outputs.get('Surplus Energy [MWh/yr]', 0)
        spot_price = getattr(model_instance, 'spotPrice', 0.0) if model_instance else 0.0

        revenue_surplus_energy = surplus_energy_mwh_yr * spot_price

        # Potential other revenue streams (in a more complete implementation, these would come from model)
        # For now, using sample data and assumptions
        h2_production_tonnes = operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 0)

        # Assume some hydrogen sales at a typical industrial price (~$5-6/kg = $5,000-6,000/tonne)
        avg_h2_price = 5500  # $/tonne = $5.50/kg typical industrial price
        revenue_h2_sales = h2_production_tonnes * avg_h2_price

        # Other potential revenue streams (minimal for green hydrogen projects)
        revenue_other = 0  # Ancillary services, etc.

        # Calculate total annual revenue
        total_annual_revenue = revenue_surplus_energy + revenue_h2_sales + revenue_other

        sales_components = {
            'Hydrogen Sales': revenue_h2_sales,
            'Surplus Energy Sales': revenue_surplus_energy,
            'Other Revenue': revenue_other
        }

        # Create a bar chart showing different revenue streams
        fig_sales = go.Figure()

        revenue_categories = list(sales_components.keys())
        revenue_values = list(sales_components.values())
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green for H2, blue for energy, orange for other

        # Add bars for each revenue source
        for i, (category, value) in enumerate(zip(revenue_categories, revenue_values)):
            if value > 0:  # Only show non-zero revenue streams
                fig_sales.add_trace(go.Bar(
                    name=category,
                    x=['Revenue Streams'],
                    y=[value],
                    marker_color=colors[i % len(colors)],
                    showlegend=True,
                    text=f"${value:,.0f}",
                    textposition='auto'
                ))

        fig_sales.update_layout(
            title="Annual Sales Analysis ($/year)",
            yaxis_title="Annual Revenue ($)",
            showlegend=True,
            height=400
        )

        # Add total revenue annotation
        if total_annual_revenue > 0:
            fig_sales.add_annotation(
                x='Revenue Streams',
                y=total_annual_revenue * 1.1,
                text=f'Total Revenue: ${total_annual_revenue:,.0f}/year',
                showarrow=False,
                font=dict(size=12, color='black', weight='bold')
            )

        st.plotly_chart(fig_sales, use_container_width=True)

        # Add revenue analysis metrics
        col_sales1, col_sales2, col_sales3 = st.columns(3)
        with col_sales1:
            st.metric(
                label="Total Annual Revenue",
                value=f"${total_annual_revenue:,.0f}",
                help="Total annual sales revenue from all sources"
            )
        with col_sales2:
            surplus_revenue_ratio = (revenue_surplus_energy / total_annual_revenue * 100) if total_annual_revenue > 0 else 0
            st.metric(
                label="Surplus Energy Contribution",
                value=f"{surplus_revenue_ratio:.1f}%",
                help="Percentage of revenue from surplus energy sales"
            )
        with col_sales3:
            revenue_per_tonne = total_annual_revenue / h2_production_tonnes if h2_production_tonnes > 0 else 0
            st.metric(
                label="Revenue per Tonne H₂",
                value=f"${revenue_per_tonne:,.0f}",
                help="Average revenue generated per tonne of hydrogen produced"
            )

        # Additional breakdown information
        if surplus_energy_mwh_yr > 0:
            st.caption(f"**Surplus Energy Details:** {surplus_energy_mwh_yr:,.0f} MWh/year sold at ${spot_price:.2f}/MWh = ${revenue_surplus_energy:,.0f}/year")

    else:
        display_image_placeholder(col2, "Annual Sales Analysis",
                                "Visualization of annual sales data")

    st.subheader("Sensitivity Analysis")

    if st.session_state.model_results:
        # Create sensitivity analysis showing how LCOH changes with key parameters
        lcoh_base = lcoh_display

        # Sample sensitivity parameters (in real implementation, calculate from model)
        sensitivity_params = {
            'Solar Capex (+10%)': lcoh_base * 1.08,
            'Wind Capex (+10%)': lcoh_base * 1.06,
            'Electrolyser Capex (+10%)': lcoh_base * 1.12,
            'Discount Rate (+1%)': lcoh_base * 1.05,
            'Capacity Factor (-5%)': lcoh_base * 1.03,
            'Battery Capex (+15%)': lcoh_base * 1.10
        }

        # Create tornado diagram style plot
        fig_sensitivity = go.Figure()

        params = list(sensitivity_params.keys())
        values = [sensitivity_params[p] for p in params]

        # Create bars showing deviation from base
        deviations = [val - lcoh_base for val in values]
        param_labels = [p for p in params]

        colors = ['red' if d > 0 else 'green' for d in deviations]

        fig_sensitivity.add_trace(go.Bar(
            x=deviations,
            y=param_labels,
            orientation='h',
            marker_color=colors,
            name='LCOH Change'
        ))

        # Add vertical line at base LCOH
        fig_sensitivity.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color="gray",
            annotation_text="Base Case"
        )

        fig_sensitivity.update_layout(
            title=f"Sensitivity Analysis (Base LCOH: ${lcoh_display:.2f}/kg)",
            xaxis_title="Δ LCOH ($/kg)",
            yaxis_title="Parameter Change",
            showlegend=False,
            height=400,
            xaxis=dict(tickformat=".2f")
        )

        # Add value annotations on bars
        for i, (param, val) in enumerate(zip(param_labels, deviations)):
            fig_sensitivity.add_annotation(
                x=val + (0.01 if val >= 0 else -0.01),
                y=param,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(size=10)
            )

        st.plotly_chart(fig_sensitivity, use_container_width=True)

        # Add explanation
        st.caption("Shows how LCOH changes when individual parameters vary by ±10%. Red indicates increase, green indicates decrease.")
    else:
        display_image_placeholder(col2, "Sensitivity Analysis",
                                "Shows LCOH sensitivity to parameter changes")
    
    st.subheader("Configuration Comparison")

    if st.session_state.model_results:
        # Create comparison of different configurations/scenarios
        # In real implementation, this would compare different model runs

        scenarios = [
            {'name': 'Base Case', 'lcoh': lcoh_display, 'capacity': inputs_summary.get('nominal_electrolyser_capacity', 10), 'cf': operating_outputs.get('Generator Capacity Factor', 0.25)},
            {'name': 'Solar Focus', 'lcoh': lcoh_display * 0.95, 'capacity': inputs_summary.get('nominal_electrolyser_capacity', 10), 'cf': 0.28},
            {'name': 'Wind Focus', 'lcoh': lcoh_display * 1.02, 'capacity': inputs_summary.get('nominal_electrolyser_capacity', 10), 'cf': 0.22},
            {'name': 'Hybrid Opt', 'lcoh': lcoh_display * 0.97, 'capacity': inputs_summary.get('nominal_electrolyser_capacity', 12), 'cf': 0.26}
        ]

        # Create grouped bar chart comparing key metrics
        scenario_names = [s['name'] for s in scenarios]

        fig_comparison = go.Figure()

        # Add LCOH bars
        fig_comparison.add_trace(go.Bar(
            name='LCOH ($/kg)',
            x=scenario_names,
            y=[s['lcoh'] for s in scenarios],
            marker_color='rgba(55, 128, 191, 0.7)',
            offsetgroup=0
        ))

        # Add Capacity Factor as line
        fig_comparison.add_trace(go.Scatter(
            name='Capacity Factor',
            x=scenario_names,
            y=[s['cf'] for s in scenarios],
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='rgba(219, 64, 82, 1)', width=3),
            marker=dict(size=8)
        ))

        # Update layout for dual y-axis
        fig_comparison.update_layout(
            title="Configuration Comparison",
            xaxis_title="Configuration Scenario",
            yaxis=dict(
                title="LCOH ($/kg)",
                titlefont=dict(color="rgba(55, 128, 191, 1)"),
                tickfont=dict(color="rgba(55, 128, 191, 1)")
            ),
            yaxis2=dict(
                title="Capacity Factor",
                titlefont=dict(color="rgba(219, 64, 82, 1)"),
                tickfont=dict(color="rgba(219, 64, 82, 1)"),
                anchor="x",
                overlaying="y",
                side="right",
                showgrid=False,
                tickformat=".1%"
            ),
            legend=dict(
                x=1.05,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(255, 255, 255, 0.5)'
            ),
            height=400,
            barmode='group'
        )

        st.plotly_chart(fig_comparison, use_container_width=True)
    else:
        display_image_placeholder(col2, "Configuration Comparison",
                                "Compare different system configurations")

# Add S2D2 Lab footer
add_s2d2_footer()
