import streamlit as st
from utils import add_s2d2_footer

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Results")

st.header("Summary of Key Inputs and Results")

if st.session_state.model_results:
    results = st.session_state.model_results
    inputs_summary = results['inputs_summary']
    operating_outputs = results['operating_outputs']
    lcoh = results['lcoh']
    business_case = results['business_case']

    st.markdown(f"""
    **Key Inputs:**
    - **Location:** {inputs_summary.get('site_location', 'N/A')}
    - **Configuration:** {inputs_summary.get('power_plant_configuration', 'N/A')}
    - **Electrolyser Capacity:** {inputs_summary.get('nominal_electrolyser_capacity', 'N/A')} MW
    - **Power Plant Capacity:** Solar: {inputs_summary.get('nominal_solar_farm_capacity', 'N/A')} MW, Wind: {inputs_summary.get('nominal_wind_farm_capacity', 'N/A')} MW
    - **Battery Capacity:** {inputs_summary.get('battery_rated_power', 'N/A')} MW ({inputs_summary.get('duration_of_storage_hours', 'N/A')} hours)

    **Key Results:**
    - **Power Plant Capacity Factor:** {operating_outputs.get('Generator Capacity Factor', 'N/A'):.2%}
    - **Time Electrolyser is at its Maximum Capacity:** {operating_outputs.get('Time Electrolyser is at its Rated Capacity', 'N/A'):.2%}
    - **Total Time Electrolyser is Operating:** {operating_outputs.get('Total Time Electrolyser is Operating', 'N/A'):.2%}
    - **Achieved Electrolyser Capacity Factor:** {operating_outputs.get('Achieved Electrolyser Capacity Factor', 'N/A'):.2%}
    - **Energy Consumed by Electrolyser:** {operating_outputs.get('Energy in to Electrolyser [MWh/yr]', 'N/A'):,.2f} MWh/yr
    - **Surplus Energy Not Utilised by Electrolyser:** {operating_outputs.get('Surplus Energy [MWh/yr]', 'N/A'):,.2f} MWh/yr
    - **Hydrogen Output (Fixed SEC):** {operating_outputs.get('Hydrogen Output for Fixed Operation [t/yr]', 'N/A'):,.2f} t/yr
    - **Hydrogen Output (Variable SEC):** {operating_outputs.get('Hydrogen Output for Variable Operation [t/yr]', 'N/A'):,.2f} t/yr
    - **LCH2:** {lcoh:.2f} $/kg
    - **Net Profit:** {business_case.get('Net Profit', 'N/A')}
    - **Return on Investment:** {business_case.get('ROI', 'N/A')}
    - **Payback Period:** {business_case.get('Payback Period', 'N/A')}
    """)
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
    display_image_placeholder(col1, "Annual Duration Curves", 
                            "Shows duration curves for power plant and electrolyser")
    
    display_image_placeholder(col1, "Interactive Hourly Capacity Factors", 
                            "Interactive plot of hourly capacity factors")
    
    display_image_placeholder(col1, "Waterfall Plot - LCH2 Components", 
                            "Breakdown of LCH2 components")
    
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
