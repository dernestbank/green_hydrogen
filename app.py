import streamlit as st
from src.models.hydrogen_model import HydrogenModel

def main():
    st.set_page_config(layout="wide")
    st.title("Green Hydrogen Production Framework")

    # Initialize all input variables with default values
    latitude = 34.0522
    longitude = -118.2437
    us_states = ["", "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
                     "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
                     "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
                     "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
                     "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
                     "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
                     "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
                     "Wisconsin", "Wyoming"]
    selected_state = ""

    nominal_electrolyser_capacity = 10.0
    nominal_power_plant_capacity = 20.0
    nominal_battery_capacity = 5.0
    battery_storage_duration = 4.0

    power_plant_scenarios = [
        "Scenario 1: Solar PV Only", "Scenario 2: Wind Only", "Scenario 3: Solar PV + Wind",
        "Scenario 4: Solar PV + Battery", "Scenario 5: Wind + Battery", "Scenario 6: Solar PV + Wind + Battery",
        "Scenario 7: Grid Only", "Scenario 8: Solar PV + Grid", "Scenario 9: Wind + Grid",
        "Scenario 10: Solar PV + Wind + Grid", "Scenario 11: Solar PV + Battery + Grid",
        "Scenario 12: Wind + Battery + Grid", "Scenario 13: Solar PV + Wind + Battery + Grid",
        "Scenario 14: Solar PV (Fixed Tilt)", "Scenario 15: Solar PV (Single-Axis Tracking)",
        "Scenario 16: Solar PV (Dual-Axis Tracking)", "Scenario 17: Wind (Onshore)", "Scenario 18: Wind (Offshore)",
        "Scenario 19: Solar PV + Wind (Optimized)", "Scenario 20: Solar PV + Battery (Optimized)",
        "Scenario 21: Wind + Battery (Optimized)", "Scenario 22: Solar PV + Wind + Battery (Optimized)",
        "Scenario 23: Grid (Time-of-Use)", "Scenario 24: Grid (Fixed Price)"
    ]
    selected_power_plant_scenario = power_plant_scenarios[0]  # Initialize with the first scenario

    electrolyser_choice = "PEM"
    sec = 50.0
    load_range_min = 10.0
    load_range_max = 100.0
    overloading = 120.0
    degradation = 1.0
    stack_replacement = 10.0
    water_requirement = 9.0
    electrolyser_capital_cost = 700.0
    electrolyser_indirect_cost = 20.0
    custom_cost_curve = False
    economies_of_scale = 0.9
    electrolyser_operating_cost = 0.5

    power_plant_degradation_rates = 0.5
    power_plant_capex = 1000.0
    power_plant_indirect_cost = 15.0
    power_plant_om = 20.0
    grid_connection_costs = 50.0
    ppa_costs = 30.0
    renewables_ninja_api_key = ""

    battery_efficiency = 90.0
    soc_min = 10.0
    soc_max = 90.0
    battery_capex = 300.0
    battery_indirect_costs = 10.0
    battery_replacement_cost = 200.0
    battery_opex = 5.0

    upfront_costs = 0.0
    annual_costs = 0.0

    surplus_electricity_retail = 0.0
    by_product_oxygen_retail = 0.0

    plant_life = 20
    discount_rate = 8.0
    investment_breakdown_equity = 30.0
    investment_breakdown_debt = 70.0
    salvage_costs = 0.0
    decommissioning_costs = 0.0
    inflation_rate = 2.0
    tax_rate = 25.0
    depreciation_profile = "Straight-line"


    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Input", "Results", "Levelised Cost Analysis", "Cash Flow Analysis", "Raw Data", "Working Data"])

    if page == "Input":
        st.header("Input Parameters")
        st.write("This is the input page where users can configure the simulation parameters.")

        st.sidebar.subheader("Site Location")
        latitude = st.sidebar.number_input("Latitude", value=latitude, format="%.4f")
        longitude = st.sidebar.number_input("Longitude", value=longitude, format="%.4f")
        selected_state = st.sidebar.selectbox("Pre-defined US States/Regions", us_states, index=us_states.index(selected_state) if selected_state in us_states else 0)

        st.sidebar.subheader("System Sizing")
        nominal_electrolyser_capacity = st.sidebar.number_input("Nominal Electrolyser Capacity (MW)", value=nominal_electrolyser_capacity, format="%.2f")
        nominal_power_plant_capacity = st.sidebar.number_input("Nominal Power Plant Capacity (MW)", value=nominal_power_plant_capacity, format="%.2f")
        nominal_battery_capacity = st.sidebar.number_input("Nominal Battery Capacity (MWh)", value=nominal_battery_capacity, format="%.2f")
        battery_storage_duration = st.sidebar.number_input("Battery Storage Duration (hours)", value=battery_storage_duration, format="%.2f")

        st.sidebar.subheader("Power Plant Configuration")
        selected_power_plant_scenario = st.sidebar.selectbox("Power Plant Configuration", power_plant_scenarios, index=power_plant_scenarios.index(selected_power_plant_scenario))

        st.subheader("Electrolyser Parameters")
        electrolyser_choice = st.selectbox("Electrolyser Choice", ["PEM", "Alkaline"], index=["PEM", "Alkaline"].index(electrolyser_choice))
        sec = st.number_input("Specific Energy Consumption (kWh/kg H2)", value=sec, format="%.2f")
        load_range_min = st.number_input("Load Range Minimum (%)", value=load_range_min, format="%.2f")
        load_range_max = st.number_input("Load Range Maximum (%)", value=load_range_max, format="%.2f")
        overloading = st.number_input("Overloading (%)", value=overloading, format="%.2f")
        degradation = st.number_input("Degradation (%/year)", value=degradation, format="%.2f")
        stack_replacement = st.number_input("Stack Replacement (years)", value=stack_replacement, format="%.2f")
        water_requirement = st.number_input("Water Requirement (L/kg H2)", value=water_requirement, format="%.2f")
        electrolyser_capital_cost = st.number_input("Electrolyser Capital Cost ($/kW)", value=electrolyser_capital_cost, format="%.2f")
        electrolyser_indirect_cost = st.number_input("Electrolyser Indirect Cost (%)", value=electrolyser_indirect_cost, format="%.2f")
        custom_cost_curve = st.checkbox("Use Custom Cost Curve", value=custom_cost_curve)
        if custom_cost_curve:
            st.write("Implement custom cost curve input here.") # Placeholder for custom curve input
        economies_of_scale = st.number_input("Economies of Scale Factor", value=economies_of_scale, format="%.2f")
        electrolyser_operating_cost = st.number_input("Electrolyser Operating Cost ($/kg H2)", value=electrolyser_operating_cost, format="%.2f")
        st.subheader("Electrochemical Design Parameters")
        # Placeholder for electrochemical design parameters
        st.write("Electrochemical design parameters will be implemented here.")

        st.subheader("Power Plant Parameters")
        power_plant_degradation_rates = st.number_input("Power Plant Degradation Rates (%/year)", value=power_plant_degradation_rates, format="%.2f")
        power_plant_capex = st.number_input("Power Plant CAPEX ($/kW)", value=power_plant_capex, format="%.2f")
        power_plant_indirect_cost = st.number_input("Power Plant Indirect Cost (%)", value=power_plant_indirect_cost, format="%.2f")
        power_plant_om = st.number_input("Power Plant O&M ($/kW/year)", value=power_plant_om, format="%.2f")
        grid_connection_costs = st.number_input("Grid Connection Costs ($/kW)", value=grid_connection_costs, format="%.2f")
        ppa_costs = st.number_input("PPA Costs ($/MWh)", value=ppa_costs, format="%.2f")
        renewables_ninja_api_key = st.text_input("Renewables Ninja API Key", value=renewables_ninja_api_key)
        
        with st.expander("Renewables Ninja Solar PV Advanced Parameters"):
            st.write("Solar PV advanced parameters will be implemented here.")
        with st.expander("Renewables Ninja Wind Advanced Parameters"):
            st.write("Wind advanced parameters will be implemented here.")

        st.subheader("Battery Parameters")
        battery_efficiency = st.number_input("Battery Efficiency (%)", value=battery_efficiency, format="%.2f")
        soc_min = st.number_input("State of Charge Minimum (%)", value=soc_min, format="%.2f")
        soc_max = st.number_input("State of Charge Maximum (%)", value=soc_max, format="%.2f")
        battery_capex = st.number_input("Battery CAPEX ($/kWh)", value=battery_capex, format="%.2f")
        battery_indirect_costs = st.number_input("Battery Indirect Costs (%)", value=battery_indirect_costs, format="%.2f")
        battery_replacement_cost = st.number_input("Battery Replacement Cost ($/kWh)", value=battery_replacement_cost, format="%.2f")
        battery_opex = st.number_input("Battery OPEX ($/kWh/year)", value=battery_opex, format="%.2f")

        st.subheader("Additional Costs")
        upfront_costs = st.number_input("Upfront Costs ($)", value=upfront_costs, format="%.2f")
        annual_costs = st.number_input("Annual Costs ($/year)", value=annual_costs, format="%.2f")

        st.subheader("Additional Revenue Streams")
        surplus_electricity_retail = st.number_input("Surplus Electricity Retail ($/MWh)", value=surplus_electricity_retail, format="%.2f")
        by_product_oxygen_retail = st.number_input("By-product Oxygen Retail ($/kg O2)", value=by_product_oxygen_retail, format="%.2f")

        st.subheader("Financing Parameters")
        plant_life = st.number_input("Plant Life (years)", value=plant_life, format="%d")
        discount_rate = st.number_input("Discount Rate (%)", value=discount_rate, format="%.2f")
        investment_breakdown_equity = st.number_input("Investment Breakdown: Equity (%)", value=investment_breakdown_equity, format="%.2f")
        investment_breakdown_debt = st.number_input("Investment Breakdown: Debt (%)", value=investment_breakdown_debt, format="%.2f")
        salvage_costs = st.number_input("Salvage Costs ($)", value=salvage_costs, format="%.2f")
        decommissioning_costs = st.number_input("Decommissioning Costs ($)", value=decommissioning_costs, format="%.2f")
        inflation_rate = st.number_input("Inflation Rate (%)", value=inflation_rate, format="%.2f")
        tax_rate = st.number_input("Tax Rate (%)", value=tax_rate, format="%.2f")
        depreciation_profile = st.selectbox("Depreciation Profile", ["Straight-line", "Declining Balance"], index=["Straight-line", "Declining Balance"].index(depreciation_profile))

    elif page == "Results":
        st.header("Simulation Results")
        st.subheader("Summary of Inputs")
        st.write(f"Latitude: {latitude}")
        st.write(f"Longitude: {longitude}")
        st.write(f"Selected State: {selected_state}")
        st.write(f"Nominal Electrolyser Capacity: {nominal_electrolyser_capacity} MW")
        st.write(f"Nominal Power Plant Capacity: {nominal_power_plant_capacity} MW")
        st.write(f"Nominal Battery Capacity: {nominal_battery_capacity} MWh")
        st.write(f"Battery Storage Duration: {battery_storage_duration} hours")
        st.write(f"Power Plant Configuration: {selected_power_plant_scenario}")
        st.write(f"Electrolyser Choice: {electrolyser_choice}")
        st.write(f"Specific Energy Consumption: {sec} kWh/kg H2")
        st.write(f"Load Range: {load_range_min}% - {load_range_max}%")
        st.write(f"Overloading: {overloading}%")
        st.write(f"Degradation: {degradation}%/year")
        st.write(f"Stack Replacement: {stack_replacement} years")
        st.write(f"Water Requirement: {water_requirement} L/kg H2")
        st.write(f"Electrolyser Capital Cost: ${electrolyser_capital_cost}/kW")
        st.write(f"Electrolyser Indirect Cost: {electrolyser_indirect_cost}%")
        st.write(f"Custom Cost Curve: {custom_cost_curve}")
        st.write(f"Economies of Scale Factor: {economies_of_scale}")
        st.write(f"Electrolyser Operating Cost: ${electrolyser_operating_cost}/kg H2")
        st.write(f"Power Plant Degradation Rates: {power_plant_degradation_rates}%/year")
        st.write(f"Power Plant CAPEX: ${power_plant_capex}/kW")
        st.write(f"Power Plant Indirect Cost: {power_plant_indirect_cost}%")
        st.write(f"Power Plant O&M: ${power_plant_om}/kW/year")
        st.write(f"Grid Connection Costs: ${grid_connection_costs}/kW")
        st.write(f"PPA Costs: ${ppa_costs}/MWh")
        st.write(f"Renewables Ninja API Key: {'Provided' if renewables_ninja_api_key else 'Not Provided'}")
        st.write(f"Battery Efficiency: {battery_efficiency}%")
        st.write(f"State of Charge: {soc_min}% - {soc_max}%")
        st.write(f"Battery CAPEX: ${battery_capex}/kWh")
        st.write(f"Battery Indirect Costs: {battery_indirect_costs}%")
        st.write(f"Battery Replacement Cost: ${battery_replacement_cost}/kWh")
        st.write(f"Battery OPEX: ${battery_opex}/kWh/year")
        st.write(f"Upfront Costs: ${upfront_costs}")
        st.write(f"Annual Costs: ${annual_costs}/year")
        st.write(f"Surplus Electricity Retail: ${surplus_electricity_retail}/MWh")
        st.write(f"By-product Oxygen Retail: ${by_product_oxygen_retail}/kg O2")
        st.write(f"Plant Life: {plant_life} years")
        st.write(f"Discount Rate: {discount_rate}%")
        st.write(f"Investment Breakdown: Equity {investment_breakdown_equity}%, Debt {investment_breakdown_debt}%")
        st.write(f"Salvage Costs: ${salvage_costs}")
        st.write(f"Decommissioning Costs: ${decommissioning_costs}")
        st.write(f"Inflation Rate: {inflation_rate}%")
        st.write(f"Tax Rate: {tax_rate}%")
        st.write(f"Depreciation Profile: {depreciation_profile}")

        st.subheader("Key Results Summary")
        
        # Run the model
        params = locals()
        model = HydrogenModel(**params)
        results = model.run()

        st.write(f"Levelised Cost of Hydrogen (LCOH): ${results['lcoh']:.2f}/kg")
        st.write(f"Annual Hydrogen Production: {results['annual_hydrogen_production']:,} kg")
        st.write(f"Capacity Factor: {results['capacity_factor']:.2%}")

        st.subheader("Visualizations")
        st.write("Figures 8-14 will be implemented here using Plotly.")
        # Placeholder for Plotly figures
        st.write("Figure 8: Annual duration curves for the power plant and electrolyser.")
        st.write("Figure 9: Interactive plot showing the hourly capacity factors of the power plant and electrolyser.")
        st.write("Figure 10: Waterfall plot showing the relative components of the LCH2.")
        st.write("Figure 11: Breakdown of the components of the capital and indirect cost.")
        st.write("Figure 12: Annual sales plot.")
        st.write("Figure 13: Annual operating cost plot.")
        st.write("Figure 14: Cumulative cash flow.")

    elif page == "Levelised Cost Analysis":
        st.header("Levelised Cost Analysis")
        st.write("This page will show the detailed cost breakdown.")
        st.subheader("Costs Summary")
        st.write("Tables/plots for costs will be displayed here.")
        st.subheader("Operational Profile")
        st.write("Tables/plots for operational profile will be displayed here.")
        st.subheader("Cash Flows")
        st.write("Tables/plots for cash flows will be displayed here.")
    elif page == "Cash Flow Analysis":
        st.header("Cash Flow Analysis")
        st.write("This page will present the cash flow analysis.")
        st.subheader("Detailed Cash Flows")
        st.write("Detailed cash flow tables/plots will be displayed here.")
        st.subheader("Net Profit, ROI, and Payback Period")
        st.write("Net Profit, ROI, and Payback Period will be displayed here.")
    elif page == "Raw Data":
        st.header("Raw Data")
        st.write("This page will display the raw solar and wind trace data.")
        st.subheader("Hourly Solar and Wind Traces")
        st.write("Tables/plots for hourly solar and wind traces will be displayed here.")
    elif page == "Working Data":
        st.header("Working Data")
        st.write("This page will display the hourly electrolyser operation and outputs.")
        st.subheader("Hourly Electrolyser Operation and Outputs")
        st.write("Tables/plots for hourly electrolyser operation and outputs will be displayed here.")

if __name__ == "__main__":
    main()

    if st.button("Run Simulation"):
        st.write("Running simulation...")
        # In a real application, you would collect all the parameters and pass them to the model
        # For now, we just show a message
        st.success("Simulation complete!")