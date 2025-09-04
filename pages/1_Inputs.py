import streamlit as st
from src.models.hydrogen_model import HydrogenModel  # Import the HydrogenModel class
from utils import add_s2d2_footer

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Inputs")

st.sidebar.header("Scope and Configuration")

# Initialize session state for results if not already present
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

with st.form("input_form"):
    # Sidebar inputs
    site_location = st.sidebar.selectbox("Site Location", [
       "Custom", "US.CA",  "US.TX", "US.NM", "US.AZ", "US.NV", "US.UT", "US.CO", "US.WY", "US.MT",
    ], key="site_location", index=0) # Set default to Custom
    
    # If site_location == "Custom", show input for custom location, set api parameters(longitude,latitude, api_key)
       # Initialize all input variables with default values
    default_latitude = 34.0522
    default_longitude = -118.2437
    if site_location == "Custom":
        st.sidebar.subheader("Custom Location Parameters")
        longitude = st.sidebar.number_input("Longitude", value=default_longitude, step=0.01, key="longitude")
        latitude = st.sidebar.number_input("Latitude", value=default_latitude, step=0.01, key="latitude")
        api_key = st.sidebar.text_input("API Key", type="password", key="api_key")
    
    power_plant_configuration = st.sidebar.selectbox("Power Plant Configuration", [
        "C1. Standalone Solar PV Generator with Electrolyser",
        "C2. Standalone Solar PV Generator with Electrolyser and Battery",
        "C3. Grid Connected Solar PV Generator with Electrolyser",
        "C4. Grid Connected Solar PV Generator with Electrolyser with Surplus Retailed to Grid",
        "C5. Grid Connected Solar PV Generator with Electrolyser and Battery",
        "C6. Grid Connected Solar PV Generator with Electrolyser and Battery with Surplus Retailed to Grid",
        "C7. Solar PPA with Electrolyser",
        "C8. Solar PPA with Electrolyser and Battery",
        "C9. Standalone Wind Generator with Electrolyser",
        "C10. Standalone Wind Generator with Electrolyser and Battery",
        "C11. Grid Connected Wind Generator with Electrolyser",
        "C12. Grid Connected Wind Generator with Electrolyser with Surplus Retailed to Grid",
        "C13. Grid Connected Wind Generator with Electrolyser and Battery",
        "C14. Grid Connected Wind Generator with Electrolyser and Battery with Surplus Retail to Gird",
        "C15. Wind PPA with Electrolyser",
        "C16. Wind PPA with Electrolyser and Battery",
        "C17. Standalone Hybrid Generator with Electrolyser",
        "C18. Standalone Hybrid Generator with Electrolyser and Battery",
        "C19. Grid Connected Hybrid Generator with Electrolyser",
        "C20. Grid Connected Hybrid Generator with Electrolyser with Surplus Retailed to Grid",
        "C21. Grid Connected Hybrid Generator with Electrolyser and Battery",
        "C22. Grid Connected Hybrid Generator with Electrolyser and Battery with Surplus Retailed to Grid",
        "C23. Hybrid PPA with Electrolyser",
        "C24. Hybrid PPA with Electrolyser and Battery"
    ], key="power_plant_configuration")
    
    electrolyser_choice = st.sidebar.selectbox("Electrolyser Choice", ["PEM"], key="electrolyser_choice")

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "System Sizing",
        "Electrolyser Parameters",
        "Power Plant Parameters",
        "Financial Parameters",
        
    ])

    user_inputs = {}

    with tab1:
        st.header("System Sizing")
        st.write("Define the capacities for the electrolyser, power plant, and battery.")

        st.subheader("Electrolyser Sizing")
        user_inputs['nominal_electrolyser_capacity'] = st.number_input(
        "Nominal Electrolyser Capacity (MW)", 
        min_value=0.0, 
        value=10.0, 
        step=0.1,
        help="The rated power capacity of the electrolyser in megawatts (MW)",
        key="nominal_electrolyser_capacity"
    )

        st.subheader("Power Plant Sizing")
        # Determine generator type based on configuration prefix
        generator_type = None
        if power_plant_configuration.startswith(("C1.", "C2.", "C3.", "C4.", "C5.", "C6.", "C7.", "C8.")):
            generator_type = "Solar PV"
        elif power_plant_configuration.startswith(("C9.", "C10.", "C11.", "C12.", "C13.", "C14.", "C15.", "C16.")):
            generator_type = "Wind"
        elif power_plant_configuration.startswith(("C17.", "C18.", "C19.", "C20.", "C21.", "C22.", "C23.", "C24.")):
            generator_type = "Hybrid"

        # Display generator type and relevant inputs
        st.write(f"Configuration Type: **{generator_type}**")
        user_inputs['generator_type'] = generator_type

        # Show relevant capacity inputs based on generator type
        if generator_type in ["Solar PV", "Hybrid"]:
            user_inputs['nominal_solar_farm_capacity'] = st.number_input(
                "Nominal Solar Farm Capacity (MW)", 
                min_value=0.0, 
                value=10.0, 
                step=0.1, 
                help="The rated power capacity of the solar farm in megawatts (MW). Consider instaled or local solar resource availability.",
                key="nominal_solar_farm_capacity"
            )

        if generator_type in ["Wind", "Hybrid"]:
            user_inputs['nominal_wind_farm_capacity'] = st.number_input(
                "Nominal Wind Farm Capacity (MW)", 
                min_value=0.0, 
                value=10.0, 
                step=0.5,
                help="The rated power capacity of the wind farm in megawatts (MW). Consider installed or local wind resource availability.",
                key="nominal_wind_farm_capacity"
            )

        # Show battery inputs only for configurations with battery
        is_battery_config = any(x in power_plant_configuration for x in ["C2.", "C5.", "C6.", "C8.", "C10.", "C13.", "C14.", "C16.", "C18.", "C21.", "C22.", "C24."])
        if is_battery_config:
            st.subheader("Battery Sizing")
            user_inputs['battery_rated_power'] = st.number_input(
                "Battery Rated Power (MW)", 
                min_value=0.0, 
                value=5.0, 
                step=0.5, 
                help="The maximum power output/input capability of the battery system in megawatts (MW)",
                key="battery_rated_power",
            
            )
            user_inputs['duration_of_storage_hours'] = st.selectbox(
                "Duration of Storage (hours)", 
                [1, 2, 4, 8], 
                help="The number of hours the battery can provide its rated power output when fully charged",
                key="duration_of_storage_hours"
            )
            Nominal_Battery_Capacity = user_inputs['battery_rated_power'] * user_inputs['duration_of_storage_hours']
            st.write(f"Nominal Battery Capacity: {Nominal_Battery_Capacity} MWh")
            
    with tab2:
        PEM_expander= st.expander("Electrolyser Design Parameters", expanded=False)
        # PEM_expander.header("Electrolyser Design Parameters")

        
        PEM_expander.subheader("Electrochemical design Parameters")
        #include here set of electrochemical and PEM electrolyser design parameters
        PEM_expander.write("Define design parameters for the electrolyser.")
        
        # active cell area, current density, anode and cathode thickness catlyst layer, membrane thickness, choice of catalysis layer, operating temperature, and pressure,limiting and exchange current densities
        # Create two columns for the input fields
        col1, col2 = PEM_expander.columns(2)

        # Column 1 inputs
        with col1:
            user_inputs['active_cell_area_m2'] = st.number_input("Active Cell Area (cm2)", min_value=0.0, value=10.0, step=0.1, key="active_cell_area_m2")
            user_inputs['current_density_amps_cm2'] = st.number_input("Current Density (A/cm2)", min_value=0.0, value=10.0, step=0.1, key="current_density_amps_cm2")
            user_inputs['anode_thickness_mm'] = st.number_input("Anode Thickness (mm)", min_value=0.0, value=0.1, step=0.01, key="anode_thickness_mm")
            user_inputs['cathode_thickness_mm'] = st.number_input("Cathode Thickness (mm)", min_value=0.0, value=0.1, step=0.01, key="cathode_thickness_mm")

        # Column 2 inputs
        with col2:
            user_inputs['membrane_thickness_mm'] = st.number_input("Membrane Thickness (mm)", min_value=0.0, value=0.1, step=0.01, key="membrane_thickness_mm")
            user_inputs['operating_temperature_c'] = st.number_input("Operating Temperature (C)", min_value=0.0, value=25.0, step=0.1, key="operating_temperature_c")
            user_inputs['operating_pressure_bar'] = st.number_input("Operating Pressure (bar)", min_value=0.0, value=1.0, step=0.1, key="operating_pressure_bar")
            user_inputs['limiting_current_density_amps_cm2'] = st.number_input("Limiting Current Density (A/cm2)", min_value=0.0, value=10.0, step=0.1, key="limiting_current_density_amps_cm2")
            user_inputs['exchange_current_density_amps_cm2'] = st.number_input("Exchange Current Density (A/cm2)", min_value=0.0, value=10.0, step=0.1, key="exchange_current_density_amps_cm2")
        
        # Electrolyser performance parameters
        Perf_params=st.expander("Electrolyser Performance Parameters", expanded=False)
        Perf_params.write("Define performance parameters for the electrolyser.")
        
        col1, col2 = Perf_params.columns(2)
        with col1:
            user_inputs['sec_at_nominal_load'] = st.number_input(
                "SEC at Nominal Load (kWh/kg)", 
                min_value=0.0, 
                value=50.0, 
                step=0.1,
                help="Specific energy consumption at nominal load",
                key="perf_sec_nominal"
            )
            user_inputs['total_system_sec'] = st.number_input(
                "Total System SEC (kWh/kg)",
                min_value=0.0, 
                value=50.0,
                help="Total system specific energy consumption",
                key="total_system_sec"
            )
            user_inputs['electrolyser_min_load'] = st.slider(
                "Minimum Load (%)", 
                min_value=0, 
                max_value=100, 
                value=10,
                help="Minimum operating load percentage",
                key="perf_min_load"
            )
        
        with col2:
            user_inputs['electrolyser_max_load'] = st.slider(
                "Maximum Load (%)", 
                min_value=0, 
                max_value=120, 
                value=100,
                help="Maximum operating load percentage",
                key="perf_max_load"
            )
            user_inputs['max_overload_duration'] = st.number_input(
                "Maximum Overload Duration (hrs)", 
                min_value=0, 
                value=0,
                help="Maximum duration system can operate in overload",
                key="max_overload_duration"
            )
            user_inputs['time_between_overload'] = st.number_input(
                "Time Between Overload (hrs)", 
                min_value=0, 
                value=0,
                help="Required cool-down time between overload operations",
                key="time_between_overload"
            )

        sys_capex = st.expander("System Capital Costs", expanded=False)
        sys_capex.write("Define capital costs for the electrolyser system.")
        
        col1, col2 = sys_capex.columns(2)
        with col1:
            user_inputs['reference_capacity'] = st.number_input(
                "Reference Capacity (kW)", 
                min_value=0.0, 
                value=1000.0,
                help="Reference capacity for cost scaling",
                key="ref_capacity"
            )
            user_inputs['reference_cost'] = st.number_input(
                "Reference Cost ($/kW)", 
                min_value=0.0, 
                value=1500.0,
                help="Reference cost at reference capacity",
                key="ref_cost"
            )
            
            #Economies of scale- Electrolyser equipment cost
            user_inputs['electrolyser_economies_of_scale_type'] = st.selectbox(
                "Economies of Scale Type", 
                ["Scale Index", "Self Defined", "Custom Curve Fitted"], 
                key="elec_eos_type"
            )
            
            # Different inputs based on economies of scale type
            if user_inputs['electrolyser_economies_of_scale_type'] == "Scale Index":
                user_inputs['scale_index'] = st.number_input(
                    "Scale Index", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.9,
                    help="Economies of scale index (0-1)",
                    key="scale_index_eos"  # Changed key to be unique
                )
            elif user_inputs['electrolyser_economies_of_scale_type'] == "Self Defined":
                user_inputs['electrolyser_economies_of_scale'] = st.number_input(
                    "Economies of Scale (% of CAPEX)", 
                    min_value=0.0, 
                    value=0.0,
                    key="elec_eos_self_defined"
                )
            elif user_inputs['electrolyser_economies_of_scale_type'] == "Custom Curve Fitted":
                # Provide coefficients for the curve-fitted model
                st.write("Provide coefficients for the curve-fitted model:")
                st.latex(r"Cost_{El}= C_1 + C_2*P_{EL,rated} + C_3*e^{C_4*P_{EL,rated}}")
                user_inputs['electrolyser_custom_c1'] = st.number_input("c1", value=1046.93, key="elec_c1")
                user_inputs['electrolyser_custom_c2'] = st.number_input("c2", value=-3.479, key="elec_c2")
                user_inputs['electrolyser_custom_c3'] = st.number_input("c3", value=61.567, key="elec_c3")
                user_inputs['electrolyser_custom_c4'] = st.number_input("c4", value=-0.261, key="elec_c4")

        with col2:
            user_inputs['land_cost_percent'] = st.number_input(
                "Land Cost (% of CAPEX)", 
                min_value=0.0, 
                value=6.0,
                help="Land procurement cost as percentage of capital cost",
                key="land_cost"
            )
            
            user_inputs['installation_cost_percent'] = st.number_input(
                "Installation Cost (% of CAPEX)", 
                min_value=0.0, 
                value=0.0,
                help="Installation cost as percentage of capital cost",
                key="installation_cost"
            )

        sys_opex = st.expander("System Operational Costs", expanded=False)
        sys_opex.write("Define operational costs for the electrolyser system.")
        
        col1, col2 = sys_opex.columns(2)
        with col1:
            user_inputs['om_cost_percent'] = st.number_input(
                "O&M Cost (% of CAPEX/year)", 
                min_value=0.0, 
                value=2.5,
                help="Annual operation and maintenance cost",
                key="om_cost"
            )
            user_inputs['stack_replacement_percent'] = st.number_input(
                "Stack Replacement (% of CAPEX)", 
                min_value=0.0, 
                value=40.0,
                help="Stack replacement cost as percentage of capital cost",
                key="stack_replacement"
            )

        with col2:
            user_inputs['water_cost'] = st.number_input(
                "Water Cost (A$/kL)", 
                min_value=0.0, 
                value=5.0,
                help="Cost of water per kiloliter",
                key="water_cost_kl"
            )
            user_inputs['other_operational_costs'] = st.number_input(
                "Other Operational Costs (A$/year)", 
                min_value=0.0, 
                value=0.0,
                help="Additional operational costs per year",
                key="other_opex"
            )

    with tab3:
        
        
        st.header("Power Plant Parameters")
        Power_sys_design= st.expander("Power System Design Parameters", expanded=False)
        Power_sys_design.subheader("Power System Design Parameters")
        # define input for the ind and pv api data call
        
        # Solar PV API call parameters
        Power_sys_design.subheader("Solar Photovoltaic Power (PV)")
        solar_col1, solar_col2 = Power_sys_design.columns(2)

        with solar_col1:
            solar_dataset = st.selectbox("Dataset", ["MERRA-2 (global)", "Other"], key="solar_dataset")
            solar_year = st.selectbox("Select a year of data", list(range(2000, 2026)), index=24, key="solar_year")
            solar_capacity_kw = st.number_input("Capacity (kW)", min_value=0.0, value=1.0, step=0.1, key="solar_capacity_kw")
            solar_system_loss = st.number_input("System loss (fraction)", min_value=0.0, value=0.1, step=0.01, key="solar_system_loss")

        with solar_col2:
            solar_tracking = st.selectbox("Tracking", ["None", "Single-axis", "Dual-axis"], key="solar_tracking")
            solar_tilt = st.number_input("Tilt (°)", min_value=0.0, value=35.0, step=1.0, key="solar_tilt")
            solar_azimuth = st.number_input("Azimuth (°)", min_value=0.0, value=180.0, step=1.0, key="solar_azimuth")
            solar_include_raw_data = st.checkbox("Include raw data", key="solar_include_raw_data")

        # Wind  Power design and API call parameters
        Power_sys_design.subheader("Wind Power")
        wind_col1, wind_col2 = Power_sys_design.columns(2)

        with wind_col1:
            wind_dataset = st.selectbox("Dataset", ["MERRA-2 (global)", "Other"], key="wind_dataset")
            wind_year = st.selectbox("Select a year of data", list(range(2000, 2026)), index=24, key="wind_year")
            wind_capacity_kw = st.number_input("Capacity (kW)", min_value=0.0, value=1.0, step=0.1, key="wind_capacity_kw")

        with wind_col2:
            wind_hub_height_m = st.number_input("Hub height (m)", min_value=0.0, value=80.0, step=1.0, key="wind_hub_height_m")
            wind_turbine_model = st.selectbox("Turbine model", ["Vestas V90 2000", "Other"], key="wind_turbine_model")
            wind_include_raw_data = st.checkbox("Include raw data", key="wind_include_raw_data")
        
        Power_sys_design.write("Define degradation rates, capital, and operating costs for the power plant.")
        Power_sys_design.subheader("Operational Parameters")
        user_inputs['solar_pv_degradation_rate_percent_year'] = Power_sys_design.number_input("Solar PV Degradation Rate (%/year)", min_value=0.0, value=0.5, step=0.01, key="solar_degradation")
        user_inputs['wind_farm_degradation_rate_percent_year'] = Power_sys_design.number_input("Wind Farm Degradation Rate (%/year)", min_value=0.0, value=0.5, step=0.01, key="wind_degradation")

        #---------------- expander---------------------
        solar_farm_params = st.expander("Solar Farm Parameters", expanded=False)
        solar_farm_params.subheader("Solar Farm Parameters")
        
        # Create two columns for solar farm parameters
        sol_col1, sol_col2 = solar_farm_params.columns(2)
        
        with sol_col1:
            # Solar Farm Build Costs
            user_inputs['solar_reference_capacity'] = st.number_input(
                "Reference Capacity of Solar Farm (kW)", 
                min_value=0.0, 
                value=1000.0, 
                key="solar_ref_capacity"
            )
            user_inputs['solar_reference_equipment_cost'] = st.number_input(
                "Reference Solar PV Farm Equipment Cost (A$/kW)", 
                min_value=0.0, 
                value=1500.0, 
                key="solar_ref_equipment_cost"
            )
            
            # Economies of Scale Profile
            user_inputs['solar_scale_index'] = st.number_input(
                "Scale Index", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.9,
                help="Scale index for economies of scale profile",
                key="solar_scale_index"
            )
            user_inputs['solar_cost_reduction'] = st.number_input(
                "Solar Farm Equipment Cost Reduction with Scale (%)", 
                min_value=0.0, 
                value=0.0,
                key="solar_cost_reduction"
            )

        with sol_col2:
            # Installation and Land Costs
            user_inputs['solar_installation_costs'] = st.number_input(
                "Installation Costs (% of CAPEX)", 
                min_value=0.0, 
                value=0.0,
                key="solar_installation_costs"
            )
            user_inputs['solar_land_cost'] = st.number_input(
                "Land Procurement Cost (% of CAPEX)", 
                min_value=0.0, 
                value=8.0,
                key="solar_land_cost"
            )
            # Operating Costs
            user_inputs['solar_opex'] = st.number_input(
                "OPEX (Fixed & Variable O&M) (A$/MW/year)", 
                min_value=0.0, 
                value=17000.0,
                key="solar_opex"
            )
            user_inputs['solar_degradation'] = st.number_input(
                "Solar Degradation (%/yr)", 
                min_value=0.0, 
                value=0.1,
                help="Decrease in solar farm output per year",
                key="solar_annual_degradation"
            )
        #---------------- expander --------------------
        wind_farm_params = st.expander("Wind Farm Parameters", expanded=False)
        wind_farm_params.subheader("Wind Farm Parameters")
        
        # Create two columns for wind farm parameters
        wind_col1, wind_col2 = wind_farm_params.columns(2)
        
        with wind_col1:
            # Wind Farm Capital Costs
            user_inputs['wind_reference_capacity'] = st.number_input(
                "Reference Capacity of Wind Farm (kW)", 
                min_value=0.0, 
                value=1000.0,
                key="wind_ref_capacity"
            )
            user_inputs['wind_reference_cost'] = st.number_input(
                "Reference Wind Farm Cost (A$/kW)", 
                min_value=0.0, 
                value=3000.0,
                key="wind_ref_cost"
            )
            
            # Economies of Scale Profile
            user_inputs['wind_scale_index'] = st.number_input(
                "Scale Index", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.9,
                help="Scale index for economies of scale profile",
                key="wind_scale_index"
            )
            user_inputs['wind_cost_reduction'] = st.number_input(
                "Wind Farm Equipment Cost Reduction with Scale (%)", 
                min_value=0.0, 
                value=0.0,
                key="wind_cost_reduction"
            )

        with wind_col2:
            # Installation and Land Costs
            user_inputs['wind_installation_costs'] = st.number_input(
                "Installation Costs (% of CAPEX)", 
                min_value=0.0, 
                value=0.0,
                key="wind_installation_costs"
            )
            user_inputs['wind_land_cost'] = st.number_input(
                "Land Procurement Cost (% of CAPEX)", 
                min_value=0.0, 
                value=8.0,
                key="wind_land_cost"
            )
            # Operating Costs
            user_inputs['wind_opex'] = st.number_input(
                "OPEX (Fixed & Variable O&M) (A$/MW/year)", 
                min_value=0.0, 
                value=25000.0,
                key="wind_opex"
            )
            user_inputs['wind_degradation'] = st.number_input(
                "Wind Degradation (%/yr)", 
                min_value=0.0, 
                value=0.1,
                help="Decrease in wind farm output per year",
                key="wind_annual_degradation"
            )

        ##--------- Grid and Battery Parameters----------------------------  
        
        grid_battery_params = st.expander("Grid and Battery Configuration Parameters", expanded=False)
    
        col1, col2 = grid_battery_params.columns(2)
        with col1:
            st.subheader("Grid Connection Configuration")  
            user_inputs['grid_connection_cost_percent'] = st.number_input(
                "Grid Connection Cost (% of CAPEX)", 
                min_value=0.0, 
                value=0.0,
                key="grid_connection_cost"
            )
            user_inputs['grid_service_charge_percent'] = st.number_input(
                "Grid Service Charge (% of CAPEX)", 
                min_value=0.0, 
                value=0.0,
                key="grid_service_charge"
            )
        with col2:
            st.subheader("PPA Configuration")
            user_inputs['principal_ppa_cost_percent'] = st.number_input(
                "Principal PPA Cost (% of CAPEX)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                key="principal_ppa_cost"
            )
            user_inputs['transmission_connection_cost_percent'] = st.number_input(
                "Transmission Connection Cost (% of CAPEX)", 
                min_value=0.0, 
                value=0.0,
                key="transmission_connection_cost"
            )

        grid_battery_params.header("Battery Parameters")
        grid_battery_params.write("Define performance and cost parameters for the battery.")
        grid_battery_params.write("Operational Parameters")
        col1, col2 , col2= grid_battery_params.columns(3)
        with col1:
            user_inputs['round_trip_efficiency'] = st.number_input("Round Trip Efficiency (%)", min_value=0.0, value=90.0, step=0.1, key="rte")
        
        # Create a single range slider
        min_soc, max_soc = grid_battery_params.slider(
            "Minimum and Maximum State of Charge (%)",
            min_value=0,
            max_value=100,
            value=(10, 90),  # A tuple sets the initial minimum and maximum values
            key="soc_range"
        )
        # Store the values in your user_inputs dictionary
        user_inputs["minimum_state_of_charge"] = min_soc
        user_inputs["maximum_state_of_charge"] = max_soc
        
        col1, col2 = grid_battery_params.columns(2)
        with col1:
            
            st.subheader("Capital Costs")
            user_inputs['battery_capex_a_kwh'] = st.number_input(
                "Battery CAPEX (A$/kWh)", 
                min_value=0.0, 
                value=300.0, 
                step=1.0, 
                key="batt_capex")
            user_inputs['battery_indirect_costs_percent_of_capex'] = st.number_input("Battery Indirect Costs (% of CAPEX)", min_value=0.0, value=10.0, step=0.1, key="batt_indirect_cost")
        with col2:
            st.subheader("Operating Costs")
            user_inputs['battery_replacement_cost_of_capex'] = st.number_input(
                "Battery Replacement Cost (% of CAPEX)", 
                min_value=0.0, 
                value=50.0, 
                step=0.1, 
                key="batt_replacement_cost"
                )
            user_inputs['battery_opex_a_mw_yr'] = st.number_input(
                "Battery OPEX (A$/MW/yr)", 
                min_value=0.0, 
                value=10.0, 
                step=0.1, 
                key="batt_opex")


    with tab4:
        st.header("Financial Parameters")
        
        # Project Basics
        project_basics = st.expander("Project Basics", expanded=True)
        col1, col2 = project_basics.columns(2)
        
        with col1:
            user_inputs['plant_life_years'] = st.number_input(
                "Plant Life (years)", 
                min_value=1, 
                value=25, 
                step=1,
                help="Expected operational lifetime of the plant",
                key="plant_life"
            )
            user_inputs['discount_rate'] = st.number_input(
                "Discount Rate (%)", 
                min_value=0.0, 
                value=5.0, 
                step=0.1,
                help="Annual discount rate for future cash flows",
                key="discount_rate"
            )
        
        with col2:
            user_inputs['inflation_rate'] = st.number_input(
                "Inflation Rate (%)", 
                min_value=0.0, 
                value=2.0, 
                step=0.1,
                help="Annual inflation rate",
                key="inflation_rate"
            )
            user_inputs['tax_rate'] = st.number_input(
                "Tax Rate (%)", 
                min_value=0.0, 
                value=30.0, 
                step=0.1,
                help="Corporate tax rate",
                key="tax_rate"
            )

        # Investment Structure
        investment = st.expander("Investment Structure", expanded=False)
        col1, col2 = investment.columns(2)
        
        with col1:
            user_inputs['financing_via_equity'] = st.number_input(
                "Financing via Equity (%)", 
                min_value=0.0, 
                value=30.0, 
                step=0.1,
                help="Percentage of project financed through equity",
                key="financing_equity"
            )
            user_inputs['direct_equity_of_total_equity'] = st.number_input(
                "Direct Equity (% of Total Equity)", 
                min_value=0.0, 
                value=100.0, 
                step=0.1,
                help="Percentage of equity that is direct investment",
                key="direct_equity"
            )
        
        with col2:
            user_inputs['loan_term_years'] = st.number_input(
                "Loan Term (years)", 
                min_value=1, 
                value=10, 
                step=1,
                help="Duration of the loan",
                key="loan_term"
            )
            user_inputs['interest_rate_on_loan_p_a'] = st.number_input(
                "Interest Rate on Loan (% p.a.)", 
                min_value=0.0, 
                value=5.0, 
                step=0.1,
                help="Annual interest rate on the loan",
                key="interest_rate"
            )

        # Depreciation Settings
        depreciation = st.expander("Depreciation Settings", expanded=False)
        depreciation.write("Define the depreciation method for the project assets.")
        user_inputs['depreciation_profile'] = depreciation.selectbox(
            "Depreciation Profile",
            [
                "Straight Line",
                "Modified Accelerated Cost Recovery System (3 years)",
                "Modified Accelerated Cost Recovery System (5 years)",
                "Modified Accelerated Cost Recovery System (7 years)",
                "Modified Accelerated Cost Recovery System (10 years)",
                "Modified Accelerated Cost Recovery System (15 years)",
                "Modified Accelerated Cost Recovery System (20 years)"
            ],
            help="Method used to calculate asset depreciation",
            key="depreciation_profile"
        )

        # End of Life Costs
        eol_costs = st.expander("End of Life Costs", expanded=False)
        col1, col2 = eol_costs.columns(2)
        
        with col1:
            user_inputs['salvage_costs_of_total_investments'] = st.number_input(
                "Salvage Value (% of Total Investments)", 
                min_value=0.0, 
                value=5.0, 
                step=0.1,
                help="Expected salvage value at end of project life",
                key="salvage_costs"
            )
        
        with col2:
            user_inputs['decommissioning_costs_of_total_investments'] = st.number_input(
                "Decommissioning Costs (% of Total Investments)", 
                min_value=0.0, 
                value=5.0, 
                step=0.1,
                help="Expected costs for decommissioning",
                key="decommissioning_costs"
            )

        # Additional Revenue and Costs
        additional = st.expander("Additional Revenue and Costs", expanded=False)
        
        additional.subheader("Additional Costs")
        col1, col2 = additional.columns(2)
        
        with col1:
            user_inputs['additional_upfront_costs_a'] = st.number_input(
                "Additional Upfront Costs (A$)", 
                min_value=0.0, 
                value=0.0, 
                step=1000.0,
                help="One-time additional costs at project start",
                key="additional_upfront_costs"
            )
        
        with col2:
            user_inputs['additional_annual_costs_a_yr'] = st.number_input(
                "Additional Annual Costs (A$/yr)", 
                min_value=0.0, 
                value=0.0, 
                step=100.0,
                help="Recurring annual additional costs",
                key="additional_annual_costs"
            )

        additional.subheader("Additional Revenue Streams")
        col1, col2 = additional.columns(2)
        
        with col1:
            user_inputs['average_electricity_spot_price_a_mwh'] = st.number_input(
                "Average Electricity Spot Price (A$/MWh)", 
                min_value=0.0, 
                value=0.0, 
                step=0.01,
                help="Expected electricity spot price for grid sales",
                key="avg_elec_spot_price"
            )
        
        with col2:
            user_inputs['oxygen_retail_price_a_kg'] = st.number_input(
                "Oxygen Retail Price (A$/kg)", 
                min_value=0.0, 
                value=0.0, 
                step=0.01,
                help="Expected retail price for oxygen byproduct",
                key="oxygen_retail_price"
            )

    submitted = st.form_submit_button("Calculate")

    if submitted:
        # Add sidebar inputs to user_inputs dictionary
        user_inputs['site_location'] = site_location
        user_inputs['power_plant_configuration'] = power_plant_configuration
        user_inputs['electrolyser_choice'] = electrolyser_choice

        try:
            model = HydrogenModel(**user_inputs)
            results = model.run()

            st.session_state.model_results = {
                "operating_outputs": results,
                "lcoh": results["lcoh"],
                "business_case": results.get("business_case", {}),
                "inputs_summary": user_inputs # Store inputs for results page
            }
            st.success("Calculation complete! Navigate to the Results page to view the outputs.")
        except KeyError as e:
            st.error(f"Missing input: {e}. Please ensure all required fields are filled.")
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")

# Add S2D2 Lab footer
add_s2d2_footer()
