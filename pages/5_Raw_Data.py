import streamlit as st
from utils import add_s2d2_footer
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Raw Data")

st.header("API Data Display with Filtering")

# Sidebar filters
st.sidebar.header("Data Filters")

if st.session_state.model_results:
    results = st.session_state.model_results

    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Solar PV", "Wind", "Hybrid"],
        help="Select which renewable energy data to display"
    )

    # Time range filter
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["All Data", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=0,
        help="Filter data by time range"
    )

    # Display options
    show_chart = st.sidebar.checkbox("Show Chart", value=True)
    show_table = st.sidebar.checkbox("Show Data Table", value=True)
    export_data = st.sidebar.checkbox("Enable Export", value=False)

    # Get the API data from session state
    api_data = results.get('api_responses', {})

    if not api_data:
        st.warning("No API data available. Please run a calculation first to generate data.")
        # Add S2D2 Lab footer
        add_s2d2_footer()
        st.stop()
    else:
        # Process and filter data based on selection
        if data_source in ["Solar PV", "Hybrid"]:
            solar_key = f"solar_{results.get('inputs_summary', {}).get('location', 'default').replace(' ', '_').lower()}"
            solar_data = api_data.get(solar_key, {})

            if solar_data:
                st.subheader("📊 Solar PV Raw Data")

                # Extract the data
                solar_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(solar_data.get('metadata', {}).get('timestamps', [])),
                    'capacity_factor': solar_data.get('data', {}).get('capacity_factor', []),
                    'irradiance': solar_data.get('data', {}).get('irradiance', []),
                    'temperature': solar_data.get('data', {}).get('temperature', [])
                }).dropna()

                # Apply time filtering
                if time_range != "All Data":
                    end_date = solar_df['timestamp'].max()
                    if time_range == "Last 24 Hours":
                        start_date = end_date - timedelta(hours=24)
                    elif time_range == "Last 7 Days":
                        start_date = end_date - timedelta(days=7)
                    else:  # Last 30 Days
                        start_date = end_date - timedelta(days=30)

                    solar_df = solar_df[solar_df['timestamp'] >= start_date]

                # Display chart
                if show_chart and not solar_df.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Capacity Factor Over Time")
                        fig_cap = go.Figure()
                        fig_cap.add_trace(go.Scatter(
                            x=solar_df['timestamp'],
                            y=solar_df['capacity_factor'],
                            mode='lines',
                            name='Capacity Factor',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                        fig_cap.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Capacity Factor",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_cap, use_container_width=True)

                    with col2:
                        st.subheader("Irradiance vs Capacity Factor")
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=solar_df['irradiance'],
                            y=solar_df['capacity_factor'],
                            mode='markers',
                            name='Hourly Data',
                            marker=dict(
                                size=4,
                                color=solar_df['temperature'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Temperature (°C)")
                            )
                        ))
                        fig_scatter.update_layout(
                            xaxis_title="Irradiance (W/m²)",
                            yaxis_title="Capacity Factor",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Display data table
                if show_table and not solar_df.empty:
                    st.subheader("Raw Solar Data Table")

                    # Format the table
                    display_df = solar_df.copy()
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    display_df['capacity_factor'] = display_df['capacity_factor'].round(4)
                    display_df['irradiance'] = display_df['irradiance'].round(1)
                    display_df['temperature'] = display_df['temperature'].round(1)

                    st.dataframe(display_df, use_container_width=True)

                    # Data statistics
                    with st.expander("Data Statistics"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Hours", len(solar_df))
                        with col2:
                            st.metric("Avg Capacity Factor", f"{solar_df['capacity_factor'].mean():.3f}")
                        with col3:
                            st.metric("Avg Irradiance", f"{solar_df['irradiance'].mean():.0f} W/m²")

                # Export functionality
                if export_data and not solar_df.empty:
                    st.subheader("Export Data")
                    csv = solar_df.to_csv(index=False)
                    st.download_button(
                        label="Download Solar Data as CSV",
                        data=csv,
                        file_name="solar_raw_data.csv",
                        mime="text/csv",
                        key="download_solar_csv"
                    )
            else:
                st.info("No solar data available for the selected location.")

        if data_source in ["Wind", "Hybrid"]:
            if data_source == "Hybrid":
                st.markdown("---")

            wind_key = f"wind_{results.get('inputs_summary', {}).get('location', 'default').replace(' ', '_').lower()}"
            wind_data = api_data.get(wind_key, {})

            if wind_data:
                st.subheader("💨 Wind Raw Data")

                # Extract the data
                wind_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(wind_data.get('metadata', {}).get('timestamps', [])),
                    'capacity_factor': wind_data.get('data', {}).get('capacity_factor', []),
                    'wind_speed': wind_data.get('data', {}).get('wind_speed', []),
                    'temperature': wind_data.get('data', {}).get('temperature', [])
                }).dropna()

                # Apply time filtering
                if time_range != "All Data":
                    end_date = wind_df['timestamp'].max()
                    if time_range == "Last 24 Hours":
                        start_date = end_date - timedelta(hours=24)
                    elif time_range == "Last 7 Days":
                        start_date = end_date - timedelta(days=7)
                    else:  # Last 30 Days
                        start_date = end_date - timedelta(days=30)

                    wind_df = wind_df[wind_df['timestamp'] >= start_date]

                # Display chart
                if show_chart and not wind_df.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Wind Capacity Factor Over Time")
                        fig_cap = go.Figure()
                        fig_cap.add_trace(go.Scatter(
                            x=wind_df['timestamp'],
                            y=wind_df['capacity_factor'],
                            mode='lines',
                            name='Capacity Factor',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        fig_cap.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Capacity Factor",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_cap, use_container_width=True)

                    with col2:
                        st.subheader("Wind Speed vs Capacity Factor")
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=wind_df['wind_speed'],
                            y=wind_df['capacity_factor'],
                            mode='markers',
                            name='Hourly Data',
                            marker=dict(
                                size=4,
                                color=wind_df['temperature'],
                                colorscale='Plasma',
                                showscale=True,
                                colorbar=dict(title="Temperature (°C)")
                            )
                        ))
                        fig_scatter.update_layout(
                            xaxis_title="Wind Speed (m/s)",
                            yaxis_title="Capacity Factor",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Display data table
                if show_table and not wind_df.empty:
                    st.subheader("Raw Wind Data Table")

                    # Format the table
                    display_df = wind_df.copy()
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    display_df['capacity_factor'] = display_df['capacity_factor'].round(4)
                    display_df['wind_speed'] = display_df['wind_speed'].round(1)
                    display_df['temperature'] = display_df['temperature'].round(1)

                    st.dataframe(display_df, use_container_width=True)

                    # Data statistics
                    with st.expander("Data Statistics"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Hours", len(wind_df))
                        with col2:
                            st.metric("Avg Capacity Factor", f"{wind_df['capacity_factor'].mean():.3f}")
                        with col3:
                            st.metric("Avg Wind Speed", f"{wind_df['wind_speed'].mean():.1f} m/s")

                # Export functionality
                if export_data and not wind_df.empty:
                    st.subheader("Export Data")
                    csv = wind_df.to_csv(index=False)
                    st.download_button(
                        label="Download Wind Data as CSV",
                        data=csv,
                        file_name="wind_raw_data.csv",
                        mime="text/csv",
                        key="download_wind_csv"
                    )
            else:
                st.info("No wind data available for the selected location.")

        # Data quality indicators
        st.markdown("---")
        st.subheader("📈 Data Quality Indicators")

        # Show data source information
        location = results.get('inputs_summary', {}).get('location', 'Unknown')
        st.write(f"**Location:** {location}")
        st.write(f"**Data Source:** Renewables Ninja API")
        st.write(f"**Total API Calls:** {len(api_data)}")

        # Cache information if available
        if 'cache_info' in results:
            cache_info = results['cache_info']
            with st.expander("Cache Information"):
                st.write(f"Cache Hits: {cache_info.get('hits', 0)}")
                st.write(f"Cache Misses: {cache_info.get('misses', 0)}")
                st.write(f"Total Requests: {cache_info.get('total', 0)}")

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

# Add S2D2 Lab footer
add_s2d2_footer()
