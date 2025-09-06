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

        # Enhanced data quality indicators
        st.markdown("---")
        st.subheader("📈 Data Quality Indicators")

        # Show data source information
        location = results.get('inputs_summary', {}).get('location', 'Unknown')
        st.write(f"**Location:** {location}")
        st.write(f"**Data Source:** Renewables Ninja API")
        st.write(f"**Total API Calls:** {len(api_data)}")

        # Data Quality Tabs
        quality_tabs = st.tabs(["Overview", "Completeness", "Outliers", "Consistency", "Performance"])

        # Overview Tab
        with quality_tabs[0]:
            col1, col2, col3 = st.columns(3)

            total_records = 0
            complete_records = 0

            # Calculate overall data quality
            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict) and 'capacity_factor' in data:
                        total_records += len(data['capacity_factor'])
                        complete_records += sum(1 for cf in data['capacity_factor'] if cf is not None and not pd.isna(cf))

            data_coverage = complete_records / total_records if total_records > 0 else 0

            with col1:
                st.metric("Total Records", f"{total_records:,}")
            with col2:
                st.metric("Complete Records", f"{complete_records:,}")
            with col3:
                st.metric("Data Coverage", f"{data_coverage:.1%}")

            # Quality score
            quality_score = data_coverage * 100
            if quality_score >= 95:
                st.success(f"✅ **Overall Quality Score: {quality_score:.1f}%** - Excellent data completeness")
            elif quality_score >= 80:
                st.warning(f"⚠️ **Overall Quality Score: {quality_score:.1f}%** - Good data completeness")
            else:
                st.error(f"❌ **Overall Quality Score: {quality_score:.1f}%** - Data completeness needs attention")

        # Completeness Tab
        with quality_tabs[1]:
            st.subheader("Data Completeness Analysis")

            completeness_data = []
            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict):
                        data_type = 'Solar' if 'irradiance' in data else 'Wind'
                        location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                        for field in ['capacity_factor', 'irradiance', 'wind_speed', 'temperature']:
                            if field in data and data[field]:
                                total = len(data[field])
                                non_null = sum(1 for val in data[field] if val is not None and not pd.isna(val))
                                completeness = non_null / total if total > 0 else 0
                                completeness_data.append({
                                    'Data Type': data_type,
                                    'Location': location_name,
                                    'Field': field.replace('_', ' ').title(),
                                    'Completeness': completeness,
                                    'Missing': total - non_null
                                })

            if completeness_data:
                comp_df = pd.DataFrame(completeness_data)
                st.dataframe(comp_df.style.format({
                    'Completeness': '{:.1%}',
                    'Missing': '{:,}'
                }), use_container_width=True)

                # Completeness visualization
                fig_comp = go.Figure()
                for data_type in comp_df['Data Type'].unique():
                    type_data = comp_df[comp_df['Data Type'] == data_type]
                    fig_comp.add_trace(go.Bar(
                        name=data_type,
                        x=type_data['Field'],
                        y=type_data['Completeness'],
                        text=[f"{x:.1%}" for x in type_data['Completeness']],
                        textposition='auto'
                    ))

                fig_comp.update_layout(
                    title="Data Completeness by Field",
                    xaxis_title="Field",
                    yaxis_title="Completeness (%)",
                    height=400
                )
                st.plotly_chart(fig_comp, use_container_width=True)

        # Outliers Tab
        with quality_tabs[2]:
            st.subheader("Outlier Detection")

            outlier_data = []
            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict) and 'capacity_factor' in data:
                        data_type = 'Solar' if 'irradiance' in data else 'Wind'
                        location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                        cf_values = [x for x in data['capacity_factor'] if x is not None and not pd.isna(x)]
                        if cf_values:
                            q1, q3 = pd.Series(cf_values).quantile([0.25, 0.75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr

                            outliers = sum(1 for x in cf_values if x < lower_bound or x > upper_bound)
                            outlier_pct = outliers / len(cf_values)

                            outlier_data.append({
                                'Data Type': data_type,
                                'Location': location_name,
                                'Total Values': len(cf_values),
                                'Outliers': outliers,
                                'Outlier Rate': outlier_pct,
                                'Range': f"[{lower_bound:.3f}, {upper_bound:.3f}]"
                            })

            if outlier_data:
                outlier_df = pd.DataFrame(outlier_data)
                st.dataframe(outlier_df.style.format({
                    'Outlier Rate': '{:.1%}'
                }), use_container_width=True)

                # Outlier visualization
                fig_outlier = go.Figure()
                for idx, row in outlier_df.iterrows():
                    fig_outlier.add_trace(go.Bar(
                        name=f"{row['Data Type']} - {row['Location']}",
                        x=['Total Values', 'Outliers'],
                        y=[row['Total Values'], row['Outliers']],
                        text=[f"{row['Total Values']:,}", f"{row['Outliers']:,}"],
                        textposition='auto'
                    ))

                fig_outlier.update_layout(
                    title="Outlier Analysis",
                    xaxis_title="Metric",
                    yaxis_title="Count",
                    height=400,
                    barmode='group'
                )
                st.plotly_chart(fig_outlier, use_container_width=True)
            else:
                st.info("No data available for outlier analysis")

        # Consistency Tab
        with quality_tabs[3]:
            st.subheader("Data Consistency Checks")

            consistency_issues = []
            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict):
                        data_type = 'Solar' if 'irradiance' in data else 'Wind'
                        location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                        if data_type == 'Solar' and 'irradiance' in data and 'capacity_factor' in data:
                            irradiance = [x for x in data['irradiance'] if x is not None and not pd.isna(x)]
                            cap_factor = [x for x in data['capacity_factor'] if x is not None and not pd.isna(x)]
                            if len(irradiance) > 10:
                                corr = pd.Series(irradiance).corr(pd.Series(cap_factor[:len(irradiance)]))
                                if corr < 0.5:
                                    consistency_issues.append({
                                        'Check': 'Solar: Irradiance vs Capacity Factor',
                                        'Location': location_name,
                                        'Correlation': corr,
                                        'Status': 'Low Correlation'
                                    })

            if consistency_issues:
                cons_df = pd.DataFrame(consistency_issues)
                st.dataframe(cons_df.style.format({
                    'Correlation': '{:.3f}'
                }), use_container_width=True)
            else:
                st.success("✅ No data consistency issues detected")

        # Performance Tab
        with quality_tabs[4]:
            st.subheader("Performance Metrics")

            # Cache information
            if 'cache_info' in results:
                cache_info = results['cache_info']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cache Hits", f"{cache_info.get('hits', 0):,}")
                with col2:
                    st.metric("Cache Misses", f"{cache_info.get('misses', 0):,}")
                with col3:
                    hit_rate = cache_info.get('hits', 0) / max(cache_info.get('total', 1), 1) * 100
                    st.metric("Hit Rate", f"{hit_rate:.1f}%")

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

# Add S2D2 Lab footer
add_s2d2_footer()
