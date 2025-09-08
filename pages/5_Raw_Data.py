import streamlit as st
from utils import add_s2d2_footer
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO

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

    # Export options - initialize defaults
    enable_export = False
    export_format = "CSV"
    export_with_quality = False
    export_timestamp = True

    with st.sidebar.expander("Export Options"):
        enable_export = st.checkbox("Enable Export Functionality", value=False)
        if enable_export:
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel"],
                help="Choose the export file format"
            )
            export_with_quality = st.checkbox(
                "Include Quality Metrics",
                help="Add data quality information to the export"
            )
            export_timestamp = st.checkbox(
                "Include Timestamp",
                value=True,
                help="Add export timestamp to filename"
            )

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

                # Enhanced Export functionality
                if enable_export and not solar_df.empty:
                    st.subheader("📥 Export Solar Data")

                    # Generate filename with timestamp if requested
                    timestamp = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if export_timestamp else ""
                    base_name = "solar_raw_data"

                    # Prepare data for export
                    export_df = solar_df.copy()
                    if export_with_quality:
                        # Add quality metrics to the data
                        export_df['data_quality_score'] = len(export_df.dropna()) / len(export_df) * 100
                        export_df['has_outliers'] = False  # Would need outlier detection logic

                    if export_format == "CSV":
                        export_data = export_df.to_csv(index=False)
                        mime_type = "text/csv"
                        file_extension = ".csv"
                    else:  # Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            export_df.to_excel(writer, sheet_name='Solar_Data', index=False)
                            if export_with_quality:
                                # Add quality metrics sheet
                                quality_df = pd.DataFrame({
                                    'Metric': ['Completeness', 'Total Records', 'Avg Capacity Factor'],
                                    'Value': [
                                        len(export_df.dropna()) / len(export_df) * 100,
                                        len(export_df),
                                        export_df['capacity_factor'].mean()
                                    ]
                                })
                                quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
                        export_data = output.getvalue()
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_extension = ".xlsx"

                    filename = f"{base_name}{timestamp}{file_extension}"

                    st.download_button(
                        label=f"Download Solar Data ({export_format})",
                        data=export_data,
                        file_name=filename,
                        mime=mime_type,
                        key="download_solar_data"
                    )

                    # Export statistics
                    with st.expander("Export Summary"):
                        st.write(f"📊 **Export Details:**")
                        st.write(f"- Records: {len(export_df):,}")
                        st.write(f"- Date Range: {export_df['timestamp'].min()} to {export_df['timestamp'].max()}")
                        st.write(f"- Avg Capacity Factor: {export_df['capacity_factor'].mean():.3f}")
                        if 'irradiance' in export_df.columns:
                            st.write(f"- Avg Irradiance: {export_df['irradiance'].mean():.0f} W/m²")
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

                # Enhanced Export functionality
                if enable_export and not wind_df.empty:
                    st.subheader("📥 Export Wind Data")

                    # Generate filename with timestamp if requested
                    timestamp = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if export_timestamp else ""
                    base_name = "wind_raw_data"

                    # Prepare data for export
                    export_df = wind_df.copy()
                    if export_with_quality:
                        # Add quality metrics to the data
                        export_df['data_quality_score'] = len(export_df.dropna()) / len(export_df) * 100
                        export_df['has_outliers'] = False  # Would need outlier detection logic

                    if export_format == "CSV":
                        export_data = export_df.to_csv(index=False)
                        mime_type = "text/csv"
                        file_extension = ".csv"
                    else:  # Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            export_df.to_excel(writer, sheet_name='Wind_Data', index=False)
                            if export_with_quality:
                                # Add quality metrics sheet
                                quality_df = pd.DataFrame({
                                    'Metric': ['Completeness', 'Total Records', 'Avg Capacity Factor'],
                                    'Value': [
                                        len(export_df.dropna()) / len(export_df) * 100,
                                        len(export_df),
                                        export_df['capacity_factor'].mean()
                                    ]
                                })
                                quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
                        export_data = output.getvalue()
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_extension = ".xlsx"

                    filename = f"{base_name}{timestamp}{file_extension}"

                    st.download_button(
                        label=f"Download Wind Data ({export_format})",
                        data=export_data,
                        file_name=filename,
                        mime=mime_type,
                        key="download_wind_data"
                    )

                    # Export statistics
                    with st.expander("Export Summary"):
                        st.write(f"📊 **Export Details:**")
                        st.write(f"- Records: {len(export_df):,}")
                        st.write(f"- Date Range: {export_df['timestamp'].min()} to {export_df['timestamp'].max()}")
                        st.write(f"- Avg Capacity Factor: {export_df['capacity_factor'].mean():.3f}")
                        if 'wind_speed' in export_df.columns:
                            st.write(f"- Avg Wind Speed: {export_df['wind_speed'].mean():.1f} m/s")
            else:
                st.info("No wind data available for the selected location.")
    
            # Bulk Export Option
            if enable_export and len(api_data) > 1:
                st.markdown("---")
                st.subheader("📤 Bulk Export All Data")
    
                if st.button("Export All Data Sources"):
                    # Prepare bulk export
                    export_data_sources = {}
    
                    for key, data_dict in api_data.items():
                        if isinstance(data_dict, dict) and 'data' in data_dict:
                            data = data_dict['data']
                            if isinstance(data, dict):
                                data_type = 'Solar' if 'irradiance' in data else 'Wind'
                                location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()
    
                                df = pd.DataFrame({
                                    'timestamp': pd.to_datetime(data.get('metadata', {}).get('timestamps', [])),
                                    'capacity_factor': data.get('capacity_factor', []),
                                    'data_type': [data_type] * len(data.get('capacity_factor', [])),
                                    'location': [location_name] * len(data.get('capacity_factor', []))
                                }).dropna()
    
                                if data_type == 'Solar' and 'irradiance' in data:
                                    df['irradiance'] = data.get('irradiance', [])[:len(df)]
                                elif data_type == 'Wind' and 'wind_speed' in data:
                                    df['wind_speed'] = data.get('wind_speed', [])[:len(df)]
    
                                export_data_sources[f"{data_type}_{location_name}"] = df
    
                    # Create Excel file with multiple sheets
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Summary sheet
                        summary_df = pd.DataFrame({
                            'Data Source': list(export_data_sources.keys()),
                            'Records': [len(df) for df in export_data_sources.values()],
                            'Date Range Start': [df['timestamp'].min() for df in export_data_sources.values()],
                            'Date Range End': [df['timestamp'].max() for df in export_data_sources.values()],
                            'Avg Capacity Factor': [df['capacity_factor'].mean() for df in export_data_sources.values()]
                        })
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
                        # Individual data sheets
                        for sheet_name, df in export_data_sources.items():
                            # Excel sheet names have length limits and can't contain special chars
                            safe_sheet_name = sheet_name.replace(' ', '_')[:30]
                            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    
                    bulk_filename = f"hydrogen_raw_data_bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
                    st.download_button(
                        label="Download Bulk Data (Excel)",
                        data=output.getvalue(),
                        file_name=bulk_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_bulk_data"
                    )
    
                    st.success(f"✅ Bulk export ready! File contains {len(export_data_sources)} data sources with {sum(len(df) for df in export_data_sources.values()):,} total records.")
    
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

        # Data Validation Reports Section
        st.markdown("---")
        st.header("📋 Data Validation Reports")

        validation_tabs = st.tabs(["Integrity Checks", "Time Series", "Physical Constraints", "Statistical", "Summary Report"])

        # Initialize variables for validation tracking
        integrity_issues = []
        time_series_issues = []
        constraint_violations = []
        statistical_report = []

        # Integrity Checks Tab
        with validation_tabs[0]:
            st.subheader("Data Integrity Validation")

            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict):
                        data_type = 'Solar' if 'irradiance' in data else 'Wind'
                        location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                        # Check for null/invalid values
                        for field in ['capacity_factor', 'irradiance', 'wind_speed', 'temperature']:
                            if field in data and data[field]:
                                values = data[field]
                                total_values = len(values)
                                null_count = sum(1 for v in values if v is None or pd.isna(v))
                                if null_count > 0:
                                    integrity_issues.append({
                                        'Validation Type': 'Missing Values',
                                        'Data Type': data_type,
                                        'Location': location_name,
                                        'Field': field.replace('_', ' ').title(),
                                        'Issues Found': null_count,
                                        'Total Records': total_values,
                                        'Severity': 'Medium' if null_count/total_values > 0.1 else 'Low'
                                    })

                                # Check for data type consistency
                                non_null_values = [v for v in values if v is not None and not pd.isna(v)]
                                if non_null_values:
                                    # Check if all values are numeric for numeric fields
                                    numeric_fields = ['capacity_factor', 'irradiance', 'wind_speed', 'temperature']
                                    if field in numeric_fields:
                                        try:
                                            numeric_check = [isinstance(v, (int, float)) or pd.api.types.is_number(v) for v in non_null_values]
                                            invalid_types = sum(not x for x in numeric_check)
                                            if invalid_types > 0:
                                                integrity_issues.append({
                                                    'Validation Type': 'Data Type',
                                                    'Data Type': data_type,
                                                    'Location': location_name,
                                                    'Field': field.replace('_', ' ').title(),
                                                    'Issues Found': invalid_types,
                                                    'Total Records': total_values,
                                                    'Severity': 'High'
                                                })
                                        except:
                                            integrity_issues.append({
                                                'Validation Type': 'Data Type Error',
                                                'Data Type': data_type,
                                                'Location': location_name,
                                                'Field': field.replace('_', ' ').title(),
                                                'Issues Found': 'Type check failed',
                                                'Total Records': total_values,
                                                'Severity': 'High'
                                            })

            if integrity_issues:
                integrity_df = pd.DataFrame(integrity_issues)
                st.dataframe(integrity_df.style.format({
                    'Issues Found': '{:,}'
                }), use_container_width=True)

                # Show severity breakdown
                severity_counts = integrity_df['Severity'].value_counts()
                colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                fig_severity = go.Figure(data=[go.Pie(
                    labels=severity_counts.index,
                    values=severity_counts.values,
                    marker_colors=[colors.get(sev, 'gray') for sev in severity_counts.index]
                )])
                fig_severity.update_layout(title="Integrity Issues by Severity")
                st.plotly_chart(fig_severity, use_container_width=True)
            else:
                st.success("✅ All data integrity checks passed!")

        # Time Series Tab
        with validation_tabs[1]:
            st.subheader("Time Series Continuity Validation")

            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'metadata' in data_dict and 'timestamps' in data_dict['metadata']:
                    timestamps = data_dict['metadata']['timestamps']
                    data_type = 'Solar' if 'irradiance' in data_dict.get('data', {}) else 'Wind'
                    location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                    if timestamps:
                        try:
                            # Convert to datetime for analysis
                            dt_series = pd.to_datetime(timestamps)
                            expected_interval = pd.Timedelta(hours=1)  # Assuming hourly data

                            # Check for duplicate timestamps
                            duplicates = dt_series.duplicated().sum()
                            if duplicates > 0:
                                time_series_issues.append({
                                    'Validation Type': 'Duplicate Timestamps',
                                    'Data Type': data_type,
                                    'Location': location_name,
                                    'Issues Found': duplicates,
                                    'Expected Interval': '1 hour',
                                    'Severity': 'High'
                                })

                            # Check for time gaps (missing hours)
                            if len(dt_series) > 1:
                                sorted_times = sorted(dt_series)
                                gaps = 0
                                for i in range(1, len(sorted_times)):
                                    if sorted_times[i] - sorted_times[i-1] > expected_interval:
                                        gaps += 1

                                if gaps > 0:
                                    time_series_issues.append({
                                        'Validation Type': 'Missing Time Periods',
                                        'Data Type': data_type,
                                        'Location': location_name,
                                        'Issues Found': gaps,
                                        'Expected Interval': '1 hour',
                                        'Severity': 'Medium'
                                    })

                            # Check for out-of-order timestamps
                            if not dt_series.is_monotonic_increasing:
                                time_series_issues.append({
                                    'Validation Type': 'Non-monotonic Timestamps',
                                    'Data Type': data_type,
                                    'Location': location_name,
                                    'Issues Found': 'Out of order',
                                    'Expected Interval': '1 hour',
                                    'Severity': 'High'
                                })

                        except Exception as e:
                            time_series_issues.append({
                                'Validation Type': 'Time Parsing Error',
                                'Data Type': data_type,
                                'Location': location_name,
                                'Issues Found': str(e),
                                'Expected Interval': '1 hour',
                                'Severity': 'High'
                            })

            if time_series_issues:
                ts_df = pd.DataFrame(time_series_issues)
                st.dataframe(ts_df, use_container_width=True)

                # Time series validation summary
                issue_types = ts_df['Validation Type'].value_counts()
                fig_ts = go.Figure(data=[go.Bar(
                    x=issue_types.index,
                    y=issue_types.values,
                    marker_color='lightcoral'
                )])
                fig_ts.update_layout(
                    title="Time Series Validation Issues",
                    xaxis_title="Issue Type",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.success("✅ All time series continuity checks passed!")

        # Physical Constraints Tab
        with validation_tabs[2]:
            st.subheader("Physical Constraints Validation")

            # Define physical constraints
            constraints = {
                'capacity_factor': {'min': 0.0, 'max': 1.5, 'unit': 'fraction'},  # Allow slight over 1.0
                'irradiance': {'min': 0.0, 'max': 1500.0, 'unit': 'W/m²'},  # Maximum solar irradiance
                'wind_speed': {'min': 0.0, 'max': 50.0, 'unit': 'm/s'},  # Maximum reasonable wind speed
                'temperature': {'min': -50.0, 'max': 60.0, 'unit': '°C'}  # Reasonable temperature range
            }

            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict):
                        data_type = 'Solar' if 'irradiance' in data else 'Wind'
                        location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                        for field, limits in constraints.items():
                            if field in data and data[field]:
                                values = data[field]
                                non_null_values = [v for v in values if v is not None and not pd.isna(v)]

                                if non_null_values:
                                    # Check range violations
                                    min_violations = sum(1 for v in non_null_values if v < limits['min'])
                                    max_violations = sum(1 for v in non_null_values if v > limits['max'])

                                    total_violations = min_violations + max_violations

                                    if total_violations > 0:
                                        constraint_violations.append({
                                            'Data Type': data_type,
                                            'Location': location_name,
                                            'Field': field.replace('_', ' ').title(),
                                            'Min Range': f"{limits['min']} {limits['unit']}",
                                            'Max Range': f"{limits['max']} {limits['unit']}",
                                            'Below Min': min_violations,
                                            'Above Max': max_violations,
                                            'Total Violations': total_violations,
                                            'Severity': 'High' if total_violations > len(non_null_values) * 0.05 else 'Medium'
                                        })

            if constraint_violations:
                const_df = pd.DataFrame(constraint_violations)
                st.dataframe(const_df, use_container_width=True)

                # Visualization of violations
                fig_const = go.Figure()
                for idx, row in const_df.iterrows():
                    fig_const.add_trace(go.Bar(
                        name=f"{row['Data Type']} - {row['Location']} - {row['Field']}",
                        x=['Below Min', 'Above Max'],
                        y=[row['Below Min'], row['Above Max']],
                        text=[row['Below Min'], row['Above Max']],
                        textposition='auto'
                    ))

                fig_const.update_layout(
                    title="Physical Constraint Violations",
                    xaxis_title="Violation Type",
                    yaxis_title="Count",
                    barmode='stack',
                    height=400
                )
                st.plotly_chart(fig_const, use_container_width=True)
            else:
                st.success("✅ All physical constraint checks passed!")

        # Statistical Tab
        with validation_tabs[3]:
            st.subheader("Statistical Data Validation")

            for key, data_dict in api_data.items():
                if isinstance(data_dict, dict) and 'data' in data_dict:
                    data = data_dict['data']
                    if isinstance(data, dict) and 'capacity_factor' in data:
                        data_type = 'Solar' if 'irradiance' in data else 'Wind'
                        location_name = key.replace('solar_', '').replace('wind_', '').replace('_', ' ').title()

                        cf_values = [x for x in data['capacity_factor'] if x is not None and not pd.isna(x)]

                        if cf_values:
                            series = pd.Series(cf_values)

                            # Basic statistical tests
                            mean_val = series.mean()
                            std_val = series.std()
                            min_val = series.min()
                            max_val = series.max()
                            median_val = series.median()

                            # Check for unrealistic statistical properties
                            issues = []

                            # Check coefficient of variation
                            cv = std_val / mean_val if mean_val != 0 else 0
                            if cv > 2:  # Very high variation for capacity factors
                                issues.append("High coefficient of variation")

                            # Check for suspicious minimum values
                            if min_val < -0.1:  # Negative capacity factor
                                issues.append("Negative capacity factors found")

                            if max_val > 1.5:  # Unrealistically high capacity factor
                                issues.append("Excessively high capacity factors")

                            statistical_report.append({
                                'Data Type': data_type,
                                'Location': location_name,
                                'Count': len(cf_values),
                                'Mean': mean_val,
                                'Std Dev': std_val,
                                'Min': min_val,
                                'Max': max_val,
                                'Median': median_val,
                                'Coefficient of Variation': cv,
                                'Issues Detected': '; '.join(issues) if issues else 'None'
                            })

            if statistical_report:
                stat_df = pd.DataFrame(statistical_report)
                st.dataframe(stat_df.style.format({
                    'Mean': '{:.4f}',
                    'Std Dev': '{:.4f}',
                    'Min': '{:.4f}',
                    'Max': '{:.4f}',
                    'Median': '{:.4f}',
                    'Coefficient of Variation': '{:.3f}'
                }), use_container_width=True)

                # Statistical summary visualization
                if len(stat_df) > 0:
                    fig_stat = go.Figure()

                    # Mean capacity factors by location
                    fig_stat.add_trace(go.Bar(
                        name='Mean Capacity Factor',
                        x=stat_df['Location'],
                        y=stat_df['Mean'],
                        marker_color='lightblue'
                    ))

                    # Standard deviation
                    fig_stat.add_trace(go.Bar(
                        name='Std Deviation',
                        x=stat_df['Location'],
                        y=stat_df['Std Dev'],
                        marker_color='orange'
                    ))

                    fig_stat.update_layout(
                        title="Statistical Summary by Location",
                        xaxis_title="Location",
                        yaxis_title="Value",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig_stat, use_container_width=True)

            # Export statistical report
            if statistical_report and enable_export and st.button("Export Statistical Report", key="stat_report"):
                stat_df = pd.DataFrame(statistical_report)
                csv = stat_df.to_csv(index=False)
                st.download_button(
                    label="Download Statistical Report (CSV)",
                    data=csv,
                    file_name=f"statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_stat_report"
                )

        # Summary Report Tab
        with validation_tabs[4]:
            st.subheader("Data Validation Summary Report")

            # Create summary table
            summary_df = pd.DataFrame({
                'Validation Category': ['Integrity Checks', 'Time Series Checks', 'Physical Constraints', 'Statistical Issues'],
                'Issues Found': [len(integrity_issues), len(time_series_issues), len(constraint_violations), len(statistical_report)],
                'Status': [
                    '❌ Issues' if len(integrity_issues) > 0 else '✅ Passed',
                    '❌ Issues' if len(time_series_issues) > 0 else '✅ Passed',
                    '❌ Issues' if len(constraint_violations) > 0 else '✅ Passed',
                    '❌ Issues' if len(statistical_report) > 0 else '✅ Passed'
                ]
            })

            st.dataframe(summary_df, use_container_width=True)

            # Overall assessment
            total_issues = len(integrity_issues) + len(time_series_issues) + len(constraint_violations) + len(statistical_report)
            if total_issues == 0:
                st.success("🎉 **Excellent! All data validation checks passed.** The dataset appears to be clean and reliable.")
            elif total_issues < 5:
                st.warning("⚠️ **Minor Issues Detected:** Found a few validation concerns that should be reviewed but don't significantly impact data quality.")
            else:
                st.error("⚠️ **Multiple Issues Found:** Several data validation concerns require attention. Consider reviewing and cleaning the dataset.")

            # Recommendations
            st.subheader("Recommendations")

            if integrity_issues:
                with st.expander("Data Integrity Recommendations"):
                    st.write("• Consider imputing or removing missing values based on your analysis needs")
                    st.write("• Implement data type validation before processing")
                    st.write("• Review data collection processes to reduce null values")

            if time_series_issues:
                with st.expander("Time Series Recommendations"):
                    st.write("• Check data collection timing and alignment")
                    st.write("• Implement duplicate removal procedures")
                    st.write("• Consider gap-filling techniques for missing timestamps")

            if constraint_violations:
                with st.expander("Physical Constraints Recommendations"):
                    st.write("• Review out-of-range values for potential data entry errors")
                    st.write("• Validate sensor calibration and measurement accuracy")
                    st.write("• Implement range checks in data collection systems")

            if statistical_report and any('Issues Detected' in r and r['Issues Detected'] != 'None' for r in statistical_report):
                with st.expander("Statistical Recommendations"):
                    st.write("• Investigate unusually high or low values")
                    st.write("• Consider statistical outlier treatment methods")
                    st.write("• Review data distribution for expected patterns")

            # Export validation summary
            if enable_export and st.button("Export Validation Summary", key="validation_summary"):
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Validation Summary (CSV)",
                    data=summary_csv,
                    file_name=f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_validation_summary"
                )

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

# Add S2D2 Lab footer
add_s2d2_footer()
