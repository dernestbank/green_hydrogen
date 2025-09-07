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

else:
    st.info("Please go to the 'Inputs' page and run the calculation first.")

# Add S2D2 Lab footer
add_s2d2_footer()
