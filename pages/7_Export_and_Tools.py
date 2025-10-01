"""
Export & Tools Page for Hydrogen Cost Analysis Tool
Provides comprehensive export, configuration, and comparison features.
"""

import streamlit as st
from utils import add_s2d2_footer
import pandas as pd
from datetime import datetime
import io
from typing import Dict, Any

# Import our new utilities
try:
    from src.utils.pdf_report_generator import create_pdf_report
    PDF_AVAILABLE = True
except ImportError as e:
    PDF_AVAILABLE = False
    pdf_error_msg = str(e)

from src.utils.data_export_manager import create_complete_data_export, DataExportManager
from src.utils.configuration_manager import create_configuration_manager, ConfigurationManager
from src.utils.results_comparison_tool import create_comparison_tool, ResultsComparisonTool

st.set_page_config(layout="wide")
st.title("Hydrogen Cost Analysis Tool - Export & Tools")

st.header("Comprehensive Export and Configuration Management")

if not st.session_state.model_results:
    st.warning("Please run a model calculation first to use these tools.")
    add_s2d2_footer()
    st.stop()

results_data = st.session_state.model_results

# Create tool instances
pdf_generator = None
export_manager = DataExportManager()
config_manager = create_configuration_manager()
comparison_tool = create_comparison_tool()

# Main tabs for different tool categories
tool_tabs = st.tabs(["📄 Reports", "💾 Data Export", "⚙️ Configurations", "📊 Comparisons"])

# Reports Tab
with tool_tabs[0]:
    st.subheader("📄 Professional Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PDF Report Generation")

        report_options = st.multiselect(
            "Report Sections to Include",
            ["Executive Summary", "System Configuration", "Technical Analysis",
             "Financial Analysis", "Operational Analysis", "Conclusions"],
            default=["Executive Summary", "System Configuration", "Technical Analysis"]
        )

        report_quality = st.selectbox(
            "Report Quality",
            ["Standard", "Professional", "Enterprise"],
            help="Affects report formatting and detail level",
            key="pdf_report_quality"
        )

        include_charts = st.checkbox("Include Charts", value=True, key="pdf_include_charts")
        include_metadata = st.checkbox("Include Metadata", value=True, key="pdf_include_metadata")

    with col2:
        st.markdown("### Report Preview")

        # Show what will be included
        st.markdown("**Selected Sections:**")
        for section in report_options:
            st.markdown(f"• {section}")

        st.markdown(f"**Report Quality:** {report_quality}")
        st.markdown(f"**Charts Included:** {'Yes' if include_charts else 'No'}")
        st.markdown(f"**Metadata Included:** {'Yes' if include_metadata else 'No'}")

        if st.button("🔄 Preview Report", key="preview_report"):
            st.info("PDF preview feature would display report structure here.")

        if PDF_AVAILABLE:
            if st.button("📄 Generate PDF Report", key="generate_pdf"):
                try:
                    # Generate comprehensive PDF report
                    pdf_bytes = create_pdf_report(results_data)

                    # Create download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"hydrogen_cost_analysis_report_{timestamp}.pdf"

                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        key="download_pdf_report"
                    )

                    st.success("✅ PDF report generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating PDF report: {str(e)}")
        else:
            st.warning(f"📄 PDF generation not available: {pdf_error_msg}")
            st.info("Install reportlab to enable PDF reports: `pip install reportlab>=4.0.0`")

# Data Export Tab
with tool_tabs[1]:
    st.subheader("💾 Comprehensive Data Export")

    export_format = st.selectbox(
        "Export Format",
        ["Excel (Complete)", "CSV Files (ZIP)", "JSON", "All Formats (ZIP)"],
        help="Choose export format for your data",
        key="data_export_format"
    )

    export_scope = st.selectbox(
        "Export Scope",
        ["All Data", "Filtered Data", "Charts Only", "Tables Only"],
        help="What data to include in export",
        key="data_export_scope"
    )

    export_options = st.expander("Advanced Export Options")

    with export_options:
        include_metadata = st.checkbox("Include Metadata", value=True, key="export_include_metadata")
        include_timestamps = st.checkbox("Include Timestamps", value=True, key="export_include_timestamps")
        compress_export = st.checkbox("Compress Export", value=True, key="export_compress")
        add_checksum = st.checkbox("Add Checksum Validation", value=False, key="export_add_checksum")

    # Export preview
    st.markdown("### Export Preview")

    if export_format == "Excel (Complete)":
        st.info("📊 **Excel Export includes:**\n"
                "• All system configuration data\n"
                "• Complete operating results\n"
                "• Financial analysis\n"
                "• Hourly operation data\n"
                "• API raw data sources\n"
                "• Metadata and timestamps")
    else:
        st.info("💾 **Selected export format:** Compact, efficient data export")

    if st.button("🚀 Generate Export", key="generate_export"):
        try:
            # Determine format type for manager
            format_map = {
                "Excel (Complete)": "excel",
                "CSV Files (ZIP)": "csv",
                "JSON": "json",
                "All Formats (ZIP)": "zip"
            }

            format_type = format_map[export_format]

            # Generate export (convert session state to dict)
            export_data = create_complete_data_export(dict(st.session_state), format_type)  # type: ignore

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = "hydrogen_analysis_complete"

            if format_type == "excel":
                filename = f"{base_name}_{timestamp}.xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif format_type == "csv":
                filename = f"{base_name}_{timestamp}.zip"
                mime_type = "application/zip"
            elif format_type == "json":
                filename = f"{base_name}_{timestamp}.json"
                mime_type = "application/json"
            else:  # ZIP
                filename = f"{base_name}_all_formats_{timestamp}.zip"
                mime_type = "application/zip"

            # Download button
            st.download_button(
                label=f"📥 Download {export_format}",
                data=export_data,
                file_name=filename,
                mime=mime_type,
                key="download_export_data"
            )

            st.success(f"✅ Export generated successfully as {filename}!")

        except Exception as e:
            st.error(f"❌ Error generating export: {str(e)}")

# Configurations Tab
with tool_tabs[2]:
    st.subheader("⚙️ Configuration Management")

    config_action = st.selectbox(
        "Configuration Action",
        ["Save Current", "Load Existing", "Use Preset", "Export Config"],
        key="config_action_select"
    )

    if config_action == "Save Current":
        st.markdown("### 💾 Save Current Configuration")

        config_name = st.text_input("Configuration Name", placeholder="e.g., Base Case Scenario")
        config_description = st.text_area("Description (Optional)", height=100,
                                        placeholder="Brief description of this configuration...")

        if st.button("💾 Save Configuration", key="save_config"):
            if config_name.strip():
                try:
                    # Prepare configuration data from session state
                    config_data = {
                        "inputs_summary": results_data.get("inputs_summary", {}),
                        "timestamp": datetime.now().isoformat(),
                        "session_metadata": {
                            "has_results": "operating_outputs" in results_data,
                            "has_financial": "cash_flows" in results_data,
                            "data_sources": len(results_data.get("api_responses", {}))
                        }
                    }

                    # Save configuration
                    saved_path = config_manager.save_configuration(
                        config_data, config_name, config_description
                    )

                    st.success(f"✅ Configuration saved successfully as '{config_name}'!")
                    st.info(f"📁 Saved to: {saved_path}")

                except Exception as e:
                    st.error(f"❌ Error saving configuration: {str(e)}")
            else:
                st.error("❌ Please enter a configuration name")

    elif config_action == "Load Existing":
        st.markdown("### 📂 Load Existing Configuration")

        # List saved configurations
        saved_configs = config_manager.list_configurations()

        if saved_configs:
            config_names = [config["name"] for config in saved_configs]
            selected_config = st.selectbox("Select Configuration", config_names)

            if selected_config:
                config_details = next(c for c in saved_configs if c["name"] == selected_config)
                st.markdown(f"**Description:** {config_details['description']}")
                st.markdown(f"**Created:** {config_details['created_date']}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load Configuration", key="load_config"):
                        try:
                            config_data = config_manager.load_configuration(
                                config_details["file_path"]
                            )
                            st.success(f"✅ Configuration '{selected_config}' loaded!")
                            st.info("⚠️ Note: Configuration loaded to memory. Apply to model as needed.")
                        except Exception as e:
                            st.error(f"❌ Error loading configuration: {str(e)}")

                with col2:
                    if st.button("🗑️ Delete Configuration", key="delete_config"):
                        if config_manager.delete_configuration(config_details["file_path"]):
                            st.success(f"✅ Configuration '{selected_config}' deleted!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete configuration")
        else:
            st.info("📝 No saved configurations found. Save your first configuration to get started!")

    elif config_action == "Use Preset":
        st.markdown("### ⭐ Use Preset Configuration")

        # Get available presets
        presets = config_manager.get_preset_configurations()

        if presets:
            preset_names = [preset["name"] for preset in presets]
            selected_preset = st.selectbox("Select Preset", preset_names)

            if selected_preset:
                preset_details = next(p for p in presets if p["name"] == selected_preset)
                st.markdown(f"**Description:** {preset_details['description']}")

                # Show preset configuration
                preset_config = preset_details["configuration"]

                config_cols = st.columns(3)
                with config_cols[0]:
                    st.metric("Electrolyser", f"{preset_config.get('nominal_electrolyser_capacity', 0)} MW")
                    st.metric("Solar", f"{preset_config.get('nominal_solar_farm_capacity', 0)} MW")
                with config_cols[1]:
                    st.metric("Wind", f"{preset_config.get('nominal_wind_farm_capacity', 0)} MW")
                    st.metric("Battery", f"{preset_config.get('battery_power_rating', 0)} MW")
                with config_cols[2]:
                    st.metric("H₂ Price", f"${preset_config.get('hydrogen_selling_price', 0):.0f}/kg")
                    st.metric("Discount", f"{(preset_config.get('discount_rate', 0) * 100):.1f}%")

                if st.button("✅ Load Preset", key="load_preset"):
                    st.success(f"✅ Preset '{selected_preset}' is ready to apply!")
                    st.info("🔄 Use the 'Inputs' page to apply this configuration to your model.")
        else:
            st.warning("⚠️ No preset configurations available")

    elif config_action == "Export Config":
        st.markdown("### 📤 Export Configuration")

        export_name = st.text_input("Export File Name", placeholder="my_configuration")
        export_format = st.selectbox("Export Format", ["YAML", "JSON"])

        if st.button("📤 Export Configuration", key="export_config"):
            if export_name.strip():
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{export_name}_{timestamp}.{'yaml' if export_format == 'YAML' else 'json'}"

                    # Export current configuration
                    config_data = {
                        "inputs_summary": results_data.get("inputs_summary", {}),
                        "export_timestamp": datetime.now().isoformat(),
                        "tool_version": "Hydrogen Cost Analysis Tool v2.0"
                    }

                    export_path = config_manager.export_configuration(
                        config_data, filename, export_format.lower()
                    )

                    with open(export_path, 'rb') as f:
                        export_bytes = f.read()

                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=export_bytes,
                        file_name=filename,
                        mime="application/yaml" if export_format == "YAML" else "application/json",
                        key="download_config_export"
                    )

                    st.success("✅ Configuration exported successfully!")

                except Exception as e:
                    st.error(f"❌ Error exporting configuration: {str(e)}")
            else:
                st.error("❌ Please enter a valid file name")

# Comparisons Tab
with tool_tabs[3]:
    st.subheader("📊 Results Comparison Tool")

    comparison_subtabs = st.tabs(["Scenario Comparison", "Ranking & Analysis", "Sensitivity Analysis"])

    with comparison_subtabs[0]:
        st.markdown("### 🔍 Scenario Comparison")

        # Check if we have scenarios to compare
        current_scenario_count = len(comparison_tool.list_scenarios())

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Scenarios Available:** {current_scenario_count}")

            # Option to add current scenario
            scenario_name = st.text_input(
                "Scenario Name",
                placeholder="e.g., Base Case Scenario",
                key="comparison_scenario_name"
            )

            if st.button("➕ Add Current Scenario", key="add_current_scenario"):
                if scenario_name.strip():
                    success = comparison_tool.add_scenario(
                        scenario_name.strip(),
                        results_data,
                        {"description": "Current session results", "added_via": "manual"}
                    )

                    if success:
                        st.success(f"✅ Scenario '{scenario_name}' added to comparison!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add scenario")
                else:
                    st.error("❌ Please enter a scenario name")

        with col2:
            st.markdown("### Available Scenarios")
            scenarios_list = comparison_tool.list_scenarios()

            if scenarios_list:
                for scenario in scenarios_list:
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.write(f"• {scenario['name']}")
                    with col_b:
                        if st.button("🗑️", key=f"remove_{scenario['name']}", help="Remove scenario"):
                            # Note: Remove functionality would need to be added to the tool
                            st.info(f"Remove functionality for {scenario['name']}")
            else:
                st.info("No scenarios added yet")

        # Show current scenarios
        if current_scenario_count >= 2:
            st.markdown("---")
            st.markdown("### 📈 Generate Comparison")

            # Set baseline scenario if there are multiple scenarios
            if current_scenario_count > 1:
                baseline_options = list(comparison_tool.scenarios.keys())
                baseline_selection = st.selectbox(
                    "Baseline Scenario (for % calculations)",
                    baseline_options,
                    index=0 if comparison_tool.baseline_scenario is None else
                          baseline_options.index(comparison_tool.baseline_scenario)
                    if comparison_tool.baseline_scenario in baseline_options else 0
                )

                if st.button("🎯 Set Baseline", key="set_baseline"):
                    comparison_tool.set_baseline_scenario(baseline_selection)
                    st.success(f"✅ Baseline set to: {baseline_selection}")

            if st.button("📊 Generate Comparison", key="generate_comparison"):
                try:
                    comparison_results = comparison_tool.compare_scenarios()

                    if comparison_results:
                        st.markdown("### 📋 Comparison Results")

                        # Show summary
                        if "summary" in comparison_results:
                            st.markdown("**Summary of Changes:**")
                            summary_col1, summary_col2, summary_col3 = st.columns(3)

                            summary_data = comparison_results["summary"]
                            if "annual_h2_production_t" in summary_data:
                                h2_summary = summary_data["annual_h2_production_t"]
                                with summary_col1:
                                    h2_avg = h2_summary.get("avg_change", 0)
                                    st.metric("Avg H₂ Production Change", f"{h2_avg:.1f}%",
                                            delta=f"{h2_avg:.1f}%" if abs(h2_avg) > 0.1 else None)

                            if "lcoh_fixed" in summary_data:
                                lcoh_summary = summary_data["lcoh_fixed"]
                                with summary_col2:
                                    lcoh_avg = lcoh_summary.get("avg_change", 0)
                                    st.metric("Avg LCOH Change", f"{lcoh_avg:.1f}%",
                                            delta=f"{'↘️' if lcoh_avg < 0 else '↗️'} {abs(lcoh_avg):.1f}%")

                            if "npv" in summary_data:
                                npv_summary = summary_data["npv"]
                                with summary_col3:
                                    npv_avg = npv_summary.get("avg_change", 0)
                                    st.metric("Avg NPV Change", f"{npv_avg:.1f}%",
                                            delta=f"{npv_avg:.1f}%")

                        # Show rankings if available
                        if "ranking" in comparison_results and comparison_results["ranking"]:
                            st.markdown("### 🏆 Scenario Rankings")

                            ranking_df = pd.DataFrame(comparison_results["ranking"])
                            # Show top 5 rankings
                            display_columns = ["scenario", "score", "rank"]
                            if not ranking_df.empty and "rank" in ranking_df.columns:
                                top_rankings = ranking_df[display_columns].head(5)
                                st.dataframe(top_rankings, use_container_width=True)

                        # Show recommendations
                        if "recommendations" in comparison_results and comparison_results["recommendations"]:
                            st.markdown("### 💡 Recommendations")
                            for rec in comparison_results["recommendations"]:
                                st.info(rec)

                        # Detailed comparison table
                        if "detailed_comparison" in comparison_results:
                            st.markdown("### 📋 Detailed Comparison")

                            # Convert to DataFrame for display
                            detailed_data = []
                            for scenario_name, metrics in comparison_results["detailed_comparison"].items():
                                for metric_name, metric_data in metrics.items():
                                    if metric_name in ["annual_h2_production_t", "lcoh_fixed", "npv"]:
                                        detailed_data.append({
                                            "Scenario": scenario_name,
                                            "Metric": metric_name.replace("_", " ").title(),
                                            "Baseline": metric_data.get("baseline", 0),
                                            "Compared": metric_data.get("scenario", 0),
                                            "Difference": metric_data.get("difference", 0),
                                            "Change %": metric_data.get("percentage_change", 0)
                                        })

                            if detailed_data:
                                detailed_df = pd.DataFrame(detailed_data)

                                # Format the columns
                                if not detailed_df.empty:
                                    st.dataframe(detailed_df.style.format({
                                        "Baseline": "{:,.2f}",
                                        "Compared": "{:,.2f}",
                                        "Difference": "{:,.2f}",
                                        "Change %": "{:.1f}%"
                                    }), use_container_width=True)

                    else:
                        st.warning("⚠️ No comparison results generated. Check scenario data.")

                except Exception as e:
                    st.error(f"❌ Error generating comparison: {str(e)}")
        else:
            st.info("💡 **Tip:** Add at least 2 scenarios to enable comparison features. "
                   "This will allow you to analyze the impact of different configurations.")

    with comparison_subtabs[1]:
        st.markdown("### 📈 Advanced Analysis")

        st.info("🔄 Advanced ranking and sensitivity analysis features will be available in the full implementation.")

        # Placeholder for future advanced features
        if st.button("🚀 Enable Advanced Features", key="enable_advanced"):
            st.success("✅ Advanced features would be enabled here!")

    with comparison_subtabs[2]:
        st.markdown("### 🔍 Sensitivity Analysis")

        st.info("📊 Sensitivity analysis tools will help you understand how different parameters affect your results.")

        # Placeholder content
        sensitivity_options = st.multiselect(
            "Parameters to Analyze",
            ["Hydrogen Price", "Electricity Price", "Discount Rate", "Capital Costs"],
            help="Select parameters for sensitivity analysis"
        )

        if sensitivity_options:
            st.info(f"Selected parameters: {', '.join(sensitivity_options)}")

            if st.button("🔬 Run Sensitivity Analysis", key="run_sensitivity"):
                st.success("✅ Sensitivity analysis would run here!")

# Footer
add_s2d2_footer()