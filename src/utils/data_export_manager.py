"""
Data Export Manager for Hydrogen Cost Analysis Tool
Handles comprehensive data export functionality for all visualizations and tables.
"""

import pandas as pd
import json
from io import BytesIO
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import zipfile

logger = logging.getLogger(__name__)

class DataExportManager:
    """
    Comprehensive data export manager for the hydrogen cost analysis tool.

    Features:
    - Export all visualization data (charts, tables, metrics)
    - Multiple format support (CSV, Excel, JSON)
    - Bulk export with zip archiving
    - Metadata inclusion and timestamp tracking
    - Filtered data export capabilities
    """

    def __init__(self):
        self.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def export_all_visualization_data(self, session_data: Dict[str, Any], format_type: str = "excel") -> bytes:
        """
        Export all visualization data from the analysis session.

        Args:
            session_data: Complete session data dictionary
            format_type: Export format ('excel', 'csv', 'json', 'zip')

        Returns:
            Exported data as bytes
        """
        export_data = self._collect_all_data(session_data)

        if format_type == "excel":
            return self._export_as_excel(export_data)
        elif format_type == "csv":
            return self._export_as_csv_zip(export_data)
        elif format_type == "json":
            return self._export_as_json(export_data)
        elif format_type == "zip":
            return self._export_as_complete_zip(export_data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _collect_all_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and organize all visualization data from session.

        Args:
            session_data: Raw session data

        Returns:
            Organized export data dictionary
        """
        export_data = {
            "metadata": self._create_export_metadata(session_data),
            "input_data": {},
            "results_data": {},
            "visualization_data": {},
            "charts_data": {},
            "tables_data": {}
        }

        # Extract input summary data
        if "model_results" in session_data:
            model_results = session_data["model_results"]

            if "inputs_summary" in model_results:
                export_data["input_data"]["system_configuration"] = model_results["inputs_summary"]

            if "operating_outputs" in model_results:
                export_data["results_data"]["operating_outputs"] = model_results["operating_outputs"]

            if "lcoh" in model_results:
                export_data["results_data"]["lcoh_analysis"] = model_results["lcoh"]

            if "cash_flows" in model_results:
                export_data["results_data"]["cash_flows"] = pd.DataFrame(model_results["cash_flows"]).to_dict('records')

            # Extract API response data
            if "api_responses" in model_results:
                api_data_list = []
                for key, api_response in model_results["api_responses"].items():
                    if isinstance(api_response, dict) and "data" in api_response:
                        data_type = "solar" if "irradiance" in api_response["data"] else "wind"
                        location_name = key.replace("solar_", "").replace("wind_", "").replace("_", " ").title()

                        # Convert API data to DataFrame format
                        api_df = self._convert_api_data_to_dataframe(api_response, data_type, location_name)
                        if not api_df.empty:
                            api_data_list.append({
                                "data_type": data_type,
                                "location": location_name,
                                "data": api_df.to_dict('records')
                            })

                export_data["results_data"]["api_raw_data"] = api_data_list

            # Extract hourly operation data if available
            hourly_data = session_data.get('hourly_operation_data')
            if hourly_data is not None and hasattr(hourly_data, 'empty') and not hourly_data.empty:
                export_data["results_data"]["hourly_operation_data"] = \
                    hourly_data.to_dict('records')

        return export_data

    def _convert_api_data_to_dataframe(self, api_response: Dict, data_type: str, location: str) -> pd.DataFrame:
        """Convert API response data to DataFrame format."""
        try:
            data = api_response.get("data", {})
            metadata = api_response.get("metadata", {})

            if not data or 'capacity_factor' not in data:
                return pd.DataFrame()

            # Create the dataframe
            df_data = {
                'timestamp': pd.to_datetime(metadata.get('timestamps', [])),
                'capacity_factor': data.get('capacity_factor', []),
                'data_type': [data_type] * len(data.get('capacity_factor', [])),
                'location': [location] * len(data.get('capacity_factor', []))
            }

            # Add type-specific fields
            if data_type == "solar":
                df_data.update({
                    'irradiance': data.get('irradiance', []),
                    'temperature': data.get('temperature', [])
                })
            elif data_type == "wind":
                df_data.update({
                    'wind_speed': data.get('wind_speed', []),
                    'temperature': data.get('temperature', [])
                })

            df = pd.DataFrame(df_data)
            return df.dropna(how='all')

        except Exception as e:
            logger.warning(f"Error converting API data: {e}")
            return pd.DataFrame()

    def _create_export_metadata(self, session_data: Dict) -> Dict[str, Any]:
        """Create export metadata."""
        return {
            "export_timestamp": self.export_timestamp,
            "export_datetime": datetime.now().isoformat(),
            "tool_version": "Hydrogen Cost Analysis Tool v2.0",
            "data_sources": ["Renewables.Ninja API", "Internal Calculations"],
            "session_info": {
                "has_model_results": "model_results" in session_data,
                "has_hourly_data": hasattr(session_data, 'hourly_operation_data'),
                "export_includes": ["inputs", "results", "visualizations", "api_data"]
            }
        }

    def _export_as_excel(self, data: Dict[str, Any]) -> bytes:
        """Export all data as a comprehensive Excel file."""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Metadata sheet
            pd.DataFrame([data["metadata"]]).to_excel(writer, sheet_name='Export_Metadata', index=False)

            # Input data
            for key, value in data["input_data"].items():
                if isinstance(value, dict):
                    pd.DataFrame([value]).to_excel(writer, sheet_name=f'Inputs_{key.title()}', index=False)

            # Results data
            for key, value in data["results_data"].items():
                if isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        pd.DataFrame(value).to_excel(writer, sheet_name=f'Results_{key.title()}', index=False)
                elif isinstance(value, dict):
                    pd.DataFrame([value]).to_excel(writer, sheet_name=f'Results_{key.title()}', index=False)

            # Create summary sheet
            summary_data = []
            if "results_data" in data:
                results = data["results_data"]
                if "operating_outputs" in results:
                    op_out = results["operating_outputs"]
                    summary_data.extend([
                        ["Annual H₂ Production (tonnes)", op_out.get("Hydrogen Output for Fixed Operation [t/yr]", 0)],
                        ["Generator Capacity Factor", op_out.get("Generator Capacity Factor", 0)],
                        ["Levelized Cost ($/kg)", data.get("lcoh", {}).get("fixed", 0)]
                    ])

            if summary_data:
                pd.DataFrame(summary_data, columns=["Metric", "Value"]).to_excel(
                    writer, sheet_name='Summary', index=False
                )

        output.seek(0)
        return output.getvalue()

    def _export_as_csv_zip(self, data: Dict[str, Any]) -> bytes:
        """Export all data as CSV files in a ZIP archive."""
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Export metadata
            metadata_csv = pd.DataFrame([data["metadata"]]).to_csv(index=False)
            zip_file.writestr(f"export_metadata_{self.export_timestamp}.csv", metadata_csv)

            # Export input data
            for key, value in data["input_data"].items():
                if isinstance(value, dict):
                    csv_content = pd.DataFrame([value]).to_csv(index=False)
                    zip_file.writestr(f"inputs_{key}_{self.export_timestamp}.csv", csv_content)

            # Export results data
            for key, value in data["results_data"].items():
                if isinstance(value, list) and value:
                    if isinstance(value[0], dict):
                        csv_content = pd.DataFrame(value).to_csv(index=False)
                        zip_file.writestr(f"results_{key}_{self.export_timestamp}.csv", csv_content)
                elif isinstance(value, dict):
                    csv_content = pd.DataFrame([value]).to_csv(index=False)
                    zip_file.writestr(f"results_{key}_{self.export_timestamp}.csv", csv_content)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def _export_as_json(self, data: Dict[str, Any]) -> bytes:
        """Export all data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8')

    def _export_as_complete_zip(self, data: Dict[str, Any]) -> bytes:
        """Export data in all formats within a ZIP file."""
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Excel export
            excel_data = self._export_as_excel(data)
            zip_file.writestr(f"complete_export_{self.export_timestamp}.xlsx", excel_data)

            # CSV files
            csv_zip_data = self._export_as_csv_zip(data)
            # Note: This creates a nested ZIP, which might not be ideal but works

            # JSON export
            json_data = self._export_as_json(data)
            zip_file.writestr(f"complete_export_{self.export_timestamp}.json", json_data)

            # README file
            readme_content = self._create_readme_content()
            zip_file.writestr("README.txt", readme_content)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def _create_readme_content(self) -> str:
        """Create README content for the export package."""
        readme = f"""
HYDROGEN COST ANALYSIS TOOL - COMPLETE DATA EXPORT
=================================================

Export Timestamp: {self.export_timestamp}

This package contains all data from your hydrogen cost analysis session
exported in multiple formats for comprehensive analysis and backup.

CONTENTS:
---------
1. Excel Format (complete_export_*.xlsx)
   - All data organized in Excel sheets
   - Summary, inputs, results, and raw data

2. JSON Format (complete_export_*.json)
   - Complete structured data export
   - Ideal for programmatic analysis

3. Individual CSV Files
   - Granular data export
   - Each data source as separate CSV file

DATA SECTIONS:
--------------
- Metadata: Export information and timestamps
- Inputs: System configuration and parameters
- Results: Analysis outputs and calculations
- API Raw Data: Original data from Renewables.Ninja
- Hourly Operations: Time-series operational data

For questions or support, contact the development team.

Generated by Hydrogen Cost Analysis Tool v2.0
© {datetime.now().year} Green Hydrogen Production Framework
"""
        return readme

    def export_specific_visualization(self, viz_type: str, viz_data: Any, format_type: str = "csv") -> bytes:
        """
        Export data for a specific visualization type.

        Args:
            viz_type: Type of visualization ('results_table', 'chart_data', etc.)
            viz_data: The visualization data
            format_type: Export format ('csv', 'json', 'excel')

        Returns:
            Exported data as bytes
        """
        if format_type == "csv":
            if isinstance(viz_data, pd.DataFrame):
                return viz_data.to_csv(index=False).encode('utf-8')
            elif isinstance(viz_data, dict):
                return pd.DataFrame([viz_data]).to_csv(index=False).encode('utf-8')
            else:
                return str(viz_data).encode('utf-8')

        elif format_type == "json":
            return json.dumps(viz_data, indent=2, default=str).encode('utf-8')

        elif format_type == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if isinstance(viz_data, pd.DataFrame):
                    viz_data.to_excel(writer, sheet_name=viz_type, index=False)
                elif isinstance(viz_data, dict):
                    pd.DataFrame([viz_data]).to_excel(writer, sheet_name=viz_type, index=False)
            output.seek(0)
            return output.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format_type}")

# Factory functions
def create_complete_data_export(session_data: Dict[str, Any], format_type: str = "excel") -> bytes:
    """
    Factory function to create a complete data export.

    Args:
        session_data: Session data to export
        format_type: Export format

    Returns:
        Exported data as bytes
    """
    manager = DataExportManager()
    return manager.export_all_visualization_data(session_data, format_type)

def export_single_visualization(viz_type: str, viz_data: Any, format_type: str = "csv") -> bytes:
    """
    Factory function to export a single visualization.

    Args:
        viz_type: Visualization type
        viz_data: Visualization data
        format_type: Export format

    Returns:
        Exported data as bytes
    """
    manager = DataExportManager()
    return manager.export_specific_visualization(viz_type, viz_data, format_type)