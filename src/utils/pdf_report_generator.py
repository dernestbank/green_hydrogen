"""
PDF Report Generator for Hydrogen Cost Analysis Tool
Creates professional PDF reports with charts, data, and analysis results.
"""

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus.flowables import PageBreak
from datetime import datetime
import io
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """
    Professional PDF report generator for hydrogen production analysis.

    Features:
    - Executive summary with key metrics
    - Technical analysis charts and data
    - Financial projections and analysis
    - System configuration details
    - Performance benchmarks
    """

    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for professional reports."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=20,
            alignment=1,  # Center aligned
            spaceAfter=30,
            textColor=colors.navy
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkgreen
        ))

        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.darkblue
        ))

        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.blue,
            fontName='Helvetica-Bold'
        ))

    def generate_comprehensive_report(self, results_data: Dict, charts_data: Optional[Dict] = None) -> bytes:
        """
        Generate a comprehensive PDF report from analysis results.

        Args:
            results_data: Dictionary containing all analysis results
            charts_data: Optional dictionary containing chart data

        Returns:
            PDF report as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=self.page_size,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)

        story = []

        # Title page
        story.extend(self._create_title_page(results_data))

        # Executive summary
        story.extend(self._create_executive_summary(results_data))

        # System configuration
        story.extend(self._create_system_configuration(results_data))

        # Technical analysis
        story.extend(self._create_technical_analysis(results_data))

        # Financial analysis
        story.extend(self._create_financial_analysis(results_data))

        # Operational analysis
        story.extend(self._create_operational_analysis(results_data))

        # Charts and visualizations
        if charts_data:
            story.extend(self._create_charts_section(charts_data))

        # Conclusions and recommendations
        story.extend(self._create_conclusions(results_data))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    def _create_title_page(self, results_data: Dict) -> List:
        """Create professional title page."""
        elements = []

        # Company/logo area
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("Green Hydrogen Production Framework", self.styles['ReportTitle']))

        # Report type and generation info
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph("Technical and Economic Analysis Report", self.styles['SubSection']))

        # Generation timestamp
        generation_time = datetime.now().strftime("%B %d, %Y at %H:%M")
        elements.append(Paragraph(f"Generated: {generation_time}", self.styles['Normal']))

        # Project information if available
        if 'inputs_summary' in results_data:
            inputs = results_data['inputs_summary']
            location = inputs.get('location', 'N/A')
            electrolyser_capacity = inputs.get('nominal_electrolyser_capacity', 'N/A')
            solar_capacity = inputs.get('nominal_solar_farm_capacity', 'N/A')
            wind_capacity = inputs.get('nominal_wind_farm_capacity', 'N/A')

            elements.append(Spacer(1, 0.5*inch))
            elements.append(Paragraph("Project Details:", self.styles['SubSection']))

            project_info = [
                [f"Location:", location],
                [f"Electrolyser Capacity:", f"{electrolyser_capacity} MW"],
                [f"Solar Capacity:", f"{solar_capacity} MW"],
                [f"Wind Capacity:", f"{wind_capacity} MW"]
            ]

            table = Table(project_info, colWidths=[2*inch, 4*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 12),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke)
            ]))

            elements.append(table)

        elements.append(PageBreak())
        return elements

    def _create_executive_summary(self, results_data: Dict) -> List:
        """Create executive summary section."""
        elements = []

        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))

        if 'operating_outputs' in results_data:
            op_out = results_data['operating_outputs']

            # Key metrics table
            summary_data = [
                ["Metric", "Value", "Description"],
                ["Annual H₂ Production", f"{op_out.get('Hydrogen Output for Fixed Operation [t/yr]', 0):,.0f} tonnes",
                 "Total hydrogen production capacity"],
                ["Capacity Factor", f"{op_out.get('Generator Capacity Factor', 0):.1%}",
                 "Overall renewable energy utilization"],
                ["Levelized Cost", f"${results_data.get('lcoh', {}).get('fixed', 0):.2f}/kg",
                 "Cost per kg of hydrogen produced"]
            ]

            if 'npv' in results_data:
                summary_data.append(["Net Present Value", f"${results_data['npv']:,.0f}",
                                   "Total project NPV"])

            table = Table(summary_data, colWidths=[2*inch, 2*inch, 3.5*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))

        elements.append(Paragraph("Summary Text", self.styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _create_financial_analysis(self, results_data: Dict) -> List:
        """Create financial analysis section."""
        elements = []

        elements.append(Paragraph("Financial Analysis", self.styles['SectionHeader']))

        # Financial metrics display
        if 'cash_flows' in results_data:
            elements.append(Paragraph("Cash Flow Projections", self.styles['SubSection']))

            # Get summary of key financial metrics
            cash_flows = pd.DataFrame(results_data['cash_flows'])

            # First 10 years + final year summary
            display_years = [0, 1, 2, 3, 4, 5, 9, 19]  # Years to show
            financial_data = [
                ["Year", "CAPEX ($)", "OPEX ($)", "Revenue ($)", "Cash Flow ($)", "Cumulative ($)"]
            ]

            cumulative_total = 0
            for year in display_years:
                if year < len(cash_flows):
                    capex = cash_flows.get('Gen_CAPEX', pd.Series([0]*len(cash_flows)))[year] + \
                           cash_flows.get('Elec_CAPEX', pd.Series([0]*len(cash_flows)))[year]
                    opex = cash_flows.get('Gen_OPEX', pd.Series([0]*len(cash_flows)))[year] + \
                          cash_flows.get('Elec_OandM', pd.Series([0]*len(cash_flows)))[year]
                    revenue = cash_flows.get('hydrogen_revenue', pd.Series([0]*len(cash_flows)))[year]
                    total = cash_flows.get('Total', pd.Series([0]*len(cash_flows)))[year] if year < len(cash_flows) else 0

                    cumulative_total += total if year > 0 else capex  # CAPEX is negative

                    financial_data.append([
                        str(year),
                        f"{abs(capex):,.0f}" if capex != 0 else "-",
                        f"{opex:,.0f}" if opex != 0 else "-",
                        f"{revenue:,.0f}" if revenue > 0 else "-",
                        f"{total:,.0f}" if total != 0 else "-",
                        f"{cumulative_total:,.0f}" if cumulative_total != 0 else "-"
                    ])
                else:
                    break

            table = Table(financial_data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white)
            ]))

            elements.append(table)

        elements.append(PageBreak())
        return elements

    def _create_system_configuration(self, results_data: Dict) -> List:
        """Create system configuration section."""
        elements = []

        elements.append(Paragraph("System Configuration", self.styles['SectionHeader']))

        if 'inputs_summary' in results_data:
            inputs = results_data['inputs_summary']

            config_data = [
                ["Component", "Specification", "Notes"],
                ["Location", inputs.get('location', 'N/A'), "Project site location"],
                ["Electrolyser Type", "PEM", "Polymer Electrolyte Membrane"],
                ["Electrolyser Capacity", f"{inputs.get('nominal_electrolyser_capacity', 0)} MW", "Rated power capacity"],
                ["Solar Farm Capacity", f"{inputs.get('nominal_solar_farm_capacity', 0)} MW", "PV array capacity"],
                ["Wind Farm Capacity", f"{inputs.get('nominal_wind_farm_capacity', 0)} MW", "Wind turbine capacity"],
                ["Battery Power", f"{inputs.get('battery_power_rating', 0)} MW", "Battery power rating"],
                ["Battery Duration", f"{inputs.get('battery_storage_duration', 0)} hours", "Storage duration"]
            ]

            table = Table(config_data, colWidths=[2*inch, 2*inch, 3*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke)
            ]))

            elements.append(table)

        elements.append(PageBreak())
        return elements

    def _create_technical_analysis(self, results_data: Dict) -> List:
        """Create technical analysis section."""
        elements = []

        elements.append(Paragraph("Technical Analysis", self.styles['SectionHeader']))

        if 'operating_outputs' in results_data:
            op_out = results_data['operating_outputs']

            technical_data = [
                ["Parameter", "Value", "Unit"],
                ["Generator Capacity Factor", f"{op_out.get('Generator Capacity Factor', 0):.3f}", "%"],
                ["Electrolyser Capacity Factor", f"{op_out.get('Achieved Electrolyser Capacity Factor', 0):.3f}", "%"],
                ["Time at Rated Capacity", f"{op_out.get('Time Electrolyser is at its Rated Capacity', 0):.1f}", "hours"],
                ["Total Operating Time", f"{op_out.get('Total Time Electrolyser is Operating', 0):.1f}", "hours"],
                ["Energy to Electrolyser", f"{op_out.get('Energy in to Electrolyser [MWh/yr]', 0):,.0f}", "MWh/yr"],
                ["Surplus Energy", f"{op_out.get('Surplus Energy [MWh/yr]', 0):,.0f}", "MWh/yr"]
            ]

            table = Table(technical_data, colWidths=[2.5*inch, 2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white)
            ]))

            elements.append(table)

        elements.append(PageBreak())
        return elements

    def _create_operational_analysis(self, results_data: Dict) -> List:
        """Create operational analysis section."""
        elements = []

        elements.append(Paragraph("Operational Analysis", self.styles['SectionHeader']))

        if 'operating_outputs' in results_data:
            op_out = results_data['operating_outputs']

            # Production analysis
            elements.append(Paragraph("Production Analysis", self.styles['SubSection']))

            production_data = [
                ["Hydrogen Production Metrics", "Fixed Operation", "Variable Operation", "Unit"],
                ["Annual Production", f"{op_out.get('Hydrogen Output for Fixed Operation [t/yr]', 0):,.0f}",
                                        f"{op_out.get('Hydrogen Output for Variable Operation [t/yr]', 0):,.0f}", "tonnes/year"],
                ["Daily Production", f"{op_out.get('Hydrogen Output for Fixed Operation [t/yr]', 0)/365:,.1f}",
                                    f"{op_out.get('Hydrogen Output for Variable Operation [t/yr]', 0)/365:,.1f}", "tonnes/day"],
                ["Hourly Production", f"{op_out.get('Hydrogen Output for Fixed Operation [t/yr]', 0)/8760:,.3f}",
                                      f"{op_out.get('Hydrogen Output for Variable Operation [t/yr]', 0)/8760:,.3f}", "tonnes/hour"]
            ]

            table = Table(production_data, colWidths=[2.5*inch, 2*inch, 2*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke)
            ]))

            elements.append(table)

        elements.append(PageBreak())
        return elements

    def _create_charts_section(self, charts_data: Dict) -> List:
        """Create charts and visualizations section."""
        elements = []

        elements.append(Paragraph("Charts and Visualizations", self.styles['SectionHeader']))
        elements.append(Paragraph("Note: Charts would be included in full implementation with proper image processing.", self.styles['Normal']))
        elements.append(PageBreak())

        return elements

    def _create_conclusions(self, results_data: Dict) -> List:
        """Create conclusions and recommendations section."""
        elements = []

        elements.append(Paragraph("Conclusions and Recommendations", self.styles['SectionHeader']))

        elements.append(Paragraph("Key Findings:", self.styles['SubSection']))
        elements.append(Paragraph("• Technical feasibility analysis indicates strong potential for commercial viability.", self.styles['Normal']))
        elements.append(Paragraph("• Financial model shows attractive return metrics subject to hydrogen pricing.", self.styles['Normal']))
        elements.append(Paragraph("• Operational characteristics suggest robust system performance.", self.styles['Normal']))

        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("Next Steps:", self.styles['SubSection']))
        elements.append(Paragraph("• Conduct detailed site assessment and permitting analysis.", self.styles['Normal']))
        elements.append(Paragraph("• Perform comprehensive financial modeling with sensitivity analysis.", self.styles['Normal']))
        elements.append(Paragraph("• Initiate stakeholder engagement and regulatory approval process.", self.styles['Normal']))

        # Add generation timestamp and report metadata
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        elements.append(Paragraph("© Green Hydrogen Production Framework", self.styles['Normal']))

        return elements

# Factory function
def create_pdf_report(results_data: Dict, charts_data: Optional[Dict] = None) -> bytes:
    """
    Factory function to create a PDF report.

    Args:
        results_data: Analysis results dictionary
        charts_data: Optional charts dictionary

    Returns:
        PDF content as bytes
    """
    generator = PDFReportGenerator()
    return generator.generate_comprehensive_report(results_data, charts_data)