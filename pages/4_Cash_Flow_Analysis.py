import streamlit as st
from utils import add_s2d2_footer

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Cash Flow Analysis")

st.header("Detailed Cash Flows")
st.write("This page will contain the detailed cash flows required for calculation of the net profit, return on investment and payback period, as described in S4. Cash Flow Analysis.")

# Placeholder for content
st.subheader("Net Profit Calculation")
st.write("Table/chart for net profit.")

st.subheader("Return on Investment (ROI)")
st.write("Table/chart for ROI.")

st.subheader("Payback Period")
st.write("Table/chart for payback period.")

# Add S2D2 Lab footer
add_s2d2_footer()
