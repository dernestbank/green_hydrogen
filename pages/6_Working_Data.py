import streamlit as st
from utils import add_s2d2_footer

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Working Data")

st.header("Hourly Electrolyser Operation and Outputs")
st.write("This page is where the hourly electrolyser operation and outputs are calculated, as described in S6. Working Data.")

# Placeholder for content
st.subheader("Hourly Electrolyser Data")
st.write("Table/chart for hourly electrolyser operation and outputs.")

# Add S2D2 Lab footer
add_s2d2_footer()
