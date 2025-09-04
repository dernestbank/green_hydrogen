import streamlit as st
from utils import add_s2d2_footer

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Raw Data")

st.header("Hourly Electricity Generation Profiles")
st.write("This page will contain the hourly electricity generation profiles in the form of capacity factor traces, as described in S5. Raw Data.")

# Placeholder for content
st.subheader("Solar Traces")
st.write("Table/chart for solar traces.")

st.subheader("Wind Traces")
st.write("Table/chart for wind traces.")

st.subheader("Custom Trace Upload")
st.write("Option to upload custom solar/wind traces.")

# Add S2D2 Lab footer
add_s2d2_footer()
