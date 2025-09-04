import streamlit as st
from utils import add_s2d2_footer

st.set_page_config(layout="wide")

st.title("Hydrogen Cost Analysis Tool - Levelized Cost Analysis")

st.header("Capital and Operating Costs Breakdown")
st.write("This page will present the annual operational profile for each year up to the lifetime as well as the discounted and non-discounted cash flows, as described in S3. Levelised Cost Analysis.")

# Placeholder for content
st.subheader("Annual Operational Profile")
st.write("Table/chart for annual operational profile.")

st.subheader("Discounted and Non-Discounted Cash Flows")
st.write("Table/chart for cash flows.")

# Add S2D2 Lab footer
add_s2d2_footer()
