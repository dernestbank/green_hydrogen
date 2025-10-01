import streamlit as st

st.set_page_config(
    page_title="Hydrogen Analysis Tool",
    layout="wide"
)

st.title("Hydrogen Cost Analysis Tool")
st.markdown("This tool is designed to provide a comprehensive techno-economic analysis of green hydrogen production.")

col1, col2 =st.columns([3,1])

with col1:
    st.markdown("""
    ### How to Use This Tool

    1.  **Navigate to the Inputs page** using the sidebar on the left.
    2.  **Inputs:** Provide all the necessary parameters for your analysis, including system configuration, component parameters, cost, and financing details.
    3.  **Results:** View the comprehensive results of the analysis, including Levelized Cost of Hydrogen (LCH2) and other key metrics.
    4.  **Detailed Analysis Pages:** Explore the Levelized Cost Analysis, Cash Flow Analysis, Raw Data, and Working Data pages for in-depth insights.

    To begin, please navigate to the **Inputs** page from the sidebar.
    """)
    

# Insert image from /images/Logo.png
st.image("images/Framework.png", caption="Hydrogen Cost Analysis Framework", use_column_width=True)

# Add footer with S2D2 Lab attribution
st.markdown("""
---
*Powered by [S2D2 LAB | PSU](https://s2d2lab.notion.site/) - Sustainable Design, Systems, and Decision-making (S2D2) Group  
Department of Chemical Engineering at Penn State*
""")
