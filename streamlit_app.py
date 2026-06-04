"""Commerce ML Lab — navigation router.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Commerce ML Lab",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar branding — shown on every page below the auto-generated nav links
with st.sidebar:
    st.caption("Richard McShinsky · [GitHub](https://github.com/richmcshinsky/commerce-ml-lab)")

pg = st.navigation(
    [
        st.Page("pages/home.py",                  title="Home",                  icon="🏠", default=True),
        st.Page("pages/1_Demand_Forecasting.py",  title="Demand Forecasting",    icon="📦"),
        st.Page("pages/2_Returns_Intelligence.py", title="Returns Intelligence",  icon="🔁"),
        st.Page("pages/3_Checkout_Uplift.py",     title="Checkout Uplift",       icon="🎯"),
    ]
)
pg.run()
