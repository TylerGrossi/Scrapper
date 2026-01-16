import streamlit as st

# Import configuration and styling
from config import setup_page, apply_styling

# Import data loader
from data_loader import load_and_filter_all_data, load_returns_data_raw

# Import tab renderers
from stock_screener import render_stock_screener_tab
from powerbi import render_powerbi_tab
from stop_loss_analysis import render_stop_loss_tab
from earnings_analysis import render_earnings_analysis_tab

# ------------------------------------
# PAGE SETUP
# ------------------------------------
setup_page()
apply_styling()

# ------------------------------------
# LOAD DATA
# ------------------------------------
# Load filtered data for analysis tabs
all_data = load_and_filter_all_data()
returns_df = all_data['returns']
hourly_df = all_data['hourly']
filter_stats = all_data['filter_stats']

# Load raw data for stock screener (includes stocks without 5D return)
raw_returns_df = load_returns_data_raw()

# ------------------------------------
# MAIN APP
# ------------------------------------
st.title("Earnings Momentum Strategy")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Stock Screener", 
    "PowerBI", 
    "Stop Loss Analysis", 
    "Earnings Analysis"
])

# Render each tab
with tab1:
    render_stock_screener_tab(raw_returns_df)

with tab2:
    render_powerbi_tab()

with tab3:
    render_stop_loss_tab(returns_df, hourly_df, filter_stats)

with tab4:
    render_earnings_analysis_tab(returns_df, filter_stats)

# FOOTER
st.markdown("---")
st.caption("Earnings Momentum Strategy")