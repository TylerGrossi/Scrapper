import streamlit as st

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
def setup_page():
    st.set_page_config(
        page_title="Earnings Momentum Strategy", 
        page_icon="📈", 
        layout="wide"
    )

# ------------------------------------
# CSS STYLING
# ------------------------------------
def apply_styling():
    st.markdown("""
    <style>
        .block-container {
            max-width: 95% !important;
            padding: 2rem 3rem;
        }
        
        /* Hide anchor links on headings */
        .stMarkdown a[href^="#"],
        a.headerlink,
        .header-link,
        [data-testid="StyledLinkIconContainer"] {
            display: none !important;
            visibility: hidden !important;
        }
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
        }
        
        /* Clean typography */
        h1, h2, h3 { font-weight: 600; }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.4rem; color: #e2e8f0; }
        h3 { font-size: 1.1rem; color: #cbd5e1; }
        
        /* Buttons */
        div.stButton > button {
            background: #1e293b;
            color: #f1f5f9;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background: #334155;
            border-color: #475569;
        }
        
        /* Tables */
        .stDataFrame { font-size: 0.9rem; }
        
        /* Remove excessive padding */
        .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
        
        /* Metric cards for backtest */
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-green { color: #22c55e; }
        .metric-red { color: #ef4444; }
        .metric-blue { color: #3b82f6; }
        .metric-yellow { color: #f59e0b; }
        .metric-white { color: #f1f5f9; }
        
        /* Exit breakdown cards */
        .exit-card {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .exit-count {
            font-size: 1.75rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .exit-pct {
            font-size: 0.9rem;
            color: #64748b;
        }
        .exit-return {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .exit-label {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }
    </style>
    """, unsafe_allow_html=True)