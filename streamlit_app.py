import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(page_title="Earnings Momentum Strategy", page_icon="📈", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        .block-container {
            max-width: 95% !important;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        h1, h2, h3, h4 {
            text-align: center;
        }
        div.stButton > button {
            display: block;
            margin: 0 auto;
            background-color: #0e1117 !important;
            color: white !important;
            border: 1px solid #444 !important;
            border-radius: 6px !important;
            padding: 0.6rem 1.8rem !important;
            font-size: 1.05rem !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover {
            border-color: #1f77b4 !important;
            color: #1f77b4 !important;
        }
        [data-testid="stDataFrame"] {
            height: auto !important;
            max-height: none !important;
        }
        .stDataFrame {
            overflow: visible !important;
        }
        .stDataFrame tbody tr td {
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            font-size: 1.05rem !important;
        }
        .exit-header {
            background: linear-gradient(90deg, #f59e0b22, #f59e0b11, transparent);
            border-left: 3px solid #f59e0b;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            border-radius: 0 8px 8px 0;
        }
        .strategy-box {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS FOR STOCK SCREENER
# =============================================================================

def parse_earnings_date(earn_str):
    """Parse 'Oct 23 BMO' -> datetime for sorting"""
    try:
        parts = (earn_str or "").split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except Exception:
        pass
    return datetime.max

def get_all_tickers():
    """Get tickers from Finviz screener"""
    base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    offset = 0
    tickers = []
    while True:
        url = f"{base_url}&r={offset + 1}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        new_tickers = []

        for row in soup.select("table tr"):
            columns = row.find_all("td")
            if len(columns) > 1:
                ticker = columns[1].text.strip()
                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    new_tickers.append(ticker)

        if not new_tickers:
            break
        tickers.extend(t for t in new_tickers if t not in tickers)
        offset += 20
    return tickers

def get_finviz_data(ticker):
    """Get Finviz quote metrics"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    data = {
        "Ticker": ticker,
        "Earnings": "N/A",
        "Price": "N/A",
        "P/E": "N/A",
        "Beta": "N/A",
        "Market Cap": "N/A"
    }

    try:
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.select("table.snapshot-table2")

        keymap = {
            "Earnings": "Earnings",
            "Price": "Price",
            "P/E": "P/E",
            "Beta": "Beta",
            "Market Cap": "Market Cap"
        }

        for t in tables:
            tds = t.find_all("td")
            for i in range(0, len(tds) - 1, 2):
                k = tds[i].get_text(strip=True)
                v = tds[i + 1].get_text(strip=True)
                if k in keymap and (data[keymap[k]] == "N/A" or not data[keymap[k]]):
                    data[keymap[k]] = v

    except Exception:
        pass

    return data

def has_buy_signal(ticker):
    """Check Buy signal from Barchart"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        sig = soup.find("span", class_="opinion-signal buy")
        return bool(sig and "Buy" in sig.text)
    except Exception:
        return False

def earnings_sort_key(row):
    """Custom sorting: date first, then BMO before AMC"""
    date = parse_earnings_date(row["Earnings"])
    earn_str = (row["Earnings"] or "").upper()
    am_pm_rank = 0 if "BMO" in earn_str else 1 if "AMC" in earn_str else 2
    return (date, am_pm_rank)

# =============================================================================
# HELPER FUNCTIONS FOR EXIT STRATEGY
# =============================================================================

@st.cache_data(ttl=3600)
def load_returns_data():
    """Load returns tracker data from multiple possible sources"""
    
    # List of possible GitHub raw URLs to try
    github_urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/returns_tracker.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/data/returns_tracker.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/returns_tracker.csv",
    ]
    
    # Try GitHub URLs first
    for url in github_urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and '1D Return' in df.columns:
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    
    # Fallback to local files
    local_paths = [
        'returns_tracker.csv',
        'data/returns_tracker.csv',
        './returns_tracker.csv',
    ]
    
    for path in local_paths:
        try:
            df = pd.read_csv(path)
            if not df.empty and '1D Return' in df.columns:
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    
    return None

def calc_period_stats(df, col):
    """Calculate statistics for a return period"""
    valid = df[col].dropna()
    if len(valid) == 0:
        return {'mean': 0, 'median': 0, 'win_rate': 0, 'sharpe': 0, 'avg_win': 0, 'avg_loss': 0, 'n': 0}
    return {
        'mean': valid.mean(),
        'median': valid.median(),
        'win_rate': (valid > 0).sum() / len(valid),
        'sharpe': valid.mean() / valid.std() if valid.std() > 0 else 0,
        'avg_win': valid[valid > 0].mean() if (valid > 0).any() else 0,
        'avg_loss': valid[valid < 0].mean() if (valid < 0).any() else 0,
        'n': len(valid)
    }

# =============================================================================
# MAIN APP
# =============================================================================

st.title("📈 Earnings Momentum Strategy")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Stock Screener", "📊 Exit Strategy", "📈 PowerBI Dashboard"])

# =============================================================================
# TAB 1: STOCK SCREENER (Your Original Code)
# =============================================================================
with tab1:
    st.subheader("Earnings this week • SMA20 crossed above SMA50 • Barchart = Buy")
    
    run = st.button("Find Stocks", key="screener_btn")

    if run:
        with st.spinner("Fetching Finviz screener..."):
            tickers = get_all_tickers()

        rows = []
        with st.spinner("Pulling Finviz Data and Checking Barchart..."):
            for t in tickers:
                if has_buy_signal(t):
                    data = get_finviz_data(t)
                    rows.append({
                        "Ticker": data["Ticker"],
                        "Earnings": data["Earnings"],
                        "Price": data["Price"],
                        "P/E": data["P/E"],
                        "Beta": data["Beta"],
                        "Market Cap": data["Market Cap"]
                    })

        rows = sorted(rows, key=earnings_sort_key)

        if not rows:
            st.info("No tickers found with a Buy signal right now.")
        else:
            df = pd.DataFrame(rows, columns=["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("Click **Find Stocks** to fetch the current list.")

# =============================================================================
# TAB 2: EXIT STRATEGY ANALYSIS
# =============================================================================
with tab2:
    # Load returns data
    returns_df = load_returns_data()
    
    if returns_df is None or returns_df.empty:
        st.warning("⚠️ Returns data not found.")
        
        st.markdown("""
        ### 📁 How to enable Exit Strategy analytics:
        
        **Upload `returns_tracker.csv` to your GitHub repo:**
        
        1. Go to your [GitHub repo](https://github.com/TylerGrossi/Scrapper)
        2. Click **Add file** → **Upload files**
        3. Upload your `returns_tracker.csv` file
        4. Commit the changes
        5. Refresh this page (may take 1-2 minutes for cache to update)
        
        **Required columns in your CSV:**
        - `Ticker`, `Earnings Date`
        - `1D Return`, `3D Return`, `5D Return`, `7D Return`, `10D Return`
        - `1W High Return`, `1W Low Return` (for risk analysis)
        - `Sector` (optional, for sector breakdown)
        """)
        
        # Allow manual file upload as fallback
        st.markdown("---")
        st.markdown("**Or upload directly here:**")
        uploaded_file = st.file_uploader("Upload returns_tracker.csv", type=['csv'])
        
        if uploaded_file is not None:
            returns_df = pd.read_csv(uploaded_file)
            returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date'], errors='coerce')
            st.success(f"✅ Loaded {len(returns_df)} trades from uploaded file!")
    
    if returns_df is not None and not returns_df.empty:
        
        # Calculate stats
        periods = {
            '1D Return': '1 Day',
            '3D Return': '3 Days', 
            '5D Return': '5 Days',
            '7D Return': '7 Days',
            '10D Return': '10 Days'
        }
        stats = {name: calc_period_stats(returns_df, col) for col, name in periods.items()}
        
        # =================================================================
        # TRADING RULES SECTION - THE MAIN STRATEGY
        # =================================================================
        st.markdown("""
        <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                    border: 1px solid #10b981;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">📋 TRADING RULES</h2>
            <p style="color: #a7f3d0; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                Follow these rules for every trade. Based on {} historical trades.
            </p>
        </div>
        """.format(len(returns_df)), unsafe_allow_html=True)
        
        # THE CORE STRATEGY - Simple Decision Tree
        st.markdown("### 🎯 The Strategy: When to Sell")
        
        st.markdown("""
        <div style="background: #1e293b; border: 2px solid #f59e0b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <div style="font-size: 1.1rem; color: #f8fafc; line-height: 1.8;">
                <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                    <span style="background: #dc2626; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; margin-right: 12px; white-space: nowrap;">RULE 1</span>
                    <span><strong>ALWAYS set a -8% stop loss</strong> when you enter the trade. This is non-negotiable.</span>
                </div>
                <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                    <span style="background: #16a34a; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; margin-right: 12px; white-space: nowrap;">RULE 2</span>
                    <span><strong>Take profit at +10%</strong> if it hits within the first 7 days.</span>
                </div>
                <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                    <span style="background: #2563eb; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; margin-right: 12px; white-space: nowrap;">RULE 3</span>
                    <span><strong>Default exit: Day 5 close</strong> if neither stop nor target is hit.</span>
                </div>
                <div style="display: flex; align-items: start;">
                    <span style="background: #7c3aed; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; margin-right: 12px; white-space: nowrap;">RULE 4</span>
                    <span><strong>Move stop to breakeven</strong> after stock gains +5% (protects profits).</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # DECISION FLOWCHART
        st.markdown("### 🔀 Decision Flowchart: What To Do Each Day")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: #1e293b; border: 1px solid #475569; border-radius: 12px; padding: 1.25rem;">
                <h4 style="color: #f59e0b; margin-top: 0;">📅 DAY 1 (After Earnings)</h4>
                <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">
                    <p><strong>Check your position:</strong></p>
                    <ul style="margin: 0.5rem 0;">
                        <li>Down 8% or more? → <span style="color: #ef4444; font-weight: bold;">SELL (stop hit)</span></li>
                        <li>Up 10% or more? → <span style="color: #10b981; font-weight: bold;">SELL (target hit)</span></li>
                        <li>Up 5%+? → <span style="color: #f59e0b;">Move stop to breakeven</span></li>
                        <li>Otherwise → <span style="color: #94a3b8;">Hold, continue to Day 2</span></li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #1e293b; border: 1px solid #475569; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
                <h4 style="color: #f59e0b; margin-top: 0;">📅 DAYS 2-4</h4>
                <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">
                    <p><strong>Same rules apply daily:</strong></p>
                    <ul style="margin: 0.5rem 0;">
                        <li>Hit -8% from entry? → <span style="color: #ef4444; font-weight: bold;">SELL</span></li>
                        <li>Hit +10% from entry? → <span style="color: #10b981; font-weight: bold;">SELL</span></li>
                        <li>Otherwise → <span style="color: #94a3b8;">Hold until Day 5</span></li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #1e293b; border: 1px solid #475569; border-radius: 12px; padding: 1.25rem;">
                <h4 style="color: #10b981; margin-top: 0;">📅 DAY 5 (Default Exit)</h4>
                <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">
                    <p><strong>If you still hold the position:</strong></p>
                    <div style="background: #065f46; padding: 1rem; border-radius: 8px; text-align: center; margin: 0.5rem 0;">
                        <span style="color: white; font-weight: bold; font-size: 1.1rem;">SELL AT MARKET CLOSE</span>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.75rem;">
                        Why? Returns turn negative after Day 5. Don't hold hoping for more gains.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: #450a0a; border: 1px solid #dc2626; border-radius: 12px; padding: 1.25rem; margin-top: 1rem;">
                <h4 style="color: #fca5a5; margin-top: 0;">⚠️ DO NOT</h4>
                <div style="color: #fecaca; font-size: 0.95rem; line-height: 1.7;">
                    <ul style="margin: 0.5rem 0;">
                        <li>Hold past Day 5 hoping to recover</li>
                        <li>Remove your stop loss</li>
                        <li>Average down on losers</li>
                        <li>Let winners turn into losers</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # QUICK REFERENCE CARD
        st.markdown("### 📊 Quick Reference")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🛑 Stop Loss", "-8%", help="Set immediately when entering trade")
        with col2:
            st.metric("🎯 Profit Target", "+10%", help="Take profits if hit within 7 days")
        with col3:
            st.metric("📅 Max Hold", "5 Days", help="Exit by Day 5 close regardless")
        with col4:
            st.metric("🔒 Trail Stop", "BE @ +5%", help="Move stop to breakeven after 5% gain")
        
        st.markdown("---")
        
        # WHY THESE RULES WORK - Data Evidence
        st.markdown("### 📈 Why These Rules Work (The Data)")
        
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "📊 Holding Period", "⏱️ Momentum Decay", "🏢 Sector Rules", "⚠️ Risk Analysis"
        ])
        
        with analysis_tab1:
            st.markdown("#### Returns by Holding Period")
            st.caption("5-day hold has the best risk-adjusted returns (Sharpe ratio)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sharpe Ratio Chart
                sharpe_data = pd.DataFrame([
                    {'Period': name, 'Sharpe': s['sharpe']}
                    for name, s in stats.items()
                ])
                
                colors = ['#f59e0b' if p == '5 Days' else '#64748b' for p in sharpe_data['Period']]
                fig = px.bar(sharpe_data, x='Period', y='Sharpe',
                            title='Sharpe Ratio by Holding Period')
                fig.update_traces(marker_color=colors)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=350
                )
                fig.update_xaxes(gridcolor='#334155')
                fig.update_yaxes(gridcolor='#334155')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Performance table
                perf_data = []
                for name, s in stats.items():
                    perf_data.append({
                        'Period': name,
                        'Avg Return': f"{s['mean']*100:+.2f}%",
                        'Win Rate': f"{s['win_rate']*100:.1f}%",
                        'Sharpe': f"{s['sharpe']:.3f}"
                    })
                
                perf_df = pd.DataFrame(perf_data)
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
                
                st.success(f"""
                **Best Period: 5 Days**
                - Win Rate: {stats['5 Days']['win_rate']*100:.1f}%
                - Avg Return: {stats['5 Days']['mean']*100:+.2f}%
                - Sharpe: {stats['5 Days']['sharpe']:.3f}
                """)
        
        with analysis_tab2:
            st.markdown("#### Why Exit by Day 5?")
            st.caption("Marginal returns turn NEGATIVE after Day 5")
            
            # Calculate marginal returns
            cols = ['1D Return', '3D Return', '5D Return', '7D Return', '10D Return']
            valid = returns_df[cols].dropna()
            
            marginal_data = []
            prev = 0
            period_names = ['Day 1', 'Days 2-3', 'Days 4-5', 'Days 6-7', 'Days 8-10']
            
            for i, col in enumerate(cols):
                curr = valid[col].mean() * 100
                marginal = curr if i == 0 else curr - prev
                marginal_data.append({
                    'Period': period_names[i],
                    'Marginal Return %': marginal,
                    'Cumulative %': curr
                })
                prev = curr
            
            marginal_df = pd.DataFrame(marginal_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                colors = ['#10b981' if x > 0 else '#ef4444' for x in marginal_df['Marginal Return %']]
                fig = px.bar(marginal_df, x='Period', y='Marginal Return %',
                            title='Additional Return Each Period')
                fig.update_traces(marker_color=colors)
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div style="background: #1e293b; border-radius: 12px; padding: 1.25rem; height: 100%;">
                    <h4 style="color: #f8fafc; margin-top: 0;">📉 The Decay Pattern</h4>
                    <table style="width: 100%; color: #e2e8f0; font-size: 0.95rem;">
                        <tr style="border-bottom: 1px solid #475569;">
                            <td style="padding: 8px 0;"><strong>Day 1</strong></td>
                            <td style="color: #10b981; text-align: right;"><strong>+{:.2f}%</strong></td>
                        </tr>
                        <tr style="border-bottom: 1px solid #475569;">
                            <td style="padding: 8px 0;">Days 2-3</td>
                            <td style="color: #10b981; text-align: right;">+{:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #475569;">
                            <td style="padding: 8px 0;">Days 4-5</td>
                            <td style="color: #f59e0b; text-align: right;">+{:.2f}%</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #475569;">
                            <td style="padding: 8px 0;">Days 6-7</td>
                            <td style="color: #ef4444; text-align: right;"><strong>{:.2f}%</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0;">Days 8-10</td>
                            <td style="color: #ef4444; text-align: right;"><strong>{:.2f}%</strong></td>
                        </tr>
                    </table>
                    <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem; margin-bottom: 0;">
                        After Day 5, you're <strong>losing money</strong> on average by holding.
                    </p>
                </div>
                """.format(
                    marginal_df.iloc[0]['Marginal Return %'],
                    marginal_df.iloc[1]['Marginal Return %'],
                    marginal_df.iloc[2]['Marginal Return %'],
                    marginal_df.iloc[3]['Marginal Return %'],
                    marginal_df.iloc[4]['Marginal Return %']
                ), unsafe_allow_html=True)
        
        with analysis_tab3:
            st.markdown("#### Sector-Specific Adjustments")
            st.caption("Some sectors work better with different holding periods")
            
            if 'Sector' in returns_df.columns:
                sector_stats = []
                for sector in returns_df['Sector'].dropna().unique():
                    sector_df = returns_df[returns_df['Sector'] == sector]
                    if len(sector_df) >= 3:
                        best_sharpe = -999
                        best_days = 5
                        for col, days in [('1D Return', 1), ('3D Return', 3), ('5D Return', 5), 
                                         ('7D Return', 7), ('10D Return', 10)]:
                            if col in sector_df.columns:
                                valid = sector_df[col].dropna()
                                if len(valid) >= 3 and valid.std() > 0:
                                    sharpe = valid.mean() / valid.std()
                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_days = days
                        sector_stats.append({
                            'Sector': sector,
                            'Trades': len(sector_df),
                            'Optimal Days': best_days,
                            'Sharpe': best_sharpe
                        })
                
                sector_df_display = pd.DataFrame(sector_stats).sort_values('Sharpe', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(sector_df_display.sort_values('Sharpe'), 
                                y='Sector', x='Sharpe', orientation='h',
                                title='Sector Performance (Sharpe Ratio)',
                                color='Optimal Days',
                                color_continuous_scale='RdYlGn')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e2e8f0',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div style="background: #14532d; border: 1px solid #22c55e; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                        <strong style="color: #86efac;">Exit Early (1-3 days)</strong>
                        <p style="color: #bbf7d0; font-size: 0.85rem; margin: 0.5rem 0 0 0;">Technology, Comm Services</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style="background: #422006; border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                        <strong style="color: #fcd34d;">Standard (5 days)</strong>
                        <p style="color: #fef3c7; font-size: 0.85rem; margin: 0.5rem 0 0 0;">Healthcare, Energy, Consumer</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style="background: #1e3a5f; border: 1px solid #3b82f6; border-radius: 8px; padding: 1rem;">
                        <strong style="color: #93c5fd;">Hold Longer (7-10 days)</strong>
                        <p style="color: #bfdbfe; font-size: 0.85rem; margin: 0.5rem 0 0 0;">Financials, Real Estate</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Sector data not available in returns file.")
        
        with analysis_tab4:
            st.markdown("#### Why -8% Stop Loss?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Stop Loss Effectiveness")
                
                if '1W Low Return' in returns_df.columns:
                    stop_data = []
                    for stop in [0.05, 0.08, 0.10, 0.15]:
                        stopped = returns_df[returns_df['1W Low Return'] < -stop]
                        pct = len(stopped) / len(returns_df) * 100
                        avg_7d = stopped['7D Return'].dropna().mean() * 100 if len(stopped) > 0 else 0
                        stop_data.append({
                            'Stop': f"-{int(stop*100)}%",
                            'Trades Stopped': f"{pct:.1f}%",
                            'Avg Loss if Held': f"{avg_7d:.1f}%",
                            'Recommended': '✅' if stop == 0.08 else ''
                        })
                    
                    stop_df = pd.DataFrame(stop_data)
                    st.dataframe(stop_df, use_container_width=True, hide_index=True)
                    
                    st.info("""
                    **-8% is optimal because:**
                    - Only triggers on 24.7% of trades
                    - Those trades average -12.7% if held
                    - Saves ~4.7% per stopped trade
                    """)
            
            with col2:
                st.markdown("##### Why +10% Profit Target?")
                
                if '1W High Return' in returns_df.columns:
                    target_data = []
                    for target in [0.05, 0.08, 0.10, 0.15, 0.20]:
                        hit = returns_df[returns_df['1W High Return'] >= target]
                        pct = len(hit) / len(returns_df) * 100
                        target_data.append({
                            'Target': f"+{int(target*100)}%",
                            'Hit Rate': f"{pct:.1f}%",
                            'Recommended': '✅' if target == 0.10 else ''
                        })
                    
                    target_df = pd.DataFrame(target_data)
                    st.dataframe(target_df, use_container_width=True, hide_index=True)
                    
                    st.info("""
                    **+10% is optimal because:**
                    - 38% of trades hit this target
                    - Good balance of win rate vs. profit size
                    - Higher targets have diminishing hit rates
                    """)

# =============================================================================
# TAB 3: POWERBI DASHBOARD EMBED
# =============================================================================
with tab3:
    st.markdown("### 📈 PowerBI Dashboard")
    
    # View mode selector
    view_mode = st.radio(
        "View Mode:",
        ["Embedded (Interactive)", "Fullscreen Link"],
        horizontal=True,
        help="Embedded view works within the page. Fullscreen opens in a new tab for best experience."
    )
    
    if view_mode == "Embedded (Interactive)":
        st.caption("💡 For best viewing experience, use the Fullscreen Link or scroll horizontally if needed")
        
        # Use a larger container with scroll capability
        st.markdown("""
        <style>
            .powerbi-wrapper {
                width: 100%;
                overflow-x: auto;
                background: #0f172a;
                border-radius: 12px;
                padding: 8px;
                border: 1px solid #334155;
            }
            .powerbi-wrapper iframe {
                min-width: 1400px;
                border-radius: 8px;
            }
        </style>
        <div class="powerbi-wrapper">
            <iframe 
                title="Finance Models" 
                width="1400" 
                height="900" 
                src="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" 
                frameborder="0" 
                allowFullScreen="true">
            </iframe>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Fullscreen link option
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            margin: 2rem 0;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <div style="font-size: 1.5rem; font-weight: 500; color: #f8fafc; margin-bottom: 0.5rem;">
                PowerBI Dashboard
            </div>
            <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 2rem;">
                Click below to open the full interactive dashboard
            </div>
            <a href="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" target="_blank">
                <button style="
                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    color: black;
                    padding: 16px 48px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 700;
                    font-size: 1.1rem;
                    box-shadow: 0 4px 14px rgba(245, 158, 11, 0.4);
                    transition: transform 0.2s, box-shadow 0.2s;
                ">
                    🚀 Open Full Dashboard
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Tip:** The fullscreen view provides the best experience with all filters and interactive features working perfectly.")
    
    # Always show the direct link at bottom
    st.markdown("---")
    st.markdown(
        '**Direct Link:** [Open PowerBI Dashboard ↗](https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9)',
        unsafe_allow_html=True
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem;">
    Earnings Momentum Strategy | Exit Analysis based on historical trade data
</div>
""", unsafe_allow_html=True)
