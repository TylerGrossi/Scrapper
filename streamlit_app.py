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
    """Load returns tracker data from GitHub or local file"""
    try:
        # Try loading from GitHub raw URL (update with your actual path)
        url = "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/returns_tracker.csv"
        df = pd.read_csv(url)
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        # Fallback to local file
        try:
            df = pd.read_csv('returns_tracker.csv')
            df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
            return df
        except:
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
        st.warning("⚠️ Returns data not found. Please upload returns_tracker.csv to your GitHub repo.")
        st.info("The Exit Strategy tab analyzes historical trade performance to determine optimal sell timing.")
    else:
        # Header
        st.markdown("""
        <div class="exit-header">
            <div style="font-size: 0.75rem; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600;">
                Quantitative Analysis
            </div>
            <div style="font-size: 1.4rem; font-weight: 500; color: #f8fafc; margin-top: 0.25rem;">
                Optimal Exit Strategy
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">
                Based on {} historical trades
            </div>
        </div>
        """.format(len(returns_df)), unsafe_allow_html=True)
        
        # Calculate stats
        periods = {
            '1D Return': '1 Day',
            '3D Return': '3 Days', 
            '5D Return': '5 Days',
            '7D Return': '7 Days',
            '10D Return': '10 Days'
        }
        stats = {name: calc_period_stats(returns_df, col) for col, name in periods.items()}
        
        # Key Metrics
        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Optimal Hold", "5 Days", help="Best risk-adjusted returns")
        with col2:
            wr = stats['5 Days']['win_rate'] * 100
            st.metric("✅ Win Rate", f"{wr:.1f}%", delta="5-day period")
        with col3:
            avg = stats['5 Days']['mean'] * 100
            st.metric("💰 Avg Return", f"+{avg:.2f}%")
        with col4:
            st.metric("🎯 Profit Target", "+10%", help="38% hit rate in 7 days")
        with col5:
            st.metric("🛑 Stop Loss", "-8%", help="24.7% trigger rate")
        
        st.markdown("---")
        
        # Strategy Rules Box
        st.markdown("### 🎯 Exit Strategy Rules")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success("""
            **PRIMARY EXIT**
            
            📅 **Day 5 Close**
            
            Captures 80%+ of momentum before decay sets in
            """)
        
        with col2:
            st.info("""
            **PROFIT TARGET**
            
            📈 **+10%**
            
            38% probability of hitting within 7 days
            """)
        
        with col3:
            st.error("""
            **STOP LOSS**
            
            📉 **-8%**
            
            Limits losses; avg loss if held: -12.7%
            """)
        
        with col4:
            st.warning("""
            **TRAIL STOP**
            
            🔒 **Breakeven @ +5%**
            
            Move stop to entry after 5% gain
            """)
        
        st.markdown("---")
        
        # Detailed Analysis Tabs
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "📈 Performance", "⏱️ Momentum Decay", "🏢 Sectors", "⚠️ Risk Mgmt"
        ])
        
        with analysis_tab1:
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
                    height=400
                )
                fig.update_xaxes(gridcolor='#334155')
                fig.update_yaxes(gridcolor='#334155')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Mean Return + Win Rate
                perf_data = pd.DataFrame([
                    {'Period': name, 'Mean Return %': s['mean']*100, 'Win Rate %': s['win_rate']*100}
                    for name, s in stats.items()
                ])
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=perf_data['Period'], y=perf_data['Mean Return %'], 
                          name='Mean Return %', marker_color='#10b981'),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=perf_data['Period'], y=perf_data['Win Rate %'],
                              name='Win Rate %', line=dict(color='#f59e0b', width=3),
                              mode='lines+markers', marker=dict(size=10)),
                    secondary_y=True
                )
                fig.update_layout(
                    title='Return & Win Rate by Holding Period',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                fig.update_xaxes(gridcolor='#334155')
                fig.update_yaxes(gridcolor='#334155', title_text="Mean Return %", secondary_y=False)
                fig.update_yaxes(gridcolor='#334155', title_text="Win Rate %", range=[50, 70], secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tab2:
            st.markdown("#### Marginal Return Analysis")
            st.caption("How much additional return does each period contribute?")
            
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
                            title='Marginal Return by Period')
                fig.update_traces(marker_color=colors)
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.area(marginal_df, x='Period', y='Cumulative %',
                             title='Cumulative Return Path')
                fig.update_traces(fill='tozeroy', line_color='#10b981',
                                 fillcolor='rgba(16, 185, 129, 0.2)')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.error("⚠️ **Key Insight:** Returns turn NEGATIVE after Day 5. Most alpha is captured in the first 3 days.")
        
        with analysis_tab3:
            st.markdown("#### Sector Performance")
            
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
                            '1D Avg': sector_df['1D Return'].mean() * 100 if '1D Return' in sector_df else 0,
                            '5D Avg': sector_df['5D Return'].mean() * 100 if '5D Return' in sector_df else 0,
                            'Optimal Days': best_days,
                            'Sharpe': best_sharpe
                        })
                
                sector_df = pd.DataFrame(sector_stats).sort_values('Sharpe', ascending=True)
                
                fig = px.bar(sector_df, y='Sector', x='Sharpe', orientation='h',
                            title='Sector Performance (Sharpe Ratio)',
                            color='Sharpe', color_continuous_scale='YlOrRd')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e2e8f0',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Sector recommendations
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success("**Hold Longer (7-10 days)**\n\n• Financial Services\n• Real Estate\n• Basic Materials")
                with col2:
                    st.warning("**Standard Hold (5 days)**\n\n• Energy\n• Consumer Cyclical\n• Healthcare")
                with col3:
                    st.error("**Exit Early (1-3 days)**\n\n• Technology\n• Communication Services\n• Industrials")
            else:
                st.info("Sector data not available in returns file.")
        
        with analysis_tab4:
            st.markdown("#### Risk Management Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Stop-Loss Analysis")
                
                if '1W Low Return' in returns_df.columns:
                    stop_data = []
                    for stop in [0.05, 0.08, 0.10, 0.15]:
                        stopped = returns_df[returns_df['1W Low Return'] < -stop]
                        pct = len(stopped) / len(returns_df) * 100
                        avg_7d = stopped['7D Return'].dropna().mean() * 100 if len(stopped) > 0 and '7D Return' in stopped else 0
                        stop_data.append({
                            'Stop Level': f"-{int(stop*100)}%",
                            'Triggered %': pct,
                            'Avg 7D if Held': avg_7d,
                            'Optimal': stop == 0.08
                        })
                    
                    stop_df = pd.DataFrame(stop_data)
                    colors = ['#f59e0b' if x else '#ef4444' for x in stop_df['Optimal']]
                    
                    fig = px.bar(stop_df, x='Stop Level', y='Triggered %',
                                title='Stop-Loss Trigger Rate')
                    fig.update_traces(marker_color=colors)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e2e8f0',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**-8% Recommended:** Only 24.7% triggered, protects from avg -12.7% loss")
                else:
                    st.info("Stop-loss data not available.")
            
            with col2:
                st.markdown("##### Profit Target Analysis")
                
                if '1W High Return' in returns_df.columns:
                    target_data = []
                    for target in [0.05, 0.08, 0.10, 0.15, 0.20]:
                        hit = returns_df[returns_df['1W High Return'] >= target]
                        pct = len(hit) / len(returns_df) * 100
                        target_data.append({
                            'Target': f"+{int(target*100)}%",
                            'Hit Rate %': pct,
                            'Optimal': target == 0.10
                        })
                    
                    target_df = pd.DataFrame(target_data)
                    colors = ['#f59e0b' if x else '#10b981' for x in target_df['Optimal']]
                    
                    fig = px.bar(target_df, x='Target', y='Hit Rate %',
                                title='Profit Target Hit Rate (within 7 days)')
                    fig.update_traces(marker_color=colors)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e2e8f0',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**+10% Recommended:** 38% hit rate, good risk/reward balance")
                else:
                    st.info("Profit target data not available.")
            
            # Summary stats
            st.markdown("---")
            st.markdown("#### Risk/Reward Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            if '1W High Return' in returns_df.columns and '1W Low Return' in returns_df.columns:
                avg_high = returns_df['1W High Return'].mean() * 100
                avg_low = returns_df['1W Low Return'].mean() * 100
                avg_range = (returns_df['1W High Return'] - returns_df['1W Low Return']).mean() * 100
                win_loss = abs(stats['5 Days']['avg_win'] / stats['5 Days']['avg_loss']) if stats['5 Days']['avg_loss'] != 0 else 0
                
                col1.metric("Avg 7D High", f"+{avg_high:.1f}%")
                col2.metric("Avg 7D Low", f"{avg_low:.1f}%")
                col3.metric("Avg 7D Range", f"{avg_range:.1f}%")
                col4.metric("Win/Loss Ratio", f"{win_loss:.2f}x")

# =============================================================================
# TAB 3: POWERBI DASHBOARD EMBED
# =============================================================================
with tab3:
    st.markdown("### 📈 PowerBI Dashboard")
    st.caption("Interactive returns tracking and performance analytics")
    
    # Embed the PowerBI dashboard directly
    st.markdown("""
    <iframe 
        title="Finance Models" 
        width="100%" 
        height="800" 
        src="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" 
        frameborder="0" 
        allowFullScreen="true"
        style="border: 1px solid #334155; border-radius: 8px;">
    </iframe>
    """, unsafe_allow_html=True)
    
    # Fullscreen option
    st.markdown("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(
            '<a href="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" target="_blank">'
            '<button style="background:#f59e0b; color:black; padding:10px 20px; border:none; border-radius:6px; cursor:pointer; font-weight:600;">Open Fullscreen ↗</button>'
            '</a>', 
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
