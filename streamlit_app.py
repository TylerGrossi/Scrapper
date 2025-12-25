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

# --- Clean, Professional CSS ---
st.markdown("""
<style>
    .block-container {
        max-width: 95% !important;
        padding: 2rem 3rem;
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
    
    /* Rule cards */
    .rule-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .rule-number {
        display: inline-block;
        background: #475569;
        color: #f1f5f9;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .rule-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.25rem;
    }
    .rule-desc {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    /* Strategy highlight */
    .strategy-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_earnings_date(earn_str):
    try:
        parts = (earn_str or "").split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except:
        pass
    return datetime.max

def get_all_tickers():
    base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    offset, tickers = 0, []
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
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    data = {"Ticker": ticker, "Earnings": "N/A", "Price": "N/A", "P/E": "N/A", "Beta": "N/A", "Market Cap": "N/A"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup.select("table.snapshot-table2"):
            tds = t.find_all("td")
            for i in range(0, len(tds) - 1, 2):
                k, v = tds[i].get_text(strip=True), tds[i + 1].get_text(strip=True)
                if k in data and data[k] == "N/A":
                    data[k] = v
    except:
        pass
    return data

def has_buy_signal(ticker):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        sig = soup.find("span", class_="opinion-signal buy")
        return bool(sig and "Buy" in sig.text)
    except:
        return False

def earnings_sort_key(row):
    date = parse_earnings_date(row["Earnings"])
    earn_str = (row["Earnings"] or "").upper()
    am_pm_rank = 0 if "BMO" in earn_str else 1 if "AMC" in earn_str else 2
    return (date, am_pm_rank)

@st.cache_data(ttl=3600)
def load_returns_data():
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/returns_tracker.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/returns_tracker.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and '1D Return' in df.columns:
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('returns_tracker.csv')
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        return None

def calc_period_stats(df, col):
    valid = df[col].dropna()
    if len(valid) == 0:
        return {'mean': 0, 'median': 0, 'win_rate': 0, 'sharpe': 0, 'n': 0}
    return {
        'mean': valid.mean(),
        'median': valid.median(),
        'win_rate': (valid > 0).sum() / len(valid),
        'sharpe': valid.mean() / valid.std() if valid.std() > 0 else 0,
        'n': len(valid)
    }

def backtest_strategy(df, stop_loss=-0.08, profit_target=None, max_days=5):
    """Backtest a strategy and return results"""
    results = []
    
    day_col = '5D Return' if max_days == 5 else '7D Return' if max_days == 7 else '10D Return'
    
    for idx, row in df.iterrows():
        high = row.get('1W High Return', np.nan)
        low = row.get('1W Low Return', np.nan)
        exit_return = row.get(day_col, np.nan)
        
        if pd.isna(low) or pd.isna(exit_return):
            continue
        
        # Determine exit
        if low <= stop_loss:
            final_return = stop_loss
            exit_reason = 'Stop Loss'
        elif profit_target and pd.notna(high) and high >= profit_target:
            final_return = profit_target
            exit_reason = 'Profit Target'
        else:
            final_return = exit_return
            exit_reason = f'Day {max_days} Exit'
        
        results.append({
            'Ticker': row.get('Ticker', ''),
            'Date': row.get('Earnings Date', ''),
            'Exit': exit_reason,
            'Return': final_return,
            'High': high if pd.notna(high) else 0,
            'Low': low if pd.notna(low) else 0,
        })
    
    return pd.DataFrame(results)

# =============================================================================
# MAIN APP
# =============================================================================

st.title("Earnings Momentum Strategy")

tab1, tab2, tab3, tab4 = st.tabs(["Stock Screener", "Exit Strategy", "Backtest", "PowerBI"])

# =============================================================================
# TAB 1: STOCK SCREENER
# =============================================================================
with tab1:
    st.markdown("**Criteria:** Earnings this week · SMA20 crossed above SMA50 · Barchart Buy Signal")
    
    if st.button("Find Stocks"):
        with st.spinner("Scanning..."):
            tickers = get_all_tickers()
        
        rows = []
        progress = st.progress(0)
        for i, t in enumerate(tickers):
            if has_buy_signal(t):
                data = get_finviz_data(t)
                rows.append(data)
            progress.progress((i + 1) / len(tickers))
        progress.empty()
        
        rows = sorted(rows, key=earnings_sort_key)
        
        if not rows:
            st.info("No tickers match criteria.")
        else:
            st.dataframe(
                pd.DataFrame(rows)[["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap"]],
                use_container_width=True, hide_index=True
            )
    else:
        st.caption("Click Find Stocks to scan.")

# =============================================================================
# TAB 2: EXIT STRATEGY
# =============================================================================
with tab2:
    returns_df = load_returns_data()
    
    if returns_df is None:
        st.warning("Returns data not found. Upload returns_tracker.csv to your GitHub repo.")
        uploaded = st.file_uploader("Or upload here:", type=['csv'])
        if uploaded:
            returns_df = pd.read_csv(uploaded)
            returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date'], errors='coerce')
    
    if returns_df is not None and not returns_df.empty:
        stats = {name: calc_period_stats(returns_df, col) for col, name in 
                 [('1D Return', '1D'), ('3D Return', '3D'), ('5D Return', '5D'), ('7D Return', '7D')]}
        
        # Strategy Summary
        st.markdown("### The Strategy")
        
        st.markdown("""
        <div class="strategy-banner">
            <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 0.5rem;">Based on {} trades</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; color: #f1f5f9;">
                <div>
                    <div style="font-size: 2rem; font-weight: 700; color: #ef4444;">-8%</div>
                    <div style="font-size: 0.9rem; color: #94a3b8;">Stop Loss</div>
                </div>
                <div>
                    <div style="font-size: 2rem; font-weight: 700; color: #22c55e;">Day 5</div>
                    <div style="font-size: 0.9rem; color: #94a3b8;">Exit</div>
                </div>
                <div>
                    <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">No Cap</div>
                    <div style="font-size: 0.9rem; color: #94a3b8;">Let winners run</div>
                </div>
            </div>
        </div>
        """.format(len(returns_df)), unsafe_allow_html=True)
        
        # Why no profit target
        with st.expander("Why no profit target?"):
            st.markdown("""
            Fixed profit targets hurt returns. The data:
            
            | Strategy | Avg Return |
            |----------|-----------|
            | -8% stop, **10% target**, Day 5 | +2.29% |
            | -8% stop, **no target**, Day 5 | +6.10% |
            
            Trades hitting +10% averaged **+19.5% by Day 7**. Selling at 10% leaves money on the table.
            """)
        
        st.markdown("---")
        
        # The Rules
        st.markdown("### Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="rule-card">
                <span class="rule-number">1</span>
                <div class="rule-title">Set -8% stop loss on entry</div>
                <div class="rule-desc">Always. No exceptions.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="rule-card">
                <span class="rule-number">2</span>
                <div class="rule-title">Exit Day 5 at close</div>
                <div class="rule-desc">Returns go negative after Day 5.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="rule-card">
                <span class="rule-number">3</span>
                <div class="rule-title">Let winners run</div>
                <div class="rule-desc">No profit cap. Big winners go +20-40%.</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="rule-card">
                <span class="rule-number">4</span>
                <div class="rule-title">Optional: Trail stop after +10%</div>
                <div class="rule-desc">Move stop to +5% to protect gains.</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performance
        st.markdown("### Performance by Hold Period")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1 Day", f"{stats['1D']['mean']*100:+.1f}%", f"{stats['1D']['win_rate']*100:.0f}% win")
        col2.metric("3 Days", f"{stats['3D']['mean']*100:+.1f}%", f"{stats['3D']['win_rate']*100:.0f}% win")
        col3.metric("5 Days", f"{stats['5D']['mean']*100:+.1f}%", f"{stats['5D']['win_rate']*100:.0f}% win")
        col4.metric("7 Days", f"{stats['7D']['mean']*100:+.1f}%", f"{stats['7D']['win_rate']*100:.0f}% win")
        
        # Charts
        st.markdown("### Why Day 5?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Marginal returns
            cols = ['1D Return', '3D Return', '5D Return', '7D Return', '10D Return']
            valid = returns_df[cols].dropna()
            prev = 0
            marginal = []
            for i, col in enumerate(cols):
                curr = valid[col].mean() * 100
                marginal.append({'Period': ['D1', 'D2-3', 'D4-5', 'D6-7', 'D8-10'][i], 
                                'Return': curr - prev if i > 0 else curr})
                prev = curr
            
            mdf = pd.DataFrame(marginal)
            colors = ['#22c55e' if x > 0 else '#ef4444' for x in mdf['Return']]
            
            fig = px.bar(mdf, x='Period', y='Return', title='Marginal Return by Period')
            fig.update_traces(marker_color=colors)
            fig.add_hline(y=0, line_dash="dash", line_color="#475569")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8', height=300, showlegend=False,
                yaxis_title='%', xaxis_title=''
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Big winner analysis
            if '1W High Return' in returns_df.columns:
                st.markdown("##### Big Winners (hit +20%)")
                big = returns_df[returns_df['1W High Return'] >= 0.20]
                st.markdown(f"""
                {len(big)} trades ({len(big)/len(returns_df)*100:.0f}% of total)
                
                - Avg Day 5: **+{big['5D Return'].mean()*100:.1f}%**
                - Avg peak: +{big['1W High Return'].mean()*100:.1f}%
                
                A 10% target would capture ~1/3 of gains.
                """)

# =============================================================================
# TAB 3: BACKTEST
# =============================================================================
with tab3:
    returns_df = load_returns_data()
    
    if returns_df is None:
        st.warning("Upload returns_tracker.csv to run backtests.")
    else:
        st.markdown("### Backtest")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            stop_loss = st.slider("Stop Loss", -20, -5, -8, 1, format="%d%%") / 100
        with col2:
            use_target = st.checkbox("Use Profit Target")
            profit_target = st.slider("Target", 5, 30, 10, 1, format="%d%%", disabled=not use_target) / 100 if use_target else None
        with col3:
            max_days = st.selectbox("Hold Days", [5, 7, 10], index=0)
        
        if st.button("Run Backtest"):
            results = backtest_strategy(returns_df, stop_loss, profit_target, max_days)
            
            if not results.empty:
                total_return = results['Return'].sum() * 100
                avg_return = results['Return'].mean() * 100
                win_rate = (results['Return'] > 0).mean() * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{total_return:+.1f}%")
                col2.metric("Avg/Trade", f"{avg_return:+.2f}%")
                col3.metric("Win Rate", f"{win_rate:.1f}%")
                col4.metric("Trades", len(results))
                
                # Exit breakdown
                st.markdown("#### Exits")
                exit_stats = results.groupby('Exit').agg({
                    'Return': ['count', 'mean']
                }).round(4)
                exit_stats.columns = ['Count', 'Avg Return']
                exit_stats['Avg Return'] = exit_stats['Avg Return'].apply(lambda x: f"{x*100:+.2f}%")
                st.dataframe(exit_stats, use_container_width=True)
                
                # Trade list
                st.markdown("#### Trades")
                display_df = results.copy()
                display_df['Return'] = display_df['Return'].apply(lambda x: f"{x*100:+.2f}%")
                display_df['High'] = display_df['High'].apply(lambda x: f"{x*100:+.1f}%")
                display_df['Low'] = display_df['Low'].apply(lambda x: f"{x*100:+.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
        
        st.markdown("---")
        st.markdown("### Compare Strategies")
        
        if st.button("Run Comparison"):
            strategies = [
                ("No target, -8% stop, 5D", -0.08, None, 5),
                ("10% target, -8% stop, 5D", -0.08, 0.10, 5),
                ("15% target, -8% stop, 5D", -0.08, 0.15, 5),
                ("No target, -5% stop, 5D", -0.05, None, 5),
                ("No target, -10% stop, 5D", -0.10, None, 5),
            ]
            
            comparison = []
            for name, stop, target, days in strategies:
                res = backtest_strategy(returns_df, stop, target, days)
                if not res.empty:
                    comparison.append({
                        'Strategy': name,
                        'Avg': f"{res['Return'].mean()*100:+.2f}%",
                        'Win%': f"{(res['Return'] > 0).mean()*100:.0f}%",
                        'Total': f"{res['Return'].sum()*100:+.0f}%",
                    })
            
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
            st.info("Best: **No target, -8% stop, Day 5** — lets winners run.")

# =============================================================================
# TAB 4: POWERBI
# =============================================================================
with tab4:
    st.markdown("### PowerBI Dashboard")
    
    view = st.radio("", ["Embedded", "New Tab"], horizontal=True, label_visibility="collapsed")
    
    if view == "Embedded":
        st.markdown("""
        <div style="background: #0f172a; border-radius: 8px; padding: 8px; border: 1px solid #334155; overflow-x: auto;">
            <iframe 
                title="Finance Models" 
                width="1400" height="850"
                src="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" 
                frameborder="0" allowFullScreen="true">
            </iframe>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #1e293b; border-radius: 8px;">
            <a href="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" target="_blank">
                <button style="background: #3b82f6; color: white; border: none; padding: 12px 32px; border-radius: 6px; font-weight: 600; cursor: pointer;">
                    Open Dashboard →
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Earnings Momentum Strategy")
