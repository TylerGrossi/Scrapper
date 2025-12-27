import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
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
    
    /* Parameter panel */
    .param-panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .param-title {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #334155;
    }
    .section-icon {
        font-size: 1.5rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    
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
    
    /* Comparison table styling */
    .best-row {
        background: rgba(34, 197, 94, 0.1) !important;
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


# =============================================================================
# DATE CHECK FUNCTIONS
# =============================================================================

def get_yfinance_earnings_date(ticker):
    try:
        earnings_df = yf.Ticker(ticker).get_earnings_dates(limit=10)
        if earnings_df is None or earnings_df.empty:
            return None
        today = datetime.today().date()
        best_date = None
        min_diff = 999
        for idx in earnings_df.index:
            try:
                idx_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                diff = abs((idx_date - today).days)
                if diff < min_diff and diff <= 60:
                    min_diff = diff
                    best_date = datetime.combine(idx_date, datetime.min.time())
            except:
                continue
        return best_date
    except:
        return None

def get_finviz_earnings_date(ticker):
    try:
        info = yf.Ticker(ticker).info
        ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
        if ts:
            return datetime.fromtimestamp(ts)
    except:
        pass
    return None

def check_date_status(earnings_date, yfinance_date):
    try:
        if earnings_date is None or yfinance_date is None:
            return "OK"
        ed_date = earnings_date.date() if hasattr(earnings_date, 'date') else earnings_date
        yf_date = yfinance_date.date() if hasattr(yfinance_date, 'date') else yfinance_date
        date_diff = abs((ed_date - yf_date).days)
        if date_diff > 14:
            return "OK"
        if yf_date < ed_date:
            return "DATE PASSED"
        return "OK"
    except:
        return "OK"

def get_date_check(ticker):
    finviz_date = get_finviz_earnings_date(ticker)
    yfinance_date = get_yfinance_earnings_date(ticker)
    status = check_date_status(finviz_date, yfinance_date)
    return {
        "Earnings Date (Finviz)": finviz_date.strftime("%Y-%m-%d") if finviz_date else "N/A",
        "Earnings Date (yfinance)": yfinance_date.strftime("%Y-%m-%d") if yfinance_date else "N/A",
        "Date Check": status
    }

def earnings_sort_key(row):
    date = parse_earnings_date(row["Earnings"])
    earn_str = (row["Earnings"] or "").upper()
    am_pm_rank = 0 if "BMO" in earn_str else 1 if "AMC" in earn_str else 2
    return (date, am_pm_rank)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

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

@st.cache_data(ttl=3600)
def load_daily_prices():
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/daily_prices.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/daily_prices.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and 'Days From Earnings' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('daily_prices.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
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


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def backtest_with_daily_prices(daily_df, stop_loss=-0.05, profit_target=None, max_days=5):
    results = []
    trades = daily_df.groupby(['Ticker', 'Earnings Date', 'Fiscal Quarter']).size().reset_index()[['Ticker', 'Earnings Date', 'Fiscal Quarter']]
    
    for _, trade in trades.iterrows():
        ticker = trade['Ticker']
        earnings_date = trade['Earnings Date']
        fiscal_quarter = trade['Fiscal Quarter']
        
        trade_data = daily_df[
            (daily_df['Ticker'] == ticker) & 
            (daily_df['Earnings Date'] == earnings_date) &
            (daily_df['Days From Earnings'] >= 0) &
            (daily_df['Days From Earnings'] <= max_days)
        ].sort_values('Days From Earnings')
        
        if trade_data.empty:
            continue
        
        company_name = trade_data['Company Name'].iloc[0] if 'Company Name' in trade_data.columns else ticker
        
        exit_day = None
        exit_reason = None
        exit_return = None
        max_return = 0
        min_return = 0
        
        for _, day in trade_data.iterrows():
            day_num = int(day['Days From Earnings'])
            day_return = day['Return From Earnings (%)']
            
            if pd.isna(day_return):
                continue
            
            day_return_decimal = day_return / 100
            max_return = max(max_return, day_return_decimal)
            min_return = min(min_return, day_return_decimal)
            
            if day_return_decimal <= stop_loss:
                exit_day = day_num
                exit_reason = 'Stop Loss'
                exit_return = stop_loss
                break
            
            if profit_target and day_return_decimal >= profit_target:
                exit_day = day_num
                exit_reason = 'Profit Target'
                exit_return = profit_target
                break
        
        if exit_return is None:
            last_day = trade_data[trade_data['Days From Earnings'] == trade_data['Days From Earnings'].max()]
            if not last_day.empty:
                final_return = last_day['Return From Earnings (%)'].iloc[0]
                if pd.notna(final_return):
                    exit_day = int(last_day['Days From Earnings'].iloc[0])
                    exit_reason = f'Day {exit_day} Exit'
                    exit_return = final_return / 100
        
        if exit_return is not None:
            results.append({
                'Ticker': ticker,
                'Company': company_name,
                'Earnings Date': earnings_date,
                'Fiscal Quarter': fiscal_quarter,
                'Exit Day': exit_day,
                'Exit Reason': exit_reason,
                'Return': exit_return,
                'Max Return': max_return,
                'Min Return': min_return,
            })
    
    return pd.DataFrame(results)

def backtest_strategy_legacy(df, stop_loss=-0.08, profit_target=None, max_days=5):
    results = []
    day_col = '5D Return' if max_days == 5 else '7D Return' if max_days == 7 else '10D Return'
    
    for idx, row in df.iterrows():
        high = row.get('1W High Return', np.nan)
        low = row.get('1W Low Return', np.nan)
        exit_return = row.get(day_col, np.nan)
        
        if pd.isna(low) or pd.isna(exit_return):
            continue
        
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
            'Company': row.get('Company Name', row.get('Ticker', '')),
            'Earnings Date': row.get('Earnings Date', ''),
            'Exit Day': max_days,
            'Exit Reason': exit_reason,
            'Return': final_return,
            'Max Return': high if pd.notna(high) else 0,
            'Min Return': low if pd.notna(low) else 0,
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
        with st.spinner("Scanning Finviz..."):
            tickers = get_all_tickers()
        
        st.info(f"Found {len(tickers)} tickers from Finviz. Checking Barchart signals...")
        
        barchart_passed = []
        progress = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(tickers):
            status_text.text(f"Checking Barchart: {t}")
            if has_buy_signal(t):
                barchart_passed.append(t)
            progress.progress((i + 1) / len(tickers))
        
        progress.empty()
        status_text.empty()
        
        st.info(f"{len(barchart_passed)} tickers passed Barchart filter. Checking earnings dates...")
        
        rows = []
        skipped = []
        progress = st.progress(0)
        status_text = st.empty()
        
        for i, t in enumerate(barchart_passed):
            status_text.text(f"Checking dates: {t}")
            data = get_finviz_data(t)
            date_info = get_date_check(t)
            
            if date_info["Date Check"] == "DATE PASSED":
                skipped.append({
                    "Ticker": t,
                    "Finviz Date": date_info["Earnings Date (Finviz)"],
                    "yfinance Date": date_info["Earnings Date (yfinance)"],
                    "Reason": "DATE PASSED"
                })
            else:
                data["Date Check"] = date_info["Date Check"]
                rows.append(data)
            
            progress.progress((i + 1) / len(barchart_passed))
        
        progress.empty()
        status_text.empty()
        
        rows = sorted(rows, key=earnings_sort_key)
        
        if not rows:
            st.warning("No tickers match all criteria.")
        else:
            st.success(f"✅ {len(rows)} tickers ready to trade")
            st.dataframe(
                pd.DataFrame(rows)[["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap", "Date Check"]],
                use_container_width=True, hide_index=True
            )
        
        if skipped:
            with st.expander(f"⚠️ {len(skipped)} tickers skipped (earnings already passed)"):
                st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
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
        period_cols = [('1D Return', '1 Day'), ('3D Return', '3 Days'), ('5D Return', '5 Days'), 
                       ('7D Return', '7 Days'), ('10D Return', '10 Days')]
        stats = {name: calc_period_stats(returns_df, col) for col, name in period_cols}
        
        st.markdown("### The Strategy")
        
        st.markdown("""
        <div class="strategy-banner">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">Based on {} trades</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; color: #f1f5f9;">
                <div>
                    <div style="font-size: 2rem; font-weight: 700; color: #ef4444;">-5%</div>
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
        
        st.markdown("### Rules")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="rule-card">
                <span class="rule-number">1</span>
                <div class="rule-title">Set -5% stop loss on entry</div>
                <div class="rule-desc">Triggers on 33% of trades. Best risk-adjusted returns.</div>
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
                <div class="rule-title">Optional: Move stop to breakeven after +5%</div>
                <div class="rule-desc">Protects gains on runners.</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Why These Rules Work (The Data)")
        
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "Holding Period", "Momentum Decay", "Sector Rules", "Risk Analysis"
        ])
        
        with analysis_tab1:
            st.markdown("#### Returns by Holding Period")
            st.caption("5-day hold has the best risk-adjusted returns (Sharpe ratio)")
            
            col1, col2 = st.columns([1.2, 1])
            
            with col1:
                sharpe_data = pd.DataFrame([
                    {'Period': name, 'Sharpe': stats[name]['sharpe']}
                    for name in ['1 Day', '3 Days', '5 Days', '7 Days', '10 Days']
                ])
                colors = ['#f59e0b' if p == '5 Days' else '#64748b' for p in sharpe_data['Period']]
                
                fig = px.bar(sharpe_data, x='Period', y='Sharpe', title='Sharpe Ratio by Holding Period')
                fig.update_traces(marker_color=colors)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=350, showlegend=False
                )
                fig.update_xaxes(gridcolor='#1e293b')
                fig.update_yaxes(gridcolor='#1e293b')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                perf_data = []
                for name in ['1 Day', '3 Days', '5 Days', '7 Days', '10 Days']:
                    s = stats[name]
                    perf_data.append({
                        'Period': name,
                        'Avg Return': f"{s['mean']*100:+.2f}%",
                        'Win Rate': f"{s['win_rate']*100:.1f}%",
                        'Sharpe': f"{s['sharpe']:.3f}"
                    })
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
                
                st.success(f"""
                **Best Period: 5 Days**
                - Win Rate: {stats['5 Days']['win_rate']*100:.1f}%
                - Avg Return: {stats['5 Days']['mean']*100:+.2f}%
                - Sharpe: {stats['5 Days']['sharpe']:.3f}
                """)
        
        with analysis_tab2:
            st.markdown("#### Why Exit by Day 5?")
            st.caption("Marginal returns turn NEGATIVE after Day 5")
            
            cols = ['1D Return', '3D Return', '5D Return', '7D Return', '10D Return']
            valid = returns_df[cols].dropna()
            prev = 0
            marginal_data = []
            period_names = ['Day 1', 'Days 2-3', 'Days 4-5', 'Days 6-7', 'Days 8-10']
            
            for i, col in enumerate(cols):
                curr = valid[col].mean() * 100
                marg = curr if i == 0 else curr - prev
                marginal_data.append({'Period': period_names[i], 'Marginal': marg, 'Cumulative': curr})
                prev = curr
            
            marginal_df = pd.DataFrame(marginal_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                colors = ['#22c55e' if x > 0 else '#ef4444' for x in marginal_df['Marginal']]
                fig = px.bar(marginal_df, x='Period', y='Marginal', title='Additional Return Each Period')
                fig.update_traces(marker_color=colors)
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8', height=350, yaxis_title='Marginal Return %'
                )
                fig.update_xaxes(gridcolor='#1e293b')
                fig.update_yaxes(gridcolor='#1e293b')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### The Decay Pattern")
                for _, row in marginal_df.iterrows():
                    color = '#22c55e' if row['Marginal'] > 0 else '#ef4444'
                    st.markdown(f"**{row['Period']}:** <span style='color:{color}'>{row['Marginal']:+.2f}%</span>", 
                               unsafe_allow_html=True)
                st.markdown("")
                st.warning("After Day 5, you're **losing money** on average by holding.")
        
        with analysis_tab3:
            st.markdown("#### Sector Performance")
            if 'Sector' in returns_df.columns:
                sector_data = []
                for sector in returns_df['Sector'].dropna().unique():
                    sdf = returns_df[returns_df['Sector'] == sector]
                    if len(sdf) >= 3:
                        best_sharpe, best_days = -999, 5
                        for col, days in [('1D Return', 1), ('3D Return', 3), ('5D Return', 5), 
                                         ('7D Return', 7), ('10D Return', 10)]:
                            if col in sdf.columns:
                                v = sdf[col].dropna()
                                if len(v) >= 3 and v.std() > 0:
                                    sharpe = v.mean() / v.std()
                                    if sharpe > best_sharpe:
                                        best_sharpe, best_days = sharpe, days
                        sector_data.append({
                            'Sector': sector, 'Trades': len(sdf),
                            'Avg 5D': sdf['5D Return'].mean() * 100,
                            'Best Hold': f"{best_days}D", 'Sharpe': best_sharpe
                        })
                
                sector_df = pd.DataFrame(sector_data).sort_values('Sharpe', ascending=False)
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    fig = px.bar(sector_df.sort_values('Sharpe'), y='Sector', x='Sharpe', 
                                orientation='h', title='Sector Sharpe Ratio',
                                color='Sharpe', color_continuous_scale='RdYlGn')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8', height=400, showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### Recommendations by Sector")
                    early = sector_df[sector_df['Best Hold'].isin(['1D', '3D'])]['Sector'].tolist()
                    standard = sector_df[sector_df['Best Hold'] == '5D']['Sector'].tolist()
                    longer = sector_df[sector_df['Best Hold'].isin(['7D', '10D'])]['Sector'].tolist()
                    
                    if early:
                        st.markdown(f"**Exit Early (1-3D):** {', '.join(early[:3])}")
                    if standard:
                        st.markdown(f"**Standard (5D):** {', '.join(standard[:3])}")
                    if longer:
                        st.markdown(f"**Hold Longer (7-10D):** {', '.join(longer[:3])}")
            else:
                st.info("Sector data not available.")
        
        with analysis_tab4:
            st.markdown("#### Stop Loss Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                if '1W Low Return' in returns_df.columns:
                    stop_data = []
                    for stop in [-0.03, -0.05, -0.08, -0.10, -0.15]:
                        stopped = returns_df[returns_df['1W Low Return'] <= stop]
                        not_stopped = returns_df[returns_df['1W Low Return'] > stop]
                        n_stopped, n_not = len(stopped), len(not_stopped)
                        
                        if n_stopped + n_not > 0:
                            stopped_ret = stop * n_stopped
                            not_stopped_ret = not_stopped['5D Return'].sum() if len(not_stopped) > 0 else 0
                            avg_ret = (stopped_ret + not_stopped_ret) / (n_stopped + n_not)
                            stop_data.append({
                                'Stop': f"{stop*100:.0f}%",
                                'Triggered': f"{n_stopped/len(returns_df)*100:.1f}%",
                                'Avg Return': f"{avg_ret*100:+.2f}%",
                            })
                    st.dataframe(pd.DataFrame(stop_data), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("##### Big Winners")
                if '1W High Return' in returns_df.columns:
                    for thresh in [0.10, 0.20, 0.30]:
                        big = returns_df[returns_df['1W High Return'] >= thresh]
                        if len(big) > 0:
                            st.markdown(f"**+{thresh*100:.0f}%:** {len(big)} trades → Avg Day 5: +{big['5D Return'].mean()*100:.1f}%")

# =============================================================================
# TAB 3: BACKTEST (Professional UI)
# =============================================================================
with tab3:
    # Load data
    daily_df = load_daily_prices()
    returns_df = load_returns_data()
    has_daily = daily_df is not None and not daily_df.empty
    has_returns = returns_df is not None and not returns_df.empty
    
    # Header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #f1f5f9;">Strategy Backtest</h2>
        <p style="color: #64748b; margin-top: 0.25rem;">Test stop loss and profit target combinations using historical daily price data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not has_daily and not has_returns:
        st.warning("No data available. Upload daily_prices.csv or returns_tracker.csv to your GitHub repo.")
        col1, col2 = st.columns(2)
        with col1:
            daily_upload = st.file_uploader("Upload daily_prices.csv:", type=['csv'], key='daily')
            if daily_upload:
                daily_df = pd.read_csv(daily_upload)
                daily_df['Date'] = pd.to_datetime(daily_df['Date'], errors='coerce')
                daily_df['Earnings Date'] = pd.to_datetime(daily_df['Earnings Date'], errors='coerce')
                has_daily = True
        with col2:
            returns_upload = st.file_uploader("Upload returns_tracker.csv:", type=['csv'], key='returns')
            if returns_upload:
                returns_df = pd.read_csv(returns_upload)
                returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date'], errors='coerce')
                has_returns = True
    
    if has_daily or has_returns:
        # Data status indicator
        if has_daily:
            n_trades = daily_df.groupby(['Ticker', 'Earnings Date']).ngroups
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.5rem;">
                <span style="color: #22c55e; font-weight: 600;">✓ Daily Prices Loaded</span>
                <span style="color: #94a3b8; margin-left: 1rem;">{len(daily_df):,} price points · {n_trades} trades</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(251, 191, 36, 0.1); border: 1px solid #fbbf24; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.5rem;">
                <span style="color: #fbbf24; font-weight: 600;">⚠ Using Returns Tracker</span>
                <span style="color: #94a3b8; margin-left: 1rem;">Less accurate - no daily stop simulation</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Parameters Panel
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">⚙️</span>
            <span class="section-title">Parameters</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="param-title">Stop Loss</div>', unsafe_allow_html=True)
            stop_loss = st.select_slider(
                "Stop Loss",
                options=[-0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10, -0.12, -0.15, -0.20],
                value=-0.05,
                format_func=lambda x: f"{x*100:.0f}%",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown('<div class="param-title">Profit Target</div>', unsafe_allow_html=True)
            use_target = st.checkbox("Enable Target", value=False)
        
        with col3:
            if use_target:
                st.markdown('<div class="param-title">Target Level</div>', unsafe_allow_html=True)
                profit_target = st.select_slider(
                    "Profit Target",
                    options=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
                    value=0.10,
                    format_func=lambda x: f"+{x*100:.0f}%",
                    label_visibility="collapsed"
                )
            else:
                profit_target = None
                st.markdown('<div class="param-title">Target Level</div>', unsafe_allow_html=True)
                st.markdown('<div style="color: #64748b; padding: 0.5rem 0;">No target (let winners run)</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="param-title">Max Hold Days</div>', unsafe_allow_html=True)
            max_days = st.selectbox("Max Days", [3, 5, 7, 10], index=1, label_visibility="collapsed")
        
        # Run Button
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        
        if run_backtest:
            with st.spinner("Running backtest..."):
                if has_daily:
                    results = backtest_with_daily_prices(daily_df, stop_loss, profit_target, max_days)
                else:
                    results = backtest_strategy_legacy(returns_df, stop_loss, profit_target, max_days)
            
            if results.empty:
                st.warning("No trades found in the data.")
            else:
                # Calculate metrics
                total_return = results['Return'].sum() * 100
                avg_return = results['Return'].mean() * 100
                win_rate = (results['Return'] > 0).mean() * 100
                n_trades = len(results)
                max_win = results['Return'].max() * 100
                max_loss = results['Return'].min() * 100
                
                if results['Return'].std() > 0:
                    sharpe = (results['Return'].mean() / results['Return'].std()) * np.sqrt(52)
                else:
                    sharpe = 0
                
                # Results Header
                st.markdown("""
                <div class="section-header">
                    <span class="section-icon">📊</span>
                    <span class="section-title">Results</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Metric Cards
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    color_class = "metric-green" if total_return > 0 else "metric-red"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value {color_class}">{total_return:+.1f}%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    color_class = "metric-green" if avg_return > 0 else "metric-red"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value {color_class}">{avg_return:+.2f}%</div>
                        <div class="metric-label">Avg per Trade</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    color_class = "metric-green" if win_rate >= 50 else "metric-yellow"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value {color_class}">{win_rate:.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value metric-blue">{sharpe:.2f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value metric-green">{max_win:+.1f}%</div>
                        <div class="metric-label">Best Trade</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value metric-red">{max_loss:+.1f}%</div>
                        <div class="metric-label">Worst Trade</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
                
                # Exit Breakdown
                st.markdown("""
                <div class="section-header">
                    <span class="section-icon">🚪</span>
                    <span class="section-title">Exit Analysis</span>
                </div>
                """, unsafe_allow_html=True)
                
                exit_groups = results.groupby('Exit Reason').agg({
                    'Return': ['count', 'mean', 'sum']
                }).reset_index()
                exit_groups.columns = ['Exit Reason', 'Count', 'Avg Return', 'Total Return']
                
                cols = st.columns(len(exit_groups))
                for i, (_, row) in enumerate(exit_groups.iterrows()):
                    with cols[i]:
                        pct = row['Count'] / n_trades * 100
                        avg_ret = row['Avg Return'] * 100
                        color = "#22c55e" if avg_ret > 0 else "#ef4444"
                        
                        # Determine icon
                        if 'Stop' in row['Exit Reason']:
                            icon = "🛑"
                        elif 'Target' in row['Exit Reason']:
                            icon = "🎯"
                        else:
                            icon = "📅"
                        
                        st.markdown(f"""
                        <div class="exit-card">
                            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                            <div class="exit-count">{row['Count']}</div>
                            <div class="exit-pct">{pct:.1f}% of trades</div>
                            <div class="exit-return" style="color: {color};">{avg_ret:+.2f}% avg</div>
                            <div class="exit-label">{row['Exit Reason']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    
                    # Create histogram with custom colors
                    returns_pct = results['Return'] * 100
                    colors = ['#22c55e' if x > 0 else '#ef4444' for x in returns_pct]
                    
                    fig.add_trace(go.Histogram(
                        x=returns_pct,
                        nbinsx=25,
                        marker_color='#3b82f6',
                        opacity=0.8
                    ))
                    
                    # Add reference lines
                    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", line_width=1)
                    fig.add_vline(x=stop_loss*100, line_dash="dash", line_color="#ef4444", line_width=2,
                                  annotation_text=f"Stop {stop_loss*100:.0f}%", annotation_position="top")
                    if profit_target:
                        fig.add_vline(x=profit_target*100, line_dash="dash", line_color="#22c55e", line_width=2,
                                      annotation_text=f"Target +{profit_target*100:.0f}%", annotation_position="top")
                    
                    fig.update_layout(
                        title="Return Distribution",
                        xaxis_title="Return %",
                        yaxis_title="Count",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=350,
                        showlegend=False,
                        margin=dict(t=50, b=50, l=50, r=30)
                    )
                    fig.update_xaxes(gridcolor='#1e293b', zerolinecolor='#334155')
                    fig.update_yaxes(gridcolor='#1e293b')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    results_sorted = results.sort_values('Earnings Date')
                    results_sorted['Cumulative'] = (1 + results_sorted['Return']).cumprod()
                    results_sorted['Cumulative Pct'] = (results_sorted['Cumulative'] - 1) * 100
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(results_sorted) + 1)),
                        y=results_sorted['Cumulative Pct'],
                        mode='lines',
                        line=dict(color='#3b82f6', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(59, 130, 246, 0.1)'
                    ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
                    
                    fig.update_layout(
                        title="Cumulative Return",
                        xaxis_title="Trade #",
                        yaxis_title="Cumulative Return %",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=350,
                        showlegend=False,
                        margin=dict(t=50, b=50, l=50, r=30)
                    )
                    fig.update_xaxes(gridcolor='#1e293b')
                    fig.update_yaxes(gridcolor='#1e293b')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade List
                st.markdown("""
                <div class="section-header">
                    <span class="section-icon">📋</span>
                    <span class="section-title">Trade Log</span>
                </div>
                """, unsafe_allow_html=True)
                
                display_df = results.copy()
                display_df['Return'] = display_df['Return'].apply(lambda x: f"{x*100:+.2f}%")
                display_df['Max Return'] = display_df['Max Return'].apply(lambda x: f"{x*100:+.1f}%")
                display_df['Min Return'] = display_df['Min Return'].apply(lambda x: f"{x*100:+.1f}%")
                if 'Earnings Date' in display_df.columns:
                    display_df['Earnings Date'] = pd.to_datetime(display_df['Earnings Date']).dt.strftime('%Y-%m-%d')
                
                # Reorder columns
                col_order = ['Ticker', 'Company', 'Earnings Date', 'Exit Day', 'Exit Reason', 'Return', 'Max Return', 'Min Return']
                col_order = [c for c in col_order if c in display_df.columns]
                display_df = display_df[col_order]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)
        
        # Comparison Section
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">⚖️</span>
            <span class="section-title">Stop Loss Comparison</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("Compare different stop loss levels to find the optimal risk/reward balance")
        
        if st.button("📊 Compare All Stop Levels", use_container_width=True):
            with st.spinner("Analyzing stop loss levels..."):
                stop_levels = [-0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.15]
                
                comparison = []
                for stop in stop_levels:
                    if has_daily:
                        res = backtest_with_daily_prices(daily_df, stop, None, max_days)
                    else:
                        res = backtest_strategy_legacy(returns_df, stop, None, max_days)
                    
                    if not res.empty:
                        avg_ret = res['Return'].mean()
                        win_rate = (res['Return'] > 0).mean()
                        total_ret = res['Return'].sum()
                        stopped = (res['Exit Reason'] == 'Stop Loss').sum()
                        sharpe = (avg_ret / res['Return'].std()) * np.sqrt(52) if res['Return'].std() > 0 else 0
                        
                        comparison.append({
                            'Stop Loss': f"{stop*100:.0f}%",
                            'Stop Value': stop,
                            'Avg Return': avg_ret * 100,
                            'Win Rate': win_rate * 100,
                            'Total Return': total_ret * 100,
                            'Stopped %': stopped / len(res) * 100,
                            'Sharpe': sharpe,
                            'Trades': len(res)
                        })
                
                comp_df = pd.DataFrame(comparison)
                best_idx = comp_df['Sharpe'].idxmax()
                
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Bar colors - highlight best
                    bar_colors = ['#22c55e' if i == best_idx else '#3b82f6' for i in range(len(comp_df))]
                    
                    fig.add_trace(
                        go.Bar(
                            x=comp_df['Stop Loss'],
                            y=comp_df['Avg Return'],
                            name='Avg Return %',
                            marker_color=bar_colors,
                            opacity=0.8
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=comp_df['Stop Loss'],
                            y=comp_df['Sharpe'],
                            name='Sharpe Ratio',
                            mode='lines+markers',
                            line=dict(color='#f59e0b', width=3),
                            marker=dict(size=10)
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title="Stop Loss Performance Comparison",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        margin=dict(t=80, b=50, l=50, r=50)
                    )
                    fig.update_xaxes(gridcolor='#1e293b', title="Stop Loss Level")
                    fig.update_yaxes(title_text="Avg Return %", secondary_y=False, gridcolor='#1e293b')
                    fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=True, gridcolor='#1e293b')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Summary card for best
                    best_row = comp_df.loc[best_idx]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.05) 100%); 
                                border: 1px solid #22c55e; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
                        <div style="color: #22c55e; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                            🏆 Optimal Stop Loss
                        </div>
                        <div style="font-size: 2.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem;">
                            {best_row['Stop Loss']}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                            <div>
                                <div style="color: #94a3b8; font-size: 0.75rem;">Avg Return</div>
                                <div style="color: #22c55e; font-weight: 600;">{best_row['Avg Return']:+.2f}%</div>
                            </div>
                            <div>
                                <div style="color: #94a3b8; font-size: 0.75rem;">Sharpe</div>
                                <div style="color: #f59e0b; font-weight: 600;">{best_row['Sharpe']:.2f}</div>
                            </div>
                            <div>
                                <div style="color: #94a3b8; font-size: 0.75rem;">Win Rate</div>
                                <div style="color: #f1f5f9; font-weight: 600;">{best_row['Win Rate']:.1f}%</div>
                            </div>
                            <div>
                                <div style="color: #94a3b8; font-size: 0.75rem;">Stopped</div>
                                <div style="color: #f1f5f9; font-weight: 600;">{best_row['Stopped %']:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Comparison table
                    display_comp = comp_df[['Stop Loss', 'Avg Return', 'Win Rate', 'Sharpe', 'Stopped %']].copy()
                    display_comp['Avg Return'] = display_comp['Avg Return'].apply(lambda x: f"{x:+.2f}%")
                    display_comp['Win Rate'] = display_comp['Win Rate'].apply(lambda x: f"{x:.1f}%")
                    display_comp['Sharpe'] = display_comp['Sharpe'].apply(lambda x: f"{x:.2f}")
                    display_comp['Stopped %'] = display_comp['Stopped %'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(display_comp, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 4: POWERBI
# =============================================================================
with tab4:
    st.markdown("### PowerBI Dashboard")
    
    st.markdown(
        '[Open in Full Screen ↗](https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9)',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    st.markdown("""
    <style>
        .powerbi-container {
            position: relative;
            width: 100%;
            padding-bottom: 55.4%;
            height: 0;
            overflow: hidden;
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        .powerbi-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 8px;
        }
    </style>
    <div class="powerbi-container">
        <iframe 
            title="Finance Models"
            src="https://app.powerbi.com/view?r=eyJrIjoiZWRlNGNjYTgtODNhYy00MjBjLThhMjctMzgyNmYzNzIwZGRiIiwidCI6IjhkMWE2OWVjLTAzYjUtNDM0NS1hZTIxLWRhZDExMmY1ZmI0ZiIsImMiOjN9" 
            allowFullScreen="true">
        </iframe>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Earnings Momentum Strategy")
