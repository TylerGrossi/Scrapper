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
from scipy import stats

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
def load_earnings_universe():
    """Load earnings universe to get Date Check info."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/earnings_universe.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/earnings_universe.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and 'Date Check' in df.columns:
                return df
        except:
            continue
    try:
        df = pd.read_csv('earnings_universe.csv')
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def load_hourly_prices():
    """Load hourly prices data."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/hourly_prices.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/hourly_prices.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and 'Trading Day' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('hourly_prices.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        return None

def filter_date_passed(hourly_df, earnings_df):
    """Filter out tickers with DATE PASSED from hourly prices."""
    if earnings_df is None or 'Date Check' not in earnings_df.columns:
        return hourly_df
    
    date_passed_tickers = earnings_df[earnings_df['Date Check'] == 'DATE PASSED']['Ticker'].tolist()
    if date_passed_tickers:
        filtered_df = hourly_df[~hourly_df['Ticker'].isin(date_passed_tickers)]
        return filtered_df
    return hourly_df

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

def backtest_with_hourly_prices(hourly_df, stop_loss=None, max_days=5):
    """
    Backtest strategy using hourly data. If stop_loss is None, no stop loss is applied.
    
    IMPORTANT: If the stock gaps down and opens below the stop loss, you take the 
    actual opening return (the gap loss), not the stop loss price. This reflects
    real trading where you can't exit at a price that was never traded.
    
    Model Rules:
    - BMO: Buy at previous day's close, track from market open on earnings day
    - AMC: Buy at earnings day close, track from next day's market open
    """
    results = []
    
    # Group by trade (Ticker + Earnings Date + Fiscal Quarter)
    trades = hourly_df.groupby(['Ticker', 'Earnings Date', 'Fiscal Quarter']).size().reset_index()[['Ticker', 'Earnings Date', 'Fiscal Quarter']]
    
    for _, trade in trades.iterrows():
        ticker = trade['Ticker']
        earnings_date = trade['Earnings Date']
        fiscal_quarter = trade['Fiscal Quarter']
        
        # Get all hourly data for this trade (Trading Day >= 1 means after base price)
        trade_data = hourly_df[
            (hourly_df['Ticker'] == ticker) & 
            (hourly_df['Earnings Date'] == earnings_date) &
            (hourly_df['Trading Day'] >= 1)
        ].sort_values('Datetime')
        
        if trade_data.empty:
            continue
        
        # Get the last hour of max_days (end of day close)
        exit_day_data = trade_data[trade_data['Trading Day'] == max_days]
        if exit_day_data.empty:
            # Try adjacent days if exact day not available
            for target_day in [max_days - 1, max_days + 1]:
                exit_day_data = trade_data[trade_data['Trading Day'] == target_day]
                if not exit_day_data.empty:
                    break
        
        if exit_day_data.empty:
            continue
        
        # Get the last hour's return (end of day)
        exit_day_return = exit_day_data['Return From Earnings (%)'].iloc[-1]
        actual_exit_day = int(exit_day_data['Trading Day'].iloc[-1])
        
        if pd.isna(exit_day_return):
            continue
        
        company_name = trade_data['Company Name'].iloc[0] if 'Company Name' in trade_data.columns else ticker
        base_price = trade_data['Base Price'].iloc[0] if 'Base Price' in trade_data.columns else None
        
        exit_day = None
        exit_reason = None
        exit_return = None
        max_return = 0
        min_return = 0
        stopped_out = False
        gap_down = False
        
        # Check each hour for stop loss
        first_candle = True
        for _, hour_data in trade_data.iterrows():
            trading_day = int(hour_data['Trading Day'])
            hour_return = hour_data['Return From Earnings (%)']
            
            if trading_day > actual_exit_day:
                break
            
            if pd.isna(hour_return):
                continue
            
            hour_return_decimal = hour_return / 100
            max_return = max(max_return, hour_return_decimal)
            min_return = min(min_return, hour_return_decimal)
            
            # Check stop loss (only if stop_loss is provided)
            if stop_loss is not None and hour_return_decimal <= stop_loss and not stopped_out:
                exit_day = trading_day
                
                # CRITICAL: If the FIRST candle opens below stop loss (gap down),
                # take the actual gap loss, not the stop loss price
                if first_candle and hour_return_decimal < stop_loss:
                    exit_return = hour_return_decimal  # Take the actual gap loss
                    exit_reason = 'Gap Down'
                    gap_down = True
                else:
                    exit_return = stop_loss  # Normal stop loss hit
                    exit_reason = 'Stop Loss'
                
                stopped_out = True
                break
            
            first_candle = False
        
        # If not stopped out, exit at the target day close
        if not stopped_out:
            exit_day = actual_exit_day
            exit_reason = f'Day {actual_exit_day} Exit'
            exit_return = exit_day_return / 100
        
        results.append({
            'Ticker': ticker,
            'Company': company_name,
            'Earnings Date': earnings_date,
            'Fiscal Quarter': fiscal_quarter,
            'Base Price': base_price,
            'Exit Day': exit_day,
            'Exit Reason': exit_reason,
            'Return': exit_return,
            'Max Return': max_return,
            'Min Return': min_return,
            'Gap Down': gap_down,
        })
    
    return pd.DataFrame(results)

def backtest_strategy_legacy(df, stop_loss=None, max_days=5):
    """Legacy backtest using returns_tracker.csv (less accurate)."""
    results = []
    day_col = '5D Return' if max_days == 5 else '7D Return' if max_days == 7 else '10D Return'
    
    for idx, row in df.iterrows():
        high = row.get('1W High Return', np.nan)
        low = row.get('1W Low Return', np.nan)
        exit_return = row.get(day_col, np.nan)
        
        if pd.isna(exit_return):
            continue
        
        # Check stop loss (only if provided)
        # Note: Legacy method can't detect gap downs accurately
        if stop_loss is not None and pd.notna(low) and low <= stop_loss:
            final_return = low if low < stop_loss else stop_loss  # Take actual low if gap down
            exit_reason = 'Gap Down' if low < stop_loss else 'Stop Loss'
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
            'Gap Down': low < stop_loss if stop_loss and pd.notna(low) else False,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN APP
# =============================================================================

st.title("Earnings Momentum Strategy")

tab1, tab2, tab3, tab4 = st.tabs(["Stock Screener", "Stop Loss Analysis", "Earnings Analysis", "PowerBI"])

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
# TAB 2: STOP LOSS ANALYSIS
# =============================================================================
with tab2:
    hourly_df = load_hourly_prices()
    returns_df = load_returns_data()
    earnings_df = load_earnings_universe()
    
    # Filter out DATE PASSED tickers
    if hourly_df is not None and earnings_df is not None:
        date_passed_tickers = earnings_df[earnings_df['Date Check'] == 'DATE PASSED']['Ticker'].tolist() if 'Date Check' in earnings_df.columns else []
        if date_passed_tickers:
            hourly_df = hourly_df[~hourly_df['Ticker'].isin(date_passed_tickers)]
    
    has_hourly = hourly_df is not None and not hourly_df.empty
    has_returns = returns_df is not None and not returns_df.empty
    
    st.subheader("Stop Loss Analysis")
    
    if not has_hourly and not has_returns:
        st.warning("No data available. Upload hourly_prices.csv or returns_tracker.csv.")
        col1, col2 = st.columns(2)
        with col1:
            hourly_upload = st.file_uploader("Upload hourly_prices.csv:", type=['csv'], key='hourly')
            if hourly_upload:
                hourly_df = pd.read_csv(hourly_upload)
                hourly_df['Datetime'] = pd.to_datetime(hourly_df['Datetime'], errors='coerce')
                hourly_df['Date'] = pd.to_datetime(hourly_df['Date'], errors='coerce')
                hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date'], errors='coerce')
                has_hourly = True
        with col2:
            returns_upload = st.file_uploader("Upload returns_tracker.csv:", type=['csv'], key='returns')
            if returns_upload:
                returns_df = pd.read_csv(returns_upload)
                returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date'], errors='coerce')
                has_returns = True
    
    if has_hourly or has_returns:
        # Data summary
        st.markdown("### 📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        
        if has_hourly:
            unique_trades = hourly_df.groupby(['Ticker', 'Earnings Date']).ngroups
            with col1:
                st.metric("Total Trades", unique_trades)
            with col2:
                st.metric("Unique Tickers", hourly_df['Ticker'].nunique())
            with col3:
                st.metric("Hourly Data Points", f"{len(hourly_df):,}")
        
        # Show filtered info
        if earnings_df is not None and 'Date Check' in earnings_df.columns:
            date_passed_count = len(earnings_df[earnings_df['Date Check'] == 'DATE PASSED'])
            if date_passed_count > 0:
                st.caption(f"ℹ️ Filtered out {date_passed_count} tickers with incorrect earnings dates (DATE PASSED)")
        
        st.markdown("---")
        
        # Parameters
        st.markdown("### ⚙️ Backtest Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stop_loss = st.select_slider(
                "Stop Loss",
                options=[None, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10, -0.12, -0.15, -0.20],
                value=-0.10,
                format_func=lambda x: "None" if x is None else f"{x*100:.0f}%"
            )
        
        with col2:
            max_days = st.selectbox("Exit Day (if not stopped)", [3, 5, 7, 10], index=1)
        
        with col3:
            st.metric("Strategy", f"Stop: {stop_loss*100:.0f}%" if stop_loss else "No Stop", f"Exit: Day {max_days}")
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            
            with st.spinner("Running backtest on hourly data..."):
                # Detailed backtest function that captures hourly stop times
                def detailed_hourly_backtest(hourly_df, returns_df, stop_loss=None, max_days=5):
                    """
                    Detailed backtest with gap down handling.
                    
                    IMPORTANT: If stock gaps down and opens below stop loss on first candle,
                    take the actual gap loss, not the stop loss price.
                    """
                    results = []
                    
                    # Get all unique trades
                    trades = hourly_df.groupby(['Ticker', 'Earnings Date']).first().reset_index()[['Ticker', 'Earnings Date', 'Fiscal Quarter', 'Company Name', 'Base Price', 'Earnings Timing']]
                    
                    for _, trade in trades.iterrows():
                        ticker = trade['Ticker']
                        earnings_date = trade['Earnings Date']
                        fiscal_quarter = trade.get('Fiscal Quarter', '')
                        company_name = trade.get('Company Name', ticker)
                        base_price = trade.get('Base Price', None)
                        earnings_timing = trade.get('Earnings Timing', '')
                        
                        # Get all hourly data for this trade
                        trade_data = hourly_df[
                            (hourly_df['Ticker'] == ticker) & 
                            (hourly_df['Earnings Date'] == earnings_date) &
                            (hourly_df['Trading Day'] >= 1)
                        ].sort_values('Datetime')
                        
                        if trade_data.empty:
                            continue
                        
                        # Get 5D return from returns_tracker for comparison
                        return_5d = None
                        if returns_df is not None:
                            match = returns_df[
                                (returns_df['Ticker'] == ticker) & 
                                (returns_df['Earnings Date'].dt.date == earnings_date.date() if pd.notna(earnings_date) else False)
                            ]
                            if not match.empty and '5D Return' in match.columns:
                                return_5d = match['5D Return'].iloc[0]
                        
                        # Find exit day data
                        exit_day_data = trade_data[trade_data['Trading Day'] == max_days]
                        if exit_day_data.empty:
                            for target_day in [max_days - 1, max_days + 1, max_days - 2]:
                                exit_day_data = trade_data[trade_data['Trading Day'] == target_day]
                                if not exit_day_data.empty:
                                    break
                        
                        if exit_day_data.empty:
                            continue
                        
                        # Get end of day return for exit day
                        exit_day_close_return = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
                        actual_exit_day = int(exit_day_data['Trading Day'].iloc[-1])
                        
                        # Track through each hour
                        exit_reason = None
                        exit_return = None
                        exit_datetime = None
                        exit_trading_day = None
                        exit_hour = None
                        max_return = 0
                        min_return = 0
                        stopped_out = False
                        gap_down = False
                        first_candle = True
                        
                        for _, hour_data in trade_data.iterrows():
                            trading_day = int(hour_data['Trading Day'])
                            hour_return = hour_data['Return From Earnings (%)']
                            
                            if trading_day > actual_exit_day:
                                break
                            
                            if pd.isna(hour_return):
                                continue
                            
                            hour_return_decimal = hour_return / 100
                            max_return = max(max_return, hour_return_decimal)
                            min_return = min(min_return, hour_return_decimal)
                            
                            # Check stop loss
                            if stop_loss is not None and hour_return_decimal <= stop_loss and not stopped_out:
                                exit_trading_day = trading_day
                                exit_hour = hour_data.get('Time', hour_data.get('Hour', ''))
                                exit_datetime = hour_data.get('Datetime', None)
                                
                                # CRITICAL: If FIRST candle opens below stop (gap down),
                                # take actual gap loss, not stop loss price
                                if first_candle and hour_return_decimal < stop_loss:
                                    exit_return = hour_return_decimal
                                    exit_reason = 'Gap Down'
                                    gap_down = True
                                else:
                                    exit_return = stop_loss
                                    exit_reason = 'Stop Loss'
                                
                                stopped_out = True
                                break
                            
                            first_candle = False
                        
                        # If not stopped, exit at day close
                        if not stopped_out:
                            exit_trading_day = actual_exit_day
                            exit_hour = exit_day_data['Time'].iloc[-1] if 'Time' in exit_day_data.columns else ''
                            exit_datetime = exit_day_data['Datetime'].iloc[-1] if 'Datetime' in exit_day_data.columns else None
                            exit_reason = f'Day {actual_exit_day} Close'
                            exit_return = exit_day_close_return
                        
                        # Calculate difference vs 5D return
                        diff_vs_5d = None
                        if return_5d is not None and exit_return is not None:
                            diff_vs_5d = exit_return - return_5d
                        
                        results.append({
                            'Ticker': ticker,
                            'Company': company_name,
                            'Earnings Date': earnings_date,
                            'Fiscal Quarter': fiscal_quarter,
                            'Earnings Timing': earnings_timing,
                            'Base Price': base_price,
                            'Exit Day': exit_trading_day,
                            'Exit Hour': exit_hour,
                            'Exit Datetime': exit_datetime,
                            'Exit Reason': exit_reason,
                            'Backtest Return': exit_return,
                            '5D Return': return_5d,
                            'Diff vs 5D': diff_vs_5d,
                            'Max Intraday': max_return,
                            'Min Intraday': min_return,
                            'Stopped Out': stopped_out,
                            'Gap Down': gap_down,
                        })
                    
                    return pd.DataFrame(results)
                
                # Run the backtest
                results = detailed_hourly_backtest(hourly_df, returns_df, stop_loss=stop_loss, max_days=max_days)
            
            if results.empty:
                st.warning("No trades found with complete data.")
            else:
                # Store results in session state
                st.session_state['backtest_results'] = results
                st.session_state['stop_loss'] = stop_loss
                st.session_state['max_days'] = max_days
        
        # Display results if available
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            stop_loss = st.session_state['stop_loss']
            max_days = st.session_state['max_days']
            
            st.markdown("---")
            st.markdown("### 📈 Backtest Results")
            
            # Key Metrics
            total_return = results['Backtest Return'].sum() * 100
            avg_return = results['Backtest Return'].mean() * 100
            win_rate = (results['Backtest Return'] > 0).mean() * 100
            n_trades = len(results)
            stopped_count = results['Stopped Out'].sum()
            held_count = n_trades - stopped_count
            
            # Compare to 5D buy & hold
            results_with_5d = results[results['5D Return'].notna()]
            if len(results_with_5d) > 0:
                avg_5d_return = results_with_5d['5D Return'].mean() * 100
                total_5d_return = results_with_5d['5D Return'].sum() * 100
                avg_diff = results_with_5d['Diff vs 5D'].mean() * 100
            else:
                avg_5d_return = None
                total_5d_return = None
                avg_diff = None
            
            # Display metrics in clear cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value metric-{'green' if total_return > 0 else 'red'}">{total_return:+.1f}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value metric-{'green' if avg_return > 0 else 'red'}">{avg_return:+.2f}%</div>
                    <div class="metric-label">Avg Return/Trade</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value metric-blue">{win_rate:.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value metric-white">{n_trades}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Strategy vs Buy & Hold comparison
            if avg_5d_return is not None:
                st.markdown("### 🔄 Strategy vs Buy & Hold (5D)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**With Stop Loss Strategy**")
                    st.metric("Total Return", f"{total_return:+.1f}%")
                    st.metric("Avg/Trade", f"{avg_return:+.2f}%")
                
                with col2:
                    st.markdown("**Buy & Hold 5 Days**")
                    st.metric("Total Return", f"{total_5d_return:+.1f}%")
                    st.metric("Avg/Trade", f"{avg_5d_return:+.2f}%")
                
                with col3:
                    st.markdown("**Difference (Strategy - B&H)**")
                    diff_total = total_return - total_5d_return
                    st.metric("Total Diff", f"{diff_total:+.1f}%", delta_color="normal")
                    st.metric("Avg Diff", f"{avg_diff:+.2f}%", delta_color="normal")
                
                if diff_total > 0:
                    st.success(f"✅ Stop loss strategy **outperformed** buy & hold by {diff_total:+.1f}%")
                else:
                    st.warning(f"⚠️ Stop loss strategy **underperformed** buy & hold by {diff_total:.1f}%")
            
            st.markdown("---")
            
            # Exit Breakdown
            st.markdown("### 🚪 Exit Breakdown")
            
            # Count different exit types
            gap_down_count = results['Gap Down'].sum() if 'Gap Down' in results.columns else 0
            stop_loss_count = stopped_count - gap_down_count
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Gap downs (opened below stop)
                gap_trades = results[results['Gap Down'] == True] if 'Gap Down' in results.columns else pd.DataFrame()
                gap_avg = gap_trades['Backtest Return'].mean() * 100 if len(gap_trades) > 0 else 0
                
                st.markdown(f"""
                <div class="exit-card">
                    <div class="exit-count">{gap_down_count}</div>
                    <div class="exit-pct">({gap_down_count/n_trades*100:.1f}% of trades)</div>
                    <div class="exit-return" style="color: #ef4444;">{gap_avg:+.2f}% avg</div>
                    <div class="exit-label">📉 Gap Down (opened below stop)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Normal stop losses
                stop_trades = results[(results['Stopped Out'] == True) & (results.get('Gap Down', False) == False)]
                stop_avg = stop_trades['Backtest Return'].mean() * 100 if len(stop_trades) > 0 else 0
                
                st.markdown(f"""
                <div class="exit-card">
                    <div class="exit-count">{stop_loss_count}</div>
                    <div class="exit-pct">({stop_loss_count/n_trades*100:.1f}% of trades)</div>
                    <div class="exit-return" style="color: #f59e0b;">{stop_avg:+.2f}% avg</div>
                    <div class="exit-label">🛑 Stop Loss Hit</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                held_trades = results[results['Stopped Out'] == False]
                held_avg = held_trades['Backtest Return'].mean() * 100 if len(held_trades) > 0 else 0
                
                st.markdown(f"""
                <div class="exit-card">
                    <div class="exit-count">{held_count}</div>
                    <div class="exit-pct">({held_count/n_trades*100:.1f}% of trades)</div>
                    <div class="exit-return" style="color: #22c55e;">{held_avg:+.2f}% avg</div>
                    <div class="exit-label">✅ Held to Day {max_days}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Additional details
            col1, col2 = st.columns(2)
            
            with col1:
                if stopped_count > 0:
                    stopped_trades = results[results['Stopped Out'] == True]
                    st.markdown("**When stops/gaps occurred:**")
                    stop_day_dist = stopped_trades['Exit Day'].value_counts().sort_index()
                    for day, count in stop_day_dist.items():
                        pct = count / len(stopped_trades) * 100
                        st.markdown(f"- Day {day}: {count} trades ({pct:.0f}%)")
                    
                    if gap_down_count > 0:
                        st.markdown(f"\n**Gap downs:** {gap_down_count} trades gapped below {stop_loss*100:.0f}% stop")
                        worst_gap = gap_trades['Backtest Return'].min() * 100
                        st.markdown(f"- Worst gap: {worst_gap:+.1f}%")
            
            with col2:
                if len(held_trades) > 0:
                    st.markdown("**Held trade performance:**")
                    winners = len(held_trades[held_trades['Backtest Return'] > 0])
                    losers = len(held_trades[held_trades['Backtest Return'] <= 0])
                    st.markdown(f"- Winners: {winners} ({winners/len(held_trades)*100:.0f}%)")
                    st.markdown(f"- Losers: {losers} ({losers/len(held_trades)*100:.0f}%)")
            
            st.markdown("---")
            
            # Charts
            st.markdown("### 📊 Performance Charts")
            
            chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Cumulative Returns", "Return Distribution", "Stop Loss Timing"])
            
            with chart_tab1:
                results_sorted = results.sort_values('Earnings Date').copy()
                
                # Use simple cumulative sum (not compounded) for clearer visualization
                results_sorted['Cumulative'] = results_sorted['Backtest Return'].cumsum() * 100
                
                if '5D Return' in results_sorted.columns:
                    results_sorted['Cumulative_5D'] = results_sorted['5D Return'].fillna(0).cumsum() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(results_sorted) + 1)),
                    y=results_sorted['Cumulative'],
                    mode='lines',
                    line=dict(color='#3b82f6', width=2),
                    name=f'Strategy ({stop_loss*100:.0f}% stop)' if stop_loss else 'Strategy (no stop)',
                    hovertemplate='Trade %{x}<br>Cumulative: %{y:+.1f}%<extra></extra>'
                ))
                
                if '5D Return' in results_sorted.columns:
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(results_sorted) + 1)),
                        y=results_sorted['Cumulative_5D'],
                        mode='lines',
                        line=dict(color='#64748b', width=2, dash='dash'),
                        name='Buy & Hold 5D',
                        hovertemplate='Trade %{x}<br>Cumulative: %{y:+.1f}%<extra></extra>'
                    ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                fig.update_layout(
                    title="Cumulative Return: Strategy vs Buy & Hold (Simple Sum)",
                    xaxis_title="Trade #",
                    yaxis_title="Cumulative Return %",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
                )
                fig.update_xaxes(gridcolor='#1e293b')
                fig.update_yaxes(gridcolor='#1e293b')
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=results['Backtest Return'] * 100,
                        nbinsx=30,
                        marker_color='#3b82f6',
                        name='Strategy Returns'
                    ))
                    fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
                    if stop_loss:
                        fig.add_vline(x=stop_loss*100, line_dash="dash", line_color="#ef4444", 
                                     annotation_text=f"Stop: {stop_loss*100:.0f}%")
                    fig.update_layout(
                        title="Strategy Return Distribution",
                        xaxis_title="Return %",
                        yaxis_title="Count",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Diff vs 5D' in results.columns:
                        diff_data = results['Diff vs 5D'].dropna() * 100
                        fig = go.Figure()
                        colors = ['#22c55e' if x >= 0 else '#ef4444' for x in diff_data]
                        fig.add_trace(go.Histogram(
                            x=diff_data,
                            nbinsx=30,
                            marker_color='#f59e0b',
                            name='Difference'
                        ))
                        fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
                        fig.update_layout(
                            title="Strategy vs 5D Return (Difference)",
                            xaxis_title="Difference %",
                            yaxis_title="Count",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#94a3b8',
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab3:
                if stopped_count > 0:
                    stopped_trades = results[results['Stopped Out'] == True].copy()
                    
                    # Create visualization of when stops were triggered
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # By day
                        day_counts = stopped_trades['Exit Day'].value_counts().sort_index().reset_index()
                        day_counts.columns = ['Day', 'Count']
                        
                        fig = px.bar(day_counts, x='Day', y='Count', 
                                    title='Stop Loss Triggers by Trading Day',
                                    color='Count', color_continuous_scale='Reds')
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#94a3b8',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # By hour
                        if 'Exit Hour' in stopped_trades.columns:
                            hour_counts = stopped_trades['Exit Hour'].value_counts().sort_index().reset_index()
                            hour_counts.columns = ['Hour', 'Count']
                            
                            fig = px.bar(hour_counts, x='Hour', y='Count',
                                        title='Stop Loss Triggers by Hour',
                                        color='Count', color_continuous_scale='Reds')
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#94a3b8',
                                height=300,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show the stopped trades
                    st.markdown("**Stopped Out Trades Detail:**")
                    stopped_display = stopped_trades[['Ticker', 'Earnings Date', 'Exit Day', 'Exit Hour', 'Backtest Return', '5D Return', 'Diff vs 5D', 'Min Intraday']].copy()
                    stopped_display['Earnings Date'] = pd.to_datetime(stopped_display['Earnings Date']).dt.strftime('%Y-%m-%d')
                    stopped_display['Backtest Return'] = stopped_display['Backtest Return'] * 100
                    stopped_display['5D Return'] = stopped_display['5D Return'] * 100
                    stopped_display['Diff vs 5D'] = stopped_display['Diff vs 5D'] * 100
                    stopped_display['Min Intraday'] = stopped_display['Min Intraday'] * 100
                    
                    st.dataframe(
                        stopped_display, 
                        use_container_width=True, 
                        hide_index=True, 
                        height=250,
                        column_config={
                            "Backtest Return": st.column_config.NumberColumn("Backtest Return", format="%+.2f%%"),
                            "5D Return": st.column_config.NumberColumn("5D Return", format="%+.2f%%"),
                            "Diff vs 5D": st.column_config.NumberColumn("Diff vs 5D", format="%+.2f%%"),
                            "Min Intraday": st.column_config.NumberColumn("Min Intraday", format="%+.2f%%"),
                        }
                    )
                else:
                    st.info("No trades were stopped out with the current stop loss setting.")
            
            st.markdown("---")
            
            # Full Trade List
            st.markdown("### 📋 All Trades")
            
            display_df = results.copy()
            display_df['Earnings Date'] = pd.to_datetime(display_df['Earnings Date']).dt.strftime('%Y-%m-%d')
            
            # Keep numeric columns as numbers for sorting, use column_config for formatting
            display_df['Backtest Return'] = display_df['Backtest Return'] * 100
            display_df['5D Return'] = display_df['5D Return'] * 100
            display_df['Diff vs 5D'] = display_df['Diff vs 5D'] * 100
            display_df['Max Intraday'] = display_df['Max Intraday'] * 100
            display_df['Min Intraday'] = display_df['Min Intraday'] * 100
            
            col_order = ['Ticker', 'Company', 'Fiscal Quarter', 'Earnings Date', 'Base Price', 
                        'Exit Day', 'Exit Hour', 'Exit Reason', 'Backtest Return', '5D Return', 
                        'Diff vs 5D', 'Max Intraday', 'Min Intraday']
            col_order = [c for c in col_order if c in display_df.columns]
            
            st.dataframe(
                display_df[col_order], 
                use_container_width=True, 
                hide_index=True, 
                height=400,
                column_config={
                    "Base Price": st.column_config.NumberColumn("Base Price", format="$%.2f"),
                    "Backtest Return": st.column_config.NumberColumn("Backtest Return", format="%+.2f%%"),
                    "5D Return": st.column_config.NumberColumn("5D Return", format="%+.2f%%"),
                    "Diff vs 5D": st.column_config.NumberColumn("Diff vs 5D", format="%+.2f%%"),
                    "Max Intraday": st.column_config.NumberColumn("Max Intraday", format="%+.1f%%"),
                    "Min Intraday": st.column_config.NumberColumn("Min Intraday", format="%+.1f%%"),
                }
            )
            
            # Download button
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"backtest_results_stop{int(stop_loss*100) if stop_loss else 'none'}_day{max_days}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            
            # Stop Loss Comparison Tool
            st.markdown("### 🔬 Compare Stop Loss Levels")
            
            if st.button("Run Full Comparison", use_container_width=True):
                stop_levels = [None, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.15]
                
                comparison = []
                progress = st.progress(0)
                
                for i, stop in enumerate(stop_levels):
                    # Quick backtest for comparison
                    res_list = []
                    trades = hourly_df.groupby(['Ticker', 'Earnings Date']).first().reset_index()
                    
                    for _, trade in trades.iterrows():
                        ticker = trade['Ticker']
                        earnings_date = trade['Earnings Date']
                        
                        trade_data = hourly_df[
                            (hourly_df['Ticker'] == ticker) & 
                            (hourly_df['Earnings Date'] == earnings_date) &
                            (hourly_df['Trading Day'] >= 1) &
                            (hourly_df['Trading Day'] <= max_days)
                        ].sort_values('Datetime')
                        
                        if trade_data.empty:
                            continue
                        
                        exit_return = None
                        stopped = False
                        gap_down = False
                        first_candle = True
                        
                        for _, hour in trade_data.iterrows():
                            ret = hour['Return From Earnings (%)'] / 100
                            if stop is not None and ret <= stop:
                                # Gap down: first candle opens below stop
                                if first_candle and ret < stop:
                                    exit_return = ret  # Take actual gap loss
                                    gap_down = True
                                else:
                                    exit_return = stop
                                stopped = True
                                break
                            first_candle = False
                        
                        if exit_return is None:
                            last_day = trade_data[trade_data['Trading Day'] == trade_data['Trading Day'].max()]
                            if not last_day.empty:
                                exit_return = last_day['Return From Earnings (%)'].iloc[-1] / 100
                        
                        if exit_return is not None:
                            res_list.append({
                                'Return': exit_return, 
                                'Stopped': stopped,
                                'Gap Down': gap_down
                            })
                    
                    if res_list:
                        res_df = pd.DataFrame(res_list)
                        comparison.append({
                            'Stop Loss': f"{stop*100:.0f}%" if stop else "None",
                            'Stop Value': stop if stop else 0,
                            'Total Return': res_df['Return'].sum() * 100,
                            'Avg Return': res_df['Return'].mean() * 100,
                            'Win Rate': (res_df['Return'] > 0).mean() * 100,
                            'Stopped %': res_df['Stopped'].mean() * 100 if stop else 0,
                            'Gap Downs': res_df['Gap Down'].sum() if stop else 0,
                            'Trades': len(res_df)
                        })
                    
                    progress.progress((i + 1) / len(stop_levels))
                
                progress.empty()
                comp_df = pd.DataFrame(comparison)
                
                # Find best
                best_idx = comp_df['Total Return'].idxmax()
                
                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    fig = go.Figure()
                    colors = ['#22c55e' if i == best_idx else '#3b82f6' for i in range(len(comp_df))]
                    fig.add_trace(go.Bar(
                        x=comp_df['Stop Loss'],
                        y=comp_df['Total Return'],
                        marker_color=colors,
                        text=comp_df['Total Return'].apply(lambda x: f"{x:+.1f}%"),
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title="Total Return by Stop Loss Level",
                        xaxis_title="Stop Loss",
                        yaxis_title="Total Return %",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=400,
                        xaxis=dict(type='category')  # Force categorical x-axis
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Comparison Table**")
                    display_comp = comp_df.copy()
                    display_comp['Total Return'] = display_comp['Total Return'].apply(lambda x: f"{x:+.1f}%")
                    display_comp['Avg Return'] = display_comp['Avg Return'].apply(lambda x: f"{x:+.2f}%")
                    display_comp['Win Rate'] = display_comp['Win Rate'].apply(lambda x: f"{x:.1f}%")
                    display_comp['Stopped %'] = display_comp['Stopped %'].apply(lambda x: f"{x:.1f}%")
                    display_comp = display_comp.drop(columns=['Stop Value'])
                    
                    st.dataframe(display_comp, use_container_width=True, hide_index=True)
                    
                    best_stop = comp_df.loc[best_idx, 'Stop Loss']
                    best_return = comp_df.loc[best_idx, 'Total Return']
                    st.success(f"**Best: {best_stop}** with {best_return:+.1f}% total return")

# =============================================================================
# TAB 3: EARNINGS ANALYSIS
# =============================================================================
with tab3:
    st.subheader("📊 Earnings Surprise Analysis")
    st.markdown("Analyze how earnings beats/misses and surprise magnitude affect stock returns")
    
    # Load data
    returns_df = load_returns_data()
    
    if returns_df is None or returns_df.empty:
        st.warning("No returns data available. Please ensure returns_tracker.csv is accessible.")
        
        uploaded_file = st.file_uploader("Upload returns_tracker.csv:", type=['csv'], key='earnings_upload')
        if uploaded_file:
            returns_df = pd.read_csv(uploaded_file)
            returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date'], errors='coerce')
            st.success(f"Loaded {len(returns_df)} rows")
    
    if returns_df is not None and not returns_df.empty:
        st.success(f"✅ Loaded {len(returns_df)} trades from returns tracker")
        
        # Show available earnings columns
        earnings_cols = ['EPS Estimate', 'Reported EPS', 'EPS Surprise (%)', 'EPS (TTM)', 'P/E', 'Forward P/E', 'Sector']
        available_earnings_cols = [c for c in earnings_cols if c in returns_df.columns]
        
        with st.expander("📋 Available Earnings Data Columns"):
            st.write(available_earnings_cols)
            st.caption(f"Total columns in dataset: {len(returns_df.columns)}")
        
        # Clean up the data
        analysis_df = returns_df.copy()
        
        # Convert EPS Surprise to numeric if needed
        if 'EPS Surprise (%)' in analysis_df.columns:
            analysis_df['EPS Surprise (%)'] = pd.to_numeric(analysis_df['EPS Surprise (%)'], errors='coerce')
        
        # Determine return column to use
        return_col = None
        for col in ['5D Return', '1D Return', '3D Return', '7D Return', '10D Return']:
            if col in analysis_df.columns:
                return_col = col
                break
        
        if return_col is None:
            st.error("No return column found in data.")
        else:
            # Convert returns to percentage for display
            analysis_df[return_col] = pd.to_numeric(analysis_df[return_col], errors='coerce') * 100
            
            st.markdown("---")
            
            # Quick Stats
            col1, col2, col3, col4 = st.columns(4)
            
            valid_surprise = analysis_df['EPS Surprise (%)'].notna().sum() if 'EPS Surprise (%)' in analysis_df.columns else 0
            avg_surprise = analysis_df['EPS Surprise (%)'].mean() if 'EPS Surprise (%)' in analysis_df.columns else 0
            
            with col1:
                st.metric("Total Trades", len(analysis_df))
            with col2:
                st.metric("With Surprise Data", valid_surprise)
            with col3:
                st.metric("Avg EPS Surprise", f"{avg_surprise:.1f}%" if pd.notna(avg_surprise) else "N/A")
            with col4:
                st.metric("Return Column", return_col)
            
            st.markdown("---")
            
            # Create analysis tabs
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
                "Beat vs Miss", "Surprise Magnitude", "Sector Analysis", "Raw Data"
            ])
            
            with analysis_tab1:
                st.markdown("#### Beat vs Miss Performance")
                
                if 'EPS Surprise (%)' in analysis_df.columns and valid_surprise > 0:
                    # Classify as beat, miss, or inline
                    def classify_surprise(x):
                        if pd.isna(x):
                            return 'Unknown'
                        elif x > 5:
                            return 'Strong Beat (>5%)'
                        elif x > 0:
                            return 'Beat (0-5%)'
                        elif x > -5:
                            return 'Miss (0 to -5%)'
                        else:
                            return 'Strong Miss (<-5%)'
                    
                    analysis_df['Surprise Category'] = analysis_df['EPS Surprise (%)'].apply(classify_surprise)
                    
                    # Filter out Unknown for main analysis
                    known_df = analysis_df[analysis_df['Surprise Category'] != 'Unknown']
                    
                    if len(known_df) > 0:
                        # Calculate metrics by category
                        category_stats = known_df.groupby('Surprise Category').agg({
                            return_col: ['mean', 'median', 'count', 'std'],
                        }).round(2)
                        category_stats.columns = ['Avg Return', 'Median Return', 'Count', 'Std Dev']
                        category_stats['Win Rate'] = known_df.groupby('Surprise Category')[return_col].apply(
                            lambda x: (x > 0).mean() * 100
                        ).round(1)
                        category_stats = category_stats.reset_index()
                        
                        # Sort by expected order
                        order = ['Strong Beat (>5%)', 'Beat (0-5%)', 'Miss (0 to -5%)', 'Strong Miss (<-5%)']
                        category_stats['sort_order'] = category_stats['Surprise Category'].apply(
                            lambda x: order.index(x) if x in order else 99
                        )
                        category_stats = category_stats.sort_values('sort_order').drop(columns=['sort_order'])
                        
                        col1, col2 = st.columns([1.5, 1])
                        
                        with col1:
                            # Bar chart
                            fig = go.Figure()
                            
                            colors = {
                                'Strong Beat (>5%)': '#22c55e',
                                'Beat (0-5%)': '#86efac',
                                'Miss (0 to -5%)': '#fca5a5',
                                'Strong Miss (<-5%)': '#ef4444',
                            }
                            
                            bar_colors = [colors.get(cat, '#3b82f6') for cat in category_stats['Surprise Category']]
                            
                            fig.add_trace(go.Bar(
                                x=category_stats['Surprise Category'],
                                y=category_stats['Avg Return'],
                                marker_color=bar_colors,
                                text=category_stats['Avg Return'].apply(lambda x: f"{x:+.2f}%"),
                                textposition='outside'
                            ))
                            
                            fig.update_layout(
                                title="Average Return by Earnings Surprise Category",
                                xaxis_title="Surprise Category",
                                yaxis_title=f"Avg {return_col} (%)",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#94a3b8',
                                height=400
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Performance by Category**")
                            st.dataframe(
                                category_stats,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Median Return": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Std Dev": st.column_config.NumberColumn(format="%.2f"),
                                }
                            )
                            
                            # Key insight
                            beats = category_stats[category_stats['Surprise Category'].str.contains('Beat', na=False)]
                            misses = category_stats[category_stats['Surprise Category'].str.contains('Miss', na=False)]
                            
                            if len(beats) > 0 and len(misses) > 0:
                                beat_avg = (beats['Avg Return'] * beats['Count']).sum() / beats['Count'].sum()
                                miss_avg = (misses['Avg Return'] * misses['Count']).sum() / misses['Count'].sum()
                                diff = beat_avg - miss_avg
                                
                                st.markdown("---")
                                st.markdown(f"""
                                **Key Insight:**
                                - Beats avg: **{beat_avg:+.2f}%**
                                - Misses avg: **{miss_avg:+.2f}%**
                                - Spread: **{diff:+.2f}%**
                                """)
                                
                                if diff > 0:
                                    st.success("✅ Beats outperform misses")
                                else:
                                    st.warning("⚠️ Misses outperform beats (unusual)")
                        
                        # Additional: Simple Beat vs Miss
                        st.markdown("---")
                        st.markdown("#### Simple Beat vs Miss (Any Amount)")
                        
                        simple_df = known_df.copy()
                        simple_df['Beat/Miss'] = simple_df['EPS Surprise (%)'].apply(
                            lambda x: 'Beat' if x > 0 else 'Miss'
                        )
                        
                        simple_stats = simple_df.groupby('Beat/Miss').agg({
                            return_col: ['mean', 'median', 'count'],
                            'EPS Surprise (%)': 'mean'
                        }).round(2)
                        simple_stats.columns = ['Avg Return', 'Median Return', 'Count', 'Avg Surprise %']
                        simple_stats['Win Rate'] = simple_df.groupby('Beat/Miss')[return_col].apply(
                            lambda x: (x > 0).mean() * 100
                        ).round(1)
                        simple_stats = simple_stats.reset_index()
                        
                        col1, col2, col3 = st.columns(3)
                        
                        beat_row = simple_stats[simple_stats['Beat/Miss'] == 'Beat']
                        miss_row = simple_stats[simple_stats['Beat/Miss'] == 'Miss']
                        
                        with col1:
                            if len(beat_row) > 0:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value metric-green">{beat_row['Avg Return'].values[0]:+.2f}%</div>
                                    <div class="metric-label">Beat Avg Return ({int(beat_row['Count'].values[0])} trades)</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            if len(miss_row) > 0:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value metric-red">{miss_row['Avg Return'].values[0]:+.2f}%</div>
                                    <div class="metric-label">Miss Avg Return ({int(miss_row['Count'].values[0])} trades)</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            if len(beat_row) > 0 and len(miss_row) > 0:
                                spread = beat_row['Avg Return'].values[0] - miss_row['Avg Return'].values[0]
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value metric-blue">{spread:+.2f}%</div>
                                    <div class="metric-label">Beat - Miss Spread</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No valid earnings surprise data to analyze.")
                else:
                    st.warning("No 'EPS Surprise (%)' column found in data.")
            
            with analysis_tab2:
                st.markdown("#### Surprise Magnitude vs Returns")
                
                if 'EPS Surprise (%)' in analysis_df.columns:
                    # Filter to valid data
                    scatter_df = analysis_df[
                        analysis_df['EPS Surprise (%)'].notna() & 
                        analysis_df[return_col].notna()
                    ].copy()
                    
                    if len(scatter_df) > 5:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Scatter plot
                            fig = px.scatter(
                                scatter_df,
                                x='EPS Surprise (%)',
                                y=return_col,
                                hover_data=['Ticker', 'Company Name', 'Sector'] if 'Company Name' in scatter_df.columns else ['Ticker'],
                                trendline='ols',
                                title="EPS Surprise % vs Stock Return",
                                color='Sector' if 'Sector' in scatter_df.columns else None
                            )
                            
                            fig.update_traces(marker=dict(size=10, opacity=0.7))
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#94a3b8',
                                height=500
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                            fig.add_vline(x=0, line_dash="dash", line_color="#475569")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Correlation stats
                            correlation = scatter_df['EPS Surprise (%)'].corr(scatter_df[return_col])
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value metric-blue">{correlation:.3f}</div>
                                <div class="metric-label">Correlation</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("")
                            
                            # R-squared from trendline
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                scatter_df['EPS Surprise (%)'], 
                                scatter_df[return_col]
                            )
                            
                            st.markdown(f"""
                            **Regression Stats:**
                            - R²: **{r_value**2:.3f}**
                            - Slope: **{slope:.4f}**
                            - P-value: **{p_value:.4f}**
                            """)
                            
                            if p_value < 0.05:
                                st.success("✅ Statistically significant")
                            elif p_value < 0.10:
                                st.info("ℹ️ Marginally significant")
                            else:
                                st.warning("⚠️ Not significant (p > 0.10)")
                            
                            st.markdown("---")
                            
                            # Quintile analysis
                            st.markdown("**Returns by Surprise Quintile**")
                            try:
                                scatter_df['Surprise Quintile'] = pd.qcut(
                                    scatter_df['EPS Surprise (%)'], 
                                    q=5, 
                                    labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'],
                                    duplicates='drop'
                                )
                                
                                quintile_stats = scatter_df.groupby('Surprise Quintile', observed=True)[return_col].agg(
                                    ['mean', 'count']
                                ).round(2)
                                quintile_stats.columns = ['Avg Return', 'Count']
                                
                                st.dataframe(
                                    quintile_stats,
                                    use_container_width=True,
                                    column_config={
                                        "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                                    }
                                )
                            except Exception as e:
                                st.caption(f"Could not create quintiles: {e}")
                        
                        # Surprise buckets analysis
                        st.markdown("---")
                        st.markdown("#### Returns by Surprise Buckets")
                        
                        # Create buckets
                        def bucket_surprise(x):
                            if pd.isna(x):
                                return None
                            elif x < -10:
                                return '< -10%'
                            elif x < -5:
                                return '-10% to -5%'
                            elif x < 0:
                                return '-5% to 0%'
                            elif x < 5:
                                return '0% to 5%'
                            elif x < 10:
                                return '5% to 10%'
                            elif x < 20:
                                return '10% to 20%'
                            else:
                                return '> 20%'
                        
                        scatter_df['Surprise Bucket'] = scatter_df['EPS Surprise (%)'].apply(bucket_surprise)
                        
                        bucket_stats = scatter_df.groupby('Surprise Bucket').agg({
                            return_col: ['mean', 'median', 'count'],
                        }).round(2)
                        bucket_stats.columns = ['Avg Return', 'Median', 'Count']
                        bucket_stats['Win Rate'] = scatter_df.groupby('Surprise Bucket')[return_col].apply(
                            lambda x: (x > 0).mean() * 100
                        ).round(1)
                        bucket_stats = bucket_stats.reset_index()
                        
                        # Sort buckets
                        bucket_order = ['< -10%', '-10% to -5%', '-5% to 0%', '0% to 5%', '5% to 10%', '10% to 20%', '> 20%']
                        bucket_stats['sort_order'] = bucket_stats['Surprise Bucket'].apply(
                            lambda x: bucket_order.index(x) if x in bucket_order else 99
                        )
                        bucket_stats = bucket_stats.sort_values('sort_order').drop(columns=['sort_order'])
                        
                        col1, col2 = st.columns([1.5, 1])
                        
                        with col1:
                            fig = go.Figure()
                            colors = ['#ef4444', '#f97316', '#fbbf24', '#a3e635', '#22c55e', '#10b981', '#059669']
                            
                            fig.add_trace(go.Bar(
                                x=bucket_stats['Surprise Bucket'],
                                y=bucket_stats['Avg Return'],
                                marker_color=colors[:len(bucket_stats)],
                                text=bucket_stats['Avg Return'].apply(lambda x: f"{x:+.2f}%"),
                                textposition='outside'
                            ))
                            
                            fig.update_layout(
                                title="Average Return by EPS Surprise Bucket",
                                xaxis_title="EPS Surprise %",
                                yaxis_title=f"Avg {return_col} (%)",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#94a3b8',
                                height=400
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.dataframe(
                                bucket_stats, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Median": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                                }
                            )
                    else:
                        st.warning("Not enough data points for scatter analysis.")
                else:
                    st.warning("No EPS Surprise data available for magnitude analysis.")
            
            with analysis_tab3:
                st.markdown("#### Sector Analysis")
                
                if 'Sector' in analysis_df.columns:
                    sector_df = analysis_df[analysis_df['Sector'].notna() & (analysis_df['Sector'] != '')].copy()
                    
                    if len(sector_df) > 0:
                        # Returns by sector
                        sector_stats = sector_df.groupby('Sector').agg({
                            return_col: ['mean', 'median', 'count', 'std'],
                        }).round(2)
                        sector_stats.columns = ['Avg Return', 'Median', 'Count', 'Std Dev']
                        sector_stats['Win Rate'] = sector_df.groupby('Sector')[return_col].apply(
                            lambda x: (x > 0).mean() * 100
                        ).round(1)
                        
                        # Add average surprise if available
                        if 'EPS Surprise (%)' in sector_df.columns:
                            sector_stats['Avg Surprise'] = sector_df.groupby('Sector')['EPS Surprise (%)'].mean().round(1)
                        
                        sector_stats = sector_stats.reset_index().sort_values('Avg Return', ascending=False)
                        
                        col1, col2 = st.columns([1.5, 1])
                        
                        with col1:
                            fig = px.bar(
                                sector_stats,
                                x='Sector',
                                y='Avg Return',
                                color='Avg Return',
                                color_continuous_scale=['#ef4444', '#fbbf24', '#22c55e'],
                                title="Average Return by Sector",
                                text='Avg Return'
                            )
                            fig.update_traces(texttemplate='%{text:+.1f}%', textposition='outside')
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#94a3b8',
                                height=450,
                                xaxis_tickangle=-45,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Sector Performance Table**")
                            st.dataframe(
                                sector_stats,
                                use_container_width=True,
                                hide_index=True,
                                height=400,
                                column_config={
                                    "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Median": st.column_config.NumberColumn(format="%.2f%%"),
                                    "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                                    "Avg Surprise": st.column_config.NumberColumn(format="%.1f%%"),
                                }
                            )
                        
                        # Sector + Surprise interaction
                        if 'EPS Surprise (%)' in analysis_df.columns and 'Surprise Category' in analysis_df.columns:
                            st.markdown("---")
                            st.markdown("#### Sector × Earnings Surprise Heatmap")
                            
                            heatmap_df = analysis_df[
                                analysis_df['Sector'].notna() & 
                                (analysis_df['Surprise Category'] != 'Unknown')
                            ]
                            
                            if len(heatmap_df) > 10:
                                pivot = heatmap_df.pivot_table(
                                    values=return_col,
                                    index='Sector',
                                    columns='Surprise Category',
                                    aggfunc='mean'
                                ).round(2)
                                
                                # Reorder columns
                                col_order = ['Strong Beat (>5%)', 'Beat (0-5%)', 'Miss (0 to -5%)', 'Strong Miss (<-5%)']
                                pivot = pivot[[c for c in col_order if c in pivot.columns]]
                                
                                fig = px.imshow(
                                    pivot,
                                    color_continuous_scale='RdYlGn',
                                    color_continuous_midpoint=0,
                                    title="Avg Return: Sector × Surprise Category",
                                    text_auto='.1f'
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#94a3b8',
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.caption("Values show average return (%). Green = positive, Red = negative.")
                    else:
                        st.warning("No sector data available after filtering.")
                else:
                    st.warning("No 'Sector' column found in data.")
                    
                    # Fallback: by earnings timing
                    if 'Earnings Timing' in analysis_df.columns:
                        st.markdown("#### Returns by Earnings Timing (BMO vs AMC)")
                        
                        timing_df = analysis_df[analysis_df['Earnings Timing'].notna()].copy()
                        
                        if len(timing_df) > 0:
                            timing_stats = timing_df.groupby('Earnings Timing').agg({
                                return_col: ['mean', 'median', 'count'],
                            }).round(2)
                            timing_stats.columns = ['Avg Return', 'Median', 'Count']
                            timing_stats['Win Rate'] = timing_df.groupby('Earnings Timing')[return_col].apply(
                                lambda x: (x > 0).mean() * 100
                            ).round(1)
                            timing_stats = timing_stats.reset_index()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.bar(
                                    timing_stats,
                                    x='Earnings Timing',
                                    y='Avg Return',
                                    color='Earnings Timing',
                                    title="Returns by Earnings Timing"
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#94a3b8',
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.dataframe(timing_stats, use_container_width=True, hide_index=True)
            
            with analysis_tab4:
                st.markdown("#### Raw Data Explorer")
                
                # Column selector
                all_cols = list(analysis_df.columns)
                default_cols = ['Ticker', 'Company Name', 'Sector', 'Earnings Date', 'EPS Estimate', 
                               'Reported EPS', 'EPS Surprise (%)', return_col]
                default_cols = [c for c in default_cols if c in all_cols]
                
                selected_cols = st.multiselect(
                    "Select columns to display:",
                    options=all_cols,
                    default=default_cols[:10]
                )
                
                if selected_cols:
                    display_data = analysis_df[selected_cols].copy()
                    
                    # Filters
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Sector' in selected_cols and 'Sector' in analysis_df.columns:
                            sectors = ['All'] + sorted(analysis_df['Sector'].dropna().unique().tolist())
                            selected_sector = st.selectbox("Filter by Sector:", sectors)
                            if selected_sector != 'All':
                                display_data = display_data[display_data['Sector'] == selected_sector]
                    
                    with col2:
                        sort_col = st.selectbox("Sort by:", selected_cols, index=0)
                    
                    with col3:
                        sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
                    
                    display_data = display_data.sort_values(
                        sort_col, 
                        ascending=(sort_order == "Ascending")
                    )
                    
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        hide_index=True,
                        height=500
                    )
                    
                    st.caption(f"Showing {len(display_data)} rows")
                    
                    # Download
                    csv = display_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Data",
                        data=csv,
                        file_name="earnings_analysis_data.csv",
                        mime="text/csv"
                    )
                
                # Summary statistics
                st.markdown("---")
                st.markdown("#### Summary Statistics")
                
                numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
                key_cols = ['EPS Surprise (%)', return_col, 'EPS Estimate', 'Reported EPS', 'P/E', 'Beta']
                key_cols = [c for c in key_cols if c in numeric_cols]
                
                if key_cols:
                    summary = analysis_df[key_cols].describe().T
                    st.dataframe(summary.round(2), use_container_width=True)

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
