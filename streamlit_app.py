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
    
    /* Surprise analysis cards */
    .surprise-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .surprise-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .surprise-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    .surprise-sub {
        font-size: 0.75rem;
        color: #64748b;
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


# =============================================================================
# EARNINGS SURPRISE ANALYSIS FUNCTIONS
# =============================================================================

def categorize_surprise(surprise_pct):
    """Categorize EPS surprise into buckets."""
    if pd.isna(surprise_pct):
        return 'N/A'
    elif surprise_pct <= -20:
        return 'Big Miss (≤-20%)'
    elif surprise_pct <= -5:
        return 'Miss (-20% to -5%)'
    elif surprise_pct < 5:
        return 'In-Line (-5% to 5%)'
    elif surprise_pct < 20:
        return 'Beat (5% to 20%)'
    elif surprise_pct < 50:
        return 'Strong Beat (20% to 50%)'
    else:
        return 'Blowout (≥50%)'

def get_surprise_color(category):
    """Get color for surprise category."""
    colors = {
        'Big Miss (≤-20%)': '#ef4444',
        'Miss (-20% to -5%)': '#f97316',
        'In-Line (-5% to 5%)': '#94a3b8',
        'Beat (5% to 20%)': '#22c55e',
        'Strong Beat (20% to 50%)': '#10b981',
        'Blowout (≥50%)': '#06b6d4',
        'N/A': '#64748b'
    }
    return colors.get(category, '#64748b')


# =============================================================================
# MAIN APP
# =============================================================================

st.title("Earnings Momentum Strategy")

tab1, tab2, tab3, tab4 = st.tabs(["Stock Screener", "Stop Loss Analysis", "Earnings Surprise Analysis", "PowerBI"])

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
                def detailed_hourly_backtest(hourly_df, returns_df, stop_loss=None, max_days=5):
                    results = []
                    trades = hourly_df.groupby(['Ticker', 'Earnings Date']).first().reset_index()[['Ticker', 'Earnings Date', 'Fiscal Quarter', 'Company Name', 'Base Price', 'Earnings Timing']]
                    
                    for _, trade in trades.iterrows():
                        ticker = trade['Ticker']
                        earnings_date = trade['Earnings Date']
                        fiscal_quarter = trade.get('Fiscal Quarter', '')
                        company_name = trade.get('Company Name', ticker)
                        base_price = trade.get('Base Price', None)
                        earnings_timing = trade.get('Earnings Timing', '')
                        
                        trade_data = hourly_df[
                            (hourly_df['Ticker'] == ticker) & 
                            (hourly_df['Earnings Date'] == earnings_date) &
                            (hourly_df['Trading Day'] >= 1)
                        ].sort_values('Datetime')
                        
                        if trade_data.empty:
                            continue
                        
                        return_5d = None
                        if returns_df is not None:
                            match = returns_df[
                                (returns_df['Ticker'] == ticker) & 
                                (returns_df['Earnings Date'].dt.date == earnings_date.date() if pd.notna(earnings_date) else False)
                            ]
                            if not match.empty and '5D Return' in match.columns:
                                return_5d = match['5D Return'].iloc[0]
                        
                        exit_day_data = trade_data[trade_data['Trading Day'] == max_days]
                        if exit_day_data.empty:
                            for target_day in [max_days - 1, max_days + 1, max_days - 2]:
                                exit_day_data = trade_data[trade_data['Trading Day'] == target_day]
                                if not exit_day_data.empty:
                                    break
                        
                        if exit_day_data.empty:
                            continue
                        
                        exit_day_close_return = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
                        actual_exit_day = int(exit_day_data['Trading Day'].iloc[-1])
                        
                        exit_reason = None
                        exit_return = None
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
                            
                            if stop_loss is not None and hour_return_decimal <= stop_loss and not stopped_out:
                                exit_trading_day = trading_day
                                exit_hour = hour_data.get('Time', hour_data.get('Hour', ''))
                                
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
                        
                        if not stopped_out:
                            exit_trading_day = actual_exit_day
                            exit_hour = exit_day_data['Time'].iloc[-1] if 'Time' in exit_day_data.columns else ''
                            exit_reason = f'Day {actual_exit_day} Close'
                            exit_return = exit_day_close_return
                        
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
                
                results = detailed_hourly_backtest(hourly_df, returns_df, stop_loss=stop_loss, max_days=max_days)
            
            if results.empty:
                st.warning("No trades found with complete data.")
            else:
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
            
            total_return = results['Backtest Return'].sum() * 100
            avg_return = results['Backtest Return'].mean() * 100
            win_rate = (results['Backtest Return'] > 0).mean() * 100
            n_trades = len(results)
            stopped_count = results['Stopped Out'].sum()
            held_count = n_trades - stopped_count
            
            results_with_5d = results[results['5D Return'].notna()]
            if len(results_with_5d) > 0:
                avg_5d_return = results_with_5d['5D Return'].mean() * 100
                total_5d_return = results_with_5d['5D Return'].sum() * 100
                avg_diff = results_with_5d['Diff vs 5D'].mean() * 100
            else:
                avg_5d_return = None
                total_5d_return = None
                avg_diff = None
            
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
                    st.metric("Total Diff", f"{diff_total:+.1f}%")
                    st.metric("Avg Diff", f"{avg_diff:+.2f}%")
                
                if diff_total > 0:
                    st.success(f"✅ Stop loss strategy **outperformed** buy & hold by {diff_total:+.1f}%")
                else:
                    st.warning(f"⚠️ Stop loss strategy **underperformed** buy & hold by {diff_total:.1f}%")
            
            st.markdown("---")
            st.markdown("### 🚪 Exit Breakdown")
            
            gap_down_count = results['Gap Down'].sum() if 'Gap Down' in results.columns else 0
            stop_loss_count = stopped_count - gap_down_count
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gap_trades = results[results['Gap Down'] == True] if 'Gap Down' in results.columns else pd.DataFrame()
                gap_avg = gap_trades['Backtest Return'].mean() * 100 if len(gap_trades) > 0 else 0
                
                st.markdown(f"""
                <div class="exit-card">
                    <div class="exit-count">{gap_down_count}</div>
                    <div class="exit-pct">({gap_down_count/n_trades*100:.1f}% of trades)</div>
                    <div class="exit-return" style="color: #ef4444;">{gap_avg:+.2f}% avg</div>
                    <div class="exit-label">📉 Gap Down</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
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
            
            st.markdown("---")
            st.markdown("### 📋 All Trades")
            
            display_df = results.copy()
            display_df['Earnings Date'] = pd.to_datetime(display_df['Earnings Date']).dt.strftime('%Y-%m-%d')
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
            
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"backtest_results_stop{int(stop_loss*100) if stop_loss else 'none'}_day{max_days}.csv",
                mime="text/csv"
            )

# =============================================================================
# TAB 3: EARNINGS SURPRISE ANALYSIS
# =============================================================================
with tab3:
    st.subheader("📊 Earnings Surprise Analysis")
    st.markdown("Analyze how EPS surprise magnitude affects post-earnings returns.")
    
    returns_df_surprise = load_returns_data()
    
    if returns_df_surprise is None or returns_df_surprise.empty:
        st.warning("No returns data available. Please ensure returns_tracker.csv is accessible.")
    else:
        df_with_surprise = returns_df_surprise[returns_df_surprise['EPS Surprise (%)'].notna()].copy()
        
        if df_with_surprise.empty:
            st.warning("No trades found with EPS Surprise data.")
        else:
            df_with_surprise['Surprise Category'] = df_with_surprise['EPS Surprise (%)'].apply(categorize_surprise)
            
            # Summary metrics
            st.markdown("### 📈 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="surprise-card">
                    <div class="surprise-value metric-white">{len(df_with_surprise)}</div>
                    <div class="surprise-label">Trades with EPS Data</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_surprise = df_with_surprise['EPS Surprise (%)'].mean()
                st.markdown(f"""
                <div class="surprise-card">
                    <div class="surprise-value metric-{'green' if avg_surprise > 0 else 'red'}">{avg_surprise:+.1f}%</div>
                    <div class="surprise-label">Avg EPS Surprise</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                beats = (df_with_surprise['EPS Surprise (%)'] > 0).sum()
                beat_pct = beats / len(df_with_surprise) * 100
                st.markdown(f"""
                <div class="surprise-card">
                    <div class="surprise-value metric-green">{beat_pct:.1f}%</div>
                    <div class="surprise-label">Beat Rate</div>
                    <div class="surprise-sub">{beats} of {len(df_with_surprise)} stocks</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if '5D Return' in df_with_surprise.columns:
                    avg_5d = df_with_surprise['5D Return'].mean() * 100
                    st.markdown(f"""
                    <div class="surprise-card">
                        <div class="surprise-value metric-{'green' if avg_5d > 0 else 'red'}">{avg_5d:+.2f}%</div>
                        <div class="surprise-label">Avg 5D Return</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Performance by Surprise Category
            st.markdown("### 🎯 Performance by Surprise Category")
            
            category_order = ['Big Miss (≤-20%)', 'Miss (-20% to -5%)', 'In-Line (-5% to 5%)', 
                             'Beat (5% to 20%)', 'Strong Beat (20% to 50%)', 'Blowout (≥50%)']
            
            category_stats = []
            for cat in category_order:
                cat_df = df_with_surprise[df_with_surprise['Surprise Category'] == cat]
                if len(cat_df) > 0:
                    stats = {
                        'Category': cat,
                        'Count': len(cat_df),
                        'Avg Surprise': cat_df['EPS Surprise (%)'].mean(),
                        '1D Return': cat_df['1D Return'].mean() * 100 if '1D Return' in cat_df.columns else None,
                        '3D Return': cat_df['3D Return'].mean() * 100 if '3D Return' in cat_df.columns else None,
                        '5D Return': cat_df['5D Return'].mean() * 100 if '5D Return' in cat_df.columns else None,
                        '7D Return': cat_df['7D Return'].mean() * 100 if '7D Return' in cat_df.columns else None,
                        'Win Rate 5D': (cat_df['5D Return'] > 0).mean() * 100 if '5D Return' in cat_df.columns else None,
                    }
                    category_stats.append(stats)
            
            cat_stats_df = pd.DataFrame(category_stats)
            
            if not cat_stats_df.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure()
                    time_periods = [('1D Return', '#3b82f6'), ('3D Return', '#8b5cf6'), 
                                   ('5D Return', '#22c55e'), ('7D Return', '#f59e0b')]
                    
                    for period, color in time_periods:
                        if period in cat_stats_df.columns and cat_stats_df[period].notna().any():
                            fig.add_trace(go.Bar(
                                name=period.replace(' Return', ''),
                                x=cat_stats_df['Category'],
                                y=cat_stats_df[period],
                                marker_color=color
                            ))
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                    fig.update_layout(
                        title="Average Returns by Surprise Category",
                        xaxis_title="Surprise Category",
                        yaxis_title="Average Return (%)",
                        barmode='group',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                        xaxis=dict(tickangle=-45)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Category Breakdown**")
                    display_stats = cat_stats_df.copy()
                    display_stats['Avg Surprise'] = display_stats['Avg Surprise'].apply(lambda x: f"{x:+.1f}%")
                    for col in ['1D Return', '3D Return', '5D Return', '7D Return']:
                        if col in display_stats.columns:
                            display_stats[col] = display_stats[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                    if 'Win Rate 5D' in display_stats.columns:
                        display_stats['Win Rate 5D'] = display_stats['Win Rate 5D'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                    
                    st.dataframe(display_stats, use_container_width=True, hide_index=True, height=300)
            
            st.markdown("---")
            
            # Correlation Analysis
            st.markdown("### 📉 Correlation: Surprise vs Returns")
            col1, col2 = st.columns(2)
            
            with col1:
                if '5D Return' in df_with_surprise.columns:
                    fig = px.scatter(
                        df_with_surprise,
                        x='EPS Surprise (%)',
                        y=df_with_surprise['5D Return'] * 100,
                        color='Surprise Category',
                        color_discrete_map={cat: get_surprise_color(cat) for cat in category_order},
                        hover_data=['Ticker', 'Company Name'] if 'Company Name' in df_with_surprise.columns else ['Ticker'],
                        title="EPS Surprise vs 5-Day Return"
                    )
                    
                    # Add trend line
                    x_clean = df_with_surprise['EPS Surprise (%)'].dropna()
                    y_clean = (df_with_surprise.loc[x_clean.index, '5D Return'] * 100).dropna()
                    common_idx = x_clean.index.intersection(y_clean.index)
                    
                    if len(common_idx) > 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            x_clean.loc[common_idx], y_clean.loc[common_idx]
                        )
                        x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
                        y_range = slope * x_range + intercept
                        
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_range,
                            mode='lines',
                            name=f'Trend (R²={r_value**2:.3f})',
                            line=dict(color='#f1f5f9', dash='dash', width=2)
                        ))
                    
                    fig.add_hline(y=0, line_dash="dot", line_color="#475569")
                    fig.add_vline(x=0, line_dash="dot", line_color="#475569")
                    
                    fig.update_layout(
                        xaxis_title="EPS Surprise (%)",
                        yaxis_title="5-Day Return (%)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=450,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.5, xanchor="center")
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Correlation matrix
                corr_cols = ['EPS Surprise (%)', '1D Return', '3D Return', '5D Return', '7D Return', '10D Return']
                corr_cols = [c for c in corr_cols if c in df_with_surprise.columns]
                
                if len(corr_cols) > 1:
                    corr_df = df_with_surprise[corr_cols].copy()
                    for col in corr_cols:
                        if 'Return' in col:
                            corr_df[col] = corr_df[col] * 100
                    
                    corr_matrix = corr_df.corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        aspect='auto',
                        title="Correlation Matrix"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=450
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if 'EPS Surprise (%)' in corr_matrix.columns:
                        st.markdown("**Key Correlations with EPS Surprise:**")
                        for col in ['1D Return', '3D Return', '5D Return', '7D Return']:
                            if col in corr_matrix.columns:
                                corr_val = corr_matrix.loc['EPS Surprise (%)', col]
                                color = 'green' if corr_val > 0 else 'red'
                                st.markdown(f"- {col}: **:{color}[{corr_val:+.3f}]**")
            
            st.markdown("---")
            
            # Distribution Analysis
            st.markdown("### 📊 Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df_with_surprise['EPS Surprise (%)'],
                    nbinsx=40,
                    marker_color='#3b82f6',
                    name='EPS Surprise'
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="#ef4444", annotation_text="0%")
                fig.add_vline(x=df_with_surprise['EPS Surprise (%)'].median(), 
                             line_dash="dash", line_color="#22c55e", 
                             annotation_text=f"Median: {df_with_surprise['EPS Surprise (%)'].median():.1f}%")
                fig.update_layout(
                    title="EPS Surprise Distribution",
                    xaxis_title="EPS Surprise (%)",
                    yaxis_title="Count",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cat_counts = df_with_surprise['Surprise Category'].value_counts()
                colors = [get_surprise_color(cat) for cat in cat_counts.index]
                
                fig = go.Figure(data=[go.Pie(
                    labels=cat_counts.index,
                    values=cat_counts.values,
                    marker_colors=colors,
                    textinfo='label+percent',
                    textposition='outside',
                    hole=0.4
                )])
                fig.update_layout(
                    title="Surprise Category Distribution",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Sector Analysis
            if 'Sector' in df_with_surprise.columns:
                st.markdown("### 🏢 Sector Analysis")
                
                sector_stats = df_with_surprise.groupby('Sector').agg({
                    'EPS Surprise (%)': ['mean', 'count'],
                    '5D Return': 'mean' if '5D Return' in df_with_surprise.columns else 'count'
                }).round(2)
                
                if '5D Return' in df_with_surprise.columns:
                    sector_stats.columns = ['Avg Surprise', 'Count', 'Avg 5D Return']
                    sector_stats['Avg 5D Return'] = sector_stats['Avg 5D Return'] * 100
                else:
                    sector_stats.columns = ['Avg Surprise', 'Count', 'Count2']
                    sector_stats = sector_stats.drop(columns=['Count2'])
                
                sector_stats = sector_stats.sort_values('Count', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        sector_stats.reset_index(),
                        x='Sector',
                        y='Avg Surprise',
                        color='Avg Surprise',
                        color_continuous_scale='RdYlGn',
                        title="Average EPS Surprise by Sector"
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="Avg EPS Surprise (%)",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        height=400,
                        xaxis=dict(tickangle=-45),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Sector Statistics**")
                    display_sector = sector_stats.reset_index()
                    display_sector['Avg Surprise'] = display_sector['Avg Surprise'].apply(lambda x: f"{x:+.1f}%")
                    if 'Avg 5D Return' in display_sector.columns:
                        display_sector['Avg 5D Return'] = display_sector['Avg 5D Return'].apply(lambda x: f"{x:+.2f}%")
                    st.dataframe(display_sector, use_container_width=True, hide_index=True, height=350)
            
            st.markdown("---")
            
            # Optimal Surprise Threshold Analysis
            st.markdown("### 🎯 Optimal Surprise Threshold")
            st.markdown("Find the EPS surprise threshold that maximizes returns.")
            
            if '5D Return' in df_with_surprise.columns:
                thresholds = list(range(-50, 110, 10))
                threshold_results = []
                
                for thresh in thresholds:
                    above_thresh = df_with_surprise[df_with_surprise['EPS Surprise (%)'] >= thresh]
                    if len(above_thresh) >= 5:
                        avg_ret = above_thresh['5D Return'].mean() * 100
                        win_rate = (above_thresh['5D Return'] > 0).mean() * 100
                        threshold_results.append({
                            'Threshold': thresh,
                            'Avg 5D Return': avg_ret,
                            'Win Rate': win_rate,
                            'Count': len(above_thresh)
                        })
                
                if threshold_results:
                    thresh_df = pd.DataFrame(threshold_results)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Scatter(
                                x=thresh_df['Threshold'],
                                y=thresh_df['Avg 5D Return'],
                                mode='lines+markers',
                                name='Avg 5D Return',
                                line=dict(color='#3b82f6', width=2),
                                marker=dict(size=8)
                            ),
                            secondary_y=False
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=thresh_df['Threshold'],
                                y=thresh_df['Win Rate'],
                                mode='lines+markers',
                                name='Win Rate',
                                line=dict(color='#22c55e', width=2, dash='dash'),
                                marker=dict(size=8)
                            ),
                            secondary_y=True
                        )
                        
                        fig.add_hline(y=0, line_dash="dot", line_color="#475569", secondary_y=False)
                        
                        fig.update_layout(
                            title="Performance by Minimum Surprise Threshold",
                            xaxis_title="Minimum EPS Surprise (%)",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#94a3b8',
                            height=400,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
                        )
                        fig.update_yaxes(title_text="Avg 5D Return (%)", secondary_y=False, gridcolor='#1e293b')
                        fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
                        fig.update_xaxes(gridcolor='#1e293b')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        best_idx = thresh_df['Avg 5D Return'].idxmax()
                        best_thresh = thresh_df.loc[best_idx, 'Threshold']
                        best_return = thresh_df.loc[best_idx, 'Avg 5D Return']
                        best_win = thresh_df.loc[best_idx, 'Win Rate']
                        best_count = thresh_df.loc[best_idx, 'Count']
                        
                        st.markdown("**Optimal Threshold Analysis**")
                        
                        st.markdown(f"""
                        <div class="surprise-card">
                            <div class="surprise-value metric-blue">≥{best_thresh}%</div>
                            <div class="surprise-label">Optimal Surprise Threshold</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric("Avg 5D Return", f"{best_return:+.2f}%")
                        st.metric("Win Rate", f"{best_win:.1f}%")
                        st.metric("Sample Size", f"{best_count} trades")
                        
                        st.markdown("---")
                        st.markdown("**Threshold Table**")
                        display_thresh = thresh_df.copy()
                        display_thresh['Threshold'] = display_thresh['Threshold'].apply(lambda x: f"≥{x}%")
                        display_thresh['Avg 5D Return'] = display_thresh['Avg 5D Return'].apply(lambda x: f"{x:+.2f}%")
                        display_thresh['Win Rate'] = display_thresh['Win Rate'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_thresh, use_container_width=True, hide_index=True, height=250)
            
            st.markdown("---")
            
            # Individual Trades Table
            st.markdown("### 📋 Individual Trades with Surprise Data")
            
            display_cols = ['Ticker', 'Company Name', 'Earnings Date', 'EPS Estimate', 'Reported EPS', 
                           'EPS Surprise (%)', 'Surprise Category', '1D Return', '3D Return', '5D Return', 'Sector']
            display_cols = [c for c in display_cols if c in df_with_surprise.columns]
            
            display_surprise = df_with_surprise[display_cols].copy()
            display_surprise['Earnings Date'] = pd.to_datetime(display_surprise['Earnings Date']).dt.strftime('%Y-%m-%d')
            
            for col in ['1D Return', '3D Return', '5D Return', '7D Return', '10D Return']:
                if col in display_surprise.columns:
                    display_surprise[col] = display_surprise[col] * 100
            
            st.dataframe(
                display_surprise.sort_values('EPS Surprise (%)', ascending=False),
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    "EPS Surprise (%)": st.column_config.NumberColumn("EPS Surprise", format="%+.1f%%"),
                    "1D Return": st.column_config.NumberColumn("1D Return", format="%+.2f%%"),
                    "3D Return": st.column_config.NumberColumn("3D Return", format="%+.2f%%"),
                    "5D Return": st.column_config.NumberColumn("5D Return", format="%+.2f%%"),
                }
            )

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
