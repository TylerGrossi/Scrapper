import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Earnings Week Momentum", page_icon="📈", layout="wide")

# --- Custom CSS ---
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
        /* ---- Table display ---- */
        [data-testid="stDataFrame"] {
            height: auto !important;  /* let it expand naturally */
            max-height: none !important;
        }
        .stDataFrame {
            overflow: visible !important;  /* disable inner scroll */
        }
        .stDataFrame tbody tr td {
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            font-size: 1.05rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Parse "Oct 23 BMO" -> datetime for sorting ---
def parse_earnings_date(earn_str):
    try:
        parts = (earn_str or "").split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except Exception:
        pass
    return datetime.max


# --- Get tickers from Finviz screener ---
def get_all_tickers():
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


# --- Get Finviz quote metrics ---
def get_finviz_data(ticker):
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
        "Market Cap (M)": "N/A"
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
            "Market Cap": "Market Cap (M)"
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


# --- Check Buy signal from Barchart ---
def has_buy_signal(ticker):
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


# --- Streamlit UI ---
st.title("📈 Stock Checker")
st.subheader("Earnings this week • SMA20 crossed above SMA50 • Barchart = Buy")

run = st.button("Find Stocks")

if run:
    with st.spinner("Fetching Finviz screener..."):
        tickers = get_all_tickers()

    rows = []
    with st.spinner("Checking Barchart and pulling Finviz data..."):
        for t in tickers:
            if has_buy_signal(t):
                data = get_finviz_data(t)
                rows.append({
                    "Ticker": data["Ticker"],
                    "Earnings": data["Earnings"],
                    "Price": data["Price"],
                    "P/E": data["P/E"],
                    "Beta": data["Beta"],
                    "Market Cap (M)": data["Market Cap (M)"],
                    "_sort_key": parse_earnings_date(data["Earnings"])
                })

    rows = sorted(rows, key=lambda r: r["_sort_key"])
    for r in rows:
        r.pop("_sort_key", None)

    if not rows:
        st.info("No tickers found with a Buy signal right now.")
    else:
        df = pd.DataFrame(rows, columns=["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap (M)"])
        st.markdown("### ✅ Tickers with Buy Signal (sorted by earliest earnings date)")
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("Click **Find Stocks** to fetch the current list.")
