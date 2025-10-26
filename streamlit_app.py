import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Earnings Week Momentum", page_icon="📈", layout="wide")

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
            height: 750px !important;
        }
        .stDataFrame tbody tr td {
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            font-size: 1.05rem !important;
        }
    </style>
""", unsafe_allow_html=True)

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
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    }

    data = {
        "Ticker": ticker,
        "Earnings": None,
        "Price": None,
        "P/E": None,
        "Dividend": None,       # maps from "Dividend %"
        "52W Range": None,
        "Beta": None,
        "Market Cap (M)": None, # keep your column name; we’ll store the Finviz-formatted string (e.g., 8.60B / 53.19M)
    }

    try:
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Finviz shows two tables with this class; scan both to be safe.
        snap_tables = soup.select("table.snapshot-table2")
        if not snap_tables:
            return data  # page didn’t load correctly or layout changed

        keymap = {
            "Earnings": "Earnings",
            "Price": "Price",
            "P/E": "P/E",
            "Dividend %": "Dividend",
            "Dividend": "Dividend",        # just in case
            "52W Range": "52W Range",
            "Beta": "Beta",
            "Market Cap": "Market Cap (M)",
        }

        for t in snap_tables:
            tds = t.find_all("td")
            # cells are label/value pairs
            for i in range(0, len(tds) - 1, 2):
                k = tds[i].get_text(strip=True)
                v = tds[i + 1].get_text(strip=True)
                if k in keymap and (data[keymap[k]] is None or data[keymap[k]] in ("-", "N/A", "")):
                    data[keymap[k]] = v

        # Clean up edge cases
        for k in ["Dividend", "52W Range", "Price", "P/E", "Beta", "Earnings", "Market Cap (M)"]:
            if not data[k] or data[k] in ("-", "—", "--"):
                data[k] = "N/A"

        # Keep Market Cap exactly as Finviz shows (B/M suffix)
        # (Nothing to convert here — we already stored the formatted string.)

    except Exception:
        # leave defaults / NAs
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

# Centered Button
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
                    "Earnings": data["Earnings"] or "N/A",
                    "Price": data["Price"] or "N/A",
                    "P/E": data["P/E"] or "N/A",
                    "Dividend": data["Dividend"] or "N/A",
                    "52W Range": data["52W Range"] or "N/A",
                    "Beta": data["Beta"] or "N/A",
                    "Market Cap (M)": data["Market Cap (M)"] or "N/A",
                    "_sort_key": parse_earnings_date(data["Earnings"])
                })

    rows = sorted(rows, key=lambda r: r["_sort_key"])
    for r in rows:
        r.pop("_sort_key", None)

    if not rows:
        st.info("No tickers found with a Buy signal right now.")
    else:
        df = pd.DataFrame(rows, columns=["Ticker", "Earnings", "Price", "P/E", "Dividend", "52W Range", "Beta", "Market Cap (M)"])
        st.markdown("### ✅ Tickers with Buy Signal (sorted by earliest earnings date)")
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("Click **Find Stocks** to fetch the current list.")
