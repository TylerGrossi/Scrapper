import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

# --- Streamlit page config ---
st.set_page_config(page_title="Earnings Week Momentum", page_icon="📈", layout="wide")

# --- Custom CSS styling ---
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
        "User-Agent": "Mozilla/5.0"
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


# --- Parse Finviz data (Snapshot + Financial) ---
def get_finviz_data(ticker):
    headers = {"User-Agent": "Mozilla/5.0"}
    base_url = f"https://finviz.com/quote.ashx?t={ticker}"
    data = {
        "Ticker": ticker,
        "Earnings": "N/A",
        "Price": "N/A",
        "Forward P/E": "N/A",
        "Dividend": "N/A",
        "52W Range": "N/A",
        "Beta": "N/A",
        "Market Cap (B)": "N/A",
    }

    try:
        # --- Snapshot section ---
        r = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if len(tables) >= 9:
            cells = tables[8].find_all("td")
            for i in range(0, len(cells), 2):
                key = cells[i].get_text(strip=True)
                val = cells[i + 1].get_text(strip=True)
                if key == "Earnings":
                    data["Earnings"] = val
                elif key == "Price":
                    data["Price"] = val
                elif key == "Dividend %":
                    data["Dividend"] = val
                elif key == "52W Range":
                    data["52W Range"] = val
                elif key == "Beta":
                    data["Beta"] = val
                elif key == "Market Cap":
                    val = val.replace(",", "")
                    if "B" in val:
                        cap = float(val.replace("B", ""))
                    elif "M" in val:
                        cap = float(val.replace("M", "")) / 1000
                    else:
                        cap = float(val) if val.replace(".", "").isdigit() else None
                    if cap is not None:
                        data["Market Cap (B)"] = f"{cap:.2f}B"

        # --- Financial section (for Forward P/E etc.) ---
        r_fin = requests.get(f"{base_url}&p=financial", headers=headers, timeout=10)
        soup_fin = BeautifulSoup(r_fin.text, "html.parser")
        fin_table = soup_fin.find("table", class_="snapshot-table2")
        if fin_table:
            cells = fin_table.find_all("td")
            for i in range(0, len(cells), 2):
                key = cells[i].get_text(strip=True)
                val = cells[i + 1].get_text(strip=True)
                if key == "Forward P/E":
                    data["Forward P/E"] = val

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
    return data


# --- Parse date string (for sorting) ---
def parse_earnings_date(earn_str):
    try:
        parts = (earn_str or "").split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except Exception:
        pass
    return datetime.max


# --- Check Buy signal from Barchart ---
def has_buy_signal(ticker):
    headers = {"User-Agent": "Mozilla/5.0"}
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
    with st.spinner("Checking Barchart and gathering fundamentals..."):
        for t in tickers:
            if has_buy_signal(t):
                d = get_finviz_data(t)
                rows.append({
                    "Ticker": d["Ticker"],
                    "Earnings": d["Earnings"],
                    "Price": d["Price"],
                    "Forward P/E": d["Forward P/E"],
                    "Dividend": d["Dividend"],
                    "52W Range": d["52W Range"],
                    "Beta": d["Beta"],
                    "Market Cap (B)": d["Market Cap (B)"],
                    "_sort_key": parse_earnings_date(d["Earnings"]),
                })

    rows = sorted(rows, key=lambda r: r["_sort_key"])
    for r in rows:
        r.pop("_sort_key", None)

    if not rows:
        st.info("No tickers found with a Buy signal right now.")
    else:
        df = pd.DataFrame(rows, columns=["Ticker", "Earnings", "Price", "Forward P/E", "Dividend", "52W Range", "Beta", "Market Cap (B)"])
        st.markdown("### ✅ Tickers with Buy Signal (sorted by earliest earnings date)")
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("Click **Find Stocks** to fetch the current list.")
