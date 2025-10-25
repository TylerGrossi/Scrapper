import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Initialize session state
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# --- Fetch tickers from Finviz screener (same logic as original) ---
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


# --- Helper: get price & earnings date from Finviz quote page ---
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    data = {"ticker": ticker, "price": None, "earnings": None}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find_all("table")[8]
        cells = table.find_all("td")

        for i in range(0, len(cells), 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            if key == "Price":
                data["price"] = value
            elif key == "Earnings":
                data["earnings"] = value
    except Exception:
        pass
    return data


# --- Helper: convert "Oct 23 BMO" → datetime for sorting ---
def parse_earnings_date(earn_str):
    try:
        parts = earn_str.split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except Exception:
        pass
    return datetime.max  # fallback for sorting if parsing fails


# --- Check Buy signals on Barchart ---
def check_buy_signal(tickers):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    buy_signals = []

    for ticker in tickers:
        url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            buy_signal = soup.find("span", class_="opinion-signal buy")
            if buy_signal and "Buy" in buy_signal.text:
                finviz_data = get_finviz_data(ticker)
                buy_signals.append(finviz_data)
        except Exception:
            continue

    buy_percentage = (len(buy_signals) / len(tickers)) * 100 if tickers else 0
    return buy_signals, buy_percentage


# --- Streamlit UI ---
st.title("📈 Stock Checker")
st.subheader("Earnings this week, SMA20 crossed SMA50 above, Buys")

if st.button("Find Stocks"):
    tickers = get_all_tickers()
    buy_stocks, buy_percentage = check_buy_signal(tickers)

    # Sort by earnings date (earliest first)
    buy_stocks = sorted(buy_stocks, key=lambda x: parse_earnings_date(x["earnings"] or ""))

    st.write("### ✅ Tickers with Buy Signal (Sorted by Earnings Date):")
    if buy_stocks:
        for stock in buy_stocks:
            ticker = stock["ticker"]
            price = stock["price"] or "N/A"
            earnings = stock["earnings"] or "N/A"
            st.markdown(
                f"<div style='padding:6px 0;'>"
                f"<b>{ticker}</b> — ${price} — {earnings} "
                f"<button onclick='navigator.clipboard.writeText(\"{ticker}\")' "
                f"style='margin-left:8px; background:#0078ff; color:white; border:none; padding:3px 8px; border-radius:4px; cursor:pointer;'>Copy</button>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.write("No tickers found with Buy signal.")

    st.write(f"**Buy Signal Percentage:** {buy_percentage:.2f}%")
