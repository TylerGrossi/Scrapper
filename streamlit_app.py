import streamlit as st
import requests
from bs4 import BeautifulSoup

# Initialize session state
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# --- Function to fetch tickers from Finviz (screener) ---
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

        tickers.extend(ticker for ticker in new_tickers if ticker not in tickers)
        offset += 20

    return tickers


# --- Helper: fetch earnings date from Finviz quote page ---
def get_earnings_date(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find_all("table")[8]
        cells = table.find_all("td")

        for i in range(0, len(cells), 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            if key == "Earnings":
                return value
    except Exception:
        return "N/A"
    return "N/A"


# --- Function to check Buy signals ---
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
                earnings = get_earnings_date(ticker)
                buy_signals.append((ticker, earnings))
        except requests.exceptions.RequestException as e:
            st.write(f"Error fetching {ticker}: {e}")

    buy_percentage = (len(buy_signals) / len(tickers)) * 100 if tickers else 0
    return buy_signals, buy_percentage


# --- Streamlit UI ---
st.title("📈 Stock Checker")
st.subheader("Earnings this week, SMA20 crossed SMA50 above, Buys")

if st.button("Find Stocks"):
    tickers = get_all_tickers()
    buy_tickers, buy_percentage = check_buy_signal(tickers)

    st.write("### ✅ Tickers with Buy Signal:")
    if buy_tickers:
        for ticker, earnings in buy_tickers:
            st.markdown(
                f"**{ticker}** — Earnings: {earnings} "
                f"[📋 Copy](javascript:navigator.clipboard.writeText('{ticker}'))",
                unsafe_allow_html=True,
            )
    else:
        st.write("No tickers found with Buy signal.")

    st.write(f"**Buy Signal Percentage:** {buy_percentage:.2f}%")
