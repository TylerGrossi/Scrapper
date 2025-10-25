import streamlit as st
import requests
from bs4 import BeautifulSoup

# Initialize session state for tickers
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# Function to fetch tickers and earnings dates from Finviz
def get_all_tickers_with_dates():
    base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    offset = 0
    data = []

    while True:
        url = f"{base_url}&r={offset + 1}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table with the screener results
        table = soup.find("table", class_="table-light")
        if not table:
            break

        rows = table.find_all("tr")[1:]  # skip header row

        new_rows = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) > 1:
                ticker = cols[1].text.strip()
                earnings = None
                # Finviz often places the "Earnings" data in one of the last columns
                for c in cols:
                    if "BMO" in c.text or "AMC" in c.text:
                        earnings = c.text.strip()
                        break
                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    new_rows.append((ticker, earnings if earnings else "N/A"))

        if not new_rows:
            break

        data.extend(new_rows)
        offset += 20

    return data

# Function to check Buy signals
def check_buy_signal(tickers_with_dates):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    buy_signals = []

    for ticker, earnings in tickers_with_dates:
        url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            buy_signal = soup.find("span", class_="opinion-signal buy")
            if buy_signal and "Buy" in buy_signal.text:
                buy_signals.append((ticker, earnings))
        except requests.exceptions.RequestException as e:
            st.write(f"Error fetching {ticker}: {e}")

    buy_percentage = (
        (len(buy_signals) / len(tickers_with_dates)) * 100 if tickers_with_dates else 0
    )
    return buy_signals, buy_percentage

# Streamlit UI
st.title("📈 Stock Checker")
st.subheader("Earnings This Week + SMA20 Crossed Above SMA50 + Buy Signal")

if st.button("Find Stocks"):
    tickers_with_dates = get_all_tickers_with_dates()
    buy_tickers, buy_percentage = check_buy_signal(tickers_with_dates)

    st.write("### ✅ Tickers with Buy Signals:")
    if buy_tickers:
        df_display = [{"Ticker": t, "Earnings Date": e} for t, e in buy_tickers]
        st.dataframe(df_display)
    else:
        st.write("No tickers found with Buy signal.")

    st.write(f"**Buy Signal Percentage:** {buy_percentage:.2f}%")
