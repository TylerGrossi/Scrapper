import streamlit as st
import requests
from bs4 import BeautifulSoup

# Initialize session state for tickers
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# --- Function to fetch tickers + earnings dates from Finviz ---
def get_all_tickers_with_dates():
    base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    offset = 0
    ticker_data = []

    while True:
        url = f"{base_url}&r={offset + 1}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        new_rows = []

        # Locate all table rows
        for row in soup.select("table tr"):
            columns = row.find_all("td")
            if len(columns) > 1:
                ticker = columns[1].text.strip()

                # Look for the earnings date text (BMO / AMC) in the same row
                earnings_date = None
                for c in columns:
                    if "BMO" in c.text or "AMC" in c.text:
                        earnings_date = c.text.strip()
                        break

                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    new_rows.append((ticker, earnings_date if earnings_date else "N/A"))

        if not new_rows:
            break

        ticker_data.extend(new_rows)
        offset += 20

    return ticker_data


# --- Function to check Buy signals ---
def check_buy_signal(ticker_data):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    buy_signals = []

    for ticker, earnings_date in ticker_data:
        url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            buy_signal = soup.find('span', class_='opinion-signal buy')
            if buy_signal and "Buy" in buy_signal.text:
                buy_signals.append((ticker, earnings_date))
        
        except requests.exceptions.RequestException as e:
            st.write(f"Error fetching page for {ticker}: {e}")

    buy_percentage = (len(buy_signals) / len(ticker_data)) * 100 if ticker_data else 0
    return buy_signals, buy_percentage


# --- Streamlit App Interface ---
st.title("📈 Stock Checker")
st.subheader("Earnings this week, SMA20 crossed SMA50 above, Buys")

if st.button("Find Stocks"):
    ticker_data = get_all_tickers_with_dates()
    buy_tickers, buy_percentage = check_buy_signal(ticker_data)
    
    # Show tickers + earnings dates for Buy signals
    st.write("### ✅ Tickers with Buy Signal:")
    if buy_tickers:
        df = [{"Ticker": t, "Earnings Date": d} for t, d in buy_tickers]
        st.dataframe(df)
    else:
        st.write("No tickers found with Buy signal.")

    st.write(f"**Buy Signal Percentage:** {buy_percentage:.2f}%")
