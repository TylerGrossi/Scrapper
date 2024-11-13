import streamlit as st
import requests
from bs4 import BeautifulSoup

# Initialize session state for tickers
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# Function to fetch tickers from Finviz
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
        soup = BeautifulSoup(response.text, 'html.parser')
        new_tickers = []

        # Locate rows containing tickers specifically
        for row in soup.select("table tr"):
            columns = row.find_all("td")
            if len(columns) > 1:
                ticker = columns[1].text.strip()
                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    new_tickers.append(ticker)

        # If no new tickers were found, we assume there are no more pages
        if not new_tickers:
            break

        tickers.extend(ticker for ticker in new_tickers if ticker not in tickers)
        offset += 20

    return tickers

# Function to check Buy signals
def check_buy_signal(tickers):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    buy_signals = []

    for ticker in tickers:
        url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            buy_signal = soup.find('span', class_='opinion-signal buy')
            if buy_signal and "Buy" in buy_signal.text:
                buy_signals.append(ticker)
        
        except requests.exceptions.RequestException as e:
            st.write(f"Error fetching page for {ticker}: {e}")

    # Calculate the percentage of tickers with a Buy signal
    buy_percentage = (len(buy_signals) / len(tickers)) * 100 if tickers else 0
    return buy_signals, buy_percentage

# Streamlit App Interface
st.title("Ticker Buy Signal Checker")

# Button to fetch all tickers
if st.button("Fetch Tickers"):
    st.session_state.tickers = get_all_tickers()
    st.write(f"Total tickers found: {len(st.session_state.tickers)}")
    st.write("Tickers:", st.session_state.tickers)

# Button to check buy signals
if st.button("Check Buy Signals"):
    if not st.session_state.tickers:
        st.write("Please fetch tickers first.")
    else:
        buy_tickers, buy_percentage = check_buy_signal(st.session_state.tickers)
        st.write("Tickers with Buy signal:", buy_tickers)
        st.write(f"Percentage of tickers with Buy signal: {buy_percentage:.2f}%")

# Streamlit App Interface
st.title("Ticker Buy Signal Checker")

if st.button("Fetch Tickers and Check Buy Signals"):
    st.session_state.tickers = get_all_tickers()
    buy_tickers, buy_percentage = check_buy_signals(st.session_state.tickers)
    
    st.write(f"Total tickers found: {len(st.session_state.tickers)}")
    st.write("All Tickers:", st.session_state.tickers)
    st.write("Tickers with Buy signal:", buy_tickers)
    st.write(f"Percentage of tickers with Buy signal: {buy_percentage:.2f}%")
