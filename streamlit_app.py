import streamlit as st
import requests
from bs4 import BeautifulSoup
import time

# Initialize session state for tickers
if "tickers" not in st.session_state:
    st.session_state.tickers = []

# Progress bar and time estimation display
progress_bar = st.progress(0)
time_display = st.empty()

# Function to fetch tickers with estimated time
def get_all_tickers():
    base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    offset = 0
    tickers = []
    estimated_total_pages = 5  # Set an estimate for the number of pages to fetch
    start_time = time.time()

    for page in range(estimated_total_pages):
        url = f"{base_url}&r={offset + 1}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Process each row for tickers
        for row in soup.select("table tr"):
            columns = row.find_all("td")
            if len(columns) > 1:
                ticker = columns[1].text.strip()
                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    tickers.append(ticker)
        
        # Update progress and estimated time
        elapsed = time.time() - start_time
        avg_time_per_page = elapsed / (page + 1)
        estimated_time_remaining = avg_time_per_page * (estimated_total_pages - (page + 1))
        
        progress = (page + 1) / estimated_total_pages
        progress_bar.progress(progress)
        time_display.text(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
        
        offset += 20
        time.sleep(0.5)  # Simulate processing time if needed

    progress_bar.progress(1.0)  # Complete progress
    return tickers

# Single button to fetch tickers and check buy signals
if st.button("Fetch Tickers with Estimated Time"):
    st.session_state.tickers = get_all_tickers()
    st.write(f"Total tickers found: {len(st.session_state.tickers)}")
    st.write("All Tickers:", st.session_state.tickers)
