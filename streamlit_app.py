import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import html
import json

st.set_page_config(page_title="Earnings Week Momentum", page_icon="📈", layout="centered")

# -----------------------------
# Original screener logic (unchanged)
# -----------------------------
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

# -----------------------------
# Finviz quote page: get Price + Earnings text
# -----------------------------
def get_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    out = {"ticker": ticker, "price": None, "earnings": None}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        tbls = soup.find_all("table")
        # Defensive: ensure correct table exists
        if len(tbls) >= 9:
            cells = tbls[8].find_all("td")
            for i in range(0, len(cells), 2):
                key = cells[i].get_text(strip=True)
                val = cells[i + 1].get_text(strip=True)
                if key == "Price":
                    out["price"] = val
                elif key == "Earnings":
                    out["earnings"] = val
    except Exception:
        pass
    return out

# -----------------------------
# Parse earnings text "Oct 23 BMO" -> datetime for sorting
# -----------------------------
def parse_earnings_date(earn_str):
    try:
        parts = (earn_str or "").split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except Exception:
        pass
    # Put unparseable/empty at the bottom when sorting
    return datetime.max

# -----------------------------
# Barchart opinion: detect Buy
# -----------------------------
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

# -----------------------------
# UI
# -----------------------------
st.title("📈 Stock Checker")
st.subheader("Earnings this week • SMA20 crossed above SMA50 • Barchart = Buy")
run = st.button("Find Stocks")

if run:
    with st.spinner("Fetching screener tickers…"):
        tickers = get_all_tickers()

    rows = []
    with st.spinner("Checking Barchart signals and pulling Finviz data…"):
        for t in tickers:
            if has_buy_signal(t):
                data = get_finviz_data(t)
                rows.append({
                    "Ticker": data["ticker"],
                    "Price": data["price"] or "N/A",
                    "Earnings": data["earnings"] or "N/A",
                    "_sort_key": parse_earnings_date(data["earnings"])
                })

    # Sort by earnings date
    rows = sorted(rows, key=lambda r: r["_sort_key"])
    for r in rows:
        r.pop("_sort_key", None)

    if not rows:
        st.info("No tickers found with a Buy signal right now.")
    else:
        # -----------------------------
        # 1) Show a Streamlit DataFrame
        # -----------------------------
        df = pd.DataFrame(rows, columns=["Ticker", "Price", "Earnings"])
        st.write("### ✅ Tickers with Buy Signal (sorted by earliest earnings date)")
        st.dataframe(df, use_container_width=True)

        # -----------------------------
        # 2) Click-to-copy Ticker (HTML + JS)
        # -----------------------------
        st.write("#### Quick Copy (click the Ticker cell to copy)")
        safe_rows = [
            {
                "Ticker": html.escape(row["Ticker"]),
                "Price": html.escape(str(row["Price"])),
                "Earnings": html.escape(str(row["Earnings"]))
            } for row in rows
        ]
        data_json = json.dumps(safe_rows)

        st.components.v1.html(
            f"""
            <div id="copy-table-wrapper" style="font-family: ui-sans-serif, system-ui; max-width: 900px;">
              <style>
                #ctable {{
                  width: 100%;
                  border-collapse: collapse;
                  font-size: 14px;
                }}
                #ctable th, #ctable td {{
                  border-bottom: 1px solid #e6e6e6;
                  padding: 8px 10px;
                  text-align: left;
                }}
                #ctable th {{
                  background: #f7f7f7;
                  font-weight: 600;
                }}
                #ctable td.ticker {{
                  color: #1166ee;
                  cursor: pointer;
                  user-select: none;
                }}
                #toast {{
                  position: fixed;
                  bottom: 20px;
                  right: 20px;
                  background: #111;
                  color: #fff;
                  padding: 10px 14px;
                  border-radius: 6px;
                  opacity: 0;
                  transform: translateY(10px);
                  transition: all .25s ease;
                  font-size: 13px;
                  z-index: 9999;
                }}
                #toast.show {{
                  opacity: 0.95;
                  transform: translateY(0);
                }}
              </style>
              <table id="ctable">
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Price</th>
                    <th>Earnings</th>
                  </tr>
                </thead>
                <tbody id="tbody"></tbody>
              </table>
              <div id="toast">Copied!</div>
            </div>
            <script>
              const data = {data_json};
              const tbody = document.getElementById("tbody");
              data.forEach(row => {{
                const tr = document.createElement("tr");
                const tdTicker = document.createElement("td");
                tdTicker.textContent = row.Ticker;
                tdTicker.className = "ticker";
                tdTicker.title = "Click to copy";
                tdTicker.addEventListener("click", async () => {{
                  try {{
                    await navigator.clipboard.writeText(row.Ticker);
                    const toast = document.getElementById("toast");
                    toast.textContent = `Copied ${row.Ticker}`;
                    toast.classList.add("show");
                    setTimeout(() => toast.classList.remove("show"), 1100);
                  }} catch (e) {{
                    // no-op
                  }}
                }});

                const tdPrice = document.createElement("td");
                tdPrice.textContent = row.Price;

                const tdEarn = document.createElement("td");
                tdEarn.textContent = row.Earnings;

                tr.appendChild(tdTicker);
                tr.appendChild(tdPrice);
                tr.appendChild(tdEarn);
                tbody.appendChild(tr);
              }});
            </script>
            """,
            height=360 + min(32 * len(rows), 400),  # dynamic-ish height
            scrolling=True,
        )
else:
    st.caption("Click **Find Stocks** to fetch the current list, then click any Ticker to copy it.")
