# ================================================
# Earnings Momentum Quant Model - with Barchart Filter
# ================================================
# Builds two CSVs:
#   1. earnings_universe.csv   → Finviz + Barchart "Buy" tickers, fundamentals, earnings dates
#   2. returns_tracker.csv     → price/returns tracking post-earnings
# ================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time, re, os

# ------------------------------------
# CONFIG
# ------------------------------------

UNIVERSE_FILE = "earnings_universe.csv"
RETURNS_FILE  = "returns_tracker.csv"
         
FINVIZ_SCREENER = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# ------------------------------------
# SCRAPERS
# ------------------------------------
def get_all_tickers():
    """Scrape all tickers from Finviz screener."""
    tickers, offset = [], 0
    while True:
        url = f"{FINVIZ_SCREENER}&r={offset + 1}"
        soup = BeautifulSoup(requests.get(url, headers=HEADERS).text, "html.parser")
        new_tickers = []
        for row in soup.select("table tr"):
            cols = row.find_all("td")
            if len(cols) > 1:
                ticker = cols[1].text.strip()
                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    new_tickers.append(ticker)
        if not new_tickers:
            break
        tickers.extend(t for t in new_tickers if t not in tickers)
        offset += 20
    return tickers


def get_company_name(ticker):
    """Get company name and sector using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName') or info.get('shortName') or ticker
        sector = info.get('sector', 'N/A')
        return name, sector
    except Exception:
        return ticker, 'N/A'


def get_finviz_stats(ticker):
    """Scrape Finviz quote page for fundamentals."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=10).text, "html.parser")
        
        # Get company name and sector using yfinance
        company_name, sector = get_company_name(ticker)
        
        table = soup.find_all("table")[8]
        cells = table.find_all("td")
        data = {}
        for i in range(0, len(cells), 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            data[key] = value
        
        return {
            "Ticker": ticker,
            "Company Name": company_name,
            "Sector": sector,
            "Price": data.get("Price"),
            "EPS (TTM)": data.get("EPS (ttm)"),
            "P/E": data.get("P/E"),
            "Forward P/E": data.get("Forward P/E"),
            "Market Cap": data.get("Market Cap"),
            "Beta": data.get("Beta"),
            "Earnings Raw": data.get("Earnings")
        }
    except Exception as e:
        company_name, sector = get_company_name(ticker)
        return {"Ticker": ticker, "Company Name": company_name, "Sector": sector, "Error": "Failed"}


def has_buy_signal(ticker):
    """Check Barchart.com for Buy signal (same logic as website)."""
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        sig = soup.find("span", class_="opinion-signal buy")
        return bool(sig and "Buy" in sig.text)
    except Exception:
        return False

# ------------------------------------
# HELPERS
# ------------------------------------
def parse_earnings_date(val, default_year):
    """Convert 'Oct 23 BMO' → datetime(YYYY,10,23) and extract timing."""
    if pd.isna(val) or not isinstance(val, str) or len(val.strip()) < 3:
        return np.nan, None, None
    
    original_val = val  # Keep original string
    
    # Extract timing (BMO/AMC)
    timing = None
    if 'BMO' in val.upper():
        timing = 'BMO'
    elif 'AMC' in val.upper():
        timing = 'AMC'
    
    m = re.match(r"([A-Za-z]{3}) (\d{1,2})", val.strip())
    if not m:
        return np.nan, timing, original_val
    try:
        month_num = datetime.strptime(m.group(1), "%b").month
        return datetime(default_year, month_num, int(m.group(2))), timing, original_val
    except:
        return np.nan, timing, original_val


def get_earnings_quarter(earnings_date):
    """Determine the fiscal quarter based on earnings date.
    Most companies report 4-6 weeks after quarter end."""
    if pd.isna(earnings_date):
        return np.nan
    
    try:
        earnings_date = pd.to_datetime(earnings_date)
        month = earnings_date.month
        
        # Approximate mapping: earnings typically reported ~1 month after quarter end
        # Jan-Mar earnings → Q4 of previous year
        # Apr-Jun earnings → Q1
        # Jul-Sep earnings → Q2
        # Oct-Dec earnings → Q3
        if month in [1, 2, 3]:
            return "Q4"
        elif month in [4, 5, 6]:
            return "Q1"
        elif month in [7, 8, 9]:
            return "Q2"
        else:  # Oct, Nov, Dec
            return "Q3"
    except:
        return np.nan


def parse_market_cap(v):
    if pd.isna(v):
        return np.nan
    v = str(v).upper().replace("$", "").strip()
    try:
        if v.endswith("B"):
            return float(v[:-1]) * 1e9
        if v.endswith("M"):
            return float(v[:-1]) * 1e6
        if v.endswith("K"):
            return float(v[:-1]) * 1e3
        return float(v)
    except:
        return np.nan


def to_num(x):
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except:
        return np.nan


def get_base_price(px, date, timing):
    """Get the base price depending on earnings timing (BMO vs AMC)."""
    try:
        if timing == 'BMO':
            # BMO (Before Market Open) - use prior day's close (4PM)
            prior_date = date - timedelta(days=1)
            prior_prices = px.loc[px.index <= prior_date]
            if prior_prices.empty:
                return np.nan
            return prior_prices.iloc[-1]
        else:
            # AMC (After Market Close) or unknown - use same day's close (4PM)
            same_day_prices = px.loc[px.index <= date]
            if same_day_prices.empty:
                return np.nan
            return same_day_prices.iloc[-1]
    except:
        return np.nan


def get_return(px, date, days, timing):
    """Calculate return N days after earnings, respecting timing."""
    try:
        # Get base price based on timing
        if timing == 'BMO':
            # BMO - use prior day's close
            prior_date = date - timedelta(days=1)
            prior_prices = px.loc[px.index <= prior_date]
            if prior_prices.empty:
                return np.nan
            base = prior_prices.iloc[-1]
        else:
            # AMC or unknown - use same day's close
            same_day_prices = px.loc[px.index <= date]
            if same_day_prices.empty:
                return np.nan
            base = same_day_prices.iloc[-1]
        
        # Get price N days after earnings
        after = px.loc[px.index > date]
        if len(after) < days:
            return np.nan
        future_price = after.iloc[days - 1]
        return (future_price / base) - 1
    except:
        return np.nan


def get_return_to_today(px, date, timing):
    """Calculate return from earnings date to most recent trading day."""
    try:
        # Get base price based on timing
        if timing == 'BMO':
            prior_date = date - timedelta(days=1)
            prior_prices = px.loc[px.index <= prior_date]
            if prior_prices.empty:
                return np.nan
            base = prior_prices.iloc[-1]
        else:
            same_day_prices = px.loc[px.index <= date]
            if same_day_prices.empty:
                return np.nan
            base = same_day_prices.iloc[-1]
        
        after = px.loc[px.index > date]
        if after.empty:
            return np.nan
        latest = after.iloc[-1]
        return (latest / base) - 1
    except:
        return np.nan


def get_week_stats(px, date, days, timing):
    """Return high/low returns and absolute prices after earnings."""
    try:
        # Get base price based on timing
        if timing == 'BMO':
            prior_date = date - timedelta(days=1)
            prior_prices = px.loc[px.index <= prior_date]
            if prior_prices.empty:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            base = prior_prices.iloc[-1]
        else:
            same_day_prices = px.loc[px.index <= date]
            if same_day_prices.empty:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            base = same_day_prices.iloc[-1]
        
        after = px.loc[px.index > date].head(days)
        if after.empty:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        
        high_px = after.max()
        low_px = after.min()
        high_ret = (high_px / base) - 1
        low_ret = (low_px / base) - 1
        range_pct = (high_px - low_px) / base
        return (high_ret, low_ret, range_pct, high_px, low_px)
    except:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

# ------------------------------------
# 1️⃣ BUILD / UPDATE UNIVERSE
# ------------------------------------
def build_universe():
    today = datetime.today().date()
    finviz_tickers = get_all_tickers()
    print(f"[{today}] Found {len(finviz_tickers)} Finviz tickers...")

    # --- Apply Barchart Buy filter ---
    filtered_tickers = []
    for t in finviz_tickers:
        if has_buy_signal(t):
            filtered_tickers.append(t)
        time.sleep(0.3)
    print(f"[{today}] {len(filtered_tickers)} tickers passed Barchart Buy filter.")

    all_rows = []
    for t in filtered_tickers:
        row = get_finviz_stats(t)
        print(f"  {t}: {row.get('Company Name', 'N/A')} - {row.get('Sector', 'N/A')}")
        all_rows.append(row)
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    yr = datetime.today().year
    
    # Parse earnings date and timing
    if "Earnings Raw" in df.columns:
        parsed = df["Earnings Raw"].apply(lambda x: pd.Series(parse_earnings_date(x, yr)))
        df["Earnings Date"] = parsed[0]
        timing_col = parsed[1]  # Keep timing internally for calculations
        # Earnings Raw already exists in the data
    else:
        df["Earnings Date"] = np.nan
        timing_col = None

    # Add quarter column
    df["Quarter"] = df["Earnings Date"].apply(get_earnings_quarter)

    # Convert numeric columns
    for c in ["Price", "EPS (TTM)", "P/E", "Forward P/E", "Beta"]:
        if c in df.columns:
            df[c] = df[c].apply(to_num)
    if "Market Cap" in df.columns:
        df["Market Cap"] = df["Market Cap"].apply(parse_market_cap)
    if "Price" in df.columns:
        df["Price"] = df["Price"].round(2)

    # Merge with existing file
    if os.path.exists(UNIVERSE_FILE):
        existing = pd.read_csv(UNIVERSE_FILE)
        
        # Remove Scrape Date and Timing if they exist
        cols_to_remove = ["Scrape Date", "Timing"]
        for col in cols_to_remove:
            if col in existing.columns:
                existing = existing.drop(columns=[col])
        
        # Ensure Earnings Date is datetime
        if "Earnings Date" in existing.columns:
            existing["Earnings Date"] = pd.to_datetime(existing["Earnings Date"], errors='coerce')
        
        # Recalculate Quarter for existing data
        if "Earnings Date" in existing.columns:
            existing["Quarter"] = existing["Earnings Date"].apply(get_earnings_quarter)
        
        # Combine
        df = pd.concat([existing, df], ignore_index=True)
        
        # Remove duplicates: same Ticker AND Quarter (keep most recent)
        df.drop_duplicates(subset=["Ticker", "Quarter"], keep="last", inplace=True)
        
        # Backfill missing company names and sectors
        missing_data = df[(df["Company Name"].isna() | (df["Company Name"] == "")) | 
                          (df["Sector"].isna() | (df["Sector"] == ""))]
        if not missing_data.empty:
            print(f"\nBackfilling {len(missing_data)} missing company names and/or sectors...")
            for idx, row in missing_data.iterrows():
                ticker = row["Ticker"]
                company_name, sector = get_company_name(ticker)
                df.at[idx, "Company Name"] = company_name
                df.at[idx, "Sector"] = sector
                print(f"  {ticker}: {company_name} - {sector}")
                time.sleep(0.2)

    # Ensure Earnings Date is datetime
    df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors='coerce')
    
    # Sort by earnings date
    df = df.sort_values(by="Earnings Date", ascending=True).reset_index(drop=True)
    
    # Reorder columns: Ticker, Quarter, Price, ... , Earnings Raw, Earnings Date, Company Name, Sector
    base_cols = ["Ticker", "Quarter", "Price", "EPS (TTM)", "P/E", "Forward P/E", "Market Cap", "Beta", 
                 "Earnings Raw", "Earnings Date", "Company Name", "Sector"]
    # Keep any extra columns that might exist
    final_cols = [c for c in base_cols if c in df.columns]
    extra_cols = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + extra_cols]
    
    df.to_csv(UNIVERSE_FILE, index=False)
    print(f"✅ Updated {UNIVERSE_FILE} ({len(df)} total tickers)")
    return df

# ------------------------------------
# 2️⃣ UPDATE RETURNS TRACKER
# ------------------------------------
def update_returns(universe_df):
    """Update returns for all tickers in universe, filling in any missing data."""
    if os.path.exists(RETURNS_FILE):
        df_returns = pd.read_csv(RETURNS_FILE, parse_dates=["Earnings Date"])
        
        # Add Quarter column to existing returns if it doesn't exist
        if "Quarter" not in df_returns.columns:
            df_returns["Quarter"] = df_returns["Earnings Date"].apply(get_earnings_quarter)
    else:
        df_returns = pd.DataFrame()

    # Get all unique tickers from universe
    tickers = universe_df["Ticker"].dropna().unique().tolist()
    start = universe_df["Earnings Date"].min() - timedelta(days=5)
    end = datetime.today() + timedelta(days=2)
    print(f"\nDownloading price data for {len(tickers)} tickers...")

    price_data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)
    updated_rows = []

    print(f"\nCalculating returns for {len(universe_df)} ticker-date combinations...")
    for idx, row in universe_df.iterrows():
        t, d = row["Ticker"], row["Earnings Date"]
        
        # Extract timing from Earnings Raw for calculations
        timing = None
        if pd.notna(row.get("Earnings Raw")):
            earnings_raw = str(row.get("Earnings Raw"))
            if 'BMO' in earnings_raw.upper():
                timing = 'BMO'
            elif 'AMC' in earnings_raw.upper():
                timing = 'AMC'
        
        # Skip if earnings date is invalid
        if pd.isna(d):
            continue
            
        try:
            # Handle single ticker vs multiple tickers data structure
            if len(tickers) == 1:
                px = price_data["Close"].dropna()
            else:
                px = price_data[t]["Close"].dropna()
            
            base_price = get_base_price(px, d, timing)
            r1 = get_return(px, d, 1, timing)
            r3 = get_return(px, d, 3, timing)
            r5 = get_return(px, d, 5, timing)
            r7 = get_return(px, d, 7, timing)
            r10 = get_return(px, d, 10, timing)
            r_today = get_return_to_today(px, d, timing)
            high_ret, low_ret, rng, high_px, low_px = get_week_stats(px, d, 7, timing)

            updated_rows.append({
                "Ticker": t,
                "Quarter": row.get("Quarter", np.nan),
                "Earnings Date": d,
                "Price": round(base_price, 2) if not pd.isna(base_price) else np.nan,
                "EPS (TTM)": row.get("EPS (TTM)"),
                "P/E": row.get("P/E"),
                "Forward P/E": row.get("Forward P/E"),
                "Market Cap": row.get("Market Cap"),
                "Beta": row.get("Beta"),
                "1D Return": round(r1, 3) if not pd.isna(r1) else np.nan,
                "3D Return": round(r3, 3) if not pd.isna(r3) else np.nan,
                "5D Return": round(r5, 3) if not pd.isna(r5) else np.nan,
                "7D Return": round(r7, 3) if not pd.isna(r7) else np.nan,
                "10D Return": round(r10, 3) if not pd.isna(r10) else np.nan,
                "Return to Today": round(r_today, 3) if not pd.isna(r_today) else np.nan,
                "1W High Return": round(high_ret, 3) if not pd.isna(high_ret) else np.nan,
                "1W Low Return": round(low_ret, 3) if not pd.isna(low_ret) else np.nan,
                "1W Range": round(rng, 3) if not pd.isna(rng) else np.nan,
                "1W High Price": round(high_px, 2) if not pd.isna(high_px) else np.nan,
                "1W Low Price": round(low_px, 2) if not pd.isna(low_px) else np.nan,
                "Company Name": row.get("Company Name", t),
                "Sector": row.get("Sector", "N/A")
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(universe_df)} entries...")
        except Exception as e:
            print(f"  ⚠️ Error processing {t} on {d}: {str(e)}")
            continue

    df_new = pd.DataFrame(updated_rows)
    
    # Combine with existing returns, keeping the most recent calculation
    if not df_returns.empty:
        df_combined = pd.concat([df_returns, df_new], ignore_index=True)
        
        # Ensure Quarter column exists before deduplication
        if "Quarter" not in df_combined.columns:
            df_combined["Quarter"] = df_combined["Earnings Date"].apply(get_earnings_quarter)
        
        # Remove duplicates by Ticker AND Quarter (keep most recent)
        df_combined.drop_duplicates(subset=["Ticker", "Quarter"], keep="last", inplace=True)
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values(by="Earnings Date", ascending=True).reset_index(drop=True)
    
    # Reorder columns to match desired format: Ticker, Quarter at the start
    preferred_order = ["Ticker", "Quarter", "Earnings Date", "Price", "EPS (TTM)", "P/E", 
                      "Forward P/E", "Market Cap", "Beta", "1D Return", "3D Return", "5D Return",
                      "7D Return", "10D Return", "Return to Today", "1W High Return", "1W Low Return",
                      "1W Range", "1W High Price", "1W Low Price", "Company Name", "Sector"]
    
    final_cols = [c for c in preferred_order if c in df_combined.columns]
    extra_cols = [c for c in df_combined.columns if c not in final_cols]
    df_combined = df_combined[final_cols + extra_cols]
    
    df_combined.to_csv(RETURNS_FILE, index=False)
    print(f"✅ Updated {RETURNS_FILE} ({len(df_combined)} total entries)")
    print(f"   - Added/updated {len(df_new)} entries this run")

# ------------------------------------
# RUN PIPELINE
# ------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("EARNINGS MOMENTUM QUANT MODEL - PIPELINE START")
    print("=" * 60)
    
    # Step 1: Build/update universe with new tickers
    universe = build_universe()
    
    # Step 2: Calculate returns for all tickers (new and existing)
    update_returns(universe)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)