# Earnings Momentum Strategy

A quantitative trading strategy that identifies stocks with positive technical momentum ahead of earnings announcements.

## Strategy Overview

### Entry Criteria (All must be met)
1. **Earnings This Week** - Stock has earnings announcement scheduled for the current week
2. **SMA20 > SMA50** - 20-day simple moving average has crossed above the 50-day SMA (golden cross)
3. **Barchart Buy Signal** - Stock shows a "Buy" opinion signal on Barchart.com

### Entry Timing
- **BMO (Before Market Open)**: Buy at previous day's close (~4pm day before earnings)
- **AMC (After Market Close)**: Buy at earnings day close (~4pm on earnings day)

### Exit Rules
1. **Stop Loss: -10%** - Exit if position drops 10% from entry price
   - If stock gaps down and opens below -10%, take the actual gap loss (not -10%)
2. **Time Exit: Day 5** - Exit at market close on trading day 5 if stop loss not triggered
3. **No Profit Cap** - Let winners run; do not cap upside

## Model Rules Summary

| Rule | Description |
|------|-------------|
| Entry Signal | SMA20 crosses above SMA50 + Barchart Buy + Earnings this week |
| Entry Price (BMO) | Previous day's close |
| Entry Price (AMC) | Earnings day close |
| Stop Loss | -10% (or actual gap if opens lower) |
| Exit Day | Day 5 close |
| Position Sizing | Equal weight per trade |

## Key Metrics (Historical Backtest)

Based on ~177 trades from the universe:

| Metric | Value |
|--------|-------|
| Total Trades | 177 |
| Average Return/Trade | ~+5-6% |
| Win Rate | ~55-60% |
| Best Stop Loss Level | -10% |

## Files

| File | Description |
|------|-------------|
| `earnings_universe.csv` | Master list of tickers meeting entry criteria |
| `returns_tracker.csv` | Daily return calculations (1D, 3D, 5D, 7D, 10D) |
| `hourly_prices.csv` | Hourly price data for backtesting stop losses |
| `earnings_momentum_hourly.py` | Python script to update all data |
| `earnings_momentum_streamlit.py` | Streamlit web app for visualization |

## Data Columns

### earnings_universe.csv
- `Ticker` - Stock symbol
- `Fiscal Quarter` - e.g., "Q3 FY25"
- `Earnings Date` - Scheduled earnings announcement date
- `Earnings Timing` - BMO (Before Market Open) or AMC (After Market Close)
- `Date Check` - "OK" or "DATE PASSED" validation
- `Company Name`, `Sector`, `Market Cap`, etc.

### hourly_prices.csv
- `Ticker` - Stock symbol
- `Datetime` - Timestamp of price observation
- `Close` - Closing price for that hour
- `Base Price` - Entry price (based on BMO/AMC timing)
- `Trading Day` - 1-5 (days after entry)
- `Return From Earnings (%)` - Return from base price

### returns_tracker.csv
- `Ticker` - Stock symbol
- `1D Return` through `10D Return` - Returns at each holding period
- `1W High Return` / `1W Low Return` - Best and worst intraday returns in first week

## Running the Model

### Daily Update (GitHub Actions)
```bash
python earnings_momentum_hourly.py
```

This script:
1. Scans Finviz for stocks meeting SMA criteria with earnings this week
2. Filters through Barchart for buy signals
3. Updates earnings data and returns
4. Exports hourly prices for backtest visualization

### Streamlit App
```bash
streamlit run earnings_momentum_streamlit.py
```

## Important Notes

### Gap Down Handling
If a stock gaps down and the first candle opens below the stop loss level:
- You take the **actual gap loss**, not the stop loss price
- Example: Stop is -10%, stock opens at -15% → You lose 15%, not 10%
- This reflects real trading where you cannot exit at a price that was never traded

### Date Validation
The model validates earnings dates by comparing Finviz dates with yfinance:
- `DATE PASSED` = Finviz shows future date but earnings already happened
- These tickers are filtered out of backtests

### Data Limitations
- Hourly data from yfinance only available for last ~730 days
- Some tickers may not have complete hourly data

## Strategy Rationale

1. **Momentum** - SMA20 > SMA50 indicates positive price momentum
2. **Catalyst** - Earnings announcement provides potential for gap up
3. **Confirmation** - Barchart buy signal adds technical confirmation
4. **Risk Management** - 10% stop loss limits downside while allowing upside
5. **Time Decay** - Exit by Day 5 captures post-earnings momentum before mean reversion

## License

For personal/educational use only. Not financial advice.
