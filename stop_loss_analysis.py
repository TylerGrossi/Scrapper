import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------
# BACKTEST FUNCTIONS
# ------------------------------------

def detailed_hourly_backtest(hourly_df, returns_df, stop_loss=None, max_days=5):
    """
    Detailed backtest with gap down handling.
    If stock gaps down and opens below stop loss on first candle,
    take the actual gap loss, not the stop loss price.
    """
    results = []
    
    trades = hourly_df.groupby(['Ticker', 'Earnings Date']).first().reset_index()[
        ['Ticker', 'Earnings Date', 'Fiscal Quarter', 'Company Name', 'Base Price', 'Earnings Timing']
    ]
    
    for _, trade in trades.iterrows():
        ticker = trade['Ticker']
        earnings_date = trade['Earnings Date']
        fiscal_quarter = trade.get('Fiscal Quarter', '')
        company_name = trade.get('Company Name', ticker)
        base_price = trade.get('Base Price', None)
        earnings_timing = trade.get('Earnings Timing', '')
        
        trade_data = hourly_df[
            (hourly_df['Ticker'] == ticker) & 
            (hourly_df['Earnings Date'] == earnings_date) &
            (hourly_df['Trading Day'] >= 1)
        ].sort_values('Datetime')
        
        if trade_data.empty:
            continue
        
        return_5d = None
        if returns_df is not None:
            match = returns_df[
                (returns_df['Ticker'] == ticker) & 
                (returns_df['Earnings Date'].dt.date == earnings_date.date() if pd.notna(earnings_date) else False)
            ]
            if not match.empty and '5D Return' in match.columns:
                return_5d = match['5D Return'].iloc[0]
        
        exit_day_data = trade_data[trade_data['Trading Day'] == max_days]
        if exit_day_data.empty:
            for target_day in [max_days - 1, max_days + 1, max_days - 2]:
                exit_day_data = trade_data[trade_data['Trading Day'] == target_day]
                if not exit_day_data.empty:
                    break
        
        if exit_day_data.empty:
            continue
        
        exit_day_close_return = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
        actual_exit_day = int(exit_day_data['Trading Day'].iloc[-1])
        
        exit_reason = None
        exit_return = None
        exit_datetime = None
        exit_trading_day = None
        exit_hour = None
        max_return = 0
        min_return = 0
        stopped_out = False
        gap_down = False
        first_candle = True
        
        for _, hour_data in trade_data.iterrows():
            trading_day = int(hour_data['Trading Day'])
            hour_return = hour_data['Return From Earnings (%)']
            
            if trading_day > actual_exit_day:
                break
            
            if pd.isna(hour_return):
                continue
            
            hour_return_decimal = hour_return / 100
            max_return = max(max_return, hour_return_decimal)
            min_return = min(min_return, hour_return_decimal)
            
            if stop_loss is not None and hour_return_decimal <= stop_loss and not stopped_out:
                exit_trading_day = trading_day
                exit_hour = hour_data.get('Time', hour_data.get('Hour', ''))
                exit_datetime = hour_data.get('Datetime', None)
                
                if first_candle and hour_return_decimal < stop_loss:
                    exit_return = hour_return_decimal
                    exit_reason = 'Gap Down'
                    gap_down = True
                else:
                    exit_return = stop_loss
                    exit_reason = 'Stop Loss'
                
                stopped_out = True
                break
            
            first_candle = False
        
        if not stopped_out:
            exit_trading_day = actual_exit_day
            exit_hour = exit_day_data['Time'].iloc[-1] if 'Time' in exit_day_data.columns else ''
            exit_datetime = exit_day_data['Datetime'].iloc[-1] if 'Datetime' in exit_day_data.columns else None
            exit_reason = f'Day {actual_exit_day} Close'
            exit_return = exit_day_close_return
        
        diff_vs_5d = None
        if return_5d is not None and exit_return is not None:
            diff_vs_5d = exit_return - return_5d
        
        results.append({
            'Ticker': ticker,
            'Company': company_name,
            'Earnings Date': earnings_date,
            'Fiscal Quarter': fiscal_quarter,
            'Earnings Timing': earnings_timing,
            'Base Price': base_price,
            'Exit Day': exit_trading_day,
            'Exit Hour': exit_hour,
            'Exit Datetime': exit_datetime,
            'Exit Reason': exit_reason,
            'Backtest Return': exit_return,
            '5D Return': return_5d,
            'Diff vs 5D': diff_vs_5d,
            'Max Intraday': max_return,
            'Min Intraday': min_return,
            'Stopped Out': stopped_out,
            'Gap Down': gap_down,
        })
    
    return pd.DataFrame(results)


# ------------------------------------
# RENDER TAB
# ------------------------------------

def render_stop_loss_tab(returns_df, hourly_df, filter_stats):
    """Render the Stop Loss Analysis tab."""
    
    has_hourly = hourly_df is not None and not hourly_df.empty
    has_returns = returns_df is not None and not returns_df.empty
    
    st.subheader("Stop Loss Analysis")
    
    if not has_hourly and not has_returns:
        st.warning("No data available. Upload hourly_prices.csv or returns_tracker.csv.")
        col1, col2 = st.columns(2)
        with col1:
            hourly_upload = st.file_uploader("Upload hourly_prices.csv:", type=['csv'], key='hourly')
            if hourly_upload:
                hourly_df = pd.read_csv(hourly_upload)
                hourly_df['Datetime'] = pd.to_datetime(hourly_df['Datetime'], errors='coerce')
                hourly_df['Date'] = pd.to_datetime(hourly_df['Date'], errors='coerce')
                hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date'], errors='coerce')
                has_hourly = True
        with col2:
            returns_upload = st.file_uploader("Upload returns_tracker.csv:", type=['csv'], key='returns')
            if returns_upload:
                returns_df = pd.read_csv(returns_upload)
                returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date'], errors='coerce')
                has_returns = True
    
    if has_hourly or has_returns:
        # Data summary
        st.markdown("### Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", filter_stats['final_count'])
        with col2:
            st.metric("Unique Tickers", len(filter_stats['final_tickers']))
        with col3:
            if has_hourly:
                st.metric("Hourly Data Points", f"{len(hourly_df):,}")
            else:
                st.metric("Data Source", "Returns Tracker")
        
        st.markdown("---")
        
        # Parameters
        st.markdown("### Backtest Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stop_loss = st.select_slider(
                "Stop Loss",
                options=[None, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10, -0.12, -0.15, -0.20],
                value=-0.10,
                format_func=lambda x: "None" if x is None else f"{x*100:.0f}%"
            )
        
        with col2:
            max_days = st.selectbox("Exit Day (if not stopped)", [3, 5, 7, 10], index=1)
        
        with col3:
            st.metric("Strategy", f"Stop: {stop_loss*100:.0f}%" if stop_loss else "No Stop", f"Exit: Day {max_days}")
        
        # Run backtest button
        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest on hourly data..."):
                results = detailed_hourly_backtest(hourly_df, returns_df, stop_loss=stop_loss, max_days=max_days)
            
            if results.empty:
                st.warning("No trades found with complete data.")
            else:
                st.session_state['backtest_results'] = results
                st.session_state['stop_loss'] = stop_loss
                st.session_state['max_days'] = max_days
        
        # Display results if available
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            stop_loss = st.session_state['stop_loss']
            max_days = st.session_state['max_days']
            
            _render_backtest_results(results, stop_loss, max_days, hourly_df)


def _render_backtest_results(results, stop_loss, max_days, hourly_df):
    """Render the backtest results section."""
    
    st.markdown("---")
    st.markdown("### Backtest Results")
    
    total_return = results['Backtest Return'].sum() * 100
    avg_return = results['Backtest Return'].mean() * 100
    win_rate = (results['Backtest Return'] > 0).mean() * 100
    n_trades = len(results)
    stopped_count = results['Stopped Out'].sum()
    held_count = n_trades - stopped_count
    
    results_with_5d = results[results['5D Return'].notna()]
    if len(results_with_5d) > 0:
        avg_5d_return = results_with_5d['5D Return'].mean() * 100
        total_5d_return = results_with_5d['5D Return'].sum() * 100
        avg_diff = results_with_5d['Diff vs 5D'].mean() * 100
    else:
        avg_5d_return = None
        total_5d_return = None
        avg_diff = None
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value metric-{'green' if total_return > 0 else 'red'}">{total_return:+.1f}%</div>
            <div class="metric-label">Total Return</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value metric-{'green' if avg_return > 0 else 'red'}">{avg_return:+.2f}%</div>
            <div class="metric-label">Avg Return/Trade</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value metric-blue">{win_rate:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value metric-white">{n_trades}</div>
            <div class="metric-label">Total Trades</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Strategy vs Buy & Hold
    if avg_5d_return is not None:
        st.markdown("### Strategy vs Buy & Hold (5D)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**With Stop Loss Strategy**")
            st.metric("Total Return", f"{total_return:+.1f}%")
            st.metric("Avg/Trade", f"{avg_return:+.2f}%")
        
        with col2:
            st.markdown("**Buy & Hold 5 Days**")
            st.metric("Total Return", f"{total_5d_return:+.1f}%")
            st.metric("Avg/Trade", f"{avg_5d_return:+.2f}%")
        
        with col3:
            st.markdown("**Difference (Strategy - B&H)**")
            diff_total = total_return - total_5d_return
            st.metric("Total Diff", f"{diff_total:+.1f}%", delta_color="normal")
            st.metric("Avg Diff", f"{avg_diff:+.2f}%", delta_color="normal")
        
        if diff_total > 0:
            st.success(f"Stop loss strategy **outperformed** buy & hold by {diff_total:+.1f}%")
        else:
            st.warning(f"Stop loss strategy **underperformed** buy & hold by {diff_total:.1f}%")
    
    st.markdown("---")
    
    # Exit Breakdown
    st.markdown("### Exit Breakdown")
    
    gap_down_count = results['Gap Down'].sum() if 'Gap Down' in results.columns else 0
    stop_loss_count = stopped_count - gap_down_count
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gap_trades = results[results['Gap Down'] == True] if 'Gap Down' in results.columns else pd.DataFrame()
        gap_avg = gap_trades['Backtest Return'].mean() * 100 if len(gap_trades) > 0 else 0
        
        st.markdown(f"""
        <div class="exit-card">
            <div class="exit-count">{gap_down_count}</div>
            <div class="exit-pct">({gap_down_count/n_trades*100:.1f}% of trades)</div>
            <div class="exit-return" style="color: #ef4444;">{gap_avg:+.2f}% avg</div>
            <div class="exit-label">Gap Down (opened below stop)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stop_trades = results[(results['Stopped Out'] == True) & (results.get('Gap Down', False) == False)]
        stop_avg = stop_trades['Backtest Return'].mean() * 100 if len(stop_trades) > 0 else 0
        
        st.markdown(f"""
        <div class="exit-card">
            <div class="exit-count">{stop_loss_count}</div>
            <div class="exit-pct">({stop_loss_count/n_trades*100:.1f}% of trades)</div>
            <div class="exit-return" style="color: #f59e0b;">{stop_avg:+.2f}% avg</div>
            <div class="exit-label">Stop Loss Hit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        held_trades = results[results['Stopped Out'] == False]
        held_avg = held_trades['Backtest Return'].mean() * 100 if len(held_trades) > 0 else 0
        
        st.markdown(f"""
        <div class="exit-card">
            <div class="exit-count">{held_count}</div>
            <div class="exit-pct">({held_count/n_trades*100:.1f}% of trades)</div>
            <div class="exit-return" style="color: #22c55e;">{held_avg:+.2f}% avg</div>
            <div class="exit-label">Held to Day {max_days}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    st.markdown("### Performance Charts")
    
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Cumulative Returns", "Return Distribution", "Stop Loss Timing"])
    
    with chart_tab1:
        _render_cumulative_chart(results, stop_loss)
    
    with chart_tab2:
        _render_distribution_charts(results, stop_loss)
    
    with chart_tab3:
        _render_stop_timing_charts(results, stopped_count, max_days)
    
    st.markdown("---")
    
    # Full Trade List
    st.markdown("### All Trades")
    _render_trade_list(results, stop_loss, max_days)
    
    st.markdown("---")
    
    # Stop Loss Comparison
    st.markdown("### Compare Stop Loss Levels")
    if st.button("Run Full Comparison", use_container_width=True):
        _run_stop_loss_comparison(hourly_df, max_days)


def _render_cumulative_chart(results, stop_loss):
    """Render cumulative returns chart."""
    results_sorted = results.sort_values('Earnings Date').copy()
    results_sorted['Cumulative'] = results_sorted['Backtest Return'].cumsum() * 100
    
    if '5D Return' in results_sorted.columns:
        results_sorted['Cumulative_5D'] = results_sorted['5D Return'].fillna(0).cumsum() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(results_sorted) + 1)),
        y=results_sorted['Cumulative'],
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        name=f'Strategy ({stop_loss*100:.0f}% stop)' if stop_loss else 'Strategy (no stop)',
        hovertemplate='Trade %{x}<br>Cumulative: %{y:+.1f}%<extra></extra>'
    ))
    
    if '5D Return' in results_sorted.columns:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(results_sorted) + 1)),
            y=results_sorted['Cumulative_5D'],
            mode='lines',
            line=dict(color='#64748b', width=2, dash='dash'),
            name='Buy & Hold 5D',
            hovertemplate='Trade %{x}<br>Cumulative: %{y:+.1f}%<extra></extra>'
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#475569")
    fig.update_layout(
        title="Cumulative Return: Strategy vs Buy & Hold",
        xaxis_title="Trade #",
        yaxis_title="Cumulative Return %",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
    )
    fig.update_xaxes(gridcolor='#1e293b')
    fig.update_yaxes(gridcolor='#1e293b')
    st.plotly_chart(fig, use_container_width=True)


def _render_distribution_charts(results, stop_loss):
    """Render return distribution charts."""
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=results['Backtest Return'] * 100,
            nbinsx=30,
            marker_color='#3b82f6',
            name='Strategy Returns'
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
        if stop_loss:
            fig.add_vline(x=stop_loss*100, line_dash="dash", line_color="#ef4444", 
                         annotation_text=f"Stop: {stop_loss*100:.0f}%")
        fig.update_layout(
            title="Strategy Return Distribution",
            xaxis_title="Return %",
            yaxis_title="Count",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Diff vs 5D' in results.columns:
            diff_data = results['Diff vs 5D'].dropna() * 100
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=diff_data,
                nbinsx=30,
                marker_color='#f59e0b',
                name='Difference'
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
            fig.update_layout(
                title="Strategy vs 5D Return (Difference)",
                xaxis_title="Difference %",
                yaxis_title="Count",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_stop_timing_charts(results, stopped_count, max_days):
    """Render stop loss timing charts."""
    if stopped_count > 0:
        stopped_trades = results[results['Stopped Out'] == True].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            day_counts = stopped_trades['Exit Day'].value_counts().sort_index().reset_index()
            day_counts.columns = ['Day', 'Count']
            
            fig = px.bar(day_counts, x='Day', y='Count', 
                        title='Stop Loss Triggers by Trading Day')
            fig.update_traces(marker_color='#3b82f6')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Exit Hour' in stopped_trades.columns:
                hour_counts = stopped_trades['Exit Hour'].value_counts().sort_index().reset_index()
                hour_counts.columns = ['Hour', 'Count']
                
                fig = px.bar(hour_counts, x='Hour', y='Count',
                            title='Stop Loss Triggers by Hour')
                fig.update_traces(marker_color='#3b82f6')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades were stopped out with the current stop loss setting.")


def _render_trade_list(results, stop_loss, max_days):
    """Render the full trade list."""
    display_df = results.copy()
    display_df['Earnings Date'] = pd.to_datetime(display_df['Earnings Date']).dt.strftime('%Y-%m-%d')
    display_df['Backtest Return'] = display_df['Backtest Return'] * 100
    display_df['5D Return'] = display_df['5D Return'] * 100
    display_df['Diff vs 5D'] = display_df['Diff vs 5D'] * 100
    display_df['Max Intraday'] = display_df['Max Intraday'] * 100
    display_df['Min Intraday'] = display_df['Min Intraday'] * 100
    
    col_order = ['Ticker', 'Company', 'Fiscal Quarter', 'Earnings Date', 'Base Price', 
                'Exit Day', 'Exit Hour', 'Exit Reason', 'Backtest Return', '5D Return', 
                'Diff vs 5D', 'Max Intraday', 'Min Intraday']
    col_order = [c for c in col_order if c in display_df.columns]
    
    st.dataframe(
        display_df[col_order], 
        use_container_width=True, 
        hide_index=True, 
        height=400,
        column_config={
            "Base Price": st.column_config.NumberColumn("Base Price", format="$%.2f"),
            "Backtest Return": st.column_config.NumberColumn("Backtest Return", format="%+.2f%%"),
            "5D Return": st.column_config.NumberColumn("5D Return", format="%+.2f%%"),
            "Diff vs 5D": st.column_config.NumberColumn("Diff vs 5D", format="%+.2f%%"),
            "Max Intraday": st.column_config.NumberColumn("Max Intraday", format="%+.1f%%"),
            "Min Intraday": st.column_config.NumberColumn("Min Intraday", format="%+.1f%%"),
        }
    )
    
    csv = results.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name=f"backtest_results_stop{int(stop_loss*100) if stop_loss else 'none'}_day{max_days}.csv",
        mime="text/csv"
    )


def _run_stop_loss_comparison(hourly_df, max_days):
    """Run comparison across multiple stop loss levels."""
    stop_levels = [None, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.15]
    
    comparison = []
    progress = st.progress(0)
    
    for i, stop in enumerate(stop_levels):
        res_list = []
        trades = hourly_df.groupby(['Ticker', 'Earnings Date']).first().reset_index()
        
        for _, trade in trades.iterrows():
            ticker = trade['Ticker']
            earnings_date = trade['Earnings Date']
            
            trade_data = hourly_df[
                (hourly_df['Ticker'] == ticker) & 
                (hourly_df['Earnings Date'] == earnings_date) &
                (hourly_df['Trading Day'] >= 1) &
                (hourly_df['Trading Day'] <= max_days)
            ].sort_values('Datetime')
            
            if trade_data.empty:
                continue
            
            exit_return = None
            stopped = False
            gap_down = False
            first_candle = True
            
            for _, hour in trade_data.iterrows():
                ret = hour['Return From Earnings (%)'] / 100
                if stop is not None and ret <= stop:
                    if first_candle and ret < stop:
                        exit_return = ret
                        gap_down = True
                    else:
                        exit_return = stop
                    stopped = True
                    break
                first_candle = False
            
            if exit_return is None:
                last_day = trade_data[trade_data['Trading Day'] == trade_data['Trading Day'].max()]
                if not last_day.empty:
                    exit_return = last_day['Return From Earnings (%)'].iloc[-1] / 100
            
            if exit_return is not None:
                res_list.append({
                    'Return': exit_return, 
                    'Stopped': stopped,
                    'Gap Down': gap_down
                })
        
        if res_list:
            res_df = pd.DataFrame(res_list)
            comparison.append({
                'Stop Loss': f"{stop*100:.0f}%" if stop else "None",
                'Stop Value': stop if stop else 0,
                'Total Return': res_df['Return'].sum() * 100,
                'Avg Return': res_df['Return'].mean() * 100,
                'Win Rate': (res_df['Return'] > 0).mean() * 100,
                'Stopped %': res_df['Stopped'].mean() * 100 if stop else 0,
                'Gap Downs': res_df['Gap Down'].sum() if stop else 0,
                'Trades': len(res_df)
            })
        
        progress.progress((i + 1) / len(stop_levels))
    
    progress.empty()
    comp_df = pd.DataFrame(comparison)
    
    best_idx = comp_df['Total Return'].idxmax()
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig = go.Figure()
        colors = ['#22c55e' if i == best_idx else '#3b82f6' for i in range(len(comp_df))]
        fig.add_trace(go.Bar(
            x=comp_df['Stop Loss'],
            y=comp_df['Total Return'],
            marker_color=colors,
            text=comp_df['Total Return'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside'
        ))
        
        y_min = comp_df['Total Return'].min()
        y_max = comp_df['Total Return'].max()
        y_padding = (y_max - y_min) * 0.15 if y_max != y_min else 10
        
        fig.update_layout(
            title="Total Return by Stop Loss Level",
            xaxis_title="Stop Loss",
            yaxis_title="Total Return %",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=400,
            xaxis=dict(type='category'),
            yaxis=dict(range=[min(y_min - y_padding, y_min * 1.1 if y_min < 0 else 0), y_max + y_padding]),
            margin=dict(t=50, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Comparison Table**")
        display_comp = comp_df.copy()
        display_comp['Total Return'] = display_comp['Total Return'].apply(lambda x: f"{x:+.1f}%")
        display_comp['Avg Return'] = display_comp['Avg Return'].apply(lambda x: f"{x:+.2f}%")
        display_comp['Win Rate'] = display_comp['Win Rate'].apply(lambda x: f"{x:.1f}%")
        display_comp['Stopped %'] = display_comp['Stopped %'].apply(lambda x: f"{x:.1f}%")
        display_comp = display_comp.drop(columns=['Stop Value'])
        
        st.dataframe(display_comp, use_container_width=True, hide_index=True)
        
        best_stop = comp_df.loc[best_idx, 'Stop Loss']
        best_return = comp_df.loc[best_idx, 'Total Return']
        st.success(f"**Best: {best_stop}** with {best_return:+.1f}% total return")
