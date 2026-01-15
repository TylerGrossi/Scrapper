import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_earnings_analysis_tab(returns_df, filter_stats):
    """Render the Earnings Analysis tab."""
    
    st.subheader("Earnings Surprise Analysis")
    st.markdown("Analyze how earnings beats/misses and surprise magnitude affect stock returns")
    
    # Use the filtered returns_df
    analysis_df = returns_df.copy() if returns_df is not None else None
    
    if analysis_df is None or analysis_df.empty:
        st.warning("No returns data available. Please ensure returns_tracker.csv is accessible.")
        
        uploaded_file = st.file_uploader("Upload returns_tracker.csv:", type=['csv'], key='earnings_upload')
        if uploaded_file:
            analysis_df = pd.read_csv(uploaded_file)
            analysis_df['Earnings Date'] = pd.to_datetime(analysis_df['Earnings Date'], errors='coerce')
            st.success(f"Loaded {len(analysis_df)} rows")
    
    if analysis_df is not None and not analysis_df.empty:
        # Convert EPS Surprise to numeric if needed
        if 'EPS Surprise (%)' in analysis_df.columns:
            analysis_df['EPS Surprise (%)'] = pd.to_numeric(analysis_df['EPS Surprise (%)'], errors='coerce')
        
        # Use 5D Return
        return_col = '5D Return'
        
        # Convert returns to percentage for display
        analysis_df[return_col] = pd.to_numeric(analysis_df[return_col], errors='coerce') * 100
        
        # Count trades with and without EPS Surprise data
        total_trades = len(analysis_df)
        valid_surprise = analysis_df['EPS Surprise (%)'].notna().sum() if 'EPS Surprise (%)' in analysis_df.columns else 0
        missing_surprise = total_trades - valid_surprise
        avg_surprise = analysis_df['EPS Surprise (%)'].mean() if 'EPS Surprise (%)' in analysis_df.columns else 0
        
        st.markdown("---")
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("With Surprise Data", valid_surprise)
        with col3:
            st.metric("Avg EPS Surprise", f"{avg_surprise:.1f}%" if pd.notna(avg_surprise) else "N/A")
        with col4:
            st.metric("Return Column", return_col)
        
        st.markdown("---")
        
        # Create analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "Beat vs Miss", "Surprise Magnitude", "Raw Data"
        ])
        
        with analysis_tab1:
            _render_beat_vs_miss(analysis_df, return_col, valid_surprise, total_trades)
        
        with analysis_tab2:
            _render_surprise_magnitude(analysis_df, return_col, valid_surprise, total_trades)
        
        with analysis_tab3:
            _render_raw_data(analysis_df, return_col)


def _render_beat_vs_miss(analysis_df, return_col, valid_surprise, total_trades):
    """Render Beat vs Miss analysis."""
    st.markdown("#### Beat vs Miss Performance")
    
    if 'EPS Surprise (%)' in analysis_df.columns and valid_surprise > 0:
        def classify_surprise(x):
            if pd.isna(x):
                return 'Unknown'
            elif x > 5:
                return 'Strong Beat (>5%)'
            elif x > 0:
                return 'Beat (0-5%)'
            elif x > -5:
                return 'Miss (0 to -5%)'
            else:
                return 'Strong Miss (<-5%)'
        
        analysis_df['Surprise Category'] = analysis_df['EPS Surprise (%)'].apply(classify_surprise)
        known_df = analysis_df[analysis_df['Surprise Category'] != 'Unknown']
        
        if len(known_df) > 0:
            category_stats = known_df.groupby('Surprise Category').agg({
                return_col: ['sum', 'mean', 'median', 'count', 'std'],
            }).round(2)
            category_stats.columns = ['Total Return', 'Avg Return', 'Median Return', 'Count', 'Std Dev']
            category_stats['Win Rate'] = known_df.groupby('Surprise Category')[return_col].apply(
                lambda x: (x > 0).mean() * 100
            ).round(1)
            category_stats = category_stats.reset_index()
            
            order = ['Strong Beat (>5%)', 'Beat (0-5%)', 'Miss (0 to -5%)', 'Strong Miss (<-5%)']
            category_stats['sort_order'] = category_stats['Surprise Category'].apply(
                lambda x: order.index(x) if x in order else 99
            )
            category_stats = category_stats.sort_values('sort_order').drop(columns=['sort_order'])
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=category_stats['Surprise Category'],
                    y=category_stats['Avg Return'],
                    marker_color='#3b82f6',
                    text=category_stats['Avg Return'].apply(lambda x: f"{x:+.2f}%"),
                    textposition='outside'
                ))
                
                y_min = category_stats['Avg Return'].min()
                y_max = category_stats['Avg Return'].max()
                y_padding = (y_max - y_min) * 0.2 if y_max != y_min else 5
                
                fig.update_layout(
                    title="Average Return by Earnings Surprise Category",
                    xaxis_title="Surprise Category",
                    yaxis_title=f"Avg {return_col} (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=400,
                    yaxis=dict(range=[min(y_min - y_padding, -2), max(y_max + y_padding, 2)]),
                    margin=dict(t=50, b=50)
                )
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Performance by Category**")
                st.dataframe(
                    category_stats,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Total Return": st.column_config.NumberColumn(format="%.1f%%"),
                        "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                        "Median Return": st.column_config.NumberColumn(format="%.2f%%"),
                        "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                        "Std Dev": st.column_config.NumberColumn(format="%.2f"),
                    }
                )
                
                beats = category_stats[category_stats['Surprise Category'].str.contains('Beat', na=False)]
                misses = category_stats[category_stats['Surprise Category'].str.contains('Miss', na=False)]
                
                if len(beats) > 0 and len(misses) > 0:
                    beat_avg = (beats['Avg Return'] * beats['Count']).sum() / beats['Count'].sum()
                    beat_total = beats['Total Return'].sum()
                    miss_avg = (misses['Avg Return'] * misses['Count']).sum() / misses['Count'].sum()
                    miss_total = misses['Total Return'].sum()
                    diff = beat_avg - miss_avg
                    
                    st.markdown("---")
                    st.markdown(f"""
                    **Key Insight:**
                    - Beats: **{beat_avg:+.2f}%** avg, **{beat_total:+.1f}%** total
                    - Misses: **{miss_avg:+.2f}%** avg, **{miss_total:+.1f}%** total
                    - Spread: **{diff:+.2f}%**
                    """)
                    
                    if diff > 0:
                        st.success("Beats outperform misses")
                    else:
                        st.warning("Misses outperform beats (unusual)")
            
            # Simple Beat vs Miss
            st.markdown("---")
            st.markdown("#### Simple Beat vs Miss (Any Amount)")
            
            simple_df = known_df.copy()
            simple_df['Beat/Miss'] = simple_df['EPS Surprise (%)'].apply(
                lambda x: 'Beat' if x > 0 else 'Miss'
            )
            
            simple_stats = simple_df.groupby('Beat/Miss').agg({
                return_col: ['sum', 'mean', 'median', 'count'],
                'EPS Surprise (%)': 'mean'
            }).round(2)
            simple_stats.columns = ['Total Return', 'Avg Return', 'Median Return', 'Count', 'Avg Surprise %']
            simple_stats['Win Rate'] = simple_df.groupby('Beat/Miss')[return_col].apply(
                lambda x: (x > 0).mean() * 100
            ).round(1)
            simple_stats = simple_stats.reset_index()
            
            col1, col2, col3 = st.columns(3)
            
            beat_row = simple_stats[simple_stats['Beat/Miss'] == 'Beat']
            miss_row = simple_stats[simple_stats['Beat/Miss'] == 'Miss']
            
            with col1:
                if len(beat_row) > 0:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value metric-green">{beat_row['Total Return'].values[0]:+.1f}%</div>
                        <div class="metric-label">Beat Total Return ({int(beat_row['Count'].values[0])} trades)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if len(miss_row) > 0:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value metric-red">{miss_row['Total Return'].values[0]:+.1f}%</div>
                        <div class="metric-label">Miss Total Return ({int(miss_row['Count'].values[0])} trades)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                if len(beat_row) > 0 and len(miss_row) > 0:
                    spread = beat_row['Avg Return'].values[0] - miss_row['Avg Return'].values[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value metric-blue">{spread:+.2f}%</div>
                        <div class="metric-label">Beat - Miss Avg Spread</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("No 'EPS Surprise (%)' column found in data.")


def _render_surprise_magnitude(analysis_df, return_col, valid_surprise, total_trades):
    """Render Surprise Magnitude analysis."""
    st.markdown("#### Surprise Magnitude vs Returns")
    
    if 'EPS Surprise (%)' in analysis_df.columns:
        scatter_df = analysis_df[
            analysis_df['EPS Surprise (%)'].notna() & 
            analysis_df[return_col].notna() &
            (analysis_df['EPS Surprise (%)'] >= -100) &
            (analysis_df['EPS Surprise (%)'] <= 100)
        ].copy()
        
        outliers = analysis_df[
            analysis_df['EPS Surprise (%)'].notna() & 
            ((analysis_df['EPS Surprise (%)'] < -100) | (analysis_df['EPS Surprise (%)'] > 100))
        ]
        if len(outliers) > 0:
            st.caption(f"Note: {len(outliers)} outliers with EPS Surprise outside -100% to 100% range excluded from chart")
        
        if len(scatter_df) > 5:
            col1, col2 = st.columns([2, 1])
            
            x = scatter_df['EPS Surprise (%)'].values
            y = scatter_df[return_col].values
            n = len(x)
            x_mean, y_mean = np.mean(x), np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
            intercept = y_mean - slope * x_mean
            
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            se = np.sqrt(ss_res / (n - 2)) if n > 2 else 0
            se_slope = se / np.sqrt(np.sum((x - x_mean) ** 2)) if np.sum((x - x_mean) ** 2) > 0 else 0
            t_stat = slope / se_slope if se_slope > 0 else 0
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=scatter_df['EPS Surprise (%)'],
                    y=scatter_df[return_col],
                    mode='markers',
                    marker=dict(size=10, opacity=0.7, color='#3b82f6'),
                    text=scatter_df['Ticker'],
                    hovertemplate='<b>%{text}</b><br>EPS Surprise: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>',
                    name='Trades'
                ))
                
                x_line = np.array([-100, 100])
                y_line = slope * x_line + intercept
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color='#f59e0b', width=2, dash='solid'),
                    name=f'Trend (R²={r_squared:.3f})'
                ))
                
                fig.update_layout(
                    title="EPS Surprise % vs Stock Return",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=500,
                    xaxis=dict(range=[-100, 100], title='EPS Surprise (%)'),
                    yaxis=dict(title=f'{return_col} (%)'),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                fig.add_vline(x=0, line_dash="dash", line_color="#475569")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                correlation = scatter_df['EPS Surprise (%)'].corr(scatter_df[return_col])
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value metric-blue">{correlation:.3f}</div>
                    <div class="metric-label">Correlation</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                
                st.markdown(f"""
                **Regression Stats:**
                - R²: **{r_squared:.3f}**
                - Slope: **{slope:.4f}**
                - T-stat: **{t_stat:.2f}**
                """)
                
                if abs(t_stat) > 1.96:
                    st.success("Statistically significant (|t| > 1.96)")
                elif abs(t_stat) > 1.65:
                    st.info("Marginally significant")
                else:
                    st.warning("Not significant")
            
            # Surprise buckets analysis
            st.markdown("---")
            st.markdown("#### Returns by Surprise Buckets")
            
            def bucket_surprise(x):
                if pd.isna(x):
                    return None
                elif x < -10:
                    return '< -10%'
                elif x < -5:
                    return '-10% to -5%'
                elif x < 0:
                    return '-5% to 0%'
                elif x < 5:
                    return '0% to 5%'
                elif x < 10:
                    return '5% to 10%'
                elif x < 20:
                    return '10% to 20%'
                else:
                    return '> 20%'
            
            scatter_df['Surprise Bucket'] = scatter_df['EPS Surprise (%)'].apply(bucket_surprise)
            
            bucket_stats = scatter_df.groupby('Surprise Bucket').agg({
                return_col: ['sum', 'mean', 'median', 'count'],
            }).round(2)
            bucket_stats.columns = ['Total Return', 'Avg Return', 'Median', 'Count']
            bucket_stats['Win Rate'] = scatter_df.groupby('Surprise Bucket')[return_col].apply(
                lambda x: (x > 0).mean() * 100
            ).round(1)
            bucket_stats = bucket_stats.reset_index()
            
            bucket_order = ['< -10%', '-10% to -5%', '-5% to 0%', '0% to 5%', '5% to 10%', '10% to 20%', '> 20%']
            bucket_stats['sort_order'] = bucket_stats['Surprise Bucket'].apply(
                lambda x: bucket_order.index(x) if x in bucket_order else 99
            )
            bucket_stats = bucket_stats.sort_values('sort_order').drop(columns=['sort_order'])
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=bucket_stats['Surprise Bucket'],
                    y=bucket_stats['Avg Return'],
                    marker_color='#3b82f6',
                    text=bucket_stats['Avg Return'].apply(lambda x: f"{x:+.2f}%"),
                    textposition='outside'
                ))
                
                y_min = bucket_stats['Avg Return'].min()
                y_max = bucket_stats['Avg Return'].max()
                y_padding = (y_max - y_min) * 0.2 if y_max != y_min else 5
                
                fig.update_layout(
                    title="Average Return by EPS Surprise Bucket",
                    xaxis_title="EPS Surprise %",
                    yaxis_title=f"Avg {return_col} (%)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=400,
                    yaxis=dict(range=[min(y_min - y_padding, -2), max(y_max + y_padding, 2)]),
                    margin=dict(t=50, b=50)
                )
                fig.add_hline(y=0, line_dash="dash", line_color="#475569")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    bucket_stats, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Total Return": st.column_config.NumberColumn(format="%.1f%%"),
                        "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                        "Median": st.column_config.NumberColumn(format="%.2f%%"),
                        "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                    }
                )
    else:
        st.warning("No EPS Surprise data available for magnitude analysis.")


def _render_raw_data(analysis_df, return_col):
    """Render Raw Data explorer."""
    st.markdown("#### Raw Data Explorer")
    
    all_cols = list(analysis_df.columns)
    default_cols = ['Ticker', 'Company Name', 'Earnings Date', 'EPS Estimate', 
                   'Reported EPS', 'EPS Surprise (%)', return_col]
    default_cols = [c for c in default_cols if c in all_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display:",
        options=all_cols,
        default=default_cols[:10]
    )
    
    if selected_cols:
        display_data = analysis_df[selected_cols].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            sort_col = st.selectbox("Sort by:", selected_cols, index=0)
        
        with col2:
            sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
        
        display_data = display_data.sort_values(
            sort_col, 
            ascending=(sort_order == "Ascending")
        )
        
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        st.caption(f"Showing {len(display_data)} rows")
        
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="earnings_analysis_data.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("#### Summary Statistics")
    
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    key_cols = ['EPS Surprise (%)', return_col, 'EPS Estimate', 'Reported EPS', 'P/E', 'Beta']
    key_cols = [c for c in key_cols if c in numeric_cols]
    
    if key_cols:
        summary = analysis_df[key_cols].describe().T
        st.dataframe(summary.round(2), use_container_width=True)
