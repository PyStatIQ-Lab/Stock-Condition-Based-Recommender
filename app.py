import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(symbol, period='1d', interval='1d'):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Technical indicator calculations
def calculate_indicators(df):
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Bollinger Bands
    df['MiddleBand'] = df['Close'].rolling(window=20).mean()
    df['UpperBand'] = df['MiddleBand'] + 2 * df['Close'].rolling(window=20).std()
    df['LowerBand'] = df['MiddleBand'] - 2 * df['Close'].rolling(window=20).std()
    
    # ATR Calculation
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = np.maximum(high_low, np.maximum(high_close, low_close)).rolling(window=14).mean()
    
    # Volume Analysis
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    return df

# Candlestick pattern detection
def detect_patterns(df):
    # Bullish Patterns
    df['Bullish_Engulfing'] = (df['Close'] > df['Open']) & \
                             (df['Close'].shift() < df['Open'].shift()) & \
                             (df['Close'] > df['Open'].shift()) & \
                             (df['Open'] < df['Close'].shift())
    
    df['Hammer'] = (df['Close'] > df['Open']) & \
                  ((df['Close'] - df['Low']) > 2 * (df['Open'] - df['Close'])) & \
                  ((df['High'] - df['Close']) < (df['Open'] - df['Close']))
    
    # Bearish Patterns
    df['Bearish_Engulfing'] = (df['Close'] < df['Open']) & \
                             (df['Close'].shift() > df['Open'].shift()) & \
                             (df['Close'] < df['Open'].shift()) & \
                             (df['Open'] > df['Close'].shift())
    
    df['Shooting_Star'] = (df['Open'] > df['Close']) & \
                         ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) & \
                         ((df['Close'] - df['Low']) < (df['Open'] - df['Close']))
    
    return df

# Stock analysis function
def analyze_stock(symbol, timeframe='1d', interval='1d'):
    data = get_stock_data(symbol, timeframe, interval)
    if data is None or len(data) < 50:
        return None
    
    data = calculate_indicators(data)
    data = detect_patterns(data)
    latest = data.iloc[-1]
    
    # Initialize result dictionary
    result = {
        'Symbol': symbol,
        'Price': round(latest['Close'], 2),
        'Change %': round(latest['Returns'] * 100, 2) if not np.isnan(latest['Returns']) else 0,
        'Volume': f"{latest['Volume']/1e6:.2f}M",
        'RSI': round(latest['RSI'], 1) if not np.isnan(latest['RSI']) else None,
        'MACD': round(latest['MACD_Hist'], 3) if not np.isnan(latest['MACD_Hist']) else None,
        'Trend': "Up" if latest['MA5'] > latest['MA20'] > latest['MA50'] else 
                "Down" if latest['MA5'] < latest['MA20'] < latest['MA50'] else "Neutral",
        'Pattern': "Bullish" if latest['Bullish_Engulfing'] or latest['Hammer'] else
                  "Bearish" if latest['Bearish_Engulfing'] or latest['Shooting_Star'] else "None",
        'Recommendation': "Neutral",
        'Stop Loss': None,
        'Target': None,
        'Confidence': "Low"
    }
    
    # Scoring system for recommendations
    score = 0
    
    # Bullish factors
    if result['RSI'] < 30: score += 1
    if result['Trend'] == "Up": score += 1
    if "Bullish" in result['Pattern']: score += 1
    if latest['MACD_Hist'] > 0: score += 1
    if latest['Close'] > latest['MA20']: score += 1
    
    # Bearish factors
    if result['RSI'] > 70: score -= 1
    if result['Trend'] == "Down": score -= 1
    if "Bearish" in result['Pattern']: score -= 1
    if latest['MACD_Hist'] < 0: score -= 1
    if latest['Close'] < latest['MA20']: score -= 1
    
    # Generate recommendation
    if score >= 3:
        result['Recommendation'] = "Strong Buy"
        result['Confidence'] = "High"
        result['Stop Loss'] = round(latest['Close'] * 0.97, 2)
        result['Target'] = round(latest['Close'] * 1.06, 2)
    elif score >= 1:
        result['Recommendation'] = "Buy"
        result['Confidence'] = "Medium"
        result['Stop Loss'] = round(latest['Close'] * 0.98, 2)
        result['Target'] = round(latest['Close'] * 1.04, 2)
    elif score <= -3:
        result['Recommendation'] = "Strong Sell"
        result['Confidence'] = "High"
        result['Stop Loss'] = round(latest['Close'] * 1.03, 2)
        result['Target'] = round(latest['Close'] * 0.94, 2)
    elif score <= -1:
        result['Recommendation'] = "Sell"
        result['Confidence'] = "Medium"
        result['Stop Loss'] = round(latest['Close'] * 1.02, 2)
        result['Target'] = round(latest['Close'] * 0.96, 2)
    
    return result

# Main Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Pure yFinance Stock Recommender")
    st.markdown("Intraday & Swing Trading Recommendations using Technical Indicators")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        analysis_type = st.radio("Analysis Type", ["Intraday", "Swing"])
        
        if analysis_type == "Intraday":
            timeframe = st.selectbox("Timeframe", ["1d", "5d"])
            interval = st.selectbox("Interval", ["5m", "15m", "30m", "60m"])
        else:
            timeframe = st.selectbox("Timeframe", ["1mo", "3mo", "6mo"])
            interval = st.selectbox("Interval", ["1d", "1wk"])
        
        min_volume = st.number_input("Min Volume (Millions)", min_value=0, value=1)
        confidence_level = st.select_slider("Min Confidence", ["Low", "Medium", "High"], value="Medium")
    
    # Load stock list
    try:
        stock_df = pd.read_excel('stocklist.xlsx')
        symbols = stock_df['Symbol'].unique().tolist()
    except:
        st.error("Error loading stock list. Please ensure 'stocklist.xlsx' exists with a 'Symbol' column.")
        return
    
    # Analysis button
    if st.button("Analyze Stocks"):
        with st.spinner("Analyzing stocks..."):
            results = []
            for symbol in symbols:
                result = analyze_stock(symbol, timeframe, interval)
                if result and float(result['Volume'].replace('M','')) >= min_volume:
                    results.append(result)
            
            if not results:
                st.warning("No stocks matched your criteria")
                return
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Filter by confidence
            conf_map = {"Low": 0, "Medium": 1, "High": 2}
            min_conf = conf_map[confidence_level]
            results_df['Conf_Score'] = results_df['Confidence'].map(conf_map)
            filtered_df = results_df[results_df['Conf_Score'] >= min_conf]
            
            # Display results
            st.subheader(f"Analysis Results ({len(filtered_df)} actionable)")
            
            # Color coding for recommendations
            def color_recommendation(val):
                color = 'green' if 'Buy' in val else 'red' if 'Sell' in val else 'gray'
                return f'color: {color}; font-weight: bold'
            
            # Display table with styled recommendations
            st.dataframe(
                filtered_df.style.applymap(color_recommendation, subset=['Recommendation']),
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                    "RSI": st.column_config.ProgressColumn(
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download All Results",
                    data=filtered_df.to_csv(index=False),
                    file_name=f"stock_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    label="Download Strong Signals Only",
                    data=filtered_df[filtered_df['Confidence'] == 'High'].to_csv(index=False),
                    file_name=f"strong_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Show charts for top recommendations
            st.subheader("Top Recommendations")
            top_recommendations = filtered_df.sort_values(by=['Conf_Score', 'Recommendation'], ascending=False).head(3)
            
            for _, row in top_recommendations.iterrows():
                with st.expander(f"{row['Symbol']} - {row['Recommendation']} (Confidence: {row['Confidence']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Price:** ${row['Price']}")
                        st.write(f"**RSI:** {row['RSI']}")
                        st.write(f"**Trend:** {row['Trend']}")
                        st.write(f"**Pattern:** {row['Pattern']}")
                    with col2:
                        st.write(f"**Stop Loss:** ${row['Stop Loss']}")
                        st.write(f"**Target:** ${row['Target']}")
                        st.write(f"**Change:** {row['Change %']}%")
                        st.write(f"**Volume:** {row['Volume']}")
                    
                    # Price chart
                    chart_data = get_stock_data(row['Symbol'], timeframe, interval)
                    if chart_data is not None:
                        st.line_chart(chart_data['Close'])

if __name__ == "__main__":
    main()
