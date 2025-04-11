import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta

# Function to fetch stock data with multiple timeframes
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

# Function to calculate technical indicators
def calculate_indicators(df):
    # Moving Averages
    df['MA5'] = talib.SMA(df['Close'], timeperiod=5)
    df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['Close'], timeperiod=50)
    
    # RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
    
    # Bollinger Bands
    df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(df['Close'], timeperiod=20)
    
    # ATR for volatility measurement
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Volume analysis
    df['Volume_MA20'] = talib.SMA(df['Volume'], timeperiod=20)
    
    return df

# Function to detect candlestick patterns
def detect_patterns(df):
    # Single candle patterns
    df['DOJI'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['HAMMER'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
    df['ENGULFING'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
    df['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    df['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
    
    return df

# Function to analyze stock condition with enhanced logic
def analyze_stock(symbol, timeframe='1d', interval='1d'):
    data = get_stock_data(symbol, period=timeframe, interval=interval)
    if data is None or len(data) < 50:  # Need enough data for indicators
        return None
    
    data = calculate_indicators(data)
    data = detect_patterns(data)
    
    latest = data.iloc[-1]
    prev_day = data.iloc[-2] if len(data) > 1 else latest
    
    current_price = round(latest['Close'], 2)
    atr = latest['ATR']
    
    # Initialize result dictionary
    result = {
        'Symbol': symbol,
        'Current Price': current_price,
        'Open': round(latest['Open'], 2),
        'High': round(latest['High'], 2),
        'Low': round(latest['Low'], 2),
        'Close': current_price,
        'Volume': latest['Volume'],
        'RSI': round(latest['RSI'], 2) if not np.isnan(latest['RSI']) else None,
        'MACD_Hist': round(latest['MACD_Hist'], 4) if not np.isnan(latest['MACD_Hist']) else None,
        'ATR': round(atr, 2) if not np.isnan(atr) else None,
        'Recommendation': "Neutral",
        'Stop Loss': None,
        'Target': None,
        'Timeframe': timeframe,
        'Interval': interval,
        'Patterns': [],
        'Trend': None,
        'Confidence': None
    }
    
    # Detect candlestick patterns
    patterns = []
    if latest['DOJI'] != 0: patterns.append("Doji")
    if latest['HAMMER'] != 0: patterns.append("Hammer")
    if latest['ENGULFING'] != 0: patterns.append("Engulfing")
    if latest['MORNING_STAR'] != 0: patterns.append("Morning Star")
    if latest['EVENING_STAR'] != 0: patterns.append("Evening Star")
    result['Patterns'] = ", ".join(patterns) if patterns else "None"
    
    # Determine trend
    trend = "Neutral"
    if latest['MA5'] > latest['MA20'] > latest['MA50']:
        trend = "Up"
    elif latest['MA5'] < latest['MA20'] < latest['MA50']:
        trend = "Down"
    result['Trend'] = trend
    
    # Enhanced recommendation logic
    recommendation = "Neutral"
    confidence = "Low"
    stop_loss = None
    target = None
    
    # Bullish conditions
    bullish_conditions = 0
    if latest['Open'] == latest['Low']: bullish_conditions += 1
    if "Hammer" in patterns: bullish_conditions += 1
    if "Morning Star" in patterns: bullish_conditions += 1
    if latest['RSI'] < 30: bullish_conditions += 1
    if latest['MACD_Hist'] > 0 and latest['MACD'] > latest['MACD_Signal']: bullish_conditions += 1
    if trend == "Up": bullish_conditions += 1
    if latest['Close'] > latest['UpperBand']: bullish_conditions -= 1  # Overbought
    
    # Bearish conditions
    bearish_conditions = 0
    if latest['Open'] == latest['High']: bearish_conditions += 1
    if "Evening Star" in patterns: bearish_conditions += 1
    if latest['RSI'] > 70: bearish_conditions += 1
    if latest['MACD_Hist'] < 0 and latest['MACD'] < latest['MACD_Signal']: bearish_conditions += 1
    if trend == "Down": bearish_conditions += 1
    if latest['Close'] < latest['LowerBand']: bearish_conditions -= 1  # Oversold
    
    # Generate recommendation
    if bullish_conditions >= 3 and bearish_conditions < 2:
        recommendation = "Buy"
        confidence = "Medium" if bullish_conditions == 3 else "High"
        stop_loss = round(current_price - (2 * atr), 2) if atr else round(current_price * 0.98, 2)
        target = round(current_price + (3 * atr), 2) if atr else round(current_price * 1.03, 2)
    elif bearish_conditions >= 3 and bullish_conditions < 2:
        recommendation = "Sell"
        confidence = "Medium" if bearish_conditions == 3 else "High"
        stop_loss = round(current_price + (2 * atr), 2) if atr else round(current_price * 1.02, 2)
        target = round(current_price - (3 * atr), 2) if atr else round(current_price * 0.97, 2)
    
    result.update({
        'Recommendation': recommendation,
        'Stop Loss': stop_loss,
        'Target': target,
        'Confidence': confidence,
        'Bullish Score': bullish_conditions,
        'Bearish Score': bearish_conditions
    })
    
    return result

# Main Streamlit app with enhanced features
def main():
    st.title("Advanced Stock Recommender for Intraday & Swing Trading")
    st.write("Analyzes stocks using multiple technical indicators and patterns")
    
    # Load stock list from Excel
    try:
        stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
    except FileNotFoundError:
        st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
        return
    
    # User inputs
    selected_sheet = st.sidebar.selectbox("Select Stock List", stock_sheets)
    analysis_type = st.sidebar.radio("Analysis Type", ["Intraday", "Swing"])
    
    if analysis_type == "Intraday":
        timeframe = st.sidebar.selectbox("Timeframe", ["1d", "5d"])
        interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "60m"])
    else:
        timeframe = st.sidebar.selectbox("Timeframe", ["1mo", "3mo", "6mo"])
        interval = st.sidebar.selectbox("Interval", ["1d", "1wk"])
    
    min_confidence = st.sidebar.select_slider("Minimum Confidence Level", ["Low", "Medium", "High"], value="Medium")
    min_volume = st.sidebar.number_input("Minimum Average Volume (millions)", min_value=0, value=1)
    
    analyze_button = st.sidebar.button("Analyze Stocks")
    
    if analyze_button:
        try:
            # Read selected sheet
            stock_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
            
            if 'Symbol' not in stock_df.columns:
                st.error("Error: The selected sheet doesn't have a 'Symbol' column.")
                return
            
            symbols = stock_df['Symbol'].tolist()
            
            # Analyze each stock
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(symbols):
                status_text.text(f"Analyzing {symbol} ({i+1}/{len(symbols)})...")
                result = analyze_stock(symbol, timeframe, interval)
                if result is not None:
                    # Filter by volume
                    if result['Volume'] >= min_volume * 1000000:  # Convert millions to actual volume
                        # Filter by confidence
                        confidence_levels = {"Low": 0, "Medium": 1, "High": 2}
                        if confidence_levels[result['Confidence']] >= confidence_levels[min_confidence]:
                            results.append(result)
                progress_bar.progress((i + 1) / len(symbols))
            
            if not results:
                st.warning("No stocks matched your criteria. Try adjusting your filters.")
                return
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Filter only Buy/Sell recommendations
            actionable_df = results_df[results_df['Recommendation'].isin(['Buy', 'Sell'])]
            
            # Display results
            st.subheader(f"Analysis Results ({analysis_type} - {timeframe} period - {interval} interval)")
            
            # Show summary stats
            st.write(f"Total Stocks Analyzed: {len(symbols)}")
            st.write(f"Actionable Recommendations Found: {len(actionable_df)}")
            
            # Display all results with expander
            with st.expander("Show All Analyzed Stocks"):
                st.dataframe(results_df)
            
            # Display actionable recommendations
            st.subheader("Actionable Recommendations")
            if not actionable_df.empty:
                # Sort by confidence and bullish/bearish score
                actionable_df = actionable_df.sort_values(
                    by=['Confidence', 'Bullish Score', 'Bearish Score'], 
                    ascending=[False, False, False]
                )
                
                st.dataframe(actionable_df)
                
                # Add charts for top recommendations
                st.subheader("Charts for Top Recommendations")
                top_recommendations = actionable_df.head(3)['Symbol'].tolist()
                
                for symbol in top_recommendations:
                    st.write(f"### {symbol}")
                    data = get_stock_data(symbol, timeframe, interval)
                    if data is not None:
                        st.line_chart(data['Close'])
                
                # Download buttons
                st.download_button(
                    label="Download All Results as CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'stock_recommendations_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
                
                st.download_button(
                    label="Download Actionable Recommendations as CSV",
                    data=actionable_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'actionable_recommendations_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            else:
                st.info("No strong Buy/Sell recommendations based on your criteria.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
