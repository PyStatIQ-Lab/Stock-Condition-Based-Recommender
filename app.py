import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta  # Alternative to talib
import numpy as np
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

# Function to calculate technical indicators using pandas_ta
def calculate_indicators(df):
    # Moving Averages
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA20'] = ta.sma(df['Close'], length=20)
    df['MA50'] = ta.sma(df['Close'], length=50)
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    df = pd.concat([df, bb], axis=1)
    
    # ATR for volatility measurement
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Volume analysis
    df['Volume_MA20'] = ta.sma(df['Volume'], length=20)
    
    return df

# Function to detect candlestick patterns using pandas_ta
def detect_patterns(df):
    # Get all patterns (returns a DataFrame with boolean columns)
    patterns = df.ta.cdl_pattern(name="all")
    
    # Add patterns to original DataFrame
    df = pd.concat([df, patterns], axis=1)
    
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
        'MACD_Hist': round(latest['MACDh_12_26_9'], 4) if 'MACDh_12_26_9' in latest else None,
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
    pattern_names = [
        'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING', 
        'CDL_MORNINGSTAR', 'CDL_EVENINGSTAR'
    ]
    patterns = [p.replace('CDL_', '') for p in pattern_names if p in latest and latest[p] != 0]
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
    if "HAMMER" in patterns: bullish_conditions += 1
    if "MORNINGSTAR" in patterns: bullish_conditions += 1
    if latest['RSI'] < 30: bullish_conditions += 1
    if 'MACDh_12_26_9' in latest and latest['MACDh_12_26_9'] > 0: bullish_conditions += 1
    if trend == "Up": bullish_conditions += 1
    if 'BBU_20_2.0' in latest and latest['Close'] > latest['BBU_20_2.0']: bullish_conditions -= 1
    
    # Bearish conditions
    bearish_conditions = 0
    if latest['Open'] == latest['High']: bearish_conditions += 1
    if "EVENINGSTAR" in patterns: bearish_conditions += 1
    if latest['RSI'] > 70: bearish_conditions += 1
    if 'MACDh_12_26_9' in latest and latest['MACDh_12_26_9'] < 0: bearish_conditions += 1
    if trend == "Down": bearish_conditions += 1
    if 'BBL_20_2.0' in latest and latest['Close'] < latest['BBL_20_2.0']: bearish_conditions -= 1
    
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

# Main Streamlit app remains the same as before
def main():
    st.title("Advanced Stock Recommender for Intraday & Swing Trading")
    st.write("Analyzes stocks using multiple technical indicators and patterns")
    
    # [Rest of your existing main() function remains unchanged]
    # ...

if __name__ == "__main__":
    main()
