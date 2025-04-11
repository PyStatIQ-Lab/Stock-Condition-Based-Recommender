import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to calculate technical indicators using only yfinance data
def calculate_indicators(df):
    # Simple Moving Averages
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
    
    # Volume Moving Average
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    return df

# Function to detect basic candlestick patterns
def detect_patterns(df):
    patterns = []
    
    # Bullish patterns
    hammer = (df['Close'] > df['Open']) & \
             ((df['Close'] - df['Low']) > 2 * (df['Open'] - df['Close'])) & \
             ((df['High'] - df['Close']) < (df['Open'] - df['Close']))
    
    # Bearish patterns
    shooting_star = (df['Open'] > df['Close']) & \
                    ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) & \
                    ((df['Close'] - df['Low']) < (df['Open'] - df['Close']))
    
    df['Hammer'] = hammer
    df['ShootingStar'] = shooting_star
    
    return df

# Function to analyze stock condition
def analyze_stock(symbol, timeframe='1d', interval='1d'):
    data = get_stock_data(symbol, timeframe, interval)
    if data is None or len(data) < 50:
        return None
    
    data = calculate_indicators(data)
    data = detect_patterns(data)
    
    latest = data.iloc[-1]
    
    result = {
        'Symbol': symbol,
        'Current Price': round(latest['Close'], 2),
        'Open': round(latest['Open'], 2),
        'High': round(latest['High'], 2),
        'Low': round(latest['Low'], 2),
        'Volume': latest['Volume'],
        'RSI': round(latest['RSI'], 2) if not np.isnan(latest['RSI']) else None,
        'MACD_Hist': round(latest['MACD_Hist'], 4) if not np.isnan(latest['MACD_Hist']) else None,
        'ATR': round(latest['ATR'], 2) if not np.isnan(latest['ATR']) else None,
        'Recommendation': "Neutral",
        'Stop Loss': None,
        'Target': None,
        'Trend': "Up" if latest['MA5'] > latest['MA20'] > latest['MA50'] else 
                "Down" if latest['MA5'] < latest['MA20'] < latest['MA50'] else "Neutral",
        'Patterns': "Hammer" if latest['Hammer'] else 
                   "Shooting Star" if latest['ShootingStar'] else "None"
    }
    
    # Recommendation logic
    bullish = 0
    bearish = 0
    
    # Bullish signals
    if result['RSI'] < 30: bullish += 1
    if result['Trend'] == "Up": bullish += 1
    if result['Patterns'] == "Hammer": bullish += 1
    if latest['MACD_Hist'] > 0: bullish += 1
    
    # Bearish signals
    if result['RSI'] > 70: bearish += 1
    if result['Trend'] == "Down": bearish += 1
    if result['Patterns'] == "Shooting Star": bearish += 1
    if latest['MACD_Hist'] < 0: bearish += 1
    
    if bullish >= 2 and bearish <= 1:
        result['Recommendation'] = "Buy"
        result['Stop Loss'] = round(latest['Close'] * 0.98, 2)
        result['Target'] = round(latest['Close'] * 1.04, 2)
    elif bearish >= 2 and bullish <= 1:
        result['Recommendation'] = "Sell"
        result['Stop Loss'] = round(latest['Close'] * 1.02, 2)
        result['Target'] = round(latest['Close'] * 0.96, 2)
    
    return result

# [Rest of your code (get_stock_data and main functions) remains the same]
# ...

if __name__ == "__main__":
    main()
