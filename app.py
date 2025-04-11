import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
st.set_page_config(layout="wide", page_title="Advanced Stock Recommender")

# Function to calculate technical indicators manually
def calculate_technical_indicators(df):
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['STD_20'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['STD_20'])
    
    # Calculate SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    return df

# Function to fetch stock data with technical indicators
def get_stock_data(symbol, period='1d', interval='1d'):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return None
            
        # Calculate technical indicators
        hist = calculate_technical_indicators(hist)
        return hist.iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to analyze stock condition with confidence scoring
def analyze_stock(symbol, trading_type='intraday'):
    data = get_stock_data(symbol, period='5d' if trading_type == 'swing' else '1d',
                         interval='60m' if trading_type == 'intraday' else '1d')
    
    if data is None:
        return None
    
    current_price = round(data['Close'], 2)
    open_price = round(data['Open'], 2)
    high = round(data['High'], 2)
    low = round(data['Low'], 2)
    
    # Initialize confidence score (0-100)
    confidence = 0
    reasons = []
    
    # Price action analysis
    if open_price == high:
        price_action = "Bearish (Open=High)"
        confidence += 15
        reasons.append("Bearish price action (Open=High)")
    elif open_price == low:
        price_action = "Bullish (Open=Low)"
        confidence += 15
        reasons.append("Bullish price action (Open=Low)")
    else:
        price_action = "Neutral"
    
    # RSI analysis
    rsi = data['RSI']
    if not np.isnan(rsi):
        if rsi < 30:
            confidence += 20
            reasons.append("Oversold (RSI < 30)")
        elif rsi > 70:
            confidence -= 20
            reasons.append("Overbought (RSI > 70)")
        else:
            confidence += 10
    
    # MACD analysis
    macd_diff = data['MACD_Diff']
    if not np.isnan(macd_diff):
        if macd_diff > 0:
            confidence += 15
            reasons.append("Bullish MACD crossover")
        else:
            confidence -= 10
    
    # Bollinger Bands analysis
    bb_lower = data['BB_Lower']
    bb_upper = data['BB_Upper']
    if not np.isnan(bb_lower) and not np.isnan(bb_upper):
        if current_price <= bb_lower:
            confidence += 15
            reasons.append("Price at lower Bollinger Band")
        elif current_price >= bb_upper:
            confidence -= 15
            reasons.append("Price at upper Bollinger Band")
    
    # Moving averages analysis
    sma_20 = data['SMA_20']
    sma_50 = data['SMA_50']
    if not np.isnan(sma_20) and not np.isnan(sma_50):
        if current_price > sma_20 and sma_20 > sma_50:
            confidence += 10
            reasons.append("Bullish moving average alignment")
        elif current_price < sma_20 and sma_20 < sma_50:
            confidence -= 10
            reasons.append("Bearish moving average alignment")
    
    # Determine recommendation based on confidence
    if confidence >= 50:
        recommendation = "Buy"
        stop_loss = round(min(low, current_price * 0.98, bb_lower if not np.isnan(bb_lower) else current_price * 0.98), 2)
        target1 = round(current_price * 1.02, 2)
        target2 = round(current_price * 1.04, 2)
        buy_range = f"{round(current_price * 0.99, 2)} - {current_price}"
    elif confidence <= -30:
        recommendation = "Sell"
        stop_loss = round(max(high, current_price * 1.02, bb_upper if not np.isnan(bb_upper) else current_price * 1.02), 2)
        target1 = round(current_price * 0.98, 2)
        target2 = round(current_price * 0.96, 2)
        buy_range = "N/A"
    else:
        recommendation = "Hold"
        stop_loss = None
        target1 = None
        target2 = None
        buy_range = "N/A"
    
    # Adjust for swing trading
    if trading_type == 'swing':
        if recommendation == "Buy":
            target1 = round(current_price * 1.06, 2)
            target2 = round(current_price * 1.12, 2)
            stop_loss = round(current_price * 0.94, 2)
        elif recommendation == "Sell":
            target1 = round(current_price * 0.94, 2)
            target2 = round(current_price * 0.88, 2)
            stop_loss = round(current_price * 1.06, 2)
    
    return {
        'Symbol': symbol,
        'Current Price': current_price,
        'Open': open_price,
        'High': high,
        'Low': low,
        'RSI': round(rsi, 2) if not np.isnan(rsi) else None,
        'MACD': round(data['MACD'], 2) if not np.isnan(data['MACD']) else None,
        'BB Lower': round(bb_lower, 2) if not np.isnan(bb_lower) else None,
        'BB Upper': round(bb_upper, 2) if not np.isnan(bb_upper) else None,
        'SMA 20': round(sma_20, 2) if not np.isnan(sma_20) else None,
        'SMA 50': round(sma_50, 2) if not np.isnan(sma_50) else None,
        'Recommendation': recommendation,
        'Confidence Score': min(100, max(0, confidence)),
        'Buy Range': buy_range,
        'Stop Loss': stop_loss,
        'Target 1': target1,
        'Target 2': target2,
        'Price Action': price_action,
        'Reasons': ", ".join(reasons) if reasons else "No strong signals"
    }

# Main Streamlit app
def main():
    st.title("Advanced Stock Recommender with Technical Analysis")
    st.write("Analyzes stocks using multiple technical indicators for intraday and swing trading")
    
    # Load stock list from Excel
    try:
        stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
    except FileNotFoundError:
        st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
        return
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        selected_sheet = st.selectbox("Select Stock List", stock_sheets)
    with col2:
        trading_type = st.selectbox("Select Trading Type", ['intraday', 'swing'])
    
    analyze_button = st.button("Analyze Stocks")
    
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
                result = analyze_stock(symbol, trading_type)
                if result is not None:
                    results.append(result)
                progress_bar.progress((i + 1) / len(symbols))
            
            if not results:
                st.warning("No valid stock data could be fetched. Please try again later.")
                return
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Filter only Buy/Sell recommendations
            actionable_df = results_df[results_df['Recommendation'].isin(['Buy', 'Sell'])]
            
            # Display results
            st.subheader("All Analyzed Stocks")
            st.dataframe(results_df.style.applymap(
                lambda x: 'background-color: lightgreen' if x == 'Buy' else 
                         'background-color: lightcoral' if x == 'Sell' else '',
                subset=['Recommendation']
            ))
            
            st.subheader("Actionable Recommendations (Buy/Sell)")
            if not actionable_df.empty:
                # Sort by confidence score
                actionable_df = actionable_df.sort_values('Confidence Score', ascending=False)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Buy Signals", len(actionable_df[actionable_df['Recommendation'] == 'Buy']))
                col2.metric("Total Sell Signals", len(actionable_df[actionable_df['Recommendation'] == 'Sell']))
                col3.metric("Highest Confidence", f"{actionable_df.iloc[0]['Confidence Score']}%")
                
                st.dataframe(actionable_df)
                
                # Download buttons
                st.download_button(
                    label="Download All Results as CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'stock_recommendations_{trading_type}_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
                
                st.download_button(
                    label="Download Actionable Recommendations as CSV",
                    data=actionable_df.to_csv(index=False).encode('utf-8'),
                    file_name=f'actionable_recommendations_{trading_type}_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
                
                # Show top recommendations
                st.subheader("Top Recommendations")
                top_buys = actionable_df[actionable_df['Recommendation'] == 'Buy'].nlargest(3, 'Confidence Score')
                top_sells = actionable_df[actionable_df['Recommendation'] == 'Sell'].nsmallest(3, 'Confidence Score')
                
                if not top_buys.empty:
                    st.write("**Best Buy Opportunities**")
                    st.dataframe(top_buys)
                
                if not top_sells.empty:
                    st.write("**Best Sell Opportunities**")
                    st.dataframe(top_sells)
                
            else:
                st.info("No strong Buy/Sell recommendations today based on the criteria.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
