import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to fetch real-time or latest stock data
def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        
        # Get intraday data if market is open
        intraday = stock.history(period='1d', interval='5m')
        
        # If we have intraday data (market is open), use it
        if not intraday.empty and len(intraday) > 1:
            latest_data = intraday.iloc[-1]
            prev_data = intraday.iloc[-2]
            is_real_time = True
        else:
            # Fall back to daily data if market is closed
            daily = stock.history(period='5d')
            if daily.empty:
                return None
            latest_data = daily.iloc[-1]
            prev_data = daily.iloc[-2] if len(daily) > 1 else latest_data
            is_real_time = False
        
        return {
            'symbol': symbol,
            'open': latest_data['Open'],
            'high': latest_data['High'],
            'low': latest_data['Low'],
            'close': latest_data['Close'],
            'volume': latest_data['Volume'],
            'prev_close': prev_data['Close'],
            'is_real_time': is_real_time,
            'last_updated': latest_data.name if hasattr(latest_data, 'name') else datetime.now()
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Technical indicators calculations
def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi[-1]

def calculate_sma(prices, window=20):
    return np.mean(prices[-window:])

def calculate_ema(prices, window=20):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.convolve(prices[-window:], weights, mode='valid')[0]

def calculate_macd(prices, slow=26, fast=12, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    return macd

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[-window:])
    upper = sma + (rolling_std * num_std)
    lower = sma - (rolling_std * num_std)
    return upper, lower

# Function to analyze stock condition with confidence score and Open-High/Low condition
def analyze_stock(symbol):
    # Get the latest data
    data = get_stock_data(symbol)
    if data is None:
        return None
    
    # Get historical data for indicators
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1mo')  # 1 month of data for indicators
        if hist.empty:
            return None
        
        close_prices = hist['Close'].values
        high_prices = hist['High'].values
        low_prices = hist['Low'].values
        
        # Calculate technical indicators
        rsi = calculate_rsi(close_prices)
        sma_20 = calculate_sma(close_prices, 20)
        sma_50 = calculate_sma(close_prices, 50)
        ema_20 = calculate_ema(close_prices, 20)
        macd = calculate_macd(close_prices)
        upper_bb, lower_bb = calculate_bollinger_bands(close_prices)
        
        current_price = data['close']
        prev_close = data['prev_close']
        price_change = ((current_price - prev_close) / prev_close) * 100
        
        # Initialize confidence score (0-100)
        confidence = 50  # Neutral baseline
        
        # Price action analysis
        is_bullish_candle = current_price > data['open']
        is_bearish_candle = current_price < data['open']
        
        # Open-High-Low analysis (your specific condition)
        open_high_condition = data['open'] == data['high']
        open_low_condition = data['open'] == data['low']
        
        # Trend analysis
        short_term_trend = "Up" if current_price > sma_20 else "Down"
        medium_term_trend = "Up" if current_price > sma_50 else "Down"
        
        # Volume analysis (simple)
        volume_avg = np.mean(hist['Volume'].values[-20:])  # 20-day average
        volume_spike = data['volume'] > volume_avg * 1.5
        
        # RSI analysis
        rsi_signal = None
        if rsi > 70:
            rsi_signal = "Overbought"
            confidence -= 10 if is_bullish_candle else 5
        elif rsi < 30:
            rsi_signal = "Oversold"
            confidence += 10 if is_bearish_candle else 5
        
        # MACD analysis
        macd_signal = "Bullish" if macd > 0 else "Bearish"
        
        # Bollinger Bands analysis
        bb_signal = None
        if current_price > upper_bb:
            bb_signal = "Overbought"
            confidence -= 5
        elif current_price < lower_bb:
            bb_signal = "Oversold"
            confidence += 5
        
        # Determine recommendation based on multiple factors including Open-High/Low
        recommendation = "Neutral"
        
        # Strong conditions from your original code
        if open_high_condition:  # Bearish condition
            recommendation = "Sell"
            confidence = max(confidence, 70)  # Boost confidence for this clear pattern
            stop_loss = round(current_price * 1.02, 2)  # 2% above current price
            target = round(current_price * 0.96, 2)     # 4% below current price
            condition = "Open=High (Bearish)"
        elif open_low_condition:  # Bullish condition
            recommendation = "Buy"
            confidence = max(confidence, 70)  # Boost confidence for this clear pattern
            stop_loss = round(current_price * 0.98, 2)  # 2% below current price
            target = round(current_price * 1.04, 2)     # 4% above current price
            condition = "Open=Low (Bullish)"
        else:
            # If no clear Open-High/Low pattern, use the multi-factor approach
            condition = "No clear Open-High/Low pattern"
            
            # Bullish factors
            bullish_factors = 0
            if is_bullish_candle: bullish_factors += 1
            if short_term_trend == "Up": bullish_factors += 1
            if medium_term_trend == "Up": bullish_factors += 1
            if rsi_signal == "Oversold": bullish_factors += 1
            if macd_signal == "Bullish": bullish_factors += 1
            if bb_signal == "Oversold": bullish_factors += 1
            if volume_spike and is_bullish_candle: bullish_factors += 2
            
            # Bearish factors
            bearish_factors = 0
            if is_bearish_candle: bearish_factors += 1
            if short_term_trend == "Down": bearish_factors += 1
            if medium_term_trend == "Down": bearish_factors += 1
            if rsi_signal == "Overbought": bearish_factors += 1
            if macd_signal == "Bearish": bearish_factors += 1
            if bb_signal == "Overbought": bearish_factors += 1
            if volume_spike and is_bearish_candle: bearish_factors += 2
            
            # Determine final recommendation
            if bullish_factors - bearish_factors >= 3:
                recommendation = "Buy"
                confidence += (bullish_factors - bearish_factors) * 5
            elif bearish_factors - bullish_factors >= 3:
                recommendation = "Sell"
                confidence += (bearish_factors - bullish_factors) * 5
            
            # Calculate stop loss and target based on volatility if no Open-High/Low pattern
            atr = np.mean(np.maximum(high_prices[-14:] - low_prices[-14:], 
                                   np.abs(high_prices[-14:] - close_prices[-14:]), 
                                   np.abs(low_prices[-14:] - close_prices[-14:])))
            
            if recommendation == "Buy":
                stop_loss = round(current_price - atr * 1.5, 2)
                target = round(current_price + atr * 3, 2)
            elif recommendation == "Sell":
                stop_loss = round(current_price + atr * 1.5, 2)
                target = round(current_price - atr * 3, 2)
            else:
                stop_loss = None
                target = None
        
        # Cap confidence between 0 and 100
        confidence = max(0, min(100, confidence))
        
        return {
            'Symbol': symbol,
            'Current Price': round(current_price, 2),
            'Change (%)': round(price_change, 2),
            'Open': round(data['open'], 2),
            'High': round(data['high'], 2),
            'Low': round(data['low'], 2),
            'Volume': f"{data['volume']:,.0f}",
            'RSI (14)': round(rsi, 2),
            'SMA (20)': round(sma_20, 2),
            'SMA (50)': round(sma_50, 2),
            'MACD': round(macd, 4),
            'Bollinger Band': f"{round(lower_bb, 2)} - {round(upper_bb, 2)}",
            'Recommendation': recommendation,
            'Confidence (%)': round(confidence),
            'Stop Loss': stop_loss,
            'Target': target,
            'Condition': condition,
            'Trend (S/M)': f"{short_term_trend}/{medium_term_trend}",
            'Data Freshness': "Real-time" if data['is_real_time'] else "EOD",
            'Last Updated': data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("📈 Real-Time Stock Analysis & Recommendation System")
    st.write("""
    This tool provides real-time stock analysis using multiple technical indicators 
    including Open-High/Low conditions, and generates recommendations with confidence scores.
    """)
    
    # Add refresh button
    if st.button("🔄 Refresh All Data"):
        st.experimental_rerun()
    
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
        min_confidence = st.slider("Minimum Confidence (%)", 0, 100, 70)
    
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
                result = analyze_stock(symbol)
                if result is not None:
                    results.append(result)
                progress_bar.progress((i + 1) / len(symbols))
            
            if not results:
                st.warning("No valid stock data could be fetched. Please try again later.")
                return
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Filter only recommendations meeting confidence threshold
            actionable_df = results_df[
                (results_df['Recommendation'].isin(['Buy', 'Sell'])) & 
                (results_df['Confidence (%)'] >= min_confidence)
            ].sort_values('Confidence (%)', ascending=False)
            
            # Display results
            st.subheader("📊 All Analyzed Stocks")
            st.dataframe(results_df.style.background_gradient(
                subset=['Confidence (%)'], 
                cmap='RdYlGn', 
                vmin=0, 
                vmax=100
            ))
            
            st.subheader("🚀 High-Confidence Actionable Recommendations")
            if not actionable_df.empty:
                st.dataframe(actionable_df.style.background_gradient(
                    subset=['Confidence (%)'], 
                    cmap='RdYlGn', 
                    vmin=min_confidence, 
                    vmax=100
                ))
                
                # Download buttons
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download All Results as CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name=f'stock_recommendations_{timestamp}.csv',
                        mime='text/csv'
                    )
                with col2:
                    st.download_button(
                        label="💾 Download Actionable Recommendations as CSV",
                        data=actionable_df.to_csv(index=False).encode('utf-8'),
                        file_name=f'actionable_recommendations_{timestamp}.csv',
                        mime='text/csv'
                    )
            else:
                st.info("No strong Buy/Sell recommendations meet your confidence threshold today.")
            
            # Show summary statistics
            st.subheader("📈 Recommendation Summary")
            if not results_df.empty:
                rec_counts = results_df['Recommendation'].value_counts()
                st.write(f"Total Stocks Analyzed: {len(results_df)}")
                st.write(f"Buy Recommendations: {rec_counts.get('Buy', 0)}")
                st.write(f"Sell Recommendations: {rec_counts.get('Sell', 0)}")
                st.write(f"Neutral Recommendations: {rec_counts.get('Neutral', 0)}")
                
                # Count Open-High/Low conditions
                open_high_count = len(results_df[results_df['Condition'] == "Open=High (Bearish)"])
                open_low_count = len(results_df[results_df['Condition'] == "Open=Low (Bullish)"])
                st.write(f"Open=High Patterns Found: {open_high_count}")
                st.write(f"Open=Low Patterns Found: {open_low_count}")
                
                avg_confidence = results_df['Confidence (%)'].mean()
                st.write(f"Average Confidence: {avg_confidence:.1f}%")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
