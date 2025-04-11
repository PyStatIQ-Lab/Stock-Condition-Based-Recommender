import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(symbol, period='1d'):
    try:
        stock = yf.Ticker(symbol + '')
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist.iloc[-1]
    except:
        return None

# Function to analyze stock condition
def analyze_stock(symbol):
    data = get_stock_data(symbol)
    if data is None:
        return None
    
    current_price = round(data['Close'], 2)
    
    # Calculate stop loss and target (simple percentage-based)
    if data['Open'] == data['High']:  # Bearish condition
        recommendation = "Sell"
        stop_loss = round(current_price * 1.02, 2)  # 2% above current price
        target = round(current_price * 0.96, 2)     # 4% below current price
    elif data['Open'] == data['Low']:  # Bullish condition
        recommendation = "Buy"
        stop_loss = round(current_price * 0.98, 2)  # 2% below current price
        target = round(current_price * 1.04, 2)     # 4% above current price
    else:
        recommendation = "Neutral"
        stop_loss = None
        target = None
    
    return {
        'Symbol': symbol,
        'Current Price': current_price,
        'Open': round(data['Open'], 2),
        'High': round(data['High'], 2),
        'Low': round(data['Low'], 2),
        'Close': current_price,
        'Recommendation': recommendation,
        'Stop Loss': stop_loss,
        'Target': target,
        'Condition': "Open=High (Bearish)" if data['Open'] == data['High'] else 
                    "Open=Low (Bullish)" if data['Open'] == data['Low'] else "No clear pattern"
    }

# Main Streamlit app
def main():
    st.title("Stock Condition-Based Recommender")
    st.write("Analyzes stocks based on Open-High/Low conditions and provides recommendations")
    
    # Load stock list from Excel
    try:
        stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
    except FileNotFoundError:
        st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
        return
    
    # User inputs
    selected_sheet = st.selectbox("Select Stock List", stock_sheets)
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
            
            # Filter only Buy/Sell recommendations
            actionable_df = results_df[results_df['Recommendation'].isin(['Buy', 'Sell'])]
            
            # Display results
            st.subheader("All Analyzed Stocks")
            st.dataframe(results_df)
            
            st.subheader("Actionable Recommendations (Buy/Sell)")
            if not actionable_df.empty:
                st.dataframe(actionable_df)
                
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
                st.info("No strong Buy/Sell recommendations today based on the criteria.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
