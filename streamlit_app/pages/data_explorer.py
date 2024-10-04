import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.blockchain_data_collector import BlockchainDataCollector
from src.data.exchange_data_collector import ExchangeDataCollector

def app():
    st.title("Data Explorer")
    
    data_source = st.selectbox("Select data source", ["Blockchain", "Exchange"])
    
    if data_source == "Blockchain":
        collector = BlockchainDataCollector(file_path='data/external/transactions.csv')
        data = collector.get_data()
    else:
        collector1 = ExchangeDataCollector(file_path='data/external/transactions1.csv')
        data = collector1.get_data()
    
    # Display the first few rows of the data
    st.write(data.head())
    
    st.subheader("Data Statistics")
    st.write(data.describe())

    # Ensure correct data types for plotting
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')  # Convert to datetime

    # Ensure specific columns are floats (replace 'price' and 'volume' with actual column names)
    if 'amount' in data.columns:
        data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
    if 'fee' in data.columns:
        data['fee'] = pd.to_numeric(data['fee'], errors='coerce')

    # Drop rows with NaN values in key columns
    data = data.dropna(subset=['timestamp', 'amount'])  # Adjust based on your needs

    st.subheader("Data Visualization")
    chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Scatter Plot"])
    
    if chart_type == "Line Chart":
        if 'timestamp' in data.columns and 'amount' in data.columns:
            st.line_chart(data.set_index('timestamp')['amount'])
        else:
            st.error("Required columns for line chart not found.")

    elif chart_type == "Bar Chart":
        if 'timestamp' in data.columns and 'fee' in data.columns:
            st.bar_chart(data.set_index('timestamp')['fee'])
        else:
            st.error("Required columns for bar chart not found.")

    elif chart_type == "Scatter Plot":
        if 'timestamp' in data.columns and 'amount' in data.columns:
            st.scatter_chart(data.set_index('timestamp')[['amount']])
        else:
            st.error("Required columns for scatter plot not found.")

# Run the app
if __name__ == "__main__":
    app()
