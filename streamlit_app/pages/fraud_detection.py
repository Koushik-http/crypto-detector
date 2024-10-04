import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models import EnsembleModel
from src.visualization import network_visualizer

def app():
    st.title("Fraud Detection")
    
    uploaded_file = st.file_uploader("Upload transaction data", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.write(data.head())
        
        if st.button("Detect Fraud"):
            with st.spinner("Analyzing transactions..."):
                results = EnsembleModel.predict(data)
            
            st.subheader("Results")
            st.write(results)
            
            st.subheader("Network Visualization")
            fig = network_visualizer.plot_transaction_network(data, results)
            st.pyplot(fig)

        st.subheader("Transaction Statistics")
        st.write(data.describe())


app()