import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.model_trainer import ModelTrainer

def app():
    st.title("Model Training")

    # File uploader for CSV data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data shape:", data.shape)
        st.write(data.head())

        # Initialize ModelTrainer
        trainer = ModelTrainer(data)

        # Model selection
        model_type = st.selectbox("Select model type", ["All", "LSTM", "Ensemble", "GNN"])

        # Hyperparameters
        st.subheader("Hyperparameters")
        if model_type in ["All", "LSTM"]:
            lstm_hidden_dim = st.number_input("LSTM Hidden dimension", min_value=8, max_value=256, value=64)
            lstm_num_layers = st.number_input("LSTM Number of layers", min_value=1, max_value=5, value=2)

        if model_type in ["All", "GNN"]:
            gnn_num_classes = st.number_input("GNN Number of classes", min_value=2, max_value=10, value=2)

        # Training button
        if st.button("Train Model(s)"):
            with st.spinner("Training model(s)..."):
                if model_type == "All":
                    trainer.train_all_models()
                elif model_type == "LSTM":
                    trainer.train_lstm(input_dim=data.shape[1]-1, hidden_dim=lstm_hidden_dim, num_layers=lstm_num_layers)
                elif model_type == "Ensemble":
                    trainer.train_ensemble()
                elif model_type == "GNN":
                    if trainer.graph_data is None:
                        st.error("Graph data is required to train GNN. Please upload graph data.")
                    else:
                        trainer.train_gnn(num_node_features=data.shape[1]-1, num_classes=gnn_num_classes)

            st.success("Model(s) trained successfully!")

            # Save models
            save_path = st.text_input("Enter path to save models", value="./models")
            if st.button("Save Models"):
                trainer.save_models(save_path)
                st.success(f"Models saved to {save_path}")

            # Option to download models
            if os.path.exists(save_path):
                for model_file in os.listdir(save_path):
                    with open(os.path.join(save_path, model_file), "rb") as f:
                        st.download_button(
                            label=f"Download {model_file}",
                            data=f,
                            file_name=model_file,
                            mime="application/octet-stream"
                        )

app()