import joblib
import numpy as np
import pandas as pd
from app.models.graph_neural_network import GraphNeuralNetwork
from app.models.ensemble_model import EnsembleModel

class FraudDetectionService:
    def __init__(self):
        self.gnn_model = GraphNeuralNetwork.load("models/gnn_model.pt")
        self.ensemble_model = joblib.load("models/ensemble_model.joblib")
        self.scaler = joblib.load("models/feature_scaler.joblib")

    def preprocess(self, transaction):
        # Convert transaction to DataFrame
        df = pd.DataFrame([transaction])
        
        # Apply feature engineering
        df = self.engineer_features(df)
        
        # Scale features
        scaled_features = self.scaler.transform(df)
        
        return scaled_features

    def engineer_features(self, df):
        # Add your feature engineering logic here
        # This is a placeholder for demonstration
        df['amount_log'] = np.log1p(df['amount'])
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        return df

    def predict(self, transaction):
        preprocessed_data = self.preprocess(transaction)
        
        # Get GNN prediction
        gnn_prediction = self.gnn_model.predict(preprocessed_data)
        
        # Get ensemble model prediction
        ensemble_prediction = self.ensemble_model.predict(preprocessed_data)
        
        # Combine predictions (you can implement more sophisticated logic here)
        final_prediction = (gnn_prediction + ensemble_prediction) / 2
        
        return float(final_prediction)

    def analyze(self, transaction):
        prediction = self.predict(transaction)
        risk_score = prediction * 100  # Convert to percentage
        
        result = {
            "risk_score": risk_score,
            "is_suspicious": risk_score > 70,  # Threshold can be adjusted
            "recommendation": "Further investigation required" if risk_score > 70 else "Transaction appears normal"
        }
        
        return result