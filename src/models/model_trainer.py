import joblib
from sklearn.model_selection import train_test_split
from .deep_learning_model import train_lstm_model
from .ensemble_model import EnsembleModel
from .graph_neural_network import TemporalFeatureExtractor


class ModelTrainer:
    def __init__(self, data, graph_data=None):
        self.data = data
        self.graph_data = graph_data
        self.lstm_model = None
        self.ensemble_model = None
        self.gnn_model = None

    def prepare_data(self):
        # Assume self.data is a pandas DataFrame
        X = self.data.drop('is_fraud', axis=1)
        y = self.data['is_fraud']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_lstm(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.lstm_model = train_lstm_model(X_train.values, y_train.values, input_dim, 
                                           hidden_dim, num_layers, output_dim)
        return self.lstm_model

    def train_ensemble(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.ensemble_model = EnsembleModel()
        self.ensemble_model.train(X_train, y_train)
        return self.ensemble_model

    def train_gnn(self, num_node_features, num_classes):
        if self.graph_data is None:
            raise ValueError("Graph data is required to train GNN")
        self.gnn_model = TemporalFeatureExtractor(num_node_features, num_classes)
        TemporalFeatureExtractor(self.gnn_model, self.graph_data)
        return self.gnn_model

    def train_all_models(self):
        print("Training LSTM model...")
        self.train_lstm(input_dim=self.data.shape[1]-1)  # -1 to exclude the target variable
        
        print("Training Ensemble model...")
        self.train_ensemble()
        
        if self.graph_data is not None:
            print("Training GNN model...")
            self.train_gnn(num_node_features=self.data.shape[1]-1, num_classes=2)
        
        print("All models trained successfully.")

    def save_models(self, path):
        joblib.dump(self.lstm_model, f"{path}/lstm_model.joblib")
        joblib.dump(self.ensemble_model, f"{path}/ensemble_model.joblib")
        if self.gnn_model:
            joblib.dump(self.gnn_model, f"{path}/gnn_model.joblib")
        print(f"All models saved to {path}")

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np

    # Generate dummy data
    data = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feature_{i}' for i in range(20)])
    data['is_fraud'] = np.random.randint(0, 2, 1000)

    trainer = ModelTrainer(data)
    trainer.train_all_models()
    trainer.save_models('./models')