import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_blockchain_data(self, input_file: str) -> pd.DataFrame:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Add more preprocessing steps as needed
        return df

    def preprocess_exchange_data(self, input_file: str) -> pd.DataFrame:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=24).std()

        # Add more preprocessing steps as needed
        return df

    def scale_features(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
        df[features] = self.scaler.fit_transform(df[features])
        return df, self.scaler

    def save_preprocessed_data(self, df: pd.DataFrame, output_file: str):
        df.to_csv(output_file, index=False)
        print(f"Saved preprocessed data to {output_file}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    blockchain_df = preprocessor.preprocess_blockchain_data("data/raw/bitcoin_transactions.csv")
    exchange_df = preprocessor.preprocess_exchange_data("data/raw/binance_btc_usdt_1h.csv")
    
    features_to_scale = ['amount', 'fee', 'volume']
    blockchain_df, _ = preprocessor.scale_features(blockchain_df, features_to_scale)
    exchange_df, _ = preprocessor.scale_features(exchange_df, features_to_scale)
    
    preprocessor.save_preprocessed_data(blockchain_df, "data/processed/preprocessed_blockchain_data.csv")
    preprocessor.save_preprocessed_data(exchange_df, "data/processed/preprocessed_exchange_data.csv")