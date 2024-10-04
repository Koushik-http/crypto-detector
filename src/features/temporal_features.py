import pandas as pd
import numpy as np
from typing import List

class TemporalFeatureExtractor:
    def __init__(self, transaction_data: pd.DataFrame):
        self.transaction_data = transaction_data
        self.transaction_data['timestamp'] = pd.to_datetime(self.transaction_data['timestamp'])

    def extract_time_based_features(self) -> pd.DataFrame:
        self.transaction_data['hour'] = self.transaction_data['timestamp'].dt.hour
        self.transaction_data['day_of_week'] = self.transaction_data['timestamp'].dt.dayofweek
        self.transaction_data['is_weekend'] = self.transaction_data['day_of_week'].isin([5, 6]).astype(int)
        return self.transaction_data

    def calculate_rolling_statistics(self, column: str, windows: List[int]) -> pd.DataFrame:
        for window in windows:
            self.transaction_data[f'{column}_rolling_mean_{window}'] = self.transaction_data.groupby('from')[column].rolling(window=window).mean().reset_index(0, drop=True)
            self.transaction_data[f'{column}_rolling_std_{window}'] = self.transaction_data.groupby('from')[column].rolling(window=window).std().reset_index(0, drop=True)
        return self.transaction_data

    def calculate_time_since_last_transaction(self) -> pd.DataFrame:
        self.transaction_data['time_since_last_tx'] = self.transaction_data.groupby('from')['timestamp'].diff().dt.total_seconds()
        return self.transaction_data

if __name__ == "__main__":
    # Load preprocessed transaction data
    transaction_data = pd.read_csv("data/processed/preprocessed_blockchain_data.csv")
    
    extractor = TemporalFeatureExtractor(transaction_data)
    transaction_data = extractor.extract_time_based_features()
    transaction_data = extractor.calculate_rolling_statistics('amount', [10, 50, 100])
    transaction_data = extractor.calculate_time_since_last_transaction()
    
    # Save extracted features
    transaction_data.to_csv("data/processed/temporal_features.csv", index=False)