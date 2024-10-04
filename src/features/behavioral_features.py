import pandas as pd
from typing import Dict

class BehavioralFeatureExtractor:
    def __init__(self, transaction_data: pd.DataFrame):
        self.transaction_data = transaction_data

    def extract_frequency_features(self) -> Dict[str, Dict[str, float]]:
        """
        Extract behavioral features related to transaction frequency.
        :return: A dictionary where keys are user IDs and values are dictionaries of behavioral features.
        """
        features = {}
        user_transaction_counts = self.transaction_data.groupby('from').size()

        for user, count in user_transaction_counts.items():
            total_amount = self.transaction_data[self.transaction_data['from'] == user]['amount'].sum()
            avg_transaction_size = total_amount / count

            features[user] = {
                'transaction_count': count,
                'total_transaction_amount': total_amount,
                'avg_transaction_size': avg_transaction_size
            }
        
        return features

    def extract_time_based_behavioral_features(self) -> Dict[str, Dict[str, float]]:
        """
        Extract behavioral features based on time intervals between transactions.
        :return: A dictionary where keys are user IDs and values are dictionaries of time-based behavioral features.
        """
        features = {}
        self.transaction_data['timestamp'] = pd.to_datetime(self.transaction_data['timestamp'])
        
        # Calculate time difference between consecutive transactions for each user
        self.transaction_data['time_diff'] = self.transaction_data.groupby('from')['timestamp'].diff().dt.total_seconds()
        
        for user, group in self.transaction_data.groupby('from'):
            time_diffs = group['time_diff'].dropna()  # Remove NaN values (first transaction has NaN)

            if len(time_diffs) > 0:
                avg_time_diff = time_diffs.mean()
                std_time_diff = time_diffs.std()
            else:
                avg_time_diff = 0
                std_time_diff = 0

            features[user] = {
                'avg_time_between_transactions': avg_time_diff,
                'std_time_between_transactions': std_time_diff
            }

        return features

    def extract_abnormal_transaction_features(self) -> Dict[str, Dict[str, float]]:
        """
        Extract behavioral features related to abnormal transaction patterns.
        :return: A dictionary where keys are user IDs and values are dictionaries of abnormal transaction features.
        """
        features = {}
        threshold = self.transaction_data['amount'].quantile(0.95)  # Define threshold for abnormal transactions (95th percentile)

        for user, group in self.transaction_data.groupby('from'):
            abnormal_transactions = group[group['amount'] > threshold]
            abnormal_count = abnormal_transactions.shape[0]
            abnormal_percentage = abnormal_count / group.shape[0] if group.shape[0] > 0 else 0

            features[user] = {
                'abnormal_transaction_count': abnormal_count,
                'abnormal_transaction_percentage': abnormal_percentage
            }

        return features

if __name__ == "__main__":
    # Load preprocessed transaction data
    transaction_data = pd.read_csv("data/processed/preprocessed_blockchain_data.csv")
    
    extractor = BehavioralFeatureExtractor(transaction_data)
    
    frequency_features = extractor.extract_frequency_features()
    time_based_features = extractor.extract_time_based_behavioral_features()
    abnormal_features = extractor.extract_abnormal_transaction_features()

    # Save extracted features
    pd.DataFrame.from_dict(frequency_features, orient='index').to_csv("data/processed/frequency_features.csv")
    pd.DataFrame.from_dict(time_based_features, orient='index').to_csv("data/processed/time_based_behavioral_features.csv")
    pd.DataFrame.from_dict(abnormal_features, orient='index').to_csv("data/processed/abnormal_transaction_features.csv")
