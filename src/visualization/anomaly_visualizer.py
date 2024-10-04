import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyVisualizer:
    def __init__(self, anomaly_data: pd.DataFrame):
        """
        Initializes the AnomalyVisualizer with the provided anomaly data.
        :param anomaly_data: DataFrame containing anomaly detection results.
        """
        self.anomaly_data = anomaly_data

    def plot_anomaly_distribution(self):
        """
        Plots the distribution of anomalies in the dataset.
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.anomaly_data, x='anomaly_label')
        plt.title('Anomaly Distribution')
        plt.xlabel('Anomaly Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_anomaly_over_time(self):
        """
        Plots anomalies over time to observe trends.
        """
        if 'timestamp' not in self.anomaly_data.columns:
            raise ValueError("The anomaly data must contain a 'timestamp' column.")
        
        # Convert timestamp to datetime if not already
        self.anomaly_data['timestamp'] = pd.to_datetime(self.anomaly_data['timestamp'])
        anomaly_counts = self.anomaly_data.set_index('timestamp').resample('D').size()

        plt.figure(figsize=(14, 7))
        plt.plot(anomaly_counts.index, anomaly_counts.values, marker='o', linestyle='-')
        plt.title('Anomalies Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Anomalies')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
