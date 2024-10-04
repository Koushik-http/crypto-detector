import pandas as pd
from typing import Dict, List

class ExchangeDataCollector:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load data from the CSV file."""
        return pd.read_csv(self.file_path)

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the exchange data."""
        df = self.load_data()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['price'] = df['price'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df

    def validate_data(self) -> Dict:
        """Validate the exchange data for missing values and data types."""
        df = self.load_data()
        missing_values = df.isnull().sum().to_dict()
        data_types = df.dtypes.astype(str).to_dict()
        validation_results = {
            'missing_values': missing_values,
            'data_types': data_types
        }
        return validation_results

    def collect_and_preprocess(self) -> Dict:
        """Collect data from CSV, preprocess, and validate it."""
        preprocessed_data = self.preprocess_data()
        validation_results = self.validate_data()
        return {
            'preprocessed_data': preprocessed_data,
            'validation': validation_results
        }

    def get_data(self) -> pd.DataFrame:
        """
        Load and return the exchange data as a DataFrame.
        """
        return self.load_data()

    def save_data_to_csv(self, output_file: str):
        """Save the preprocessed data to a CSV file."""
        preprocessed_data = self.preprocess_data()
        preprocessed_data.to_csv(output_file, index=False)
        print(f"Saved preprocessed exchange data to {output_file}")

if __name__ == "__main__":
    collector = ExchangeDataCollector(file_path="path/to/your/exchange_data.csv")
    # Accessing the preprocessed data
    preprocessed_data = collector.preprocess_data()
    print(preprocessed_data.head())
    
    # Accessing validation results
    validation_results = collector.validate_data()
    print(validation_results)
    
    # Saving preprocessed data to another CSV if needed
    collector.save_data_to_csv("data/raw/exchange_data_processed.csv")
