import pandas as pd
from typing import List, Dict

class BlockchainDataCollector:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Load data from the CSV file."""
        return pd.read_csv(self.file_path)

    def get_transactions(self) -> List[Dict]:
        """Get transactions as a list of dictionaries."""
        df = self.load_data()
        return df.to_dict(orient='records')

    def save_transactions_to_csv(self, transactions: List[Dict], output_file: str):
        """Save transactions to a CSV file."""
        df = pd.DataFrame(transactions)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(transactions)} transactions to {output_file}")

    def get_data(self) -> pd.DataFrame:
        """
        This method loads and returns the transaction data as a DataFrame.
        """
        return self.load_data()

if __name__ == "__main__":
    collector = BlockchainDataCollector(file_path="path/to/your/bitcoin_transactions.csv")
    transactions = collector.get_transactions()
    # You can save it to another CSV if needed
    collector.save_transactions_to_csv(transactions, "data/raw/bitcoin_transactions_processed.csv")
