import pandas as pd
import numpy as np
from typing import List, Dict

class DataValidator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validation_results = {}

    def check_missing_values(self) -> Dict[str, float]:
        missing_percentages = (self.data.isnull().sum() / len(self.data)) * 100
        self.validation_results['missing_values'] = missing_percentages.to_dict()
        return self.validation_results['missing_values']

    def check_data_types(self) -> Dict[str, str]:
        self.validation_results['data_types'] = self.data.dtypes.astype(str).to_dict()
        return self.validation_results['data_types']

    def check_value_ranges(self, numeric_columns: List[str]) -> Dict[str, Dict[str, float]]:
        ranges = {}
        for col in numeric_columns:
            ranges[col] = {
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'mean': self.data[col].mean(),
                'median': self.data[col].median()
            }
        self.validation_results['value_ranges'] = ranges
        return ranges

    def check_unique_values(self, categorical_columns: List[str]) -> Dict[str, int]:
        unique_counts = {col: self.data[col].nunique() for col in categorical_columns}
        self.validation_results['unique_values'] = unique_counts
        return unique_counts

    def check_correlations(self, threshold: float = 0.8) -> Dict[str, List[str]]:
        corr_matrix = self.data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = (upper_tri > threshold).any()
        high_corr_features = high_corr[high_corr].index.tolist()
        self.validation_results['high_correlations'] = {
            'threshold': threshold,
            'features': high_corr_features
        }
        return self.validation_results['high_correlations']

    def validate_all(self, numeric_columns: List[str], categorical_columns: List[str]) -> Dict:
        self.check_missing_values()
        self.check_data_types()
        self.check_value_ranges(numeric_columns)
        self.check_unique_values(categorical_columns)
        self.check_correlations()
        return self.validation_results

    def generate_report(self) -> str:
        report = "Data Validation Report\n"
        report += "=" * 25 + "\n\n"

        for check, results in self.validation_results.items():
            report += f"{check.replace('_', ' ').title()}:\n"
            report += "-" * 25 + "\n"
            if isinstance(results, dict):
                for key, value in results.items():
                    report += f"  {key}: {value}\n"
            elif isinstance(results, list):
                for item in results:
                    report += f"  {item}\n"
            report += "\n"

        return report

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np

    # Generate dummy data
    data = pd.DataFrame({
        'amount': np.random.randint(1, 1000, 1000),
        'transaction_type': np.random.choice(['A', 'B', 'C'], 1000),
        'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.99, 0.01])
    })

    validator = DataValidator(data)
    results = validator.validate_all(
        numeric_columns=['amount'],
        categorical_columns=['transaction_type', 'is_fraud']
    )

    print(validator.generate_report())