from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class EnsembleModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42)
    
    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        self.lr_model.fit(X_train, y_train)
        
        print("Random Forest Performance:")
        print(classification_report(y_val, self.rf_model.predict(X_val)))
        
        print("\nGradient Boosting Performance:")
        print(classification_report(y_val, self.gb_model.predict(X_val)))
        
        print("\nLogistic Regression Performance:")
        print(classification_report(y_val, self.lr_model.predict(X_val)))
    
    def predict(self, X):
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        gb_pred = self.gb_model.predict_proba(X)[:, 1]
        lr_pred = self.lr_model.predict_proba(X)[:, 1]
        
        # Simple averaging of probabilities
        ensemble_pred = (rf_pred + gb_pred + lr_pred) / 3
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        rf_pred = self.rf_model.predict_proba(X)[:, 1]
        gb_pred = self.gb_model.predict_proba(X)[:, 1]
        lr_pred = self.lr_model.predict_proba(X)[:, 1]
        
        return (rf_pred + gb_pred + lr_pred) / 3

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate dummy data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    ensemble = EnsembleModel()
    ensemble.train(X, y)
    
    # Make predictions
    test_sample = X[:10]
    predictions = ensemble.predict(test_sample)
    probabilities = ensemble.predict_proba(test_sample)
    
    print("\nEnsemble Predictions:")
    print(predictions)
    print("\nEnsemble Probabilities:")
    print(probabilities)