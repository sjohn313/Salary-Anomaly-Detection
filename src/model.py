from sklearn.ensemble import IsolationForest
import numpy as np

def create_model(contamination=0.05):
    """
    Initializes the Isolation Forest model.
    Business context: 'contamination' is the expected % of payroll anomalies.
    """
    return IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )

def train_model(model, X):
    """
    Fits the model to the processed feature set.
    """
    model.fit(X)
    return model

def get_predictions(model, X):
    """
    Returns boolean predictions: True if anomaly, False if normal.
    Isolation Forest returns -1 for outliers and 1 for inliers.
    """
    preds = model.predict(X)
    return [True if p == -1 else False for p in preds]