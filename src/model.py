from sklearn.ensemble import IsolationForest
import pickle

def train_isolation_forest(data, contamination=0.05):
    # Initialize the model with a specified contamination rate
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    
    # Fit model and predict (-1 for anomalies, 1 for normal)
    data['anomaly_score'] = model.fit_predict(data)
    
    # Save the model to the models directory
    with open('models/anomaly_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    return data