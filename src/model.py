import os
import pickle
from sklearn.ensemble import IsolationForest

def train_isolation_forest(data, contamination=0.05):
    # Initialize the model
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    
    # Fit and get labels (-1 for anomaly, 1 for normal)
    data['anomaly_label'] = model.fit_predict(data)
    
    # Capture the raw 'score'. Lower/more negative values are more anomalous.
    # We drop the label column for the score calculation to avoid bias
    data['anomaly_score'] = model.decision_function(data.drop(columns=['anomaly_label']))
    
    # Save the model artifact
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'anomaly_model.pkl')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    return data