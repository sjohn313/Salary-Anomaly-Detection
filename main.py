from src.data_loader import load_and_preprocess
from src.model import train_isolation_forest
import os

def main():
    # Define paths based on your directory structure
    data_path = 'data/raw/revised_scm_data.csv'
    
    # Ensure model directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    print("--- Starting Anomaly Detection Workflow ---")
    
    # Load and preprocess
    processed_data = load_and_preprocess(data_path)
    
    # Detect anomalies
    results = train_isolation_forest(processed_data)
    
    # Output results
    anomalies = results[results['anomaly_score'] == -1]
    print(f"Total Records Analyzed: {len(results)}")
    print(f"Anomalies Detected: {len(anomalies)}")
    print("\nSample Anomalous Records:")
    print(anomalies.head())

if __name__ == "__main__":
    main()