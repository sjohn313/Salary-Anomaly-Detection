import os
import pandas as pd
from src.data_loader import load_and_preprocess
from src.model import train_isolation_forest

def main():
    # Paths
    raw_data_path = os.path.join('data', 'raw', 'revised_scm_data.csv')
    output_csv = 'flagged_anomalies.csv'
    
    print("--- Initializing Anomaly Report Generation ---")
    
    # 1. Load the original data for the final report
    original_df = pd.read_csv(raw_data_path)
    
    # 2. Get the preprocessed (encoded) data for the model
    processed_df = load_and_preprocess(raw_data_path)
    
    # 3. Train model and get scores
    # results_df will have the anomaly_label and anomaly_score columns
    results_df = train_isolation_forest(processed_df)
    
    # 4. Map the results back to the original readable dataframe
    original_df['anomaly_label'] = results_df['anomaly_label']
    original_df['anomaly_score'] = results_df['anomaly_score']
    
    # 5. Filter for ONLY anomalies (label == -1)
    anomalies_only = original_df[original_df['anomaly_label'] == -1].copy()
    
    # Sort by anomaly_score (most anomalous first)
    anomalies_only = anomalies_only.sort_values(by='anomaly_score')
    
    # 6. Save only the anomalies to the root directory
    anomalies_only.to_csv(output_csv, index=False)
    
    # Statistical Summary for Console
    total_count = len(original_df)
    anomaly_count = len(anomalies_only)
    
    print("\n" + "="*30)
    print("ANOMALY DETECTION SUMMARY")
    print("="*30)
    print(f"Total Records Scanned: {total_count}")
    print(f"Anomalies Identified:  {anomaly_count}")
    print(f"Contamination Rate:    { (anomaly_count/total_count)*100:.2f}%")
    print("-" * 30)
    print(f"SUCCESS: Filtered report saved to: {os.path.abspath(output_csv)}")
    print("="*30)

if __name__ == "__main__":
    main()