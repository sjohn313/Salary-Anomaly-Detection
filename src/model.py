import os
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'data', 'revised_scm_data.csv')
    output_dir = os.path.join(base, 'output')
    output_path = os.path.join(output_dir, 'flagged_anomalies.csv')
    model_dir = os.path.join(base, 'models')

    print("--- Initializing Anomaly Report Generation ---")

    df = pd.read_csv(data_path)

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['anomaly_label'] = model.fit_predict(df)
    df['anomaly_score'] = model.decision_function(df.drop(columns=['anomaly_label']))

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'anomaly_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    os.makedirs(output_dir, exist_ok=True)
    anomalies = df[df['anomaly_label'] == -1].sort_values('anomaly_score')
    anomalies.to_csv(output_path, index=False)

    print("\n" + "=" * 30)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 30)
    print(f"Total Records Scanned: {len(df)}")
    print(f"Anomalies Identified:  {len(anomalies)}")
    print(f"Contamination Rate:    {len(anomalies) / len(df) * 100:.2f}%")
    print("-" * 30)
    print(f"SUCCESS: Report saved to: {output_path}")
    print("=" * 30)

if __name__ == '__main__':
    main()
