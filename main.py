import os
import pandas as pd
from pathlib import Path
# If your folder is named 'source', change 'src' to 'source' below
from src.data_loader import load_raw_data, validate_scm_data
from src.features import encode_categories
from src.model import create_model, train_model, get_predictions
from src.utils import setup_logging

def main():
    # 1. Initialize Logging
    logger = setup_logging()
    logger.info("--- Starting Anomaly Detection Pipeline ---")

    # 2. Define File Paths
    # We assume your raw data is in the root or data/raw
    BASE_DIR = Path(__file__).resolve().parent
    
    input_path = os.path.join(BASE_DIR, "data", "raw", "revised_scm_data.csv")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    output_report = os.path.join(BASE_DIR, "final_anomaly_report.csv")
    
    # 3. Load and Validate Data
    try:
        df_raw = load_raw_data(input_path)
        if not validate_scm_data(df_raw):
            logger.error("Data validation failed. Exiting.")
            return
    except Exception as e:
        logger.error(f"Error during loading: {e}")
        return

    # 4. Process Data (Encoding text to numbers)
    logger.info("Translating categorical data...")
    df_processed, _ = encode_categories(df_raw)

    # Ensure processed folder exists and save the math-version
    os.makedirs(processed_dir, exist_ok=True)
    df_processed.to_csv(os.path.join(processed_dir, "encoded_data.csv"), index=False)

    # 5. Train the Model (The Brain)
    # Contamination 0.05 means we expect 5% of data to be anomalies
    model = create_model(contamination=0.05)
    trained_model = train_model(model, df_processed)

    # 6. Generate Predictions
    logger.info("Running anomaly detection...")
    predictions = get_predictions(trained_model, df_processed)
    
    # Add the results back to the original human-readable data
    df_raw['model_is_anomaly'] = predictions

    # 7. Save Final Results
    df_anomalies = df_raw[df_raw['model_is_anomaly'] == True].copy()

    if not df_anomalies.empty:
        df_anomalies.to_csv(output_report, index=False)
        logger.info(f"SUCCESS: Found {len(df_anomalies)} anomalies. Report saved: {output_report}")
    else:
        logger.info("No anomalies detected based on current model parameters.")

if __name__ == "__main__":
    main()