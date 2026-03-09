import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_raw_data(file_path):
    """Reads the SCM salary data from a CSV file."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Could not find {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise

def validate_scm_data(df):
    """
    Checks if the specific SCM columns are present.
    This acts as a 'Quality Check' gate.
    """
    required_cols = [
        'age', 'years_experience', 'market_rate_salary', 
        'negotiated_salary', 'is_anomaly'
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Data is missing critical columns: {missing}")
        return False
    
    logger.info("Data validation successful: All required columns present.")
    return True