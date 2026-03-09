import pandas as pd
import logging

logger = logging.getLogger(__name__)

def encode_categories(df):
    
    # 2. Identify text-based (categorical) columns
    categorical_cols = df_features.select_dtypes(include=['object', 'bool']).columns.tolist()
    logger.info(f"Encoding categorical columns: {categorical_cols}")

    # 3. One-Hot Encoding (Translates 'Gender' into gender_Male, gender_Female, etc.)
    # We use drop_first=True to avoid redundant data (the 'dummy variable trap')
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)
    
    logger.info(f"Encoding complete. Features expanded from {len(df_features.columns)} to {len(df_encoded.columns)}.")
    
    return df_encoded, target