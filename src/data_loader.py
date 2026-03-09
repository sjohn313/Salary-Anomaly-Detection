import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Categorical features to encode
    cat_cols = ['education', 'gender', 'sex', 'race', 'ethnicity', 
                'pregnancy', 'department', 'position_level']
    
    # Fill missing values if any and encode
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df