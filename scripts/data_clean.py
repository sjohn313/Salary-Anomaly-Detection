import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop protected/irrelevant columns
    df.drop(columns=['gender', 'sex', 'race', 'ethnicity', 'pregnancy', 'department'], inplace=True)

    # Encode education ordinally
    edu_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df['education'] = df['education'].map(edu_map)

    # Encode position_level hierarchically
    level_map = {'Entry': 1, 'Mid': 2, 'Director': 3, 'Executive': 4}
    df['position_level'] = df['position_level'].map(level_map)

    # Create salary_gap feature and drop raw salary columns
    df['salary_gap'] = (df['negotiated_salary'] - df['market_rate_salary']) / df['market_rate_salary']
    df.drop(columns=['market_rate_salary', 'negotiated_salary'], inplace=True)

    # Standardise all numeric columns
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(df.head())

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, 'data', 'scm_data.csv')
    output_path = os.path.join(base, 'data', 'revised_scm_data.csv')
    preprocess(input_path, output_path)
