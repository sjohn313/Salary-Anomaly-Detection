# Salary Anomaly Detection

A machine learning project that identifies unusual salary patterns in employee compensation data using an Isolation Forest algorithm.

## Overview

This tool analyzes salary records to flag statistical anomalies — cases where an employee's compensation significantly deviates from expected patterns given their experience, education, and position level. Potential anomalies include overpayment, underpayment, data entry errors, or other irregular compensation cases.

The model excludes demographic attributes (gender, race, ethnicity, etc.) both for fairness and for a structural reason specific to Isolation Forest — see the [Data Preprocessing](#1-data-preprocessing-scriptsdatacleanpy) section for the full explanation.

## Project Structure

```
Salary Anomaly Detection/
├── data/
│   ├── scm_data.csv              # Raw input dataset (1,000 records)
│   └── revised_scm_data.csv     # Preprocessed dataset used for training
├── scripts/
│   └── data_clean.py            # Data preprocessing pipeline
├── src/
│   └── model.py                 # Anomaly detection model
├── output/
│   └── flagged_anomalies.csv    # Detected anomalies (model output)
└── models/
    └── anomaly_model.pkl        # Saved trained model
```

## How It Works

### 1. Data Preprocessing ([`scripts/data_clean.py`](scripts/data_clean.py))

The original dataset contained several demographic variables (gender, sex, race, ethnicity, pregnancy, department). These were dropped before training — not only for fairness, but for a model-specific reason tied to how Isolation Forest works.

**Why demographic variables were removed**

Isolation Forest isolates anomalies by recursively splitting the feature space at randomly selected features. A data point that can be separated quickly (in fewer splits) is considered anomalous; one that requires many splits is considered normal.

The problem with including one-hot encoded categorical variables is one of proportion. A variable like race/ethnicity, encoded into 7 binary columns, contributes 7 features to the pool of candidates at each node split — while a single continuous variable like `years_experience` contributes only 1. Because the algorithm selects features randomly at each node, one-hot encoded columns are far more likely to be picked simply because they outnumber the continuous ones.

This creates a distortion: a genuinely anomalous salary record might follow an unusual compensation pattern, but if the algorithm keeps branching on race/ethnicity dummy columns (which carry no salary signal), the path to isolate that record becomes artificially long — making it look statistically normal. The anomaly gets buried.

By removing all demographic columns and using ordinal encoding for the remaining categorical features (education, position level), we ensure that every feature in the random selection pool is one that meaningfully contributes to salary anomaly detection.

**Preprocessing steps:**

- **Drops demographic columns**: gender, sex, race, ethnicity, pregnancy, department
- **Encodes education** ordinally: High School (1) → Bachelor's (2) → Master's (3) → PhD (4)
- **Encodes position level** hierarchically: Entry (1) → Mid (2) → Director (3) → Executive (4)
- **Creates `salary_gap` feature**: `(negotiated_salary - market_rate_salary) / market_rate_salary`
- **Standardizes** all numeric features using `RobustScaler` (IQR/median-based — see below)

**Why ordinal features can be standardized**

Education and position level are encoded as integers not because they are arbitrary categories, but because they are hierarchical — a PhD is meaningfully more than a Bachelor's degree in the same way an executive role outranks an entry-level one. This ordering means they behave more like continuous variables than categorical ones, and it is valid to standardize them alongside age and years of experience.

**Why robust standardization instead of z-score**

All continuous features — age, education, position level, and years of experience — are scaled using robust standardization, which positions each data point relative to the dataset's median and IQR rather than its mean and standard deviation.

Z-score standardization computes a point's position using the standard deviation of the entire dataset. This creates a problem for anomaly detection: a large outlier directly inflates the standard deviation. A sufficiently extreme value can widen the distribution enough that the algorithm no longer sees it as far from center — the outlier effectively masks itself by distorting the very metric used to measure it.

Robust standardization avoids this by using the median and IQR, both of which are resistant to extreme values. An outlier cannot pull the median far from the true center of the data, so it cannot hide behind a skewed scale.

This also supports post-hoc analysis. Because each feature is independently scaled on a comparable axis, it is possible to inspect the standardized output and identify which specific features — age, years of experience, education, position level — are driving a record's anomaly score. Raw values alone do not reveal this; a standardized feature far from center is a concrete signal that it may have contributed to the flag.

**Why `salary_gap` is a critical feature**

Rather than feeding `negotiated_salary` and `market_rate_salary` into the model as two separate columns, they are combined into a single derived feature that captures the relationship between the two:

```
salary_gap = (negotiated_salary - market_rate_salary) / market_rate_salary
```

A positive `salary_gap` means the employee's negotiated salary exceeds the market rate — the larger the value, the greater the premium above market. A negative value means the employee is being paid below market rate — the more negative, the larger the shortfall.

This feature is particularly powerful for anomaly detection because extreme values in either direction are meaningful signals worth investigating:

- A **large positive gap** may indicate favoritism or nepotism — someone receiving compensation far above what the role warrants
- A **large negative gap** may indicate potential discrimination — someone being systematically underpaid relative to their market rate

By consolidating two raw salary figures into one normalized ratio, the model can split on a single, high-signal feature rather than two correlated columns, making it easier to isolate data points where compensation is genuinely out of line with expectations. Flagged records can then be reviewed to determine whether the disparity reflects a legitimate reason or a potential violation of compensation policy.

### 2. Anomaly Detection ([`src/model.py`](src/model.py))

- Trains an **Isolation Forest** model on the preprocessed data
- Contamination rate: **5%** (flags approximately 50 of 1,000 records as anomalies)
- Outputs an `anomaly_score` per record — the lower the score, the more anomalous
- Saves the trained model to `models/anomaly_model.pkl` for reuse
- Exports flagged records to `output/flagged_anomalies.csv`, sorted by anomaly score

## Setup

### Requirements

- Python 3.8+
- scikit-learn
- pandas

Install dependencies:

```bash
pip install scikit-learn pandas
```

### Running the Project

**Step 1 — Preprocess the data:**

```bash
python scripts/data_clean.py
```

**Step 2 — Run the anomaly detection model:**

```bash
python src/model.py
```

Results will be saved to `output/flagged_anomalies.csv`.

## Output

The output CSV contains all flagged salary records with the following columns:

| Column | Description |
|---|---|
| `age` | Employee age (standardized) |
| `education` | Encoded education level (1–4) |
| `years_experience` | Years of experience (standardized) |
| `position_level` | Encoded position level (1–4) |
| `salary_gap` | Normalized difference between negotiated and market salary |
| `anomaly_label` | Always `-1` for flagged anomalies |
| `anomaly_score` | Decision function score (lower = more anomalous) |

## License

MIT License — Sebastiaan Johnson, 2026
