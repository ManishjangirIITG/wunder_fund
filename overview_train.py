import pandas as pd
import numpy as np

# 1. Load the dataset
file_path = "datasets/train.parquet"
df = pd.read_parquet(file_path)

# 2. Basic dataset shapes & counts
num_rows = len(df)
num_sequences = df['seq_ix'].nunique()

# Identify feature columns vs metadata columns
metadata_cols = {'seq_ix', 'step_in_seq', 'need_prediction'}
feature_cols = [col for col in df.columns if col not in metadata_cols]
num_features = len(feature_cols)

# 3. Rows per sequence metrics
rows_per_seq = df.groupby('seq_ix').size()

# 4. Prediction steps and counts
need_pred_counts = df['need_prediction'].value_counts().to_dict()

# 5. Missing values per feature
missing_per_col = df[feature_cols].isnull().sum()

# 6. Feature statistics (mean, std, min, max)
feature_stats = df[feature_cols].agg(['mean', 'std', 'min', 'max']).T

# 7. Lag-1 correlation per feature (computed within each sequence boundary)
lag1_corrs = {}
for col in feature_cols:
    lag_series = df.groupby('seq_ix')[col].shift(1)
    lag1_corrs[col] = df[col].corr(lag_series)

lag1_df = pd.DataFrame.from_dict(lag1_corrs, orient='index', columns=['lag1_autocorr'])

# --- COMBINE & EXPORT METRICS ---

# A. Detailed Feature-level Summary Table
feature_summary = feature_stats.join(lag1_df)
feature_summary['missing_count'] = missing_per_col
feature_summary.index.name = 'feature'

# Save feature-level stats to CSV
feature_summary.to_csv("feature_statistics.csv")
print("Saved feature statistics to 'feature_statistics.csv'")

# B. General Dataset Overview Table
dataset_overview = pd.DataFrame([
    {"metric": "Total Rows", "value": num_rows},
    {"metric": "Total Sequences", "value": num_sequences},
    {"metric": "Number of Features", "value": num_features},
    {"metric": "Rows per Seq (Min)", "value": rows_per_seq.min()},
    {"metric": "Rows per Seq (Max)", "value": rows_per_seq.max()},
    {"metric": "Rows per Seq (Mean)", "value": round(rows_per_seq.mean(), 2)},
    {"metric": "Total Prediction Steps", "value": df['need_prediction'].sum()},
    {"metric": "Total Missing Values", "value": df.isnull().sum().sum()},
])

# Append need_prediction counts into overview
for val, count in need_pred_counts.items():
    dataset_overview = pd.concat([
        dataset_overview, 
        pd.DataFrame([{"metric": f"need_prediction_count (val={val})", "value": count}])
    ], ignore_index=True)

# Save high-level overview to CSV
dataset_overview.to_csv("dataset_summary.csv", index=False)
print("Saved high-level summary report to 'dataset_summary.csv'")