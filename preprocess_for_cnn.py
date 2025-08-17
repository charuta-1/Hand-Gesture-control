import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and clean CSV
file_path = 'gesture_data.csv'
cleaned_lines = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 64:  # 63 features + 1 label
            cleaned_lines.append(','.join(parts))

# Step 2: Save cleaned data
cleaned_file = 'gesture_data_cleaned.csv'
with open(cleaned_file, 'w') as f:
    for line in cleaned_lines:
        f.write(line + '\n')

print(f"Original lines: {len(open(file_path).readlines())}")
print(f"Clean lines (64 columns): {len(cleaned_lines)}")

# Step 3: Load cleaned data into DataFrame
df = pd.read_csv(cleaned_file, header=None)
print(f"Data shape: {df.shape}")

# Step 4: Separate features and labels
num_features = df.shape[1] - 1
num_landmarks = num_features // 3
print(f"Feature columns: {num_features}")
print(f"Number of landmarks inferred: {num_landmarks}")

# Step 5: Drop rows with missing or malformed features
df_clean = df[df.iloc[:, :-1].apply(lambda row: len(row.dropna()) == num_features, axis=1)]
print(f"Total rows: {df.shape[0]}")
print(f"Valid rows with full features: {df_clean.shape[0]}")
print(f"Invalid rows with missing features: {df.shape[0] - df_clean.shape[0]}")

# Step 6: Normalize each row of landmarks
def normalize_row(row):
    features = np.array(row).astype(float)
    landmarks = features.reshape(num_landmarks, 3)
    wrist = landmarks[0]
    normalized = landmarks - wrist
    return normalized.flatten()

normalized_features = df_clean.iloc[:, :-1].apply(lambda row: normalize_row(row), axis=1, result_type='expand')

# Step 7: Encode string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df_clean.iloc[:, -1])  # e.g., 'FORWARD' → 0

# Step 8: Create final DataFrame
final_df = pd.concat([normalized_features, pd.Series(labels, name='label')], axis=1)
print(f"Final normalized data shape: {final_df.shape}")

# Step 9: Save final preprocessed CSV
final_file = 'gesture_data_preprocessed.csv'
final_df.to_csv(final_file, index=False, header=False)
print(f"Preprocessed data saved to {final_file}")

# Optional: Save label mapping
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping (string → int):")
for label, idx in label_map.items():
    print(f"{label} → {idx}")





