import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load preprocessed data
file_path = "gesture_data_preprocessed.csv"
df = pd.read_csv(file_path)

# Convert features and labels
X = df.iloc[:, :-1].values  # All columns except the last one (features)
y = df.iloc[:, -1].values   # Last column (labels)

# Reshape X into CNN-compatible input (num_samples, 21, 3, 1)
X_reshaped = X.reshape(-1, 21, 3, 1).astype('float32')

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

# Save as NumPy files
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Data successfully split and saved as NumPy arrays:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
