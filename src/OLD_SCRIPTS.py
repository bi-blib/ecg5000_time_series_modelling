#Mesh all samples, cap by total samples
"""
y_all_5000 = np.concatenate([train[:, 0], test[:, 0]]).astype(int)
X_all_5000 = np.concatenate([train[:, 1:], test[:, 1:]], axis=0)

y_all = y_all_5000[:TOTAL_SAMPLES]
X_all = X_all_5000[:TOTAL_SAMPLES, :]

# ECG5000 labels are 1-indexed (1..5); most PyTorch loss functions for
# classification expect 0-indexed class targets.
y_all = y_all - 1

# 70/15/15 train/val/test, stratified so each split keeps the same class
# ratios as the full dataset.
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
)
"""