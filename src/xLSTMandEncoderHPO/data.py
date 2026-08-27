import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from config import N_SPLITS, RANDOM_STATE


class ClassificationDataset(Dataset):
    def __init__(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
    ):
        self.signal = torch.tensor(signal)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Number of samples (rows), not the length of one signal.
        return len(self.signal)

    def __getitem__(self, idx):
        x = self.signal[idx]
        y = self.labels[idx]

        return x, y


def loadRawDataFromFile():
    train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
    test  = np.loadtxt("ECG5000/ECG5000_TEST.txt")
    return train, test


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loadTaskData(task):
    """Load complete ECG5000 data and labels for the selected task.

    For "binary", y is the raw 5-way subtype label (0=normal, 1-4=abnormal
    subtypes) rather than the collapsed healthy/unhealthy label. Splitting
    and stratification happen on this subtype label so rare abnormal
    subtypes get fair representation in every split. Binary training is not
    oversampled; call relabelForTask("binary", y) before the label reaches
    the model/loss/metrics.
    """
    train, test = loadRawDataFromFile()
    complete = np.concatenate([train, test])
    X = complete[:, 1:].astype(np.float32)
    y = complete[:, 0].astype(np.int64) - 1
    if task == "binary":
        return X, y
    if task == "multiclass":
        abnormal = y > 0
        return X[abnormal], y[abnormal] - 1
    raise ValueError("task must be 'binary' or 'multiclass'")


def toBinaryLabels(y):
    """Collapse ECG5000's 5-way subtype label to healthy(0)/unhealthy(1).

    Only call this after stratified splitting and after oversampling - both
    need the true subtype label so rare abnormal subtypes aren't left to
    chance in a split, or drowned out by whichever subtype happens to
    dominate the pooled "unhealthy" bucket during oversampling.
    """
    return (y > 0).astype(np.int64)


def relabelForTask(task, y):
    """No-op for multiclass; collapses to healthy/unhealthy for binary.

    Apply only where a task-facing label is needed (model input, loss, or
    metrics), never before stratified splitting. Multiclass oversampling uses
    the multiclass labels; binary has no oversampling.
    """
    return toBinaryLabels(y) if task == "binary" else y


def multiclassOversampleStrategy(y, amount=20):
    """Add a fixed number of samples to each minority class, with a cap."""
    class_counts = np.bincount(np.asarray(y, dtype=np.int64))
    majority_count = int(class_counts.max())
    majority_class = int(class_counts.argmax())
    return {
        class_id: (
            count
            if class_id == majority_class
            else min(count + amount, majority_count)
        )
        for class_id, count in enumerate(class_counts)
        if count > 0
    }


def _loader(X, y, batch_size, shuffle):
    dataset = ClassificationDataset(
        signal=X.reshape(-1, X.shape[1], 1).astype(np.float32), labels=y
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def makeFolds(X, y):
    """Create the five disjoint folds reused by every Optuna trial."""
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    return list(splitter.split(X, y))
