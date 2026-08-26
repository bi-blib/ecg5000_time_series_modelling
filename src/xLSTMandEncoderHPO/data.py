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
    subtypes) rather than the collapsed healthy/unhealthy label. Splitting,
    stratification, and oversampling must all happen on this subtype label
    so rare abnormal subtypes get fair representation in every split and in
    the oversampled training set - call relabelForTask("binary", y) right
    before the label reaches the model/loss/metrics.
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

    Apply only where a task-facing label is needed (model input, loss,
    metrics) - never to the label array driving stratified splitting or the
    RandomOverSampler oversample target, which must stay in the raw 5-way
    subtype space for binary.
    """
    return toBinaryLabels(y) if task == "binary" else y


def binaryOversampleStrategy(y):
    """RandomOverSampler target for the binary task's raw subtype label.

    Brings each abnormal subtype up toward healthy_count / num_abnormal_
    subtypes, never down, so every rare subtype gets real representation
    once merged into "unhealthy" - without fully equalizing all 5 raw
    classes, which would leave "unhealthy" ~4x the size of "healthy" after
    merging and require re-weighting the loss to compensate.
    """
    classes, counts = np.unique(y, return_counts=True)
    counts_by_class = dict(zip(classes.tolist(), counts.tolist()))
    healthy_count = counts_by_class.get(0, 0)
    abnormal_classes = [c for c in counts_by_class if c != 0]
    target = healthy_count // len(abnormal_classes)
    return {c: max(counts_by_class[c], target) for c in abnormal_classes}


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
