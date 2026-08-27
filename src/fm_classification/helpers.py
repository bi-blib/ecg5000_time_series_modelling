"""Shared MantisV2 model, data, training, and reporting helpers for ECG5000.

This adapts ``sess15_classification.py`` to binary and abnormal-only multiclass
ECG tasks, early stopping, and complete metric reporting. The pretrained encoder
is frozen and its embeddings are cached once, making live head training practical
on CPU.

Oversampling strategy (applied to embeddings after extraction):
- Binary task:     no oversampling — class distribution is already reasonable.
- Multiclass task: +20 oversampling — each minority class is duplicated by 20
                   extra samples (capped at the majority class count), which was
                   found to give the best validation Macro F1 in a grid search.
"""

import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from mantis.architecture import MantisV2
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "ECG5000"
TRAIN_PATH = DATA_DIR / "ECG5000_TRAIN.txt"
TEST_PATH = DATA_DIR / "ECG5000_TEST.txt"

SEED = 42
TARGET_LENGTH = 512
BATCH_SIZE = 32
EMBEDDING_BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 300
EARLY_STOP_PATIENCE = 10

TASK_METADATA = {
    "binary": {
        "prefix": "b_mantis_frozen",
        "class_names": ["Normal", "Abnormal"],
    },
    "multiclass": {
        "prefix": "mc_mantis_frozen",
        "class_names": [
            "R-on-T PVC",
            "PVC",
            "Supraventricular/ectopic",
            "Unclassified",
        ],
    },
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SignalDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.signals[index], self.labels[index]


class FrozenMantisFeatureExtractor(nn.Module):
    """Frozen MantisV2 encoder using the example's final CLS representation."""

    def __init__(self, device: torch.device, target_length: int = TARGET_LENGTH):
        super().__init__()
        self.target_length = target_length
        encoder = MantisV2(
            device=str(device),
            return_transf_layer=-1,
            output_token="cls_token",
        )
        self.encoder = encoder.from_pretrained("paris-noah/MantisV2")
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # ECG loader shape: (batch, 140, 1). Mantis expects (batch, 1, length).
        signals = signals.transpose(1, 2)
        signals = F.interpolate(
            signals,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        )
        return self.encoder(signals)


def load_ecg5000_task(task: str):
    """Create a fixed, stratified train/validation/test partition."""
    train = np.loadtxt(TRAIN_PATH)
    test = np.loadtxt(TEST_PATH)

    X_train = train[:, 1:]
    y_train = (train[:, 0] - 1).astype(np.int64)
    X_temp = test[:, 1:]
    y_temp = (test[:, 0] - 1).astype(np.int64)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    if task == "binary":
        y_train = (y_train > 0).astype(np.int64)
        y_val = (y_val > 0).astype(np.int64)
        y_test = (y_test > 0).astype(np.int64)
    elif task == "multiclass":
        train_mask = y_train > 0
        val_mask = y_val > 0
        test_mask = y_test > 0
        X_train, y_train = X_train[train_mask], y_train[train_mask] - 1
        X_val, y_val = X_val[val_mask], y_val[val_mask] - 1
        X_test, y_test = X_test[test_mask], y_test[test_mask] - 1
    else:
        raise ValueError(f"Unknown task: {task}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_data(X_train, X_val, X_test):
    """Fit channel-wise scaling on train only, and scale all splits."""
    scaler = StandardScaler()
    train_shape, val_shape, test_shape = X_train.shape, X_val.shape, X_test.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(train_shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(val_shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(test_shape)
    return (
        X_train_scaled.astype(np.float32),
        X_val_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        scaler,
    )


# Number of extra samples added to each minority class for the multiclass task.
_MC_OVERSAMPLE_AMOUNT = 20


def apply_oversampling(
    embeddings: torch.Tensor, targets: torch.Tensor, task: str
) -> tuple:
    """Apply the fixed oversampling strategy to training embeddings.

    - Binary: no oversampling.
    - Multiclass: each minority class gets +20 extra samples
      (capped at the majority class size).
    """
    if task == "binary":
        return embeddings, targets

    y = targets.numpy()
    class_counts = np.bincount(y)
    majority_count = int(class_counts.max())
    majority_cls = int(class_counts.argmax())

    sampling_strategy = {
        cls: (count if cls == majority_cls else min(count + _MC_OVERSAMPLE_AMOUNT, majority_count))
        for cls, count in enumerate(class_counts)
        if count > 0
    }
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=SEED)
    X_res, y_res = ros.fit_resample(embeddings.numpy(), y)
    return torch.tensor(X_res, dtype=torch.float32), torch.tensor(y_res, dtype=torch.long)


def signal_loader(signals, labels, batch_size, shuffle=False):
    return DataLoader(
        SignalDataset(signals[..., np.newaxis], labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def extract_embeddings(extractor, loader, device, label):
    embeddings, labels = [], []
    extractor.eval()
    started = time.time()
    with torch.no_grad():
        for batch_number, (signals, batch_labels) in enumerate(loader, start=1):
            embeddings.append(extractor(signals.to(device)).cpu())
            labels.append(batch_labels)
            print(
                f"\r{label}: embedding batch {batch_number}/{len(loader)}",
                end="",
                flush=True,
            )
    print(f" ({time.time() - started:.1f}s)", flush=True)
    return torch.cat(embeddings), torch.cat(labels)


def train_head(
    head,
    train_loader,
    val_loader,
    device,
    checkpoint_path,
    epochs,
    patience,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    train_losses, val_losses, epoch_times = [], [], []
    best_val_loss = float("inf")
    best_val_epoch = 0

    for epoch in range(epochs):
        started = time.time()
        head.train()
        train_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(head(embeddings), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                val_loss += criterion(head(embeddings), labels).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(time.time() - started)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(head.state_dict(), checkpoint_path)

        print(
            f"Epoch: {epoch + 1:3d} | Train loss: {train_loss:.8f} | "
            f"Val loss: {val_loss:.8f} | Best Val loss: {best_val_loss:.8f} | "
            f"Best Val epoch: {best_val_epoch + 1}",
            flush=True,
        )
        if epoch - best_val_epoch >= patience:
            print(f"No validation improvement for {patience} epochs; stopping.", flush=True)
            break

    return train_losses, val_losses, epoch_times


def predict(head, loader, device):
    logits, labels = [], []
    head.eval()
    with torch.no_grad():
        for embeddings, batch_labels in loader:
            logits.append(head(embeddings.to(device)).cpu())
            labels.append(batch_labels)
    logits = torch.cat(logits)
    return logits.argmax(dim=1).numpy(), torch.cat(labels).numpy(), logits


def save_loss_plot(train_losses, val_losses, path):
    best_epoch = int(np.argmin(val_losses))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses, label="Validation loss")
    ax.axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch {best_epoch + 1}")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.set_yscale("log")
    ax.grid(linestyle="dashed", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plot(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Counts")
    ConfusionMatrixDisplay(cm_normalized, display_labels=class_names).plot(
        ax=axes[1], cmap="Blues", colorbar=False, values_format=".1%"
    )
    axes[1].set_title("Proportion of true class")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def save_roc_pr_plot(y_true, logits, class_names, path):
    scores = torch.softmax(logits, dim=1).numpy()
    n_classes = len(class_names)
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(13, 5))
    if n_classes == 2:
        positive_scores = scores[:, 1]
        fpr, tpr, _ = roc_curve(y_true, positive_scores)
        precision, recall, _ = precision_recall_curve(y_true, positive_scores)
        macro_auroc = roc_auc_score(y_true, positive_scores)
        macro_auprc = average_precision_score(y_true, positive_scores)
        roc_ax.plot(fpr, tpr, label=f"AUROC={macro_auroc:.3f}")
        pr_ax.plot(recall, precision, label=f"AUPRC={macro_auprc:.3f}")
    else:
        binary_targets = label_binarize(y_true, classes=np.arange(n_classes))
        macro_auroc = roc_auc_score(y_true, scores, average="macro", multi_class="ovr")
        macro_auprc = average_precision_score(binary_targets, scores, average="macro")
        for class_index, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(binary_targets[:, class_index], scores[:, class_index])
            precision, recall, _ = precision_recall_curve(
                binary_targets[:, class_index], scores[:, class_index]
            )
            roc_ax.plot(fpr, tpr, label=class_name)
            pr_ax.plot(recall, precision, label=class_name)

    roc_ax.plot([0, 1], [0, 1], "--", color="gray")
    roc_ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    roc_ax.legend(fontsize=8)
    pr_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(macro_auroc), float(macro_auprc)


def run_task(task, extractor, device, args):
    metadata = TASK_METADATA[task]
    prefix = metadata["prefix"]
    class_names = metadata["class_names"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_ecg5000_task(task)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(
        X_train, X_val, X_test
    )
    print(f"\nTask: {task}")
    print(f"Train/validation/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Original training class counts: {np.bincount(y_train)}", flush=True)

    # Extract embeddings from the original (un-oversampled) training signals.
    # Oversampling is applied afterwards on the cheap embedding vectors, so the
    # slow encoder forward pass runs only once.
    train_signals = signal_loader(
        X_train_scaled, y_train, args.embedding_batch_size, shuffle=False
    )
    val_signals = signal_loader(X_val_scaled, y_val, args.embedding_batch_size)
    test_signals = signal_loader(X_test_scaled, y_test, args.embedding_batch_size)

    train_embeddings, train_targets = extract_embeddings(
        extractor, train_signals, device, f"{task} train"
    )
    val_embeddings, val_targets = extract_embeddings(
        extractor, val_signals, device, f"{task} validation"
    )
    test_embeddings, test_targets = extract_embeddings(
        extractor, test_signals, device, f"{task} test"
    )

    # Apply the fixed oversampling strategy to the training embeddings.
    train_embeddings, train_targets = apply_oversampling(train_embeddings, train_targets, task)
    print(
        f"Training class counts after oversampling: {np.bincount(train_targets.numpy())}",
        flush=True,
    )

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(train_embeddings, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(val_embeddings, val_targets), batch_size=args.batch_size
    )
    test_loader = DataLoader(
        TensorDataset(test_embeddings, test_targets), batch_size=args.batch_size
    )

    head = nn.Linear(train_embeddings.shape[1], len(class_names)).to(device)
    checkpoint_path = output_dir / f"{prefix}_best_model.pth"
    train_losses, val_losses, epoch_times = train_head(
        head,
        train_loader,
        val_loader,
        device,
        checkpoint_path,
        args.epochs,
        args.patience,
    )
    save_loss_plot(train_losses, val_losses, output_dir / f"{prefix}_loss.png")

    head.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    y_pred, y_true, logits = predict(head, test_loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cm = save_confusion_plot(
        y_true, y_pred, class_names, output_dir / f"{prefix}_cf.png"
    )
    macro_auroc, macro_auprc = save_roc_pr_plot(
        y_true, logits, class_names, output_dir / f"{prefix}_roc_pr.png"
    )

    result = {
        "task": task,
        "model": "MantisV2FrozenClassifier",
        "pretrained_model": "paris-noah/MantisV2",
        "representation": "final_cls_token",
        "oversampling": "none" if task == "binary" else f"+{_MC_OVERSAMPLE_AMOUNT}",
        "split_sizes": {
            "train_original": int(len(y_train)),
            "train_after_oversampling": int(len(train_targets)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "best_epoch": int(np.argmin(val_losses) + 1),
        "best_validation_loss": float(np.min(val_losses)),
        "epochs_run": len(val_losses),
        "test_accuracy": float(accuracy),
        "test_macro_precision": float(precision),
        "test_macro_recall": float(recall),
        "test_macro_f1": float(macro_f1),
        "test_balanced_accuracy": float(balanced_accuracy),
        "test_macro_auroc": macro_auroc,
        "test_macro_auprc": macro_auprc,
        "mean_head_epoch_seconds": float(np.mean(epoch_times)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    }
    results_path = output_dir / f"{prefix}_results.json"
    results_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n{task} test results")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro precision:   {precision:.4f}")
    print(f"Macro recall:      {recall:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro AUROC:       {macro_auroc:.4f}")
    print(f"Macro AUPRC:       {macro_auprc:.4f}")
    print(f"Saved results:     {results_path}", flush=True)
    return result
