"""Shared model, data, and evaluation helpers for the encoder package."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset


EPOCHS = 300
EARLY_STOP_NO_IMPROVEMENT = 10


class ClassificationDataset(Dataset):
    def __init__(self, signal: np.ndarray, labels: np.ndarray):
        self.signal = torch.as_tensor(signal, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, index):
        return self.signal[index], self.labels[index]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, inputs):
        return inputs + self.pe[:, : inputs.size(1)]


class BaseTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        dim_ff,
        n_classes,
        num_layers,
        n_heads,
        n_tokens,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=n_tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model * n_tokens, n_classes)

    def forward(self, inputs):
        embeddings = self.input_projection(inputs)
        encoded = self.encoder(self.positional_encoding(embeddings))
        return self.output_projection(encoded.flatten(start_dim=1))


def getCriterion(y_train, device):
    """Create a balanced cross-entropy loss from training labels only."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    return nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )


def loadRawDataFromFile():
    """Load ECG5000 data independently of the caller's working directory."""
    data_dir = Path(__file__).resolve().parents[2] / "ECG5000"
    train = np.loadtxt(data_dir / "ECG5000_TRAIN.txt")
    test = np.loadtxt(data_dir / "ECG5000_TEST.txt")
    return train, test


def plotLossCurve(train_losses, val_losses, save_path=None):
    best_epoch = int(np.argmin(val_losses))
    best_val_loss = val_losses[best_epoch]
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.axvline(
        best_epoch,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch {best_epoch + 1} (val loss={best_val_loss:.4f})",
    )
    plt.ylabel("log(Loss)", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def performTestingLoop(best_model, test_loader, device):
    best_model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            all_logits.append(best_model(x_batch.to(device)).cpu())
            all_targets.append(y_batch.cpu())
    all_logits = torch.cat(all_logits)
    y_true = torch.cat(all_targets).numpy()
    y_pred = all_logits.argmax(dim=1).numpy()
    return y_pred, y_true, all_logits


def performMetrics(y_true, y_pred, all_logits, class_labels, prefix, roc_suffix="roc"):
    """Print and plot test classification, ROC, and PR metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test Macro-averaged precision: {precision:.4f}")
    print(f"Test Macro-averaged recall: {recall:.4f}")
    print(f"Test Macro-averaged F1_score: {f1:.4f}")
    print(f"Test balanced accuracy: {balanced_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    matrix = confusion_matrix(y_true, y_pred)
    matrix_normalized = confusion_matrix(y_true, y_pred, normalize="true")
    print("Confusion matrix:")
    print(matrix)
    figure, (counts_axis, proportions_axis) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(matrix, display_labels=class_labels).plot(
        cmap="Blues", ax=counts_axis, colorbar=False
    )
    counts_axis.set_title("Counts")
    ConfusionMatrixDisplay(
        matrix_normalized, display_labels=class_labels
    ).plot(cmap="Blues", ax=proportions_axis, values_format=".1%")
    proportions_axis.set_title("Proportion of true class")
    figure.suptitle("Test set confusion matrix")
    plt.tight_layout()
    plt.savefig(f"{prefix}_cf.png", dpi=150, bbox_inches="tight")
    plt.show()

    scores = torch.softmax(all_logits, dim=1).numpy()
    is_binary = len(class_labels) == 2
    if is_binary:
        positive_scores = scores[:, 1]
        macro_auroc = roc_auc_score(y_true, positive_scores)
        macro_auprc = average_precision_score(y_true, positive_scores)
    else:
        binary_targets = label_binarize(y_true, classes=class_labels)
        macro_auroc = roc_auc_score(
            y_true, scores, average="macro", multi_class="ovr"
        )
        macro_auprc = average_precision_score(
            binary_targets, scores, average="macro"
        )
    print(f"\nMacro-average AUROC: {macro_auroc:.4f}")
    print(f"Macro-average AUPRC: {macro_auprc:.4f}")

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(12, 5))
    if is_binary:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, positive_scores
        )
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"Class {int(class_labels[1]) + 1} (AUROC={macro_auroc:.3f})",
        )
        curve_precision, curve_recall, _ = precision_recall_curve(
            y_true, positive_scores
        )
        pr_axis.plot(
            curve_recall,
            curve_precision,
            label=f"Class {int(class_labels[1]) + 1} (AUPRC={macro_auprc:.3f})",
        )
    else:
        for index, class_label in enumerate(class_labels):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_targets[:, index], scores[:, index]
            )
            class_auroc = roc_auc_score(binary_targets[:, index], scores[:, index])
            roc_axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"Class {int(class_label) + 1} (AUROC={class_auroc:.3f})",
            )
            curve_precision, curve_recall, _ = precision_recall_curve(
                binary_targets[:, index], scores[:, index]
            )
            class_auprc = average_precision_score(
                binary_targets[:, index], scores[:, index]
            )
            pr_axis.plot(
                curve_recall,
                curve_precision,
                label=f"Class {int(class_label) + 1} (AUPRC={class_auprc:.3f})",
            )

    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    roc_axis.set_title(f"ROC curves (macro AUROC={macro_auroc:.3f})")
    pr_axis.set(xlabel="Recall", ylabel="Precision")
    pr_axis.set_title(f"Precision-Recall curves (macro AUPRC={macro_auprc:.3f})")
    for axis in (roc_axis, pr_axis):
        axis.legend(fontsize=8)
        axis.grid(linestyle="dashed")
    plt.tight_layout()
    plt.savefig(f"{prefix}_{roc_suffix}.png", dpi=150, bbox_inches="tight")
    plt.show()
