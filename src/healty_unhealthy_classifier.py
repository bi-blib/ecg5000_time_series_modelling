import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

from helpers import *

### Load the data
# ECG5000 ships pre-split train/test files, but we want our own 70/15/15
# train/validation/test split with class ratios preserved, so combine
# both files into one pool and re-split below.
train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
test  = np.loadtxt("ECG5000/ECG5000_TEST.txt")

X_train = train[:, 1:]
y_train = train[:, 0]
X_temp = test[:, 1:]
y_temp = test[:, 0]
y_train = y_train - 1
y_temp = y_temp - 1


y_train = np.array([1 if y > 0 else 0 for y in y_train])
y_temp = np.array([1 if y > 0 else 0 for y in y_temp])

X_val, X_test, y_val, y_test = train_test_split( 
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("Train Signal Shape:", X_train.shape)
print("Val Signal Shape:", X_val.shape)
print("Test Signal Shape:", X_test.shape)
print("Train Label Shape:", y_train.shape)
print("Val Label Shape:", y_val.shape)
print("Test Label Shape:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

### Create the dataset
train_loader, val_loader, test_loader = getLoaders(
    X_train_scaled=X_train_scaled,
    X_val_scaled=X_val_scaled,
    X_test_scaled=X_test_scaled,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)

x_batch, y_batch = next(iter(train_loader))

print("Input shape:", x_batch.shape) #batch size, n of time steps
print("Target shape:", y_batch.shape) #batch size, n of output labels

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

### Initialize model and optimizer
input_size = 1
output_size = len(np.unique(y_train))

model = LSTMClassifier(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=output_size
).to(device)

# Loss function
criterion = getCriterion(y_train=y_train, device=device)

### Training loop
train_losses, val_losses = performTrainingLoop(model, train_loader, val_loader, device, criterion)
plotLossCurve(train_losses, val_losses)

best_model = LSTMClassifier(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=output_size
).to(device)
best_model.load_state_dict(torch.load("best_model.pth")) #load best model (by val loss)

y_pred, y_true, all_logits = performTestingLoop(best_model, test_loader, device)

### Metrics
# Use y_true (returned by performTestingLoop, aligned to y_pred's order)
# rather than y_test - test_loader order doesn't match y_test's order.
test_accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest accuracy: {test_accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(cm)

### Plot the results
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title("Test set confusion matrix")
plt.tight_layout()
plt.show()

### AUROC / AUPRC
# Class scores (probabilities) are needed for threshold-based metrics
# like AUROC/AUPRC - argmax predictions (y_pred) collapse that info away.
y_score = torch.softmax(all_logits, dim=1).numpy()
class_labels = np.unique(y_train)
# label_binarize collapses to a single column when there are exactly 2
# classes, which doesn't match the 2-column softmax output below, so
# one-hot encode explicitly instead.
y_true_bin = np.eye(len(class_labels))[y_true.astype(int)]

macro_auroc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr")
macro_auprc = average_precision_score(y_true_bin, y_score, average="macro")

print(f"\nMacro-average AUROC: {macro_auroc:.4f}")
print(f"Macro-average AUPRC: {macro_auprc:.4f}")

# Plot one-vs-rest ROC and Precision-Recall curves, one line per class.
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

for i, class_label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    class_auroc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    ax_roc.plot(fpr, tpr, label=f"Class {int(class_label) + 1} (AUROC={class_auroc:.3f})")

    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    class_auprc = average_precision_score(y_true_bin[:, i], y_score[:, i])
    ax_pr.plot(recall, precision, label=f"Class {int(class_label) + 1} (AUPRC={class_auprc:.3f})")

ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title(f"ROC curves (macro AUROC={macro_auroc:.3f})")
ax_roc.legend(fontsize=8)
ax_roc.grid(linestyle="dashed")

ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"Precision-Recall curves (macro AUPRC={macro_auprc:.3f})")
ax_pr.legend(fontsize=8)
ax_pr.grid(linestyle="dashed")

plt.tight_layout()
plt.show()
