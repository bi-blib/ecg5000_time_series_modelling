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
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

#Statics
BATCH_SIZE = 32
D_MODEL = 64
DIM_FF = 128
N_HEADS = 4
NUM_LAYERS = 1
LR = 0.0001
EPOCHS = 300
#TOTAL_SAMPLES = 2500 #/5000, not currently used

class ClassificationDataset(Dataset):
    def __init__(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
    ):
        # XHINT: ECG5000 already gives you one full heartbeat per row plus a
        # separate label array (y_train/y_test from main()). What extra
        # argument does this __init__ need so __getitem__ can return a label?
        self.signal = torch.tensor(signal)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Number of samples (rows), not the length of one signal.
        return len(self.signal)

    def __getitem__(self, idx):
        # XHINT: x should be the whole signal for sample idx (one heartbeat),
        # and y should be that sample's class label - not "the next value".
        x = self.signal[idx]
        y = self.labels[idx]

        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


### Transformer Encoder Model
class BaseTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        dim_ff: int,
        n_classes: int,
        num_layers: int,
        n_heads: int,
        n_tokens: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=n_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model * n_tokens, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_embed = self.input_projection(x)
        x_embed_pe = self.positional_encoding(x_embed)

        z = self.encoder(x_embed_pe)

        # Flatten the output of the encoder
        z_flat = z.flatten(start_dim=1)

        logits = self.output_projection(z_flat)

        return logits

def getCriterion(y_train, device):
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(class_weights)
    return nn.CrossEntropyLoss(weight=class_weights_tensor)

def getLoaders(X_train_scaled, X_val_scaled, X_test_scaled,
               y_train, y_val, y_test):
    
    train_dataset = ClassificationDataset(
        signal=X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_train,
    )

    val_dataset = ClassificationDataset(
        signal=X_val_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_val,
    )

    test_dataset = ClassificationDataset(
        signal=X_test_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def performTrainingLoop(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
) -> None:

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(x_batch)

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad(): #don't track gradients for optimizer, we don't want to learn validation
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)

                loss = criterion(
                    predictions,
                    y_batch,
                )

                #No backward pass = no learning
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            #save best validation loss use later; protects against overfitting
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                torch.save(model.state_dict(), "best_model.pth")


            print(
                f"Epoch: {epoch + 1:3d} | "
                f"Train loss: {train_loss:.8f}"
                f" | "
                f"| Val loss: {val_loss:.8f}"
                f" | "
                f"Best Val loss: {best_val_loss:.8f}"
                f" | "
                f"Best Val epoch: {best_val_epoch + 1}"
            )

        if epoch == best_val_epoch + 100: #early stopping if we haven't had a best val epoch in 100 epochs
            print("No reduction in validation loss in 100 epochs. Stopping training early.")
            break

    return train_losses, val_losses

def plotLossCurve(train_losses, val_losses):
    # Plot loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.ylabel("log(Loss)", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.legend()
    plt.show()

def performTestingLoop(best_model, test_loader, device):
    best_model.eval()
    all_logits = []
    all_targets = []

    #testing loop
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            logits = best_model(x_batch)

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    y_pred = all_logits.argmax(dim=1).numpy()
    y_true = all_targets.numpy()

    return y_pred, y_true, all_logits

def loadRawDataFromFile():
    train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
    test  = np.loadtxt("ECG5000/ECG5000_TEST.txt")
    return train, test

def cleanAndSplitRaw(train, test):
    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_temp = test[:, 1:]
    y_temp = test[:, 0]
    y_train = y_train - 1
    y_temp = y_temp - 1

    X_val, X_test, y_val, y_test = train_test_split( 
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test

def performMetrics(y_true, y_pred, all_logits, class_labels):
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
    n_classes = len(class_labels)
    # label_binarize collapses to a single column for exactly 2 classes, which
    # breaks the one-vs-rest machinery below, so binary needs its own path.
    is_binary = n_classes == 2

    if is_binary:
        pos_score = y_score[:, 1]
        macro_auroc = roc_auc_score(y_true, pos_score)
        macro_auprc = average_precision_score(y_true, pos_score)
    else:
        y_true_bin = label_binarize(y_true, classes=class_labels)
        macro_auroc = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
        macro_auprc = average_precision_score(y_true_bin, y_score, average="macro")

    print(f"\nMacro-average AUROC: {macro_auroc:.4f}")
    print(f"Macro-average AUPRC: {macro_auprc:.4f}")

    # Plot one-vs-rest ROC and Precision-Recall curves, one line per class.
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, pos_score)
        ax_roc.plot(fpr, tpr, label=f"Class {int(class_labels[1]) + 1} (AUROC={macro_auroc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, pos_score)
        ax_pr.plot(recall, precision, label=f"Class {int(class_labels[1]) + 1} (AUPRC={macro_auprc:.3f})")
    else:
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