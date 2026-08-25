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
from sklearn.preprocessing import label_binarize, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import RandomOverSampler

#Statics
BATCH_SIZE = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 0.0001
EPOCHS = 300
TOTAL_SAMPLES = 2500 #/5000

### Dataset
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


### LSTM Model
# HINT: this architecture (LSTM -> last time step -> Linear) works fine for
# classification too - a many-to-one setup. The main thing to change is what
# output_size means: instead of "1 value to predict", it should be "1 score
# per class". Where output_size is set in main(), what should it equal for
# ECG5000?
class LSTMClassifier(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True) #batch_first because batch comes first in loader batch info
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1]
        output = self.linear(last_output)
        return output


def main():
    # HINT: ECG5000 already ships pre-split train/test files (loaded below),
    # so there's no single continuous signal to carve a train_fraction out
    # of - is this variable still needed for classification?
    #train_fraction = 0.80

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
    X_val, X_test, y_val, y_test = train_test_split( 
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("Train Signal Shape:", X_train.shape)
    print("Val Signal Shape:", X_val.shape)
    print("Test Signal Shape:", X_test.shape)
    print("Train Label Shape:", y_train.shape)
    print("Val Label Shape:", y_val.shape)
    print("Test Label Shape:", y_test.shape)
    #quit()

    ### Scale the data
    # HINT: StandardScaler expects 2D input shaped (n_samples, n_features).
    # Each row of X_train is one whole heartbeat here, not one time step -
    # think about what "a sample" and "a feature" mean for this scaler now.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    ros = RandomOverSampler(random_state=0)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)

    print(
        "Train class counts before oversampling:",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )
    print(
        "Train class counts after oversampling:",
        dict(zip(*np.unique(y_train_ros, return_counts=True))),
    )


    ### Create the dataset
    # HINT: pass X_train (and the new label argument you added to
    # ClassificationDataset above) instead of a single windowed signal. The
    # LSTM also expects input shaped (batch, seq_len, input_size) - X_train
    # is currently 2D (n_samples, seq_len), so it'll need an extra dimension
    # of size 1 added somewhere before it reaches the model.
    train_dataset = ClassificationDataset(
        signal=X_train_ros.reshape(-1, X_train_ros.shape[1], 1),
        labels=y_train_ros,
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
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    x_batch, y_batch = next(iter(train_loader))

    print("Input shape:", x_batch.shape) #batch size, n of time steps
    print("Target shape:", y_batch.shape) #batch size, n of output labels

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    ### Initialize model and optimizer
    # Define input and output size
    # HINT: input_size = features per time step = 1 (single-lead ECG, so
    # each time step is just one number). Hardcode this as 1, or better,
    # take it from your reshaped X_train's last dimension.
    #
    # HINT: output_size = number of classes. ECG5000 has 5 (labels 1-5), so
    # this should end up being 5 - but instead of hardcoding 5, derive it:
    #     output_size = len(np.unique(y_train))
    # That way the code stays correct even if the label set changes.
    #
    # Either way, `train_signal` and `test_signal` are leftover forecasting
    # variables (from the dead block you're replacing near the top of
    # main()) and won't exist once that's cleaned up - both lines below need
    # to reference your new X_train/y_train instead.
    input_size = 1
    output_size = len(np.unique(y_train))


    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size
    ).to(device)

    # Loss function
    # HINT: MSELoss compares continuous predictions to continuous targets -
    # not what you want when predicting one of several classes. What loss
    # function does PyTorch provide for multi-class classification, and what
    # dtype/shape does it expect the targets (y_batch) to be in?
    # RandomOverSampler has already balanced the training classes. Applying
    # balanced class weights as well would correct for the same imbalance
    # twice, so use ordinary cross-entropy here.
    criterion = nn.CrossEntropyLoss()

    # Optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    ### Training loop

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
        if epoch == best_val_epoch + 100:
            print("No reduction in validation loss in 100 epochs. Stop training early")
            break

    # Plot loss curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.ylabel("log(Loss)", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.legend()
    plt.show()


    best_model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size
    ).to(device)
    best_model.load_state_dict(torch.load("best_model.pth")) #load best model (by val loss)

    # Scale X_test with the scaler fit on X_train (never re-fit on test
    # data), then reshape to (n_samples, seq_len, 1) same as the train set.
    X_test_tensor = torch.tensor(
        X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1),
        device=device,
    )

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

    ### Metrics
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
    y_true_bin = label_binarize(y_true, classes=class_labels)

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


if __name__ == "__main__":
    main()
