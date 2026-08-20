import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

#Statics
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 0.0001
EPOCHS = 300

### Dataset
class ClassificationDataset(Dataset):
    def __init__(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
        context_length: int,
    ):
        self.signal = torch.tensor(signal) #signal is a tensor of the ECG record
        self.labels = labels
        self.context_length = context_length

    def __len__(self):
        # Number of samples (rows), not the length of one signal.
        return len(self.signal)

    def __getitem__(self, idx):
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
    context_length = 100

    # HINT: ECG5000 already ships pre-split train/test files (loaded below),
    # so there's no single continuous signal to carve a train_fraction out
    # of - is this variable still needed for classification?
    #train_fraction = 0.80

    ### Load the data
    train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
    test  = np.loadtxt("ECG5000/ECG5000_TEST.txt")

    y_train, X_train = train[:, 0].astype(int), train[:, 1:]
    y_test,  X_test  = test[:, 0].astype(int),  test[:, 1:]

    print("Train Signal Shape:", X_train.shape)
    print("Test Signal Shape:", X_test.shape)
    print("Train Label Shape:", y_train.shape)
    print("Test Label Shape:", y_test.shape)

    # HINT: remove this quit() once you start wiring up the block below to
    # use X_train/y_train/X_test/y_test instead of the old single `data`
    # array. Also - ECG5000 labels are 1-indexed (1..5); most PyTorch loss
    # functions for classification expect 0-indexed class targets.
    y_train = y_train - 1
    y_test = y_test - 1
    #print(y_train.min())

    ### Scale the data
    # HINT: StandardScaler expects 2D input shaped (n_samples, n_features).
    # Each row of X_train is one whole heartbeat here, not one time step -
    # think about what "a sample" and "a feature" mean for this scaler now.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)

    ### Create the dataset
    # HINT: pass X_train (and the new label argument you added to
    # ClassificationDataset above) instead of a single windowed signal. The
    # LSTM also expects input shaped (batch, seq_len, input_size) - X_train
    # is currently 2D (n_samples, seq_len), so it'll need an extra dimension
    # of size 1 added somewhere before it reaches the model.
    train_dataset = ClassificationDataset(
        signal=X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_train,
        context_length=context_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
    criterion = nn.CrossEntropyLoss()

    # Optimizer

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    ### Training loop

    train_losses = []

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

        print(f"Epoch: {epoch + 1:3d} | " f"Train loss: {train_loss:.8f}")

    # Plot loss curve
    plt.plot(train_losses)
    plt.ylabel("log(loss)")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    model.eval()

    # Scale X_test with the scaler fit on X_train (never re-fit on test
    # data), then reshape to (n_samples, seq_len, 1) same as the train set.
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    X_test_tensor = torch.tensor(
        X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1),
        device=device,
    )

    with torch.no_grad():
        test_logits = model(X_test_tensor)

    # Highest-scoring class per sample = the model's predicted label
    y_pred = test_logits.argmax(dim=1).cpu().numpy()

    ### Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    ### Plot the results
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Test set confusion matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
