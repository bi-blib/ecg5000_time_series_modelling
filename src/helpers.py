import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

#Statics
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 1
LR = 0.0001
EPOCHS = 100
TOTAL_SAMPLES = 2500 #/5000

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