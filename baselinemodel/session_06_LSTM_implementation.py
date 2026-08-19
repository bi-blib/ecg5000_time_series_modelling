import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


### Dataset
class ClassificationDataset(Dataset):
    def __init__(
        self,
        signal: np.ndarray,
        context_length: int,
    ):
        self.signal = torch.tensor(signal)

        self.context_length = context_length

    def __len__(self):
        # Defines how many samples exist
        return len(self.signal) - self.context_length

    def __getitem__(self, idx):
        #TODO: Change to get a specific sample
        # Previous context_length values
        x = self.signal[idx : idx + self.context_length]  # 0 ... 100

        # Immediately following value
        y = self.signal[idx + self.context_length]  # 101

        return x, y


### LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x):
        # x:
        # (batch_size, context_length, 1)

        lstm_output, _ = self.lstm(x)

        # LSTM output at the final context time step
        last_output = lstm_output[:, -1, :]

        # Predict the next signal value
        prediction = self.fc(last_output)

        return prediction


def main():
    context_length = 100
    train_fraction = 0.80

    ### Load the data
    data = np.load(
        "teaching_material/datasets/DampedSineSignal/damped_sine_signal.npy"
    )
    print("Signal shape:", data.shape)
    time = data[:, 0]
    signal = data[:, 1:2]

    n_time_steps = len(signal)
    train_end = int(train_fraction * n_time_steps)

    ### Split into train and test sets
    train_signal = signal[:train_end]
    test_signal = signal[train_end:]

    print("Training points:", len(train_signal))
    print("Test points:", len(test_signal))

    ### Scale the data
    scaler = StandardScaler()

    scaler.fit(train_signal)  # Fit only on the train signal

    signal_scaled = scaler.transform(signal).astype(np.float32)
    train_signal_scaled = signal_scaled[:train_end]

    ### Create the dataset
    train_dataset = ForecastingDataset(
        signal=train_signal_scaled,
        context_length=context_length,
    )

    bs = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
    )

    x_batch, y_batch = next(iter(train_loader))

    print("Input shape:", x_batch.shape)
    print("Target shape:", y_batch.shape)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    ### Initialize model and optimizer
    # Define input and output size
    input_size = train_signal.shape[-1]
    output_size = test_signal.shape[-1]

    hidden_size = 64
    num_layers = 1
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=0.0,
    ).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    lr = 0.0001
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    ### Training loop
    epochs = 500
    train_losses = []

    for epoch in range(epochs):
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
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.grid(True)
    plt.show()

    model.eval()

    # The initial context contains only training values
    window = torch.tensor(
        signal_scaled[train_end - context_length : train_end],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # Shape:
    # (1, context_length, 1)
    print("Initial test context:", window.shape)

    test_predictions_scaled = []

    with torch.no_grad():
        for _ in range(len(test_signal)):
            # Predict the next value
            next_value = model(window)

            test_predictions_scaled.append(next_value.cpu().numpy())

            # Shape:
            # (1, 1) -> (1, 1, 1)
            next_value = next_value.unsqueeze(1)

            # Remove oldest value and append prediction
            window = torch.cat(
                [
                    window[:, 1:, :],
                    next_value,
                ],
                dim=1,
            )

    test_predictions_scaled = np.concatenate(
        test_predictions_scaled,
        axis=0,
    )

    # Convert prediction back to original scale
    test_predictions = scaler.inverse_transform(test_predictions_scaled)

    test_target = test_signal

    ### Metrics
    test_mse = np.mean((test_predictions - test_target) ** 2)
    r2 = r2_score(test_target, test_predictions)

    print(f"\nTest MSE: {test_mse:.8f}")
    print(f"Test r2: {r2:.8f}")

    ### Plot the results
    plt.figure(figsize=(12, 5))

    # Training part of the true signal
    plt.plot(
        time[:train_end],
        signal[:train_end, 0],
        linewidth=2,
        label="Training signal",
        color="black",
    )

    # Test part of the true signal
    plt.plot(
        time[train_end - 1 :],
        signal[train_end - 1 :, 0],
        linewidth=2,
        label="Test ground truth",
        color="tab:blue",
    )

    # Predicted complete test region
    plt.plot(
        time[train_end:],
        test_predictions,
        linewidth=2,
        label="Test prediction",
        color="tab:orange",
    )

    # Train-test boundary
    plt.axvline(
        time[train_end],
        linestyle=":",
        label="Train-test split",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
