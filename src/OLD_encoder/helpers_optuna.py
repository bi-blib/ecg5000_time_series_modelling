import torch
from torch.utils.data import DataLoader

from helpers import BaseTransformerClassifier, ClassificationDataset

# Optuna HPO settings. Trials use a smaller epoch budget than the final
# training run (helpers.EPOCHS / helpers.EARLY_STOP_NO_IMPROVEMENT) so a
# full search finishes in reasonable time; the winning config is then
# retrained with the full budget.
N_TRIALS = 25
HPO_EPOCHS = 60
HPO_EARLY_STOP_NO_IMPROVEMENT = 8


def suggestTransformerHyperparams(trial):
    """Search space for BaseTransformerClassifier + training hyperparameters.

    n_heads is drawn from values that evenly divide every d_model choice
    below (32/64/128 are all divisible by 2 and 4), so every sampled
    combination is a valid nn.MultiheadAttention config - no need to
    couple the two suggest_* calls together.
    """
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_ff": trial.suggest_categorical("dim_ff", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def buildModel(params, input_size, n_classes, n_tokens, device):
    return BaseTransformerClassifier(
        input_size=input_size,
        d_model=params["d_model"],
        dim_ff=params["dim_ff"],
        n_classes=n_classes,
        num_layers=params["num_layers"],
        n_heads=params["n_heads"],
        n_tokens=n_tokens,
    ).to(device)


def getLoadersOptuna(X_train_scaled, X_val_scaled, X_test_scaled,
                      y_train, y_val, y_test, batch_size):
    """Same as helpers.getLoaders, but batch_size is an argument instead of
    the fixed BATCH_SIZE constant, so Optuna can tune it per trial."""
    train_dataset = ClassificationDataset(
        signal=X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_train,
    )
    val_dataset = ClassificationDataset(
        signal=X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1),
        labels=y_val,
    )
    test_dataset = ClassificationDataset(
        signal=X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1),
        labels=y_test,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def performTrainingLoopOptuna(model, train_loader, val_loader, device, criterion,
                               lr, epochs, early_stop_patience,
                               checkpoint_path=None, verbose=True):
    """Same shape as helpers.performTrainingLoop, but lr/epochs/early-stop
    patience/checkpoint path are all arguments instead of module constants,
    so Optuna trials can vary them and skip checkpointing entirely
    (checkpoint_path=None) - trials only need the val loss, not the weights.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)

            if verbose:
                print(
                    f"Epoch: {epoch + 1:3d} | "
                    f"Train loss: {train_loss:.8f} | "
                    f"Val loss: {val_loss:.8f} | "
                    f"Best Val loss: {best_val_loss:.8f} | "
                    f"Best Val epoch: {best_val_epoch + 1}"
                )

        if epoch == best_val_epoch + early_stop_patience:
            if verbose:
                print(
                    f"No reduction in validation loss in {early_stop_patience} "
                    "epochs. Stopping training early."
                )
            break

    return train_losses, val_losses
