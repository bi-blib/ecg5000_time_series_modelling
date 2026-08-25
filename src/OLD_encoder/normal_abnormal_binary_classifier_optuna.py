import numpy as np
import optuna
import torch
from sklearn.preprocessing import StandardScaler

from helpers import (
    EARLY_STOP_NO_IMPROVEMENT,
    EPOCHS,
    cleanAndSplitRaw,
    getCriterion,
    loadRawDataFromFile,
    performMetrics,
    performTestingLoop,
    plotLossCurve,
)
from helpers_optuna import (
    HPO_EARLY_STOP_NO_IMPROVEMENT,
    HPO_EPOCHS,
    N_TRIALS,
    buildModel,
    getLoadersOptuna,
    performTrainingLoopOptuna,
    suggestTransformerHyperparams,
)


def main():
    ### Load the data
    train, test = loadRawDataFromFile()
    X_train, y_train, X_val, y_val, X_test, y_test = cleanAndSplitRaw(train, test)

    y_train = np.array([1 if y > 0 else 0 for y in y_train])
    y_val = np.array([1 if y > 0 else 0 for y in y_val])
    y_test = np.array([1 if y > 0 else 0 for y in y_test])

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    input_size = 1
    output_size = len(np.unique(y_train))
    n_tokens = X_train.shape[1]

    ### Optuna hyperparameter search
    # Each trial suggests its own hyperparameters (including batch size, so
    # the DataLoaders are rebuilt per trial), trains for a reduced epoch
    # budget, and is scored on the held-out validation set only - the test
    # set stays untouched until the very end, after the winning config is
    # retrained with the full epoch budget.
    def objective(trial: optuna.trial.Trial) -> float:
        params = suggestTransformerHyperparams(trial)

        train_loader, val_loader, _ = getLoadersOptuna(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            batch_size=params["batch_size"],
        )

        model = buildModel(params, input_size, output_size, n_tokens, device)
        criterion = getCriterion(y_train=y_train, device=device)

        _, val_losses = performTrainingLoopOptuna(
            model, train_loader, val_loader, device, criterion,
            lr=params["lr"],
            epochs=HPO_EPOCHS,
            early_stop_patience=HPO_EARLY_STOP_NO_IMPROVEMENT,
            checkpoint_path=None,
            verbose=False,
        )

        return min(val_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest trial:")
    print(f"  Validation loss: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    best_params = study.best_params

    ### Retrain the winning config with the full training budget
    train_loader, val_loader, test_loader = getLoadersOptuna(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        batch_size=best_params["batch_size"],
    )

    model = buildModel(best_params, input_size, output_size, n_tokens, device)
    criterion = getCriterion(y_train=y_train, device=device)

    checkpoint_path = "b_optuna_best_model.pth"
    train_losses, val_losses = performTrainingLoopOptuna(
        model, train_loader, val_loader, device, criterion,
        lr=best_params["lr"],
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_NO_IMPROVEMENT,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    plotLossCurve(train_losses, val_losses, save_path="b_optuna_loss.png")

    best_model = buildModel(best_params, input_size, output_size, n_tokens, device)
    best_model.load_state_dict(torch.load(checkpoint_path))  # load best model (by val loss)

    y_pred, y_true, all_logits = performTestingLoop(best_model, test_loader, device)

    ### Metrics
    class_labels = np.unique(y_train)
    performMetrics(y_true, y_pred, all_logits, class_labels, prefix="b_optuna", roc_suffix="auc")


if __name__ == "__main__":
    main()
