from collections import Counter

import numpy as np
import optuna
import torch
from sklearn.preprocessing import StandardScaler

from helpers import (
    EARLY_STOP_NO_IMPROVEMENT,
    EPOCHS,
    HPO_EARLY_STOP_NO_IMPROVEMENT,
    HPO_EPOCHS,
    MODEL_TYPE,
    N_TRIALS,
    NUM_THREADS,
    buildModel,
    cleanAndSplitRaw,
    getCriterion,
    getLoadersOptuna,
    loadRawDataFromFile,
    performMetrics,
    performTestingLoop,
    performTrainingLoopOptuna,
    plotLossCurve,
    resolveBestParams,
    suggestHyperparams,
)


def main():
    # ECG5000 epochs are small enough that spreading each matmul over every core
    # costs more in thread synchronisation than it saves.
    torch.set_num_threads(NUM_THREADS)

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
    print("Model type:", MODEL_TYPE)

    input_size = 1
    output_size = len(np.unique(y_train))
    n_tokens = X_train.shape[1]

    ### Optuna hyperparameter search
    # Each trial suggests its own architecture (including the block mix, so this
    # replaces the interactive mLSTM/sLSTM/xLSTM prompt) and its own batch size,
    # trains on a reduced epoch budget, and is scored on the validation set only.
    # The test set stays untouched until the winning config has been retrained.
    def objective(trial: optuna.trial.Trial) -> float:
        params = suggestHyperparams(trial)

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
            weight_decay=params["weight_decay"],
            epochs=HPO_EPOCHS,
            early_stop_patience=HPO_EARLY_STOP_NO_IMPROVEMENT,
            checkpoint_path=None,
            verbose=False,
            trial=trial,
        )

        return min(val_losses)

    # MedianPruner stops a trial once its val loss falls behind the median of
    # completed trials at the same epoch - the single biggest wall-clock saving
    # on CPU, since bad configs are usually obvious within ~15 epochs.
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nTrial states:", Counter(t.state.name for t in study.trials))
    print("Best trial:")
    print(f"  Validation loss: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # study.best_params only holds values that went through trial.suggest_*, so
    # the injected model_type and the fixed qkv_proj_blocksize have to be put
    # back before buildModel sees the dict.
    best_params = resolveBestParams(study)

    ### Retrain the winning config with the full training budget
    train_loader, val_loader, test_loader = getLoadersOptuna(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        batch_size=best_params["batch_size"],
    )

    model = buildModel(best_params, input_size, output_size, n_tokens, device)
    criterion = getCriterion(y_train=y_train, device=device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Best model has {num_params} trainable parameters.")

    # Artifact names follow MODEL_TYPE so switching architectures does not
    # overwrite the previous model's results under the wrong name.
    prefix = f"b_{MODEL_TYPE}_optuna"

    checkpoint_path = f"{prefix}_best_model.pth"
    train_losses, val_losses = performTrainingLoopOptuna(
        model, train_loader, val_loader, device, criterion,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_NO_IMPROVEMENT,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    plotLossCurve(train_losses, val_losses, save_path=f"{prefix}_loss.png")

    best_model = buildModel(best_params, input_size, output_size, n_tokens, device)
    best_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )  # load best model (by val loss)

    y_pred, y_true, all_logits = performTestingLoop(best_model, test_loader, device)

    ### Metrics
    class_labels = np.unique(y_train)
    performMetrics(
        y_true, y_pred, all_logits, class_labels,
        prefix=prefix, roc_suffix="auc",
    )


if __name__ == "__main__":
    main()
