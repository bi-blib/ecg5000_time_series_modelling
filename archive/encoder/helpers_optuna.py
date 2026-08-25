"""Five-fold cross-validated Optuna model selection for encoder classifiers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import optuna
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    from .helpers import (
        BaseTransformerClassifier,
        ClassificationDataset,
        EARLY_STOP_NO_IMPROVEMENT,
        EPOCHS,
        getCriterion,
        loadRawDataFromFile,
        performMetrics,
        performTestingLoop,
        plotLossCurve,
    )
except ImportError:
    from ecg5000_time_series_modelling.archive.encoder.helpers import (
        BaseTransformerClassifier,
        ClassificationDataset,
        EARLY_STOP_NO_IMPROVEMENT,
        EPOCHS,
        getCriterion,
        loadRawDataFromFile,
        performMetrics,
        performTestingLoop,
        plotLossCurve,
    )


N_TRIALS = 25
N_SPLITS = 5
HPO_EPOCHS = 60
HPO_EARLY_STOP_NO_IMPROVEMENT = 8
RANDOM_STATE = 42


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suggestTransformerHyperparams(trial):
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_ff": trial.suggest_categorical("dim_ff", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def buildModel(params, input_size, n_classes, n_tokens, device):
    if params["d_model"] % params["n_heads"] != 0:
        raise ValueError("d_model must be divisible by n_heads")
    return BaseTransformerClassifier(
        input_size=input_size,
        d_model=params["d_model"],
        dim_ff=params["dim_ff"],
        n_classes=n_classes,
        num_layers=params["num_layers"],
        n_heads=params["n_heads"],
        n_tokens=n_tokens,
    ).to(device)


def _loader(X, y, batch_size, shuffle):
    dataset = ClassificationDataset(
        X.reshape(-1, X.shape[1], 1).astype(np.float32), y
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def getLoadersOptuna(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    y_train,
    y_val,
    y_test,
    batch_size,
):
    return (
        _loader(X_train_scaled, y_train, batch_size, True),
        _loader(X_val_scaled, y_val, batch_size, False),
        _loader(X_test_scaled, y_test, batch_size, False),
    )


def performTrainingLoopOptuna(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    lr,
    epochs,
    early_stop_patience,
    checkpoint_path=None,
    verbose=True,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_val_epoch = -1

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item() * y_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)

        if verbose:
            print(
                f"Epoch: {epoch + 1:3d} | Train loss: {train_loss:.8f} | "
                f"Val loss: {val_loss:.8f} | Best Val loss: "
                f"{best_val_loss:.8f} | Best Val epoch: {best_val_epoch + 1}"
            )
        if epoch - best_val_epoch >= early_stop_patience:
            if verbose:
                print(
                    f"No validation improvement for {early_stop_patience} "
                    "epochs. Stopping early."
                )
            break
    return train_losses, val_losses


def loadTaskData(task):
    """Load complete ECG5000 data and map labels for the selected task."""
    train, test = loadRawDataFromFile()
    complete = np.concatenate([train, test])
    X = complete[:, 1:].astype(np.float32)
    y = complete[:, 0].astype(np.int64) - 1
    if task == "binary":
        return X, (y > 0).astype(np.int64)
    if task == "multiclass":
        abnormal = y > 0
        return X[abnormal], y[abnormal] - 1
    raise ValueError("task must be 'binary' or 'multiclass'")


def makeFolds(X, y):
    """Create the five disjoint folds reused by every candidate lambda."""
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    return list(splitter.split(X, y))


def crossValidatedScore(params, X_development, y_development, folds, device):
    """Return mean five-fold validation loss and all fold-level scores."""
    fold_scores = []
    n_classes = len(np.unique(y_development))
    for fold_number, (train_index, val_index) in enumerate(folds, start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_development[train_index]).astype(np.float32)
        X_val = scaler.transform(X_development[val_index]).astype(np.float32)
        y_train, y_val = y_development[train_index], y_development[val_index]

        setSeed(RANDOM_STATE + fold_number)
        train_loader = _loader(X_train, y_train, params["batch_size"], True)
        val_loader = _loader(X_val, y_val, params["batch_size"], False)
        model = buildModel(
            params, 1, n_classes, X_development.shape[1], device
        )
        _, val_losses = performTrainingLoopOptuna(
            model,
            train_loader,
            val_loader,
            device,
            getCriterion(y_train, device),
            params["lr"],
            HPO_EPOCHS,
            HPO_EARLY_STOP_NO_IMPROVEMENT,
            verbose=False,
        )
        score = float(min(val_losses))
        fold_scores.append(score)
        print(f"  fold {fold_number}/{N_SPLITS}: validation loss={score:.6f}")
    return float(np.mean(fold_scores)), fold_scores


def runCrossValidatedModelSelection(task, output_prefix, roc_suffix):
    """Select lambda*, refit on a fresh split, then test exactly once."""
    setSeed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    X, y = loadTaskData(task)

    # The test set is reserved before CV and never passed to an objective.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    folds = makeFolds(X_development, y_development)

    def objective(trial):
        params = suggestTransformerHyperparams(trial)
        print(f"\nTrial {trial.number + 1}/{N_TRIALS}: {params}")
        mean_score, fold_scores = crossValidatedScore(
            params, X_development, y_development, folds, device
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_std", float(np.std(fold_scores, ddof=1)))
        print(
            f"  mean validation loss={mean_score:.6f} ± "
            f"{trial.user_attrs['fold_std']:.6f}"
        )
        return mean_score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    print("\nSelected lambda*:", best_params)
    print(f"Mean five-fold validation loss: {study.best_value:.6f}")

    # Fresh train/validation division gives 70/15/15 overall proportions.
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.15 / 0.85,
        stratify=y_development,
        random_state=RANDOM_STATE + 1,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    train_loader, val_loader, test_loader = getLoadersOptuna(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        best_params["batch_size"],
    )

    checkpoint_path = f"{output_prefix}_best_model.pth"
    final_model = buildModel(best_params, 1, len(np.unique(y)), X.shape[1], device)
    train_losses, val_losses = performTrainingLoopOptuna(
        final_model,
        train_loader,
        val_loader,
        device,
        getCriterion(y_train, device),
        best_params["lr"],
        EPOCHS,
        EARLY_STOP_NO_IMPROVEMENT,
        checkpoint_path=checkpoint_path,
    )
    plotLossCurve(train_losses, val_losses, f"{output_prefix}_loss.png")

    best_model = buildModel(best_params, 1, len(np.unique(y)), X.shape[1], device)
    best_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    # The untouched test loader is consumed only here.
    y_pred, y_true, logits = performTestingLoop(best_model, test_loader, device)
    test_score = float(balanced_accuracy_score(y_true, y_pred))
    performMetrics(
        y_true, y_pred, logits, np.unique(y), output_prefix, roc_suffix
    )

    report = {
        "task": task,
        "lambda_star": best_params,
        "mean_cv_validation_loss": study.best_value,
        "best_fold_scores": study.best_trial.user_attrs["fold_scores"],
        "test_performance_indicator": "balanced_accuracy",
        "test_balanced_accuracy": test_score,
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
    }
    Path(f"{output_prefix}_cv_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
