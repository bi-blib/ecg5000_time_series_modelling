"""Leakage-free five-fold cross-validated model selection.

The validation risk used to choose lambda is the minimum validation
cross-entropy reached by early stopping.  The final reported performance is
balanced accuracy on a test set which is never used during model selection.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from src.encoder_no_HPO.helpers import BaseTransformerClassifier, ClassificationDataset


N_SPLITS = 5
RANDOM_STATE = 42

# ParameterGrid evaluates the Cartesian product.  All n_heads values must
# divide all d_model values.  Pass a JSON grid to the CLI to replace this.
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "d_model": [32, 64],
    "n_heads": [4],
    "num_layers": [1, 2],
    "dim_ff": [128],
    "lr": [1e-4, 3e-4],
    "batch_size": [32],
}


@dataclass(frozen=True)
class CandidateResult:
    """The five validation risks for one candidate lambda."""

    params: dict[str, Any]
    fold_scores: list[float]
    mean_score: float
    std_score: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_complete_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Combine the supplied ECG5000 train/test files into one data pool."""
    data_dir = Path(data_dir)
    train = np.loadtxt(data_dir / "ECG5000_TRAIN.txt")
    test = np.loadtxt(data_dir / "ECG5000_TEST.txt")
    complete = np.concatenate([train, test], axis=0)
    return complete[:, 1:].astype(np.float32), complete[:, 0].astype(np.int64) - 1


def prepare_task(
    X: np.ndarray, y: np.ndarray, task: str
) -> tuple[np.ndarray, np.ndarray]:
    """Map ECG5000 labels for the binary or abnormal-only task."""
    if task == "binary":
        return X, (y > 0).astype(np.int64)
    if task == "multiclass":
        abnormal = y > 0
        return X[abnormal], (y[abnormal] - 1).astype(np.int64)
    raise ValueError("task must be 'binary' or 'multiclass'")


def make_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create disjoint, stratified folds once for reuse by every candidate."""
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return list(splitter.split(X, y))


def _dataset_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = ClassificationDataset(
        signal=X.reshape(-1, X.shape[1], 1).astype(np.float32), labels=y
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_model(
    params: Mapping[str, Any], n_tokens: int, n_classes: int, device: torch.device
) -> BaseTransformerClassifier:
    if params["d_model"] % params["n_heads"] != 0:
        raise ValueError("d_model must be divisible by n_heads")
    return BaseTransformerClassifier(
        input_size=1,
        d_model=int(params["d_model"]),
        dim_ff=int(params["dim_ff"]),
        n_classes=n_classes,
        num_layers=int(params["num_layers"]),
        n_heads=int(params["n_heads"]),
        n_tokens=n_tokens,
    ).to(device)


def _criterion(y_train: np.ndarray, device: torch.device) -> nn.CrossEntropyLoss:
    classes = np.unique(y_train)
    expected = np.arange(classes.size)
    if not np.array_equal(classes, expected):
        raise ValueError(
            "training labels must contain every class and be contiguous from zero"
        )
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )


def _fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, float, list[float], list[float], int]:
    """Fit a model and restore the state from its lowest validation loss."""
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must both be positive")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y_batch.size(0)
        train_losses.append(running_loss / len(train_loader.dataset))

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                running_loss += (
                    criterion(model(X_batch), y_batch).item() * y_batch.size(0)
                )
        # Weight by observations, not batches.  This keeps the risk comparable
        # when batch_size itself is part of lambda.
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:  # Defensive: max_epochs is validated above.
        raise RuntimeError("training completed without producing a model state")
    model.load_state_dict(best_state)
    return model, best_loss, train_losses, val_losses, best_epoch + 1


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    candidates: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    max_epochs: int = 60,
    patience: int = 8,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Any], list[CandidateResult]]:
    """Evaluate all lambdas on the same five folds and return argmin mean risk."""
    folds = make_folds(X, y, random_state=random_state)
    candidate_list = [dict(candidate) for candidate in candidates]
    if not candidate_list:
        raise ValueError("at least one candidate configuration is required")

    results: list[CandidateResult] = []
    for candidate_number, params in enumerate(candidate_list, start=1):
        scores: list[float] = []
        print(f"\nCandidate {candidate_number}/{len(candidate_list)}: {params}")
        for fold_number, (train_index, val_index) in enumerate(folds, start=1):
            # A scaler is fit only to this fold's training observations.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_index]).astype(np.float32)
            X_val = scaler.transform(X[val_index]).astype(np.float32)
            y_train, y_val = y[train_index], y[val_index]

            set_seed(random_state + fold_number)
            train_loader = _dataset_loader(
                X_train, y_train, int(params["batch_size"]), shuffle=True
            )
            val_loader = _dataset_loader(
                X_val, y_val, int(params["batch_size"]), shuffle=False
            )
            model = _build_model(params, X.shape[1], len(np.unique(y)), device)
            criterion = _criterion(y_train, device)
            _, score, _, _, _ = _fit_with_early_stopping(
                model,
                train_loader,
                val_loader,
                criterion,
                device,
                lr=float(params["lr"]),
                max_epochs=max_epochs,
                patience=patience,
            )
            scores.append(score)
            print(f"  fold {fold_number}/{N_SPLITS}: validation loss={score:.6f}")

        result = CandidateResult(
            params=params,
            fold_scores=scores,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores, ddof=1)),
        )
        results.append(result)
        print(f"  mean validation loss={result.mean_score:.6f} ± {result.std_score:.6f}")

    best = min(results, key=lambda result: result.mean_score)
    return dict(best.params), results


def _predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits.append(model(X_batch.to(device)).cpu())
            targets.append(y_batch)
    return (
        torch.cat(logits).argmax(dim=1).numpy(),
        torch.cat(targets).numpy(),
    )


def _test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def run_model_selection(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    task: str = "binary",
    param_grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]] = DEFAULT_PARAM_GRID,
    cv_epochs: int = 60,
    cv_patience: int = 8,
    final_epochs: int = 300,
    final_patience: int = 10,
    random_state: int = RANDOM_STATE,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Select lambda*, refit it, and evaluate the untouched test set once."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(random_state)

    X, y = prepare_task(*load_complete_data(data_dir), task)

    # Reserve the test observations before CV.  Only the development portion
    # is passed to cross_validate, keeping the final test score unbiased.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )

    candidates = list(ParameterGrid(param_grid))
    best_params, cv_results = cross_validate(
        X_development,
        y_development,
        candidates,
        device=device,
        max_epochs=cv_epochs,
        patience=cv_patience,
        random_state=random_state,
    )
    print(f"\nSelected lambda*: {best_params}")

    # Create a fresh train/validation division after selection.  The 15/85
    # fraction makes validation 15% of the original complete data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.15 / 0.85,
        stratify=y_development,
        random_state=random_state + 1,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    set_seed(random_state)
    batch_size = int(best_params["batch_size"])
    train_loader = _dataset_loader(X_train_scaled, y_train, batch_size, True)
    val_loader = _dataset_loader(X_val_scaled, y_val, batch_size, False)
    test_loader = _dataset_loader(X_test_scaled, y_test, batch_size, False)
    final_model = _build_model(best_params, X.shape[1], len(np.unique(y)), device)
    final_model, best_val_loss, train_losses, val_losses, best_epoch = (
        _fit_with_early_stopping(
            final_model,
            train_loader,
            val_loader,
            _criterion(y_train, device),
            device,
            lr=float(best_params["lr"]),
            max_epochs=final_epochs,
            patience=final_patience,
        )
    )

    # This is the only point at which the test loader is consumed.
    y_pred, y_true = _predict(final_model, test_loader, device)
    test_metrics = _test_metrics(y_true, y_pred)
    report = {
        "task": task,
        "selection_metric": "mean 5-fold minimum validation cross-entropy",
        "lambda_star": best_params,
        "cv_results": [asdict(result) for result in cv_results],
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "final_best_epoch": best_epoch,
        "final_best_validation_loss": best_val_loss,
        "test_performance_indicator": "balanced_accuracy",
        "test_metrics": test_metrics,
    }

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
        "lambda_star": best_params,
        "task": task,
        "n_tokens": X.shape[1],
        "n_classes": len(np.unique(y)),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
    torch.save(checkpoint, output_dir / f"{task}_final_model.pth")
    (output_dir / f"{task}_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez(
        output_dir / f"{task}_losses.npz",
        train=np.asarray(train_losses),
        validation=np.asarray(val_losses),
    )

    print(
        "Final untouched-test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    return report
