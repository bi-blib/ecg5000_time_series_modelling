"""Compare training-only augmentation strategies for the binary mLSTM model."""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import helpers


SEED = 42
MODEL_TYPE = "mLSTM"
RESULTS_DIR = Path("archive/xLSTM/augmentation_results")
SCENARIOS = (
    "oversampling_only",
    "full_reversal",
    "partial_reversal",
    "ecg_perturbation",
    "ecg_perturbation_10pct",
    "ecg_perturbation_25pct",
    "ecg_perturbation_50pct",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stratified_indices(y, fraction, rng):
    selected = []
    for class_label in np.unique(y):
        candidates = np.flatnonzero(y == class_label)
        count = max(1, int(round(len(candidates) * fraction)))
        selected.extend(rng.choice(candidates, size=count, replace=False))
    return np.asarray(selected, dtype=np.int64)


def _shift_without_wrap(signal, amount):
    if amount > 0:
        return np.concatenate((np.repeat(signal[0], amount), signal[:-amount]))
    if amount < 0:
        amount = -amount
        return np.concatenate((signal[amount:], np.repeat(signal[-1], amount)))
    return signal.copy()


def augment_training_data(X_train, y_train, scenario, seed=SEED):
    """Return originals plus the requested training-only augmentation."""
    if scenario == "oversampling_only":
        return X_train.copy(), y_train.copy()

    rng = np.random.default_rng(seed)
    if scenario == "full_reversal":
        augmented = X_train[:, ::-1].copy()
        augmented_labels = y_train
    elif scenario == "partial_reversal":
        indices = _stratified_indices(y_train, fraction=0.20, rng=rng)
        augmented = X_train[indices, ::-1].copy()
        augmented_labels = y_train[indices]
    elif scenario == "ecg_perturbation" or scenario.startswith("ecg_perturbation_"):
        if scenario == "ecg_perturbation":
            source_indices = np.arange(len(y_train))
        else:
            fraction = int(scenario.removeprefix("ecg_perturbation_").removesuffix("pct")) / 100
            source_indices = _stratified_indices(y_train, fraction=fraction, rng=rng)
        source_signals = X_train[source_indices]
        augmented = np.empty_like(source_signals)
        phase = np.linspace(0.0, 2.0 * np.pi, X_train.shape[1], endpoint=False)
        for index, signal in enumerate(source_signals):
            signal_scale = max(float(np.std(signal)), 1e-6)
            amplitude = rng.normal(loc=1.0, scale=0.05)
            noise = rng.normal(loc=0.0, scale=0.01 * signal_scale, size=signal.shape)
            drift_amplitude = rng.normal(loc=0.0, scale=0.02 * signal_scale)
            drift_phase = rng.uniform(0.0, 2.0 * np.pi)
            shift = int(rng.integers(-2, 3))
            perturbed = amplitude * signal + noise + drift_amplitude * np.sin(phase + drift_phase)
            augmented[index] = _shift_without_wrap(perturbed, shift)
        augmented_labels = y_train[source_indices]
    else:
        raise ValueError(f"Unknown augmentation scenario: {scenario}")

    return (
        np.concatenate((X_train, augmented), axis=0),
        np.concatenate((y_train, augmented_labels), axis=0),
    )


def run_scenario(scenario, train, test, device, seed=SEED, task="binary"):
    set_seed(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = helpers.cleanAndSplitRaw(train, test)
    if task == "binary":
        y_train = (y_train > 0).astype(np.int64)
        y_val = (y_val > 0).astype(np.int64)
        y_test = (y_test > 0).astype(np.int64)
    elif task == "multiclass":
        train_mask = y_train > 0
        val_mask = y_val > 0
        test_mask = y_test > 0
        X_train, y_train = X_train[train_mask], y_train[train_mask] - 1
        X_val, y_val = X_val[val_mask], y_val[val_mask] - 1
        X_test, y_test = X_test[test_mask], y_test[test_mask] - 1
    else:
        raise ValueError(f"Unknown task: {task}")

    X_train_augmented, y_train_augmented = augment_training_data(
        X_train, y_train, scenario, seed=seed
    )

    # Fit preprocessing only on this scenario's training data. Reversal and
    # perturbation happen first, so moved timesteps receive the destination
    # column's normalization rather than retaining the source column's scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_augmented).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    ros = RandomOverSampler(random_state=seed)
    X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train_augmented)
    train_loader, val_loader, test_loader = helpers.getLoaders(
        X_train_ros, X_val_scaled, X_test_scaled,
        y_train_ros, y_val, y_test,
    )

    model = helpers.chooseTypeLSTM(
        MODEL_TYPE,
        input_size=1,
        n_classes=len(np.unique(y_train)),
        n_timesteps=X_train.shape[1],
        device=device,
    )
    checkpoint_path = RESULTS_DIR / f"{task}_{MODEL_TYPE}_{scenario}_seed{seed}.pth"
    train_losses, val_losses, times = helpers.performTrainingLoop(
        model,
        train_loader,
        val_loader,
        device,
        torch.nn.CrossEntropyLoss(),
        str(checkpoint_path),
    )

    best_model = helpers.chooseTypeLSTM(
        MODEL_TYPE,
        input_size=1,
        n_classes=len(np.unique(y_train)),
        n_timesteps=X_train.shape[1],
        device=device,
    )
    best_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    y_pred, y_true, _ = helpers.performTestingLoop(best_model, test_loader, device)

    return {
        "scenario": scenario,
        "task": task,
        "seed": seed,
        "model_type": MODEL_TYPE,
        "original_training_samples": int(len(y_train)),
        "post_augmentation_samples": int(len(y_train_augmented)),
        "post_oversampling_samples": int(len(y_train_ros)),
        "epochs_run": len(val_losses),
        "best_epoch": int(np.argmin(val_losses) + 1),
        "best_validation_loss": float(np.min(val_losses)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mean_epoch_seconds": float(np.mean(times)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train, test = helpers.loadRawDataFromFile()

    results_path = RESULTS_DIR / "binary_mLSTM_augmentation_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    completed = {(result["scenario"], result["seed"]) for result in results}
    for scenario in SCENARIOS:
        if (scenario, SEED) in completed:
            print(f"Skipping completed scenario: {scenario}, seed {SEED}")
            continue
        print(f"\n{'=' * 20} {scenario} {'=' * 20}", flush=True)
        result = run_scenario(scenario, train, test, device)
        results.append(result)
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)

    for seed in (7, 123):
        for scenario in ("oversampling_only", "ecg_perturbation_50pct"):
            if (scenario, seed) in completed:
                print(f"Skipping completed scenario: {scenario}, seed {seed}")
                continue
            print(f"\n{'=' * 20} {scenario}, seed {seed} {'=' * 20}", flush=True)
            result = run_scenario(scenario, train, test, device, seed=seed)
            results.append(result)
            results_path.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)

    print("\nFinal comparison:")
    for result in sorted(results, key=lambda item: item["best_validation_loss"]):
        print(
            f"{result['scenario']:20s} val_loss={result['best_validation_loss']:.6f} "
            f"test_bal_acc={result['test_balanced_accuracy']:.4f} "
            f"macro_f1={result['test_macro_f1']:.4f}"
        )

    print("\nThree-seed finalist means:")
    for scenario in ("oversampling_only", "ecg_perturbation_50pct"):
        finalist_results = [result for result in results if result["scenario"] == scenario]
        print(
            f"{scenario:25s} "
            f"val_loss={np.mean([r['best_validation_loss'] for r in finalist_results]):.6f} "
            f"test_bal_acc={np.mean([r['test_balanced_accuracy'] for r in finalist_results]):.4f} "
            f"macro_f1={np.mean([r['test_macro_f1'] for r in finalist_results]):.4f}"
        )

    multiclass_path = RESULTS_DIR / "multiclass_mLSTM_augmentation_results.json"
    multiclass_results = json.loads(multiclass_path.read_text()) if multiclass_path.exists() else []
    multiclass_completed = {
        (result["scenario"], result["seed"]) for result in multiclass_results
    }
    for scenario in ("oversampling_only", "full_reversal", "ecg_perturbation_50pct"):
        if (scenario, SEED) in multiclass_completed:
            print(f"Skipping completed multiclass scenario: {scenario}, seed {SEED}")
            continue
        print(f"\n{'=' * 20} multiclass {scenario} {'=' * 20}", flush=True)
        result = run_scenario(scenario, train, test, device, task="multiclass")
        multiclass_results.append(result)
        multiclass_path.write_text(json.dumps(multiclass_results, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)

    print("\nMulticlass comparison:")
    for result in sorted(multiclass_results, key=lambda item: item["best_validation_loss"]):
        print(
            f"{result['scenario']:25s} val_loss={result['best_validation_loss']:.6f} "
            f"test_bal_acc={result['test_balanced_accuracy']:.4f} "
            f"macro_f1={result['test_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
