import json
from pathlib import Path

import numpy as np
import optuna
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from config import (
    BINARY_THRESHOLD_BETA,
    EARLY_STOP_NO_IMPROVEMENT,
    EPOCHS,
    EVAL_BATCH_SIZE,
    HPO_EARLY_STOP_NO_IMPROVEMENT,
    HPO_EPOCHS,
    HPO_PRUNER_N_STARTUP_TRIALS,
    HPO_PRUNER_N_WARMUP_STEPS,
    MULTICLASS_THRESHOLD_BETA,
    N_SPLITS,
    N_TRIALS,
    NUM_THREADS,
    RANDOM_STATE,
    TEST_FRACTION,
)
from data import _loader, loadTaskData, makeFolds, multiclassOversampleStrategy, relabelForTask, setSeed
from evaluation import applyMulticlassThresholds, chooseMulticlassThresholds, chooseThresholdMaxFbeta, evaluateAndReport
from hpo_search import resolveBestParams, suggestHyperparams
from models import buildModel, countTrainableParameters
from training import performInferenceLoop, performTrainingLoopOptuna


def scoreHyperparams(params, X_train, y_train, X_val, y_val, device, trial, task):
    """Train one Optuna trial's config on a single fixed train/val split of the
    development pool and return its best validation loss.

    trial is passed through to performTrainingLoopOptuna (not None): with
    exactly one split per trial the epoch counter increases monotonically for
    the whole trial (unlike the old per-fold CV objective, where the epoch
    counter reset every fold), so trial.report/should_prune can safely act on
    it and unpromising trials get pruned instead of burning the full budget.

    y_train/y_val arrive in the raw subtype label space for the binary task
    (see loadTaskData) - oversampling and relabelForTask below both need
    that raw label, so the collapse to healthy/unhealthy only happens after
    resampling.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Duplicate minority rows in the training split only, on the raw subtype
    # label; the validation split keeps its natural distribution so the
    # score still reflects real-world performance.
    if task == "multiclass":
        ros = RandomOverSampler(
            sampling_strategy=multiclassOversampleStrategy(y_train),
            random_state=RANDOM_STATE,
        )
        X_train_scaled, y_train_ros = ros.fit_resample(X_train_scaled, y_train)
    else:
        y_train_ros = y_train

    y_train_ros = relabelForTask(task, y_train_ros)
    y_val = relabelForTask(task, y_val)

    n_classes = len(np.unique(np.concatenate([y_train_ros, y_val])))
    n_tokens = X_train.shape[1]

    # Constant seed (not trial-dependent) so every trial sees the same batch
    # shuffle order - isolates the effect of the hyperparameters from the
    # effect of randomness in data order.
    setSeed(RANDOM_STATE)
    train_loader = _loader(X_train_scaled, y_train_ros, params["batch_size"], True)
    val_loader = _loader(X_val_scaled, y_val, params["batch_size"], False)
    model = buildModel(params, 1, n_classes, n_tokens, device)
    print(
        f"Optuna trial trainable parameters: {countTrainableParameters(model)}",
        flush=True,
    )

    _, val_losses, _ = performTrainingLoopOptuna(
        model, train_loader, val_loader, device,
        # Oversampling and class-weighting together would over-correct the loss.
        nn.CrossEntropyLoss(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        epochs=HPO_EPOCHS,
        early_stop_patience=HPO_EARLY_STOP_NO_IMPROVEMENT,
        verbose=False,
        trial=trial,
    )
    return float(min(val_losses))


def runFoldCV(params, X_development, y_development, folds, device, output_prefix, task):
    """Train one model per fold at the full EPOCHS budget (not the HPO
    budget), checkpointing each fold's best-val-loss weights to
    '{output_prefix}_fold{n}_model.pth'.

    y_development arrives in the raw subtype label space for the binary task
    (see loadTaskData) - each fold's stratified indices, and the
    oversampling below, both need that raw label; the collapse to
    healthy/unhealthy happens per-fold, right after resampling.

    Returns a list of per-fold dicts (fold, val_loss, checkpoint_path,
    val_loader, scaler) - the val_loader/scaler ride along so the caller can
    score the winning fold's model on its own validation split and on the
    test set without re-deriving either from indices.
    """
    n_classes = len(np.unique(relabelForTask(task, y_development)))
    n_tokens = X_development.shape[1]
    fold_results = []

    for fold_number, (train_index, val_index) in enumerate(folds, start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_development[train_index]).astype(np.float32)
        X_val = scaler.transform(X_development[val_index]).astype(np.float32)
        y_train, y_val = y_development[train_index], y_development[val_index]

        # Duplicate minority rows in the training fold only, on the raw
        # subtype label; the validation fold keeps its natural distribution
        # so the fold score still reflects real-world performance.
        if task == "multiclass":
            ros = RandomOverSampler(
                sampling_strategy=multiclassOversampleStrategy(y_train),
                random_state=RANDOM_STATE,
            )
            X_train, y_train = ros.fit_resample(X_train, y_train)

        y_train = relabelForTask(task, y_train)
        y_val = relabelForTask(task, y_val)

        setSeed(RANDOM_STATE + fold_number)
        train_loader = _loader(X_train, y_train, params["batch_size"], True)
        val_loader = _loader(X_val, y_val, params["batch_size"], False)
        model = buildModel(params, 1, n_classes, n_tokens, device)
        trainable_parameters = countTrainableParameters(model)
        print(
            f"Fold {fold_number}/{N_SPLITS} trainable parameters: "
            f"{trainable_parameters}",
            flush=True,
        )

        checkpoint_path = f"{output_prefix}_fold{fold_number}_model.pth"
        _, val_losses, epoch_times = performTrainingLoopOptuna(
            model, train_loader, val_loader, device,
            nn.CrossEntropyLoss(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP_NO_IMPROVEMENT,
            checkpoint_path=checkpoint_path,
            verbose=False,
            trial=None,
            label=f"Fold {fold_number}/{N_SPLITS}",
            leave=False,
        )
        score = float(min(val_losses))
        print(f"  fold {fold_number}/{N_SPLITS}: validation loss={score:.6f}")
        fold_results.append({
            "fold": fold_number,
            "val_loss": score,
            "checkpoint_path": checkpoint_path,
            "val_loader": val_loader,
            "scaler": scaler,
            "time_per_epoch": epoch_times,
            "trainable_parameters": trainable_parameters,
        })

    return fold_results


def selectBestFold(fold_results):
    """Pick the fold with the lowest validation loss and print the summary
    line. Returns (best, fold_val_losses, fold_std).
    """
    fold_val_losses = [r["val_loss"] for r in fold_results]
    best = min(fold_results, key=lambda r: r["val_loss"])
    fold_std = float(np.std(fold_val_losses, ddof=1))
    print(
        f"\nFold validation losses: {[f'{s:.6f}' for s in fold_val_losses]}\n"
        f"Best fold: {best['fold']}/{N_SPLITS} (val loss={best['val_loss']:.6f}), "
        f"mean={np.mean(fold_val_losses):.6f} +/- {fold_std:.6f}"
    )
    return best, fold_val_losses, fold_std


def cleanupNonWinningCheckpoints(fold_results, best):
    """Keep only the winning fold's checkpoint - it is the single source of
    truth for "checkpoint_path" in the JSON report - and delete the rest to
    avoid five checkpoints of clutter per run.
    """
    for r in fold_results:
        if r is not best:
            Path(r["checkpoint_path"]).unlink(missing_ok=True)


def loadBestFoldModel(params, best, input_size, n_classes, n_tokens, device):
    """Rebuild the winning fold's architecture and reload its checkpointed
    weights.
    """
    model = buildModel(params, input_size, n_classes, n_tokens, device)
    model.load_state_dict(
        torch.load(best["checkpoint_path"], map_location=device, weights_only=True)
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Best fold model has {num_params} trainable parameters.")
    return model


def chooseDecisionThreshold(y_val_true, y_val_score, class_labels):
    """Select the decision threshold(s) on the validation set - binary gets
    one F-beta-optimal threshold, multiclass gets one per class.

    Returns (decision_thresholds, threshold_info) - the latter is the
    JSON-serializable dict that goes straight into the report.
    """
    if len(class_labels) == 2:
        threshold, best_fbeta, thr_precision, thr_recall = chooseThresholdMaxFbeta(
            y_val_true, y_val_score[:, 1], beta=BINARY_THRESHOLD_BETA
        )
        print(
            f"\nValidation-selected decision threshold "
            f"(F{BINARY_THRESHOLD_BETA:g}-optimal, recall-prioritized): "
            f"{threshold:.4f} (F{BINARY_THRESHOLD_BETA:g}={best_fbeta:.4f}, "
            f"precision={thr_precision:.4f}, recall={thr_recall:.4f})"
        )
        threshold_info = {"beta": BINARY_THRESHOLD_BETA, "threshold": threshold}
        return threshold, threshold_info

    thresholds, fbetas, precisions, recalls = chooseMulticlassThresholds(
        y_val_true, y_val_score, class_labels, beta=MULTICLASS_THRESHOLD_BETA
    )
    print(
        f"\nValidation-selected per-class decision thresholds "
        f"(F{MULTICLASS_THRESHOLD_BETA:g}-optimal, balanced):"
    )
    for label, thr, fb, prec, rec in zip(class_labels, thresholds, fbetas, precisions, recalls):
        print(
            f"  class {int(label) + 1}: threshold={thr:.4f}, "
            f"F{MULTICLASS_THRESHOLD_BETA:g}={fb:.4f}, precision={prec:.4f}, recall={rec:.4f}"
        )
    threshold_info = {"beta": MULTICLASS_THRESHOLD_BETA, "thresholds": thresholds.tolist()}
    return thresholds, threshold_info


def applyDecisionThreshold(y_test_score, decision_thresholds, class_labels):
    """Turn validation-selected threshold(s) into test-set predicted labels."""
    if len(class_labels) == 2:
        return (y_test_score[:, 1] >= decision_thresholds).astype(np.int64)
    return applyMulticlassThresholds(y_test_score, decision_thresholds)


def evaluateOnTestSet(model, scaler, X_test, y_test, batch_size, device):
    """Scale X_test with the winning fold's own scaler, run inference, and
    return (test_logits, y_true, y_test_score).

    The test set must be scaled with the WINNING fold's own scaler (each
    fold fit its own in runFoldCV) - reusing a different fold's scaler
    would leak that fold's train distribution into the test transform.
    """
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    test_loader = _loader(X_test_scaled, y_test, batch_size, False)
    test_logits, test_targets = performInferenceLoop(model, test_loader, device)
    y_true = test_targets.numpy()
    y_test_score = torch.softmax(test_logits, dim=1).numpy()
    return test_logits, y_true, y_test_score


def buildSelectionReport(task, model_type, params, fold_val_losses, fold_std, best,
                          threshold_info, test_score, split_sizes, extra_report_fields=None):
    report = {
        "task": task,
        "model_type": model_type,
        "lambda_star": params,
        "cv_fold_val_losses": fold_val_losses,
        "cv_fold_val_loss_mean": float(np.mean(fold_val_losses)),
        "cv_fold_val_loss_std": fold_std,
        "best_fold": best["fold"],
        "checkpoint_path": best["checkpoint_path"],
        "decision_threshold": threshold_info,
        "test_performance_indicator": "balanced_accuracy",
        "test_balanced_accuracy": test_score,
        "trainable_parameters": best["trainable_parameters"],
        "time_per_epoch": best["time_per_epoch"],
        "average_time_per_epoch": float(np.mean(best["time_per_epoch"])),
        "split_sizes": split_sizes,
    }
    if extra_report_fields:
        report.update(extra_report_fields)
    return report


def writeSelectionReport(report, output_prefix):
    Path(f"{output_prefix}_cv_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def runFinalModelSelection(task, model_type, output_prefix, roc_suffix, params,
                            X, y, X_development, y_development, X_test, y_test,
                            folds, device, extra_report_fields=None):
    """5-fold CV over the dev pool at the full training budget; keep the
    single fold whose model had the lowest validation loss ("best fold
    model") as the deployed model, evaluate it once on the untouched test
    set, write the JSON report.

    Shared tail of runHPOModelSelection (params = HPO winner) and
    runFixedParamsModelSelection (params = hand-picked dict) - both just
    build params/folds differently and hand off here.

    y and y_test are already task-facing labels (relabelForTask applied by
    the caller); y_development is still the raw subtype label for binary,
    since runFoldCV needs it that way to stratify-oversample each fold.
    """
    fold_results = runFoldCV(params, X_development, y_development, folds, device, output_prefix, task)
    best, fold_val_losses, fold_std = selectBestFold(fold_results)
    cleanupNonWinningCheckpoints(fold_results, best)

    input_size = 1
    n_classes = len(np.unique(y))
    n_tokens = X.shape[1]
    best_model = loadBestFoldModel(params, best, input_size, n_classes, n_tokens, device)

    class_labels = np.unique(y)

    # Threshold is selected on the winning fold's own validation split
    # (natural class distribution, never oversampled or trained on) and then
    # frozen before it ever touches the test set.
    val_logits, val_targets = performInferenceLoop(best_model, best["val_loader"], device)
    y_val_true = val_targets.numpy()
    y_val_score = torch.softmax(val_logits, dim=1).numpy()

    decision_thresholds, threshold_info = chooseDecisionThreshold(y_val_true, y_val_score, class_labels)

    eval_batch_size = max(params["batch_size"], EVAL_BATCH_SIZE)
    test_logits, y_true, y_test_score = evaluateOnTestSet(
        best_model, best["scaler"], X_test, y_test, eval_batch_size, device
    )
    y_pred = applyDecisionThreshold(y_test_score, decision_thresholds, class_labels)

    test_score = float(balanced_accuracy_score(y_true, y_pred))
    test_metrics = evaluateAndReport(
        y_true, y_pred, test_logits, class_labels,
        prefix=output_prefix, roc_suffix=roc_suffix,
        decision_thresholds=decision_thresholds,
        val_true=y_val_true, val_logits=val_logits,
        time_per_epoch=best["time_per_epoch"],
    )

    best_train_index, best_val_index = folds[best["fold"] - 1]
    split_sizes = {
        "development": len(y_development),
        "best_fold_train": len(best_train_index),
        "best_fold_validation": len(best_val_index),
        "test": len(y_test),
    }
    report = buildSelectionReport(
        task, model_type, params, fold_val_losses, fold_std, best,
        threshold_info, test_score, split_sizes,
        extra_report_fields={
            **(extra_report_fields or {}),
            "test_macro_f1": test_metrics["f1"],
            "test_macro_auroc": test_metrics["macro_auroc"],
            "test_macro_auprc": test_metrics["macro_auprc"],
        },
    )
    writeSelectionReport(report, output_prefix)
    return report


def runHPOModelSelection(task, model_type, output_prefix, roc_suffix):
    """Search hyperparameters with Optuna on a single train/val split of the
    dev pool (fast, prunable), then hand the winner to runFinalModelSelection
    for a proper 5-fold CV fit/evaluation.
    """
    setSeed(RANDOM_STATE)
    torch.set_num_threads(NUM_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model type:", model_type)

    # y_raw is the raw subtype label for binary (see loadTaskData) - every
    # split/fold below stratifies on it so rare abnormal subtypes land in
    # every split, not just the pooled healthy/unhealthy ratio.
    X, y_raw = loadTaskData(task)

    # The test set is reserved before HPO/CV and never passed to an objective.
    X_development, X_test, y_development, y_test_raw = train_test_split(
        X, y_raw, test_size=TEST_FRACTION, stratify=y_raw, random_state=RANDOM_STATE
    )
    print(f"Development pool shape: {X_development.shape} (used for HPO + {N_SPLITS}-fold CV)")
    print(f"Held-out test shape: {X_test.shape}")

    # Single fixed split for HPO trials - same proportions as one CV fold
    # (test_size = 1/N_SPLITS), computed once and reused by every trial
    # instead of paying for N_SPLITS folds per trial.
    X_hpo_train, X_hpo_val, y_hpo_train, y_hpo_val = train_test_split(
        X_development, y_development,
        test_size=1 / N_SPLITS, stratify=y_development, random_state=RANDOM_STATE + 1,
    )

    def objective(trial):
        params = suggestHyperparams(trial, model_type)
        print(f"\nTrial {trial.number + 1}/{N_TRIALS}: {params}")
        score = scoreHyperparams(
            params, X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val, device, trial, task
        )
        print(f"  validation loss={score:.6f}")
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=HPO_PRUNER_N_STARTUP_TRIALS,
            n_warmup_steps=HPO_PRUNER_N_WARMUP_STEPS,
        ),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = resolveBestParams(study, model_type)
    print("\nSelected lambda*:", best_params)
    print(f"HPO single-split validation loss: {study.best_value:.6f}")

    folds = makeFolds(X_development, y_development)
    # Only relabel here, for the labels that reach runFinalModelSelection's
    # metrics/report directly (y, y_test) - y_development stays raw so
    # runFoldCV can stratify-oversample each fold on the true subtype label.
    y = relabelForTask(task, y_raw)
    y_test = relabelForTask(task, y_test_raw)
    return runFinalModelSelection(
        task, model_type, output_prefix, roc_suffix, best_params,
        X, y, X_development, y_development, X_test, y_test, folds, device,
        extra_report_fields={"hpo_validation_loss": study.best_value},
    )


def runFixedParamsModelSelection(task, model_type, output_prefix, roc_suffix, params):
    """Same 5-fold CV final-model fit/evaluation as runHPOModelSelection's
    tail, skipping the Optuna search - params is a hand-picked dict shaped
    like buildModel expects (see suggestXLSTMHyperparams /
    suggestTransformerHyperparams for key names, resolveBestParams for which
    extra keys are needed, e.g. "model_type").
    """
    setSeed(RANDOM_STATE)
    torch.set_num_threads(NUM_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model type:", model_type)
    print("Fixed params:", params)

    # y_raw is the raw subtype label for binary (see loadTaskData) - every
    # split/fold below stratifies on it so rare abnormal subtypes land in
    # every split, not just the pooled healthy/unhealthy ratio.
    X, y_raw = loadTaskData(task)
    X_development, X_test, y_development, y_test_raw = train_test_split(
        X, y_raw, test_size=TEST_FRACTION, stratify=y_raw, random_state=RANDOM_STATE
    )
    print(f"Development pool shape: {X_development.shape} (used for {N_SPLITS}-fold CV)")
    print(f"Held-out test shape: {X_test.shape}")

    folds = makeFolds(X_development, y_development)
    # Only relabel here, for the labels that reach runFinalModelSelection's
    # metrics/report directly (y, y_test) - y_development stays raw so
    # runFoldCV can stratify-oversample each fold on the true subtype label.
    y = relabelForTask(task, y_raw)
    y_test = relabelForTask(task, y_test_raw)
    return runFinalModelSelection(
        task, model_type, output_prefix, roc_suffix, params,
        X, y, X_development, y_development, X_test, y_test, folds, device,
    )
