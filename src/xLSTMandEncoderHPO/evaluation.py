import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

from plotting import plotConfusionMatrices, plotRocPrPanel


def chooseThresholdMaxFbeta(y_true, y_score, beta=1.0):
    """Pick the decision threshold on a precision_recall_curve that maximizes
    F-beta for one binary (or one-vs-rest) score column.

    beta=1 is plain F1 (precision and recall weighted equally). beta>1 shifts
    the optimum towards higher-recall thresholds, i.e. it is willing to trade
    some precision (more false positives) for fewer false negatives - this is
    how "prioritize catching positives, even at the cost of a few more false
    positives" gets encoded as a single number instead of picking a threshold
    by hand.

    Returns (threshold, fbeta, precision, recall) at the best point.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve appends a final (precision=1, recall=0) point for
    # an implicit threshold of +inf that has no corresponding entry in
    # `thresholds`, so drop it before pairing the arrays up.
    precision, recall = precision[:-1], recall[:-1]

    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    fbeta = np.divide(
        (1 + beta_sq) * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    best_idx = int(np.argmax(fbeta))
    return (
        float(thresholds[best_idx]),
        float(fbeta[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
    )


def chooseMulticlassThresholds(y_true, y_score, class_labels, beta=1.0):
    """One-vs-rest F-beta-optimal threshold per class.

    Each class gets its own threshold from chooseThresholdMaxFbeta run
    against its own one-vs-rest column, so classes are treated independently
    rather than pooled - a rare class does not get drowned out by a common
    one when picking its cutoff.
    """
    y_true_bin = label_binarize(y_true, classes=class_labels)
    n_classes = len(class_labels)
    thresholds = np.zeros(n_classes)
    fbetas = np.zeros(n_classes)
    precisions = np.zeros(n_classes)
    recalls = np.zeros(n_classes)

    for i in range(n_classes):
        thr, fbeta, prec, rec = chooseThresholdMaxFbeta(
            y_true_bin[:, i], y_score[:, i], beta=beta
        )
        thresholds[i], fbetas[i], precisions[i], recalls[i] = thr, fbeta, prec, rec

    return thresholds, fbetas, precisions, recalls


def applyMulticlassThresholds(y_score, thresholds):
    """Turn per-class one-vs-rest thresholds into a single predicted label.

    Each sample goes to the class whose score clears its own threshold by the
    largest margin. Independently tuned thresholds don't guarantee any class
    clears its bar for every sample, so when none do, fall back to plain
    argmax (equivalent to the untuned decision rule) rather than leaving the
    sample unclassified.
    """
    margin = y_score - thresholds[np.newaxis, :]
    any_cleared = (margin > 0).any(axis=1)
    return np.where(
        any_cleared,
        np.argmax(margin, axis=1),
        np.argmax(y_score, axis=1),
    )


def computeScalarMetrics(y_true, y_pred):
    """Accuracy/precision/recall/F1/balanced-accuracy, macro-averaged where
    applicable.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def printMetricsSummary(metrics, y_true, y_pred, time_per_epoch=None):
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Macro-averaged precision: {metrics['precision']:.4f}")
    print(f"Test Macro-averaged recall: {metrics['recall']:.4f}")
    print(f"Test Macro-averaged F1_score: {metrics['f1']:.4f}")
    print(f"Test balanced accuracy: {metrics['balanced_accuracy']:.4f}")

    if time_per_epoch:
        average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
        print(f"Average time per epoch: {average_time_per_epoch:.4f}s")

    print("\nClassification reports:")
    print(classification_report(y_true, y_pred))


def evaluateAndReport(y_true, y_pred, all_logits, class_labels, prefix, roc_suffix="roc",
                       time_per_epoch=None, decision_thresholds=None,
                       val_true=None, val_logits=None):
    """Print scalar metrics + classification report, plot the confusion
    matrix, and plot test-set (and optional validation-set) ROC/PR curves.

    Same call shape as the old performMetrics: one call per final model
    evaluation, from runFinalModelSelection.
    """
    metrics = computeScalarMetrics(y_true, y_pred)
    printMetricsSummary(metrics, y_true, y_pred, time_per_epoch=time_per_epoch)

    plotConfusionMatrices(y_true, y_pred, class_labels, save_path=f"{prefix}_cf.png")

    # Class scores (probabilities) are needed for threshold-based metrics
    # like AUROC/AUPRC - argmax predictions (y_pred) collapse that info away.
    y_score = torch.softmax(all_logits, dim=1).numpy()

    # No threshold markers here - the decision threshold was chosen on the
    # validation set, not this curve, so a marker here doesn't reflect an
    # actual decision made from this data.
    macro_auroc, macro_auprc = plotRocPrPanel(
        y_true, y_score, class_labels, title="Test set",
        save_path=f"{prefix}_{roc_suffix}.png", annotate_threshold=False,
    )
    print(f"\nMacro-average AUROC: {macro_auroc:.4f}")
    print(f"Macro-average AUPRC: {macro_auprc:.4f}")

    # Also plot the validation-set curves, when given - the decision
    # threshold(s) were chosen on validation data, so that's the only curve
    # where a threshold marker reflects an actual decision made from the data.
    if val_true is not None and val_logits is not None:
        y_val_score = torch.softmax(val_logits, dim=1).numpy()
        val_auroc, val_auprc = plotRocPrPanel(
            val_true, y_val_score, class_labels, title="Validation set (threshold selection)",
            save_path=f"{prefix}_{roc_suffix}_val.png", decision_thresholds=decision_thresholds,
        )
        print(f"\nValidation macro AUROC: {val_auroc:.4f}")
        print(f"Validation macro AUPRC: {val_auprc:.4f}")
