import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def plotConfusionMatrices(y_true, y_pred, class_labels, save_path):
    """Compute, print, and plot the test-set confusion matrix.

    Row-normalized panel shows what fraction of each true class landed in
    each predicted bucket, which raw counts hide when classes are imbalanced
    (e.g. mc_cf.png: class 1 only has ~43 test samples total).
    """
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig_cm, (ax_cm_counts, ax_cm_pct) = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(cmap="Blues", ax=ax_cm_counts, colorbar=False)
    ax_cm_counts.set_title("Counts")

    ConfusionMatrixDisplay(cm_norm, display_labels=class_labels).plot(
        cmap="Blues", ax=ax_cm_pct, values_format=".1%"
    )
    ax_cm_pct.set_title("Proportion of true class")

    fig_cm.suptitle("Test set confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plotRocPrCurves(y_true, y_score, class_labels, ax_roc, ax_pr,
                     decision_thresholds=None, annotate_threshold=True):
    """Draw one-vs-rest ROC/PR curves onto the given axes, optionally with
    decision-threshold markers on the PR axes. Markers are recomputed from
    y_true/y_score - the same arrays that produce the curves - so they
    always land exactly on the plotted line, rather than being reused from a
    different split.

    Returns (macro_auroc, macro_auprc).
    """
    n_classes = len(class_labels)
    # label_binarize collapses to a single column for exactly 2 classes, which
    # breaks the one-vs-rest machinery below, so binary needs its own path.
    is_binary = n_classes == 2

    if is_binary:
        pos_score = y_score[:, 1]
        macro_auroc = roc_auc_score(y_true, pos_score)
        macro_auprc = average_precision_score(y_true, pos_score)
    else:
        y_true_bin = label_binarize(y_true, classes=class_labels)
        macro_auroc = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
        macro_auprc = average_precision_score(y_true_bin, y_score, average="macro")

    operating_points = []

    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, pos_score)
        ax_roc.plot(fpr, tpr, label=f"Class {int(class_labels[1]) + 1} (AUROC={macro_auroc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, pos_score)
        ax_pr.plot(recall, precision, label=f"Class {int(class_labels[1]) + 1} (AUPRC={macro_auprc:.3f})")

        if annotate_threshold and decision_thresholds is not None:
            thr = float(decision_thresholds)
            y_at_thr = (pos_score >= thr).astype(np.int64)
            op_precision = precision_score(y_true, y_at_thr, zero_division=0)
            op_recall = recall_score(y_true, y_at_thr, zero_division=0)
            operating_points.append(
                (op_recall, op_precision, f"Class {int(class_labels[1]) + 1} thr={thr:.3f}")
            )
    else:
        for i, class_label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            class_auroc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
            ax_roc.plot(fpr, tpr, label=f"Class {int(class_label) + 1} (AUROC={class_auroc:.3f})")

            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            class_auprc = average_precision_score(y_true_bin[:, i], y_score[:, i])
            ax_pr.plot(recall, precision, label=f"Class {int(class_label) + 1} (AUPRC={class_auprc:.3f})")

            if annotate_threshold and decision_thresholds is not None:
                thr = float(decision_thresholds[i])
                y_at_thr = (y_score[:, i] >= thr).astype(np.int64)
                op_precision = precision_score(y_true_bin[:, i], y_at_thr, zero_division=0)
                op_recall = recall_score(y_true_bin[:, i], y_at_thr, zero_division=0)
                operating_points.append(
                    (op_recall, op_precision, f"Class {int(class_label) + 1} thr={thr:.3f}")
                )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC curves (macro AUROC={macro_auroc:.3f})")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(linestyle="dashed")

    # Mark the threshold(s) chosen on the validation set, so the PR curve
    # shows where the deployed decision rule actually sits on the tradeoff.
    for j, (op_recall, op_precision, op_label) in enumerate(operating_points):
        ax_pr.scatter(
            op_recall, op_precision,
            marker="o", s=40, color="black", zorder=5,
            label="Selected threshold" if j == 0 else None,
        )
        ax_pr.annotate(
            op_label, (op_recall, op_precision),
            textcoords="offset points", xytext=(6, -8), fontsize=7,
        )

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall curves (macro AUPRC={macro_auprc:.3f})")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(linestyle="dashed")

    return macro_auroc, macro_auprc


def plotRocPrPanel(y_true, y_score, class_labels, title, save_path,
                    decision_thresholds=None, annotate_threshold=True):
    """Build a ROC+PR figure, draw both curves via plotRocPrCurves, save and
    show it. Returns (macro_auroc, macro_auprc) for the caller to print.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    macro_auroc, macro_auprc = plotRocPrCurves(
        y_true, y_score, class_labels, ax_roc, ax_pr,
        decision_thresholds=decision_thresholds, annotate_threshold=annotate_threshold,
    )
    fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return macro_auroc, macro_auprc
