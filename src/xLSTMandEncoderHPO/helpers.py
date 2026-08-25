import json
import random
import time
from pathlib import Path

import optuna
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

#Statics
BATCH_SIZE = 32
D_MODEL = 64
DIM_FF = 128
NUM_BLOCKS = 1 # set to 2 or more for xLSTM with mLSTM and sLSTM block
NUM_HEADS = 4
CONV1D_KERNEL_SIZE= 4 # the amount of timesteps we should use to get the representation that we want
QVK_PROJ_BLOCKSIZE=4
NUM_LAYERS = 1
LR = 0.0001
EPOCHS = 300
EARLY_STOP_NO_IMPROVEMENT = 10

# Optuna HPO settings. Trials run on a smaller epoch budget than the final
# training run so a full search finishes in reasonable time on CPU; the
# winning config is then retrained with the full EPOCHS budget.
N_TRIALS = 20
HPO_EPOCHS = 40
HPO_EARLY_STOP_NO_IMPROVEMENT = 8

# Cross-validated model selection settings (runCrossValidatedModelSelection).
# TEST_FRACTION is deliberately inverted from the usual small-test-set
# convention: only 40% of the data is pooled for CV/HPO, 60% is held out.
RANDOM_STATE = 42
N_SPLITS = 5
TEST_FRACTION = 0.60

# Validation/test batches only ever run forward, so they can be far larger
# than the trial's training batch size. The val set is ~2250 samples and is
# evaluated every epoch, so this is the cheapest speedup available on CPU.
EVAL_BATCH_SIZE = 256

# Decision-threshold selection (runCrossValidatedModelSelection), done on the
# validation set and then frozen before touching the test set.
#   Binary: F-beta with beta>1 weights recall over precision, so missing an
#   abnormal ECG (false negative) costs more than a false alarm.
#   Multiclass: beta=1 (plain F1), applied per class one-vs-rest, so no class
#   is favoured over another.
BINARY_THRESHOLD_BETA = 2.0
MULTICLASS_THRESHOLD_BETA = 1.0

# ECG5000 epochs are tiny (500 samples), so the per-op matmuls are small and
# spreading them over every core costs more in thread sync than it saves.
NUM_THREADS = 4

# Per-trial epoch progress bars. Set False for clean logs when redirecting
# output to a file.
SHOW_PROGRESS = True

# Which architecture the Optuna workflow searches over.
#   "xlstm"       -> TunableXLSTMClassifier (mLSTM / mixed / sLSTM block stack)
#   "transformer" -> BaseTransformerClassifier (the original encoder)
# This is the single switch: it picks both the search space and the model that
# buildModel constructs. Nothing else needs to change.
MODEL_TYPE = "transformer"


class ClassificationDataset(Dataset):
    def __init__(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
    ):
        self.signal = torch.tensor(signal)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Number of samples (rows), not the length of one signal.
        return len(self.signal)

    def __getitem__(self, idx):
        x = self.signal[idx]
        y = self.labels[idx]

        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


### Transformer Encoder Model
class BaseTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        dim_ff: int,
        n_classes: int,
        num_layers: int,
        n_heads: int,
        n_tokens: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=n_tokens)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model * n_tokens, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_embed = self.input_projection(x)
        x_embed_pe = self.positional_encoding(x_embed)

        z = self.encoder(x_embed_pe)

        # Flatten the output of the encoder
        z_flat = z.flatten(start_dim=1)

        logits = self.output_projection(z_flat)

        return logits

### s/m/xLSTM

#mLSTMCalssifier
class MLSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size:int,
            d_model:int,
            n_classes:int,
            num_blocks:int, # number of layers
            num_head:int,
            context_length:int,
            conv1d_kernel_size: int,
            qkv_proj_blocksize:int, 
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size,d_model)

        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    num_heads=num_head,
                    conv1d_kernel_size=conv1d_kernel_size,
                    qkv_proj_blocksize=qkv_proj_blocksize
                )
            ),
            context_length=context_length, 
            num_blocks=num_blocks,
            embedding_dim=d_model, 

        )

        self.mlstm = xLSTMBlockStack(xlstm_config)

        #Classifiaction head
        self.output_projection=nn.Linear(d_model*context_length, n_classes)


    def forward(self, x:torch.tensor)-> torch.Tensor:
        #(bs,n_timesteps, n_features)
        x= self.input_projection(x) # (bs, n_timestamps, d_model)

        z = self.mlstm(x) # (bs, n_timesteps, d_model)

        z_flat=z.flatten(start_dim=1) #(bs,n_timesteps*d_model)

        output=self.output_projection(z_flat) # (bs, n_classes)

        return output


#sLTSMClassifier
class SLSTMClassifier(nn.Module):
    def __init__(
            self,
            input_size:int,
            d_model:int,
            n_classes:int,
            num_blocks:int, # number of layers
            num_head:int,
            context_length:int,
            conv1d_kernel_size: int,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size,d_model)

        xlstm_config = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_head,
                    conv1d_kernel_size=conv1d_kernel_size,
                    backend="vanilla" #vanilla = Pytorch GPU allocation, cuda = custom cuda kernel
            ),
            ),
            context_length=context_length, 
            num_blocks=num_blocks, 
            embedding_dim=d_model, 
    
        )

        self.slstm = xLSTMBlockStack(xlstm_config) 

        #Classifiaction head
        self.output_projection=nn.Linear(d_model*context_length, n_classes)

    def forward(self, x:torch.tensor)-> torch.Tensor:
        #(bs,n_timesteps, n_features)
        x= self.input_projection(x) # (bs, n_timestamps, d_model)

        z = self.slstm(x) # (bs, n_timesteps, d_model)

        z_flat=z.flatten(start_dim=1) #(bs,n_timesteps*d_model)

        output=self.output_projection(z_flat) # (bs, n_classes)

        return output

### xLSTM
class XLSTMClassifier(nn.Module):
    def __init__(
                self,
                input_size:int,
                d_model:int,
                n_classes:int,
                num_blocks:int, # number of layers
                num_head:int,
                context_length:int,
                conv1d_kernel_size: int,
                qkv_proj_blocksize:int, 
        ):
            super().__init__()
    
            self.input_projection = nn.Linear(input_size,d_model)
    
            xlstm_config = xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(
                    mlstm=mLSTMLayerConfig(
                        num_heads=num_head,
                        conv1d_kernel_size=conv1d_kernel_size,
                        qkv_proj_blocksize=qkv_proj_blocksize
                    )
                ),

                slstm_block=sLSTMBlockConfig(
                    slstm=sLSTMLayerConfig(
                        num_heads=num_head,
                        conv1d_kernel_size=conv1d_kernel_size,
                        backend="vanilla" #vanilla = Pytorch GPU allocation, cuda = custom cuda kernel ) 
                    ),
                ),

                context_length=context_length, 
                num_blocks=num_blocks, 
                embedding_dim=d_model,
                slstm_at=[1]
    
            )
    
            self.xlstm = xLSTMBlockStack(xlstm_config) 
    
            #Classifiaction head
            self.output_projection=nn.Linear(d_model*context_length, n_classes)
    
    def forward(self, x:torch.tensor)-> torch.Tensor:
        #(bs,n_timesteps, n_features)
        x= self.input_projection(x) # (bs, n_timestamps, d_model)

        z = self.xlstm(x) # (bs, n_timesteps, d_model)

        z_flat=z.flatten(start_dim=1) #(bs,n_timesteps*d_model)

        output=self.output_projection(z_flat) # (bs, n_classes)

        return output

def chooseTypeLSTM(opt, input_size,n_classes,n_timesteps, device:torch.device):
    if(opt =="mLSTM" ): 
        return MLSTMClassifier(
            input_size=input_size,
            n_classes=n_classes,
            d_model=D_MODEL,
            num_blocks=NUM_BLOCKS,
            num_head=NUM_HEADS,
            context_length=n_timesteps,
            conv1d_kernel_size=CONV1D_KERNEL_SIZE,
            qkv_proj_blocksize=QVK_PROJ_BLOCKSIZE, # Only for MLSTM
        ).to(device)
    elif (opt =="sLSTM"): 
        return SLSTMClassifier(
            input_size=input_size,
            n_classes=n_classes,
            d_model=D_MODEL,
            num_blocks=NUM_BLOCKS,
            num_head=NUM_HEADS,
            context_length=n_timesteps,
            conv1d_kernel_size=CONV1D_KERNEL_SIZE,
        ).to(device) 
    else: 
        print("Using default model xLSTM")
        return XLSTMClassifier(
            input_size=input_size,
            n_classes=n_classes,
            d_model=D_MODEL,
            num_blocks=2, #we have to have min 2 blocks because we have an slstm/ mlstm
            num_head=NUM_HEADS,
            context_length=n_timesteps,
            conv1d_kernel_size=CONV1D_KERNEL_SIZE,
            qkv_proj_blocksize=QVK_PROJ_BLOCKSIZE, # Only for MLSTM
        ).to(device)
        

def getCriterion(y_train, device):
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(class_weights)
    return nn.CrossEntropyLoss(weight=class_weights_tensor)

def getLoaders(X_train_scaled, X_val_scaled, X_test_scaled,
               y_train, y_val, y_test):
    
    train_dataset = ClassificationDataset(
        signal=X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_train,
    )

    val_dataset = ClassificationDataset(
        signal=X_val_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_val,
    )

    test_dataset = ClassificationDataset(
        signal=X_test_scaled.reshape(-1, X_train_scaled.shape[1], 1),
        labels=y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def performTrainingLoop(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
) -> None:

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    time_per_epoch = []

    for epoch in range(EPOCHS):
        start_time = time.time() # Start time per epoch

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

        val_loss = 0.0
        model.eval()
        with torch.no_grad(): #don't track gradients for optimizer, we don't want to learn validation
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(x_batch)

                loss = criterion(
                    predictions,
                    y_batch,
                )

                #No backward pass = no learning
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        end_time = time.time() #timestamp 
        time_per_epoch.append(end_time - start_time) # Appends the time per epoch


        #save best validation loss use later; protects against overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")


        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f}"
            f" | "
            f"Val loss: {val_loss:.8f}"
            f" | "
            f"Best Val loss: {best_val_loss:.8f}"
            f" | "
            f"Best Val epoch: {best_val_epoch + 1}"
        )

        if epoch == best_val_epoch + EARLY_STOP_NO_IMPROVEMENT: #early stopping if we haven't had a best val epoch in EARLY_STOP_NO_IMPROVEMENT epochs
            print("No reduction in validation loss in 10 epochs. Stopping training early.")
            break

    return train_losses, val_losses, time_per_epoch

def plotLossCurve(train_losses, val_losses, save_path=None):
    # Best epoch = lowest val loss; this is also the epoch performTrainingLoop
    # checkpoints to best_model.pth, so marking it ties the plot back to the
    # model actually used for testing.
    best_epoch = int(np.argmin(val_losses))
    best_val_loss = val_losses[best_epoch]

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.axvline(
        best_epoch,
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"Best epoch {best_epoch + 1} (val loss={best_val_loss:.4f})",
    )
    plt.ylabel("log(Loss)", fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

def performInferenceLoop(model, loader, device):
    """Forward-only pass over a loader, returning raw logits and targets.

    Shared by performTestingLoop (argmax predictions) and threshold selection
    (which needs class probabilities instead of a collapsed prediction).
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_logits, all_targets


def performTestingLoop(best_model, test_loader, device):
    all_logits, all_targets = performInferenceLoop(best_model, test_loader, device)

    y_pred = all_logits.argmax(dim=1).numpy()
    y_true = all_targets.numpy()

    return y_pred, y_true, all_logits


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

def loadRawDataFromFile():
    train = np.loadtxt("ECG5000/ECG5000_TRAIN.txt")
    test  = np.loadtxt("ECG5000/ECG5000_TEST.txt")
    return train, test


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

def performMetrics(y_true, y_pred, all_logits, class_labels, prefix, roc_suffix="roc", time_per_epoch=None, decision_thresholds=None, val_true=None, val_logits=None):
    if time_per_epoch is None:
        time_per_epoch = []
        
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    test_precision = precision_score(y_true, y_pred, average="macro")
    print(f"Test Macro-averaged precision: {test_precision:.4f}")

    test_recall = recall_score(y_true, y_pred, average="macro")
    print(f"Test Macro-averaged recall: {test_recall:.4f}")

    test_f1score = f1_score(y_true, y_pred, average="macro")
    print(f"Test Macro-averaged F1_score: {test_f1score:.4f}")

    test_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"Test balanced accuracy: {test_balanced_accuracy:.4f}")

    if time_per_epoch:
        average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
        print(f"Average time per epoch: {average_time_per_epoch:.4f}s")

    print("\nClassification reports:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    ### Plot the results
    # Row-normalized panel shows what fraction of each true class landed in
    # each predicted bucket, which raw counts hide when classes are imbalanced
    # (e.g. mc_cf.png: class 1 only has ~43 test samples total).
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
    plt.savefig(f"{prefix}_cf.png", dpi=150, bbox_inches="tight")
    plt.show()

    ### AUROC / AUPRC
    # Class scores (probabilities) are needed for threshold-based metrics
    # like AUROC/AUPRC - argmax predictions (y_pred) collapse that info away.
    y_score = torch.softmax(all_logits, dim=1).numpy()
    n_classes = len(class_labels)
    # label_binarize collapses to a single column for exactly 2 classes, which
    # breaks the one-vs-rest machinery below, so binary needs its own path.
    is_binary = n_classes == 2

    def _plotRocPr(y_true_, y_score_, ax_roc_, ax_pr_, annotate_threshold=True):
        """Draw one-vs-rest ROC/PR curves, optionally with decision-threshold
        markers on the PR axes. Markers are recomputed from y_true_/y_score_ -
        the same arrays that produce the curves - so they always land exactly
        on the plotted line, rather than being reused from a different split.
        """
        if is_binary:
            pos_score_ = y_score_[:, 1]
            macro_auroc_ = roc_auc_score(y_true_, pos_score_)
            macro_auprc_ = average_precision_score(y_true_, pos_score_)
        else:
            y_true_bin_ = label_binarize(y_true_, classes=class_labels)
            macro_auroc_ = roc_auc_score(y_true_, y_score_, average="macro", multi_class="ovr")
            macro_auprc_ = average_precision_score(y_true_bin_, y_score_, average="macro")

        operating_points_ = []

        if is_binary:
            fpr, tpr, _ = roc_curve(y_true_, pos_score_)
            ax_roc_.plot(fpr, tpr, label=f"Class {int(class_labels[1]) + 1} (AUROC={macro_auroc_:.3f})")

            precision, recall, _ = precision_recall_curve(y_true_, pos_score_)
            ax_pr_.plot(recall, precision, label=f"Class {int(class_labels[1]) + 1} (AUPRC={macro_auprc_:.3f})")

            if annotate_threshold and decision_thresholds is not None:
                thr = float(decision_thresholds)
                y_at_thr = (pos_score_ >= thr).astype(np.int64)
                op_precision = precision_score(y_true_, y_at_thr, zero_division=0)
                op_recall = recall_score(y_true_, y_at_thr, zero_division=0)
                operating_points_.append(
                    (op_recall, op_precision, f"Class {int(class_labels[1]) + 1} thr={thr:.3f}")
                )
        else:
            for i, class_label in enumerate(class_labels):
                fpr, tpr, _ = roc_curve(y_true_bin_[:, i], y_score_[:, i])
                class_auroc = roc_auc_score(y_true_bin_[:, i], y_score_[:, i])
                ax_roc_.plot(fpr, tpr, label=f"Class {int(class_label) + 1} (AUROC={class_auroc:.3f})")

                precision, recall, _ = precision_recall_curve(y_true_bin_[:, i], y_score_[:, i])
                class_auprc = average_precision_score(y_true_bin_[:, i], y_score_[:, i])
                ax_pr_.plot(recall, precision, label=f"Class {int(class_label) + 1} (AUPRC={class_auprc:.3f})")

                if annotate_threshold and decision_thresholds is not None:
                    thr = float(decision_thresholds[i])
                    y_at_thr = (y_score_[:, i] >= thr).astype(np.int64)
                    op_precision = precision_score(y_true_bin_[:, i], y_at_thr, zero_division=0)
                    op_recall = recall_score(y_true_bin_[:, i], y_at_thr, zero_division=0)
                    operating_points_.append(
                        (op_recall, op_precision, f"Class {int(class_label) + 1} thr={thr:.3f}")
                    )

        ax_roc_.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax_roc_.set_xlabel("False Positive Rate")
        ax_roc_.set_ylabel("True Positive Rate")
        ax_roc_.set_title(f"ROC curves (macro AUROC={macro_auroc_:.3f})")
        ax_roc_.legend(fontsize=8)
        ax_roc_.grid(linestyle="dashed")

        # Mark the threshold(s) chosen on the validation set, so the PR curve
        # shows where the deployed decision rule actually sits on the tradeoff.
        for j, (op_recall, op_precision, op_label) in enumerate(operating_points_):
            ax_pr_.scatter(
                op_recall, op_precision,
                marker="o", s=40, color="black", zorder=5,
                label="Selected threshold" if j == 0 else None,
            )
            ax_pr_.annotate(
                op_label, (op_recall, op_precision),
                textcoords="offset points", xytext=(6, -8), fontsize=7,
            )

        ax_pr_.set_xlabel("Recall")
        ax_pr_.set_ylabel("Precision")
        ax_pr_.set_title(f"Precision-Recall curves (macro AUPRC={macro_auprc_:.3f})")
        ax_pr_.legend(fontsize=8)
        ax_pr_.grid(linestyle="dashed")

        return macro_auroc_, macro_auprc_

    # Plot one-vs-rest ROC and Precision-Recall curves, one line per class,
    # for the test set (the split the reported metrics come from). No
    # threshold markers here - the decision threshold was chosen on the
    # validation set, not this curve, so a marker here doesn't reflect an
    # actual decision made from this data.
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    macro_auroc, macro_auprc = _plotRocPr(y_true, y_score, ax_roc, ax_pr, annotate_threshold=False)
    fig.suptitle("Test set")

    print(f"\nMacro-average AUROC: {macro_auroc:.4f}")
    print(f"Macro-average AUPRC: {macro_auprc:.4f}")

    plt.tight_layout()
    plt.savefig(f"{prefix}_{roc_suffix}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Also plot the validation-set curves, when given - the decision
    # threshold(s) were chosen on validation data, so that's the only curve
    # where a threshold marker reflects an actual decision made from the data.
    if val_true is not None and val_logits is not None:
        y_val_score = torch.softmax(val_logits, dim=1).numpy()
        fig_val, (ax_roc_val, ax_pr_val) = plt.subplots(1, 2, figsize=(12, 5))
        val_auroc, val_auprc = _plotRocPr(val_true, y_val_score, ax_roc_val, ax_pr_val)
        fig_val.suptitle("Validation set (threshold selection)")

        print(f"\nValidation macro AUROC: {val_auroc:.4f}")
        print(f"Validation macro AUPRC: {val_auprc:.4f}")

        plt.tight_layout()
        plt.savefig(f"{prefix}_{roc_suffix}_val.png", dpi=150, bbox_inches="tight")
        plt.show()




### Optuna-tunable xLSTM

class TunableXLSTMClassifier(nn.Module):
    """One xLSTM classifier covering all three block layouts.

    MLSTMClassifier / SLSTMClassifier / XLSTMClassifier above each hard-code a
    single layout and a flatten head. Optuna needs to move between layouts and
    head types inside one search space, so this class takes both as arguments:

      block_mix  "mlstm" -> every block is an mLSTM block
                 "mixed" -> mLSTM blocks with one sLSTM block last
                 "slstm" -> every block is an sLSTM block
      pooling    "last" / "mean" / "flatten" - how (B, T, d_model) collapses to
                 a single vector before the classification head.

    No positional encoding: xLSTM is recurrent and causal, so position is
    already implicit in the scan.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_classes: int,
        n_tokens: int,
        num_blocks: int,
        num_heads: int,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
        proj_factor: float,
        dropout: float,
        block_mix: str,
        pooling: str,
    ):
        super().__init__()
        self.pooling = pooling

        # "mixed" needs somewhere to put the mLSTM block; with a single block it
        # would silently collapse to an all-sLSTM stack, so name it as such.
        if block_mix == "mixed" and num_blocks < 2:
            block_mix = "mlstm"
        self.block_mix = block_mix

        self.input_projection = nn.Linear(input_size, d_model)

        slstm_at = {"mlstm": [], "mixed": [num_blocks - 1], "slstm": "all"}[block_mix]

        # Built fresh every time: xLSTMBlockStackConfig.__post_init__ mutates the
        # nested block configs (embedding_dim, dropout, context_length), so a
        # config object must never be shared across models or trials.
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=None if block_mix == "slstm" else mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size,
                    qkv_proj_blocksize=qkv_proj_blocksize,
                    num_heads=num_heads,
                    proj_factor=proj_factor,
                )
            ),
            slstm_block=None if block_mix == "mlstm" else sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    # Both are required on a CPU-only machine: sLSTMCellConfig
                    # defaults to backend="cuda" (which JIT-compiles CUDA
                    # kernels) and dtype="bfloat16".
                    backend="vanilla",
                    dtype="float32",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=n_tokens,  # sizes the mLSTM causal mask - must be the seq len
            num_blocks=num_blocks,
            embedding_dim=d_model,
            dropout=dropout,
            # Must be passed here, not assigned afterwards: __post_init__ turns
            # it into the block map that decides which block class goes where.
            slstm_at=slstm_at,
        )

        self.encoder = xLSTMBlockStack(xlstm_config)

        head_in = d_model * n_tokens if pooling == "flatten" else d_model
        self.output_projection = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, input_size) -> (B, T, d_model)
        z = self.encoder(self.input_projection(x))

        if self.pooling == "last":
            z = z[:, -1, :]
        elif self.pooling == "mean":
            z = z.mean(dim=1)
        else:
            z = z.flatten(start_dim=1)

        return self.output_projection(z)


def suggestXLSTMHyperparams(trial):
    """Search space for TunableXLSTMClassifier + training hyperparameters.

    Every combination below is valid by construction. The mLSTM inner dim is
    round_up(proj_factor * d_model, multiple_of=64), so num_heads and
    qkv_proj_blocksize always divide it; sLSTM sets hidden_size = d_model, and
    both 2 and 4 divide each of 32/64/128.

    Ordered roughly by expected impact on this dataset (500 training samples,
    140 timesteps, heavy class imbalance) - regularisation and head size matter
    more here than raw capacity.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "pooling": trial.suggest_categorical("pooling", ["last", "mean", "flatten"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "block_mix": trial.suggest_categorical("block_mix", ["mlstm", "mixed", "slstm"]),
        "conv1d_kernel_size": trial.suggest_categorical("conv1d_kernel_size", [2, 4, 8]),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
        "proj_factor": trial.suggest_categorical("proj_factor", [1.0, 2.0]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        # Fixed rather than searched: it is effectively a re-parameterisation of
        # head count, and 25 trials is already thin for the dimensions above.
        # Promote to [2, 4, 8] if the search budget grows.
        "qkv_proj_blocksize": 4,
    }


def suggestTransformerHyperparams(trial):
    """Search space for BaseTransformerClassifier + training hyperparameters.

    n_heads is drawn from values that evenly divide every d_model choice below
    (32/64/128 are all divisible by 2 and 4), so every sampled combination is a
    valid nn.MultiheadAttention config.

    weight_decay is included so this dict is interchangeable with the xLSTM one
    at every call site - performTrainingLoopOptuna takes it either way.
    """
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_ff": trial.suggest_categorical("dim_ff", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def suggestHyperparams(trial, model_type=MODEL_TYPE):
    """Dispatch to the right search space and tag the result with model_type.

    The tag rides along in the params dict so buildModel knows what to build
    without a second argument threaded through every call site.
    """
    if model_type == "transformer":
        params = suggestTransformerHyperparams(trial)
    elif model_type == "xlstm":
        params = suggestXLSTMHyperparams(trial)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}, expected 'xlstm' or 'transformer'"
        )

    params["model_type"] = model_type
    return params


def resolveBestParams(study, model_type=MODEL_TYPE):
    """Rebuild the full param dict for the winning trial.

    study.best_params only contains values that went through a trial.suggest_*
    call, so anything injected (model_type) or fixed (qkv_proj_blocksize) is
    missing from it and has to be put back before buildModel sees it.
    """
    best_params = dict(study.best_params)
    best_params["model_type"] = model_type
    best_params.setdefault("qkv_proj_blocksize", QVK_PROJ_BLOCKSIZE)
    return best_params

def buildModel(params, input_size, n_classes, n_tokens, device):
    """Single choke point for model construction.

    Called three times per run with the same params (trial, retrain, reload), so
    every architectural knob comes from params rather than module constants -
    otherwise the reloaded model would not match the saved state_dict.

    Dispatches on params["model_type"], which suggestHyperparams tags onto the
    dict. Defaults to "xlstm" when the tag is absent.
    """
    if params.get("model_type", "xlstm") == "transformer":
        return BaseTransformerClassifier(
            input_size=input_size,
            d_model=params["d_model"],
            dim_ff=params["dim_ff"],
            n_classes=n_classes,
            num_layers=params["num_layers"],
            n_heads=params["n_heads"],
            n_tokens=n_tokens,
        ).to(device)

    return TunableXLSTMClassifier(
        input_size=input_size,
        d_model=params["d_model"],
        n_classes=n_classes,
        n_tokens=n_tokens,
        num_blocks=params["num_blocks"],
        num_heads=params["num_heads"],
        conv1d_kernel_size=params["conv1d_kernel_size"],
        qkv_proj_blocksize=params.get("qkv_proj_blocksize", QVK_PROJ_BLOCKSIZE),
        proj_factor=params["proj_factor"],
        dropout=params["dropout"],
        block_mix=params["block_mix"],
        pooling=params["pooling"],
    ).to(device)


def getLoadersOptuna(X_train_scaled, X_val_scaled, X_test_scaled,
                     y_train, y_val, y_test, batch_size,
                     eval_batch_size=None):
    """Same as getLoaders, but batch_size is an argument so Optuna can tune it,
    and val/test get their own larger batch size - they never backprop, and the
    val set is evaluated every single epoch."""
    eval_bs = eval_batch_size or max(batch_size, EVAL_BATCH_SIZE)

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
    val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False)

    return train_loader, val_loader, test_loader


def performTrainingLoopOptuna(model, train_loader, val_loader, device, criterion,
                              lr, epochs, early_stop_patience,
                              checkpoint_path=None, verbose=True,
                              trial=None, weight_decay=0.0,
                              label=None, leave=None):
    """Train/validate loop with lr, epochs, patience, checkpoint path and weight
    decay as arguments instead of module constants.

    Pass trial=... to report intermediate val loss to Optuna, so unpromising
    trials are pruned partway through instead of burning the full epoch budget.
    Trials pass checkpoint_path=None - they only need the val loss, not weights.

    label/leave override the tqdm bar's caption and leave-behind behaviour.
    Left as None, they fall back to the trial-derived defaults below - callers
    that train once per fold (no per-epoch pruning, trial=None) pass their own
    label so the bar still identifies which trial/fold is running.
    """
    # AdamW rather than Adam so weight decay is decoupled from the gradient;
    # weight_decay=0.0 makes this identical to performTrainingLoop's Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0

    # One bar per trial, showing elapsed time, s/epoch and ETA so the cost of a
    # trial is visible while it runs. Trial bars clear themselves on completion
    # (leave=False) so the log does not fill up with 25 finished bars; the final
    # retrain keeps its bar.
    if label is None:
        label = f"Trial {trial.number + 1}/{N_TRIALS}" if trial is not None else "Final training"
    if leave is None:
        leave = trial is None
    bar = tqdm(
        range(epochs),
        desc=label,
        unit="ep",
        leave=leave,
        dynamic_ncols=True,
        disable=not SHOW_PROGRESS,
    )

    started = time.perf_counter()
    status = "completed"
    epochs_run = 0

    try:
        for epoch in bar:
            epochs_run = epoch + 1
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
            with torch.inference_mode():
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

            bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}@{best_val_epoch + 1}",
            )

            if verbose:
                # bar.write, not print - print would tear the bar in half.
                bar.write(
                    f"Epoch: {epoch + 1:3d} | "
                    f"Train loss: {train_loss:.8f} | "
                    f"Val loss: {val_loss:.8f} | "
                    f"Best Val loss: {best_val_loss:.8f} | "
                    f"Best Val epoch: {best_val_epoch + 1}"
                )

            # Report before the early-stop check, so a trial that stops early
            # still contributes its final value to the pruner's statistics.
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    status = "pruned"
                    raise optuna.TrialPruned()

            if epoch - best_val_epoch >= early_stop_patience:
                status = "early-stopped"
                if verbose:
                    bar.write(
                        f"No reduction in validation loss in {early_stop_patience} "
                        "epochs. Stopping training early."
                    )
                break
    finally:
        bar.close()
        elapsed = time.perf_counter() - started
        per_epoch = elapsed / max(epochs_run, 1)
        # Printed for every trial, pruned ones included, so the per-trial cost
        # stays on the record after the bar has cleared itself.
        print(
            f"{label}: {status} after {epochs_run} epochs in {elapsed:.1f}s "
            f"({per_epoch:.2f}s/epoch) | best val loss "
            f"{best_val_loss:.6f} @ epoch {best_val_epoch + 1}",
            flush=True,
        )

    return train_losses, val_losses


def _loader(X, y, batch_size, shuffle):
    dataset = ClassificationDataset(
        signal=X.reshape(-1, X.shape[1], 1).astype(np.float32), labels=y
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def makeFolds(X, y):
    """Create the five disjoint folds reused by every Optuna trial."""
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    return list(splitter.split(X, y))


def crossValidatedScore(params, X_development, y_development, folds, device, trial):
    """Return mean five-fold validation loss and all fold-level scores.

    Each fold trains with trial=None (no per-epoch pruning): performTrainingLoopOptuna
    reports (val_loss, epoch) to trial.report(), and epoch resets to 0 every fold,
    so reporting across folds on the same trial would violate Optuna's requirement
    that a trial's reported steps increase monotonically.
    """
    fold_scores = []
    n_classes = len(np.unique(y_development))
    n_tokens = X_development.shape[1]
    for fold_number, (train_index, val_index) in enumerate(folds, start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_development[train_index]).astype(np.float32)
        X_val = scaler.transform(X_development[val_index]).astype(np.float32)
        y_train, y_val = y_development[train_index], y_development[val_index]

        # Duplicate minority-class rows in the training fold only; the
        # validation fold keeps its natural distribution so the CV score
        # still reflects real-world performance.
        ros = RandomOverSampler(random_state=RANDOM_STATE)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        setSeed(RANDOM_STATE + fold_number)
        train_loader = _loader(X_train, y_train, params["batch_size"], True)
        val_loader = _loader(X_val, y_val, params["batch_size"], False)
        model = buildModel(params, 1, n_classes, n_tokens, device)
        _, val_losses = performTrainingLoopOptuna(
            model,
            train_loader,
            val_loader,
            device,
            # RandomOverSampler already balances the training distribution;
            # also class-weighting the loss would over-correct.
            nn.CrossEntropyLoss(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            epochs=HPO_EPOCHS,
            early_stop_patience=HPO_EARLY_STOP_NO_IMPROVEMENT,
            verbose=False,
            trial=None,
            label=f"Trial {trial.number + 1}/{N_TRIALS} fold {fold_number}/{N_SPLITS}",
            leave=False,
        )
        score = float(min(val_losses))
        fold_scores.append(score)
        print(f"  fold {fold_number}/{N_SPLITS}: validation loss={score:.6f}")
    return float(np.mean(fold_scores)), fold_scores


def runCrossValidatedModelSelection(task, model_type, output_prefix, roc_suffix):
    """Select the winning hyperparameters via 5-fold CV over a 40% dev pool,
    refit on a fresh split of that pool, then test once on the held-out 60%.
    """
    setSeed(RANDOM_STATE)
    torch.set_num_threads(NUM_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model type:", model_type)

    X, y = loadTaskData(task)

    # The test set is reserved before CV and never passed to an objective.
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Development pool shape: {X_development.shape} (used for {N_SPLITS}-fold CV)")
    print(f"Held-out test shape: {X_test.shape}")
    folds = makeFolds(X_development, y_development)

    def objective(trial):
        params = suggestHyperparams(trial, model_type)
        print(f"\nTrial {trial.number + 1}/{N_TRIALS}: {params}")
        mean_score, fold_scores = crossValidatedScore(
            params, X_development, y_development, folds, device, trial
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_std", float(np.std(fold_scores, ddof=1)))
        print(
            f"  mean validation loss={mean_score:.6f} +/- "
            f"{trial.user_attrs['fold_std']:.6f}"
        )
        return mean_score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = resolveBestParams(study, model_type)
    print("\nSelected lambda*:", best_params)
    print(f"Mean five-fold validation loss: {study.best_value:.6f}")

    # Fresh train/validation division of the dev pool, matching the 5-fold
    # proportions used during CV (test_size = 1/N_SPLITS).
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=1 / N_SPLITS,
        stratify=y_development,
        random_state=RANDOM_STATE + 1,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Duplicate minority-class rows in the training set only. The
    # validation and test sets retain their natural class distributions.
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_scaled, y_train_ros = ros.fit_resample(X_train_scaled, y_train)
    print(
        "Training class counts before/after oversampling:",
        np.bincount(y_train), "->", np.bincount(y_train_ros),
    )

    input_size = 1
    n_classes = len(np.unique(y))
    n_tokens = X.shape[1]

    train_loader, val_loader, test_loader = getLoadersOptuna(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train_ros, y_val, y_test,
        batch_size=best_params["batch_size"],
    )

    model = buildModel(best_params, input_size, n_classes, n_tokens, device)
    # RandomOverSampler already balances the training distribution; also
    # class-weighting the loss would over-correct.
    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Best model has {num_params} trainable parameters.")

    checkpoint_path = f"{output_prefix}_best_model.pth"
    train_losses, val_losses = performTrainingLoopOptuna(
        model, train_loader, val_loader, device, criterion,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_NO_IMPROVEMENT,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    plotLossCurve(train_losses, val_losses, save_path=f"{output_prefix}_loss.png")

    best_model = buildModel(best_params, input_size, n_classes, n_tokens, device)
    best_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )

    class_labels = np.unique(y)
    is_binary = len(class_labels) == 2

    # Threshold is selected on the validation set (natural class distribution,
    # never oversampled or trained on) and then frozen before it ever touches
    # the test set.
    val_logits, val_targets = performInferenceLoop(best_model, val_loader, device)
    y_val_true = val_targets.numpy()
    y_val_score = torch.softmax(val_logits, dim=1).numpy()

    test_logits, test_targets = performInferenceLoop(best_model, test_loader, device)
    y_true = test_targets.numpy()
    y_test_score = torch.softmax(test_logits, dim=1).numpy()

    if is_binary:
        threshold, best_fbeta, thr_precision, thr_recall = chooseThresholdMaxFbeta(
            y_val_true, y_val_score[:, 1], beta=BINARY_THRESHOLD_BETA
        )
        print(
            f"\nValidation-selected decision threshold "
            f"(F{BINARY_THRESHOLD_BETA:g}-optimal, recall-prioritized): "
            f"{threshold:.4f} (F{BINARY_THRESHOLD_BETA:g}={best_fbeta:.4f}, "
            f"precision={thr_precision:.4f}, recall={thr_recall:.4f})"
        )
        y_pred = (y_test_score[:, 1] >= threshold).astype(np.int64)
        threshold_info = {"beta": BINARY_THRESHOLD_BETA, "threshold": threshold}
        decision_thresholds = threshold
    else:
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
        y_pred = applyMulticlassThresholds(y_test_score, thresholds)
        threshold_info = {"beta": MULTICLASS_THRESHOLD_BETA, "thresholds": thresholds.tolist()}
        decision_thresholds = thresholds

    test_score = float(balanced_accuracy_score(y_true, y_pred))
    performMetrics(
        y_true, y_pred, test_logits, class_labels,
        prefix=output_prefix, roc_suffix=roc_suffix,
        decision_thresholds=decision_thresholds,
        val_true=y_val_true, val_logits=val_logits,
    )

    report = {
        "task": task,
        "model_type": model_type,
        "lambda_star": best_params,
        "mean_cv_validation_loss": study.best_value,
        "best_fold_scores": study.best_trial.user_attrs["fold_scores"],
        "decision_threshold": threshold_info,
        "test_performance_indicator": "balanced_accuracy",
        "test_balanced_accuracy": test_score,
        "split_sizes": {
            "development": len(y_development),
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
    }
    Path(f"{output_prefix}_cv_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
