"""MantisV2 for RacketSports time-series classification.

Two variants are compared:

1) MantisV2FrozenClassifier
   - Pretrained MantisV2 encoder is frozen.
   - Only the classification output layer is trained.

2) MantisV2FineTunedClassifier
   - Pretrained MantisV2 encoder is trainable.
   - Encoder and classification output layer are trained.

Both models use exactly the same final CLS-token representation.
"""

import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from scipy.io.arff import loadarff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from torch.utils.data import Dataset, DataLoader

from mantis.architecture import MantisV2

# Repo root is two levels up from this script (teaching_material/scripts/).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_DATASET_DIR = os.path.join(
    _REPO_ROOT,
    "teaching_material",
    "datasets",
    "RacketSports",
)

_TRAIN_PATH = os.path.join(_DATASET_DIR, "RacketSports_TRAIN.arff")
_TEST_PATH = os.path.join(_DATASET_DIR, "RacketSports_TEST.arff")

# Anchored to the script directory so the checkpoint location does not
# depend on the process's current working directory.
_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pth",
)


def save_checkpoint_safely(
    state_dict: dict,
    path: str,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> None:
    """Save a model checkpoint, tolerating transient Windows file locks.

    Saves to a temporary file first and atomically replaces the target,
    so a crash or lock never leaves a corrupted checkpoint behind.
    A file that is briefly locked by a virus scanner, sync client (e.g.
    OneDrive), or another process is retried instead of failing training.
    """

    tmp_path = f"{path}.tmp-{os.getpid()}"

    for attempt in range(max_retries):

        try:
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, path)
            return

        except (RuntimeError, OSError, PermissionError) as error:

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if attempt == max_retries - 1:
                print(
                    f"Warning: could not save checkpoint to {path} "
                    f"after {max_retries} attempts ({error}). "
                    "Continuing training without saving this checkpoint."
                )
                return

            print(
                f"Warning: checkpoint save attempt {attempt + 1} failed "
                f"({error}). Retrying in {retry_delay:.1f}s..."
            )

            time.sleep(retry_delay)



# MantisV2 Models

class MantisV2FrozenClassifier(nn.Module):
    """MantisV2 classifier with frozen pretrained encoder."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        device: torch.device,
        target_length: int = 512,
    ):
        super().__init__()

        self.target_length = target_length

        # Load pretrained MantisV2 encoder.
        #
        # return_transf_layer=-1:
        #     Use the output of the final Transformer layer.
        #
        # output_token="cls_token":
        #     Use the final CLS token as sequence representation.
        self.encoder = MantisV2(
            device=str(device),
            return_transf_layer=-1,
            output_token="cls_token",
        )

        self.encoder = self.encoder.from_pretrained(
            "paris-noah/MantisV2"
        )

        # Freeze all pretrained encoder parameters.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        # MantisV2 CLS embedding dimension = 256.
        embedding_dim = self.encoder.hidden_dim

        # Each sensor channel is encoded independently.
        #
        # RacketSports:
        # 6 channels * 256 embeddings = 1536 features.
        self.output_projection = nn.Linear(
            input_size * embedding_dim,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input:
        # (B, T, C)
        #
        # RacketSports:
        # (B, 30, 6)

        # Change to channel-first representation
        x = x.transpose(1, 2)   # (B, 30, 6) -> (B, 6, 30)

        # Resize signal length. Recommendation by the authors, because the mode model was trained on a sequence length of 512
        x = F.interpolate(
            x,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        ) # (B, 30, 6) -> (B, 6, 512)

        # Move channels into batch dimension
        batch_size, n_channels, seq_len = x.shape

        x = x.reshape(
            batch_size * n_channels,
            1,
            seq_len,
        ) # (B, 6, 512) -> (B*6, 1, 512)


        # Frozen MantisV2 encoder
        # model.train() from the outer training loop would otherwise put the encoder into training mode.
        self.encoder.eval()

        with torch.no_grad():
            embeddings = self.encoder(x) # (B*6, 256)

        # Restore channel dimension
        embeddings = embeddings.reshape(
            batch_size,
            n_channels,
            -1,
        ) # (B*6, 256) -> (B, 6, 256)

        # Concatenate channel embeddings
        embeddings = embeddings.flatten(start_dim=1) # (B, 6, 256) -> (B, 1536)

        # Classification
        output = self.output_projection(embeddings) # (B, 1536) -> (B, 4)

        return output


class MantisV2FineTunedClassifier(nn.Module):
    """MantisV2 classifier with fine-tuned pretrained encoder."""

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        device: torch.device,
        target_length: int = 512,
    ):
        super().__init__()

        self.target_length = target_length

        # Load exactly the same pretrained MantisV2 encoder
        # representation as for the frozen model.
        self.encoder = MantisV2(
            device=str(device),
            return_transf_layer=-1,
            output_token="cls_token",
        )

        self.encoder = self.encoder.from_pretrained(
            "paris-noah/MantisV2"
        )

        # No parameters are frozen.
        # Gradients therefore propagate through MantisV2.

        embedding_dim = self.encoder.hidden_dim

        # RacketSports:
        # 6 channels * 256 embeddings = 1536 features.
        self.output_projection = nn.Linear(
            input_size * embedding_dim,
            n_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input:
        # (B, T, C)
        #
        # RacketSports:
        # (B, 30, 6)

        # Change to channel-first representation
        x = x.transpose(1, 2)   # (B, 30, 6) -> (B, 6, 30)

        # Resize signal length. Recommendation by the authors, because the mode model was trained on a sequence length of 512
        x = F.interpolate(
            x,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        ) # (B, 30, 6) -> (B, 6, 512)

        # Move channels into batch dimension
        batch_size, n_channels, seq_len = x.shape

        x = x.reshape(
            batch_size * n_channels,
            1,
            seq_len,
        ) # (B, 6, 512) -> (B*6, 1, 512)

        embeddings = self.encoder(x) # (B*6, 256)

        # Restore channel dimension
        embeddings = embeddings.reshape(
            batch_size,
            n_channels,
            -1,
        ) # (B*6, 256) -> (B, 6, 256)

        # Concatenate channel embeddings
        embeddings = embeddings.flatten(start_dim=1) # (B, 6, 256) -> (B, 1536)

        # Classification
        output = self.output_projection(embeddings) # (B, 1536) -> (B, 4)

        return output


def load_racket_sports_arff(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the RacketSports dataset from an ARFF file.

    Args:
        file_path:
            Path to the RacketSports dataset.

    Returns:
        x:
            Array of shape
            (n_examples, n_timesteps, n_features).

        y:
            Class labels.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"RacketSports data file not found: {file_path}\n"
            "Download the ARFF files from "
            "https://www.timeseriesclassification.com/description.php?Dataset=RacketSports "
            f"and place them in {_DATASET_DIR}"
        )

    data, metadata = loadarff(file_path)

    relational_data = data["relationalAtt"]
    channel_names = relational_data[0].dtype.names

    x = np.stack(
        [
            np.column_stack(
                [
                    sample[channel]
                    for channel in channel_names
                ]
            )
            for sample in relational_data
        ]
    ).astype(np.float32)

    y = np.array(
        [
            (
                label.decode("utf-8")
                if isinstance(label, (bytes, np.bytes_))
                else str(label)
            )
            for label in data["activity"]
        ]
    )

    # (N, C, T) -> (N, T, C)
    return x.transpose(0, 2, 1), y


def plot_sample(
    data: np.ndarray,
    labels: np.ndarray,
    sample_idx: int,
) -> None:

    sample = data[sample_idx]
    label = labels[sample_idx]

    time_steps = np.arange(sample.shape[0])

    channel_names = [
        "Accelerometer x",
        "Accelerometer y",
        "Accelerometer z",
        "Gyroscope x",
        "Gyroscope y",
        "Gyroscope z",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11, 6),
    )

    axes = axes.ravel()

    for i in range(6):

        axes[i].plot(
            time_steps,
            sample[:, i],
            linewidth=1.8,
        )

        axes[i].set_title(
            channel_names[i],
            fontsize=11,
        )

        axes[i].set_xlabel("Time step")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    fig.suptitle(
        f"Class: {label}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    true_indices: np.ndarray, predicted_indices: np.ndarray
) -> None:
    class_names = [
        "Badminton Clear",
        "Badminton Smash",
        "Squash Forehand",
        "Squash Backhand",
    ]

    cm = confusion_matrix(
        true_indices,
        predicted_indices,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
    )

    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix – RacketSports")
    plt.tight_layout()
    plt.show()


class RacketSportsDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():

    torch.manual_seed(42)

    train_data, train_labels = load_racket_sports_arff(_TRAIN_PATH)

    test_data, test_labels = load_racket_sports_arff(_TEST_PATH)

    print(
        "Train data shape:",
        train_data.shape,
    )

    print(
        "Test data shape:",
        test_data.shape,
    )

    plot_sample(
        train_data,
        train_labels,
        sample_idx=0,
    )

    #### One Hot Encoding of the labels
    ohe = OneHotEncoder()
    train_labels_ohe = ohe.fit_transform(train_labels.reshape(-1, 1)).toarray()
    test_labels_ohe = ohe.transform(test_labels.reshape(-1, 1)).toarray()

    ### Validation split
    train_data, val_data, train_labels_ohe, val_labels_ohe = train_test_split(
        train_data,
        train_labels_ohe,
        test_size=0.1,
        random_state=42,
    )

    scaler = StandardScaler()

    # Fit scaler only on training data.
    train_scaled = scaler.fit_transform(
        train_data.reshape(
            -1,
            train_data.shape[-1],
        )
    ).reshape(
        train_data.shape
    )

    val_scaled = scaler.transform(
        val_data.reshape(
            -1,
            val_data.shape[-1],
        )
    ).reshape(
        val_data.shape
    )

    test_scaled = scaler.transform(
        test_data.reshape(
            -1,
            test_data.shape[-1],
        )
    ).reshape(
        test_data.shape
    )

    train_dataset = RacketSportsDataset(
        data=train_scaled,
        labels=train_labels_ohe,
    )

    val_dataset = RacketSportsDataset(
        data=val_scaled,
        labels=val_labels_ohe,
    )

    test_dataset = RacketSportsDataset(
        data=test_scaled,
        labels=test_labels_ohe,
    )

    bs = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
    )

    input_size = train_data.shape[-1]

    n_classes = train_labels_ohe.shape[1]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # SELECT MODEL HERE

    # Option 1:
    model_class = MantisV2FrozenClassifier

    # Option 2:
    # model_class = MantisV2FineTunedClassifier

    model = model_class(
        input_size=input_size,
        n_classes=n_classes,
        device=device,
    ).to(device)

    model_name = model.__class__.__name__

    print(
        f"\nModel: {model_name}"
    )

    # Parameter count
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    lr = 0.0001

    optimizer = None # To omit the warning
    if model_name == "MantisV2FrozenClassifier":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), # To train only the parameters which require gradients
            lr=lr,
        )

    elif model_name == "MantisV2FineTunedClassifier":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )


    criterion = nn.CrossEntropyLoss()

    epochs = 100

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_val_epoch = 0

    time_per_epoch = []

    for epoch in range(epochs):

        start_time = time.time()

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

        train_losses.append(
            train_loss
        )

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(
                    x_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        val_losses.append(
            val_loss
        )

        end_time = time.time()

        time_per_epoch.append(
            end_time - start_time
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_val_epoch = epoch

            save_checkpoint_safely(
                model.state_dict(),
                _CHECKPOINT_PATH,
            )

        print(
            f"Epoch: {epoch + 1:3d} | "
            f"Train loss: {train_loss:.8f} | "
            f"Val loss: {val_loss:.8f} | "
            f"Best Val loss: {best_val_loss:.8f} | "
            f"Best Val epoch: {best_val_epoch + 1}"
        )

    plt.plot(
        train_losses,
        label="Train Loss",
    )

    plt.plot(
        val_losses,
        label="Validation Loss",
    )

    plt.ylabel(
        "Loss",
        fontsize=12,
    )

    plt.xlabel(
        "Epoch",
        fontsize=12,
    )

    plt.yscale("log")
    plt.grid(linestyle="dashed")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Testing
    # Instantiate exactly the same model class that
    # was used during training.
    best_model = model_class(
        input_size=input_size,
        n_classes=n_classes,
        device=device,
    ).to(device)

    best_model.load_state_dict(torch.load(_CHECKPOINT_PATH))

    best_model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            logits = best_model(x_batch)

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    predicted_indices = all_logits.argmax(dim=1).numpy()
    true_indices = all_targets.argmax(dim=1).numpy()

    ### Metrics
    accuracy = accuracy_score(
        true_indices,
        predicted_indices,
    )

    precision = precision_score(true_indices, predicted_indices, average="macro")
    recall = recall_score(true_indices, predicted_indices, average="macro")
    f1_macro = f1_score(true_indices, predicted_indices, average="macro")

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision (Macro): {precision:.2f}")
    print(f"Recall (Macro: {recall:.2f}")
    print(f"F1 (Macro): {f1_macro:.4f}")

    average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
    print(f"Average time per epoch: {average_time_per_epoch:.4f}s")

    ### Confusion Matrix
    plot_confusion_matrix(true_indices, predicted_indices)

if __name__ == "__main__":
    main()