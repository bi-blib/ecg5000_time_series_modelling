"""Run the ECG5000 MantisV2 binary and/or multiclass workflows."""

import argparse
import json
from pathlib import Path

import torch

try:
    from . import abnormal_multiclassifier as multiclass
    from . import normal_abnormal_binary_classifier as binary
    from .helpers import (
        BATCH_SIZE,
        EARLY_STOP_PATIENCE,
        EMBEDDING_BATCH_SIZE,
        EPOCHS,
        PACKAGE_DIR,
        FrozenMantisFeatureExtractor,
        set_seed,
    )
except ImportError:
    import abnormal_multiclassifier as multiclass
    import normal_abnormal_binary_classifier as binary
    from helpers import (
        BATCH_SIZE,
        EARLY_STOP_PATIENCE,
        EMBEDDING_BATCH_SIZE,
        EPOCHS,
        PACKAGE_DIR,
        FrozenMantisFeatureExtractor,
        set_seed,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("binary", "multiclass", "both"), default="both"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE
    )
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_DIR / "results"),
    )
    return parser.parse_args()


def main():
    config = parse_args()
    set_seed()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading pretrained MantisV2 encoder...", flush=True)
    extractor = FrozenMantisFeatureExtractor(device).to(device)
    print(
        f"Encoder parameters: {sum(p.numel() for p in extractor.parameters()):,} "
        "(frozen)",
        flush=True,
    )

    results = []
    if config.task in ("binary", "both"):
        results.append(binary.main(extractor, device, config))
    if config.task in ("multiclass", "both"):
        results.append(multiclass.main(extractor, device, config))

    combined_path = Path(config.output_dir) / "mantis_frozen_results.json"
    combined_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Combined results: {combined_path}")


if __name__ == "__main__":
    main()
