"""Command-line entry point for five-fold ECG5000 model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cross_validation import DEFAULT_PARAM_GRID, run_model_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select lambda* by 5-fold CV, refit, and test once."
    )
    parser.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--data-dir", type=Path, default=Path("ECG5000"))
    parser.add_argument("--output-dir", type=Path, default=Path("cross_validation_results"))
    parser.add_argument(
        "--grid",
        type=Path,
        help="Optional JSON object (or list of objects) accepted by sklearn ParameterGrid.",
    )
    parser.add_argument("--cv-epochs", type=int, default=60)
    parser.add_argument("--cv-patience", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=300)
    parser.add_argument("--final-patience", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = DEFAULT_PARAM_GRID
    if args.grid:
        grid = json.loads(args.grid.read_text(encoding="utf-8"))
    run_model_selection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task=args.task,
        param_grid=grid,
        cv_epochs=args.cv_epochs,
        cv_patience=args.cv_patience,
        final_epochs=args.final_epochs,
        final_patience=args.final_patience,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
