"""Abnormal-only encoder with embedded five-fold model selection."""

try:
    from .helpers_optuna import runCrossValidatedModelSelection
except ImportError:
    from ecg5000_time_series_modelling.archive.encoder.helpers_optuna import runCrossValidatedModelSelection


def main():
    return runCrossValidatedModelSelection("multiclass", "mc_optuna", "roc")


if __name__ == "__main__":
    main()
