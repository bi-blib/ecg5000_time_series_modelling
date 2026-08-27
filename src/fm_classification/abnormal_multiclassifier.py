"""Abnormal-only ECG5000 multiclass classification with MantisV2."""

try:
    from .helpers import run_task
except ImportError:
    from helpers import run_task


def main(extractor, device, config):
    return run_task("multiclass", extractor, device, config)
