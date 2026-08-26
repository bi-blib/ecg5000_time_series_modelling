"""Binary normal-versus-abnormal ECG5000 classification with MantisV2."""

try:
    from .helpers import run_task
except ImportError:
    from helpers import run_task


def main(extractor, device, config):
    return run_task("binary", extractor, device, config)
