"""Compatibility entry point for the structured ECG5000 MantisV2 workflow."""

try:
    from .runner import main
except ImportError:
    from runner import main


if __name__ == "__main__":
    main()
