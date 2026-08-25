from helpers import MODEL_TYPE, runCrossValidatedModelSelection


def main():
    prefix = f"mc_{MODEL_TYPE}_optuna"
    return runCrossValidatedModelSelection("multiclass", MODEL_TYPE, prefix, "roc")


if __name__ == "__main__":
    main()
