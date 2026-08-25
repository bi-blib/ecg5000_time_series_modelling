from helpers import MODEL_TYPE, runCrossValidatedModelSelection


def main():
    prefix = f"b_{MODEL_TYPE}_optuna"
    return runCrossValidatedModelSelection("binary", MODEL_TYPE, prefix, "auc")


if __name__ == "__main__":
    main()
