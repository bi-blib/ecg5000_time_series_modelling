from helpers import MODEL_TYPE, runHPOModelSelection


def main():
    prefix = f"mc_{MODEL_TYPE}_optuna"
    return runHPOModelSelection("multiclass", MODEL_TYPE, prefix, "roc")


if __name__ == "__main__":
    main()
