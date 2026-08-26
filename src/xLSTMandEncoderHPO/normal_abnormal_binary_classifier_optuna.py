from config import MODEL_TYPE
from model_selection import runHPOModelSelection


def main():
    prefix = f"b_{MODEL_TYPE}_optuna"
    return runHPOModelSelection("binary", MODEL_TYPE, prefix, "auc")


if __name__ == "__main__":
    main()
