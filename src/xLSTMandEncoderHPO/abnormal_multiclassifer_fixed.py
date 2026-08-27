from config import MODEL_TYPE
from model_selection import runFixedParamsModelSelection

# Hand-picked starting hyperparameters, one dict per model_type so this file
# keeps working if MODEL_TYPE is flipped in helpers.py without editing here.
# These are rough midpoints of the Optuna search spaces in
# suggestTransformerHyperparams / suggestXLSTMHyperparams (helpers.py) - NOT
# tuned. Replace with your own study's "lambda_star" (see the
# *_optuna_cv_results.json this same task produces via runner_optuna.py)
# once you have HPO results worth trusting. Duplicated deliberately from
# normal_abnormal_binary_classifier_fixed.py's FIXED_PARAMS - each task file
# is meant to be independently hand-tuned.
FIXED_PARAMS = {
    "transformer": {
        "model_type": "transformer",
        "d_model": 64,
        "n_heads": 2,
        "num_layers": 1,
        "dim_ff": 256,
        "lr": 9.324140221663475e-05,
        "weight_decay": 0.00011889469769530483,
        "batch_size": 128,
    },
    "xlstm": {
        "model_type": "xlstm",
        "d_model": 64,
        "num_blocks": 2,
        "num_heads": 4,
        "conv1d_kernel_size": 4,
        "qkv_proj_blocksize": 4,
        "proj_factor": 2.0,
        "dropout": 0.1,
        "block_mix": "mixed",
        "pooling": "mean",
        "lr": 7e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
    },
}


def main():
    prefix = f"mc_{MODEL_TYPE}_fixed"
    return runFixedParamsModelSelection(
        "multiclass", MODEL_TYPE, prefix, "roc", FIXED_PARAMS[MODEL_TYPE]
    )


if __name__ == "__main__":
    main()
