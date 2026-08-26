from config import MODEL_TYPE
from model_selection import runFixedParamsModelSelection

# Hand-picked starting hyperparameters, one dict per model_type so this file
# keeps working if MODEL_TYPE is flipped in helpers.py without editing here.
# These are rough midpoints of the Optuna search spaces in
# suggestTransformerHyperparams / suggestXLSTMHyperparams (helpers.py) - NOT
# tuned. Replace with your own study's "lambda_star" (see the
# *_optuna_cv_results.json this same task produces via runner_optuna.py)
# once you have HPO results worth trusting.
FIXED_PARAMS = {
    "transformer": {
        "model_type": "transformer",
        "d_model": 64,
        "n_heads": 2,
        "num_layers": 2,
        "dim_ff": 256,
        "lr": 0.002115429079726122,
        "weight_decay": 0.005727904470799624,
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
    prefix = f"b_{MODEL_TYPE}_fixed"
    return runFixedParamsModelSelection(
        "binary", MODEL_TYPE, prefix, "auc", FIXED_PARAMS[MODEL_TYPE]
    )


if __name__ == "__main__":
    main()
