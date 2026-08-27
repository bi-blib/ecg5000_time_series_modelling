from config import MODEL_TYPE, QVK_PROJ_BLOCKSIZE


def suggestXLSTMHyperparams(trial):
    """Search space for TunableXLSTMClassifier + training hyperparameters.

    Every combination below is valid by construction. The mLSTM inner dim is
    round_up(proj_factor * d_model, multiple_of=64), so num_heads and
    qkv_proj_blocksize always divide it; sLSTM sets hidden_size = d_model, and
    both 2 and 4 divide each of 32/64/128.

    Ordered roughly by expected impact on this dataset (500 training samples,
    140 timesteps, heavy class imbalance) - regularisation and head size matter
    more here than raw capacity.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "pooling": trial.suggest_categorical("pooling", ["last", "mean", "flatten"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "block_mix": trial.suggest_categorical("block_mix", ["mlstm", "mixed", "slstm"]),
        "conv1d_kernel_size": trial.suggest_categorical("conv1d_kernel_size", [2, 4, 8]),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
        "proj_factor": trial.suggest_categorical("proj_factor", [1.0, 2.0]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        # Fixed rather than searched: it is effectively a re-parameterisation of
        # head count, and 25 trials is already thin for the dimensions above.
        # Promote to [2, 4, 8] if the search budget grows.
        "qkv_proj_blocksize": 4,
    }


def suggestTransformerHyperparams(trial):
    """Search space for BaseTransformerClassifier + training hyperparameters.

    n_heads is drawn from values that evenly divide every d_model choice below
    (32/64/128 are all divisible by 2 and 4), so every sampled combination is a
    valid nn.MultiheadAttention config.

    weight_decay is included so this dict is interchangeable with the xLSTM one
    at every call site - performTrainingLoopOptuna takes it either way.
    """
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_ff": trial.suggest_categorical("dim_ff", [64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def suggestHyperparams(trial, model_type=MODEL_TYPE):
    """Dispatch to the right search space and tag the result with model_type.

    The tag rides along in the params dict so buildModel knows what to build
    without a second argument threaded through every call site.
    """
    if model_type == "transformer":
        params = suggestTransformerHyperparams(trial)
    elif model_type == "xlstm":
        params = suggestXLSTMHyperparams(trial)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}, expected 'xlstm' or 'transformer'"
        )

    params["model_type"] = model_type
    return params


def resolveBestParams(study, model_type=MODEL_TYPE):
    """Rebuild the full param dict for the winning trial.

    study.best_params only contains values that went through a trial.suggest_*
    call, so anything injected (model_type) or fixed (qkv_proj_blocksize) is
    missing from it and has to be put back before buildModel sees it.
    """
    best_params = dict(study.best_params)
    best_params["model_type"] = model_type
    best_params.setdefault("qkv_proj_blocksize", QVK_PROJ_BLOCKSIZE)
    return best_params
