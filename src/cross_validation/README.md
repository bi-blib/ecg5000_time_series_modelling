# Five-fold cross-validation

This module performs model selection exactly once per candidate on the same
five stratified folds. It minimizes mean validation cross-entropy to obtain
`lambda_star`, then trains that configuration on a fresh 70/15/15
train/validation/test split and reports balanced accuracy on the untouched
test set.

From the repository root:

```bash
.venv/bin/python -m src.cross_validation.run_cross_validation --task binary
.venv/bin/python -m src.cross_validation.run_cross_validation --task multiclass
```

The default search has eight candidate configurations. To use another search
space, pass `--grid grid.json`. The JSON must be an object of parameter lists
accepted by scikit-learn's `ParameterGrid`, for example:

```json
{
  "d_model": [32, 64],
  "n_heads": [4],
  "num_layers": [1, 2],
  "dim_ff": [128],
  "lr": [0.0001, 0.0003],
  "batch_size": [32]
}
```

Outputs are written under `cross_validation_results/`: the final checkpoint,
the complete CV/test report as JSON, and the final train/validation losses.
