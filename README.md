# ECG5000 Time Series Modelling

This repository contains the code, data, experiments, and presentation material for time-series classification research on the ECG5000 heartbeat dataset. It includes exploratory analysis, baseline models, Transformer and xLSTM classifiers, foundation-model experiments, archived experiment results, and assets used to document the work.

## Repository overview

| Path                      | Contents                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------ |
| `ECG5000/`                | ECG5000 dataset files in TXT, ARFF, and TS formats                                               |
| `src/xLSTMandEncoderHPO/` | Transformer and xLSTM classifiers, training, evaluation, and Optuna hyperparameter optimization  |
| `src/fm_classification/`  | Experiments using time-series foundation models for classification                               |
| `exploratory_analysis/`   | Exploratory notebooks and baseline experiments, including oversampling and hint-based approaches |
| `archive/`                | Earlier encoder and xLSTM implementations, checkpoints, HPO runs, and result artifacts           |
| `presentation_assets/`    | Scripts and generated figures used for presentations                                             |
| `dataset_description.md`  | Description, format, and provenance of the ECG5000 dataset                                       |
| `requirements.txt`        | Python dependencies for the project                                                              |

The main implementation areas are independent experiment tracks that use the shared ECG5000 data. The sections below document the Transformer and xLSTM classification track in detail; the other directories retain their own notebooks, scripts, and experiment artifacts.

# xLSTM and Encoder HPO

Classifiers for the ECG5000 heartbeat dataset, comparing a Transformer encoder against an xLSTM (mLSTM/sLSTM block stack) architecture. Two classification tasks are supported:

- **Binary** — normal vs. abnormal heartbeat.
- **Multiclass** — which of the four abnormal subtypes, restricted to abnormal beats.

For each task, hyperparameters can either be a hand-picked fixed set, or searched with Optuna. Both paths end in the same model-selection procedure: 5-fold cross-validation on a pooled train/validation split, selection of the best fold, and a final evaluation against a held-out test set.

## Prerequisites

- **Python 3.10 or 3.11.** `torch==2.2.2` (pinned in `requirements.txt`) has no wheels for newer interpreters (e.g. 3.13/3.14), so check `python --version` and use a matching interpreter (`py -3.11` on Windows) if your default `python` is newer.
- A virtual environment is recommended:
  ```
  python -m venv .venv
  .venv\Scripts\activate      # Windows
  source .venv/bin/activate   # macOS/Linux
  ```
- Install dependencies **from the repo root** (`requirements.txt` lives there, not in this folder):
  ```
  pip install -r requirements.txt
  ```
  This file also pins `mantis-tsfm`, which belongs to the unrelated `src/fm_classification` module — it's not used here, but harmless to have installed.

## Dataset

The ECG5000 data already ships in this repo at `ECG5000/ECG5000_TRAIN.txt` and `ECG5000/ECG5000_TEST.txt` (repo root) — there's nothing to download. See `../../dataset_description.md` for the dataset's format and provenance.

## Running the scripts — run from the repo root

**Every command below must be run from the repo root, not from inside this folder.** `data.py` loads the dataset via the relative path `"ECG5000/ECG5000_TRAIN.txt"`, which is resolved against the process's current working directory. If you `cd` into `src/xLSTMandEncoderHPO/` first, there's no `ECG5000/` folder there and the script will fail with `FileNotFoundError`.

```
python src/xLSTMandEncoderHPO/runner_fixed.py
```

| Script                                        | Task                | Hyperparameters      |
| --------------------------------------------- | ------------------- | -------------------- |
| `runner_fixed.py`                             | binary + multiclass | fixed (fast)         |
| `runner_optuna.py`                            | binary + multiclass | Optuna search (slow) |
| `normal_abnormal_binary_classifier_fixed.py`  | binary only         | fixed                |
| `normal_abnormal_binary_classifier_optuna.py` | binary only         | Optuna search        |
| `abnormal_multiclassifer_fixed.py`            | multiclass only     | fixed                |
| `abnormal_multiclassifer_optuna.py`           | multiclass only     | Optuna search        |

The `runner_*.py` scripts are just a convenience — each calls the `main()` of its two matching task scripts in sequence. None of the scripts take command-line arguments; all configuration is via `config.py` and the hardcoded `FIXED_PARAMS` dicts in the `*_fixed.py` files.

The Optuna variants are considerably slower than the fixed variants: they run a full hyperparameter search (`N_TRIALS` trials at a reduced epoch budget) before the same 5-fold CV / test-evaluation pass. On CPU, expect the fixed variants to take a few minutes and the Optuna variants tens of minutes, depending on `config.py`'s settings.

## Configuration (`config.py`)

The single most important switch:

```python
MODEL_TYPE = "transformer"   # or "xlstm"
```

This picks the architecture (`BaseTransformerClassifier` vs. the xLSTM block-stack classifier) used by every entry point above — edit it before running.

Other knobs (see the comments in `config.py` for the reasoning behind each default):

| Constant                                              | Meaning                                                |
| ----------------------------------------------------- | ------------------------------------------------------ |
| `EPOCHS`                                              | Max epochs for the final (post-selection) training run |
| `N_TRIALS`                                            | Number of Optuna trials                                |
| `HPO_EPOCHS`                                          | Epoch budget per Optuna trial (smaller than `EPOCHS`)  |
| `N_SPLITS`                                            | Number of CV folds                                     |
| `TEST_FRACTION`                                       | Fraction of data held out as the final test set        |
| `NUM_THREADS`                                         | PyTorch CPU thread count                               |
| `SHOW_PROGRESS`                                       | Toggle tqdm progress bars                              |
| `BINARY_THRESHOLD_BETA` / `MULTICLASS_THRESHOLD_BETA` | F-beta used to pick each task's decision threshold     |

If you want to run with your own tuned hyperparameters instead of Optuna, edit the `FIXED_PARAMS` dict in `normal_abnormal_binary_classifier_fixed.py` or `abnormal_multiclassifer_fixed.py` — a natural source is the `"lambda_star"` field of a previous Optuna run's `*_cv_results.json` (see Outputs below).

## Outputs

Every run writes its outputs to the **current working directory** (the repo root, if invoked as instructed above), using a filename prefix of `{b|mc}_{MODEL_TYPE}_{fixed|optuna}` — e.g. `b_transformer_optuna` for a binary run of the Optuna variant with `MODEL_TYPE = "transformer"`.

| File                                                          | Contents                                                                                                                         |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `{prefix}_cf.png`                                             | Confusion matrix (counts + row-normalized)                                                                                       |
| `{prefix}_auc.png` (binary) / `{prefix}_roc.png` (multiclass) | Test-set ROC + PR curves                                                                                                         |
| `{prefix}_auc_val.png` / `{prefix}_roc_val.png`               | Validation-set ROC + PR curves, with the chosen decision threshold marked                                                        |
| `{prefix}_fold{N}_model.pth`                                  | Checkpoint of the winning CV fold (the other folds' checkpoints are deleted)                                                     |
| `{prefix}_cv_results.json`                                    | Hyperparameters, per-fold losses, best fold, decision threshold(s), test metrics, split sizes, timing, trainable parameter count |

`example_outputs/` in this folder is a worked example (binary + multiclass, Transformer architecture, Optuna variant) including `run_log.txt`, the full console output of that run — useful as a reference for what a complete run looks like.

## File map

| File                                                        | Purpose                                                                                                     |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `config.py`                                                 | All tunable constants, including the `MODEL_TYPE` switch                                                    |
| `data.py`                                                   | ECG5000 loading, task labeling, oversampling, stratified fold splitting                                     |
| `models.py`                                                 | `BaseTransformerClassifier` and the xLSTM-based classifier, plus the `buildModel()` dispatcher              |
| `training.py`                                               | Train/validate loop with early stopping and Optuna pruning hooks                                            |
| `hpo_search.py`                                             | Optuna search-space definitions per architecture                                                            |
| `model_selection.py`                                        | Orchestration: runs Optuna or fixed params, 5-fold CV, best-fold selection, test evaluation, report writing |
| `evaluation.py`                                             | Decision-threshold selection and metric computation                                                         |
| `plotting.py`                                               | Confusion-matrix and ROC/PR plot rendering                                                                  |
| `normal_abnormal_binary_classifier_fixed.py` / `_optuna.py` | Entry points for the binary task                                                                            |
| `abnormal_multiclassifer_fixed.py` / `_optuna.py`           | Entry points for the multiclass task                                                                        |
| `runner_fixed.py` / `runner_optuna.py`                      | Entry points that run both tasks in sequence                                                                |
