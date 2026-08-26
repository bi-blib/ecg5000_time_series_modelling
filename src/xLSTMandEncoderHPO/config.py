# Model/training constants shared across the ECG5000 classification pipeline.

QVK_PROJ_BLOCKSIZE = 4
EPOCHS = 300
EARLY_STOP_NO_IMPROVEMENT = 10

# Optuna HPO settings. Trials run on a smaller epoch budget than the final
# training run so a full search finishes in reasonable time on CPU; the
# winning config is then retrained with the full EPOCHS budget.
N_TRIALS = 10
HPO_EPOCHS = 40
HPO_EARLY_STOP_NO_IMPROVEMENT = 8

# MedianPruner for the HPO objective (runHPOModelSelection): a trial is
# pruned if its val loss at a given epoch is worse than the median of other
# trials' values at that same epoch.
#   n_startup_trials: trials that always run to completion, so there is a
#   distribution to compare against before pruning starts.
#   n_warmup_steps: epochs within every trial that are never pruned, so a
#   config that starts slow isn't killed before it has any signal.
HPO_PRUNER_N_STARTUP_TRIALS = 5
HPO_PRUNER_N_WARMUP_STEPS = 10

# Cross-validated model selection settings (runHPOModelSelection,
# runFixedParamsModelSelection). TEST_FRACTION is deliberately inverted from
# the usual small-test-set convention: only 40% of the data is pooled for
# HPO/CV, 60% is held out.
RANDOM_STATE = 42
N_SPLITS = 5
TEST_FRACTION = 0.60

# Validation/test batches only ever run forward, so they can be far larger
# than the trial's training batch size. The val set is ~2250 samples and is
# evaluated every epoch, so this is the cheapest speedup available on CPU.
EVAL_BATCH_SIZE = 256

# Decision-threshold selection (runFinalModelSelection), done on the
# validation set and then frozen before touching the test set.
#   Binary: F-beta with beta>1 weights recall over precision, so missing an
#   abnormal ECG (false negative) costs more than a false alarm.
#   Multiclass: beta=1 (plain F1), applied per class one-vs-rest, so no class
#   is favoured over another.
BINARY_THRESHOLD_BETA = 2.0
MULTICLASS_THRESHOLD_BETA = 1.0

# ECG5000 epochs are tiny (500 samples), so the per-op matmuls are small and
# spreading them over every core costs more in thread sync than it saves.
NUM_THREADS = 4

# Per-trial epoch progress bars. Set False for clean logs when redirecting
# output to a file.
SHOW_PROGRESS = True

# Which architecture the Optuna workflow searches over.
#   "xlstm"       -> TunableXLSTMClassifier (mLSTM / mixed / sLSTM block stack)
#   "transformer" -> BaseTransformerClassifier (the original encoder)
# This is the single switch: it picks both the search space and the model that
# buildModel constructs. Nothing else needs to change.
MODEL_TYPE = "transformer"
