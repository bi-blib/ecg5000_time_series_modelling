import time

import optuna
import torch
from tqdm.auto import tqdm

from config import N_TRIALS, SHOW_PROGRESS


def performInferenceLoop(model, loader, device):
    """Forward-only pass over a loader, returning raw logits and targets.

    Shared by test-set evaluation (argmax predictions) and validation-set
    threshold selection (which needs class probabilities instead of a
    collapsed prediction).
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            logits = model(x_batch)

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_logits, all_targets


def performTrainingLoopOptuna(model, train_loader, val_loader, device, criterion,
                              lr, epochs, early_stop_patience,
                              checkpoint_path=None, verbose=True,
                              trial=None, weight_decay=0.0,
                              label=None, leave=None):
    """Train/validate loop with lr, epochs, patience, checkpoint path and weight
    decay as arguments instead of module constants.

    Pass trial=... to report intermediate val loss to Optuna, so unpromising
    trials are pruned partway through instead of burning the full epoch budget.
    Trials pass checkpoint_path=None - they only need the val loss, not weights.

    label/leave override the tqdm bar's caption and leave-behind behaviour.
    Left as None, they fall back to the trial-derived defaults below - callers
    that train once per fold (no per-epoch pruning, trial=None) pass their own
    label so the bar still identifies which trial/fold is running.
    """
    # AdamW rather than Adam so weight decay is decoupled from the gradient.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_val_epoch = 0

    # One bar per trial, showing elapsed time, s/epoch and ETA so the cost of a
    # trial is visible while it runs. Trial bars clear themselves on completion
    # (leave=False) so the log does not fill up with 25 finished bars; the final
    # retrain keeps its bar.
    if label is None:
        label = f"Trial {trial.number + 1}/{N_TRIALS}" if trial is not None else "Final training"
    if leave is None:
        leave = trial is None
    bar = tqdm(
        range(epochs),
        desc=label,
        unit="ep",
        leave=leave,
        dynamic_ncols=True,
        disable=not SHOW_PROGRESS,
    )

    started = time.perf_counter()
    status = "completed"
    epochs_run = 0

    try:
        for epoch in bar:
            epochs_run = epoch + 1
            model.train()
            train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            val_loss = 0.0
            model.eval()
            with torch.inference_mode():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(x_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)

            bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}@{best_val_epoch + 1}",
            )

            if verbose:
                # bar.write, not print - print would tear the bar in half.
                bar.write(
                    f"Epoch: {epoch + 1:3d} | "
                    f"Train loss: {train_loss:.8f} | "
                    f"Val loss: {val_loss:.8f} | "
                    f"Best Val loss: {best_val_loss:.8f} | "
                    f"Best Val epoch: {best_val_epoch + 1}"
                )

            # Report before the early-stop check, so a trial that stops early
            # still contributes its final value to the pruner's statistics.
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    status = "pruned"
                    raise optuna.TrialPruned()

            if epoch - best_val_epoch >= early_stop_patience:
                status = "early-stopped"
                if verbose:
                    bar.write(
                        f"No reduction in validation loss in {early_stop_patience} "
                        "epochs. Stopping training early."
                    )
                break
    finally:
        bar.close()
        elapsed = time.perf_counter() - started
        per_epoch = elapsed / max(epochs_run, 1)
        # Printed for every trial, pruned ones included, so the per-trial cost
        # stays on the record after the bar has cleared itself.
        print(
            f"{label}: {status} after {epochs_run} epochs in {elapsed:.1f}s "
            f"({per_epoch:.2f}s/epoch) | best val loss "
            f"{best_val_loss:.6f} @ epoch {best_val_epoch + 1}",
            flush=True,
        )

    return train_losses, val_losses
