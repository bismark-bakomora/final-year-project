import json
import logging
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.cnn_model import build_cnn, decode_hyperparameters, train_cnn
from src.memory_utils import aggressive_memory_cleanup, reset_fitness_gc_counter
from src.tf_config import set_random_seeds

logger = logging.getLogger("heart_disease.fitness")

# ─────────────────────────────────────────
# FITNESS FUNCTION
# Paper Section 3.3.2, Equation 23:
# f(x) = 1 - validation_accuracy
# ─────────────────────────────────────────

_X_train = None
_y_train = None
_X_val = None
_y_val = None
_y_train_raw = None

fitness_history = []
_eval_count = 0
_log_interval = 25

# Runtime configuration (set via configure_fitness)
_cv_folds = 0
_persist_checkpoints = True
_gap_penalty = 0.0
_early_stopping_patience = 5
_ensemble_top_k = 3
_fitness_seed = 42
_checkpoint_dir: Path | None = None

# Best search checkpoint (single best)
_best_fitness = float("inf")
_best_val_accuracy = 0.0
_best_hyperparams: dict | None = None
_best_weights_path: Path | None = None

# Top-K checkpoints for ensemble [{fitness, val_accuracy, hyperparams, weights_path}]
_top_checkpoints: list[dict] = []


def configure_fitness_logging(log_interval: int = 25) -> None:
    global _log_interval
    _log_interval = max(1, int(log_interval))


def configure_fitness(
    *,
    cv_folds: int = 0,
    persist_checkpoints: bool = True,
    validation_gap_penalty: float = 0.0,
    early_stopping_patience: int = 5,
    ensemble_top_k: int = 3,
    checkpoint_dir: Path | str | None = None,
    fitness_seed: int = 42,
) -> None:
    """Apply pipeline training settings to the fitness evaluator."""
    global _cv_folds, _persist_checkpoints, _gap_penalty
    global _early_stopping_patience, _ensemble_top_k, _checkpoint_dir
    global _fitness_seed

    _cv_folds = max(0, int(cv_folds))
    _persist_checkpoints = bool(persist_checkpoints)
    _gap_penalty = max(0.0, float(validation_gap_penalty))
    _early_stopping_patience = max(1, int(early_stopping_patience))
    _ensemble_top_k = max(1, int(ensemble_top_k))
    _fitness_seed = int(fitness_seed)
    _checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None


def set_data(X_train, y_train, X_val, y_val, y_train_raw=None):
    """Register train/val arrays used during hyperparameter search."""
    global _X_train, _y_train, _X_val, _y_val, _y_train_raw
    _X_train = X_train
    _y_train = y_train
    _X_val = X_val
    _y_val = y_val
    if y_train_raw is None:
        _y_train_raw = np.argmax(y_train, axis=1)
    else:
        _y_train_raw = y_train_raw


def _checkpoint_root() -> Path:
    if _checkpoint_dir is not None:
        root = _checkpoint_dir
    else:
        root = Path("models") / "search_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _hyperparams_match(a: dict, b: dict) -> bool:
    if a is None or b is None:
        return False
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _train_and_evaluate(
    hyperparams: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    seed: int,
) -> tuple[float, float, object]:
    """Train one candidate and return (fitness, val_accuracy, model)."""
    model = build_cnn(hyperparams)
    train_cnn(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=hyperparams["batch_size"],
        max_epoch=hyperparams["max_epoch"],
        patience=_early_stopping_patience,
        seed=seed,
    )
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    gap = max(0.0, float(train_acc) - float(val_accuracy))
    fitness = 1.0 - float(val_accuracy) + _gap_penalty * gap
    return fitness, float(val_accuracy), model


def _evaluate_kfold(hyperparams: dict, seed: int) -> tuple[float, float, object]:
    """K-fold CV on training data; last fold model returned for checkpointing."""
    kf = StratifiedKFold(
        n_splits=_cv_folds,
        shuffle=True,
        random_state=seed,
    )
    fold_fitness: list[float] = []
    fold_val_acc: list[float] = []
    last_model = None

    for fold_idx, (tr_idx, va_idx) in enumerate(
        kf.split(_X_train, _y_train_raw)
    ):
        X_tr = _X_train[tr_idx]
        y_tr = _y_train[tr_idx]
        X_va = _X_train[va_idx]
        y_va = _y_train[va_idx]
        fold_seed = seed + fold_idx
        fitness, val_acc, model = _train_and_evaluate(
            hyperparams, X_tr, y_tr, X_va, y_va, fold_seed
        )
        fold_fitness.append(fitness)
        fold_val_acc.append(val_acc)
        if last_model is not None:
            del last_model
        last_model = model

    mean_fitness = float(np.mean(fold_fitness))
    mean_val_acc = float(np.mean(fold_val_acc))
    return mean_fitness, mean_val_acc, last_model


def _save_checkpoint(
    model, hyperparams: dict, fitness: float, val_accuracy: float
) -> tuple[Path, Path]:
    """Persist weights and hyperparameter metadata for a candidate model."""
    root = _checkpoint_root()
    tag = f"ckpt_{_eval_count:05d}_{val_accuracy:.4f}"
    weights_path = root / f"{tag}.weights.h5"
    hp_path = root / f"{tag}.json"
    model.save_weights(weights_path)
    hp_path.write_text(json.dumps(hyperparams, indent=2), encoding="utf-8")
    return weights_path, hp_path


def _register_checkpoint(
    model,
    hyperparams: dict,
    fitness: float,
    val_accuracy: float,
) -> None:
    """Track global best and top-K ensemble checkpoints."""
    global _best_fitness, _best_val_accuracy, _best_hyperparams, _best_weights_path

    weights_path, hp_path = _save_checkpoint(
        model, hyperparams, fitness, val_accuracy
    )
    entry = {
        "fitness": fitness,
        "val_accuracy": val_accuracy,
        "hyperparams": hyperparams.copy(),
        "weights_path": str(weights_path),
        "meta_path": str(hp_path),
    }

    _top_checkpoints.append(entry)
    _top_checkpoints.sort(key=lambda e: e["fitness"])
    while len(_top_checkpoints) > _ensemble_top_k:
        dropped = _top_checkpoints.pop()
        try:
            Path(dropped["weights_path"]).unlink(missing_ok=True)
            Path(dropped["meta_path"]).unlink(missing_ok=True)
        except OSError:
            pass

    if fitness < _best_fitness:
        _best_fitness = fitness
        _best_val_accuracy = val_accuracy
        _best_hyperparams = hyperparams.copy()
        _best_weights_path = weights_path

        best_link = _checkpoint_root() / "best.weights.h5"
        best_meta = _checkpoint_root() / "best.json"
        shutil.copy2(weights_path, best_link)
        best_meta.write_text(
            json.dumps(
                {
                    "fitness": fitness,
                    "val_accuracy": val_accuracy,
                    "hyperparams": hyperparams,
                    "weights_path": str(best_link),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _best_weights_path = best_link
        logger.info(
            "New search-best checkpoint | fitness=%.4f val_acc=%.2f%%",
            fitness,
            val_accuracy * 100,
        )


def fitness_function(x):
    """
    Evaluate hyperparameter vector x.
    Returns 1 - validation_accuracy (lower is better).
    """
    global fitness_history, _eval_count

    if _X_train is None:
        raise RuntimeError("Data not set. Call fitness.set_data() first.")

    try:
        set_random_seeds(_fitness_seed + _eval_count)
        hyperparams = decode_hyperparameters(x)

        if _cv_folds >= 2:
            fitness, val_accuracy, model = _evaluate_kfold(
                hyperparams, _fitness_seed + _eval_count
            )
        else:
            fitness, val_accuracy, model = _train_and_evaluate(
                hyperparams,
                _X_train,
                _y_train,
                _X_val,
                _y_val,
                _fitness_seed + _eval_count,
            )

        fitness_history.append(fitness)
        _eval_count += 1

        if _eval_count % _log_interval == 0:
            logger.info(
                "Fitness eval #%d | latest=%.4f acc=%.2f%% | history_len=%d",
                _eval_count,
                fitness,
                val_accuracy * 100,
                len(fitness_history),
            )

        if _persist_checkpoints:
            _register_checkpoint(model, hyperparams, fitness, val_accuracy)

        del model
        aggressive_memory_cleanup(label="fitness_ok")
        return float(fitness)

    except Exception as e:
        logger.warning("Fitness evaluation failed: %s", e)
        fitness_history.append(1.0)
        _eval_count += 1
        aggressive_memory_cleanup(label="fitness_fail")
        return 1.0


def reset_history():
    """Reset fitness history between optimizer runs."""
    global fitness_history, _eval_count
    fitness_history = []
    _eval_count = 0
    reset_fitness_gc_counter()


def reset_search_checkpoints(clear_disk: bool = True) -> None:
    """Clear in-memory and on-disk search checkpoints (e.g. new seed run)."""
    global _best_fitness, _best_val_accuracy, _best_hyperparams
    global _best_weights_path, _top_checkpoints

    _best_fitness = float("inf")
    _best_val_accuracy = 0.0
    _best_hyperparams = None
    _best_weights_path = None
    _top_checkpoints = []

    if clear_disk:
        root = _checkpoint_root()
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)


def get_best_search_checkpoint() -> dict | None:
    """Return best persisted search checkpoint metadata."""
    if _best_weights_path is None or _best_hyperparams is None:
        return None
    return {
        "fitness": _best_fitness,
        "val_accuracy": _best_val_accuracy,
        "hyperparams": _best_hyperparams.copy(),
        "weights_path": str(_best_weights_path),
    }


def get_top_checkpoints() -> list[dict]:
    """Return top-K checkpoint metadata sorted by fitness (best first)."""
    return [dict(c) for c in _top_checkpoints]


def get_history():
    return fitness_history.copy()
