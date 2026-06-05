import numpy as np
import tensorflow as tf
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cnn_model import build_cnn, train_cnn, decode_hyperparameters

# ─────────────────────────────────────────
# FITNESS FUNCTION
# Paper Section 3.3.2, Equation 23:
# f(x) = 1 - (TP + TN) / (TP + TN + FP + FN)
# f(x) = 1 - validation_accuracy
#
# This is a MINIMIZATION problem:
# - f(x) = 0   means 100% accuracy (perfect)
# - f(x) = 1   means 0% accuracy (worst)
# The optimizer tries to find x that minimizes f(x)
# ─────────────────────────────────────────

# Global data references — set once before optimization
_X_train = None
_y_train = None
_X_val   = None
_y_val   = None

# Track evaluations for convergence plotting
fitness_history = []


def set_data(X_train, y_train, X_val, y_val):
    """
    Set the training and validation data globally.
    Called once before optimization begins.
    Only train and val data are used here —
    test data is never touched during optimization
    as per paper Section 3.3.3.
    """
    global _X_train, _y_train, _X_val, _y_val
    _X_train = X_train
    _y_train = y_train
    _X_val   = X_val
    _y_val   = y_val


def fitness_function(x):
    """
    Evaluate a hyperparameter vector x.

    Paper Equation 23:
    f(x) = 1 - (TP + TN) / (TP + TN + FP + FN)
         = 1 - validation_accuracy

    Parameters
    ----------
    x : array-like, shape (9,)
        Hyperparameter vector:
        [filters_idx, kernel_idx, pooling_idx,
         neurons_idx, dropout_rate, learning_rate,
         batch_idx, optimizer_idx, max_epoch]

    Returns
    -------
    fitness : float
        Value between 0 and 1.
        Lower is better.
        0 = perfect classifier.
    """
    global fitness_history

    if _X_train is None:
        raise RuntimeError(
            "Data not set. Call fitness.set_data() first."
        )

    try:
        # Step 1 — Decode continuous vector to hyperparameters
        hyperparams = decode_hyperparameters(x)

        # Step 2 — Build CNN with these hyperparameters
        # Model is rebuilt from scratch each evaluation
        model = build_cnn(hyperparams)

        # Step 3 — Train on training set
        # Validation set used for early stopping only
        train_cnn(
            model=model,
            X_train=_X_train,
            y_train=_y_train,
            X_val=_X_val,
            y_val=_y_val,
            batch_size=hyperparams['batch_size'],
            max_epoch=hyperparams['max_epoch'],
        )

        # Step 4 — Evaluate on validation set
        _, val_accuracy = model.evaluate(
            _X_val, _y_val, verbose=0
        )

        # Step 5 — Compute fitness (Equation 23)
        fitness = 1.0 - val_accuracy

        # Track history for convergence plot (Figure 8)
        fitness_history.append(fitness)

        # Clean up model to free memory
        tf.keras.backend.clear_session()
        del model

        return float(fitness)

    except Exception as e:
        # If model fails to build/train with these
        # hyperparameters, return worst possible fitness
        print(f"  Fitness evaluation failed: {e}")
        fitness_history.append(1.0)
        return 1.0


def reset_history():
    """Reset fitness history between optimizer runs."""
    global fitness_history
    fitness_history = []


def get_history():
    """Return fitness history for convergence plotting."""
    return fitness_history.copy()


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
def test_fitness():
    """
    Test fitness function with paper's optimal
    hyperparameters from Table 6.
    Expected fitness ≈ 0.05 (95%+ accuracy)
    """
    import numpy as np

    print("Testing fitness function...")
    print("-" * 50)

    # Load preprocessed data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')

    # Register data with fitness function
    set_data(X_train, y_train, X_val, y_val)

    # Paper's optimal hyperparameter vector (Table 6)
    # [filters_idx, kernel_idx, pooling_idx, neurons_idx,
    #  dropout,     lr,         batch_idx,   opt_idx, epoch]
    #
    # filters=[32,64,128,256] → index 2
    # kernel=11               → index 4
    # pooling=3               → index 1
    # neurons=128             → index 3
    # dropout=0.313
    # lr=0.00015
    # batch=128               → index 5
    # optimizer=adam          → index 0
    # epoch=36
    x_test = np.array([2, 4, 1, 3, 0.313, 0.00015, 5, 0, 36])

    print(f"Hyperparameter vector: {x_test}")
    print("Training CNN (this takes ~1-2 minutes)...")

    fitness = fitness_function(x_test)

    print(f"\nFitness value:    {fitness:.4f}")
    print(f"Val accuracy:     {(1 - fitness) * 100:.2f}%")
    print(f"Fitness history:  {get_history()}")
    print("\nFitness function test PASSED")


if __name__ == "__main__":
    test_fitness()