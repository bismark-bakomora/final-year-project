import numpy as np
from src.preprocess import run_preprocessing
from src.hybrid_optimizer import HybridOptimizer
from src.cnn_model import (
    LOWER_BOUNDS, UPPER_BOUNDS,
    build_cnn, train_cnn, decode_hyperparameters
)
from src.fitness import set_data
from src.evaluate import run_full_evaluation

if __name__ == "__main__":

    print("=" * 50)
    print("GWO-WOA-AOA Heart Disease Prediction")
    print("QUICK TEST — small population")
    print("=" * 50)

    # ── Step 1: Preprocessing ──
    run_preprocessing(
        'data/raw/heart_statlog_cleveland_hungary_final.csv'
    )

    # ── Load processed data ──
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')
    X_test  = np.load('data/processed/X_test.npy')
    y_test  = np.load('data/processed/y_test.npy')
    y_test_raw  = np.load(
        'data/processed/y_test_raw.npy'
    )

    # ── Step 2: Hybrid Optimization (small test) ──
    hybrid = HybridOptimizer(
        population_size=20,   # full: 20
        gwo_iterations=10,    # full: 10
        woa_iterations=10,    # full: 10
        aoa_iterations=10,    # full: 10
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS
    )

    best_hp, best_fitness, conv_curve = hybrid.optimize(
        X_train, y_train, X_val, y_val, verbose=True
    )

    # ── Step 3: Train final model ──
    final_model = hybrid.train_final_model(
        X_train, y_train, X_val, y_val, verbose=True
    )

    # ── Step 4: Build NO-CNN baseline ──
    print("\nTraining NO-CNN baseline...")
    no_cnn_hp = decode_hyperparameters(
        [1, 1, 1, 2, 0.3, 0.001, 2, 0, 50]
    )
    no_cnn = build_cnn(no_cnn_hp)
    train_cnn(
        no_cnn, X_train, y_train,
        X_val, y_val,
        no_cnn_hp['batch_size'],
        no_cnn_hp['max_epoch']
    )
    print("NO-CNN training complete.")

    # ── Step 5: Evaluate both models ──
    models_dict = {
        'NO-CNN':          no_cnn,
        'GWO-WOA-AOA-CNN': final_model,
    }

    convergence_curves = {
        'GWO-WOA-AOA': conv_curve,
    }

    hp_dict = {
        'GWO-WOA-AOA-CNN': best_hp,
    }

    results = run_full_evaluation(
        models_dict=models_dict,
        X_test=X_test,
        y_test_cat=y_test,
        y_test_raw=y_test_raw,
        convergence_curves=convergence_curves,
        hp_dict=hp_dict
    )

    print("\nQuick test complete.")
    print("Check outputs/ folder:")
    print("  outputs/figures/confusion_matrices.png")
    print("  outputs/figures/roc_curves.png")
    print("  outputs/figures/convergence_curves.png")
    print("  outputs/figures/performance_comparison.png")
    print("  outputs/results/metrics_comparison.csv")
    print("  outputs/results/hyperparameters.csv")