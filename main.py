import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace'
)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding='utf-8', errors='replace'
)
import numpy as np
from src.preprocess import run_preprocessing
from src.hybrid_optimizer import HybridOptimizer
from src.cnn_model import (
    LOWER_BOUNDS, UPPER_BOUNDS,
    build_cnn, train_cnn, decode_hyperparameters,
    train_model_with_retries
)
from src.standalone_optimizers import (
    run_standalone_gwo, run_standalone_woa,
    run_standalone_aoa, run_standalone_rime
)
from src.fitness import set_data, reset_history
from src.evaluate import run_full_evaluation
from src.shap_analysis import run_shap_analysis
from src.smote_analysis import run_smote_analysis
import joblib

# ─────────────────────────────────────────
# CONFIGURATION
# Adjust these for quick test vs full run
# ─────────────────────────────────────────
POPULATION_SIZE = 6  # full paper: 20
ITERATIONS      = 3  # full paper: 10 (per algorithm)
RUN_LABEL       = "quick run — medium population"



if __name__ == "__main__":

    print("=" * 50)
    print("GWO-WOA-AOA Heart Disease Prediction")
    print(RUN_LABEL)
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
    y_test_raw = np.load('data/processed/y_test_raw.npy')

    set_data(X_train, y_train, X_val, y_val)

    models_dict        = {}
    convergence_curves = {}
    hp_dict            = {}

    # =================================================
    # MODEL 1 — NO-CNN (baseline, no optimization)
    # =================================================
    print("\n" + "=" * 50)
    print("MODEL 1/6 — NO-CNN (baseline)")
    print("=" * 50)
    no_cnn_hp = decode_hyperparameters(
        [1, 1, 1, 2, 0.3, 0.001, 2, 0, 50]
    )
    no_cnn = train_model_with_retries(
        no_cnn_hp, X_train, y_train, X_val, y_val,
        n_attempts=3, label="NO-CNN"
    )
    models_dict['NO-CNN'] = no_cnn

    # =================================================
    # MODEL 2 — GWO-CNN (standalone GWO)
    # =================================================
    print("\n" + "=" * 50)
    print("MODEL 2/6 — GWO-CNN (standalone)")
    print("=" * 50)
    reset_history()
    gwo_pos, gwo_fit, gwo_curve = run_standalone_gwo(
        POPULATION_SIZE, ITERATIONS,
        LOWER_BOUNDS, UPPER_BOUNDS, verbose=True
    )
    gwo_hp = decode_hyperparameters(gwo_pos)
    gwo_model = train_model_with_retries(
        gwo_hp, X_train, y_train, X_val, y_val,
        n_attempts=3, label="GWO-CNN"
    )
    models_dict['GWO-CNN'] = gwo_model
    convergence_curves['GWO'] = gwo_curve
    hp_dict['GWO-CNN'] = gwo_hp

    # =================================================
    # MODEL 3 — WOA-CNN (standalone WOA)
    # =================================================
    print("\n" + "=" * 50)
    print("MODEL 3/6 — WOA-CNN (standalone)")
    print("=" * 50)
    reset_history()
    woa_pos, woa_fit, woa_curve = run_standalone_woa(
        POPULATION_SIZE, ITERATIONS,
        LOWER_BOUNDS, UPPER_BOUNDS, verbose=True
    )
    woa_hp = decode_hyperparameters(woa_pos)
    woa_model = train_model_with_retries(
        woa_hp, X_train, y_train, X_val, y_val,
        n_attempts=3, label="WOA-CNN"
    )
    models_dict['WOA-CNN'] = woa_model
    convergence_curves['WOA'] = woa_curve
    hp_dict['WOA-CNN'] = woa_hp

    # =================================================
    # MODEL 4 — AOA-CNN (standalone AOA)
    # =================================================
    print("\n" + "=" * 50)
    print("MODEL 4/6 — AOA-CNN (standalone)")
    print("=" * 50)
    reset_history()
    aoa_pos, aoa_fit, aoa_curve = run_standalone_aoa(
        POPULATION_SIZE, ITERATIONS,
        LOWER_BOUNDS, UPPER_BOUNDS, verbose=True
    )
    aoa_hp = decode_hyperparameters(aoa_pos)
    aoa_model = train_model_with_retries(
        aoa_hp, X_train, y_train, X_val, y_val,
        n_attempts=3, label="AOA-CNN"
    )
    models_dict['AOA-CNN'] = aoa_model
    convergence_curves['AOA'] = aoa_curve
    hp_dict['AOA-CNN'] = aoa_hp

    # =================================================
    # MODEL 5 — RIME-CNN (standalone RIME)
    # =================================================
    print("\n" + "=" * 50)
    print("MODEL 5/6 — RIME-CNN (standalone)")
    print("=" * 50)
    reset_history()
    rime_pos, rime_fit, rime_curve = run_standalone_rime(
        POPULATION_SIZE, ITERATIONS,
        LOWER_BOUNDS, UPPER_BOUNDS, verbose=True
    )
    rime_hp = decode_hyperparameters(rime_pos)
    rime_model = train_model_with_retries(
        rime_hp, X_train, y_train, X_val, y_val,
        n_attempts=3, label="RIME-CNN"
    )
    models_dict['RIME-CNN'] = rime_model
    convergence_curves['RIME'] = rime_curve
    hp_dict['RIME-CNN'] = rime_hp

    # =================================================
    # MODEL 6 — GWO-WOA-AOA-CNN (the hybrid)
    # =================================================
    print("\n" + "=" * 50)
    print("MODEL 6/6 — GWO-WOA-AOA-CNN (hybrid)")
    print("=" * 50)
    hybrid = HybridOptimizer(
        population_size=POPULATION_SIZE,
        gwo_iterations=ITERATIONS,
        woa_iterations=ITERATIONS,
        aoa_iterations=ITERATIONS,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS
    )
    best_hp, best_fitness, hybrid_curve = hybrid.optimize(
        X_train, y_train, X_val, y_val, verbose=True
    )
    final_model = hybrid.train_final_model(
        X_train, y_train, X_val, y_val, verbose=True
    )
    models_dict['GWO-WOA-AOA-CNN'] = final_model
    convergence_curves['GWO-WOA-AOA'] = hybrid_curve
    hp_dict['GWO-WOA-AOA-CNN'] = best_hp

    # =================================================
    # EVALUATION — Table 7, Figures 8, 9, 10
    # =================================================
    # Reorder to match paper Table 7 order
    ordered_models = {
        'NO-CNN':          models_dict['NO-CNN'],
        'RIME-CNN':        models_dict['RIME-CNN'],
        'AOA-CNN':         models_dict['AOA-CNN'],
        'WOA-CNN':         models_dict['WOA-CNN'],
        'GWO-CNN':         models_dict['GWO-CNN'],
        'GWO-WOA-AOA-CNN': models_dict['GWO-WOA-AOA-CNN'],
    }

    results = run_full_evaluation(
        models_dict=ordered_models,
        X_test=X_test,
        y_test_cat=y_test,
        y_test_raw=y_test_raw,
        convergence_curves=convergence_curves,
        hp_dict=hp_dict
    )

    # =================================================
    # SHAP EXPLAINABILITY ANALYSIS
    # Paper Section 4.4 — Figures 14 and 15
    # =================================================
    print("\n" + "=" * 50)
    print("SHAP ANALYSIS")
    print("=" * 50)

    if final_model is not None:
        shap_values, mean_shap = run_shap_analysis(
            model=final_model,
            X_train=X_train,
            X_test=X_test,
            n_background=100,  # increase for accuracy
            n_explain=200       # explain all test samples
        )
    else:
        print("No final model available for SHAP analysis.")

    # =================================================
    # SMOTE AUGMENTATION ANALYSIS
    # Paper Section 4.5 — Table 10 and Figure 16
    # =================================================
    print("\n" + "=" * 50)
    print("SMOTE AUGMENTATION ANALYSIS")
    print("=" * 50)

    # Load scaler fitted during preprocessing
    scaler = joblib.load('models/scaler.pkl')

    # Load raw integer labels for SMOTE
    y_train_raw = np.load('data/processed/y_train_raw.npy')
    y_val_raw   = np.load('data/processed/y_val_raw.npy')
    y_test_raw  = np.load('data/processed/y_test_raw.npy')

    smote_results = run_smote_analysis(
        best_hyperparams=best_hp,
        X_train_raw=X_train,
        y_train_raw=y_train_raw,
        X_val_raw=X_val,
        y_val_raw=y_val_raw,
        X_test_raw=X_test,
        y_test_raw=y_test_raw,
        scaler=scaler,
        framingham_path=None,  
        n_attempts=5
    )

    print("\nFull comparison complete.")
    print("Check outputs/ folder for all figures and results.")