import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

from src.cnn_model import (
    build_cnn, train_cnn, train_model_with_retries
)
from src.evaluate import compute_metrics

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

# ─────────────────────────────────────────
# FEATURE COLUMNS
# Must match preprocess.py order
# ─────────────────────────────────────────
FEATURE_COLS = [
    'age', 'sex', 'chest pain type', 'resting bp s',
    'cholesterol', 'fasting blood sugar', 'resting ecg',
    'max heart rate', 'exercise angina', 'oldpeak',
    'ST slope'
]


# ─────────────────────────────────────────
# DATASET 1 — ORIGINAL
# Already prepared in preprocess.py
# 1190 records: 561 class 0, 629 class 1
# ─────────────────────────────────────────
def prepare_original(X_train_raw, y_train_raw,
                     X_val_raw, y_val_raw,
                     X_test_raw, y_test_raw):
    """
    Returns the original processed dataset.
    No augmentation applied.
    Paper: 1190 records, 47.14% / 52.86% split.
    """
    print("\nDataset: ORIGINAL")
    print(f"  Total: {len(y_train_raw)+len(y_val_raw)+len(y_test_raw)}")
    print(f"  Train: {len(y_train_raw)}, "
          f"Val: {len(y_val_raw)}, "
          f"Test: {len(y_test_raw)}")

    return (X_train_raw, to_categorical(y_train_raw, 2),
            X_val_raw,   to_categorical(y_val_raw, 2),
            X_test_raw,  to_categorical(y_test_raw, 2),
            y_test_raw)


# ─────────────────────────────────────────
# DATASET 2 — BALANCED
# SMOTE applied to minority class (class 0)
# to match majority class (class 1 = 629)
# Total: 1258 samples
# ─────────────────────────────────────────
def prepare_balanced(X_train_raw, y_train_raw,
                     X_val_raw, y_val_raw,
                     X_test_raw, y_test_raw,
                     scaler):
    """
    Apply SMOTE to training data only.
    Minority class (0) oversampled to match
    majority class (1).
    Paper: 1258 total samples after balancing.
    """
    print("\nDataset: BALANCED (SMOTE x1)")

    # Squeeze from (N, 11, 1, 1) to (N, 11)
    X_tr_2d = X_train_raw.squeeze(axis=(2, 3))

    # Un-normalize to apply SMOTE on original scale
    # then re-normalize after
    X_tr_orig = scaler.inverse_transform(X_tr_2d)

    print(f"  Before SMOTE: "
          f"{np.sum(y_train_raw==0)} class 0, "
          f"{np.sum(y_train_raw==1)} class 1")

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(
        X_tr_orig, y_train_raw
    )

    print(f"  After SMOTE:  "
          f"{np.sum(y_resampled==0)} class 0, "
          f"{np.sum(y_resampled==1)} class 1")
    print(f"  Total training: {len(y_resampled)}")

    # Re-normalize
    X_resampled_norm = scaler.transform(X_resampled)

    # Reshape to CNN format
    X_resampled_cnn = X_resampled_norm.reshape(-1, 11, 1, 1)

    return (X_resampled_cnn,
            to_categorical(y_resampled, 2),
            X_val_raw,
            to_categorical(y_val_raw, 2),
            X_test_raw,
            to_categorical(y_test_raw, 2),
            y_test_raw)


# ─────────────────────────────────────────
# DATASET 3 — DOUBLE-BALANCED
# SMOTE applied twice — both classes doubled
# Total: 2516 samples
# ─────────────────────────────────────────
def prepare_double_balanced(X_train_raw, y_train_raw,
                             X_val_raw, y_val_raw,
                             X_test_raw, y_test_raw,
                             scaler):
    """
    Apply SMOTE twice to double the dataset size.
    Paper: 2516 total samples.
    Achieves 97.42% accuracy in paper.
    """
    print("\nDataset: DOUBLE-BALANCED (SMOTE x2)")

    X_tr_2d = X_train_raw.squeeze(axis=(2, 3))
    X_tr_orig = scaler.inverse_transform(X_tr_2d)

    print(f"  Before SMOTE: "
          f"{np.sum(y_train_raw==0)} class 0, "
          f"{np.sum(y_train_raw==1)} class 1")

    # First SMOTE pass — balance classes
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_tr_orig, y_train_raw)

    print(f"  After SMOTE 1: "
          f"{np.sum(y_bal==0)} class 0, "
          f"{np.sum(y_bal==1)} class 1")

    # Second SMOTE pass — double both classes
    smote2 = SMOTE(
        random_state=42,
        sampling_strategy={
            0: np.sum(y_bal==0) * 2,
            1: np.sum(y_bal==1) * 2
        }
    )
    X_double, y_double = smote2.fit_resample(X_bal, y_bal)

    print(f"  After SMOTE 2: "
          f"{np.sum(y_double==0)} class 0, "
          f"{np.sum(y_double==1)} class 1")
    print(f"  Total training: {len(y_double)}")

    # Re-normalize and reshape
    X_double_norm = scaler.transform(X_double)
    X_double_cnn  = X_double_norm.reshape(-1, 11, 1, 1)

    return (X_double_cnn,
            to_categorical(y_double, 2),
            X_val_raw,
            to_categorical(y_val_raw, 2),
            X_test_raw,
            to_categorical(y_test_raw, 2),
            y_test_raw)


# ─────────────────────────────────────────
# DATASET 4 — FRAMINGHAM
# External generalizability test
# 3658 records: 3101 negative, 557 positive
# Paper: tests model on different source
# ─────────────────────────────────────────
def prepare_framingham(filepath, scaler):
    """
    Load and prepare Framingham Heart Study dataset.
    Tests generalizability of the trained model.
    Paper: 3658 records, 84.77% / 15.23% split.

    Note: Framingham has different feature names.
    We map available features to our 11-feature format.
    Missing features are imputed with column median.

    Parameters
    ----------
    filepath : str
        Path to Framingham CSV file.
    scaler : StandardScaler
        Scaler fitted on original training data.

    Returns
    -------
    X_test_cnn, y_test_raw or None if file not found
    """
    import pandas as pd

    if not os.path.exists(filepath):
        print(f"\nFramingham dataset not found at {filepath}")
        print("  Skipping Framingham evaluation.")
        print("  To include it, download from:")
        print("  https://www.kaggle.com/datasets/"
              "amanajmera1/framingham-heart-study-dataset")
        return None, None

    print(f"\nDataset: FRAMINGHAM")
    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Framingham target column
    # 'TenYearCHD' = 10-year coronary heart disease risk
    target_col = 'TenYearCHD'
    if target_col not in df.columns:
        # Try alternative names
        for col in ['target', 'HeartDisease', 'CHD']:
            if col in df.columns:
                target_col = col
                break

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col].values.astype(int)

    print(f"  Class 0 (no risk): {np.sum(y_raw==0)} "
          f"({np.sum(y_raw==0)/len(y_raw)*100:.2f}%)")
    print(f"  Class 1 (risk):    {np.sum(y_raw==1)} "
          f"({np.sum(y_raw==1)/len(y_raw)*100:.2f}%)")

    # Map Framingham features to our 11-feature format
    # Our features: age, sex, chest pain type, resting bp s,
    # cholesterol, fasting blood sugar, resting ecg,
    # max heart rate, exercise angina, oldpeak, ST slope

    # Framingham features available:
    # male, age, education, currentSmoker, cigsPerDay,
    # BPMeds, prevalentStroke, prevalentHyp, diabetes,
    # totChol, sysBP, diaBP, BMI, heartRate, glucose

    X = np.zeros((len(df), 11))

    # age (feature 0)
    if 'age' in df.columns:
        X[:, 0] = df['age'].fillna(df['age'].median())

    # sex — male column in Framingham (1=male, 0=female)
    if 'male' in df.columns:
        X[:, 1] = df['male'].fillna(0)

    # chest pain type — not available, use 0 (asymptomatic)
    X[:, 2] = 4  # asymptomatic (most common in paper data)

    # resting bp s — sysBP
    if 'sysBP' in df.columns:
        X[:, 3] = df['sysBP'].fillna(
            df['sysBP'].median()
        ).clip(80, 200)

    # cholesterol — totChol
    if 'totChol' in df.columns:
        X[:, 4] = df['totChol'].fillna(
            df['totChol'].median()
        ).clip(0, 603)

    # fasting blood sugar — glucose proxy
    if 'glucose' in df.columns:
        X[:, 5] = (df['glucose'].fillna(
            df['glucose'].median()
        ) > 120).astype(int)

    # resting ecg — not available, use 0 (normal)
    X[:, 6] = 0

    # max heart rate — heartRate
    if 'heartRate' in df.columns:
        X[:, 7] = df['heartRate'].fillna(
            df['heartRate'].median()
        ).clip(60, 202)

    # exercise angina — not available, use 0
    X[:, 8] = 0

    # oldpeak — not available, use 0
    X[:, 9] = 0

    # ST slope — not available, use 2 (flat, most common)
    X[:, 10] = 2

    # Normalize using original scaler
    X_norm = scaler.transform(X)
    X_cnn  = X_norm.reshape(-1, 11, 1, 1)

    print(f"  Prepared {len(X_cnn)} samples for evaluation")

    return X_cnn, y_raw


# ─────────────────────────────────────────
# TRAIN AND EVALUATE ON DATASET CONFIG
# ─────────────────────────────────────────
def train_and_evaluate(hyperparams,
                       X_train, y_train_cat,
                       X_val, y_val_cat,
                       X_test, y_test_raw,
                       dataset_name,
                       n_attempts=5):
    """
    Train model with given hyperparameters on
    augmented dataset and evaluate on test set.
    """
    print(f"\nTraining on {dataset_name}...")

    model = train_model_with_retries(
        hyperparams=hyperparams,
        X_train=X_train,
        y_train=y_train_cat,
        X_val=X_val,
        y_val=y_val_cat,
        n_attempts=n_attempts,
        verbose=True,
        label=dataset_name
    )

    if model is None:
        print(f"  WARNING: Model training failed for "
              f"{dataset_name}")
        return None

    # Evaluate on test set
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_test_raw, y_pred)

    print(f"\n  {dataset_name} Test Results:")
    print(f"  Accuracy:    {metrics['accuracy']:.2f}%")
    print(f"  F1-Score:    {metrics['f1']:.2f}%")
    print(f"  Sensitivity: {metrics['sensitivity']:.2f}%")
    print(f"  Precision:   {metrics['precision']:.2f}%")
    print(f"  NPV:         {metrics['npv']:.2f}%")
    print(f"  MCC:         {metrics['mcc']:.2f}%")
    print(f"  Kappa:       {metrics['kappa']:.2f}%")

    return metrics


# ─────────────────────────────────────────
# PLOT TABLE 10 — Figure 16
# Performance comparison across dataset
# configurations
# ─────────────────────────────────────────
def plot_dataset_comparison(results_dict,
                             save_path='outputs/figures/dataset_comparison.png'):
    """
    Bar chart comparing performance across
    4 dataset configurations.
    Reproduces Figure 16 from the paper.
    """
    datasets = list(results_dict.keys())
    metrics  = ['accuracy', 'f1', 'sensitivity',
                'precision', 'npv', 'mcc', 'kappa']
    metric_labels = ['Accuracy', 'F1-Score',
                     'Sensitivity', 'Precision',
                     'NPV', 'MCC', 'Kappa']

    colors = ['#1f77b4', '#ff7f0e',
              '#2ca02c', '#d62728']

    x     = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (dataset, color) in enumerate(
            zip(datasets, colors)):
        if results_dict[dataset] is None:
            continue
        values = [results_dict[dataset][m]
                  for m in metrics]
        ax.bar(
            x + i * width, values,
            width, label=dataset,
            color=color, alpha=0.85
        )

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(
        'Performance Comparison — Different Dataset Configurations\n'
        'Table 10: GWO-WOA-AOA CNN on Original, '
        'Balanced, Double-Balanced, Framingham',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim([60, 100])
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Dataset comparison chart saved -> {save_path}")


# ─────────────────────────────────────────
# PRINT TABLE 10
# ─────────────────────────────────────────
def print_table_10(results_dict):
    """Print Table 10 from paper."""
    import csv

    print("\n" + "=" * 85)
    print("TABLE 10 — Performance on Different Dataset Configurations")
    print("=" * 85)
    print(f"{'Dataset':<20} {'Acc%':>7} {'F1%':>7} "
          f"{'Sens%':>7} {'Prec%':>7} "
          f"{'NPV%':>7} {'MCC%':>7} {'Kappa%':>7}")
    print("-" * 85)

    csv_path = 'outputs/results/table10_dataset_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dataset', 'Accuracy', 'F1_Score',
            'Sensitivity', 'Precision',
            'NPV', 'MCC', 'Kappa'
        ])

        for dataset, m in results_dict.items():
            if m is None:
                print(f"{dataset:<20} {'N/A':>7}")
                continue
            print(
                f"{dataset:<20} "
                f"{m['accuracy']:>7.2f} "
                f"{m['f1']:>7.2f} "
                f"{m['sensitivity']:>7.2f} "
                f"{m['precision']:>7.2f} "
                f"{m['npv']:>7.2f} "
                f"{m['mcc']:>7.2f} "
                f"{m['kappa']:>7.2f}"
            )
            writer.writerow([
                dataset,
                m['accuracy'], m['f1'],
                m['sensitivity'], m['precision'],
                m['npv'], m['mcc'], m['kappa']
            ])

    print("=" * 85)
    print(f"Table 10 saved -> {csv_path}")


# ─────────────────────────────────────────
# FULL SMOTE PIPELINE
# ─────────────────────────────────────────
def run_smote_analysis(best_hyperparams,
                       X_train_raw, y_train_raw,
                       X_val_raw, y_val_raw,
                       X_test_raw, y_test_raw,
                       scaler,
                       framingham_path=None,
                       n_attempts=5):
    """
    Run full SMOTE augmentation analysis.
    Reproduces Table 10 and Figure 16.

    Parameters
    ----------
    best_hyperparams : dict
        Optimal hyperparameters from hybrid optimizer.
    X_train_raw : array, shape (N, 11, 1, 1)
        Normalized training features.
    y_train_raw : array, shape (N,)
        Integer training labels.
    scaler : StandardScaler
        Fitted on original training data.
    framingham_path : str or None
        Path to Framingham CSV, or None to skip.
    """
    print("\n" + "=" * 50)
    print("SMOTE DATA AUGMENTATION ANALYSIS")
    print("Paper Section 4.5")
    print("=" * 50)

    results_dict = {}

    # ── Config 1: Original ──
    (X_tr, y_tr, X_vl, y_vl,
     X_te, y_te, y_te_raw) = prepare_original(
        X_train_raw, y_train_raw,
        X_val_raw,   y_val_raw,
        X_test_raw,  y_test_raw
    )
    results_dict['Original'] = train_and_evaluate(
        best_hyperparams,
        X_tr, y_tr, X_vl, y_vl,
        X_te, y_te_raw,
        'Original', n_attempts
    )

    # ── Config 2: Balanced ──
    (X_tr, y_tr, X_vl, y_vl,
     X_te, y_te, y_te_raw) = prepare_balanced(
        X_train_raw, y_train_raw,
        X_val_raw,   y_val_raw,
        X_test_raw,  y_test_raw,
        scaler
    )
    results_dict['Balanced'] = train_and_evaluate(
        best_hyperparams,
        X_tr, y_tr, X_vl, y_vl,
        X_te, y_te_raw,
        'Balanced', n_attempts
    )

    # ── Config 3: Double-Balanced ──
    (X_tr, y_tr, X_vl, y_vl,
     X_te, y_te, y_te_raw) = prepare_double_balanced(
        X_train_raw, y_train_raw,
        X_val_raw,   y_val_raw,
        X_test_raw,  y_test_raw,
        scaler
    )
    results_dict['Double-Balanced'] = train_and_evaluate(
        best_hyperparams,
        X_tr, y_tr, X_vl, y_vl,
        X_te, y_te_raw,
        'Double-Balanced', n_attempts
    )

    # ── Config 4: Framingham ──
    if framingham_path:
        X_fram, y_fram = prepare_framingham(
            framingham_path, scaler
        )
        if X_fram is not None:
            # Use original training data for Framingham
            # Then test on Framingham test set
            (X_tr, y_tr, X_vl, y_vl,
             X_te, y_te, y_te_raw) = prepare_original(
                X_train_raw, y_train_raw,
                X_val_raw,   y_val_raw,
                X_test_raw,  y_test_raw
            )
            # Train on original, test on Framingham
            model = train_model_with_retries(
                hyperparams=best_hyperparams,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_vl,
                y_val=y_vl,
                n_attempts=n_attempts,
                verbose=True,
                label='Framingham'
            )
            if model is not None:
                y_prob = model.predict(X_fram, verbose=0)
                y_pred = np.argmax(y_prob, axis=1)
                metrics = compute_metrics(y_fram, y_pred)
                results_dict['Framingham'] = metrics
                print(f"\n  Framingham Test Results:")
                for k, v in metrics.items():
                    if k not in ['TP','TN','FP','FN','cm']:
                        print(f"  {k}: {v:.2f}%")
            else:
                results_dict['Framingham'] = None
        else:
            results_dict['Framingham'] = None
    else:
        print("\nFramingham dataset path not provided.")
        print("Skipping Framingham evaluation.")
        results_dict['Framingham'] = None

    # ── Print Table 10 ──
    print_table_10(results_dict)

    # ── Plot Figure 16 ──
    plot_dataset_comparison(results_dict)

    print("\nSMOTE Analysis Complete.")
    print("  outputs/figures/dataset_comparison.png (Figure 16)")
    print("  outputs/results/table10_dataset_comparison.csv")

    return results_dict