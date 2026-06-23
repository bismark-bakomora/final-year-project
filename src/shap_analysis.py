import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import shap
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs/figures', exist_ok=True)

# ─────────────────────────────────────────
# FEATURE NAMES
# Must match the order in the dataset
# and paper Figure 14
# ─────────────────────────────────────────
FEATURE_NAMES = [
    'age',
    'sex',
    'chest pain type',
    'resting bp s',
    'cholesterol',
    'fasting blood sugar',
    'resting ecg',
    'max heart rate',
    'exercise angina',
    'oldpeak',
    'ST slope'
]


# ─────────────────────────────────────────
# COMPUTE SHAP VALUES
# Uses GradientExplainer for Keras CNN models
# Paper Section 4.4
# ─────────────────────────────────────────
def compute_shap_values(model, X_train, X_test,
                        n_background=100,
                        n_explain=200):
    """
    Compute SHAP values using GradientExplainer.

    GradientExplainer is suitable for deep learning
    models and uses a background dataset to estimate
    the expected output.

    Parameters
    ----------
    model : keras Model
        Trained GWO-WOA-AOA CNN model.
    X_train : array, shape (N, 11, 1, 1)
        Training data used as SHAP background.
    X_test : array, shape (N, 11, 1, 1)
        Test data to explain.
    n_background : int
        Number of background samples.
        More = more accurate but slower.
    n_explain : int
        Number of test samples to explain.

    Returns
    -------
    shap_values : array, shape (n_explain, 11)
        SHAP values for positive class (disease=1).
    X_explain : array, shape (n_explain, 11)
        Feature values for explained samples.
    """
    print("Computing SHAP values...")
    print(f"  Background samples: {n_background}")
    print(f"  Samples to explain: {n_explain}")

    # Sample background data
    bg_idx = np.random.choice(
        len(X_train), n_background, replace=False
    )
    background = X_train[bg_idx]

    # Samples to explain
    exp_idx = np.random.choice(
        len(X_test), min(n_explain, len(X_test)),
        replace=False
    )
    X_explain = X_test[exp_idx]

    # GradientExplainer — works with Keras models
    explainer = shap.GradientExplainer(
        model, background
    )

    # Compute SHAP values
    # Returns list of arrays [class0_shap, class1_shap]
    # each shape (n_explain, 11, 1, 1)
    raw_shap = explainer.shap_values(X_explain)

    # We focus on class 1 (heart disease present)
    # Squeeze from (N, 11, 1, 1) or (N, 11, 1, 1, 2) to (N, 11)
    if isinstance(raw_shap, list):
        # Multi-output list: take class 1 (disease)
        shap_values = raw_shap[1].squeeze()
        if shap_values.ndim > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
    else:
        arr = np.array(raw_shap).squeeze()
        # If shape is (N, features, classes), take class 1
        if arr.ndim == 3:
            shap_values = arr[..., 1]
        else:
            shap_values = arr

    # Squeeze X_explain from (N, 11, 1, 1) to (N, 11)
    X_explain_2d = X_explain.squeeze(axis=(2, 3))

    print(f"  SHAP values shape: {shap_values.shape}")
    print("  SHAP computation complete.")

    return shap_values, X_explain_2d


# ─────────────────────────────────────────
# PLOT FIGURE 14 — SHAP Feature Importance
# Bar chart of mean absolute SHAP values
# Paper: "ranked by their average absolute
# SHAP values"
# ─────────────────────────────────────────
def plot_shap_importance(shap_values,
                         save_path='outputs/figures/shap_importance.png'):
    """
    Plot SHAP feature importance bar chart.
    Reproduces Figure 14 from the paper.

    Features sorted by mean |SHAP value|,
    highest at top (horizontal bar chart).
    """
    # Mean absolute SHAP value per feature (collapse samples and classes)
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, -1)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    # Sort ascending so highest appears at top
    # when plotted as horizontal bar chart
    sorted_idx  = np.argsort(mean_abs_shap)
    sorted_vals = mean_abs_shap[sorted_idx]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(
        range(len(sorted_names)),
        sorted_vals,
        color='#4472C4',
        alpha=0.85,
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel(
        'mean(|SHAP value|) (average impact on model output magnitude)',
        fontsize=11
    )
    ax.set_title(
        'SHAP Feature Importance Ranking\n'
        'Importance levels ranked by average absolute SHAP values',
        fontsize=12, fontweight='bold', pad=12
    )
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP importance plot saved -> {save_path}")

    # Print values matching paper text
    print("\nSHAP Feature Importance (mean |SHAP|):")
    print("-" * 45)
    for i in reversed(sorted_idx):
        print(f"  {FEATURE_NAMES[i]:25s}: {mean_abs_shap[i]:.3f}")

    return mean_abs_shap


# ─────────────────────────────────────────
# PLOT FIGURE 15 — SHAP Summary Plot
# Beeswarm plot showing each sample's SHAP
# value coloured by feature value
# Paper: "detailed visualization of the
# impact of each feature on individual
# predictions"
# ─────────────────────────────────────────
def plot_shap_summary(shap_values, X_explain,
                      save_path='outputs/figures/shap_summary.png'):
    """
    Plot SHAP summary beeswarm plot.
    Reproduces Figure 15 from the paper.

    Each dot = one patient.
    X axis = SHAP value (impact on prediction).
    Colour = feature value (red=high, blue=low).
    Features sorted by mean |SHAP value|.
    """
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=FEATURE_NAMES,
        plot_type='dot',        # beeswarm
        show=False,             # don't display — save instead
        color_bar=True,
        plot_size=(10, 8),
        alpha=0.7,
        max_display=11          # show all 11 features
    )

    plt.title(
        'SHAP Summary Plot\n'
        'Detailed visualization of feature impact on individual predictions',
        fontsize=12, fontweight='bold', pad=12
    )
    plt.xlabel('SHAP value (impact on model output)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved -> {save_path}")


# ─────────────────────────────────────────
# SAVE SHAP VALUES TO CSV
# ─────────────────────────────────────────
def save_shap_results(shap_values):
    """Save mean absolute SHAP values to CSV."""
    import csv
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    csv_path = 'outputs/results/shap_importance.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'Mean_Abs_SHAP',
                         'Rank'])
        for rank, i in enumerate(sorted_idx, 1):
            writer.writerow([
                FEATURE_NAMES[i],
                round(float(mean_abs_shap[i]), 4),
                rank
            ])
    print(f"SHAP results saved -> {csv_path}")


# ─────────────────────────────────────────
# FULL SHAP PIPELINE
# ─────────────────────────────────────────
def run_shap_analysis(model, X_train, X_test,
                      n_background=100,
                      n_explain=200):
    """
    Run complete SHAP analysis pipeline.
    Produces Figure 14 and Figure 15 from paper.

    Parameters
    ----------
    model : keras Model
        Trained GWO-WOA-AOA CNN model.
    X_train : array, shape (N, 11, 1, 1)
    X_test  : array, shape (N, 11, 1, 1)
    n_background : int
        Background samples for GradientExplainer.
    n_explain : int
        Test samples to explain.
    """
    print("\n" + "=" * 50)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("Paper Section 4.4")
    print("=" * 50)

    # Compute SHAP values
    shap_values, X_explain = compute_shap_values(
        model, X_train, X_test,
        n_background=n_background,
        n_explain=n_explain
    )

    # Figure 14 — Feature importance bar chart
    print("\nGenerating Figure 14 — Feature Importance...")
    mean_abs_shap = plot_shap_importance(shap_values)

    # Figure 15 — Summary beeswarm plot
    print("\nGenerating Figure 15 — Summary Plot...")
    plot_shap_summary(shap_values, X_explain)

    # Save to CSV
    save_shap_results(shap_values)

    print("\nSHAP Analysis Complete.")
    print("  outputs/figures/shap_importance.png  (Figure 14)")
    print("  outputs/figures/shap_summary.png     (Figure 15)")
    print("  outputs/results/shap_importance.csv")

    return shap_values, mean_abs_shap