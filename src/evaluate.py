import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, recall_score, precision_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# OUTPUT DIRECTORIES
# ─────────────────────────────────────────
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)


# ─────────────────────────────────────────
# PERFORMANCE METRICS
# Paper Section 4.1, Equations 24-30
# ─────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all metrics from paper Section 4.1.

    Paper Equations:
    24: Accuracy  = (TP+TN) / (TP+TN+FP+FN)
    25: F1        = 2*P*R / (P+R)
    26: Sensitivity = TP / (TP+FN)
    27: Precision   = TP / (TP+FP)
    28: NPV         = TN / (TN+FN)
    29: MCC         = (TP*TN - FP*FN) /
                      sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    30: Kappa       = (Po - Pe) / (1 - Pe)

    Parameters
    ----------
    y_true : array, true binary labels
    y_pred : array, predicted binary labels
    y_prob : array, predicted probabilities (for ROC)

    Returns
    -------
    dict of all metrics
    """
    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Equation 24 — Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Equation 25 — F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Equation 26 — Sensitivity / Recall
    sensitivity = recall_score(
        y_true, y_pred, zero_division=0
    )

    # Equation 27 — Precision
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    # Equation 28 — NPV
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    # Equation 29 — MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Equation 30 — Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'accuracy':    round(accuracy * 100, 2),
        'f1':          round(f1 * 100, 2),
        'sensitivity': round(sensitivity * 100, 2),
        'precision':   round(precision * 100, 2),
        'npv':         round(npv * 100, 2),
        'mcc':         round(mcc * 100, 2),
        'kappa':       round(kappa * 100, 2),
        'TP': int(TP), 'TN': int(TN),
        'FP': int(FP), 'FN': int(FN),
        'cm': cm
    }


# ─────────────────────────────────────────
# EVALUATE MODEL ON TEST SET
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test_cat,
                   y_test_raw):
    """
    Evaluate trained model on held-out test set.
    Returns predictions and all metrics.
    """
    # Get predictions
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_prob_pos = y_prob[:, 1]  # probability of class 1

    # Compute all metrics
    metrics = compute_metrics(
        y_test_raw, y_pred, y_prob_pos
    )

    return y_pred, y_prob_pos, metrics


# ─────────────────────────────────────────
# PLOT CONFUSION MATRIX — Figure 9
# Paper: 6 confusion matrices side by side
# ─────────────────────────────────────────
def plot_confusion_matrix(cm, title, ax):
    """
    Plot a single confusion matrix.
    Matches paper Figure 9 style.
    Blue=correct, Pink=incorrect.
    """
    # Custom colormap matching paper colors
    colors = np.array([
        ['#4472C4', '#D9A0A8'],  # TN=blue, FP=pink
        ['#D9A0A8', '#4472C4']   # FN=pink, TP=blue
    ])

    ax.set_facecolor('white')

    # Draw colored cells
    for i in range(2):
        for j in range(2):
            color = colors[i][j]
            ax.add_patch(plt.Rectangle(
                (j, 1-i), 1, 1,
                color=color, transform=ax.transData
            ))
            ax.text(
                j + 0.5, 1.5 - i,
                str(cm[i, j]),
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                color='white'
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['No Risk (0)', 'Risk (1)'],
                       fontsize=9)
    ax.set_yticklabels(['Risk (1)', 'No Risk (0)'],
                       fontsize=9)
    ax.set_xlabel('Predicted Class', fontsize=9)
    ax.set_ylabel('True Class', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold',
                 pad=8)
    ax.tick_params(length=0)


def plot_all_confusion_matrices(results_dict):
    """
    Plot all 6 confusion matrices as in Figure 9.

    Parameters
    ----------
    results_dict : dict
        Keys are method names, values are metric dicts
        containing 'cm' (confusion matrix).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        'Confusion Matrices — Heart Disease Prediction',
        fontsize=14, fontweight='bold', y=1.02
    )

    methods = list(results_dict.keys())
    axes_flat = axes.flatten()

    for idx, method in enumerate(methods):
        cm = results_dict[method]['cm']
        plot_confusion_matrix(cm, method, axes_flat[idx])

    plt.tight_layout()
    save_path = 'outputs/figures/confusion_matrices.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved → {save_path}")


# ─────────────────────────────────────────
# PLOT ROC CURVES — Figure 10
# ─────────────────────────────────────────
def plot_roc_curves(roc_data_dict):
    """
    Plot ROC curves for all methods.
    Matches paper Figure 10 style.

    Parameters
    ----------
    roc_data_dict : dict
        Keys are method names.
        Values are (fpr, tpr, auc_score) tuples.
    """
    plt.figure(figsize=(8, 7))

    # Line styles matching paper Figure 10
    styles = {
        'NO-CNN':          ('o-',  '#1f77b4'),
        'RIME-CNN':        ('s-',  '#ff7f0e'),
        'AOA-CNN':         ('^-',  '#2ca02c'),
        'WOA-CNN':         ('D-',  '#d62728'),
        'GWO-CNN':         ('v-',  '#9467bd'),
        'GWO-WOA-AOA-CNN': ('*-',  '#8c564b'),
    }

    for method, (fpr, tpr, auc_score) in \
            roc_data_dict.items():
        marker, color = styles.get(
            method, ('-', 'gray')
        )
        plt.plot(
            fpr, tpr,
            marker,
            color=color,
            label=f'{method} (AUC={auc_score:.3f})',
            linewidth=2,
            markersize=4,
            markevery=10
        )

    # Random guess line
    plt.plot([0, 1], [0, 1], 'k--',
             linewidth=1, label='Random Guess')

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curves — Heart Disease Prediction',
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.tight_layout()

    save_path = 'outputs/figures/roc_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved → {save_path}")


# ─────────────────────────────────────────
# PLOT CONVERGENCE CURVES — Figure 8
# ─────────────────────────────────────────
def plot_convergence_curves(curves_dict):
    """
    Plot convergence curves for all methods.
    Matches paper Figure 8 style.

    Parameters
    ----------
    curves_dict : dict
        Keys are method names.
        Values are lists of fitness per iteration.
    """
    plt.figure(figsize=(9, 6))

    styles = {
        'RIME':        ('+-',  '#1f77b4'),
        'AOA':         ('o-',  '#ff7f0e'),
        'GWO':         ('s-',  '#2ca02c'),
        'WOA':         ('^-',  '#d62728'),
        'GWO-WOA-AOA': ('*-',  '#9467bd'),
    }

    for method, curve in curves_dict.items():
        iterations = list(range(1, len(curve) + 1))
        marker, color = styles.get(
            method, ('-', 'gray')
        )
        plt.plot(
            iterations, curve,
            marker,
            color=color,
            label=method,
            linewidth=2,
            markersize=5
        )

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title(
        'Convergence Curves — GWO-WOA-AOA vs Others',
        fontsize=13, fontweight='bold'
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = 'outputs/figures/convergence_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence curves saved → {save_path}")


# ─────────────────────────────────────────
# PRINT RESULTS TABLE — Table 7
# ─────────────────────────────────────────
def print_results_table(results_dict):
    """
    Print comparison table matching Table 7.
    Also saves to CSV.
    """
    print("\n" + "=" * 85)
    print("TABLE 7 — Classification Results Comparison")
    print("=" * 85)
    print(f"{'Method':<22} {'Acc%':>7} {'F1%':>7} "
          f"{'Sens%':>7} {'Prec%':>7} "
          f"{'NPV%':>7} {'MCC%':>7} {'Kappa%':>7}")
    print("-" * 85)

    # Save to CSV
    import csv
    csv_path = 'outputs/results/metrics_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Accuracy', 'F1_Score',
            'Sensitivity', 'Precision',
            'NPV', 'MCC', 'Kappa'
        ])

        for method, m in results_dict.items():
            print(
                f"{method:<22} "
                f"{m['accuracy']:>7.2f} "
                f"{m['f1']:>7.2f} "
                f"{m['sensitivity']:>7.2f} "
                f"{m['precision']:>7.2f} "
                f"{m['npv']:>7.2f} "
                f"{m['mcc']:>7.2f} "
                f"{m['kappa']:>7.2f}"
            )
            writer.writerow([
                method,
                m['accuracy'], m['f1'],
                m['sensitivity'], m['precision'],
                m['npv'], m['mcc'], m['kappa']
            ])

    print("=" * 85)
    print(f"\nResults saved → {csv_path}")


# ─────────────────────────────────────────
# STATISTICAL SIGNIFICANCE — Table 8
# Wilcoxon signed-rank test, 5-fold CV
# Paper Section 4.2
# ─────────────────────────────────────────
def statistical_significance_test(
        models_dict, X, y_raw, cv=5):
    """
    Wilcoxon signed-rank test with 5-fold CV.
    Matches paper Table 8.

    Parameters
    ----------
    models_dict : dict
        Keys=method names, values=compiled models
    X : array
        Full feature array (train+test combined)
    y_raw : array
        Full integer labels
    cv : int
        Number of folds. Paper uses 5.
    """
    print("\n" + "=" * 70)
    print("TABLE 8 — Statistical Significance (Wilcoxon, CV=5)")
    print("=" * 70)

    skf = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=42
    )

    # Collect CV scores per method
    cv_scores = {}
    for method_name, model in models_dict.items():
        scores = []
        for train_idx, val_idx in skf.split(X, y_raw):
            X_tr = X[train_idx]
            y_tr = y_raw[train_idx]
            X_vl = X[val_idx]
            y_vl = y_raw[val_idx]

            # Predict
            y_prob = model.predict(
                X_vl, verbose=0
            )
            y_pred = np.argmax(y_prob, axis=1)
            acc = np.mean(y_pred == y_vl)
            scores.append(acc)

        cv_scores[method_name] = np.array(scores)

    # NO-CNN is the reference (baseline)
    reference = cv_scores.get('NO-CNN')

    print(f"\n{'Method':<22} {'Mean Acc%':>10} "
          f"{'95% CI':>20} {'p-value':>10}")
    print("-" * 70)

    import csv
    csv_path = 'outputs/results/statistical_analysis.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Mean_Accuracy',
            'CI_Lower', 'CI_Upper', 'p_value'
        ])

        for method, scores in cv_scores.items():
            mean_acc = scores.mean() * 100
            ci_lower = (scores.mean() -
                        1.96 * scores.std() /
                        np.sqrt(cv)) * 100
            ci_upper = (scores.mean() +
                        1.96 * scores.std() /
                        np.sqrt(cv)) * 100

            # Wilcoxon test vs NO-CNN baseline
            if method == 'NO-CNN' or reference is None:
                p_val = '-'
                sig = ''
            else:
                try:
                    _, p_val = wilcoxon(
                        scores, reference
                    )
                    sig = '*' if p_val < 0.05 else ''
                    p_val = f"{p_val:.3f}{sig}"
                except Exception:
                    p_val = 'N/A'

            ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
            print(f"{method:<22} {mean_acc:>10.2f} "
                  f"{ci_str:>20} {str(p_val):>10}")

            writer.writerow([
                method, round(mean_acc, 2),
                round(ci_lower, 2),
                round(ci_upper, 2),
                p_val
            ])

    print("=" * 70)
    print("* p<0.05 = statistically significant")
    print(f"\nStatistical analysis saved → {csv_path}")


# ─────────────────────────────────────────
# SAVE HYPERPARAMETER RESULTS — Table 6
# ─────────────────────────────────────────
def save_hyperparameter_results(hp_dict):
    """
    Save hyperparameter results matching Table 6.

    Parameters
    ----------
    hp_dict : dict
        Keys=method names, values=hyperparameter dicts
    """
    import csv
    csv_path = 'outputs/results/hyperparameters.csv'

    # Get all hyperparameter keys
    hp_keys = [
        'filters', 'kernel_size', 'pooling_size',
        'neurons', 'dropout_rate', 'learning_rate',
        'batch_size', 'optimizer', 'max_epoch'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Hyperparameter'] +
                        list(hp_dict.keys()))

        for key in hp_keys:
            row = [key]
            for method, hp in hp_dict.items():
                row.append(hp.get(key, 'N/A'))
            writer.writerow(row)

    print(f"Hyperparameters saved → {csv_path}")


# ─────────────────────────────────────────
# PLOT PERFORMANCE BAR CHART
# ─────────────────────────────────────────
def plot_performance_comparison(results_dict):
    """
    Bar chart comparing all methods across
    all metrics. Supplements Table 7.
    """
    methods = list(results_dict.keys())
    metrics = ['accuracy', 'f1', 'sensitivity',
               'precision', 'npv', 'mcc', 'kappa']
    metric_labels = [
        'Accuracy', 'F1-Score', 'Sensitivity',
        'Precision', 'NPV', 'MCC', 'Kappa'
    ]

    x = np.arange(len(metrics))
    width = 0.12
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b'
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (method, color) in enumerate(
            zip(methods, colors)):
        values = [results_dict[method][m]
                  for m in metrics]
        bars = ax.bar(
            x + i * width, values,
            width, label=method,
            color=color, alpha=0.85
        )

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(
        'Performance Comparison — All Methods',
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim([75, 100])
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    save_path = \
        'outputs/figures/performance_comparison.png'
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Performance chart saved → {save_path}")


# ─────────────────────────────────────────
# FULL EVALUATION PIPELINE
# ─────────────────────────────────────────
def run_full_evaluation(models_dict,
                        X_test, y_test_cat,
                        y_test_raw,
                        convergence_curves=None,
                        hp_dict=None):
    """
    Run complete evaluation pipeline.
    Produces all figures and tables from paper.

    Parameters
    ----------
    models_dict : dict
        Keys=method names, values=trained Keras models
    X_test : array
        Test features (N, 11, 1, 1)
    y_test_cat : array
        One-hot test labels (N, 2)
    y_test_raw : array
        Integer test labels (N,)
    convergence_curves : dict, optional
        Convergence curves per method
    hp_dict : dict, optional
        Hyperparameters per method
    """
    print("\n" + "=" * 50)
    print("RUNNING FULL EVALUATION")
    print("Paper Section 4")
    print("=" * 50)

    results_dict = {}
    roc_data_dict = {}

    # Evaluate each model
    for method, model in models_dict.items():
        print(f"\nEvaluating {method}...")

        y_pred, y_prob, metrics = evaluate_model(
            model, X_test, y_test_cat, y_test_raw
        )
        results_dict[method] = metrics

        # ROC data
        fpr, tpr, _ = roc_curve(y_test_raw, y_prob)
        auc_score = auc(fpr, tpr)
        roc_data_dict[method] = (fpr, tpr, auc_score)

        print(f"  Accuracy:    {metrics['accuracy']:.2f}%")
        print(f"  F1-Score:    {metrics['f1']:.2f}%")
        print(f"  Sensitivity: {metrics['sensitivity']:.2f}%")
        print(f"  Precision:   {metrics['precision']:.2f}%")
        print(f"  NPV:         {metrics['npv']:.2f}%")
        print(f"  MCC:         {metrics['mcc']:.2f}%")
        print(f"  Kappa:       {metrics['kappa']:.2f}%")

    # Print and save Table 7
    print_results_table(results_dict)

    # Plot Figure 9 — Confusion matrices
    plot_all_confusion_matrices(results_dict)

    # Plot Figure 10 — ROC curves
    plot_roc_curves(roc_data_dict)

    # Plot performance bar chart
    plot_performance_comparison(results_dict)

    # Plot Figure 8 — Convergence curves
    if convergence_curves:
        plot_convergence_curves(convergence_curves)

    # Save Table 6 — Hyperparameters
    if hp_dict:
        save_hyperparameter_results(hp_dict)

    print("\nAll outputs saved to outputs/ folder.")
    return results_dict