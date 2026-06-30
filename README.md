# Heart Disease Prediction using Hybrid GWO-WOA-AOA CNN

A final-year research implementation that reproduces and extends the hybrid metaheuristic approach described in:

> **Lale et al.** — *A novel hybrid metaheuristic for optimizing hyperparameters of convolutional neural network in heart disease prediction* (Cluster Computing, 2026).

The system automatically tunes a 2D convolutional neural network (CNN) for binary heart-disease classification using a **sequential three-stage optimizer**: Grey Wolf Optimizer (GWO) → Whale Optimization Algorithm (WOA) → Arithmetic Optimization Algorithm (AOA). Results are compared against standalone optimizers (GWO, WOA, AOA, RIME) and a non-optimised baseline (NO-CNN).

This repository is structured as an **industry-style ML pipeline**: staged CLI commands, central configuration, structured logging, on-disk artifacts, and memory-safe execution so long experiments do not exhaust RAM or terminate silently.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Scientific background](#scientific-background)
3. [System architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Quick start](#quick-start)
7. [CLI reference](#cli-reference)
8. [Configuration](#configuration)
9. [Pipeline stages explained](#pipeline-stages-explained)
10. [Outputs and artifacts](#outputs-and-artifacts)
11. [Expected runtimes](#expected-runtimes)
12. [Project structure](#project-structure)
13. [Reproducing paper results](#reproducing-paper-results)
14. [Troubleshooting](#troubleshooting)
15. [References](#references)

---

## What this project does

| Goal | How |
|------|-----|
| Predict heart disease (yes/no) | 2D-CNN on 11 clinical features |
| Find best CNN hyperparameters | Metaheuristic search minimising \(f(x) = 1 - \text{validation accuracy}\) |
| Compare methods | NO-CNN, GWO-CNN, WOA-CNN, AOA-CNN, RIME-CNN, GWO-WOA-AOA-CNN (Table 7) |
| Explain predictions | SHAP analysis on the hybrid model (Section 4.4) |
| Test data augmentation | SMOTE on original, balanced, and double-balanced sets (Section 4.5) |

**Important design choice:** The reference paper reports ~25–36 minutes **per individual optimizer run**. This project runs **one experiment per CLI command**, saves models to disk, and frees memory between stages. That matches the paper’s methodology and avoids the multi-hour, out-of-memory failures caused by running everything in a single script.

For a detailed runtime analysis, see `docs/Runtime_Analysis_Report.docx`.

---

## Scientific background

### The hybrid optimizer (Section 3.2.4)

The proposed method chains three nature-inspired algorithms:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  GWO stage  │ ──► │  WOA stage  │ ──► │  AOA stage  │
│ Exploration │     │ Exploitation│     │ Fine-tuning │
│ 10 iter     │     │ 10 iter     │     │ 10 iter     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                    Best hyperparameters
                           │
                    Final CNN model
```

- **GWO** (Section 3.2.1): global search across the hyperparameter space.
- **WOA** (Section 3.2.2): local refinement starting from GWO’s best solution.
- **AOA** (Section 3.2.3): precise fine-tuning starting from WOA’s best solution.

Each stage uses population size **ps = 20**. In hybrid mode each stage runs **10 iterations** (30 total). Standalone comparison algorithms run **30 iterations** each (Section 4.2).

### Fitness function (Equation 23, Section 3.3.2)

The optimisers minimise:

\[
f(x) = 1 - \frac{TP + TN}{TP + TN + FP + FN} = 1 - \text{validation accuracy}
\]

For each candidate hyperparameter vector \(x\), a CNN is built, trained on the training set (with early stopping on validation loss, patience = 5), and evaluated on the validation set. Lower fitness is better; 0 means perfect validation accuracy.

### CNN architecture (Section 3.3.1, Figure 7)

- Input shape: `(11, 1, 1)` — eleven features as a 2D tensor.
- Four blocks: `Conv2D → BatchNorm → ReLU → MaxPooling2D`.
- `GlobalAveragePooling2D` → dense layer → dropout → softmax (2 classes).

Nine hyperparameters are optimised (Table 5): filter configuration, kernel size, pooling size, FC neurons, dropout, learning rate, batch size, optimizer, and max epochs.

---

## System architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    src/pipeline.py                               │
│  Orchestrates stages · logging · artifacts · memory cleanup      │
└─┬──────────┬──────────┬──────────┬──────────┬────────────────────┘
  │          │          │          │          │
  ▼          ▼          ▼          ▼          ▼
preprocess  hybrid/    fitness    evaluate   shap / smote
            standalone  + CNN      + plots    (optional)
            optimizers
```

### Core modules

| Module | Responsibility |
|--------|----------------|
| `config.yaml` / `src/config.py` | Central settings (`paper` vs `quick` presets) |
| `src/logging_config.py` | Console + rotating file logs with stage tags |
| `src/memory_utils.py` | TensorFlow session cleanup between stages |
| `src/artifacts.py` | Save models, hyperparameters, run manifests |
| `src/pipeline.py` | End-to-end orchestration |
| `src/preprocess.py` | Data loading, cleaning, splitting, scaling |
| `src/cnn_model.py` | Dynamic CNN build/train; hyperparameter decode |
| `src/fitness.py` | Fitness evaluations during optimisation |
| `src/hybrid_optimizer.py` | GWO → WOA → AOA chain |
| `src/gwo.py`, `woa.py`, `aoa.py`, `rime.py` | Individual optimisers |
| `src/evaluate.py` | Metrics, confusion matrices, ROC, convergence plots |
| `src/shap_analysis.py` | SHAP explainability (Section 4.4) |
| `src/smote_analysis.py` | SMOTE experiments (Section 4.5) |

---

## Dataset

**Source:** Combined Cleveland + Hungary heart disease data (Statlog format).

| Property | Value |
|----------|-------|
| File | `data/raw/heart_statlog_cleveland_hungary_final.csv` |
| Samples | 1,190 |
| Features | 11 (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise angina, oldpeak, ST slope) |
| Target | Binary: 0 = no disease, 1 = disease |
| Split | 70% train / 10% validation / 20% test (stratified, Section 3.1) |
| Scaling | Z-score normalisation (fit on train only) |

Preprocessing handles invalid values (e.g. resting BP = 0, ST slope = 0) via median imputation or clipping, as described in Section 3.1.2.

---

## Installation

### Requirements

- **Python 3.11** (required for TensorFlow 2.15 on Windows)
- **16 GB RAM** recommended (paper hardware: Intel i7-6500U, 16 GB)
- CPU execution is supported; GPU is optional

### Steps

```powershell
# Clone or navigate to the project
cd final-year-project

# Create virtual environment (must use Python 3.11)
py -3.11 -m venv venv311

# Activate (Windows PowerShell)
.\venv311\Scripts\Activate.ps1

# Verify Python version
python --version   # should show 3.11.x

# Install dependencies (~300 MB download for TensorFlow)
pip install --upgrade pip
pip install -r requirements.txt --default-timeout=1000

# Verify TensorFlow
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
```

If the TensorFlow download times out, retry with a longer timeout or use a stable network connection. The wheel is ~300 MB.

---

## Quick start

### 1. Prepare data (run once)

```powershell
python main.py preprocess
```

Creates `data/processed/*.npy` and `models/scaler.pkl`.

### 2. Smoke test (~5 minutes)

Confirms the pipeline works end-to-end with tiny populations:

```powershell
python main.py run hybrid --preset quick --evaluate
```

### 3. Paper hybrid run (~35 minutes)

Reproduces the paper’s main method (Section 4.2):

```powershell
python main.py run hybrid --preset paper --evaluate
```

Or use the helper script:

```powershell
.\scripts\run_paper_hybrid.ps1
```

### 4. Optional follow-up analyses

Run **after** a successful hybrid or compare run:

```powershell
python main.py shap --run-id latest
python main.py smote --run-id latest
```

---

## CLI reference

```
python main.py [--preset paper|quick] [--run-id ID] <command> [options]
```

### Commands

| Command | Description | Paper reference |
|---------|-------------|-----------------|
| `preprocess` | Load, clean, split, and normalise data | Section 3.1 |
| `run hybrid` | GWO→WOA→AOA optimisation + final model | Sections 3.2.4, 4.2 |
| `run baseline` | NO-CNN baseline (fixed hyperparameters) | Table 7 |
| `run standalone <algo>` | One standalone optimiser (`gwo`, `woa`, `aoa`, `rime`) | Section 4.2 |
| `run compare` | All Table 7 models, run sequentially | Section 4.2 |
| `evaluate` | Test-set metrics and figures for a saved run | Table 7, Figures 8–10 |
| `shap` | SHAP explainability on hybrid model | Section 4.4 |
| `smote` | SMOTE augmentation study | Section 4.5 |

### `run` modes

```powershell
python main.py run hybrid   [--evaluate] [--force-preprocess]
python main.py run baseline
python main.py run standalone --algorithm gwo
python main.py run compare  [--evaluate]
```

### Flags

| Flag | Description |
|------|-------------|
| `--preset paper` | Full paper settings (default) |
| `--preset quick` | Small populations for testing |
| `--run-id ID` | Custom run identifier (default: UTC timestamp) |
| `--evaluate` | Run test-set evaluation after training |
| `--force-preprocess` | Rebuild processed data even if it exists |

### Examples

```powershell
# Single standalone algorithm (30 iterations, ~26–31 min)
python main.py run standalone --algorithm gwo --preset paper --evaluate

# Full Table 7 comparison (~2.5 hours, memory-safe)
python main.py run compare --preset paper --evaluate

# Re-evaluate a previous run without retraining
python main.py evaluate --run-id latest

# Re-evaluate a specific run
python main.py evaluate --run-id 20250629_143022
```

---

## Configuration

All settings live in `config.yaml`. Two presets are provided:

### `paper` preset (default)

Matches Section 4.2 experimental setup:

| Setting | Value |
|---------|-------|
| Population size | 20 |
| Hybrid iterations per stage | 10 (30 total) |
| Standalone iterations | 30 |
| Final model training attempts | 3 |
| Standalone training attempts | 1 |
| Early stopping patience | 5 |

### `quick` preset

For development and CI-style smoke tests:

| Setting | Value |
|---------|-------|
| Population size | 3 |
| Hybrid iterations per stage | 2 |
| Standalone iterations | 2 |
| Training attempts | 1 |

### Other settings

```yaml
shap:
  n_background: 50    # SHAP background samples
  n_explain: 100      # test samples to explain

smote:
  n_attempts: 3       # training retries per dataset config

logging:
  level: INFO
  max_bytes: 10485760 # 10 MB rotating log files
```

Edit `config.yaml` to tune these without changing code.

---

## Pipeline stages explained

### Preprocessing (`preprocess`)

1. Load CSV and verify categorical encodings.
2. Handle missing/invalid values.
3. Stratified split: 70% / 10% / 20%.
4. Z-score scaling (train statistics only).
5. Reshape to `(N, 11, 1, 1)` for Conv2D.
6. One-hot encode labels.
7. Save arrays to `data/processed/`.

### Training (`run`)

Each training command:

1. Initialises a **run directory** under `outputs/runs/<run_id>/`.
2. Starts **structured logging** to console and `outputs/logs/<run_id>.log`.
3. Runs the requested optimisation (if applicable).
4. Trains the final CNN with the best hyperparameters.
5. **Saves** `model.keras`, `hyperparameters.json`, and `convergence.json`.
6. **Releases** the model from memory before the next stage.
7. Updates `manifest.json` with stage timings and status.

### Evaluation (`evaluate`)

1. Loads saved models from the run directory.
2. Computes accuracy, F1, sensitivity, precision, NPV, MCC, Kappa (Section 4.1).
3. Writes figures to `outputs/figures/` and CSV to `outputs/results/`.

### SHAP (`shap`)

GradientExplainer analysis on the hybrid model. Run separately because it is memory-intensive and not part of the paper’s optimisation timing.

### SMOTE (`smote`)

Trains the hybrid model on three dataset configurations (original, balanced, double-balanced) and reports Table 10 metrics. Also run separately.

---

## Outputs and artifacts

After a run you will find:

```
outputs/
├── logs/
│   └── 20250629_143022.log          # Full timestamped log
├── runs/
│   ├── latest_run.txt               # Points to most recent run
│   └── 20250629_143022/
│       ├── manifest.json            # Stage status, durations, errors
│       ├── NO-CNN/
│       │   ├── model.keras
│       │   └── hyperparameters.json
│       ├── GWO-WOA-AOA-CNN/
│       │   ├── model.keras
│       │   ├── hyperparameters.json
│       │   └── convergence.json
│       └── ...
├── figures/
│   ├── confusion_matrices.png       # Figure 9
│   ├── roc_curves.png               # Figure 10
│   ├── convergence_curves.png       # Figure 8
│   ├── performance_comparison.png
│   ├── shap_importance.png          # Figure 14 (after shap command)
│   └── dataset_comparison.png       # Figure 16 (after smote command)
└── results/
    ├── metrics_comparison.csv       # Table 7
    ├── hyperparameters.csv          # Table 6
    └── metrics_<run_id>.json
```

### `manifest.json` example

```json
{
  "run_id": "20250629_143022",
  "preset": "paper",
  "mode": "hybrid",
  "status": "completed",
  "stages": {
    "GWO-WOA-AOA-CNN": {
      "status": "completed",
      "duration_sec": 2134.5
    },
    "evaluate": {
      "status": "completed",
      "duration_sec": 12.3
    }
  }
}
```

If a stage fails, `status` becomes `"failed"` and the error is recorded — check the log file for the full traceback.

---

## Expected runtimes

On hardware similar to the paper (Intel i7-6500U, 16 GB RAM, CPU):

| Command | Approximate time |
|---------|------------------|
| `preprocess` | < 1 min |
| `run hybrid --preset quick` | ~5 min |
| `run hybrid --preset paper` | ~35 min |
| `run standalone --algorithm gwo --preset paper` | ~27 min |
| `run compare --preset paper` | ~2.5 h |
| `shap --run-id latest` | 30 min – 2 h |
| `smote --run-id latest` | ~45 min – 1.5 h |

These align with the paper’s reported optimisation times (Section 4.2): RIME 30.8 min, AOA 25.4 min, GWO 26.9 min, WOA 28.3 min, hybrid 35.6 min.

---

## Project structure

```
final-year-project/
├── main.py                          # CLI entry point
├── config.yaml                      # Central configuration
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                         # Original CSV (not generated)
│   └── processed/                   # Generated .npy arrays (gitignored)
│
├── models/
│   └── scaler.pkl                   # Fitted StandardScaler (gitignored)
│
├── outputs/                         # Logs, runs, figures, results (gitignored)
│
├── docs/
│   ├── Runtime_Analysis_Report.docx # Why old single-script runs were slow
│   └── figures/
│
├── scripts/
│   ├── run_paper_hybrid.ps1         # One-click paper hybrid run
│   └── generate_runtime_analysis_doc.py
│
└── src/
    ├── config.py
    ├── logging_config.py
    ├── memory_utils.py
    ├── artifacts.py
    ├── pipeline.py
    ├── preprocess.py
    ├── cnn_model.py
    ├── fitness.py
    ├── hybrid_optimizer.py
    ├── standalone_optimizers.py
    ├── gwo.py, woa.py, aoa.py, rime.py
    ├── evaluate.py
    ├── shap_analysis.py
    └── smote_analysis.py
```

---

## Reproducing paper results

Recommended order for a full thesis reproduction:

| Step | Command | Produces |
|------|---------|----------|
| 1 | `python main.py preprocess` | Processed data |
| 2 | `python main.py run compare --preset paper --evaluate` | Table 7, Figures 8–10 |
| 3 | `python main.py shap --run-id latest` | Figures 14–15 |
| 4 | `python main.py smote --run-id latest` | Table 10, Figure 16 |

If time is limited, **Step 2 can be replaced** with:

```powershell
python main.py run hybrid --preset paper --evaluate
```

That reproduces the paper’s main hybrid result (~96% accuracy target) in ~35 minutes.

### Hyperparameter search space (Table 5)

| Hyperparameter | Search range |
|----------------|--------------|
| Conv2D filters | [8,16,32,64], [16,32,64,128], [32,64,128,256], [64,128,256,512] |
| Kernel size | 3, 5, 7, 9, 11 |
| Pooling size | 2, 3, 4, 5, 6 |
| FC neurons | 16, 32, 64, 128, 256 |
| Dropout | 0.1 – 0.5 |
| Learning rate | 0.0001 – 0.01 |
| Batch size | 8, 16, 32, 64, 100, 128 |
| Optimizer | adam, sgd, rmsprop |
| Max epochs | 10 – 50 |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`

Activate the virtual environment and install dependencies:

```powershell
.\venv311\Scripts\Activate.ps1
pip install -r requirements.txt --default-timeout=1000
```

### `No matching distribution found for tensorflow==2.15.0`

Your venv is likely **not** Python 3.11. Recreate it:

```powershell
py -3.11 -m venv venv311 --clear
.\venv311\Scripts\Activate.ps1
pip install -r requirements.txt --default-timeout=1000
```

### Process terminates after several hours

The old workflow ran all optimisers, SHAP, and SMOTE in one script (~1,500+ CNN trainings). Use **staged commands** instead:

```powershell
python main.py run hybrid --preset paper --evaluate   # first
python main.py shap --run-id latest                   # then separately
```

See `docs/Runtime_Analysis_Report.docx` for details.

### Out of memory during SHAP

Reduce samples in `config.yaml`:

```yaml
shap:
  n_background: 20
  n_explain: 50
```

### Pip download timeout

```powershell
pip install -r requirements.txt --default-timeout=1000
```

### Checking run progress

Tail the log file:

```powershell
Get-Content outputs\logs\<run_id>.log -Wait -Tail 30
```

Or inspect `outputs/runs/<run_id>/manifest.json` for completed stages.

---

## License

This project is provided for academic purposes as part of a final-year research project.
