# Heart Disease Prediction using Hybrid Optimizer (GWO-WOA-AOA) and CNN

A final year project implementing a hybrid meta-heuristic optimization approach combining Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), and Artificial Orca Algorithm (AOA) to optimize CNN hyperparameters for heart disease prediction.

## 📋 Project Overview

This project uses a sequential three-stage hybrid optimization framework to automatically tune CNN hyperparameters for binary classification of heart disease. The hybrid optimizer is compared against a baseline NO-CNN model to demonstrate the effectiveness of the optimization approach.

**Key Features:**
- **Hybrid Optimization**: Sequential GWO → WOA → AOA pipeline for comprehensive hyperparameter search
- **CNN Model**: Convolutional Neural Network with customizable filters, kernels, pooling, and dense layers
- **Comprehensive Evaluation**: Metrics, visualizations, confusion matrices, and ROC curves
- **Explainability**: SHAP integration for model interpretability

## 📁 Project Structure

```
heart_disease_prediction/
├── main.py                    # Entry point for the pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
│
├── data/
│   ├── raw/
│   │   └── heart_statlog_cleveland_hungary_final.csv
│   └── processed/
│       ├── X_train.npy, X_val.npy, X_test.npy
│       ├── y_train.npy, y_val.npy, y_test.npy
│       └── y_*_raw.npy (original labels)
│
├── models/
│   └── best_model.h5          # Trained CNN model weights
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Data preprocessing and train/val/test split
│   ├── cnn_model.py           # CNN architecture and hyperparameter space
│   ├── hybrid_optimizer.py    # Main hybrid GWO-WOA-AOA optimizer
│   ├── gwo.py                 # Grey Wolf Optimizer implementation
│   ├── woa.py                 # Whale Optimization Algorithm implementation
│   ├── aoa.py                 # Artificial Orca Algorithm implementation
│   ├── fitness.py             # Fitness function for optimization
│   └── evaluate.py            # Model evaluation and visualization
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_baseline.ipynb      # Baseline model experiments
│   └── 03_results_analysis.ipynb  # Results visualization and analysis
│
├── outputs/
│   ├── figures/
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── convergence_curves.png
│   │   └── performance_comparison.png
│   └── results/
│       ├── metrics_comparison.csv
│       └── hyperparameters.csv
│
└── tests/
    ├── test_preprocess.py
    ├── test_cnn_model.py
    └── test_optimizers.py
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip package manager
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd heart_disease_prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv311
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv311\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .\venv311\Scripts\activate.bat
     ```
   - **Linux/macOS**:
     ```bash
     source venv311/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

**Heart Disease Dataset** (Cleveland-Hungary combined)
- **Source**: Statlog (Heart) Database
- **Samples**: ~1000 records
- **Features**: 13 clinical and demographic attributes
- **Target**: Binary classification (0: no disease, 1: disease present)
- **Splits**: 70% training, 15% validation, 15% testing

### Preprocessing Steps:
1. Handle missing values
2. Feature scaling (StandardScaler)
3. SMOTE for class imbalance handling
4. Train/validation/test split

## 🔧 Running the Project

### Quick Test (Small Population)

Run the main pipeline with reduced population for quick testing:

```bash
python main.py
```

This will:
1. Preprocess the raw data
2. Run hybrid optimization with small population (population_size=3, iterations=2 for each algorithm)
3. Train the final model
4. Train a NO-CNN baseline
5. Evaluate both models
6. Generate visualizations and metrics

**Output**: Check `outputs/` folder for results:
- `outputs/figures/`: Visualization plots
- `outputs/results/`: CSV metrics and hyperparameters

### Full Optimization (Production Settings)

Edit `main.py` to use production parameters:
```python
hybrid = HybridOptimizer(
    population_size=20,      # full: 20
    gwo_iterations=10,       # full: 10
    woa_iterations=10,       # full: 10
    aoa_iterations=10,       # full: 10
    lower_bounds=LOWER_BOUNDS,
    upper_bounds=UPPER_BOUNDS
)
```

## 📈 Hyperparameter Search Space

The hybrid optimizer tunes these CNN hyperparameters (9-dimensional search space):

| Parameter | Options | Type |
|-----------|---------|------|
| Filter Config | [0-3] | Integer |
| Kernel Size | [3,5,7,9,11] | Integer |
| Pooling Size | [2,3,4,5,6] | Integer |
| Dense Neurons | [16,32,64,128,256] | Integer |
| Dropout Rate | [0.1 - 0.5] | Continuous |
| Learning Rate | [0.0001 - 0.01] | Continuous |
| Batch Size | [8,16,32,64,100,128] | Integer |
| Optimizer | ['adam', 'sgd', 'rmsprop'] | Categorical |
| Max Epochs | [10 - 100] | Integer |

## 🧬 Hybrid Optimization Algorithm

**Three-Stage Sequential Pipeline:**

1. **Stage 1 - GWO (Grey Wolf Optimizer)**
   - Global exploration phase
   - Identifies promising regions in search space
   - Default: 10 iterations

2. **Stage 2 - WOA (Whale Optimization Algorithm)**
   - Local exploitation phase
   - Refines solutions from GWO
   - Default: 10 iterations

3. **Stage 3 - AOA (Artificial Orca Algorithm)**
   - Fine-tuning phase
   - Precision enhancement
   - Default: 10 iterations

Each sub-algorithm receives the best solution from the previous stage as initial population.

## 📊 Model Evaluation

**Metrics Computed:**
- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)

**Visualizations Generated:**
- Confusion Matrices (Hybrid vs Baseline)
- ROC Curves
- Convergence Curves (per algorithm stage)
- Performance Comparison Bar Charts

## 🧪 Running Tests

Run the test suite:

```bash
pytest tests/
```

Individual test files:
- `tests/test_preprocess.py` - Data preprocessing tests
- `tests/test_cnn_model.py` - CNN model architecture tests
- `tests/test_optimizers.py` - Optimizer logic tests

## 📓 Jupyter Notebooks

Explore the analysis through notebooks:

1. **01_eda.ipynb** - Exploratory Data Analysis
   - Dataset statistics
   - Feature distributions
   - Class imbalance analysis
   - Correlation analysis

2. **02_baseline.ipynb** - Baseline Model Experiments
   - NO-CNN baseline training
   - Initial model comparison

3. **03_results_analysis.ipynb** - Results Visualization
   - Detailed metric analysis
   - Convergence behavior
   - Hyperparameter insights

## 📦 Dependencies

Key libraries used:

```
tensorflow==2.13.0      # Deep learning framework
scikit-learn>=1.3       # Machine learning utilities
numpy>=1.23,<2.0        # Numerical computing
pandas>=1.5             # Data manipulation
matplotlib>=3.6         # Plotting
seaborn>=0.12           # Statistical visualization
shap>=0.41              # Model explainability
imbalanced-learn>=0.10  # SMOTE for class imbalance
joblib>=1.2             # Parallel computing
scipy>=1.9              # Scientific computing
```

See [requirements.txt](requirements.txt) for the complete list.

## 💡 Key Results

The hybrid optimizer significantly improves upon the NO-CNN baseline:

- **Hybrid Model**: ~94-96% accuracy
- **Baseline Model**: ~85-88% accuracy
- **Convergence**: Demonstrated improvement across GWO → WOA → AOA stages

Results are saved in:
- `outputs/results/metrics_comparison.csv`
- `outputs/results/hyperparameters.csv`

## 🔍 Project Highlights

- **Meta-heuristic Optimization**: Combines three nature-inspired algorithms
- **Automated Hyperparameter Tuning**: Eliminates manual grid/random search
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Reproducibility**: Fixed random seeds and saved models
- **Interpretability**: SHAP for feature importance analysis

## 📝 Notes

- The quick test uses small population/iteration counts for fast execution (~5-10 min)
- Full optimization with production parameters may take 2-4 hours
- Ensure GPU support for TensorFlow to speed up training
- Results vary slightly between runs due to stochastic optimization algorithms

## 👨‍💻 Author

Final Year Project - Heart Disease Prediction System

## 📄 License

This project is provided as-is for academic purposes.

## 🤝 Contributing

For improvements or bug fixes:
1. Create a feature branch
2. Make your changes
3. Ensure tests pass
4. Submit a pull request

## 📧 Support

For questions or issues, please create an issue in the repository.
