import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


# CONSTANTS
FEATURE_COLS = [
    'age', 'sex', 'chest pain type', 'resting bp s',
    'cholesterol', 'fasting blood sugar', 'resting ecg',
    'max heart rate', 'exercise angina', 'oldpeak', 'ST slope'
]
TARGET_COL = 'target'

EXPECTED_RANGES = {
    'age':                  (28, 77),
    'sex':                  (0, 1),
    'chest pain type':      (1, 4),
    'resting bp s':         (80, 200),
    'cholesterol':          (0, 603),
    'fasting blood sugar':  (0, 1),
    'resting ecg':          (0, 2),
    'max heart rate':       (60, 202),
    'exercise angina':      (0, 1),
    'oldpeak':              (0.0, 6.2),
    'ST slope':             (1, 3),
}


# STEP 1.1 — LOAD DATA
def load_data(filepath):
    """Load and display basic info about the dataset."""
    df = pd.read_csv(filepath)
    
    print("=" * 50)
    print("DATASET AUDIT")
    print("=" * 50)
    print(f"Shape:        {df.shape}")
    print(f"Columns:      {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")
    print(f"\nTarget %:\n{df[TARGET_COL].value_counts(normalize=True).round(4)}")
    
    return df



# STEP 1.3 — ENCODE CATEGORICAL VARIABLES

def encode_categoricals(df):
    """
    Verify categorical variables are numerically encoded
    as per paper Section 3.1.2.
    
    Paper specifies:
    - sex:                0=Female, 1=Male
    - chest pain type:    1=Typical angina, 4=Asymptomatic
    - resting ecg:        0=Normal, 1=ST-T abnormality, 2=LVH
    - fasting blood sugar:0=<=120 mg/dl, 1=>120 mg/dl
    - exercise angina:    0=No, 1=Yes
    - ST slope:           1=Rising, 2=Flat, 3=Descending
    """
    
    cat_cols = {
        'sex':                  [0, 1],
        'chest pain type':      [1, 2, 3, 4],
        'resting ecg':          [0, 1, 2],
        'fasting blood sugar':  [0, 1],
        'exercise angina':      [0, 1],
        'ST slope':             [1, 2, 3],
    }
    
    print("\nCategorical encoding verification (Paper Section 3.1.2):")
    print("-" * 55)
    
    all_ok = True
    for col, expected_values in cat_cols.items():
        actual_values = sorted(df[col].unique().tolist())
        dtype = df[col].dtype
        
        # Check if numeric
        is_numeric = str(dtype) in ['int64', 'float64', 'int32']
        
        # Check if values match expected
        values_ok = all(v in expected_values for v in actual_values)
        
        status = "OK" if (is_numeric and values_ok) else "WARNING"
        if status == "WARNING":
            all_ok = False
            
        print(f"  {col:25s}: dtype={str(dtype):8s} "
              f"values={actual_values}  {status}")
        
        # Only encode if not already numeric
        if not is_numeric:
            if col == 'sex':
                df[col] = df[col].map({'F': 0, 'M': 1})
            elif col == 'exercise angina':
                df[col] = df[col].map({'N': 0, 'Y': 1})
            elif col == 'ST slope':
                df[col] = df[col].map({'Up': 1, 'Flat': 2, 'Down': 3})
            elif col == 'resting ecg':
                df[col] = df[col].map({'Normal': 0, 'ST': 1, 'LVH': 2})
            elif col == 'chest pain type':
                df[col] = df[col].map({'TA': 1, 'ATA': 2, 'NAP': 3, 'ASY': 4})
            print(f"    → Encoded {col} to numeric")
    
    if all_ok:
        print("\n  All categorical variables correctly encoded.")
    else:
        print("\n  Some variables were re-encoded — check warnings above.")
    
    return df



# STEP 1.4 — HANDLE MISSING VALUES

def handle_missing(df):
    """
    Check and handle missing values and invalid entries
    as per paper Section 3.1.2.
    
    Keeps all 1190 rows by using median imputation
    instead of dropping invalid rows.
    """
    
    # Check standard missing values
    missing = df[FEATURE_COLS + [TARGET_COL]].isnull().sum()
    if missing.sum() == 0:
        print("\nNo NaN missing values detected.")
    else:
        print(f"\nMissing values found:\n{missing[missing > 0]}")
        df = df.dropna(subset=[TARGET_COL])
        for col in FEATURE_COLS:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Filled {col} with median: {median_val}")

    print("\nHandling invalid values (Paper Section 3.1.2):")
    print("-" * 55)

    # Fix 1 — ST slope: replace 0 with median
    # Paper Table 1 specifies valid range is 1-3 only
    invalid_st = df[df['ST slope'] == 0].shape[0]
    if invalid_st > 0:
        median_st = df[df['ST slope'] != 0]['ST slope'].median()
        df.loc[df['ST slope'] == 0, 'ST slope'] = median_st
        print(f"  ST slope = 0:      replaced {invalid_st} row(s) "
              f"with median ({median_st})")
    else:
        print(f"  ST slope:          no invalid values")

    # Fix 2 — Resting BP: replace 0 with median
    # Blood pressure of 0 is physiologically impossible
    invalid_bp = df[df['resting bp s'] == 0].shape[0]
    if invalid_bp > 0:
        median_bp = df[df['resting bp s'] != 0]['resting bp s'].median()
        df.loc[df['resting bp s'] == 0, 'resting bp s'] = median_bp
        print(f"  Resting BP = 0:    replaced {invalid_bp} row(s) "
              f"with median ({median_bp})")
    else:
        print(f"  Resting BP:        no invalid values")

    # Fix 3 — Oldpeak: clip negative values to 0
    # Paper specifies range 0.0-6.2
    # Negative values exist in source datasets as measurement artifacts
    invalid_op = df[df['oldpeak'] < 0].shape[0]
    if invalid_op > 0:
        df['oldpeak'] = df['oldpeak'].clip(lower=0.0)
        print(f"  Oldpeak < 0:       clipped {invalid_op} value(s) to 0")
    else:
        print(f"  Oldpeak:           no invalid values")

    print(f"\nShape after cleaning: {df.shape}")
    return df



# STEP 1.5 — VALIDATE RANGES

def validate_ranges(df):
    """Cross-check feature ranges against Table 1 in the paper."""
    
    print("\nRange validation (Table 1):")
    print("-" * 55)
    all_ok = True
    
    for col, (lo, hi) in EXPECTED_RANGES.items():
        actual_min = df[col].min()
        actual_max = df[col].max()
        ok = actual_min >= lo and actual_max <= hi
        status = "OK" if ok else "WARNING"
        if not ok:
            all_ok = False
        print(f"  {col:25s}: [{actual_min:.2f}, {actual_max:.2f}]  {status}")
    
    if all_ok:
        print("\nAll ranges match Table 1.")
    else:
        print("\nSome ranges outside expected — review above warnings.")
    
    return df



# STEP 1.6 — TRAIN / VAL / TEST SPLIT

def split_data(df):
    """
    Split into 70% train / 10% val / 20% test
    as specified in the paper (Section 3).
    """
    X = df[FEATURE_COLS].values   # (1190, 11)
    y = df[TARGET_COL].values     # (1190,)
    
    # First: carve out 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    # Then: split remainder into 70% train / 10% val
    # 10% of total = 12.5% of the remaining 80%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.125,
        random_state=42,
        stratify=y_temp
    )
    
    print("\nSplit sizes:")
    print(f"  Train:      {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")
    
    # Verify class balance preserved
    print("\nClass balance per split:")
    for name, y_s in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        pos = y_s.sum() / len(y_s)
        print(f"  {name}: {pos:.3f} positive rate")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# STEP 1.7 — Z-SCORE NORMALIZATION

def normalize_data(X_train, X_val, X_test):
    """
    Apply z-score normalization as per paper Section 3.1.2.
    Fit ONLY on training data to prevent data leakage.
    xnorm = (x - mean) / std
    """
    scaler = StandardScaler()
    
    # Fit on train only
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform val and test using train statistics
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    print("\nNormalization check (train set):")
    print(f"  Mean (should be ~0): {X_train_scaled.mean(axis=0).round(3)}")
    print(f"  Std  (should be ~1): {X_train_scaled.std(axis=0).round(3)}")
    
    # Save scaler for clinical interface later
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nScaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler



# STEP 1.8 — RESHAPE FOR 2D CNN

def reshape_for_cnn(X_train, X_val, X_test):
    """
    Reshape from (N, 11) to (N, 11, 1, 1) for Conv2D input
    as specified in paper Section 3.
    """
    X_train_cnn = X_train.reshape(-1, 11, 1, 1)
    X_val_cnn   = X_val.reshape(-1, 11, 1, 1)
    X_test_cnn  = X_test.reshape(-1, 11, 1, 1)
    
    print("\nCNN input shapes:")
    print(f"  Train: {X_train_cnn.shape}")
    print(f"  Val:   {X_val_cnn.shape}")
    print(f"  Test:  {X_test_cnn.shape}")
    
    return X_train_cnn, X_val_cnn, X_test_cnn



# STEP 1.9 — ONE-HOT ENCODE LABELS

def encode_labels(y_train, y_val, y_test):
    """Convert integer labels to one-hot for categorical crossentropy."""
    
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat   = to_categorical(y_val,   num_classes=2)
    y_test_cat  = to_categorical(y_test,  num_classes=2)
    
    print("\nLabel shapes:")
    print(f"  Train: {y_train_cat.shape}")
    print(f"  Val:   {y_val_cat.shape}")
    print(f"  Test:  {y_test_cat.shape}")
    
    return y_train_cat, y_val_cat, y_test_cat



# STEP 1.10 — SAVE PROCESSED DATA

def save_processed(X_train, X_val, X_test,
                   y_train, y_val, y_test,
                   y_train_raw, y_val_raw, y_test_raw):
    """Save all processed arrays to disk."""
    
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy',   X_val)
    np.save('data/processed/X_test.npy',  X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy',   y_val)
    np.save('data/processed/y_test.npy',  y_test)
    
    # Raw integer labels needed for metrics later
    np.save('data/processed/y_train_raw.npy', y_train_raw)
    np.save('data/processed/y_val_raw.npy',   y_val_raw)
    np.save('data/processed/y_test_raw.npy',  y_test_raw)
    
    print("\nAll arrays saved to data/processed/")



# MAIN PIPELINE

def run_preprocessing(filepath):
    """Run the full preprocessing pipeline."""
    
    print("\n")
    print("STARTING PREPROCESSING PIPELINE")
    print("\n")
    
    df = load_data(filepath)
    df = encode_categoricals(df)
    df = handle_missing(df)
    df = validate_ranges(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    X_train_s, X_val_s, X_test_s, _ = normalize_data(
        X_train, X_val, X_test
    )
    
    X_train_cnn, X_val_cnn, X_test_cnn = reshape_for_cnn(
        X_train_s, X_val_s, X_test_s
    )
    
    y_train_cat, y_val_cat, y_test_cat = encode_labels(
        y_train, y_val, y_test
    )
    
    save_processed(
        X_train_cnn, X_val_cnn, X_test_cnn,
        y_train_cat, y_val_cat, y_test_cat,
        y_train, y_val, y_test
    )
    
    print("\n")
    print("PREPROCESSING COMPLETE")
    print("\n")
    
    return (X_train_cnn, X_val_cnn, X_test_cnn,
            y_train_cat, y_val_cat, y_test_cat,
            y_train, y_val, y_test)