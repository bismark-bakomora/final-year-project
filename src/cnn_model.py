import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, GlobalAveragePooling2D, Dropout,
    Dense, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ─────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACE
# Matches Table 5 in the paper exactly
# ─────────────────────────────────────────

# Filter configurations for 4 Conv2D layers
FILTER_OPTIONS = [
    [8,  16,  32,  64],   # index 0
    [16, 32,  64,  128],  # index 1
    [32, 64,  128, 256],  # index 2
    [64, 128, 256, 512],  # index 3
]

KERNEL_OPTIONS   = [3, 5, 7, 9, 11]        # index 0-4
POOLING_OPTIONS  = [2, 3, 4, 5, 6]         # index 0-4
NEURON_OPTIONS   = [16, 32, 64, 128, 256]  # index 0-4
BATCH_OPTIONS    = [8, 16, 32, 64, 100, 128] # index 0-5
OPTIMIZER_OPTIONS = ['adam', 'sgd', 'rmsprop'] # index 0-2

# Continuous ranges
DROPOUT_MIN  = 0.1;  DROPOUT_MAX  = 0.5
LR_MIN       = 0.0001; LR_MAX     = 0.01
EPOCH_MIN    = 10;   EPOCH_MAX    = 100

# Hyperparameter vector indices
# x = [filters_idx, kernel_idx, pooling_idx, neurons_idx,
#       dropout_rate, learning_rate, batch_idx,
#       optimizer_idx, max_epoch]
HP_FILTERS_IDX   = 0
HP_KERNEL_IDX    = 1
HP_POOLING_IDX   = 2
HP_NEURONS_IDX   = 3
HP_DROPOUT       = 4
HP_LR            = 5
HP_BATCH_IDX     = 6
HP_OPTIMIZER_IDX = 7
HP_EPOCH         = 8

# Search bounds for optimizers (continuous representation)
LOWER_BOUNDS = [0, 0, 0, 0, DROPOUT_MIN, LR_MIN, 0, 0, EPOCH_MIN]
UPPER_BOUNDS = [
    len(FILTER_OPTIONS) - 1,
    len(KERNEL_OPTIONS) - 1,
    len(POOLING_OPTIONS) - 1,
    len(NEURON_OPTIONS) - 1,
    DROPOUT_MAX,
    LR_MAX,
    len(BATCH_OPTIONS) - 1,
    len(OPTIMIZER_OPTIONS) - 1,
    EPOCH_MAX
]


# ─────────────────────────────────────────
# DECODE HYPERPARAMETER VECTOR
# Converts continuous optimizer values to
# actual hyperparameter values
# ─────────────────────────────────────────
def decode_hyperparameters(x):
    """
    Decode a continuous hyperparameter vector x
    into actual usable values.

    Paper Table 5 hyperparameters:
    x[0] filters index  → one of 4 filter configs
    x[1] kernel index   → one of [3,5,7,9,11]
    x[2] pooling index  → one of [2,3,4,5,6]
    x[3] neurons index  → one of [16,32,64,128,256]
    x[4] dropout rate   → continuous [0.1, 0.5]
    x[5] learning rate  → continuous [0.0001, 0.01]
    x[6] batch index    → one of [8,16,32,64,100,128]
    x[7] optimizer index→ one of ['adam','sgd','rmsprop']
    x[8] max epoch      → continuous [10, 100]
    """
    filters_idx   = int(np.clip(round(x[HP_FILTERS_IDX]),
                                0, len(FILTER_OPTIONS)-1))
    kernel_idx    = int(np.clip(round(x[HP_KERNEL_IDX]),
                                0, len(KERNEL_OPTIONS)-1))
    pooling_idx   = int(np.clip(round(x[HP_POOLING_IDX]),
                                0, len(POOLING_OPTIONS)-1))
    neurons_idx   = int(np.clip(round(x[HP_NEURONS_IDX]),
                                0, len(NEURON_OPTIONS)-1))
    dropout_rate  = float(np.clip(x[HP_DROPOUT],
                                  DROPOUT_MIN, DROPOUT_MAX))
    learning_rate = float(np.clip(x[HP_LR],
                                  LR_MIN, LR_MAX))
    batch_idx     = int(np.clip(round(x[HP_BATCH_IDX]),
                                0, len(BATCH_OPTIONS)-1))
    optimizer_idx = int(np.clip(round(x[HP_OPTIMIZER_IDX]),
                                0, len(OPTIMIZER_OPTIONS)-1))
    max_epoch     = int(np.clip(round(x[HP_EPOCH]),
                                EPOCH_MIN, EPOCH_MAX))

    return {
        'filters':      FILTER_OPTIONS[filters_idx],
        'kernel_size':  KERNEL_OPTIONS[kernel_idx],
        'pooling_size': POOLING_OPTIONS[pooling_idx],
        'neurons':      NEURON_OPTIONS[neurons_idx],
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size':   BATCH_OPTIONS[batch_idx],
        'optimizer':    OPTIMIZER_OPTIONS[optimizer_idx],
        'max_epoch':    max_epoch,
    }


# ─────────────────────────────────────────
# BUILD DYNAMIC CNN MODEL
# Implements Figure 7 from the paper
# ─────────────────────────────────────────
def build_cnn(hyperparams):
    """
    Dynamically build 2D-CNN architecture as per
    Figure 7 and Section 3.3.1 of the paper.

    Architecture:
    Input (11,1,1)
    → 4x [Conv2D → BatchNorm → ReLU → MaxPooling2D]
    → GlobalAveragePooling2D
    → Dense (FC layer) → ReLU → BatchNorm
    → Dropout
    → Dense (2, softmax)  ← binary classification
    """
    filters      = hyperparams['filters']       # list of 4
    kernel_size  = hyperparams['kernel_size']   # single int
    pooling_size = hyperparams['pooling_size']  # single int
    neurons      = hyperparams['neurons']       # FC neurons
    dropout_rate = hyperparams['dropout_rate']
    learning_rate = hyperparams['learning_rate']
    optimizer_name = hyperparams['optimizer']

    # Input shape: (11 features, 1, 1)
    inputs = Input(shape=(11, 1, 1), name='input_layer')
    x = inputs

    # ── 4 Conv2D + BatchNorm + ReLU + MaxPooling blocks ──
    for i, num_filters in enumerate(filters):

        # Kernel size must not exceed spatial dimension
        # Input height shrinks after each pooling
        # Use padding='same' to handle small spatial dims
        x = Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, 1),
            padding='same',
            name=f'conv2d_{i+1}'
        )(x)
        x = BatchNormalization(name=f'batchnorm_{i+1}')(x)
        x = Activation('relu', name=f'relu_{i+1}')(x)
        x = MaxPooling2D(
            pool_size=(pooling_size, 1),
            padding='same',
            name=f'maxpool_{i+1}'
        )(x)

    # ── Global Average Pooling ──
    # Converts feature maps to single vector
    # Reduces parameters significantly
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # ── First Fully Connected Layer ──
    x = Dense(neurons, name='fc_layer')(x)
    x = Activation('relu', name='fc_relu')(x)
    x = BatchNormalization(name='fc_batchnorm')(x)

    # ── Dropout Layer ──
    x = Dropout(dropout_rate, name='dropout')(x)

    # ── Output Layer ──
    # 2 neurons for binary classification (no disease / disease)
    outputs = Dense(2, activation='softmax',
                    name='output_layer')(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name='GWO_WOA_AOA_CNN')

    # ── Compile ──
    # Select optimizer as per paper Section 3.3.1
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,       # paper default
            beta_2=0.999,     # paper default
            epsilon=1e-7      # paper default
        )
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate
        )
    else:  # rmsprop
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    # Categorical crossentropy as per paper Section 3.3.1
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ─────────────────────────────────────────
# TRAIN CNN MODEL
# ─────────────────────────────────────────
def train_cnn(model, X_train, y_train, X_val, y_val,
              batch_size, max_epoch,
              save_path='models/best_model.weights.h5'):
    """
    Train the CNN model with early stopping.
    Paper Section 3.3.1:
    'Early stopping terminates training if there is
    no improvement in validation loss over 5 epochs.'
    """
    os.makedirs('models', exist_ok=True)

    callbacks = [
        # Early stopping — paper specifies patience=5
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        # Removed ModelCheckpoint — causes format conflicts
        # Best weights restored via EarlyStopping instead
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epoch,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    return history


# ─────────────────────────────────────────
# GET VALIDATION ACCURACY
# Used by fitness function
# ─────────────────────────────────────────
def get_val_accuracy(model, X_val, y_val):
    """Evaluate model on validation set."""
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return accuracy


# ─────────────────────────────────────────
# QUICK TEST — verify model builds correctly
# ─────────────────────────────────────────
def test_cnn_build():
    """
    Test CNN builds with the paper's optimal
    hyperparameters from Table 6.
    """
    print("Testing CNN build with paper's optimal hyperparameters...")
    print("(Table 6: GWO-WOA-AOA column)")
    print("-" * 50)

    # Paper's reported optimal hyperparameters (Table 6)
    test_hp = {
        'filters':       [32, 64, 128, 256],
        'kernel_size':   11,
        'pooling_size':  3,
        'neurons':       128,
        'dropout_rate':  0.313,
        'learning_rate': 0.00015,
        'batch_size':    128,
        'optimizer':     'adam',
        'max_epoch':     36,
    }

    model = build_cnn(test_hp)
    model.summary()

    # Test with dummy data matching our shapes
    dummy_X = np.random.randn(10, 11, 1, 1).astype(np.float32)
    dummy_y = tf.keras.utils.to_categorical(
        np.random.randint(0, 2, 10), num_classes=2
    )

    pred = model.predict(dummy_X, verbose=0)
    print(f"\nInput shape:  {dummy_X.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Output sum (should be ~1 per row): "
          f"{pred.sum(axis=1).round(4)}")
    print("\nCNN build test PASSED")

    return model


if __name__ == "__main__":
    test_cnn_build()