import json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, ReLU, Add, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import hamming
import tensorflow as tf

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Function to load prebuilt ground keys
def load_keys(file_path):
    """Load pre-built ground truth keys from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to load valid ECG segments
def load_valid_ecg_segments(directory, required_length=170):
    """Load ECG segmentation data with the required number of data points."""
    ecg_segments = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            segment_path = os.path.join(directory, filename)
            segment_data = pd.read_csv(segment_path, header=None, skiprows=1).values.flatten()
            if len(segment_data) == required_length:
                ecg_segments.append(segment_data)
    return ecg_segments

# Residual block with L2 regularization and optional dropout
def residual_block(x, filters, downsample=False, dropout_rate=0.3):
    shortcut = x
    if downsample:
        x = Conv1D(filters, 3, strides=2, padding='same', kernel_regularizer=l2(0.001))(x)
        shortcut = Conv1D(filters, 1, strides=2, padding='same', kernel_regularizer=l2(0.001))(shortcut)
    else:
        x = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

# ResNet encoder with reduced complexity
def resnet_encoder(input_dim):
    inputs = Input(shape=(input_dim, 1))
    x = Conv1D(64, 7, strides=2, padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 512, downsample=True)
    encoded = GlobalAveragePooling1D()(x)
    return inputs, encoded

# Function to build key prediction model
def build_key_prediction_model(input_dim, output_dim=256):
    inputs, encoded = resnet_encoder(input_dim)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(encoded)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(output_dim, activation='sigmoid', name='key_output')(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

# Train and test the model with pre-built keys using cross-validation
def train_and_test_prebuilt_keys(base_directory, keys_file, log_file_path="cross_validation_log.txt"):
    max_length = 170
    prebuilt_keys = load_keys(keys_file)
    all_persons_data = []
    all_person_ids = []

    # Load data and associate with pre-built ground truth keys
    for person_dir in sorted(os.listdir(base_directory)):
        if not person_dir.startswith("Person_"):
            continue

        person_id = int(person_dir.split('_')[-1])  # Extract person ID as integer
        if person_id == 74:  # Skip Person_74 (missing folder)
            continue

        # Adjust IDs greater than 73 to align with keys
        adjusted_person_id = person_id - 1 if person_id > 74 else person_id

        person_path = os.path.join(base_directory, person_dir, 'rec_2_filtered')
        if not os.path.isdir(person_path):
            print(f"Skipping non-existent or invalid folder: {person_dir}")
            continue

        key_name = f"Person_{adjusted_person_id:02d}"
        if key_name not in prebuilt_keys:
            print(f"Skipping folder without corresponding key: {person_dir}")
            continue

        segments = load_valid_ecg_segments(person_path, required_length=max_length)
        if segments:
            all_persons_data.append((np.array(segments).reshape(-1, max_length, 1), prebuilt_keys[key_name]))
            all_person_ids.extend([adjusted_person_id] * len(segments))
        else:
            print(f"No valid segments found in: {person_dir}")

    if not all_persons_data:
        print("No data loaded. Exiting.")
        return

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    with open(log_file_path, 'w') as log_file:
        log_file.write("Cross-Validation Log with Pre-Built Keys\n")
        log_file.write("========================================\n\n")

        for fold, (train_index, val_index) in enumerate(skf.split(all_person_ids, all_person_ids)):
            log_file.write(f"\n--- Fold {fold + 1} ---\n")

            X_train, Y_train = [], []
            X_val, Y_val = [], []

            for i in train_index:
                segments, ground_truth_key = all_persons_data[i]
                ground_truth_key_array = np.tile(ground_truth_key, (len(segments), 1))
                X_train.append(segments)
                Y_train.append(ground_truth_key_array)

            for i in val_index:
                segments, ground_truth_key = all_persons_data[i]
                ground_truth_key_array = np.tile(ground_truth_key, (len(segments), 1))
                X_val.append(segments)
                Y_val.append(ground_truth_key_array)

            X_train = np.concatenate(X_train, axis=0)
            Y_train = np.concatenate(Y_train, axis=0)
            X_val = np.concatenate(X_val, axis=0)
            Y_val = np.concatenate(Y_val, axis=0)

            model = build_key_prediction_model(input_dim=max_length)
            model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_val, Y_val))

            # Evaluate model
            val_predictions = model.predict(X_val)
            averaged_val_keys = np.mean(val_predictions, axis=0)
            representative_key = (averaged_val_keys > 0.5).astype(int)
            hamming_distance = hamming(representative_key, Y_val[0]) * len(representative_key)
            log_file.write(f"Validation Hamming Distance: {hamming_distance:.2f}\n")

    print(f"Cross-validation completed. Results saved to {log_file_path}")

if __name__ == "__main__":
    base_directory = '/path/to/segmented_ecg_data'
    keys_file = '/path/to/ground_keys.json'
    train_and_test_prebuilt_keys(base_directory, keys_file)
