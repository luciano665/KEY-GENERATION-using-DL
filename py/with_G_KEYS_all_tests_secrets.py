import json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from scipy.spatial.distance import hamming
import tensorflow as tf

# Set seeds for reproducibility if desired
tf.random.set_seed(42)
np.random.seed(42)

# Function to load prebuilt ground keys
def load_keys(file_path):
    """Load pre-built ground truth keys from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to load valid ECG segments
def load_valid_ecg_segments(directory, required_length=170):
    """Load the ECG segmentation data with the required # of data points on each CSV segment."""
    ecg_segments = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            segment_path = os.path.join(directory, filename)
            segment_data = pd.read_csv(segment_path, header=None, skiprows=1).values.flatten()
            if len(segment_data) == required_length:
                ecg_segments.append(segment_data)
    return ecg_segments

# ResNet-like architecture encoder: Building blocks
def conv_block(x, filters, kernel_size=3, strides=1, padding='same'):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Residual block
def residual_block(x, filters, downsample=False):
    shortcut = x
    if downsample:
        x = conv_block(x, filters, strides=2)
        shortcut = Conv1D(filters, 1, strides=2, padding='same')(shortcut)
    else:
        x = conv_block(x, filters)
    x = Conv1D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

# ResNet encoder
def resnet24_encoder(input_dim):
    """
    ResNet 24 layers to extract features from ECG signals.
    """
    inputs = Input(shape=(input_dim, 1))
    x = conv_block(inputs, filters=64, kernel_size=7, strides=2)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=128, downsample=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=256, downsample=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=512, downsample=True)
    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)
    encoded = GlobalAveragePooling1D()(x)
    return inputs, encoded

# Function to build key prediction model
def build_key_prediction_model(input_dim, output_dim=256):
    """
    Build key prediction using the encoder ResNet Model.
    """
    inputs, encoded = resnet24_encoder(input_dim)
    x = Dense(512, activation='relu')(encoded)
    x = Dense(256, activation='relu')(x)
    key_output = Dense(output_dim, activation='sigmoid', name='key_output')(x)

    model = Model(inputs, key_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

# Train and test with the pre-built keys using cross-validation
def train_and_test_prebuilt_keys(base_directory, keys_file, log_file_path="cross_validation_log_with_prebuilt_keys_and_all_tests_secrets.txt"):
    """Train and test the model with pre-built ground truth keys using cross-validation."""
    max_length = 170
    prebuilt_keys = load_keys(keys_file)
    all_persons_data = []

    # Load data and associate with pre-built ground truth keys
    # Each entry in all_persons_data: (segments_array, ground_truth_key)
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
        else:
            print(f"No valid segments found in: {person_dir}")

    if not all_persons_data:
        print("No data loaded. Exiting.")
        return

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Dictionary to store representative keys across folds for each person
    person_keys_across_folds = {i: [] for i in range(len(all_persons_data))}

    with open(log_file_path, 'w') as log_file:
        log_file.write("Cross-Validation Log with Pre-Built Keys\n")
        log_file.write("========================================\n\n")

        for fold, (train_index, val_index) in enumerate(kf.split(all_persons_data)):
            log_file.write(f"\n--- Fold {fold + 1} ---\n")

            # Combine all training data for this fold
            X_train = []
            Y_train = []
            for person_idx in train_index:
                person_segments, ground_truth_key = all_persons_data[person_idx]
                ground_truth_key_array = np.tile(ground_truth_key, (len(person_segments), 1))
                X_train.append(person_segments)
                Y_train.append(ground_truth_key_array)

            X_train = np.concatenate(X_train, axis=0)
            Y_train = np.concatenate(Y_train, axis=0)

            # Build and train a single model for this fold
            model = build_key_prediction_model(input_dim=max_length)
            model.fit(
                X_train,
                Y_train,
                epochs=100,
                batch_size=16,
                verbose=1
            )

            # Testing phase
            log_file.write(f"\n--- Validation Results for Fold {fold + 1} ---\n")
            person_representative_keys = []
            for person_idx in val_index:
                person_segments, ground_truth_key = all_persons_data[person_idx]

                # Predict keys for all segments using the fold's model
                predictions = model.predict(person_segments)
                # Average predicted keys across all segments for this person
                averaged_prediction = np.mean(predictions, axis=0)
                representative_key = (averaged_prediction > 0.5).astype(int)

                person_representative_keys.append(representative_key)
                person_keys_across_folds[person_idx].append(representative_key)

                # Check intra-person consistency: Now irrelevant since we averaged,
                # but we can still look at how each segment differs from the averaged key.
                segmentwise_keys = (predictions > 0.5).astype(int)
                hamming_distances = [
                    hamming(representative_key, segmentwise_keys[i]) * len(representative_key)
                    for i in range(len(segmentwise_keys))
                ]
                avg_intra_distance = np.mean(hamming_distances) if len(hamming_distances) > 0 else 0.0
                log_file.write(
                    f"Person {person_idx + 1} - Average Intra-Person Hamming Distance (vs rep. key): {avg_intra_distance:.2f}\n"
                )

                # Check match with ground truth key
                ground_truth_array = np.array(ground_truth_key)
                # Compare representative key with ground truth
                match_distance = hamming(ground_truth_array, representative_key) * len(ground_truth_array)
                log_file.write(
                    f"Person {person_idx + 1} - Representative Key Distance with Ground Truth: {match_distance:.2f}\n"
                )

            # Inter-person Hamming distance among persons in the validation set
            log_file.write(f"\n--- Inter-Person Hamming Distances (Fold {fold + 1}) ---\n")
            num_keys = len(person_representative_keys)
            for i in range(num_keys):
                for j in range(i + 1, num_keys):
                    distance = hamming(person_representative_keys[i], person_representative_keys[j]) * len(person_representative_keys[i])
                    log_file.write(
                        f"Hamming distance between Person {val_index[i]+1} and Person {val_index[j]+1}: {distance:.2f}\n"
                    )
            log_file.write("\n")

        # Compare representative keys across folds for each person
        # This checks how stable the predicted keys are across different folds
        log_file.write("\n--- Consistency of Representative Keys Across Folds ---\n")
        for person_idx, keys in person_keys_across_folds.items():
            if len(keys) > 1:
                # Compute the cross-fold distances for this person
                cross_fold_distances = [
                    hamming(keys[i], keys[j]) * len(keys[i]) for i in range(len(keys)) for j in range(i + 1, len(keys))
                ]
                avg_cross_fold_distance = np.mean(cross_fold_distances) if len(cross_fold_distances) > 0 else 0.0
                log_file.write(
                    f"Person {person_idx + 1} - Average Cross-Fold Hamming Distance: {avg_cross_fold_distance:.2f}\n"
                )

    print(f"Cross-validation completed. Results saved to {log_file_path}")

if __name__ == "__main__":
    base_directory = '/Users/lrm00020/PycharmProjects/KEY-GENERATION-using-DL/segmented_ecg_data1'
    keys_file = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\ground_keys\secrets_random_keys.json'
    train_and_test_prebuilt_keys(base_directory, keys_file)
