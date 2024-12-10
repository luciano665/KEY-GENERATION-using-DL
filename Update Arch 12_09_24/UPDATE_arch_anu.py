import json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     BatchNormalization, ReLU, Add, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import hamming
import tensorflow as tf

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_keys(file_path):
    """Load pre-built ground truth keys from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_valid_ecg_segments(directory, required_length=170):
    """Load ECG segments of the required length from a directory of CSV files."""
    ecg_segments = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            segment_path = os.path.join(directory, filename)
            segment_data = pd.read_csv(segment_path, header=None, skiprows=1).values.flatten()
            if len(segment_data) == required_length:
                ecg_segments.append(segment_data)
    return ecg_segments

def conv_block(x, filters, kernel_size=3, strides=1, padding='same'):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding,
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def residual_block(x, filters, downsample=False):
    shortcut = x
    if downsample:
        x = conv_block(x, filters, strides=2)
        shortcut = Conv1D(filters, 1, strides=2, padding='same',
                          kernel_regularizer=l2(1e-4))(shortcut)
    else:
        x = conv_block(x, filters)
    x = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def resnet24_encoder(input_dim):
    """
    Build a ResNet-like encoder to extract features from ECG signals.
    """
    inputs = Input(shape=(input_dim, 1))
    x = conv_block(inputs, filters=64, kernel_size=7, strides=2)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    # 4 times residual_block with filters=64
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    # downsample and 3 times with 128
    x = residual_block(x, filters=128, downsample=True)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    # downsample and 3 times with 256
    x = residual_block(x, filters=256, downsample=True)
    x = residual_block(x, filters=256)
    x = residual_block(x, filters=256)
    # downsample and 3 times with 512
    x = residual_block(x, filters=512, downsample=True)
    x = residual_block(x, filters=512)
    x = residual_block(x, filters=512)
    encoded = GlobalAveragePooling1D()(x)
    return inputs, encoded

def build_key_prediction_model(input_dim, output_dim=256):
    """
    Build a key prediction model using the encoder ResNet.
    Add moderate dropout and L2 regularization to help generalization.
    """
    inputs, encoded = resnet24_encoder(input_dim)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(encoded)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    key_output = Dense(output_dim, activation='sigmoid', name='key_output',
                       kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs, key_output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy')
    return model


def train_and_test_prebuilt_keys(base_directory, keys_file, log_file_path="segment_level_cv_log_ANU.txt"):
    """
    Train and test the model with pre-built random keys using segment-level cross-validation.
    Each fold uses one segment from each person as the validation segment, ensuring all persons
    are present in training every time (except for the held-out validation segments).
    """
    max_length = 170
    prebuilt_keys = load_keys(keys_file)

    # Load all persons and their segments
    all_persons_data = []
    for person_dir in sorted(os.listdir(base_directory)):
        if not person_dir.startswith("Person_"):
            continue
        person_id = int(person_dir.split('_')[-1])
        if person_id == 74:
            # Skip missing folder scenario
            continue
        # Adjust person_id > 74
        adjusted_person_id = person_id - 1 if person_id > 74 else person_id

        key_name = f"Person_{adjusted_person_id:02d}"
        if key_name not in prebuilt_keys:
            # Skip if no corresponding key
            continue

        person_path = os.path.join(base_directory, person_dir, 'rec_2_filtered')
        if not os.path.isdir(person_path):
            continue

        segments = load_valid_ecg_segments(person_path, required_length=max_length)
        if len(segments) < 2:
            # Need at least 2 segments to do any validation
            continue

        ground_key = prebuilt_keys[key_name]
        all_persons_data.append({
            'person_id': person_id,
            'segments': segments,
            'ground_key': ground_key
        })

    if not all_persons_data:
        print("No data loaded. Exiting.")
        return

    # Determine the number of folds based on the minimum number of segments across all persons
    # Each fold will hold out one segment index per person
    min_segments = min(len(p['segments']) for p in all_persons_data)
    num_folds = min_segments

    with open(log_file_path, 'w') as log_file:
        log_file.write("Segment-Level Cross-Validation Log\n")
        log_file.write("=================================\n\n")

        person_keys_across_folds = {p['person_id']: [] for p in all_persons_data}

        for fold in range(num_folds):
            log_file.write(f"\n--- Fold {fold + 1} ---\n")

            X_train, Y_train = [], []
            X_val, Y_val = [], []
            val_person_ids = []

            # Build training and validation sets for this fold
            for p in all_persons_data:
                segments = p['segments']
                ground_key = p['ground_key']
                person_id = p['person_id']

                if len(segments) <= fold:
                    # This person doesn't have enough segments for this fold
                    # Skip this person entirely for this fold
                    continue

                # Validation segment for this person in this fold
                val_segment = segments[fold]
                # Training segments are all others
                train_segments = [s for i, s in enumerate(segments) if i != fold]

                # Add training data
                for s in train_segments:
                    X_train.append(s)
                    Y_train.append(ground_key)

                # Add validation data
                X_val.append(val_segment)
                Y_val.append(ground_key)
                val_person_ids.append(person_id)

            # Convert to numpy arrays
            X_train = np.array(X_train).reshape(-1, max_length, 1)
            Y_train = np.array(Y_train)
            X_val = np.array(X_val).reshape(-1, max_length, 1)
            Y_val = np.array(Y_val)

            # Build and train the model
            model = build_key_prediction_model(input_dim=max_length)
            callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1,
                      validation_data=(X_val, Y_val), callbacks=[callback])

            log_file.write(f"\n--- Validation Results for Fold {fold + 1} ---\n")

            # Evaluate on the validation segments
            predictions = model.predict(X_val)
            # For each person in the validation set, compute representative key and metrics
            # Because there's only one segment per person in validation this fold, that segment's
            # prediction is itself the "representative key" for validation.

            # Convert each prediction to binary key
            bin_predictions = (predictions > 0.5).astype(int)

            # Store results per person
            person_rep_keys = {}
            for i, person_id in enumerate(val_person_ids):
                predicted_key = bin_predictions[i]
                ground_key = Y_val[i]
                person_rep_keys[person_id] = predicted_key
                person_keys_across_folds[person_id].append(predicted_key)

                # Intra-person distance (only one segment in val, so it's trivially 0)
                # If we had multiple val segments per person, we'd average here.
                intra_distance = 0.0

                # Distance with ground truth
                gt_dist = hamming(ground_key, predicted_key) * len(ground_key)

                log_file.write(
                    f"Person {person_id} - Average Intra-Person Hamming Distance (vs rep. key): {intra_distance:.2f}\n"
                )
                log_file.write(
                    f"Person {person_id} - Representative Key Distance with Ground Truth: {gt_dist:.2f}\n"
                )

            # Inter-person distances
            log_file.write(f"\n--- Inter-Person Hamming Distances (Fold {fold + 1}) ---\n")
            person_ids_in_val = list(person_rep_keys.keys())
            for i in range(len(person_ids_in_val)):
                for j in range(i + 1, len(person_ids_in_val)):
                    p1 = person_ids_in_val[i]
                    p2 = person_ids_in_val[j]
                    dist = hamming(person_rep_keys[p1], person_rep_keys[p2]) * len(person_rep_keys[p1])
                    log_file.write(f"Hamming distance between Person {p1} and Person {p2}: {dist:.2f}\n")

        # Check consistency across folds
        log_file.write("\n--- Consistency of Representative Keys Across Folds ---\n")
        for pid, keys_list in person_keys_across_folds.items():
            if len(keys_list) > 1:
                # Compute cross-fold distances
                cross_fold_distances = [
                    hamming(keys_list[i], keys_list[j]) * len(keys_list[i])
                    for i in range(len(keys_list)) for j in range(i + 1, len(keys_list))
                ]
                avg_cross_fold_distance = np.mean(cross_fold_distances) if cross_fold_distances else 0.0
                log_file.write(f"Person {pid} - Average Cross-Fold Hamming Distance: {avg_cross_fold_distance:.2f}\n")

    print(f"Cross-validation completed. Results saved to {log_file_path}")


if __name__ == "__main__":
    base_directory = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\segmented_ecg_data1'
    keys_file = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\ground_keys\ANU.json'
    train_and_test_prebuilt_keys(base_directory, keys_file)

