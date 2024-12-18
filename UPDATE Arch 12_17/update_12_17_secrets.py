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
from sklearn.model_selection import KFold  # For cross-validation

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

def conv_block(x, filters, kernel_size=3, strides=1, padding='same', l2_reg=1e-3):
    """A single Conv1D + BN + ReLU block with stronger L2 regularization."""
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding,
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def residual_block(x, filters, downsample=False, l2_reg=1e-3):
    """A single residual block with optional downsampling."""
    shortcut = x
    if downsample:
        x = conv_block(x, filters, strides=2, l2_reg=l2_reg)
        shortcut = Conv1D(filters, 1, strides=2, padding='same',
                          kernel_regularizer=l2(l2_reg))(shortcut)
    else:
        x = conv_block(x, filters, l2_reg=l2_reg)

    x = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def bigger_resnet_encoder(input_dim):
    """
    A bigger ResNet-like encoder, doubling filter sizes for stronger representation.
    """
    inputs = Input(shape=(input_dim, 1))

    x = Conv1D(128, kernel_size=7, strides=2, padding='same', kernel_regularizer=l2(1e-3))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Stage 1: filters=128
    x = residual_block(x, filters=128, l2_reg=1e-3)
    x = residual_block(x, filters=128, l2_reg=1e-3)
    x = residual_block(x, filters=128, l2_reg=1e-3)

    # Stage 2: filters=256
    x = residual_block(x, filters=256, downsample=True, l2_reg=1e-3)
    x = residual_block(x, filters=256, l2_reg=1e-3)
    x = residual_block(x, filters=256, l2_reg=1e-3)

    # Stage 3: filters=512
    x = residual_block(x, filters=512, downsample=True, l2_reg=1e-3)
    x = residual_block(x, filters=512, l2_reg=1e-3)
    x = residual_block(x, filters=512, l2_reg=1e-3)

    # Stage 4: filters=1024
    x = residual_block(x, filters=1024, downsample=True, l2_reg=1e-3)
    x = residual_block(x, filters=1024, l2_reg=1e-3)
    x = residual_block(x, filters=1024, l2_reg=1e-3)

    encoded = GlobalAveragePooling1D()(x)
    return inputs, encoded

def build_key_prediction_model(input_dim, output_dim=256):
    """
    Build a key prediction model with the bigger ResNet encoder + stronger regularization.
    """
    inputs, encoded = bigger_resnet_encoder(input_dim)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-3))(encoded)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.3)(x)
    key_output = Dense(output_dim, activation='sigmoid', name='key_output',
                       kernel_regularizer=l2(1e-3))(x)

    model = Model(inputs, key_output)
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy')
    return model

def evaluate_intra_person(bin_preds):
    """Compute average pairwise Hamming distance among predictions (intra-person)."""
    num_segs = len(bin_preds)
    if num_segs < 2:
        return 0.0
    distances = []
    for i in range(num_segs):
        for j in range(i+1, num_segs):
            dist = hamming(bin_preds[i], bin_preds[j]) * len(bin_preds[i])
            distances.append(dist)
    return np.mean(distances)

def evaluate_to_gt(bin_preds, ground_key):
    """Compute average distance to the ground truth key."""
    if len(bin_preds) == 0:
        return 0.0
    gt_dists = [hamming(ground_key, bp)*len(ground_key) for bp in bin_preds]
    return np.mean(gt_dists)

def train_and_test_prebuilt_keys(base_directory, keys_file, log_file_path="train_val_test_cv_results_secrets.txt"):
    """
    Train and test the model with:
      1) Single 70/20/10 split
      2) 3-Fold cross-validation per person
    """
    k_folds = 3
    max_length = 170
    prebuilt_keys = load_keys(keys_file)

    # Gather data per person
    all_persons_data = []
    for person_dir in sorted(os.listdir(base_directory)):
        if not person_dir.startswith("Person_"):
            continue
        person_id = int(person_dir.split('_')[-1])
        if person_id == 74:
            # Skip missing folder scenario
            continue
        adjusted_person_id = person_id - 1 if person_id > 74 else person_id

        key_name = f"Person_{adjusted_person_id:02d}"
        if key_name not in prebuilt_keys:
            continue

        person_path_2 = os.path.join(base_directory, person_dir, 'rec_2_filtered')
        if not os.path.isdir(person_path_2):
            continue
        segments_2 = load_valid_ecg_segments(person_path_2, required_length=max_length)

        # Also load rec_1_filtered
        person_path_1 = os.path.join(base_directory, person_dir, 'rec_1_filtered')
        segments_1 = []
        if os.path.isdir(person_path_1):
            segments_1 = load_valid_ecg_segments(person_path_1, required_length=max_length)

        segments = segments_1 + segments_2
        if len(segments) < 3:
            continue

        ground_key = prebuilt_keys[key_name]
        all_persons_data.append({
            'person_id': person_id,
            'segments': np.array(segments),
            'ground_key': ground_key
        })

    if not all_persons_data:
        print("No data loaded. Exiting.")
        return

    # PART 1: Single 70/20/10 Split
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    single_split_persons_data = []

    for p in all_persons_data:
        pid = p['person_id']
        ground_key = p['ground_key']
        segments = p['segments']

        np.random.shuffle(segments)
        total = len(segments)
        train_count = int(0.7 * total)
        val_count = int(0.2 * total)
        test_count = total - train_count - val_count

        train_segments = segments[:train_count]
        val_segments = segments[train_count:train_count+val_count]
        test_segments = segments[train_count+val_count:]

        single_split_persons_data.append({
            'person_id': pid,
            'train_segments': train_segments,
            'val_segments': val_segments,
            'test_segments': test_segments,
            'ground_key': ground_key
        })

        for seg in train_segments:
            X_train.append(seg)
            Y_train.append(ground_key)
        for seg in val_segments:
            X_val.append(seg)
            Y_val.append(ground_key)

    X_train = np.array(X_train).reshape(-1, max_length, 1)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val).reshape(-1, max_length, 1)
    Y_val = np.array(Y_val)

    # Train single-split model
    model = build_key_prediction_model(input_dim=max_length)
    callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=1,
              validation_data=(X_val, Y_val), callbacks=[callback])

    with open(log_file_path, 'w') as log_file:
        # --- Single-Split Results ---
        log_file.write("Secrets Keys - RESULTS (Single 70/20/10 Split)\n")
        log_file.write("============================================================\n\n")

        # Evaluate (intra-person, distance to GT) per person on combined train+val+test
        for p in single_split_persons_data:
            pid = p['person_id']
            ground_key = p['ground_key']
            all_segments = np.concatenate([p['train_segments'], p['val_segments'], p['test_segments']], axis=0)
            preds = model.predict(all_segments.reshape(-1, max_length, 1))
            bin_preds = (preds > 0.5).astype(int)

            avg_intra = evaluate_intra_person(bin_preds)
            avg_gt_dist = evaluate_to_gt(bin_preds, ground_key)

            log_file.write(f"Person {pid}: Avg Intra-Person Distance: {avg_intra:.2f}, "
                           f"Avg Distance to GT: {avg_gt_dist:.2f}\n")

        # Inter-Person Distances (Single Split)
        log_file.write("\n--- Inter-Person Hamming Distances (Single Split) ---\n")
        person_keys = {}
        for p in single_split_persons_data:
            pid = p['person_id']
            all_segments = np.concatenate([p['train_segments'], p['val_segments'], p['test_segments']], axis=0)
            preds = model.predict(all_segments.reshape(-1, max_length, 1))
            bin_preds = (preds > 0.5).astype(int)
            person_keys[pid] = bin_preds

        person_ids = sorted(person_keys.keys())
        for i in range(len(person_ids)):
            for j in range(i+1, len(person_ids)):
                pid1 = person_ids[i]
                pid2 = person_ids[j]
                segs_pid1 = person_keys[pid1]
                segs_pid2 = person_keys[pid2]

                inter_distances = []
                for k1 in range(len(segs_pid1)):
                    for k2 in range(len(segs_pid2)):
                        dist = hamming(segs_pid1[k1], segs_pid2[k2]) * len(segs_pid1[k1])
                        inter_distances.append(dist)
                avg_inter = np.mean(inter_distances) if inter_distances else 0.0
                log_file.write(f"Hamming distance between Person {pid1} and Person {pid2}: {avg_inter:.2f}\n")

        # --- Cross-Validation (3-Folds) ---
        log_file.write("\n\n3-FOLD CROSS VALIDATION RESULTS\n")
        log_file.write("================================\n\n")

        kfold = KFold(n_splits=3, shuffle=True, random_state=42)

        for p in all_persons_data:
            pid = p['person_id']
            ground_key = p['ground_key']
            segments = p['segments']

            if len(segments) < 3:
                log_file.write(f"Person {pid}: Not enough segments ({len(segments)}) for 3-fold CV.\n")
                continue

            # Cross-validation over the segments for this person
            all_intra = []
            all_gt = []
            fold_idx = 0

            for train_index, test_index in kfold.split(segments):
                fold_idx += 1
                train_segments = segments[train_index]
                test_segments = segments[test_index]

                X_cv_train = np.array(train_segments).reshape(-1, max_length, 1)
                Y_cv_train = np.array([ground_key]*len(train_segments))

                # For each fold, build a new model
                cv_model = build_key_prediction_model(input_dim=max_length)
                cv_callback = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                cv_model.fit(X_cv_train, Y_cv_train, epochs=100, batch_size=16, verbose=0,
                             callbacks=[cv_callback])

                # Evaluate on the fold's test set
                X_cv_test = np.array(test_segments).reshape(-1, max_length, 1)
                preds_cv = cv_model.predict(X_cv_test)
                bin_preds_cv = (preds_cv > 0.5).astype(int)

                fold_intra = evaluate_intra_person(bin_preds_cv)
                fold_gt = evaluate_to_gt(bin_preds_cv, ground_key)
                all_intra.append(fold_intra)
                all_gt.append(fold_gt)

            if len(all_intra) > 0:
                avg_intra_cv = np.mean(all_intra)
                avg_gt_cv = np.mean(all_gt)
                log_file.write(f"Person {pid} [3-fold CV]: Avg Intra Distance: {avg_intra_cv:.2f}, "
                               f"Avg Distance to GT: {avg_gt_cv:.2f}\n")
            else:
                log_file.write(f"Person {pid}: CV not performed.\n")

    print(f"Training + single-split evaluation + 3-fold cross-validation completed. Results saved to {log_file_path}")

if __name__ == "__main__":
    base_directory = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\segmented_ecg_data1'  # Replace with your ECG data directory
    keys_file = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\ground_keys\secrets_random_keys.json'       # Replace with your ground truth keys JSON
    train_and_test_prebuilt_keys(base_directory, keys_file, "train_val_test_cv_results.txt")
