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
    # Residual blocks
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

def build_key_prediction_model(input_dim, output_dim=256):
    """
    Build a key prediction model using the encoder ResNet.
    Add dropout and L2 regularization to help generalization.
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

def train_and_test_prebuilt_keys(base_directory, keys_file, log_file_path="train_val_test_results_12_12_secrets.txt"):
    """
    Train and test the model with pre-built random keys using a 70/20/10 split per person.
    No representative keys. Each segment is treated individually.
    """
    max_length = 170
    prebuilt_keys = load_keys(keys_file)

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
            continue

        person_path = os.path.join(base_directory, person_dir, 'rec_2_filtered')
        if not os.path.isdir(person_path):
            continue

        segments = load_valid_ecg_segments(person_path, required_length=max_length)
        if len(segments) < 3:
            # Need at least 3 segments for a meaningful 70/20/10 split
            continue

        ground_key = prebuilt_keys[key_name]
        segments = np.array(segments)
        np.random.shuffle(segments)

        total = len(segments)
        train_count = int(0.7 * total)
        val_count = int(0.2 * total)
        test_count = total - train_count - val_count

        train_segments = segments[:train_count]
        val_segments = segments[train_count:train_count+val_count]
        test_segments = segments[train_count+val_count:]

        all_persons_data.append({
            'person_id': person_id,
            'train_segments': train_segments,
            'val_segments': val_segments,
            'test_segments': test_segments,
            'ground_key': ground_key
        })

    if not all_persons_data:
        print("No data loaded. Exiting.")
        return

    # Build training and validation sets
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    for p in all_persons_data:
        for seg in p['train_segments']:
            X_train.append(seg)
            Y_train.append(p['ground_key'])
        for seg in p['val_segments']:
            X_val.append(seg)
            Y_val.append(p['ground_key'])

    X_train = np.array(X_train).reshape(-1, max_length, 1)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val).reshape(-1, max_length, 1)
    Y_val = np.array(Y_val)

    model = build_key_prediction_model(input_dim=max_length)
    callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1,
              validation_data=(X_val, Y_val), callbacks=[callback])

    # Evaluate on train, val, and test sets
    with open(log_file_path, 'w') as log_file:
        log_file.write("No Representative Key Approach - Results\n")
        log_file.write("========================================\n\n")

        # Collect predictions for all segments
        for p in all_persons_data:
            pid = p['person_id']
            ground_key = p['ground_key']

            # Predict keys for all segments of this person
            all_segments = np.concatenate([p['train_segments'], p['val_segments'], p['test_segments']], axis=0)
            preds = model.predict(all_segments.reshape(-1, max_length, 1))
            bin_preds = (preds > 0.5).astype(int)

            # Intra-Person Consistency: Compute average pairwise Hamming distances among all segments
            # If the person has N segments, we compute distances between each pair
            num_segs = len(all_segments)
            intra_distances = []
            for i in range(num_segs):
                for j in range(i+1, num_segs):
                    dist = hamming(bin_preds[i], bin_preds[j]) * len(bin_preds[i])
                    intra_distances.append(dist)
            avg_intra = np.mean(intra_distances) if intra_distances else 0.0

            # Distance to Ground Truth: Compare each segment's predicted key to the ground truth key
            gt_dists = [hamming(ground_key, bp)*len(ground_key) for bp in bin_preds]
            avg_gt_dist = np.mean(gt_dists)

            log_file.write(f"Person {pid}: Avg Intra-Person Distance: {avg_intra:.2f}, Avg Distance to GT: {avg_gt_dist:.2f}\n")

        # Inter-Person Distances: Compute average distance between each pair of persons’ segment keys
        # For each person, we already have predictions in bin_preds. Let's store them first:
        person_keys = {}
        for p in all_persons_data:
            pid = p['person_id']
            all_segments = np.concatenate([p['train_segments'], p['val_segments'], p['test_segments']], axis=0)
            preds = model.predict(all_segments.reshape(-1, max_length, 1))
            bin_preds = (preds > 0.5).astype(int)
            person_keys[pid] = bin_preds

        log_file.write("\n--- Inter-Person Hamming Distances ---\n")
        person_ids = sorted(person_keys.keys())
        for i in range(len(person_ids)):
            for j in range(i+1, len(person_ids)):
                pid1 = person_ids[i]
                pid2 = person_ids[j]
                # Compute the average distance between all segments of pid1 and all segments of pid2
                # This is a cross combination of their segments
                segs_pid1 = person_keys[pid1]
                segs_pid2 = person_keys[pid2]

                inter_distances = []
                for k1 in range(len(segs_pid1)):
                    for k2 in range(len(segs_pid2)):
                        dist = hamming(segs_pid1[k1], segs_pid2[k2]) * len(segs_pid1[k1])
                        inter_distances.append(dist)
                avg_inter = np.mean(inter_distances) if inter_distances else 0.0
                log_file.write(f"Hamming distance between Person {pid1} and Person {pid2}: {avg_inter:.2f}\n")

    print(f"Training and evaluation completed. Results saved to {log_file_path}")

if __name__ == "__main__":
    base_directory = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\segmented_ecg_data1'
    keys_file = '/Users\lrm00020\PycharmProjects\KEY-GENERATION-using-DL\ground_keys\secrets_random_keys.json'
    train_and_test_prebuilt_keys(base_directory, keys_file)
