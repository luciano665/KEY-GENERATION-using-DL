import json
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, \
    ReLU, Add, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from scipy.spatial.distance import hamming
import tensorflow as tf


#Load ground_truth keys from json files
def load_keys(file_path):
    """Load_ground truth keys from JSON file"""
    dict_storage = {}
    with open(file_path, 'r') as f:
        for person_id, keys in json.load(f).items():
            dict_storage[person_id] = keys
    return dict_storage

#Traverse over keys to assign correct keys when using this ground keys to teh correct subject



def load_valid_ecg_segments(directory, required_length=170):
    ecg_segments = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.csv'):
            segment_path = os.path.join(directory, filename)
            segment_data = pd.read_csv(segment_path, header=None, skiprows=1).values.flatten()
            if len(segment_data) == required_length:
                ecg_segments.append(segment_data)
    return ecg_segments


def conv_block(x, filters, kernel_size=3, strides=1, padding='same'):
    x = Conv1D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def residual_block(x, filters, downsample=False):
    shortcut = x
    if downsample:
        x = conv_block(x, filters, strides=2)
        shortcut = Conv1D(filters, 1, strides=2, padding="same")(shortcut)
    else:
        x = conv_block(x, filters)
    x = Conv1D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x


def resnet24_encoder(input_dim):
    """
    Resnet 24 layers
    Learn the most important features of the ECG signal of person
    :param input_dim: 170 data points -> segment
    :return: segments of a person in very low dimesion = latent space and also the inputs as they are for ground truth key
    """
    inputs = Input(shape=(input_dim, 1))
    x = conv_block(inputs, filters=64, kernel_size=7, strides=2)
    x = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)
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

#Funtion we don't need anymore since we already have 89 keys generated for all subjects

"""
def build_key_only_model(input_dim, output_dim=256, ground_truth_key=None):
    inputs, encoded = resnet24_encoder(input_dim)
    x = Dense(512, activation='relu')(encoded)
    x = Dense(256, activation='relu')(x)
    key_output = Dense(output_dim, activation="sigmoid", name="key_output")(x)

    def custom_key_loss(y_true, y_pred):
        def loss_value():
            ground_truth_key_expanded = tf.expand_dims(ground_truth_key, axis=0)
            ground_truth_key_expanded = tf.tile(ground_truth_key_expanded, [tf.shape(y_pred)[0], 1])
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(ground_truth_key_expanded, y_pred))

        return tf.cond(
            tf.logical_and(tf.size(y_true) > 0, tf.size(y_pred) > 0),
            loss_value,
            lambda: tf.constant(0.0)
        )

    model = Model(inputs, key_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_key_loss)
    return model
"""




def generate_pv_keys_with_cross_validation(base_directory, log_file_path="summary_results_with_prototype_keys.txt"):
    max_length = 170
    all_persons_data = []
    ground_truth_keys = []

    # Load data from `rec_2_filtered` only and precompute ground truth keys
    for person_dir in sorted(os.listdir(base_directory)):
        person_path = os.path.join(base_directory, person_dir, 'rec_2_filtered')
        if os.path.isdir(person_path):
            segments = load_valid_ecg_segments(person_path, required_length=max_length)
            if segments:
                all_persons_data.append(np.array(segments).reshape(-1, max_length, 1))

                # Generate and store a prototype key for each person (computed once)
                prototype_model = build_key_only_model(max_length)
                segment_keys = []
                for segment in segments:
                    key = prototype_model.predict(np.array([segment]))
                    segment_keys.append((key[0] > 0.5).astype(int))
                ground_truth_key = np.mean(segment_keys, axis=0) > 0.5
                ground_truth_keys.append(ground_truth_key.astype(float))

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_keys_per_fold = []

    with open(log_file_path, 'w') as log_file:
        log_file.write("Detailed Cross-validation Logging\n")
        log_file.write("=================================\n\n")

        for fold, (train_index, val_index) in enumerate(kf.split(all_persons_data)):
            log_file.write(f"\n--- Fold {fold + 1} ---\n")
            fold_keys = []

            for person_idx, person_segments in enumerate(all_persons_data):
                ground_truth_key = ground_truth_keys[person_idx]
                autoencoder = build_key_only_model(max_length, ground_truth_key=ground_truth_key)
                autoencoder.fit(
                    person_segments,
                    np.tile(ground_truth_key, (len(person_segments), 1)),
                    epochs=10,
                    batch_size=16,
                    verbose=1
                )

                # Generate keys and log them only once for simplicity
                person_keys = []
                for segment in person_segments:
                    key = autoencoder.predict(np.array([segment]))
                    binary_key = (key[0] > 0.5).astype(int)
                    person_keys.append(binary_key)

                # Generate keys and log only a representative key for simplicity
                representative_key = person_keys[0]  # Choose the first key as the representative key for logging
                if fold == 0:  # Log only once for the first fold to save space
                    log_file.write(f"\nPerson {person_idx} Representative Key:\n")
                    log_file.write(f"{''.join(map(str, representative_key))}\n")

                # Calculate and log Hamming distances (intra-person consistency)
                intra_person_distances = [hamming(person_keys[0], k) * 256 for k in person_keys[1:]]
                avg_intra_distance = np.mean(intra_person_distances)
                log_file.write(
                    f"Average Intra-Person Hamming Distance for Person {person_idx} (expected ~0): {avg_intra_distance}\n")

                fold_keys.append(person_keys)
            all_keys_per_fold.append(fold_keys)

        # Cross-fold consistency check
        log_file.write("\n--- Cross-Fold Consistency Check ---\n")
        for person_idx in range(len(all_persons_data)):
            person_fold_keys = [all_keys_per_fold[fold][person_idx][0] for fold in range(n_splits)]
            fold_hamming_distances = [hamming(person_fold_keys[i], person_fold_keys[j]) * 256
                                      for i in range(n_splits) for j in range(i + 1, n_splits)]
            avg_fold_distance = np.mean(fold_hamming_distances)
            log_file.write(
                f"Average Cross-Fold Hamming Distance for Person {person_idx} (expected ~0): {avg_fold_distance}\n")

        # Inter-person consistency check (comparing one key of each person with another's key)
        log_file.write("\n--- Inter-Person Key Comparison ---\n")
        for i in range(len(all_persons_data) - 1):
            inter_person_hamming = hamming(all_keys_per_fold[0][i][0], all_keys_per_fold[0][i + 1][0]) * 256
            log_file.write(
                f"Hamming distance between Person {i} and Person {i + 1} (expected high): {inter_person_hamming}\n")

    print(f"Results saved to {log_file_path}")


# Example usage
if __name__ == '__main__':
    base_directory = '/Users/lucianomaldonado/PycharmProjects/ECG-PV-generation/segmented_ecg_data1'
    generate_pv_keys_with_cross_validation(base_directory)
