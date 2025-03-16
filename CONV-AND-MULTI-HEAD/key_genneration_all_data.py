import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MultiHeadAttention, LayerNormalization, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. Data Loader with rec_2_filtered Handling
# ==============================================================================
class ECGKeyLoader:
    def __init__(self, data_dir, key_path):
        self.data_dir = data_dir
        self.key_map = self._load_keys(key_path)
        self.persons = self._load_persons()
        self._validate_dataset()

    def _load_keys(self, key_path):
        with open(key_path) as f:
            return {int(k.split('_')[-1]): np.array(v, dtype=np.float32)
                    for k, v in json.load(f).items()}

    def _load_persons(self):
        persons = []
        valid_ids = set(self.key_map.keys())

        for dir_name in sorted(os.listdir(self.data_dir)):
            if not dir_name.startswith("Person_"):
                continue

            try:
                person_id = int(dir_name.split('_')[-1].lstrip('0'))
                if person_id == 0:  # Handle Person_00 case if exists
                    person_id = int(dir_name.split('_')[-1])
            except ValueError:
                print(f"Skipping invalid directory: {dir_name}")
                continue

            if person_id not in valid_ids:
                print(f"No key found for {dir_name}, skipping")
                continue

            person_path = os.path.join(self.data_dir, dir_name)

            segments = self._load_segments(person_path)
            if len(segments) == 0:
                print(f"No valid segments in {dir_name}, skipping")
                continue

            persons.append({
                'id': person_id,
                'segments': segments,
                'key': self.key_map[person_id]
            })
            print(f"Loaded {len(segments)} segments from {dir_name}")

        return persons

    def _validate_dataset(self):
        if not self.persons:
            raise ValueError("No valid persons with both keys and ECG segments found")

        total_segments = sum(len(p['segments']) for p in self.persons)
        print(f"Dataset contains {len(self.persons)} persons with {total_segments} total segments")

        if total_segments < 10:
            raise ValueError("Insufficient data for training (min 10 segments required)")

    def _load_segments(self, person_path, seq_len=170):
        segments = []
        for root, dirs, files in os.walk(person_path):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                file_path = os.path.join(root, file)
                try:
                    #Skip first row
                    ecg = np.loadtxt(file_path, delimiter=',', skiprows=1)
                    if ecg.ndim != 1 or len(ecg) != seq_len:
                        print(f"Invalid ECG format in {file_path}")
                        continue
                    segments.append(ecg.astype(np.float32))
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        return  np.array(segments)

    def get_train_data(self, test_size=0.2):
        X, y, ids = [], [], []
        for p in self.persons:
            X.extend(p['segments'])
            y.extend([p['key']] * len(p['segments']))
            ids.extend([p['id']] * len(p['segments']))

        X = np.array(X).reshape(-1, 170, 1)
        y = np.array(y)

        if len(X) < 2:
            raise ValueError(f"Need at least 2 samples, got {len(X)}")

        return train_test_split(X, y, test_size=test_size, stratify=ids)


# ==============================================================================
# 2. Hybrid Transformer-CNN Model
# ==============================================================================
class BioKeyTransformer(Model):
    def __init__(self, num_persons, key_bits=256):
        super().__init__()
        self.num_persons = num_persons
        self.key_bits = key_bits

        # Input processing
        self.conv_stack = tf.keras.Sequential([
            Conv1D(64, 15, activation='relu', padding='same'),
            Conv1D(128, 10, activation='relu', padding='same'),
            Conv1D(256, 5, activation='relu', padding='same')
        ])

        # Transformer components
        self.attention = MultiHeadAttention(num_heads=8, key_dim=64)
        self.norm = LayerNormalization(epsilon=1e-6)

        # Key projection
        self.gap = GlobalAveragePooling1D()
        self.key_proj = Dense(key_bits, activation='sigmoid')

    def call(self, inputs):
        x = self.conv_stack(inputs)
        attn_output = self.attention(x, x)
        x = self.norm(x + attn_output)
        x = self.gap(x)
        return self.key_proj(x)


# ==============================================================================
# 3. Training System with Key Generation
# ==============================================================================
class KeyGenerationSystem:
    def __init__(self, data_dir, key_path):
        self.loader = ECGKeyLoader(data_dir, key_path)
        self.model = None

    def train(self, epochs=100, batch_size=32):
        X_train, X_val, y_train, y_val = self.loader.get_train_data()

        self.model = BioKeyTransformer(num_persons=len(self.loader.persons))
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
        )
        return history

    def generate_key(self, ecg_segments, threshold=0.5):
        """Generate final key from multiple segments"""
        if ecg_segments is None or len(ecg_segments) == 0:
            raise ValueError("No ECG segments provided for key generation")

        predictions = self.model.predict(np.array(ecg_segments))
        avg_prob = np.mean(predictions, axis=0)
        return (avg_prob > threshold).astype(np.int32)


# ==============================================================================
# 4. Main Execution with Error Handling
# ==============================================================================
if __name__ == "__main__":
    DATA_DIR = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/segmented_ecg_data"
    KEY_FILE = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/secrets_random_keys.json"

    try:
        print("Initializing system...")
        kgs = KeyGenerationSystem(DATA_DIR, KEY_FILE)

        print("\nStarting training...")
        kgs.train(epochs=100)

        # ----------------------------
        # Test key generation for all persons
        # ----------------------------
        print("\nTesting key generation for all persons:")

        # Dictionary to hold aggregated keys for inter-person comparisons
        aggregated_keys = {}
        all_intra_keys = []

        for person in kgs.loader.persons:
            segments = person['segments']
            # Generate aggregated key from all segments for this person
            aggregated_key = kgs.generate_key(segments)
            ground_truth = person['key'].astype(np.int32)
            accuracy = np.mean(aggregated_key == ground_truth)

            print(f"\nPerson {person['id']}:")
            print(f"  Aggregated Key Accuracy: {accuracy:.2%}")
            print(f"  Aggregated Key: {aggregated_key[:24]}...")
            print(f"  Ground Truth:   {ground_truth[:24]}...")

            aggregated_keys[person['id']] = aggregated_key

            # ----------------------------
            # Compute Intra-Person Hamming Distance
            # ----------------------------
            # Predict keys for individual segments (each row in predictions)
            predictions = kgs.model.predict(np.array(segments))
            individual_keys = (predictions > 0.5).astype(np.int32)
            num_keys = individual_keys.shape[0]

            if num_keys > 1:
                distances = []
                for i in range(num_keys):
                    for j in range(i + 1, num_keys):
                        # Hamming distance: count of differing bits
                        d = np.sum(individual_keys[i] != individual_keys[j])
                        distances.append(d)
                avg_distance = np.mean(distances)
                print(f"  Intra-person average Hamming distance: {avg_distance:.2f} bits")
                all_intra_keys.extend(distances)
            else:
                print("  Not enough segments to compute intra-person Hamming distance.")
                # Overall intra HD statistics
            if all_intra_keys:
                overall_intra_mean = np.mean(all_intra_keys)
                overall_intra_std = np.std(all_intra_keys)
                print("\nOverall Intra-person Hamming Distance: "
                        f"mean= {overall_intra_mean:.2f} bits, std= {overall_intra_std:.2f} bits")
            else:
                print("\nNo data available to compute mean and std for Intra-person Hamming Distance.")
        # ----------------------------
        # Compute Inter-Person Hamming Distances (aggregated keys)
        # ----------------------------
        person_ids = sorted(aggregated_keys.keys())
        person_inter_dists = {p: [] for p in person_ids}
        print("\nInter-person Hamming distances (aggregated keys):")
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                key1 = aggregated_keys[person_ids[i]]
                key2 = aggregated_keys[person_ids[j]]
                d = int(np.sum(key1 != key2))
                # Save the distance for both persons
                person_inter_dists[person_ids[i]].append(d)
                person_inter_dists[person_ids[j]].append(d)
                print(f"  Distance between Person {person_ids[i]} and Person {person_ids[j]}: {d} bits")

        # Compute overall inter-person statistics by flattening the dictionary values:
        all_inter_distances = []
        for dist_list in person_inter_dists.values():
            all_inter_distances.extend(dist_list)
        if all_inter_distances:
            overall_inter_mean = np.mean(all_inter_distances)
            overall_inter_std = np.std(all_inter_distances)
            print("\nOverall Inter-person Hamming Distance: "
                  f"mean = {overall_inter_mean:.2f} bits, std = {overall_inter_std:.2f} bits")
        else:
            print("\nNo data available to compute inter-person Hamming Distance statistics.")

        # ----------------------------
        # Save the raw distance data for later plotting:
        with open("all_intra_distances.pkl", "wb") as f:
            pickle.dump(all_intra_keys, f)

        with open("person_inter_dists.pkl", "wb") as f:
            pickle.dump(person_inter_dists, f)


    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Verification Checklist:")
        print("1. Directory structure: Person_XX/rec_2_filtered/*.csv")
        print("2. CSV files contain exactly 170 values, no headers")
        print("3. JSON keys match Person_XX numbering (1-89)")
        print("4. Minimum 10 segments across all persons")
