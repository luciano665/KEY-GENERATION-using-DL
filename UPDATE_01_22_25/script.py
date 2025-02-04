import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import hamming
from collections import defaultdict

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ======================================================================
# 1. Enhanced Data Loading & Preprocessing
# ======================================================================
class ECGDataLoader:
    def __init__(self, base_dir, key_path):
        self.base_dir = base_dir
        self.key_map = self._load_keys(key_path)
        self.person_data = self._load_persons()

    def _load_keys(self, key_path):
        with open(key_path) as f:
            raw_keys = json.load(f)
        return {
            self._adjust_id(int(k.split('_')[-1])): np.array(v, dtype=np.float32)
            for k, v in raw_keys.items()
        }

    def _adjust_id(self, person_id):
        return person_id - 1 if person_id > 74 else person_id

    def _load_persons(self):
        persons = []
        for dir_name in os.listdir(self.base_dir):
            if not dir_name.startswith("Person_"):
                continue
            original_id = int(dir_name.split('_')[-1])
            if original_id == 74:
                continue

            adjusted_id = self._adjust_id(original_id)
            if adjusted_id not in self.key_map:
                continue

            segments = self._load_segments(os.path.join(self.base_dir, dir_name, 'rec_2_filtered'))
            if len(segments) >= 3:
                persons.append({
                    'id': adjusted_id,
                    'original_id': original_id,
                    'segments': segments,
                    'key': self.key_map[adjusted_id]
                })
        return persons

    def _load_segments(self, dir_path, required_length=170):
        segments = []
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.endswith('.csv'):
                    segment = pd.read_csv(os.path.join(dir_path, fname),
                                          header=None).values.flatten()
                    if len(segment) == required_length:
                        segments.append(segment)
        return np.array(segments)


# ======================================================================
# 2. Advanced Data Augmentation
# ======================================================================
class ECGAugmenter:
    @staticmethod
    def augment(segment):
        # Randomly apply transformations
        if np.random.rand() > 0.5:
            segment = ECGAugmenter._add_noise(segment)
        if np.random.rand() > 0.3:
            segment = ECGAugmenter._time_warp(segment)
        return segment

    @staticmethod
    def _add_noise(segment, noise_level=0.03):
        return segment + np.random.normal(0, noise_level, segment.shape)

    @staticmethod
    def _time_warp(segment, max_warp=0.2):
        length = len(segment)
        warp = int(length * max_warp * np.random.uniform(-1, 1))
        return np.interp(np.arange(length), np.arange(length) + warp, segment)


# ======================================================================
# 3. Transformer Architecture with Triplet Learning
# ======================================================================
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:tf.shape(x)[1], :]


class KeyGenerator(Model):
    def __init__(self, num_persons, key_bits=256):
        super().__init__()
        self.encoder = self._build_encoder()
        self.key_head = Dense(key_bits, activation='sigmoid')
        self.embedding_head = Dense(128, activation=None)
        self.consistency_head = Dense(num_persons, activation='softmax')

    def _build_encoder(self):
        inputs = Input(shape=(170, 1))
        x = Dense(128)(inputs)
        x = PositionalEncoding(128)(x)
        x = TransformerBlock(128, 8, 512)(x)
        x = GlobalAveragePooling1D()(x)
        return Model(inputs, x)

    def call(self, inputs):
        embeddings = self.encoder(inputs)
        return {
            'key': self.key_head(embeddings),
            'embedding': self.embedding_head(embeddings),
            'consistency': self.consistency_head(embeddings)
        }


# ======================================================================
# 4. Hybrid Loss Function
# ======================================================================
class KeyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # BCE weight
        self.beta = beta  # Triplet weight
        self.gamma = gamma  # Consistency weight
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.triplet = tfa.losses.TripletSemiHardLoss()
        self.kl = tf.keras.losses.KLDivergence()

    def call(self, y_true, y_pred):
        key_loss = self.bce(y_true['key'], y_pred['key'])
        triplet_loss = self.triplet(y_true['embedding'], y_pred['embedding'])
        consistency_loss = self.kl(y_true['consistency'], y_pred['consistency'])
        return self.alpha * key_loss + self.beta * triplet_loss + self.gamma * consistency_loss


# ======================================================================
# 5. Training Pipeline
# ======================================================================
class KeyTrainingSystem:
    def __init__(self, data_loader):
        self.data = data_loader
        self.model = KeyGenerator(num_persons=len(data_loader.key_map))
        self.optimizer = tfa.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
        self.model.compile(optimizer=self.optimizer, loss=KeyLoss())

    def train(self, epochs=100):
        train_data, val_data = self._prepare_datasets()
        early_stop = EarlyStopping(patience=15, restore_best_weights=True)
        return self.model.fit(
            self._data_generator(train_data),
            validation_data=self._data_generator(val_data),
            epochs=epochs,
            callbacks=[early_stop]
        )

    def _prepare_datasets(self, test_split=0.1, val_split=0.2):
        # Implementation similar to previous prepare_datasets
        # ... (omitted for brevity)
        return train_data, val_data

    def _data_generator(self, data, batch_size=32):
        while True:
            batch = {
                'key': [],
                'embedding': [],
                'consistency': []
            }
            # Batch construction logic
            # ... (omitted for brevity)
            yield batch


# ======================================================================
# 6. Evaluation & Cryptographic Validation
# ======================================================================
class KeyEvaluator:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.results = defaultdict(dict)

    def evaluate(self):
        self._calculate_distances()
        self._cryptographic_analysis()
        return self.results

    def _calculate_distances(self):
        for person in self.data:
            segments = np.concatenate([person['train'], person['val'], person['test']])
            outputs = self.model.predict(segments.reshape(-1, 170, 1))
            final_key = self._generate_final_key(outputs['key'])

            # Intra-person consistency
            intra_dist = np.mean([hamming(final_key, k) * 256
                                  for k in (outputs['key'] > 0.5).astype(int)])

            # Store results
            self.results[person['id']]['intra'] = intra_dist
            self.results[person['id']]['key'] = final_key

        # Calculate inter-person distances
        keys = {pid: data['key'] for pid, data in self.results.items()}
        for pid1 in keys:
            for pid2 in keys:
                if pid1 != pid2:
                    dist = hamming(keys[pid1], keys[pid2]) * 256
                    self.results[pid1][f'dist_{pid2}'] = dist

    def _generate_final_key(self, preds, threshold=0.5):
        avg_pred = np.mean(preds, axis=0)
        return (avg_pred > threshold).astype(int)

    def _cryptographic_analysis(self):
        all_keys = [v['key'] for v in self.results.values()]
        # Bit balance analysis
        bit_balance = np.mean([np.mean(k) for k in all_keys])
        # Avalanche effect
        diffs = [np.mean(k1 != k2) for i, k1 in enumerate(all_keys)
                 for j, k2 in enumerate(all_keys) if i < j]

        print(f"\nCryptographic Analysis:")
        print(f"Average Bit Balance: {bit_balance:.3f} (ideal 0.5)")
        print(f"Avalanche Effect: {np.mean(diffs):.3f} (ideal 0.5)")


# ======================================================================
# 7. Main Execution
# ======================================================================
if __name__ == "__main__":
    # Configuration
    BASE_DIR = "/path/to/ecg_data"
    KEY_FILE = "/path/to/ground_truth_keys.json"

    # Initialize system
    loader = ECGDataLoader(BASE_DIR, KEY_FILE)
    trainer = KeyTrainingSystem(loader)

    # Train model
    print("Starting training...")
    history = trainer.train(epochs=100)

    # Evaluate
    evaluator = KeyEvaluator(trainer.model, loader.person_data)
    results = evaluator.evaluate()

    # Print summary
    print("\nFinal Results:")
    for pid, data in results.items():
        print(f"Person {pid}:")
        print(f"  Intra-Distance: {data['intra']:.2f} bits")
        print(f"  Inter-Distances: {[f'{k}: {v:.2f}' for k, v in data.items() if 'dist' in k]}")