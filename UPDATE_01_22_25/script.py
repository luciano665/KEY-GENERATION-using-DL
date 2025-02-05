import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Lambda
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
                continue  # Skip person 74

            adjusted_id = self._adjust_id(original_id)
            if adjusted_id not in self.key_map:
                continue

            person_path = os.path.join(self.base_dir, dir_name, 'rec_2_filtered')
            segments = self._load_segments(person_path)

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
                    segment = pd.read_csv(
                        os.path.join(dir_path, fname),
                        header=None
                    ).values.flatten()
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
# 3. Transformer Architecture Components
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


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='gelu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class KeyGenerator(Model):
    def __init__(self, num_persons, key_bits=256):
        super().__init__()
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.num_layers = 4

        # Input processing
        self.input_proj = Dense(self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.dff)
            for _ in range(self.num_layers)
        ]

        # Output heads
        self.global_pool = GlobalAveragePooling1D()
        self.key_head = Dense(key_bits, activation='sigmoid')
        self.embedding_head = Dense(128, activation='tanh')
        self.consistency_head = Dense(num_persons, activation='softmax')

    def call(self, inputs):
        x = self.input_proj(inputs)
        x = self.pos_encoding(x)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        pooled = self.global_pool(x)

        return {
            'key': self.key_head(pooled),
            'embedding': self.embedding_head(pooled),
            'consistency': self.consistency_head(pooled)
        }


# ======================================================================
# 4. Hybrid Loss Function
# ======================================================================
class KeyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
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
        return (self.alpha * key_loss +
                self.beta * triplet_loss +
                self.gamma * consistency_loss)


# ======================================================================
# 5. Complete Training Pipeline
# ======================================================================
class KeyTrainingSystem:
    def __init__(self, data_loader):
        self.data = data_loader.person_data
        self.num_persons = len(data_loader.key_map)
        self.model = KeyGenerator(self.num_persons)
        self.optimizer = tfa.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
        self.model.compile(optimizer=self.optimizer, loss=KeyLoss())

    def train(self, epochs=100, batch_size=32):
        train_data, val_data = self._prepare_datasets()
        train_gen = self._data_generator(train_data, batch_size)
        val_gen = self._data_generator(val_data, batch_size)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        return self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[early_stop],
            steps_per_epoch=len(train_data) // batch_size,
            validation_steps=len(val_data) // batch_size
        )

    def _prepare_datasets(self, test_split=0.1, val_split=0.2):
        train_data = []
        val_data = []

        for person in self.data:
            segments = person['segments']
            n_segments = len(segments)

            # Split indices
            val_idx = int(n_segments * (1 - val_split - test_split))
            test_idx = int(n_segments * (1 - test_split))

            # Store splits
            train_data.append({
                **person,
                'segments': segments[:val_idx],
                'split': 'train'
            })
            val_data.append({
                **person,
                'segments': segments[val_idx:test_idx],
                'split': 'val'
            })

        return train_data, val_data

    def _data_generator(self, data, batch_size=32):
        while True:
            batch = {
                'inputs': [],
                'targets': {
                    'key': [],
                    'embedding': [],
                    'consistency': []
                }
            }

            # Select random persons
            selected_persons = np.random.choice(data, size=batch_size // 4)

            for person in selected_persons:
                # Select 4 segments per person
                segs = person['segments']
                if len(segs) < 4:
                    segs = np.repeat(segs, 4 // len(segs) + 1, axis=0)

                indices = np.random.choice(len(segs), 4, replace=False)
                selected_segs = segs[indices]

                # Apply augmentation
                for i in range(4):
                    if np.random.rand() > 0.5 and person['split'] == 'train':
                        selected_segs[i] = ECGAugmenter.augment(selected_segs[i])

                # Store in batch
                batch['inputs'].extend(selected_segs)
                batch['targets']['key'].extend([person['key']] * 4)
                batch['targets']['embedding'].extend([person['id']] * 4)
                batch['targets']['consistency'].extend(
                    [tf.one_hot(person['id'], self.num_persons)] * 4
                )

            # Convert to numpy arrays
            inputs = np.array(batch['inputs']).reshape(-1, 170, 1).astype(np.float32)
            targets = {
                'key': np.array(batch['targets']['key']),
                'embedding': np.array(batch['targets']['embedding']),
                'consistency': np.array(batch['targets']['consistency'])
            }

            yield inputs, targets


# ======================================================================
# 6. Complete Evaluation System
# ======================================================================
class KeyEvaluator:
    def __init__(self, model, data, log_file="results.txt"):
        self.model = model
        self.data = data
        self.log_file = log_file
        self.results = defaultdict(dict)

    def evaluate(self):
        self._calculate_distances()
        self._cryptographic_analysis()
        self._save_results()
        return self.results

    def _calculate_distances(self):
        for person in self.data:
            segments = person['segments'].reshape(-1, 170, 1).astype(np.float32)
            outputs = self.model.predict(segments, verbose=0)

            # Generate final key using temporal averaging
            final_key = (np.mean(outputs['key'], axis=0) > 0.5).astype(int)

            # Calculate intra-person distances
            binary_preds = (outputs['key'] > 0.5).astype(int)
            intra_dists = [hamming(final_key, k) * 256 for k in binary_preds]

            # Calculate ground truth distance
            gt_dist = hamming(final_key, person['key']) * 256

            self.results[person['id']] = {
                'intra_mean': np.mean(intra_dists),
                'intra_std': np.std(intra_dists),
                'gt_dist': gt_dist,
                'key': final_key
            }

    def _cryptographic_analysis(self):
        all_keys = [v['key'] for v in self.results.values()]

        # Bit balance analysis
        bit_balance = np.mean([np.mean(k) for k in all_keys])

        # Avalanche effect
        diffs = []
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                diffs.append(hamming(all_keys[i], all_keys[j]))

        # Unique keys
        unique_keys = len(set(map(tuple, all_keys)))

        self.results['metrics'] = {
            'bit_balance': bit_balance,
            'avalanche_effect': np.mean(diffs),
            'unique_keys': f"{unique_keys}/{len(all_keys)}"
        }

    def _save_results(self):
        with open(self.log_file, 'w') as f:
            # Header
            f.write("ECG Cryptographic Key Generation Report\n")
            f.write("========================================\n\n")

            # Individual Results
            f.write("Per-Subject Results:\n")
            f.write("-" * 40 + "\n")
            for pid in sorted(self.results.keys()):
                if pid == 'metrics':
                    continue
                res = self.results[pid]
                f.write(f"Subject {pid:03d}:\n")
                f.write(f"  Intra-Segment Consistency: {res['intra_mean']:.2f} ± {res['intra_std']:.2f} bits\n")
                f.write(f"  Ground Truth Distance:     {res['gt_dist']:.2f} bits\n")
                f.write(f"  Generated Key:             {self._truncate_key(res['key'])}\n\n")

            # Cryptographic Metrics
            f.write("\nCryptographic Properties:\n")
            f.write("-" * 40 + "\n")
            metrics = self.results['metrics']
            f.write(f"Bit Balance (0-1):           {metrics['bit_balance']:.3f}\n")
            f.write(f"Avalanche Effect (%% changed): {metrics['avalanche_effect'] * 100:.1f}%%\n")
            f.write(f"Unique Keys Generated:       {metrics['unique_keys']}\n")

    def _truncate_key(self, key, length=16):
        return ''.join(map(str, key[:length])) + '...'


# ======================================================================
# 7. Main Execution
# ======================================================================
if __name__ == "__main__":
    # Configuration
    BASE_DIR = "/path/to/ecg_data"
    KEY_FILE = "/path/to/ground_truth_keys.json"
    LOG_FILE = "/path/to/results.txt"

    # Initialize data loader
    loader = ECGDataLoader(BASE_DIR, KEY_FILE)

    # Initialize training system
    trainer = KeyTrainingSystem(loader)

    # Train model
    print("Starting training...")
    history = trainer.train(epochs=100)
    print("Training completed!")

    # Evaluate model
    print("\nStarting evaluation...")
    evaluator = KeyEvaluator(trainer.model, loader.person_data, LOG_FILE)
    results = evaluator.evaluate()
    print(f"Evaluation complete! Results saved to {LOG_FILE}")