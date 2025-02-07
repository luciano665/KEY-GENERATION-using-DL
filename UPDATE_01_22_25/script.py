import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW  # Using AdamW from Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import hamming
from collections import defaultdict
from tensorflow.keras.losses import BinaryCrossentropy, KLDivergence


# -----------------------------------------------------------------------------
# Helper functions & custom Triplet Semi-Hard Loss implementation
# -----------------------------------------------------------------------------
def _pairwise_distances(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)
    if not squared:
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)
    return distances


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    labels = tf.cast(labels, tf.int32)
    batch_size = tf.shape(embeddings)[0]
    pdist_matrix = _pairwise_distances(embeddings, squared=True)

    loss_list = []
    for i in tf.range(batch_size):
        anchor_label = labels[i]
        anchor_distance = pdist_matrix[i]
        positive_mask = tf.logical_and(
            tf.equal(labels, anchor_label),
            tf.not_equal(tf.range(batch_size), i)
        )
        positive_indices = tf.where(positive_mask)[:, 0]
        for j in positive_indices:
            d_ap = anchor_distance[j]
            negative_mask = tf.not_equal(labels, anchor_label)
            negative_distances = tf.boolean_mask(anchor_distance, negative_mask)
            valid_negatives = tf.boolean_mask(negative_distances, negative_distances > d_ap)
            valid_negatives = tf.boolean_mask(valid_negatives, valid_negatives < d_ap + margin)
            if tf.size(valid_negatives) > 0:
                d_an = tf.reduce_min(valid_negatives)
                loss_list.append(tf.maximum(d_ap - d_an + margin, 0.0))
            else:
                loss_list.append(0.0)
    if len(loss_list) == 0:
        return 0.0
    return tf.reduce_mean(tf.stack(loss_list))


class TripletSemiHardLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, **kwargs):
        super(TripletSemiHardLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        return triplet_semihard_loss(y_true, y_pred, margin=self.margin)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# 1. Enhanced Data Loading & Preprocessing
# -----------------------------------------------------------------------------
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
            try:
                original_id = int(dir_name.split('_')[-1])
            except ValueError:
                print(f"Skipping invalid directory: {dir_name}")
                continue
            print(f"Processing: {dir_name} (ID: {original_id})")
            if original_id == 74:
                print(f"Skipping Person_74")
                continue
            adjusted_id = self._adjust_id(original_id)
            if adjusted_id not in self.key_map:
                print(f"No key found for adjusted ID {adjusted_id}")
                continue
            person_path = os.path.join(self.base_dir, dir_name, 'rec_2_filtered')
            if not os.path.isdir(person_path):
                print(f"Missing rec_2_filtered in {dir_name}")
                continue
            segments = self._load_segments(person_path)
            if len(segments) < 3:
                print(f"Insufficient segments in {dir_name} ({len(segments)} found)")
                continue
            persons.append({
                'id': adjusted_id,
                'original_id': original_id,
                'segments': segments,
                'key': self.key_map[adjusted_id]
            })
        print(f"Loaded {len(persons)} valid persons")
        return persons

    def _load_segments(self, dir_path, required_length=170):
        segments = []
        if os.path.exists(dir_path):
            for fname in os.listdir(dir_path):
                if fname.endswith('.csv'):
                    segment_path = os.path.join(dir_path, fname)
                    try:
                        df = pd.read_csv(segment_path, header=None, skiprows=1)
                        segment = df.squeeze().values.astype(np.float32)
                        if len(segment) == required_length:
                            segments.append(segment)
                        else:
                            print(f"Invalid length in {fname}: {len(segment)}")
                    except Exception as e:
                        print(f"Error reading {fname}: {str(e)}")
        return np.array(segments)


# -----------------------------------------------------------------------------
# 2. Advanced Data Augmentation
# -----------------------------------------------------------------------------
class ECGAugmenter:
    @staticmethod
    def augment(segment):
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


# -----------------------------------------------------------------------------
# 3. Transformer Architecture Components
# -----------------------------------------------------------------------------
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
        self.key_head = Dense(key_bits, activation='sigmoid')
        self.embedding_head = Dense(128, activation='tanh')
        self.consistency_head = Dense(num_persons, activation='softmax')

    def call(self, inputs):
        x = self.input_proj(inputs)
        x = self.pos_encoding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        pooled = self.global_pool(x)
        key_out = self.key_head(pooled)
        embedding_out = self.embedding_head(pooled)
        consistency_out = self.consistency_head(pooled)
        # Return outputs as a list (order must match target order)
        return [key_out, embedding_out, consistency_out]

    @property
    def global_pool(self):
        return GlobalAveragePooling1D()


# -----------------------------------------------------------------------------
# 4. Hybrid Loss Function
# -----------------------------------------------------------------------------
class KeyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Binary Crossentropy weight
        self.beta = beta  # Triplet loss weight
        self.gamma = gamma  # Consistency loss weight
        self.bce = BinaryCrossentropy()
        self.triplet = TripletSemiHardLoss(margin=1.0)
        self.kl = KLDivergence()

    def call(self, y_true, y_pred):
        # y_true and y_pred are lists with:
        # [key_target, embedding_target, consistency_target]
        key_loss = self.bce(y_true[0], y_pred[0])
        triplet_loss = self.triplet(y_true[1], y_pred[1])
        consistency_loss = self.kl(y_true[2], y_pred[2])
        return (self.alpha * key_loss +
                self.beta * triplet_loss +
                self.gamma * consistency_loss)


# -----------------------------------------------------------------------------
# 5. Complete Training Pipeline
# -----------------------------------------------------------------------------
class KeyTrainingSystem:
    def __init__(self, data_loader):
        self.data = data_loader.person_data
        self.num_persons = len(data_loader.key_map)
        self.model = KeyGenerator(self.num_persons)
        self.optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-4)
        self.model.compile(optimizer=self.optimizer, loss=KeyLoss())

    def _data_generator(self, data, batch_size=32):
        while True:
            if len(data) == 0:
                raise ValueError("No training data available. Check your data loading configuration.")
            batch = {
                'inputs': [],
                'targets': {
                    'key': [],
                    'embedding': [],
                    'consistency': []
                }
            }
            # Select random persons for the batch
            selected_persons = np.random.choice(data, size=batch_size // 4)
            for person in selected_persons:
                segs = person['segments']
                if len(segs) < 4:
                    segs = np.repeat(segs, 4 // len(segs) + 1, axis=0)
                indices = np.random.choice(len(segs), 4, replace=False)
                selected_segs = segs[indices]
                for i in range(4):
                    if np.random.rand() > 0.5 and person['split'] == 'train':
                        selected_segs[i] = ECGAugmenter.augment(selected_segs[i])
                batch['inputs'].extend(selected_segs)
                batch['targets']['key'].extend([person['key']] * 4)
                batch['targets']['embedding'].extend([person['id']] * 4)
                batch['targets']['consistency'].extend(
                    [tf.one_hot(person['id'], self.num_persons)] * 4
                )
            inputs = np.array(batch['inputs']).reshape(-1, 170, 1).astype(np.float32)
            targets = [
                np.array(batch['targets']['key'], dtype=np.float32),
                np.array(batch['targets']['embedding'], dtype=np.int32),
                np.array(batch['targets']['consistency'], dtype=np.float32)
            ]
            yield inputs, targets

    def _create_dataset(self, data, batch_size):
        return tf.data.Dataset.from_generator(
            lambda: self._data_generator(data, batch_size),
            output_signature=(
                tf.TensorSpec(shape=(None, 170, 1), dtype=tf.float32),  # inputs
                (
                    tf.TensorSpec(shape=(None, 256), dtype=tf.float32),  # key target
                    tf.TensorSpec(shape=(None,), dtype=tf.int32),  # embedding target
                    tf.TensorSpec(shape=(None, self.num_persons), dtype=tf.float32)  # consistency target
                )
            )
        )

    def _prepare_datasets(self, test_split=0.1, val_split=0.2):
        train_data = []
        val_data = []
        for person in self.data:
            segments = person['segments']
            n_segments = len(segments)
            val_idx = int(n_segments * (1 - val_split - test_split))
            test_idx = int(n_segments * (1 - test_split))
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

    def train(self, epochs=100, batch_size=32):
        train_data, val_data = self._prepare_datasets()
        train_ds = self._create_dataset(train_data, batch_size)
        val_ds = self._create_dataset(val_data, batch_size)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stop],
            steps_per_epoch=len(train_data) // batch_size,
            validation_steps=len(val_data) // batch_size
        )


# -----------------------------------------------------------------------------
# 6. Complete Evaluation System
# -----------------------------------------------------------------------------
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
            final_key = (np.mean(outputs[0], axis=0) > 0.5).astype(int)
            binary_preds = (outputs[0] > 0.5).astype(int)
            intra_dists = [hamming(final_key, k) * 256 for k in binary_preds]
            gt_dist = hamming(final_key, person['key']) * 256
            self.results[person['id']] = {
                'intra_mean': np.mean(intra_dists),
                'intra_std': np.std(intra_dists),
                'gt_dist': gt_dist,
                'key': final_key
            }

    def _cryptographic_analysis(self):
        all_keys = [v['key'] for k, v in self.results.items() if k != 'metrics']
        bit_balance = np.mean([np.mean(k) for k in all_keys])
        diffs = []
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                diffs.append(hamming(all_keys[i], all_keys[j]))
        unique_keys = len(set(map(tuple, all_keys)))
        self.results['metrics'] = {
            'bit_balance': bit_balance,
            'avalanche_effect': np.mean(diffs),
            'unique_keys': f"{unique_keys}/{len(all_keys)}"
        }

    def _save_results(self):
        with open(self.log_file, 'w') as f:
            f.write("ECG Cryptographic Key Generation Report\n")
            f.write("========================================\n\n")
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
            f.write("\nCryptographic Properties:\n")
            f.write("-" * 40 + "\n")
            metrics = self.results['metrics']
            f.write(f"Bit Balance (0-1):           {metrics['bit_balance']:.3f}\n")
            f.write(f"Avalanche Effect (%% changed): {metrics['avalanche_effect'] * 100:.1f}%%\n")
            f.write(f"Unique Keys Generated:       {metrics['unique_keys']}\n")

    def _truncate_key(self, key, length=16):
        return ''.join(map(str, key[:length])) + '...'


# -----------------------------------------------------------------------------
# 7. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Update these paths as needed
    BASE_DIR = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/segmented_ecg_data"
    KEY_FILE = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/secrets_random_keys.json"
    LOG_FILE = "results_transformer_secrets.txt"

    # Initialize data loader
    loader = ECGDataLoader(BASE_DIR, KEY_FILE)
    print(f"Total persons loaded: {len(loader.person_data)}")
    for person in loader.person_data:
        print(f"Person {person['id']}: {len(person['segments'])} segments")

    # Initialize training system
    trainer = KeyTrainingSystem(loader)
    print("Starting training...")
    history = trainer.train(epochs=100)
    print("Training completed!")

    # Evaluate model
    print("\nStarting evaluation...")
    evaluator = KeyEvaluator(trainer.model, loader.person_data, LOG_FILE)
    results = evaluator.evaluate()
    print(f"Evaluation complete! Results saved to {LOG_FILE}")
