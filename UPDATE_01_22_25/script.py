import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import hamming
from collections import defaultdict
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy


# =============================================================================
# 1. Data Loading & Preprocessing
# =============================================================================
class ECGDataLoader:
    def __init__(self, base_dir, key_path):
        self.base_dir = base_dir
        self.key_map = self._load_keys(key_path)
        self.person_data = self._load_persons()

    def _load_keys(self, key_path):
        with open(key_path) as f:
            raw_keys = json.load(f)
        # Convert each key to a NumPy array.
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
                print("Skipping Person_74")
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


# =============================================================================
# 2. Data Augmentation
# =============================================================================
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


# =============================================================================
# 3. Transformer Architecture Components
# =============================================================================
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


# Here we build a multi-task model.
class KeyGenerator(Model):
    def __init__(self, num_persons, key_units=256):
        super().__init__()
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.num_layers = 4

        self.input_proj = Dense(self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        self.transformer_blocks = [TransformerBlock(self.d_model, self.num_heads, self.dff)
                                   for _ in range(self.num_layers)]
        self.global_pool = GlobalAveragePooling1D()
        self.key_head = Dense(key_units, activation='sigmoid')
        self.class_head = Dense(num_persons, activation='softmax')
        # Build model with a fixed input shape.
        self.build((None, 170, 1))

    def call(self, inputs):
        x = self.input_proj(inputs)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.global_pool(x)
        key_out = self.key_head(x)  # (batch, 256)
        class_out = self.class_head(x)  # (batch, num_persons)
        return [key_out, class_out]


# =============================================================================
# 4. Loss Function: MultiTaskContrastiveLoss
# =============================================================================
class MultiTaskContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, alpha=1.0, beta=1.0, gamma=0.5):
        """
        alpha: Weight for key reconstruction (BCE loss)
        beta: Weight for subject classification (CCE loss)
        gamma: Weight for bit balance loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = BinaryCrossentropy()
        self.cce = CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # y_true: tuple (key_target, subject_target)
        # y_pred: list [key_pred, class_pred]
        key_target = y_true[0]  # shape (batch, 256)
        subject_target = tf.reshape(y_true[1], [-1])  # shape (batch,)
        key_pred = y_pred[0]  # shape (batch, 256)
        class_pred = y_pred[1]  # shape (batch, num_classes)

        loss_key = self.bce(key_target, key_pred)
        subject_onehot = tf.one_hot(tf.cast(subject_target, tf.int32), depth=self.num_classes)
        loss_class = self.cce(subject_onehot, class_pred)
        loss_balance = tf.reduce_mean(tf.square(tf.reduce_mean(key_pred, axis=0) - 0.5))
        return self.alpha * loss_key + self.beta * loss_class + self.gamma * loss_balance


# =============================================================================
# 5. Training Pipeline
# =============================================================================
class KeyTrainingSystem:
    def __init__(self, data_loader):
        self.data = data_loader.person_data
        self.num_persons = len(data_loader.key_map)
        self.model = KeyGenerator(self.num_persons, key_units=256)
        self.optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-4)
        self.model.compile(optimizer=self.optimizer,
                           loss=MultiTaskContrastiveLoss(num_classes=self.num_persons))

    def _data_generator(self, data, batch_size=32):
        """
        Each sample is created by randomly selecting one ECG segment from a random subject.
        Yields:
            inputs: ECG segment of shape (170, 1)
            targets: tuple (ground_truth_key, subject_id) where:
                     ground_truth_key is (256,) and subject_id is an integer.
        """
        while True:
            inputs, key_targets, subject_ids = [], [], []
            for _ in range(batch_size):
                person = np.random.choice(data, size=1)[0]
                segs = person['segments']
                idx = np.random.choice(len(segs))
                segment = segs[idx]
                segment = ECGAugmenter.augment(segment)
                inputs.append(segment.reshape(170, 1))
                key = np.atleast_1d(np.array(person['key'], dtype=np.float32))
                if key.shape[0] < 256:
                    key = np.pad(key, (0, 256 - key.shape[0]), mode='constant')
                elif key.shape[0] > 256:
                    key = key[:256]
                key_targets.append(key)
                subject_ids.append(person['id'])
            inputs = np.array(inputs, dtype=np.float32).reshape(batch_size, 170, 1)
            key_targets = np.array(key_targets, dtype=np.float32)
            subject_ids = np.array(subject_ids, dtype=np.int32)
            yield inputs, (key_targets, subject_ids)

    def _create_dataset(self, data, batch_size=32):
        return tf.data.Dataset.from_generator(
            lambda: self._data_generator(data, batch_size),
            output_signature=(
                tf.TensorSpec(shape=(None, 170, 1), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(None, 256), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            )
        )

    def train(self, epochs=100, batch_size=32):
        dataset = self._create_dataset(self.data, batch_size)
        early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        return self.model.fit(dataset, epochs=epochs, steps_per_epoch=100, callbacks=[early_stop])


# =============================================================================
# 6. Evaluation
# =============================================================================
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
        # For each subject, predict keys for all segments and average them.
        for person in self.data:
            segs = person['segments']
            inputs = np.array(segs, dtype=np.float32).reshape(-1, 170, 1)
            preds = self.model.predict(inputs, verbose=0)  # preds[0]: predicted keys (num_segments, 256)
            final_key = (np.mean(preds[0], axis=0) > 0.5).astype(int)
            binary_preds = (preds[0] > 0.5).astype(int)
            intra_dists = [hamming(final_key, k) * 256 for k in binary_preds]
            gt_dist = hamming(final_key, person['key']) * 256
            self.results[person['id']] = {
                'intra_mean': np.mean(intra_dists),
                'intra_std': np.std(intra_dists),
                'gt_dist': gt_dist,
                'key': final_key
            }

    def _cryptographic_analysis(self):
        all_keys = [v['key'] for k, v in self.results.items() if isinstance(k, int)]
        bit_balance = np.mean([np.mean(k) for k in all_keys])
        inter_dists = []
        for i in range(len(all_keys)):
            for j in range(i + 1, len(all_keys)):
                inter_dists.append(hamming(all_keys[i], all_keys[j]) * 256)
        unique_keys = len(set(map(tuple, all_keys)))
        self.results['metrics'] = {
            'bit_balance': bit_balance,
            'mean_inter_person_distance': np.mean(inter_dists),
            'std_inter_person_distance': np.std(inter_dists),
            'unique_keys': f"{unique_keys}/{len(all_keys)}"
        }

    def _save_results(self):
        with open(self.log_file, 'w') as f:
            f.write("ECG Cryptographic Key Generation Report\n")
            f.write("========================================\n\n")
            f.write("Per-Subject Results:\n")
            f.write("-" * 40 + "\n")
            for pid in sorted([k for k in self.results.keys() if isinstance(k, int)]):
                res = self.results[pid]
                f.write(f"Subject {pid:03d}:\n")
                f.write(f"  Intra-Segment Consistency: {res['intra_mean']:.2f} ± {res['intra_std']:.2f} bits\n")
                f.write(f"  Ground Truth Distance:     {res['gt_dist']:.2f} bits\n")
                f.write(f"  Generated Key:             {self._truncate_key(res['key'])}\n\n")
            f.write("\nCryptographic Properties:\n")
            f.write("-" * 40 + "\n")
            metrics = self.results['metrics']
            f.write(f"Bit Balance (0-1):           {metrics['bit_balance']:.3f}\n")
            f.write(f"Mean Inter-Person Distance:  {metrics['mean_inter_person_distance']:.2f} bits\n")
            f.write(f"Std Inter-Person Distance:   {metrics['std_inter_person_distance']:.2f} bits\n")
            f.write(f"Unique Keys Generated:       {metrics['unique_keys']}\n")

    def _truncate_key(self, key, length=16):
        return ''.join(map(str, key[:length])) + '...'


# =============================================================================
# 7. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Update these paths as needed.
    BASE_DIR = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/segmented_ecg_data"
    KEY_FILE = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/secrets_random_keys.json"
    LOG_FILE = "results_transformer_secrets.txt"

    # Initialize data loader.
    loader = ECGDataLoader(BASE_DIR, KEY_FILE)
    print(f"Total persons loaded: {len(loader.person_data)}")
    for person in loader.person_data:
        print(f"Person {person['id']}: {len(person['segments'])} segments")

    # Initialize training system.
    trainer = KeyTrainingSystem(loader)
    print("Starting training...")
    history = trainer.train(epochs=100, batch_size=32)
    print("Training completed!")

    # Evaluate model.
    print("\nStarting evaluation...")
    evaluator = KeyEvaluator(trainer.model, loader.person_data, LOG_FILE)
    results = evaluator.evaluate()
    print(f"Evaluation complete! Results saved to {LOG_FILE}")
