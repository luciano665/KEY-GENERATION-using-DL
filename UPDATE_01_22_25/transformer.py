import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, GlobalAveragePooling1D,
    MultiHeadAttention, Conv1D, MaxPooling1D, Embedding, Input
)
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from scipy.spatial.distance import hamming
from collections import defaultdict
from tensorflow.keras.losses import BinaryCrossentropy

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. Enhanced Data Loading & Preprocessing
# ==============================================================================
class ECGDataLoader:
    def __init__(self, base_dir, key_path):
        self.base_dir = base_dir
        self.key_map = self._load_and_validate_keys(key_path)
        self.person_data = self._load_persons()
        self._verify_data_integrity()

    def _load_and_validate_keys(self, key_path):
        with open(key_path) as f:
            raw_keys = json.load(f)

        key_map = {}
        for k, v in raw_keys.items():
            try:
                person_id = int(k.split('_')[-1])
            except ValueError:
                raise ValueError(f"Invalid key format: {k}")

            if person_id in key_map:
                raise ValueError(f"Duplicate person ID: {person_id}")

            key_array = np.array(v, dtype=np.float32)
            if len(key_array) != 256:
                raise ValueError(f"Key for {k} has invalid length: {len(key_array)}")

            key_map[person_id] = key_array
        return key_map

    def _load_persons(self):
        persons = []
        valid_ids = set()

        for dir_name in os.listdir(self.base_dir):
            if not dir_name.startswith("Person_"):
                continue

            try:
                person_id = int(dir_name.split('_')[-1])
            except ValueError:
                print(f"Skipping invalid directory: {dir_name}")
                continue

            if person_id in valid_ids:
                print(f"Duplicate person ID detected: {person_id}")
                continue

            person_path = os.path.join(self.base_dir, dir_name, 'rec_2_filtered')
            if not os.path.isdir(person_path):
                print(f"Missing rec_2_filtered in {dir_name}")
                continue

            segments = self._load_segments(person_path)
            if len(segments) < 10:
                print(f"Insufficient segments in {dir_name} ({len(segments)} found)")
                continue

            if person_id not in self.key_map:
                print(f"No key found for ID {person_id}")
                continue

            persons.append({
                'id': person_id,
                'segments': segments,
                'key': self.key_map[person_id]
            })
            valid_ids.add(person_id)

        print(f"Loaded {len(persons)} valid persons")
        return persons

    def _load_segments(self, dir_path, required_length=170):
        segments = []
        for fname in os.listdir(dir_path):
            if fname.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(dir_path, fname), header=None, skiprows=1)
                    segment = df.squeeze().values.astype(np.float32)
                    if len(segment) == required_length:
                        segments.append(segment)
                except Exception as e:
                    print(f"Error loading {fname}: {str(e)}")
        return np.array(segments)

    def _verify_data_integrity(self):
        key_lengths = [len(p['key']) for p in self.person_data]
        if len(set(key_lengths)) != 1:
            raise ValueError("Inconsistent key lengths detected")
        print("Data integrity check passed")


# ==============================================================================
# 2. Advanced Data Augmentation
# ==============================================================================
class ECGAugmenter:
    @staticmethod
    def augment(segment):
        if np.random.rand() > 0.5:
            segment = ECGAugmenter._add_structured_noise(segment)
        if np.random.rand() > 0.3:
            segment = ECGAugmenter._time_warp(segment)
        if np.random.rand() > 0.3:
            segment = ECGAugmenter._amplitude_shift(segment)
        return segment

    @staticmethod
    def _add_structured_noise(segment, noise_level=0.05):
        noise = np.random.normal(0, noise_level, segment.shape)
        noise = np.convolve(noise, np.ones(5) / 5, mode='same')
        return segment + noise

    @staticmethod
    def _time_warp(segment, max_warp=0.15):
        length = len(segment)
        warp_points = sorted(np.random.randint(0, length, 3))
        warped = []
        prev = 0
        for wp in warp_points + [length]:
            chunk = segment[prev:wp]
            warp_factor = 1 + np.random.uniform(-max_warp, max_warp)
            new_length = int(len(chunk) * warp_factor)
            warped.append(np.interp(np.linspace(0, len(chunk), new_length),
                                    np.arange(len(chunk)), chunk))
            prev = wp
        return np.concatenate(warped)[:length]

    @staticmethod
    def _amplitude_shift(segment, max_shift=0.2):
        return segment * (1 + np.random.uniform(-max_shift, max_shift))


# ==============================================================================
# 3. Hybrid Conv-Transformer Architecture
# ==============================================================================
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


class BioKeyGenerator(Model):
    def __init__(self, num_persons, key_bits=256):
        super().__init__()
        self.num_persons = num_persons

        # Input layers
        self.ecg_input = Input(shape=(170, 1), name='ecg_input')
        self.subject_input = Input(shape=(), dtype=tf.int32, name='subject_id')

        # Convolutional stem
        self.conv_stack = tf.keras.Sequential([
            Conv1D(64, 15, activation='relu', padding='same'),
            MaxPooling1D(3),
            Conv1D(128, 10, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(256, 5, activation='relu', padding='same'),
            GlobalAveragePooling1D()
        ])

        # Subject-aware modulation
        self.subject_embedding = Embedding(num_persons, 256)
        self.modulation_scale = Dense(256)
        self.modulation_shift = Dense(256)

        # Transformer tower with Lambda wrappers
        self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))
        self.transformer_blocks = [TransformerBlock(256, 8, 1024) for _ in range(6)]
        self.squeeze_dims = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 1))

        # Key projection
        self.key_proj = Dense(key_bits, activation='sigmoid')

        # Build the model using Functional API
        x = self.conv_stack(self.ecg_input)

        # Subject-specific modulation
        emb = self.subject_embedding(self.subject_input)
        scale = self.modulation_scale(emb)
        shift = self.modulation_shift(emb)
        x = tf.keras.layers.multiply([x, scale]) + shift

        # Transformer processing
        x = self.expand_dims(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.squeeze_dims(x)

        outputs = self.key_proj(x)

        # Initialize the model
        super().__init__(
            inputs=[self.ecg_input, self.subject_input],
            outputs=outputs
        )

    def call(self, inputs):
        # This method will be used during actual execution
        ecg_data, subject_ids = inputs
        x = self.conv_stack(ecg_data)

        # Subject-specific modulation
        emb = self.subject_embedding(subject_ids)
        scale = self.modulation_scale(emb)
        shift = self.modulation_shift(emb)
        x = x * scale + shift

        # Transformer processing
        x = tf.expand_dims(x, 1)
        for block in self.transformer_blocks:
            x = block(x)
        x = tf.squeeze(x, 1)

        return self.key_proj(x)


# ==============================================================================
# 4. Enhanced Loss Function with Contrastive Learning
# ==============================================================================
class EnhancedKeyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=5.0, beta=0.2, gamma=0.1, margin=0.3):
        super().__init__()
        self.alpha = alpha  # Ground truth matching
        self.beta = beta  # Intra-subject consistency
        self.gamma = gamma  # Bit balance
        self.margin = margin  # Inter-subject separation
        self.bce = BinaryCrossentropy()

    def call(self, y_true, y_pred, subject_ids):
        # Basic reconstruction loss
        bce_loss = self.bce(y_true, y_pred)

        # Intra-subject consistency
        unique_subjects, _ = tf.unique(subject_ids)
        consistency_loss = 0.0
        for subj in unique_subjects:
            mask = tf.equal(subject_ids, subj)
            subj_preds = tf.boolean_mask(y_pred, mask)
            consistency_loss += tf.math.reduce_variance(subj_preds)
        consistency_loss /= tf.cast(tf.size(unique_subjects), tf.float32)

        # Bit balance regularization
        bit_balance_loss = tf.reduce_mean(
            tf.square(tf.reduce_mean(y_pred, axis=0) - 0.5))

        # Contrastive separation loss
        norm_preds = tf.math.l2_normalize(y_pred, axis=1)
        similarity = tf.matmul(norm_preds, norm_preds, transpose_b=True)
        mask = tf.expand_dims(subject_ids, 1) != tf.expand_dims(subject_ids, 0)
        separation_loss = tf.reduce_mean(
            tf.maximum(self.margin - similarity, 0) * tf.cast(mask, tf.float32))

        total_loss = (self.alpha * bce_loss +
                      self.beta * consistency_loss +
                      self.gamma * bit_balance_loss +
                      separation_loss)

        # Add metric tracking
        self.add_metric(bce_loss, name='bce_loss')
        self.add_metric(consistency_loss, name='consistency_loss')
        self.add_metric(bit_balance_loss, name='bit_balance_loss')
        self.add_metric(separation_loss, name='separation_loss')

        return total_loss


# ==============================================================================
# 5. Stratified Batch Generation
# ==============================================================================
class StratifiedBatchGenerator:
    def __init__(self, data, subjects_per_batch=8, segments_per_subject=4):
        self.subjects_per_batch = subjects_per_batch
        self.segments_per_subject = segments_per_subject

        # Create subject-based registry
        self.subject_registry = {}
        for person in data:
            self.subject_registry[person['id']] = {
                'segments': person['segments'],
                'key': person['key']
            }

        self.subjects = list(self.subject_registry.keys())
        self.num_subjects = len(self.subjects)

    def __call__(self):
        while True:
            # Select subjects ensuring diversity
            selected_subjects = np.random.choice(
                self.subjects,
                size=min(self.subjects_per_batch, self.num_subjects),
                replace=False
            )

            batch_ecg = []
            batch_subjects = []
            batch_keys = []

            for subj_id in selected_subjects:
                subj_data = self.subject_registry[subj_id]
                segments = subj_data['segments']

                # Handle small sample sizes
                if len(segments) < self.segments_per_subject:
                    segments = np.tile(segments, (self.segments_per_subject // len(segments) + 1, 1))

                # Randomly select and augment segments
                indices = np.random.choice(len(segments), self.segments_per_subject, replace=False)
                selected_segments = [ECGAugmenter.augment(segments[i]) for i in indices]

                batch_ecg.extend(selected_segments)
                batch_subjects.extend([subj_id] * self.segments_per_subject)
                batch_keys.extend([subj_data['key']] * self.segments_per_subject)

            # Convert to numpy arrays
            inputs = {
                'ecg_input': np.array(batch_ecg).reshape(-1, 170, 1).astype(np.float32),
                'subject_id': np.array(batch_subjects)
            }
            targets = np.array(batch_keys)

            yield inputs, targets


# ==============================================================================
# 6. Training Pipeline
# ==============================================================================
def lr_schedule(epoch, lr):
    """Learning rate warmup and decay"""
    if epoch < 10:
        return 1e-5 * (10 ** (epoch / 10))
    elif epoch < 100:
        return 3e-4
    else:
        return 3e-4 * tf.math.exp(0.1 * (100 - epoch))


class KeyTrainingSystem:
    def __init__(self, data_loader):
        self.data = data_loader.person_data
        self.num_persons = len(data_loader.key_map)

        # Initialize model
        self.model = BioKeyGenerator(self.num_persons)
        self.optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-5)

        # Compile with custom loss
        self.model.compile(
            optimizer=self.optimizer,
            loss=EnhancedKeyLoss(alpha=5.0, beta=0.2, gamma=0.1),
        )

    def train(self, epochs=200, batch_size=32):
        # Split data
        np.random.shuffle(self.data)
        split_idx = int(0.8 * len(self.data))
        train_data = self.data[:split_idx]
        val_data = self.data[split_idx:]

        # Create data generators
        train_gen = StratifiedBatchGenerator(train_data)()
        val_gen = StratifiedBatchGenerator(val_data)()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            LearningRateScheduler(lr_schedule)
        ]

        # Train model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=50,
            validation_steps=20
        )
        return history


# ==============================================================================
# 7. Cryptographic Evaluation
# ==============================================================================
class CryptoEvaluator:
    def __init__(self, model, data):
        self.model = model
        self.data = {p['id']: p for p in data}
        self.results = defaultdict(dict)

    def evaluate(self, num_consensus=100):
        # Generate consensus keys
        for pid, person in self.data.items():
            consensus_key = self._generate_consensus_key(person, num_consensus)
            self.results[pid] = {
                'key': consensus_key,
                'gt_key': person['key']
            }

        # Calculate metrics
        self._calculate_intra_consistency()
        self._calculate_inter_distance()
        self._calculate_crypto_metrics()
        return self.results

    def _generate_consensus_key(self, person, num_samples):
        segments = person['segments']
        predictions = []

        for _ in range(num_samples):
            # Randomly select and augment segments
            idx = np.random.choice(len(segments), size=4, replace=True)
            batch = [ECGAugmenter.augment(segments[i]) for i in idx]

            # Predict
            inputs = {
                'ecg_input': np.array(batch).reshape(-1, 170, 1),
                'subject_id': np.array([person['id']] * 4)
            }
            preds = self.model.predict(inputs, verbose=0)
            predictions.append((preds > 0.5).astype(int))

        # Compute bit stability
        stability = np.mean(predictions, axis=0)

        # Apply consensus rules
        consensus = np.zeros(256, dtype=int)
        for bit in range(256):
            bit_stability = stability[:, bit]
            if np.std(bit_stability) < 0.1:  # Stable bit
                consensus[bit] = int(np.mean(bit_stability) > 0.5)
            else:  # Unstable bit, use majority vote
                consensus[bit] = np.round(np.mean(bit_stability))

        return consensus

    def _calculate_intra_consistency(self):
        for pid in self.results:
            person = self.data[pid]
            samples = []
            for _ in range(10):
                idx = np.random.choice(len(person['segments']), 4)
                batch = [person['segments'][i] for i in idx]
                inputs = {
                    'ecg_input': np.array(batch).reshape(-1, 170, 1),
                    'subject_id': np.array([pid] * 4)
                }
                preds = (self.model.predict(inputs, verbose=0) > 0.5).astype(int)
                samples.append(preds)

            # Calculate intra-consistency
            variances = []
            for bit in range(256):
                bit_values = [s[:, bit].mean() for s in samples]
                variances.append(np.var(bit_values))
            self.results[pid]['intra_var'] = np.mean(variances)

    def _calculate_inter_distance(self):
        keys = [v['key'] for v in self.results.values()]
        distances = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                distances.append(hamming(keys[i], keys[j]) * 256)
        self.results['metrics'] = {
            'mean_inter_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }

    def _calculate_crypto_metrics(self):
        keys = [v['key'] for v in self.results.values()]
        # Bit balance
        bit_balance = np.mean([np.mean(k) for k in keys])
        # Unique keys
        unique_keys = len(set(map(tuple, keys)))

        self.results['metrics'].update({
            'bit_balance': bit_balance,
            'unique_keys': f"{unique_keys}/{len(keys)}"
        })

    def save_report(self, filename="crypto_report.txt"):
        with open(filename, 'w') as f:
            f.write("Cryptographic Key Generation Report\n")
            f.write("====================================\n\n")

            # Per-subject results
            f.write("Per-Subject Metrics:\n")
            f.write("-" * 50 + "\n")
            for pid in sorted([k for k in self.results if isinstance(k, int)]):
                res = self.results[pid]
                f.write(f"Subject {pid:03d}:\n")
                f.write(f"  Intra-Consistency Variance: {res['intra_var']:.4f}\n")
                f.write(f"  Key: {self._truncate_key(res['key'])}\n")
                f.write(f"  GT Distance: {hamming(res['key'], res['gt_key']) * 256:.1f} bits\n\n")

            # Cryptographic metrics
            metrics = self.results['metrics']
            f.write("\nOverall Cryptographic Metrics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Bit Balance: {metrics['bit_balance']:.4f}\n")
            f.write(f"Unique Keys: {metrics['unique_keys']}\n")
            f.write(f"Mean Inter-Distance: {metrics['mean_inter_distance']:.1f} bits\n")
            f.write(f"Min Distance: {metrics['min_distance']:.1f} bits\n")
            f.write(f"Max Distance: {metrics['max_distance']:.1f} bits\n")

    def _truncate_key(self, key, length=16):
        return ''.join(map(str, key[:length])) + '...'


# ==============================================================================
# 8. Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Configure paths
    BASE_DIR = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/segmented_ecg_data"
    KEY_FILE = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/secrets_random_keys.json"

    # Initialize data loader
    print("Loading data...")
    loader = ECGDataLoader(BASE_DIR, KEY_FILE)

    # Initialize training system
    print("\nInitializing training system...")
    trainer = KeyTrainingSystem(loader)

    # Train model
    print("\nStarting training...")
    history = trainer.train(epochs=200)
    print("Training completed!")

    # Evaluate cryptographic properties
    print("\nStarting evaluation...")
    evaluator = CryptoEvaluator(trainer.model, loader.person_data)
    results = evaluator.evaluate(num_consensus=100)
    evaluator.save_report("final_report.txt")
    print("Evaluation complete! Results saved to final_report.txt")