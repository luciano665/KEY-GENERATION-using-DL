
BASE_DIR = "/content/drive/MyDrive/ECG_Data"  # Update with your path
KEY_FILE = "/content/drive/MyDrive/ECG_Data/ground_truth_keys.json"
LOG_FILE = "/content/drive/MyDrive/ECG_Results/results.txt"


import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import hamming
from collections import defaultdict

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)


# [markdown]
# ### 5. Complete Data Loader
#
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


#  [markdown]
# ### 6. Model Components
#
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
        self.encoder = self._build_encoder()
        self.key_head = Dense(key_bits, activation='sigmoid')
        self.embedding_head = Dense(128, activation='tanh')
        self.consistency_head = Dense(num_persons, activation='softmax')

    def _build_encoder(self):
        inputs = Input(shape=(170, 1))
        x = Dense(self.d_model)(inputs)
        x = PositionalEncoding(self.d_model)(x)
        for _ in range(4):
            x = TransformerBlock(self.d_model, 8, 512)(x)
        x = GlobalAveragePooling1D()(x)
        return Model(inputs, x)

    def call(self, inputs):
        embeddings = self.encoder(inputs)
        return {
            'key': self.key_head(embeddings),
            'embedding': self.embedding_head(embeddings),
            'consistency': self.consistency_head(embeddings)
        }


#  [markdown]
# ### 7. Hybrid Loss Function
#
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


# [markdown]
# ### 8. Data Generators
#
def create_data_generator(person_data, batch_size=32):
    while True:
        batch = {
            'key': [],
            'embedding': [],
            'consistency': []
        }

        # Select random persons
        selected_persons = np.random.choice(person_data, size=batch_size // 4)

        for person in selected_persons:
            # Select 4 segments per person
            segs = person['segments']
            if len(segs) < 4:
                segs = np.repeat(segs, 4 // len(segs) + 1, axis=0)

            indices = np.random.choice(len(segs), 4, replace=False)
            selected_segs = segs[indices]

            # Add to batch
            batch['key'].extend([person['key']] * 4)
            batch['embedding'].extend([person['id']] * 4)
            batch['consistency'].extend([tf.one_hot(person['id'], len(person_data))] * 4)

            yield (selected_segs.reshape(-1, 170, 1).astype(np.float32),
                   {'key': np.array(batch['key']),
                    'embedding': np.array(batch['embedding']),
                    'consistency': np.array(batch['consistency'])})


#  [markdown]
# ### 9. Complete Training Setup
#
# Initialize data loader
loader = ECGDataLoader(BASE_DIR, KEY_FILE)

# Create data generators
train_data = [p for p in loader.person_data if len(p['segments']) >= 8]
train_gen = create_data_generator(train_data)
val_gen = create_data_generator(train_data)  # Use same for demonstration

# Build model
model = KeyGenerator(num_persons=len(loader.key_map))

# Compile with custom loss
optimizer = tfa.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss=KeyLoss())

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# [markdown]
# ### 10. Training Execution
#
print("Starting training...")
history = model.fit(
    train_gen,
    steps_per_epoch=100,
    validation_data=val_gen,
    validation_steps=20,
    epochs=100,
    callbacks=[early_stop]
)
print("Training completed!")


#  [markdown]
# ### 11. Complete Evaluation System
#
class KeyEvaluator:
    def __init__(self, model, data, log_file):
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
            # Process all segments
            segments = person['segments'].reshape(-1, 170, 1).astype(np.float32)
            outputs = self.model.predict(segments, verbose=0)

            # Generate final key
            final_key = (np.mean(outputs['key'], axis=0) > 0.5).astype(int)

            # Calculate metrics
            intra_dist = np.mean([hamming(final_key, k) * 256
                                  for k in (outputs['key'] > 0.5).astype(int)])
            gt_dist = hamming(final_key, person['key']) * 256

            self.results[person['id']] = {
                'intra': intra_dist,
                'gt': gt_dist,
                'key': final_key
            }

    def _cryptographic_analysis(self):
        all_keys = [v['key'] for v in self.results.values()]
        self.results['metrics'] = {
            'bit_balance': np.mean([np.mean(k) for k in all_keys]),
            'avalanche': np.mean([hamming(k1, k2) for i, k1 in enumerate(all_keys)
                                  for j, k2 in enumerate(all_keys) if i < j]),
            'unique': len(set(map(tuple, all_keys)))
        }

    def _save_results(self):
        with open(self.log_file, 'w') as f:
            # Header
            f.write("ECG Cryptographic Key Generation Report\n")
            f.write("=======================================\n\n")

            # Individual Results
            f.write("Per-Subject Results:\n")
            f.write("-" * 40 + "\n")
            for pid in sorted(self.results.keys()):
                if pid == 'metrics':
                    continue
                res = self.results[pid]
                f.write(f"Subject {pid:03d}:\n")
                f.write(f"  Intra-Segment Consistency: {res['intra']:.2f} bits\n")
                f.write(f"  Ground Truth Distance:     {res['gt']:.2f} bits\n")
                f.write(f"  Generated Key:             {''.join(map(str, res['key'][:16]))}...\n\n")

            # Cryptographic Metrics
            f.write("\nOverall Cryptographic Metrics:\n")
            f.write("-" * 40 + "\n")
            metrics = self.results['metrics']
            f.write(f"Bit Balance (0-1):           {metrics['bit_balance']:.3f}\n")
            f.write(f"Avalanche Effect (%% changed): {metrics['avalanche'] * 100:.1f}%%\n")
            f.write(f"Unique Keys Generated:       {metrics['unique']}/{len(self.data)}\n")


#  [markdown]
# ### 12. Run Evaluation
#
print("\nStarting evaluation...")
evaluator = KeyEvaluator(model, loader.person_data, LOG_FILE)
results = evaluator.evaluate()
print(f"Evaluation complete! Results saved to {LOG_FILE}")