import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MultiHeadAttention, LayerNormalization, Dense, GlobalAveragePooling1D, \
    Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. Data Loader with rec_2_filtered Handling (unchanged)
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

            rec_path = os.path.join(self.data_dir, dir_name, "rec_2_filtered")
            if not os.path.exists(rec_path):
                print(f"Missing rec_2_filtered in {dir_name}, skipping")
                continue

            segments = self._load_segments(rec_path)
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

    def _load_segments(self, rec_path, seq_len=170):
        segments = []
        for fname in os.listdir(rec_path):
            if not fname.endswith('.csv'):
                continue

            file_path = os.path.join(rec_path, fname)
            try:
                # Skip first row with skiprows=1
                ecg = np.loadtxt(file_path, delimiter=',', skiprows=1)
                if ecg.ndim != 1 or len(ecg) != seq_len:
                    print(f"Invalid ECG format in {fname}")
                    continue

                segments.append(ecg.astype(np.float32))
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        return np.array(segments)

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
# 2. Transformer-Based Model for Key Generation
# ==============================================================================
# Transformer encoder block for 1D signals
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(mlp_dim, activation='relu')
        self.dense2 = Dense(embed_dim)
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Self-attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        # Feed-forward network with residual connection
        ffn_output = self.dense1(out1)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


# Patch embedding layer: splits the 1D input into patches and projects them
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = Conv1D(filters=embed_dim,
                                 kernel_size=patch_size,
                                 strides=patch_size,
                                 padding='valid')

    def call(self, x):
        # x: (batch, seq_len, channels)
        x = self.projection(x)  # (batch, num_patches, embed_dim)
        return x


# Complete Transformer model for ECG-based key generation
class TransformerKeyGenerator(tf.keras.Model):
    def __init__(self, seq_len=170, patch_size=10, embed_dim=64,
                 num_heads=4, mlp_dim=128, num_transformer_blocks=4,
                 key_bits=256, dropout_rate=0.1):
        super(TransformerKeyGenerator, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = seq_len // patch_size
        self.key_bits = key_bits

        # Embed patches
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        # Create trainable positional embeddings
        self.pos_embed = self.add_weight(name="pos_embed",
                                         shape=(1, self.num_patches, embed_dim),
                                         initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        # Stack of transformer encoder blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]
        # Global average pooling over patches
        self.gap = GlobalAveragePooling1D()
        # Final dense layer to project to key bits
        self.key_proj = Dense(key_bits, activation='sigmoid')

    def call(self, inputs, training=False):
        # Patch embedding and add positional information
        x = self.patch_embed(inputs)  # shape: (batch, num_patches, embed_dim)
        x = x + self.pos_embed

        # Pass through transformer encoder blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Global pooling to get latent representation
        x = self.gap(x)
        # Final projection to produce key
        return self.key_proj(x)


# ==============================================================================
# 3. Training and Key Generation System
# ==============================================================================
class KeyGenerationSystem:
    def __init__(self, data_dir, key_path):
        self.loader = ECGKeyLoader(data_dir, key_path)
        self.model = None

    def train(self, epochs=100, batch_size=32):
        X_train, X_val, y_train, y_val = self.loader.get_train_data()
        print("X_train shape:" , X_train.shape)

        # Initialize the transformer-based model.
        # We assume each key is a binary vector (default length 256).
        self.model = TransformerKeyGenerator(
            seq_len=170,
            patch_size=10,  # 170/10 gives 17 patches per segment
            embed_dim=64,
            num_heads=4,
            mlp_dim=128,
            num_transformer_blocks=4,
            key_bits=y_train.shape[1] if len(y_train.shape) > 1 else 256,
            dropout_rate=0.1
        )
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
        """Generate a final aggregated key from multiple segments."""
        if ecg_segments is None or len(ecg_segments) == 0:
            raise ValueError("No ECG segments provided for key generation")
        ecg_segments = np.array(ecg_segments)
        if ecg_segments.ndim == 2:
            ecg_segments = ecg_segments[..., np.newaxis]
        predictions = self.model.predict(ecg_segments)
        avg_prob = np.mean(predictions, axis=0)
        return (avg_prob > threshold).astype(np.int32)


# ==============================================================================
# 4. Main Execution with Error Handling
# ==============================================================================
if __name__ == "__main__":
    # Update these paths to your local directories/files
    DATA_DIR = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/segmented_ecg_data"
    KEY_FILE = "/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/secrets_random_keys.json"

    try:
        print("Initializing system...")
        kgs = KeyGenerationSystem(DATA_DIR, KEY_FILE)

        print("\nStarting training...")
        kgs.train(epochs=100)

        #Forcing model to build with expected  input shape, Try -1
        #kgs.model.build((None, 170, 1))
        #perform a dummy forward pass
        _ = kgs.model(tf.zeros((1, 170, 1)))

        # ----------------------------
        # Test key generation for all persons
        # ----------------------------
        print("\nTesting key generation for all persons:")

        # Dictionary to hold aggregated keys for inter-person comparisons
        aggregated_keys = {}

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
            predictions = kgs.model.predict(np.array(segments))
            individual_keys = (predictions > 0.5).astype(np.int32)
            num_keys = individual_keys.shape[0]

            if num_keys > 1:
                distances = []
                for i in range(num_keys):
                    for j in range(i + 1, num_keys):
                        d = np.sum(individual_keys[i] != individual_keys[j])
                        distances.append(d)
                avg_distance = np.mean(distances)
                print(f"  Intra-person average Hamming distance: {avg_distance:.2f} bits")
            else:
                print("  Not enough segments to compute intra-person Hamming distance.")

        # ----------------------------
        # Compute Inter-Person Hamming Distances (aggregated keys)
        # ----------------------------
        print("\nInter-person Hamming distances (aggregated keys):")
        person_ids = sorted(aggregated_keys.keys())
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                key1 = aggregated_keys[person_ids[i]]
                key2 = aggregated_keys[person_ids[j]]
                distance = np.sum(key1 != key2)
                print(f"  Distance between Person {person_ids[i]} and Person {person_ids[j]}: {distance} bits")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Verification Checklist:")
        print("1. Directory structure: Person_XX/rec_2_filtered/*.csv")
        print("2. CSV files contain exactly 170 values, no headers")
        print("3. JSON keys match Person_XX numbering (1-89)")
        print("4. Minimum 10 segments across all persons")
