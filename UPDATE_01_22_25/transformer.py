import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from scipy.spatial.distance import hamming

# --------------------------------------------------------------------
# 1. Set Seeds for Reproducibility
# --------------------------------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------------------------------------------
# 2. Data Loading Functions
# --------------------------------------------------------------------
def load_keys(file_path):
    """
    - Keys represent a stable label for each subject. Aiming for every ECG segment
        of the subject to produce the same final 256-bit output after training
    - File should contain an entry for each subject dataset
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def load_valid_ecg_segments(directory, required_length=170):
    """
     - We assume each CSV file is a pre-processed segment of the ECG recording
     - The model relies on consistent segment length. Anomalies or variable lengths can degrade
        learned representation or require padding.
    - In production may need uniform signal length involving windowing or resampling the ECG fixed dim
    """
    ecg_segments = []
    if not os.path.isdir(directory):
        return ecg_segments
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            segment_path = os.path.join(directory, filename)
            segment_data = pd.read_csv(segment_path, header=None, skiprows=1).values.flatten()
            if len(segment_data) == required_length:
                ecg_segments.append(segment_data)
    return  ecg_segments

def augment_ecg(ecg_segment, noise_std=0.01):
    """
    - ECG signal can vary due to noise. By artificially injecting noise, we encourage
        the model to lean invariances, to improve intra-persons consistency
    - Too much augmentation may become a problem, by distorting critical cues for a subject discrimination. Tuning may be essential
    """
    noise = np.random.normal(0, noise_std, size=ecg_segment.shape)
    return  ecg_segment + noise


# --------------------------------------------------------------------
# 3. Sinusoidal Positional Encoding Layer
# --------------------------------------------------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    """
     - For 1D signals (ECG), positional encoding will help the self-attention
        mechanism distinguish temporal positions.
    - Though ECG has a pattern like a periodic pattern but not excatly the same.
        So, we rely on sinusoidal embeddings as a baseline.
    """
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        """
        - Broadcasting adds positional information to each element.
        - The self-attention layer can then factor in absolute/relative
            positions in subsequent computations.
        """
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]

# --------------------------------------------------------------------
# 4. Transformer Encoder Block
# --------------------------------------------------------------------
class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    - The Multi-head self-attention (MHA) allows each time step to attend to every other time step,
        capturing global ECG morphology relationships.
    - Residual connections facilitate gradient flow, critical in deeper networks, ensuring stable training.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
         # Multi-head attention: each head learns different
        # "attention patterns" across the sequence
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        # Position-wise feed-forward: transforms each time step's embedding
        # in a per-position manner
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        # Layer Norm ensures stable distribution of activations
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout for regularization
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    def call(self, x, training=False, mask=None):
        # 1) Self-Attention sub-layer
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 2) Feed-Forward sub-layer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return  out2

class TransformerEncoder(tf.keras.layers.Layer):
    """
    - Layer stacking deepens representational capacity.
    - Increasing num_layers can help capture more intricate ECG
        morphological features, but at the risk of overfitting if the data size is small
    """
    def __init__(self, num_layers, d_model, num_heads, dff, max_len=200, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(rate)

    def call(self, x, training=False, mask=None):
        # Positional encoding to preserve ordering info
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return  x #shape (batch_size, seq_len, d_model)

# --------------------------------------------------------------------
# 5. Straight-Through Binarization Layer
# --------------------------------------------------------------------
class BinaryStraightThrough(tf.keras.layers.Layer):
    """
    - Forward pass: threshold at 0.5 -> 0/1
    - Backward pass: pass gradients as if it were still continuous
    - The straight-through approach encourages network weights to stabilize
      around binary output states during training, reducing thresholding
      mismatch.
    """
    def call(self, inputs):
        # Forward pass -> discrete binary
        binary = tf.cast(inputs > 0.5, tf.float32)
        # Backward pass -> pass gradient as if it was 'inputs'
        return  inputs + tf.stop_gradient(binary - inputs)

# --------------------------------------------------------------------
# 6. Build the Transformer Model
# --------------------------------------------------------------------
def build_transformer_model(input_dim=170, d_model=128, num_heads=8, num_layers=4, dff=512, dropout_rate=0.1, output_dim=256):
    """
    -  Assembles the entire Transformer-based pipeline for ECG -> 256-bit key:
      1) Linear projection from (batch, 170, 1) -> (batch, 170, d_model)
      2) Stacked TransformerEncoder
      3) Global average pooling (aggregates across time steps)
      4) Dense layers + straight-through binarization
    - GlobalAveragePooling1D merges time dimension, akin to a final
      "learned" global context. Alternatives like attention pooling or
      max pooling could be studied.
    - The final 256 output bits correspond to cryptographic key length, a
      typical standard for security
    """

    inputs = tf.keras.INput(shape=(input_dim, 1), name="ecg_input")

    # Step 1: Map single-channel ECG to d_model dims
    x = Dense(d_model)(inputs)

    # Step 2: Transformer blocks
    transformer = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        max_len=input_dim,
        rate=dropout_rate
    )
    x = transformer(x)

    # Step 3: Aggregate over sequence dimension
    x = GlobalAveragePooling1D()(x)

    # Step 4: Additional FC layers for capacity
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Final 256-bit output
    key_output = Dense(output_dim, activation='sigmoid')(x)
    key_output = BinaryStraightThrough()(key_output)

    model = Model(inputs=inputs, outputs=key_output)
    return  model

# --------------------------------------------------------------------
# 7. Main Training + Evaluation
# --------------------------------------------------------------------
def train_and_test_prebuilt_keys(
        base_directory,
        keys_file,
        log_file_path="transformer.txt",
        max_length=170
):
    """
    End-to-end script:
    1) Load ground-truth keys from `keys_file`.
    2) Gather ECG segments from each Person_X, skipping ID=74 (missing).
       - 70/20/10 split per person
    3) Train a Transformer model with BCE loss (256 bits).
    4) Evaluate inter/intra-person Hamming distances, averaging segment
       predictions for final key.
    """
    print("Loading ground_truth keys...")
    prebuilt_keys = load_keys(keys_file)

    print("Gathering person data...")
    all_persons_data = []
    for person_dir in sorted(os.listdir(base_directory)):
        if not person_dir.startswith("Person_"):
            continue
        person_id = int(person_dir.split('_')[-1])
        if person_id == 74:
            # Skip missing folder scenario, as per user's data structure
            continue
        # Adjust ID if > 74 to match JSON index
        adjusted_person_id = person_id - 1 if person_id > 74 else person_id
        key_name = f"Person_{adjusted_person_id:02d}"
        if key_name not in prebuilt_keys:
            continue
        # Load form rec_2_filtered; adapt if multiple recordings
        person_path = os.path.join(base_directory, person_dir, 'rec_2_filtered')
        if not os.path.isdir(person_path):
            continue

        segments = load_valid_ecg_segments(person_path, required_length=max_length)
        if len(segments) < 3:
            continue
        ground_key = np.array(prebuilt_keys[key_name], dtype=np.float32)

        # Shuffle segments for random split
        segments = np.array(segments)
        np.random.shuffle(segments)
        total = len(segments)
        train_count = int(0.7 * total)
        val_count = int(0.2 * total)
        # test_count = total - train_count - val_count

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

    print(f"Collected data for {len(all_persons_data)} persons.")

    # Build train/val sets
    X_train, Y_train = [], []
    X_val, Y_val  = [], []
    for p in all_persons_data:
        for seg in p['train_segments']:
            # Optional augmentation:
            # seg = augment_ecg(seg, noise_std=0.01)
            X_train.append(seg)
            Y_train.append(p['ground_key'])
        for seg in p['val_segments']:
            X_val.append(seg)
            Y_val.append(p['ground_key'])
    # Reshape to (N, length, 1)
    X_train = np.array(X_train).reshape(-1, max_length, 1)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val).reshape(-1, max_length, 1)
    Y_val = np.array(Y_val)

    print("Building Transformer model...")
    model = build_transformer_model(
        input_dim=max_length,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dff=512,
        dropout_rate=0.1,
        output_dim=256
    )

    # Learning rate schedule + AdamW for stability
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        initial_learning_rate=initial_lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tfa.optimizer.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    model.compile(optimizer=optimizer, loss=BinaryCrossentropy())

    print("Training...")
    callback = EarlyStopping(monito='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train,
        Y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=[callback],
        verbose=1
    )

    # ------------------------------------------------------------
    # Final Evaluation: Intra/Inter-Person Distances + Post-Processing
    # ------------------------------------------------------------
    with open(log_file_path, "w") as log_file:
        log_file.write("=== Transformer + Binarization Results ===\n\n")

        person_keys = {}
        for p in all_persons_data:
            pid = p['person_id']
            ground_key = p['ground_key']

            # Combine all segments for final evaluation
            all_segments = np.concatenate([
                p['train_segments'],
                p['val_segments'],
                p['test_segments']
            ], axis=0)
            if len(all_segments) == 0:
                continue

            X_all = all_segments.reshape(-1, max_length, 1)
            preds = model.predict(X_all, verbose=0) # -> shape: (num_Segments, 256)

            # "raw_preds" are in [0,1] range due to straight-through approach
            # For final key, we average over all segments -> reduce intra-person noise
            avg_key = np.mean(preds, axis=0)
            final_bin_key = (avg_key > 0.5).astype(int)

            # Intra-Person Consistency:
            # measure pairwise Hamming among binarized segment outputs
            seg_bin_preds = (preds > 0.5).astype(int)
            num_segs = len(all_segments)
            intra_distances = []
            for i in range(num_segs):
                for j in range(i + 1, num_segs):
                    dist = hamming(seg_bin_preds[i], seg_bin_preds[j]) * len(seg_bin_preds[i])
                    intra_distances.append(dist)
            avg_intra = np.mean(intra_distances) if intra_distances else 0.0

            # Distance to Ground Truth for the final, aggregated key
            gt_dist = hamming(ground_key, final_bin_key) * len(ground_key)

            log_file.write(
                f"Person {pid}: "
                f"Avg Intra-Person Dist={avg_intra:.2f}, "
                f"Dist to GT (Final Key)={gt_dist:.2f}\n"
            )

            person_keys[pid] = final_bin_key

            # Inter-Person Distances (using the final aggregated key)
        log_file.write("\n--- Inter-Person Distances (Final Keys) ---\n")
        person_ids = sorted(person_keys.keys())
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                pid1 = person_ids[i]
                pid2 = person_ids[j]
                key1 = person_keys[pid1]
                key2 = person_keys[pid2]
                dist = hamming(key1, key2) * len(key1)
                log_file.write(f"Person {pid1} vs Person {pid2}: {dist:.2f}\n")

    print(f"Training and evaluation completed. Results in {log_file_path}")

# --------------------------------------------------------------------
# 8. Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = ""
    keys_file = ""

    train_and_test_prebuilt_keys(
        base_dir,
        keys_file,
        log_file_path="transformer_results.txt",
        max_length=170
    )



