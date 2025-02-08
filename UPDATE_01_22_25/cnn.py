import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Dense
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
        # Mild augmentation for robustness.
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
# 3. Model Architecture: CNN-based Multi-Task Network
# =============================================================================
def build_cnn_model(num_classes, key_units=256):
    # Input: ECG segment of length 170 with 1 channel.
    inputs = Input(shape=(170, 1))
    x = Conv1D(filters=32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)  # -> (85, 32)
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)  # -> (42, 64)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)  # -> (21, 128)
    x = GlobalAveragePooling1D()(x)  # -> (128,)
    embedding = Dense(128, activation='relu')(x)
    key_out = Dense(key_units, activation='sigmoid', name='key_output')(embedding)
    # We expect the classification target to be one-hot encoded.
    class_out = Dense(num_classes, activation='softmax', name='class_output')(embedding)
    model = Model(inputs=inputs, outputs={'key_output': key_out, 'class_output': class_out})
    return model


# =============================================================================
# 4. Loss Function: Multi-Task Loss
# =============================================================================
def get_losses(num_classes, alpha=1.0, beta=1.0, gamma=0.5):
    # Standard losses.
    bce = BinaryCrossentropy()
    cce = CategoricalCrossentropy()

    def loss_fn(y_true, y_pred):
        # Here, y_true might come as a dictionary or as a tuple.
        if isinstance(y_true, dict):
            key_target = y_true['key_output']
            class_target = y_true['class_output']
        else:
            key_target, class_target = y_true
        # The classification target should be one-hot encoded. If not, do it here.
        if len(class_target.shape) == 1 or class_target.shape[-1] != num_classes:
            class_target = tf.one_hot(tf.cast(class_target, tf.int32), depth=num_classes)
        key_pred = y_pred['key_output'] if isinstance(y_pred, dict) else y_pred[0]
        class_pred = y_pred['class_output'] if isinstance(y_pred, dict) else y_pred[1]
        loss_key = bce(key_target, key_pred)
        loss_class = cce(class_target, class_pred)
        loss_balance = tf.reduce_mean(tf.square(tf.reduce_mean(key_pred, axis=0) - 0.5))
        return alpha * loss_key + beta * loss_class + gamma * loss_balance

    return loss_fn


# =============================================================================
# 5. Training Pipeline
# =============================================================================
class KeyTrainingSystem:
    def __init__(self, data_loader):
        self.data = data_loader.person_data
        self.num_persons = len(data_loader.key_map)
        self.model = build_cnn_model(num_classes=self.num_persons, key_units=256)
        self.optimizer = AdamW(learning_rate=3e-4, weight_decay=1e-4)
        self.model.compile(optimizer=self.optimizer, loss=get_losses(self.num_persons))

    def _process_key(self, key, desired_length=256):
        key = np.atleast_1d(np.array(key, dtype=np.float32))
        if key.shape[0] < desired_length:
            key = np.pad(key, (0, desired_length - key.shape[0]), mode='constant')
        elif key.shape[0] > desired_length:
            key = key[:desired_length]
        return key

    def _data_generator(self, data, batch_size=32):
        """
        Each sample is created by randomly selecting one ECG segment from a random subject.
        Yields:
          inputs: an ECG segment (shape: (170, 1))
          targets: a dictionary with keys:
              'key_output': ground_truth_key (shape: (256,))
              'class_output': one-hot encoded subject label (shape: (num_classes,))
        """
        while True:
            inputs_list, key_targets, class_targets = [], [], []
            for _ in range(batch_size):
                person = np.random.choice(data, size=1)[0]
                segs = person['segments']
                idx = np.random.choice(len(segs))
                seg = segs[idx]
                seg = ECGAugmenter.augment(seg)
                inputs_list.append(seg.reshape(170, 1))
                key_targets.append(self._process_key(person['key'], desired_length=256))
                # One-hot encode subject label.
                one_hot = tf.one_hot(person['id'], depth=self.num_persons).numpy()
                class_targets.append(one_hot)
            inputs_array = np.array(inputs_list, dtype=np.float32)
            key_targets_array = np.array(key_targets, dtype=np.float32)
            class_targets_array = np.array(class_targets, dtype=np.float32)
            targets = {'key_output': key_targets_array, 'class_output': class_targets_array}
            yield inputs_array, targets

    def _create_dataset(self, data, batch_size=32):
        return tf.data.Dataset.from_generator(
            lambda: self._data_generator(data, batch_size),
            output_signature=(
                tf.TensorSpec(shape=(None, 170, 1), dtype=tf.float32),
                {
                    'key_output': tf.TensorSpec(shape=(None, 256), dtype=tf.float32),
                    'class_output': tf.TensorSpec(shape=(None, self.num_persons), dtype=tf.float32)
                }
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

    def _process_key(self, key, desired_length=256):
        key = np.atleast_1d(np.array(key, dtype=np.float32))
        if key.shape[0] < desired_length:
            key = np.pad(key, (0, desired_length - key.shape[0]), mode='constant')
        elif key.shape[0] > desired_length:
            key = key[:desired_length]
        return key

    def evaluate(self):
        self._calculate_keys()
        self._cryptographic_analysis()
        self._save_results()
        return self.results

    def _calculate_keys(self):
        # For each subject, run all segments through the model and average the predicted keys.
        for person in self.data:
            segs = person['segments']
            inputs = np.array(segs, dtype=np.float32).reshape(-1, 170, 1)
            preds = self.model.predict(inputs, verbose=0)['key_output']  # shape (num_segments, 256)
            avg_pred = np.mean(preds, axis=0)
            final_key = (avg_pred > 0.5).astype(int)
            binary_preds = (preds > 0.5).astype(int)
            intra_dists = [hamming(final_key, k) * 256 for k in binary_preds]
            gt_key = self._process_key(person['key'], desired_length=256)
            gt_dist = hamming(final_key, gt_key) * 256
            self.results[person['id']] = {
                'final_key': final_key,
                'intra_mean': np.mean(intra_dists),
                'intra_std': np.std(intra_dists),
                'gt_dist': gt_dist
            }

    def _cryptographic_analysis(self):
        all_keys = [v['final_key'] for k, v in self.results.items() if isinstance(k, int)]
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
                f.write(f"  Final Generated Key:       {self._truncate_key(res['final_key'])}\n\n")
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
    LOG_FILE = "results_cnn_multitask.txt"

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
