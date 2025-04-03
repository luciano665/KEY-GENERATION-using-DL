import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dropout, Dense, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle






# ==============================================================================
# 1. Data Loader with rec_2_filtered Handling (unchanged)
# ==============================================================================
class ECGKeyLoader:
    """
       Class to load ECG segments and their corresponding keys.
       It traverses the data directory, loads each person's ECG segments, and
       validates the dataset to ensure that each person has a corresponding key.
    """
    def __init__(self, data_dir, key_path):
        self.data_dir = data_dir
         # Load ground truth keys
        self.key_map = self._load_keys(key_path)
        # Load person data including segments and keys
        self.persons = self._load_persons()
        # Validate dataset to ensure enough data is loaded
        self._validate_dataset()

    def _load_keys(self, key_path):
        """
        Load ground truth keys from JSON file
        """
        with open(key_path) as f:
            # Convertion of key values to numpy float32 arrays
            return {int(k.split('_')[-1]): np.array(v, dtype=np.float32)
                    for k, v in json.load(f).items()}

    def _load_persons(self):
        """
        Traverse the data directory and load all persons' ECG segments, only valid person segments'
        """
        persons = []
        valid_ids = set(self.key_map.keys())

        # Sort directories to ensure consistent ordering
        for dir_name in sorted(os.listdir(self.data_dir)):
            if not dir_name.startswith("Person_"):
                continue

            try:
                #Extract the person id form directory name
                person_id = int(dir_name.split('_')[-1].lstrip('0'))
                if person_id == 0:
                    person_id = int(dir_name.split('_')[-1])
            except ValueError:
                print(f"Skipping invalid directory {dir_name}")
                continue

            # Skip person if no key is found
            if person_id not in valid_ids:
                print(f"Key not found for {dir_name}, skipping")
                continue

            person_path = os.path.join(self.data_dir, dir_name)
            # Load ECG segments for the person
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

        return  persons

    def _validate_dataset(self):
        """
        Validates the dataset by ensuring that there is at least one person
        with valid ECG segments and that total segments across persons exceed min threshold
        """

        if not self.persons:
            raise ValueError("No valid persons with both keys and ECG segments")

        total_segments = sum(len(p['segments']) for p in self.persons)
        print(f"Dataset contains {len(self.persons)} persons with {total_segments} total segments.")

        if total_segments < 10:
            raise ValueError("Insufficient data for training (min 10 segments required)")


    def _load_segments(self, person_path, seq_len=170):
        """
        Load ECG segments from a person directory. It recursively scans subdirectories
        and loads CSV files that represent ECG segments. Each CSV must contain 170 data points.
        """
        segments = []

        # Walk through all subdirectories in person's folder
        for root, dirs, files in os.walk(person_path):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                file_path = os.path.join(root, file)
                try:
                    # Skip header row and load ECG segments
                    ecg = np.loadtxt(file_path, delimiter=',', skiprows=1, ndmin=1)
                    # Ensure the ECG segment has 170 data points as expected
                    if ecg.ndim != 1 or len(ecg) != seq_len:
                        print(f"Invalid ECG format in {file_path}")
                        continue
                    segments.append(ecg.astype(np.float32))
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        return np.array(segments)

    def get_train_data(self, test_size=0.2):
        """
        Aggregate ECG segments and their corresponding ground truth keys from all persons
        Reshape the ECG segment to the format (batch, seq_len, 1) and perform a train-test split
        The split is stratified by person ID
        """
        X, y, ids = [], [], []
        for p in self.persons:
            X.extend(p['segments'])
            y.extend([p['key']] * len(p['segments']))
            ids.extend([p['id']] * len(p['segments']))

        # Reshape to add the channel dimension
        X = np.array(X).reshape((-1, 170, 1))
        y = np.array(y)

        if len(X) < 1:
            raise ValueError(f"Need at least 2 samples, got {len(X)}.")

        # Perform stratified train-test split
        return train_test_split(X, y, test_size=test_size, stratify=ids)


# ==============================================================================
# 2. WaveNet-Based Model for Key Generation
# ==============================================================================
class WaveNetResidualBlock(tf.keras.layers.Layer):
    """
    Residual block for WaveNet.
    It consists of two dilated conv layers (filter and gate), followed by a gating mechanism,
    dropout, and residual connections.
    """
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.1):
        super(WaveNetResidualBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        """
        # Conv layer for filter part
        self.conv_filter = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')
        # Conv for gate part
        self.conv_gate = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal')
        # Dropout layer for regularization
        self.dropout = Dropout(dropout_rate)
        # 1x1 Conv layer to produce the residual output
        self.conv_residual = Conv1D(filters, 1, padding='same')
        # 1x1 Conv layer to produce the skip connection output
        self.conv_skip = Conv1D(filters, 1, padding='same')
        """

    def build(self, input_shape):
        # Create layers with known input shape during build()
        # Conv layer for filter part
        self.conv_filter = Conv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate, padding='causal')
        # Conv for gate part
        self.conv_gate = Conv1D(self.filters, self.kernel_size, dilation_rate=self.dilation_rate, padding='causal')
        # Dropout layer for regularization
        self.dropout = Dropout(self.dropout_rate)
        # 1x1 Conv layer to produce the residual output
        self.conv_residual = Conv1D(self.filters, 1, padding='same')
        # 1x1 Conv layer to produce the skip connection output
        self.conv_skip = Conv1D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs, training=False):
        """
        Apply the dilated convolutions, gating mechanism, dropout,
        and compute both the residual and skip outputs
        """
        # Apply filter and gate convolutions
        x_filter = self.conv_filter(inputs)
        x_gate = self.conv_gate(inputs)
        # Apply activation functions : tanh for filter and sigmoid for gate
        x_filter =  tf.tanh(x_filter)
        x_gate = tf.sigmoid(x_gate)
        # Multiply filter and gate outputs (gating mechanism)
        x  = x_filter * x_gate
        # Apply dropout for regularization
        x = self.dropout(x, training=training)
        # Compute the skip connection using 1x1 convolution
        residual = self.conv_residual(x)
        # Compute the skip connection using 1x1 convolution
        skip = self.conv_skip(x)
        # Add residual to the original inputs for the residual connection
        output = inputs + residual
        return  output, skip

class WaveNetKeyGenerator(tf.keras.Model):
    """
    Wavenet-based model for ECG key generation
    Model starts with initial convolution, followed by multiple Wavenet residual blocks.
    Skip connections are aggregated, and then globally pooled before a final dense projection.
    """
    def __init__(self, seq_len=170, num_filters=64, num_wavenet_blocks=4,
                 kernel_size=3, key_bits=256, dropout_rate=0.1):
        super(WaveNetKeyGenerator, self).__init__()
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.num_wavenet_blocks = num_wavenet_blocks
        self.kernel_size = kernel_size
        self.key_bits = key_bits
        self.dropout_rate = dropout_rate

    def build(self, input_shape):

        # Initial 1x1 convolution to adjust input dimensions
        self.initial_conv = Conv1D(self.num_filters, kernel_size=1, padding='same')
        # Create list to hold all Wavenet residual blocks with increasing dilation rates
        self.wavenet_blocks = []
        for i in range(self.num_wavenet_blocks):
            dilation_rate = 2 ** i # Increase dilation rate exponentially
            block = WaveNetResidualBlock(self.num_filters, self.kernel_size, dilation_rate, self.dropout_rate)
            self.wavenet_blocks.append(block)

        # Activation layer
        self.relu = ReLU()
        # Post-processing convolution after aggregating skip connections
        self.conv_post = Conv1D(self.num_filters, kernel_size=1, padding='same')
        # Global average pooling to reduce temporal dimensions
        self.gap = GlobalAveragePooling1D()
        # Final dense layer to project to the ground truth key bits with sigmoid activation
        self.key_proj = Dense(self.key_bits, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs, training=False):
        """
        Forward pass of WaveNet:
        1. Process the input with initial convolution
        2. Pass the result through each WaveNet residual block while collecting skip connections.
        3. Sum skip connections, apply activation and post-processing convolution
        4. Perform global average pooling and produce final key
        """
        # Initial Convolution to adjust the inputs channels
        x = self.initial_conv(inputs)
        skip_connections = []
        # Process through each Wavenet residual Block
        for block in self.wavenet_blocks:
            x, skip = block(x, training=training)
            skip_connections.append(skip)

        # Sum all skip connections
        x = tf.add_n(skip_connections)
        # Apply activation and post-processing convolution
        x = self.relu(x)
        x = self.conv_post(x)
        x = self.relu(x)
        # Global average pooling to create a fixed-sized representation
        x = self.gap(x)
        # Final projection to generate key
        return  self.key_proj(x)

# ==============================================================================
# 3. Training and Key Generation System
# ==============================================================================
class KeyGenerationSystem:
    """
    System to handle training and key generation
    It uses ECGKeyLoader for data loading and WaveNetKeyGenerator.
    """
    def __init__(self, data_dir, key_path):
        # Init data loader
        self.loader = ECGKeyLoader(data_dir, key_path)
        self.model = None

    def train(self, epochs=100, batch_size=32):
        """
        Train the model phase
        - Loads training and validation data
        - Initializes the model with appropriate parameters
        - Compiles and fits model using early stopping
        """

        # Retrieve train-test split from loader
        X_train, X_val, y_train, y_val = self.loader.get_train_data()
        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("y_train shape:", y_train.shape)
        print("y_val shape:", y_val.shape)

        # Initializes the model
        # Assuming each key is a binary vector (length 256 bits)
        self.model = WaveNetKeyGenerator(
            seq_len=170,
            num_filters=64,
            num_wavenet_blocks=3,
            kernel_size=3,
            key_bits=y_train.shape[1] if len(y_train.shape) > 1 else 256,
            dropout_rate=0.1
        )

        # Build the model with a known input shape to avoid unknown TensorShape issues
        self.model.build((None, 170, 1))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        # Train the model with early stopping to prevent overfitting
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)]
        )

        return  history

    def generate_key(self, ecg_segments, threshold=0.5):
        """
        Generation of final key for a set of ECG segments
        Model predicts keys for each segment and the averages predictions
        The final key is determined by thresholding the average probability
        """
        if ecg_segments is None or len(ecg_segments) == 0:
            raise ValueError("No ECG segments provided for key generation")
        ecg_segments = np.array(ecg_segments, dtype=np.float32)
        # Ensure the segments have a shape (batch, seq_len, channels)
        if ecg_segments.ndim == 2:
            ecg_segments =  ecg_segments.reshape(ecg_segments.shape[0], ecg_segments.shape[1], 1)

        # Get prediction from model
        predictions = self.model.predict(ecg_segments)
        # Average the prediction across all segments
        ave_prob = np.mean(predictions, axis=0)
        # Return final binary key based on threshold
        return  (ave_prob > threshold).astype(np.int32)

# ==============================================================================
# 4. Main Execution with Error Handling
# ==============================================================================
if __name__ == "__main__":
    DATA_DIR = ""
    KEY_FILE = ""

    try:
        # Init the key generation system with data directories and key file
        print("Initializing system...")
        kgs = KeyGenerationSystem(DATA_DIR, KEY_FILE)

        # Start training phase
        print("\nStarting training...")
        kgs.train(epochs=100)

        # Force the model to build with expected input shape by performing a dummy forward pass
        _ = kgs.model(tf.zeros((1, 170, 1)))

        # ----------------------------
        # Test key generation for all persons
        # ----------------------------
        print("\nTesting key generation for all persons:")

        # Dictionary to store aggregated keys for each person for later inter-person H-distance comparison
        # List of all keys generated for intra hamming distance
        aggregated_keys = {}
        all_intra_distances = []

        # Loop through each person loaded
        for person in kgs.loader.persons:
            segments = person['segments']

            # Ensure segments have shape correct (batch, seq_len, 1)
            if segments.ndim == 2:
                segments = segments.reshape(segments.shape[0], segments.shape[1], 1)

            #Generate an aggregated key for the person from all its segments
            aggregated_key = kgs.generate_key(segments)
            ground_truth = person['key'].astype(np.int32)
            accuracy = np.mean(aggregated_key ==  ground_truth)

            print(f"\nPerson {person['id']}:")
            print(f"  Aggregated key accuracy: {accuracy:.2%}")
            print(f"  Aggregated key: {aggregated_key[:24]}...")
            print(f"  Ground truth: {ground_truth[:24]}...")

            aggregated_keys[person['id']] = aggregated_key

            # ----------------------------
            # Compute Intra-Person Hamming Distance
            # ----------------------------
            predictions = kgs.model.predict(segments)
            individual_keys = (predictions > 0.5).astype(np.int32)
            num_keys = individual_keys.shape[0]

            # Compute pairwise Hamming distances if there is more than one segment
            if num_keys > 1:
                distances = []
                for i in range(num_keys):
                    for j in range(i+1, num_keys):
                        d = np.sum(individual_keys[i] != individual_keys[j])
                        distances.append(d)
                avg_distance = np.mean(distances)
                print(f"  Intra-person average Hamming distance: {avg_distance:.2f} bits")
                all_intra_distances.extend(distances)
            else:
                print("   Not enough segments to compute intra-person Hamming distance.")

            # ----------------------------
            # Overall intra-person Hamming distance statistics
            # ----------------------------
            person_ids = sorted(aggregated_keys.keys())
            person_inter_dists = {p: [] for p in person_ids}
            print("\nInter-person hamming distances (aggregated keys):")
            for i in range(len(person_ids)):
                for j in range(i+1, len(person_ids)):
                    key1 = aggregated_keys[person_ids[i]]
                    key2 = aggregated_keys[person_ids[j]]
                    # Calculate Hamming distance between 2 aggregated keys
                    d = int(np.sum(key1 != key2))
                    # Save the distance for persons
                    person_inter_dists[person_ids[i]].append(d)
                    person_inter_dists[person_ids[j]].append(d)
                    print(f"  Distance between Person {person_ids[i]} and Person {person_ids[j]}: {d} bits.")


            # Compute overall inter-person Hamming distance statistics
            all_inter_distances = []
            for dist_list in person_inter_dists.values():
                all_inter_distances.extend(dist_list)
            if all_inter_distances:
                overall_inter_mean = np.mean(all_inter_distances)
                overall_inter_std = np.std(all_inter_distances)
                print("\nOverall Inter-person Hamming distance: "
                      f"mean = {overall_inter_mean:.2f} bits, std = {overall_inter_std:.2f} bits")
            else:
                print("\nNo data available to compute inter-person hamming distance statistics.")

            # Save the raw distances data for plotting:
            with open("all_intra_distances_wavenet.pkl", "wb") as f:
                pickle.dump(all_intra_distances, f)
            with open("all_inter_distances_wavenet.pkl", "wb") as f:
                pickle.dump(person_inter_dists, f)


    except Exception as e:
        # Print error message and verification checklist if something goes wrong.
        print(f"\nError: {str(e)}")
        print("Verification Checklist:")
        print("1. Directory structure: Person_XX/rec_N_filtered/*.csv")
        print("2. CSV files contain exactly 170 values, no headers")
        print("3. JSON keys match Person_XX numbering (1-89)")
        print("4. Minimum 10 segments across all persons")
        print("Something went wrong check again.")


