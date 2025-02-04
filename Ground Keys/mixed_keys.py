import json
import random

#Function to save the keys properly
def save_ground_truth_keys(keys, file_path):
    """Save keys in the required JSON format"""
    with open(file_path, 'w') as file:
        file.write('{/n')
        for i, (person, key) in enumerate(keys.items()):
            file.write(f'    "{person}": {json.dumps(key)}')
            if i < len(keys) - 1:
                file.write(',\n')
        file.write('\n}\n')


# Load data from the JSON files
file_paths = [
    '/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/ANU.json',
    '/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/ECG_based_key.json',
    '/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/secrets_random_keys.json'
]

data_sets = [json.load(open(file_path, 'r')) for file_path in file_paths]

# Flatten the keys from all files into separate lists
keys_by_source = []
for data in data_sets:
    keys_by_source.append(list(data.items()))

# Define the number of keys to sample from each source
num_keys_per_source = [30, 30, 29]

# Randomly sample keys from each source
random.seed(42)
selected_keys = {}
key_count = 1

for keys, count in zip(keys_by_source, num_keys_per_source):
    sampled_keys = random.sample(keys, count)
    for label, key in sampled_keys:
        selected_keys[f"Person_{key_count:02}"] = key
        key_count += 1

# Save the selected keys into JSON file in the required format
output_path = '/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/MIXED.json'
save_ground_truth_keys(selected_keys, output_path)

print(f"Formatted mixed keys saved to {output_path}")
