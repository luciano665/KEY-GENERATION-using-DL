import json
from itertools import combinations
import pandas as pd

#Load teh JSON data from the uploaded file
file_path = '/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/Ground Keys/MIXED.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to calculate Hamming distance and percentage difference
def hamming_distance(key1, key2):
    if len(key1) != len(key2):
        raise ValueError('Keys must be of the same length.')
    distance = sum(k1 != k2 for k1, k2 in zip(key1, key2))
    percentage_difference = (distance / len(key1)) * 100
    return distance, percentage_difference

# Prepare comparisons and calculate results
results = []
for (person1, key1), (person2, key2) in combinations(data.items(), 2):
    distance, percentage = hamming_distance(key1, key2)
    results.append({
        "Person 1": person1,
        "Person 2": person2,
        "Hamming Distance": distance,
        "Percentage Difference": percentage
    })

# Save results to a Dataframe
df_results = pd.DataFrame(results)

#Save results to a CSV file
output_path = '/Ground-Key-Hamming-Distances/h_distance_mixed.csv'
df_results.to_csv(output_path, index=False)

output_path