import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load intra-person distances (flat list)
with open("/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/VIT/all_intra_distances.pkl", "rb") as f:
    intra_hd = pickle.load(f)

# Compute statistics for intra-person distances
intra_mean = np.mean(intra_hd)
intra_std = np.std(intra_hd)
print("Intra-person Hamming distances: mean = {:.2f} bits, std = {:.2f} bits".format(intra_mean, intra_std))

# Plot Intra-person Boxplot
plt.figure(figsize=(8, 6))
bp_intra = plt.boxplot(intra_hd, patch_artist=True, showfliers=True)
for patch in bp_intra['boxes']:
    patch.set_facecolor('skyblue')
plt.xlabel("Intra-person", fontsize=16)
plt.ylabel("Hamming Distance (bits)", fontsize=20)
plt.title("Intra-person Hamming Distances", fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 1-----------------------------
# Load inter-person distances (grouped by person)
with open("/Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/VIT/person_inter_dists.pkl", "rb") as f:
    person_dists = pickle.load(f)

# Convert dictionary to a list ordered by person number:
inter_data = [person_dists[p] for p in sorted(person_dists.keys())]

# Compute overall inter-person statistics
all_inter_distances = []
for dist_list in inter_data:
    all_inter_distances.extend(dist_list)
overall_inter_mean = np.mean(all_inter_distances)
overall_inter_std = np.std(all_inter_distances)
print("Inter-person Hamming distances: mean = {:.2f} bits, std = {:.2f} bits".format(overall_inter_mean, overall_inter_std))

# 2-----------------------------
# Divide persons into two groups:
group1 = inter_data[:44]   # Persons 1 to 44 (44 persons)
group2 = inter_data[44:]   # Persons 45 to 89 (45 persons)

# 2.1-----------------------------
# Plot Inter-person Boxplot for Group 1 (Persons 1-44)
plt.figure(figsize=(12, 6))
bp1 = plt.boxplot(group1, patch_artist=True, showfliers=True)
colors1 = plt.cm.hsv(np.linspace(0, 1, len(group1)))
for patch, color in zip(bp1['boxes'], colors1):
    patch.set_facecolor(color)
plt.xlabel("Person", fontsize=16)
plt.ylabel("Hamming Distance (bits)", fontsize=20)
plt.title("Inter-person Hamming Distances for Persons 1–44", fontsize=20)
plt.xticks(range(1, 45), [str(p) for p in range(1, 45)], rotation=90, fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------------------------------
# Plot Inter-person Boxplot for Group 2 (Persons 45-89)
plt.figure(figsize=(12, 6))
bp2 = plt.boxplot(group2, patch_artist=True, showfliers=True)
colors2 = plt.cm.hsv(np.linspace(0, 1, len(group2)))
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)
plt.xlabel("Person", fontsize=16)
plt.ylabel("Hamming Distance (bits)", fontsize=20)
plt.title("Inter-person Hamming Distances for Persons 45–89", fontsize=20)
plt.xticks(range(1, len(group2)+1), [str(p) for p in range(45, 90)], rotation=90, fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
