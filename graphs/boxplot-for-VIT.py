import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paste the full text into a multi-line string:
text = r"""


Person 1:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1]...
  Ground Truth:   [1 1 0 1 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 24.97 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 148ms/step

Person 2:
  Aggregated Key Accuracy: 85.16%
  Aggregated Key: [1 1 0 0 1 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 56.09 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 3:
  Aggregated Key Accuracy: 85.16%
  Aggregated Key: [1 0 1 1 1 1 1 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 0 1 1 0 1 0 0 1 1 0 1 0 1 1 0 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 28.43 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 4:
  Aggregated Key Accuracy: 76.17%
  Aggregated Key: [1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 0]...
  Ground Truth:   [1 1 1 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 38.89 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 5:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0]...
  Ground Truth:   [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 22.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 6:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 0 1 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 44.86 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 7:
  Aggregated Key Accuracy: 83.98%
  Aggregated Key: [0 0 1 0 0 1 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 1]...
  Ground Truth:   [0 1 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 56.90 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 8:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 13.26 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 9:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 1 1 1 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 1 0 0 1 1 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 30.07 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 10:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 24.83 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 11:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [1 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
  Ground Truth:   [0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 39.89 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 12:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
  Ground Truth:   [0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 8.94 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 13:
  Aggregated Key Accuracy: 79.69%
  Aggregated Key: [0 1 0 1 1 0 1 0 1 1 0 0 0 1 0 1 1 1 0 1 0 1 0 0]...
  Ground Truth:   [0 1 0 1 1 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 42.53 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 14:
  Aggregated Key Accuracy: 78.12%
  Aggregated Key: [1 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
  Ground Truth:   [1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 65.86 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 15:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [1 1 0 1 0 0 0 1 1 1 0 1 0 1 0 1 0 1 1 1 0 0 0 0]...
  Ground Truth:   [1 1 0 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 23.57 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 16:
  Aggregated Key Accuracy: 75.78%
  Aggregated Key: [1 1 0 1 0 1 0 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1]...
  Ground Truth:   [0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 71.42 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 17:
  Aggregated Key Accuracy: 91.80%
  Aggregated Key: [1 0 0 0 1 0 1 0 1 1 1 0 1 1 0 1 0 1 1 0 0 0 1 1]...
  Ground Truth:   [1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 24.72 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 18:
  Aggregated Key Accuracy: 87.89%
  Aggregated Key: [1 0 0 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 1 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 43.87 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 19:
  Aggregated Key Accuracy: 85.94%
  Aggregated Key: [1 0 1 0 0 1 1 1 1 1 1 1 0 1 0 1 0 1 0 0 0 1 0 0]...
  Ground Truth:   [1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 0 0 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 55.45 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 20:
  Aggregated Key Accuracy: 99.22%
  Aggregated Key: [1 0 1 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1]...
  Ground Truth:   [1 0 1 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 26.44 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 21:
  Aggregated Key Accuracy: 81.64%
  Aggregated Key: [1 0 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 0 1 1 0 1 1 0]...
  Ground Truth:   [1 0 0 0 0 1 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 42.59 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 22:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
  Ground Truth:   [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 17.25 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 23:
  Aggregated Key Accuracy: 84.38%
  Aggregated Key: [1 0 0 0 0 1 0 1 0 1 0 1 1 0 1 1 1 0 1 1 1 0 0 1]...
  Ground Truth:   [1 0 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 19.77 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 24:
  Aggregated Key Accuracy: 77.73%
  Aggregated Key: [0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0]...
  Ground Truth:   [0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 62.37 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 25:
  Aggregated Key Accuracy: 78.12%
  Aggregated Key: [1 0 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 68.47 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 26:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 0 0 0]...
  Ground Truth:   [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 19.25 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 27:
  Aggregated Key Accuracy: 89.84%
  Aggregated Key: [1 0 0 0 1 1 1 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 32.59 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 28:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 1 0 1 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 38.90 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 29:
  Aggregated Key Accuracy: 95.70%
  Aggregated Key: [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 21.20 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 30:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0]...
  Ground Truth:   [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 27.83 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 31:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [0 1 1 0 0 0 0 1 1 1 0 0 1 1 1 1 0 1 0 0 1 0 0 1]...
  Ground Truth:   [0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 13.57 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 32:
  Aggregated Key Accuracy: 85.94%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0]...
  Ground Truth:   [1 1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 45.72 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 33:
  Aggregated Key Accuracy: 79.69%
  Aggregated Key: [1 0 1 0 1 1 1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 1 0 0]...
  Ground Truth:   [1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 40.60 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 34:
  Aggregated Key Accuracy: 88.28%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 0 1 1 0 0 1 0 1 0 1 1 1 0 0]...
  Ground Truth:   [1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 1 0 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 27.79 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 35:
  Aggregated Key Accuracy: 96.48%
  Aggregated Key: [1 0 0 0 1 0 1 1 1 0 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
  Ground Truth:   [1 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 14.74 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 36:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 1 0 1 1 0 0 0 1 0 1 0 0 0 1 0 1 1 1 1 0 0 1 1]...
  Ground Truth:   [0 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 22.29 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 37:
  Aggregated Key Accuracy: 82.42%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 48.41 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 38:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 47.39 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 39:
  Aggregated Key Accuracy: 69.53%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 1 1 0 0 0 1 0 1 0 0 0 1 1 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 64.91 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 40:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [1 0 0 0 1 1 0 1 1 1 1 1 1 0 0 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 51.39 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 41:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0]...
  Ground Truth:   [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 9.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 42:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [0 1 0 0 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1 0 0 1 0 1]...
  Ground Truth:   [0 1 1 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 50.09 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 43:
  Aggregated Key Accuracy: 86.33%
  Aggregated Key: [1 1 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 50.66 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 44:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 0 1 0 0 1 1 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 45.97 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 45:
  Aggregated Key Accuracy: 86.72%
  Aggregated Key: [0 1 0 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 0 0 1 0 1 0]...
  Ground Truth:   [0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 1 1 0 1 0 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 54.21 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 46:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 1 1 1]...
  Ground Truth:   [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 19.82 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 47:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [0 0 1 0 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 1 0]...
  Ground Truth:   [0 0 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 58.99 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 48:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [1 0 1 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
  Ground Truth:   [1 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 7.03 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 49:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
  Ground Truth:   [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
  Intra-person average Hamming distance: 17.78 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 50:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 11.52 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 51:
  Aggregated Key Accuracy: 76.95%
  Aggregated Key: [0 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 0 1 1]...
  Ground Truth:   [0 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 24.60 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 52:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 0 1 1 1 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 19.54 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 53:
  Aggregated Key Accuracy: 94.14%
  Aggregated Key: [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
  Ground Truth:   [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 20.45 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 54:
  Aggregated Key Accuracy: 86.33%
  Aggregated Key: [0 0 0 1 1 0 0 1 1 0 0 0 1 0 1 1 0 0 0 1 1 1 0 0]...
  Ground Truth:   [0 0 0 0 1 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 29.85 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 55:
  Aggregated Key Accuracy: 74.22%
  Aggregated Key: [0 1 0 1 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 1 1 1 0 0]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 1 1 1 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 64.76 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 56:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [0 0 0 0 1 1 0 0 1 0 0 1 0 1 1 1 1 0 0 0 0 0 1 0]...
  Ground Truth:   [0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 50.48 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 57:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
  Ground Truth:   [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 10.23 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step

Person 58:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 0 1 1 0 1 0 1 1 0 1 1 1 1 0 1 1 0 1 0 1 0 0 1]...
  Ground Truth:   [0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 20.53 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 59:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
  Intra-person average Hamming distance: 15.99 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 60:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [0 0 0 1 1 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 0 1 1 1]...
  Ground Truth:   [0 0 0 1 1 0 0 1 0 0 0 1 0 0 1 0 1 1 0 1 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 26.32 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 61:
  Aggregated Key Accuracy: 83.98%
  Aggregated Key: [0 0 1 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 0 1 1 1 1]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 47.05 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 62:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0]...
  Ground Truth:   [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 30.22 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 63:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 18.25 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 64:
  Aggregated Key Accuracy: 86.72%
  Aggregated Key: [0 1 0 1 1 0 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 28.85 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 65:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
  Ground Truth:   [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 19.88 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step

Person 66:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 41.73 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 67:
  Aggregated Key Accuracy: 80.47%
  Aggregated Key: [1 0 1 0 1 1 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 1 0 1]...
  Ground Truth:   [0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 21.60 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Person 68:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 0 1 1]...
  Ground Truth:   [0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
  Intra-person average Hamming distance: 20.27 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step

Person 69:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [0 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 27.40 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 70:
  Aggregated Key Accuracy: 73.05%
  Aggregated Key: [0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 1 1 1 1 1 0 1 1 0]...
  Ground Truth:   [0 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 44.40 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step

Person 71:
  Aggregated Key Accuracy: 89.84%
  Aggregated Key: [1 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
  Ground Truth:   [1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 28.91 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 72:
  Aggregated Key Accuracy: 90.23%
  Aggregated Key: [0 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
  Ground Truth:   [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 15.27 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Person 73:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [1 0 1 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1 0 0 0 1 0 0]...
  Ground Truth:   [1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 44.79 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

Person 74:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
  Intra-person average Hamming distance: 0.33 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 75:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 1 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 0 0 1 0]...
  Ground Truth:   [0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
  Intra-person average Hamming distance: 55.93 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 

Person 76:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
  Ground Truth:   [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 
  Intra-person average Hamming distance: 1.67 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 77:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 14.35 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 78:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [1 1 0 1 1 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 24.50 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 

Person 79:
  Aggregated Key Accuracy: 96.48%
  Aggregated Key: [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 0]...
  Ground Truth:   [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
  Intra-person average Hamming distance: 16.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 80:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 0 0 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 1 1 0 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 30.49 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 81:
  Aggregated Key Accuracy: 87.50%
  Aggregated Key: [1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 0 1]...
  Ground Truth:   [1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
  Intra-person average Hamming distance: 35.49 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

Person 82:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
  Intra-person average Hamming distance: 9.48 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step

Person 83:
  Aggregated Key Accuracy: 97.27%
  Aggregated Key: [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
  Intra-person average Hamming distance: 10.16 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 84:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 44.09 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 

Person 85:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
  Intra-person average Hamming distance: 12.05 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 86:
  Aggregated Key Accuracy: 77.73%
  Aggregated Key: [0 0 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1]...
  Ground Truth:   [0 0 0 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
  Intra-person average Hamming distance: 59.82 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 87:
  Aggregated Key Accuracy: 87.11%
  Aggregated Key: [0 1 0 1 0 0 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 0 0 1]...
  Ground Truth:   [0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 21.70 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 88:
  Aggregated Key Accuracy: 88.28%
  Aggregated Key: [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1]...
  Ground Truth:   [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 32.59 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 89:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 0 1 0 0 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 1]...
  Ground Truth:   [1 0 1 1 0 0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 44.06 bits

Inter-person Hamming distances (aggregated keys):
  Distance between Person 1 and Person 2: 120 bits
  Distance between Person 1 and Person 3: 126 bits
  Distance between Person 1 and Person 4: 105 bits
  Distance between Person 1 and Person 5: 135 bits
  Distance between Person 1 and Person 6: 135 bits
  Distance between Person 1 and Person 7: 123 bits
  Distance between Person 1 and Person 8: 139 bits
  Distance between Person 1 and Person 9: 134 bits
  Distance between Person 1 and Person 10: 136 bits
  Distance between Person 1 and Person 11: 135 bits
  Distance between Person 1 and Person 12: 120 bits
  Distance between Person 1 and Person 13: 119 bits
  Distance between Person 1 and Person 14: 113 bits
  Distance between Person 1 and Person 15: 153 bits
  Distance between Person 1 and Person 16: 86 bits
  Distance between Person 1 and Person 17: 127 bits
  Distance between Person 1 and Person 18: 142 bits
  Distance between Person 1 and Person 19: 124 bits
  Distance between Person 1 and Person 20: 126 bits
  Distance between Person 1 and Person 21: 131 bits
  Distance between Person 1 and Person 22: 145 bits
  Distance between Person 1 and Person 23: 145 bits
  Distance between Person 1 and Person 24: 123 bits
  Distance between Person 1 and Person 25: 109 bits
  Distance between Person 1 and Person 26: 119 bits
  Distance between Person 1 and Person 27: 105 bits
  Distance between Person 1 and Person 28: 138 bits
  Distance between Person 1 and Person 29: 146 bits
  Distance between Person 1 and Person 30: 132 bits
  Distance between Person 1 and Person 31: 125 bits
  Distance between Person 1 and Person 32: 135 bits
  Distance between Person 1 and Person 33: 104 bits
  Distance between Person 1 and Person 34: 122 bits
  Distance between Person 1 and Person 35: 135 bits
  Distance between Person 1 and Person 36: 123 bits
  Distance between Person 1 and Person 37: 132 bits
  Distance between Person 1 and Person 38: 145 bits
  Distance between Person 1 and Person 39: 143 bits
  Distance between Person 1 and Person 40: 139 bits
  Distance between Person 1 and Person 41: 129 bits
  Distance between Person 1 and Person 42: 117 bits
  Distance between Person 1 and Person 43: 128 bits
  Distance between Person 1 and Person 44: 113 bits
  Distance between Person 1 and Person 45: 135 bits
  Distance between Person 1 and Person 46: 96 bits
  Distance between Person 1 and Person 47: 125 bits
  Distance between Person 1 and Person 48: 130 bits
  Distance between Person 1 and Person 49: 118 bits
  Distance between Person 1 and Person 50: 136 bits
  Distance between Person 1 and Person 51: 134 bits
  Distance between Person 1 and Person 52: 139 bits
  Distance between Person 1 and Person 53: 142 bits
  Distance between Person 1 and Person 54: 131 bits
  Distance between Person 1 and Person 55: 145 bits
  Distance between Person 1 and Person 56: 139 bits
  Distance between Person 1 and Person 57: 140 bits
  Distance between Person 1 and Person 58: 86 bits
  Distance between Person 1 and Person 59: 105 bits
  Distance between Person 1 and Person 60: 134 bits
  Distance between Person 1 and Person 61: 132 bits
  Distance between Person 1 and Person 62: 122 bits
  Distance between Person 1 and Person 63: 124 bits
  Distance between Person 1 and Person 64: 137 bits
  Distance between Person 1 and Person 65: 99 bits
  Distance between Person 1 and Person 66: 115 bits
  Distance between Person 1 and Person 67: 102 bits
  Distance between Person 1 and Person 68: 144 bits
  Distance between Person 1 and Person 69: 90 bits
  Distance between Person 1 and Person 70: 126 bits
  Distance between Person 1 and Person 71: 135 bits
  Distance between Person 1 and Person 72: 102 bits
  Distance between Person 1 and Person 73: 108 bits
  Distance between Person 1 and Person 74: 130 bits
  Distance between Person 1 and Person 75: 123 bits
  Distance between Person 1 and Person 76: 127 bits
  Distance between Person 1 and Person 77: 126 bits
  Distance between Person 1 and Person 78: 106 bits
  Distance between Person 1 and Person 79: 111 bits
  Distance between Person 1 and Person 80: 116 bits
  Distance between Person 1 and Person 81: 122 bits
  Distance between Person 1 and Person 82: 125 bits
  Distance between Person 1 and Person 83: 130 bits
  Distance between Person 1 and Person 84: 118 bits
  Distance between Person 1 and Person 85: 119 bits
  Distance between Person 1 and Person 86: 118 bits
  Distance between Person 1 and Person 87: 126 bits
  Distance between Person 1 and Person 88: 141 bits
  Distance between Person 1 and Person 89: 140 bits
  Distance between Person 2 and Person 3: 106 bits
  Distance between Person 2 and Person 4: 117 bits
  Distance between Person 2 and Person 5: 133 bits
  Distance between Person 2 and Person 6: 131 bits
  Distance between Person 2 and Person 7: 137 bits
  Distance between Person 2 and Person 8: 119 bits
  Distance between Person 2 and Person 9: 130 bits
  Distance between Person 2 and Person 10: 114 bits
  Distance between Person 2 and Person 11: 155 bits
  Distance between Person 2 and Person 12: 116 bits
  Distance between Person 2 and Person 13: 125 bits
  Distance between Person 2 and Person 14: 105 bits
  Distance between Person 2 and Person 15: 137 bits
  Distance between Person 2 and Person 16: 110 bits
  Distance between Person 2 and Person 17: 133 bits
  Distance between Person 2 and Person 18: 124 bits
  Distance between Person 2 and Person 19: 122 bits
  Distance between Person 2 and Person 20: 122 bits
  Distance between Person 2 and Person 21: 115 bits
  Distance between Person 2 and Person 22: 123 bits
  Distance between Person 2 and Person 23: 135 bits
  Distance between Person 2 and Person 24: 125 bits
  Distance between Person 2 and Person 25: 119 bits
  Distance between Person 2 and Person 26: 111 bits
  Distance between Person 2 and Person 27: 99 bits
  Distance between Person 2 and Person 28: 122 bits
  Distance between Person 2 and Person 29: 124 bits
  Distance between Person 2 and Person 30: 122 bits
  Distance between Person 2 and Person 31: 149 bits
  Distance between Person 2 and Person 32: 127 bits
  Distance between Person 2 and Person 33: 118 bits
  Distance between Person 2 and Person 34: 104 bits
  Distance between Person 2 and Person 35: 127 bits
  Distance between Person 2 and Person 36: 123 bits
  Distance between Person 2 and Person 37: 106 bits
  Distance between Person 2 and Person 38: 147 bits
  Distance between Person 2 and Person 39: 125 bits
  Distance between Person 2 and Person 40: 129 bits
  Distance between Person 2 and Person 41: 133 bits
  Distance between Person 2 and Person 42: 113 bits
  Distance between Person 2 and Person 43: 114 bits
  Distance between Person 2 and Person 44: 117 bits
  Distance between Person 2 and Person 45: 119 bits
  Distance between Person 2 and Person 46: 112 bits
  Distance between Person 2 and Person 47: 127 bits
  Distance between Person 2 and Person 48: 128 bits
  Distance between Person 2 and Person 49: 132 bits
  Distance between Person 2 and Person 50: 122 bits
  Distance between Person 2 and Person 51: 36 bits
  Distance between Person 2 and Person 52: 111 bits
  Distance between Person 2 and Person 53: 140 bits
  Distance between Person 2 and Person 54: 115 bits
  Distance between Person 2 and Person 55: 127 bits
  Distance between Person 2 and Person 56: 113 bits
  Distance between Person 2 and Person 57: 144 bits
  Distance between Person 2 and Person 58: 130 bits
  Distance between Person 2 and Person 59: 101 bits
  Distance between Person 2 and Person 60: 140 bits
  Distance between Person 2 and Person 61: 124 bits
  Distance between Person 2 and Person 62: 110 bits
  Distance between Person 2 and Person 63: 136 bits
  Distance between Person 2 and Person 64: 111 bits
  Distance between Person 2 and Person 65: 127 bits
  Distance between Person 2 and Person 66: 133 bits
  Distance between Person 2 and Person 67: 122 bits
  Distance between Person 2 and Person 68: 114 bits
  Distance between Person 2 and Person 69: 122 bits
  Distance between Person 2 and Person 70: 114 bits
  Distance between Person 2 and Person 71: 133 bits
  Distance between Person 2 and Person 72: 130 bits
  Distance between Person 2 and Person 73: 120 bits
  Distance between Person 2 and Person 74: 120 bits
  Distance between Person 2 and Person 75: 133 bits
  Distance between Person 2 and Person 76: 115 bits
  Distance between Person 2 and Person 77: 118 bits
  Distance between Person 2 and Person 78: 130 bits
  Distance between Person 2 and Person 79: 125 bits
  Distance between Person 2 and Person 80: 106 bits
  Distance between Person 2 and Person 81: 140 bits
  Distance between Person 2 and Person 82: 113 bits
  Distance between Person 2 and Person 83: 124 bits
  Distance between Person 2 and Person 84: 110 bits
  Distance between Person 2 and Person 85: 117 bits
  Distance between Person 2 and Person 86: 142 bits
  Distance between Person 2 and Person 87: 132 bits
  Distance between Person 2 and Person 88: 129 bits
  Distance between Person 2 and Person 89: 128 bits
  Distance between Person 3 and Person 4: 113 bits
  Distance between Person 3 and Person 5: 125 bits
  Distance between Person 3 and Person 6: 123 bits
  Distance between Person 3 and Person 7: 151 bits
  Distance between Person 3 and Person 8: 111 bits
  Distance between Person 3 and Person 9: 136 bits
  Distance between Person 3 and Person 10: 118 bits
  Distance between Person 3 and Person 11: 123 bits
  Distance between Person 3 and Person 12: 150 bits
  Distance between Person 3 and Person 13: 117 bits
  Distance between Person 3 and Person 14: 143 bits
  Distance between Person 3 and Person 15: 141 bits
  Distance between Person 3 and Person 16: 126 bits
  Distance between Person 3 and Person 17: 135 bits
  Distance between Person 3 and Person 18: 136 bits
  Distance between Person 3 and Person 19: 138 bits
  Distance between Person 3 and Person 20: 144 bits
  Distance between Person 3 and Person 21: 127 bits
  Distance between Person 3 and Person 22: 117 bits
  Distance between Person 3 and Person 23: 135 bits
  Distance between Person 3 and Person 24: 87 bits
  Distance between Person 3 and Person 25: 139 bits
  Distance between Person 3 and Person 26: 123 bits
  Distance between Person 3 and Person 27: 111 bits
  Distance between Person 3 and Person 28: 138 bits
  Distance between Person 3 and Person 29: 112 bits
  Distance between Person 3 and Person 30: 106 bits
  Distance between Person 3 and Person 31: 141 bits
  Distance between Person 3 and Person 32: 107 bits
  Distance between Person 3 and Person 33: 122 bits
  Distance between Person 3 and Person 34: 96 bits
  Distance between Person 3 and Person 35: 113 bits
  Distance between Person 3 and Person 36: 121 bits
  Distance between Person 3 and Person 37: 90 bits
  Distance between Person 3 and Person 38: 145 bits
  Distance between Person 3 and Person 39: 81 bits
  Distance between Person 3 and Person 40: 117 bits
  Distance between Person 3 and Person 41: 129 bits
  Distance between Person 3 and Person 42: 141 bits
  Distance between Person 3 and Person 43: 128 bits
  Distance between Person 3 and Person 44: 127 bits
  Distance between Person 3 and Person 45: 119 bits
  Distance between Person 3 and Person 46: 130 bits
  Distance between Person 3 and Person 47: 123 bits
  Distance between Person 3 and Person 48: 112 bits
  Distance between Person 3 and Person 49: 134 bits
  Distance between Person 3 and Person 50: 114 bits
  Distance between Person 3 and Person 51: 112 bits
  Distance between Person 3 and Person 52: 119 bits
  Distance between Person 3 and Person 53: 128 bits
  Distance between Person 3 and Person 54: 145 bits
  Distance between Person 3 and Person 55: 129 bits
  Distance between Person 3 and Person 56: 127 bits
  Distance between Person 3 and Person 57: 130 bits
  Distance between Person 3 and Person 58: 144 bits
  Distance between Person 3 and Person 59: 127 bits
  Distance between Person 3 and Person 60: 142 bits
  Distance between Person 3 and Person 61: 130 bits
  Distance between Person 3 and Person 62: 120 bits
  Distance between Person 3 and Person 63: 136 bits
  Distance between Person 3 and Person 64: 111 bits
  Distance between Person 3 and Person 65: 127 bits
  Distance between Person 3 and Person 66: 111 bits
  Distance between Person 3 and Person 67: 126 bits
  Distance between Person 3 and Person 68: 118 bits
  Distance between Person 3 and Person 69: 134 bits
  Distance between Person 3 and Person 70: 132 bits
  Distance between Person 3 and Person 71: 117 bits
  Distance between Person 3 and Person 72: 128 bits
  Distance between Person 3 and Person 73: 126 bits
  Distance between Person 3 and Person 74: 124 bits
  Distance between Person 3 and Person 75: 147 bits
  Distance between Person 3 and Person 76: 119 bits
  Distance between Person 3 and Person 77: 124 bits
  Distance between Person 3 and Person 78: 138 bits
  Distance between Person 3 and Person 79: 107 bits
  Distance between Person 3 and Person 80: 128 bits
  Distance between Person 3 and Person 81: 130 bits
  Distance between Person 3 and Person 82: 117 bits
  Distance between Person 3 and Person 83: 120 bits
  Distance between Person 3 and Person 84: 110 bits
  Distance between Person 3 and Person 85: 131 bits
  Distance between Person 3 and Person 86: 136 bits
  Distance between Person 3 and Person 87: 128 bits
  Distance between Person 3 and Person 88: 149 bits
  Distance between Person 3 and Person 89: 148 bits
  Distance between Person 4 and Person 5: 126 bits
  Distance between Person 4 and Person 6: 96 bits
  Distance between Person 4 and Person 7: 144 bits
  Distance between Person 4 and Person 8: 126 bits
  Distance between Person 4 and Person 9: 139 bits
  Distance between Person 4 and Person 10: 127 bits
  Distance between Person 4 and Person 11: 146 bits
  Distance between Person 4 and Person 12: 135 bits
  Distance between Person 4 and Person 13: 130 bits
  Distance between Person 4 and Person 14: 102 bits
  Distance between Person 4 and Person 15: 116 bits
  Distance between Person 4 and Person 16: 105 bits
  Distance between Person 4 and Person 17: 124 bits
  Distance between Person 4 and Person 18: 113 bits
  Distance between Person 4 and Person 19: 123 bits
  Distance between Person 4 and Person 20: 155 bits
  Distance between Person 4 and Person 21: 118 bits
  Distance between Person 4 and Person 22: 142 bits
  Distance between Person 4 and Person 23: 140 bits
  Distance between Person 4 and Person 24: 116 bits
  Distance between Person 4 and Person 25: 112 bits
  Distance between Person 4 and Person 26: 112 bits
  Distance between Person 4 and Person 27: 104 bits
  Distance between Person 4 and Person 28: 139 bits
  Distance between Person 4 and Person 29: 135 bits
  Distance between Person 4 and Person 30: 141 bits
  Distance between Person 4 and Person 31: 136 bits
  Distance between Person 4 and Person 32: 130 bits
  Distance between Person 4 and Person 33: 123 bits
  Distance between Person 4 and Person 34: 53 bits
  Distance between Person 4 and Person 35: 126 bits
  Distance between Person 4 and Person 36: 142 bits
  Distance between Person 4 and Person 37: 121 bits
  Distance between Person 4 and Person 38: 120 bits
  Distance between Person 4 and Person 39: 136 bits
  Distance between Person 4 and Person 40: 74 bits
  Distance between Person 4 and Person 41: 100 bits
  Distance between Person 4 and Person 42: 122 bits
  Distance between Person 4 and Person 43: 115 bits
  Distance between Person 4 and Person 44: 128 bits
  Distance between Person 4 and Person 45: 116 bits
  Distance between Person 4 and Person 46: 127 bits
  Distance between Person 4 and Person 47: 138 bits
  Distance between Person 4 and Person 48: 119 bits
  Distance between Person 4 and Person 49: 131 bits
  Distance between Person 4 and Person 50: 139 bits
  Distance between Person 4 and Person 51: 127 bits
  Distance between Person 4 and Person 52: 106 bits
  Distance between Person 4 and Person 53: 129 bits
  Distance between Person 4 and Person 54: 144 bits
  Distance between Person 4 and Person 55: 114 bits
  Distance between Person 4 and Person 56: 138 bits
  Distance between Person 4 and Person 57: 129 bits
  Distance between Person 4 and Person 58: 117 bits
  Distance between Person 4 and Person 59: 116 bits
  Distance between Person 4 and Person 60: 121 bits
  Distance between Person 4 and Person 61: 141 bits
  Distance between Person 4 and Person 62: 121 bits
  Distance between Person 4 and Person 63: 129 bits
  Distance between Person 4 and Person 64: 132 bits
  Distance between Person 4 and Person 65: 116 bits
  Distance between Person 4 and Person 66: 120 bits
  Distance between Person 4 and Person 67: 123 bits
  Distance between Person 4 and Person 68: 123 bits
  Distance between Person 4 and Person 69: 127 bits
  Distance between Person 4 and Person 70: 113 bits
  Distance between Person 4 and Person 71: 122 bits
  Distance between Person 4 and Person 72: 129 bits
  Distance between Person 4 and Person 73: 123 bits
  Distance between Person 4 and Person 74: 121 bits
  Distance between Person 4 and Person 75: 142 bits
  Distance between Person 4 and Person 76: 134 bits
  Distance between Person 4 and Person 77: 135 bits
  Distance between Person 4 and Person 78: 117 bits
  Distance between Person 4 and Person 79: 118 bits
  Distance between Person 4 and Person 80: 113 bits
  Distance between Person 4 and Person 81: 119 bits
  Distance between Person 4 and Person 82: 132 bits
  Distance between Person 4 and Person 83: 127 bits
  Distance between Person 4 and Person 84: 117 bits
  Distance between Person 4 and Person 85: 114 bits
  Distance between Person 4 and Person 86: 123 bits
  Distance between Person 4 and Person 87: 147 bits
  Distance between Person 4 and Person 88: 138 bits
  Distance between Person 4 and Person 89: 145 bits
  Distance between Person 5 and Person 6: 144 bits
  Distance between Person 5 and Person 7: 132 bits
  Distance between Person 5 and Person 8: 122 bits
  Distance between Person 5 and Person 9: 113 bits
  Distance between Person 5 and Person 10: 131 bits
  Distance between Person 5 and Person 11: 112 bits
  Distance between Person 5 and Person 12: 145 bits
  Distance between Person 5 and Person 13: 130 bits
  Distance between Person 5 and Person 14: 130 bits
  Distance between Person 5 and Person 15: 108 bits
  Distance between Person 5 and Person 16: 113 bits
  Distance between Person 5 and Person 17: 122 bits
  Distance between Person 5 and Person 18: 127 bits
  Distance between Person 5 and Person 19: 127 bits
  Distance between Person 5 and Person 20: 141 bits
  Distance between Person 5 and Person 21: 136 bits
  Distance between Person 5 and Person 22: 100 bits
  Distance between Person 5 and Person 23: 132 bits
  Distance between Person 5 and Person 24: 118 bits
  Distance between Person 5 and Person 25: 132 bits
  Distance between Person 5 and Person 26: 128 bits
  Distance between Person 5 and Person 27: 120 bits
  Distance between Person 5 and Person 28: 113 bits
  Distance between Person 5 and Person 29: 141 bits
  Distance between Person 5 and Person 30: 123 bits
  Distance between Person 5 and Person 31: 126 bits
  Distance between Person 5 and Person 32: 130 bits
  Distance between Person 5 and Person 33: 137 bits
  Distance between Person 5 and Person 34: 133 bits
  Distance between Person 5 and Person 35: 132 bits
  Distance between Person 5 and Person 36: 112 bits
  Distance between Person 5 and Person 37: 129 bits
  Distance between Person 5 and Person 38: 128 bits
  Distance between Person 5 and Person 39: 120 bits
  Distance between Person 5 and Person 40: 124 bits
  Distance between Person 5 and Person 41: 114 bits
  Distance between Person 5 and Person 42: 144 bits
  Distance between Person 5 and Person 43: 131 bits
  Distance between Person 5 and Person 44: 136 bits
  Distance between Person 5 and Person 45: 132 bits
  Distance between Person 5 and Person 46: 133 bits
  Distance between Person 5 and Person 47: 140 bits
  Distance between Person 5 and Person 48: 109 bits
  Distance between Person 5 and Person 49: 97 bits
  Distance between Person 5 and Person 50: 135 bits
  Distance between Person 5 and Person 51: 125 bits
  Distance between Person 5 and Person 52: 112 bits
  Distance between Person 5 and Person 53: 135 bits
  Distance between Person 5 and Person 54: 124 bits
  Distance between Person 5 and Person 55: 106 bits
  Distance between Person 5 and Person 56: 122 bits
  Distance between Person 5 and Person 57: 125 bits
  Distance between Person 5 and Person 58: 119 bits
  Distance between Person 5 and Person 59: 122 bits
  Distance between Person 5 and Person 60: 125 bits
  Distance between Person 5 and Person 61: 125 bits
  Distance between Person 5 and Person 62: 115 bits
  Distance between Person 5 and Person 63: 137 bits
  Distance between Person 5 and Person 64: 128 bits
  Distance between Person 5 and Person 65: 134 bits
  Distance between Person 5 and Person 66: 144 bits
  Distance between Person 5 and Person 67: 125 bits
  Distance between Person 5 and Person 68: 121 bits
  Distance between Person 5 and Person 69: 133 bits
  Distance between Person 5 and Person 70: 139 bits
  Distance between Person 5 and Person 71: 130 bits
  Distance between Person 5 and Person 72: 115 bits
  Distance between Person 5 and Person 73: 129 bits
  Distance between Person 5 and Person 74: 135 bits
  Distance between Person 5 and Person 75: 142 bits
  Distance between Person 5 and Person 76: 140 bits
  Distance between Person 5 and Person 77: 119 bits
  Distance between Person 5 and Person 78: 129 bits
  Distance between Person 5 and Person 79: 126 bits
  Distance between Person 5 and Person 80: 115 bits
  Distance between Person 5 and Person 81: 119 bits
  Distance between Person 5 and Person 82: 156 bits
  Distance between Person 5 and Person 83: 121 bits
  Distance between Person 5 and Person 84: 109 bits
  Distance between Person 5 and Person 85: 138 bits
  Distance between Person 5 and Person 86: 119 bits
  Distance between Person 5 and Person 87: 127 bits
  Distance between Person 5 and Person 88: 136 bits
  Distance between Person 5 and Person 89: 117 bits
  Distance between Person 6 and Person 7: 146 bits
  Distance between Person 6 and Person 8: 124 bits
  Distance between Person 6 and Person 9: 145 bits
  Distance between Person 6 and Person 10: 111 bits
  Distance between Person 6 and Person 11: 136 bits
  Distance between Person 6 and Person 12: 141 bits
  Distance between Person 6 and Person 13: 132 bits
  Distance between Person 6 and Person 14: 110 bits
  Distance between Person 6 and Person 15: 134 bits
  Distance between Person 6 and Person 16: 131 bits
  Distance between Person 6 and Person 17: 128 bits
  Distance between Person 6 and Person 18: 101 bits
  Distance between Person 6 and Person 19: 117 bits
  Distance between Person 6 and Person 20: 137 bits
  Distance between Person 6 and Person 21: 122 bits
  Distance between Person 6 and Person 22: 132 bits
  Distance between Person 6 and Person 23: 104 bits
  Distance between Person 6 and Person 24: 140 bits
  Distance between Person 6 and Person 25: 128 bits
  Distance between Person 6 and Person 26: 144 bits
  Distance between Person 6 and Person 27: 124 bits
  Distance between Person 6 and Person 28: 147 bits
  Distance between Person 6 and Person 29: 123 bits
  Distance between Person 6 and Person 30: 123 bits
  Distance between Person 6 and Person 31: 146 bits
  Distance between Person 6 and Person 32: 114 bits
  Distance between Person 6 and Person 33: 131 bits
  Distance between Person 6 and Person 34: 115 bits
  Distance between Person 6 and Person 35: 136 bits
  Distance between Person 6 and Person 36: 138 bits
  Distance between Person 6 and Person 37: 119 bits
  Distance between Person 6 and Person 38: 130 bits
  Distance between Person 6 and Person 39: 118 bits
  Distance between Person 6 and Person 40: 94 bits
  Distance between Person 6 and Person 41: 124 bits
  Distance between Person 6 and Person 42: 118 bits
  Distance between Person 6 and Person 43: 123 bits
  Distance between Person 6 and Person 44: 120 bits
  Distance between Person 6 and Person 45: 106 bits
  Distance between Person 6 and Person 46: 127 bits
  Distance between Person 6 and Person 47: 120 bits
  Distance between Person 6 and Person 48: 117 bits
  Distance between Person 6 and Person 49: 137 bits
  Distance between Person 6 and Person 50: 127 bits
  Distance between Person 6 and Person 51: 143 bits
  Distance between Person 6 and Person 52: 120 bits
  Distance between Person 6 and Person 53: 127 bits
  Distance between Person 6 and Person 54: 126 bits
  Distance between Person 6 and Person 55: 108 bits
  Distance between Person 6 and Person 56: 122 bits
  Distance between Person 6 and Person 57: 127 bits
  Distance between Person 6 and Person 58: 129 bits
  Distance between Person 6 and Person 59: 112 bits
  Distance between Person 6 and Person 60: 123 bits
  Distance between Person 6 and Person 61: 143 bits
  Distance between Person 6 and Person 62: 125 bits
  Distance between Person 6 and Person 63: 109 bits
  Distance between Person 6 and Person 64: 116 bits
  Distance between Person 6 and Person 65: 132 bits
  Distance between Person 6 and Person 66: 116 bits
  Distance between Person 6 and Person 67: 127 bits
  Distance between Person 6 and Person 68: 131 bits
  Distance between Person 6 and Person 69: 125 bits
  Distance between Person 6 and Person 70: 135 bits
  Distance between Person 6 and Person 71: 98 bits
  Distance between Person 6 and Person 72: 153 bits
  Distance between Person 6 and Person 73: 121 bits
  Distance between Person 6 and Person 74: 123 bits
  Distance between Person 6 and Person 75: 120 bits
  Distance between Person 6 and Person 76: 116 bits
  Distance between Person 6 and Person 77: 137 bits
  Distance between Person 6 and Person 78: 123 bits
  Distance between Person 6 and Person 79: 146 bits
  Distance between Person 6 and Person 80: 113 bits
  Distance between Person 6 and Person 81: 117 bits
  Distance between Person 6 and Person 82: 116 bits
  Distance between Person 6 and Person 83: 119 bits
  Distance between Person 6 and Person 84: 143 bits
  Distance between Person 6 and Person 85: 152 bits
  Distance between Person 6 and Person 86: 115 bits
  Distance between Person 6 and Person 87: 151 bits
  Distance between Person 6 and Person 88: 144 bits
  Distance between Person 6 and Person 89: 147 bits
  Distance between Person 7 and Person 8: 140 bits
  Distance between Person 7 and Person 9: 123 bits
  Distance between Person 7 and Person 10: 139 bits
  Distance between Person 7 and Person 11: 134 bits
  Distance between Person 7 and Person 12: 131 bits
  Distance between Person 7 and Person 13: 132 bits
  Distance between Person 7 and Person 14: 120 bits
  Distance between Person 7 and Person 15: 140 bits
  Distance between Person 7 and Person 16: 123 bits
  Distance between Person 7 and Person 17: 128 bits
  Distance between Person 7 and Person 18: 131 bits
  Distance between Person 7 and Person 19: 109 bits
  Distance between Person 7 and Person 20: 123 bits
  Distance between Person 7 and Person 21: 102 bits
  Distance between Person 7 and Person 22: 132 bits
  Distance between Person 7 and Person 23: 110 bits
  Distance between Person 7 and Person 24: 122 bits
  Distance between Person 7 and Person 25: 134 bits
  Distance between Person 7 and Person 26: 128 bits
  Distance between Person 7 and Person 27: 122 bits
  Distance between Person 7 and Person 28: 119 bits
  Distance between Person 7 and Person 29: 109 bits
  Distance between Person 7 and Person 30: 139 bits
  Distance between Person 7 and Person 31: 108 bits
  Distance between Person 7 and Person 32: 158 bits
  Distance between Person 7 and Person 33: 115 bits
  Distance between Person 7 and Person 34: 139 bits
  Distance between Person 7 and Person 35: 142 bits
  Distance between Person 7 and Person 36: 132 bits
  Distance between Person 7 and Person 37: 119 bits
  Distance between Person 7 and Person 38: 122 bits
  Distance between Person 7 and Person 39: 150 bits
  Distance between Person 7 and Person 40: 128 bits
  Distance between Person 7 and Person 41: 120 bits
  Distance between Person 7 and Person 42: 128 bits
  Distance between Person 7 and Person 43: 123 bits
  Distance between Person 7 and Person 44: 122 bits
  Distance between Person 7 and Person 45: 124 bits
  Distance between Person 7 and Person 46: 121 bits
  Distance between Person 7 and Person 47: 122 bits
  Distance between Person 7 and Person 48: 147 bits
  Distance between Person 7 and Person 49: 115 bits
  Distance between Person 7 and Person 50: 141 bits
  Distance between Person 7 and Person 51: 131 bits
  Distance between Person 7 and Person 52: 122 bits
  Distance between Person 7 and Person 53: 111 bits
  Distance between Person 7 and Person 54: 130 bits
  Distance between Person 7 and Person 55: 104 bits
  Distance between Person 7 and Person 56: 124 bits
  Distance between Person 7 and Person 57: 113 bits
  Distance between Person 7 and Person 58: 133 bits
  Distance between Person 7 and Person 59: 142 bits
  Distance between Person 7 and Person 60: 101 bits
  Distance between Person 7 and Person 61: 115 bits
  Distance between Person 7 and Person 62: 145 bits
  Distance between Person 7 and Person 63: 125 bits
  Distance between Person 7 and Person 64: 152 bits
  Distance between Person 7 and Person 65: 114 bits
  Distance between Person 7 and Person 66: 136 bits
  Distance between Person 7 and Person 67: 113 bits
  Distance between Person 7 and Person 68: 143 bits
  Distance between Person 7 and Person 69: 117 bits
  Distance between Person 7 and Person 70: 105 bits
  Distance between Person 7 and Person 71: 138 bits
  Distance between Person 7 and Person 72: 131 bits
  Distance between Person 7 and Person 73: 133 bits
  Distance between Person 7 and Person 74: 133 bits
  Distance between Person 7 and Person 75: 108 bits
  Distance between Person 7 and Person 76: 140 bits
  Distance between Person 7 and Person 77: 113 bits
  Distance between Person 7 and Person 78: 143 bits
  Distance between Person 7 and Person 79: 132 bits
  Distance between Person 7 and Person 80: 113 bits
  Distance between Person 7 and Person 81: 151 bits
  Distance between Person 7 and Person 82: 122 bits
  Distance between Person 7 and Person 83: 107 bits
  Distance between Person 7 and Person 84: 127 bits
  Distance between Person 7 and Person 85: 124 bits
  Distance between Person 7 and Person 86: 125 bits
  Distance between Person 7 and Person 87: 131 bits
  Distance between Person 7 and Person 88: 112 bits
  Distance between Person 7 and Person 89: 111 bits
  Distance between Person 8 and Person 9: 113 bits
  Distance between Person 8 and Person 10: 125 bits
  Distance between Person 8 and Person 11: 124 bits
  Distance between Person 8 and Person 12: 127 bits
  Distance between Person 8 and Person 13: 122 bits
  Distance between Person 8 and Person 14: 122 bits
  Distance between Person 8 and Person 15: 118 bits
  Distance between Person 8 and Person 16: 123 bits
  Distance between Person 8 and Person 17: 122 bits
  Distance between Person 8 and Person 18: 109 bits
  Distance between Person 8 and Person 19: 153 bits
  Distance between Person 8 and Person 20: 133 bits
  Distance between Person 8 and Person 21: 134 bits
  Distance between Person 8 and Person 22: 114 bits
  Distance between Person 8 and Person 23: 148 bits
  Distance between Person 8 and Person 24: 122 bits
  Distance between Person 8 and Person 25: 110 bits
  Distance between Person 8 and Person 26: 140 bits
  Distance between Person 8 and Person 27: 138 bits
  Distance between Person 8 and Person 28: 121 bits
  Distance between Person 8 and Person 29: 117 bits
  Distance between Person 8 and Person 30: 137 bits
  Distance between Person 8 and Person 31: 144 bits
  Distance between Person 8 and Person 32: 132 bits
  Distance between Person 8 and Person 33: 131 bits
  Distance between Person 8 and Person 34: 115 bits
  Distance between Person 8 and Person 35: 114 bits
  Distance between Person 8 and Person 36: 130 bits
  Distance between Person 8 and Person 37: 107 bits
  Distance between Person 8 and Person 38: 136 bits
  Distance between Person 8 and Person 39: 116 bits
  Distance between Person 8 and Person 40: 112 bits
  Distance between Person 8 and Person 41: 134 bits
  Distance between Person 8 and Person 42: 118 bits
  Distance between Person 8 and Person 43: 119 bits
  Distance between Person 8 and Person 44: 124 bits
  Distance between Person 8 and Person 45: 116 bits
  Distance between Person 8 and Person 46: 129 bits
  Distance between Person 8 and Person 47: 116 bits
  Distance between Person 8 and Person 48: 125 bits
  Distance between Person 8 and Person 49: 137 bits
  Distance between Person 8 and Person 50: 137 bits
  Distance between Person 8 and Person 51: 131 bits
  Distance between Person 8 and Person 52: 116 bits
  Distance between Person 8 and Person 53: 125 bits
  Distance between Person 8 and Person 54: 108 bits
  Distance between Person 8 and Person 55: 128 bits
  Distance between Person 8 and Person 56: 110 bits
  Distance between Person 8 and Person 57: 123 bits
  Distance between Person 8 and Person 58: 119 bits
  Distance between Person 8 and Person 59: 130 bits
  Distance between Person 8 and Person 60: 121 bits
  Distance between Person 8 and Person 61: 151 bits
  Distance between Person 8 and Person 62: 137 bits
  Distance between Person 8 and Person 63: 117 bits
  Distance between Person 8 and Person 64: 122 bits
  Distance between Person 8 and Person 65: 126 bits
  Distance between Person 8 and Person 66: 138 bits
  Distance between Person 8 and Person 67: 121 bits
  Distance between Person 8 and Person 68: 107 bits
  Distance between Person 8 and Person 69: 133 bits
  Distance between Person 8 and Person 70: 141 bits
  Distance between Person 8 and Person 71: 130 bits
  Distance between Person 8 and Person 72: 147 bits
  Distance between Person 8 and Person 73: 111 bits
  Distance between Person 8 and Person 74: 121 bits
  Distance between Person 8 and Person 75: 114 bits
  Distance between Person 8 and Person 76: 132 bits
  Distance between Person 8 and Person 77: 107 bits
  Distance between Person 8 and Person 78: 131 bits
  Distance between Person 8 and Person 79: 122 bits
  Distance between Person 8 and Person 80: 125 bits
  Distance between Person 8 and Person 81: 127 bits
  Distance between Person 8 and Person 82: 132 bits
  Distance between Person 8 and Person 83: 135 bits
  Distance between Person 8 and Person 84: 119 bits
  Distance between Person 8 and Person 85: 132 bits
  Distance between Person 8 and Person 86: 145 bits
  Distance between Person 8 and Person 87: 131 bits
  Distance between Person 8 and Person 88: 122 bits
  Distance between Person 8 and Person 89: 141 bits
  Distance between Person 9 and Person 10: 132 bits
  Distance between Person 9 and Person 11: 121 bits
  Distance between Person 9 and Person 12: 122 bits
  Distance between Person 9 and Person 13: 139 bits
  Distance between Person 9 and Person 14: 151 bits
  Distance between Person 9 and Person 15: 115 bits
  Distance between Person 9 and Person 16: 138 bits
  Distance between Person 9 and Person 17: 125 bits
  Distance between Person 9 and Person 18: 122 bits
  Distance between Person 9 and Person 19: 136 bits
  Distance between Person 9 and Person 20: 118 bits
  Distance between Person 9 and Person 21: 125 bits
  Distance between Person 9 and Person 22: 131 bits
  Distance between Person 9 and Person 23: 105 bits
  Distance between Person 9 and Person 24: 135 bits
  Distance between Person 9 and Person 25: 121 bits
  Distance between Person 9 and Person 26: 145 bits
  Distance between Person 9 and Person 27: 135 bits
  Distance between Person 9 and Person 28: 104 bits
  Distance between Person 9 and Person 29: 114 bits
  Distance between Person 9 and Person 30: 148 bits
  Distance between Person 9 and Person 31: 113 bits
  Distance between Person 9 and Person 32: 131 bits
  Distance between Person 9 and Person 33: 108 bits
  Distance between Person 9 and Person 34: 132 bits
  Distance between Person 9 and Person 35: 145 bits
  Distance between Person 9 and Person 36: 143 bits
  Distance between Person 9 and Person 37: 124 bits
  Distance between Person 9 and Person 38: 135 bits
  Distance between Person 9 and Person 39: 129 bits
  Distance between Person 9 and Person 40: 139 bits
  Distance between Person 9 and Person 41: 111 bits
  Distance between Person 9 and Person 42: 125 bits
  Distance between Person 9 and Person 43: 116 bits
  Distance between Person 9 and Person 44: 129 bits
  Distance between Person 9 and Person 45: 129 bits
  Distance between Person 9 and Person 46: 126 bits
  Distance between Person 9 and Person 47: 139 bits
  Distance between Person 9 and Person 48: 136 bits
  Distance between Person 9 and Person 49: 148 bits
  Distance between Person 9 and Person 50: 132 bits
  Distance between Person 9 and Person 51: 114 bits
  Distance between Person 9 and Person 52: 151 bits
  Distance between Person 9 and Person 53: 126 bits
  Distance between Person 9 and Person 54: 107 bits
  Distance between Person 9 and Person 55: 147 bits
  Distance between Person 9 and Person 56: 133 bits
  Distance between Person 9 and Person 57: 124 bits
  Distance between Person 9 and Person 58: 124 bits
  Distance between Person 9 and Person 59: 131 bits
  Distance between Person 9 and Person 60: 124 bits
  Distance between Person 9 and Person 61: 136 bits
  Distance between Person 9 and Person 62: 136 bits
  Distance between Person 9 and Person 63: 106 bits
  Distance between Person 9 and Person 64: 143 bits
  Distance between Person 9 and Person 65: 157 bits
  Distance between Person 9 and Person 66: 141 bits
  Distance between Person 9 and Person 67: 104 bits
  Distance between Person 9 and Person 68: 124 bits
  Distance between Person 9 and Person 69: 124 bits
  Distance between Person 9 and Person 70: 136 bits
  Distance between Person 9 and Person 71: 151 bits
  Distance between Person 9 and Person 72: 126 bits
  Distance between Person 9 and Person 73: 116 bits
  Distance between Person 9 and Person 74: 128 bits
  Distance between Person 9 and Person 75: 131 bits
  Distance between Person 9 and Person 76: 129 bits
  Distance between Person 9 and Person 77: 120 bits
  Distance between Person 9 and Person 78: 98 bits
  Distance between Person 9 and Person 79: 149 bits
  Distance between Person 9 and Person 80: 126 bits
  Distance between Person 9 and Person 81: 106 bits
  Distance between Person 9 and Person 82: 123 bits
  Distance between Person 9 and Person 83: 128 bits
  Distance between Person 9 and Person 84: 128 bits
  Distance between Person 9 and Person 85: 131 bits
  Distance between Person 9 and Person 86: 126 bits
  Distance between Person 9 and Person 87: 120 bits
  Distance between Person 9 and Person 88: 113 bits
  Distance between Person 9 and Person 89: 134 bits
  Distance between Person 10 and Person 11: 139 bits
  Distance between Person 10 and Person 12: 136 bits
  Distance between Person 10 and Person 13: 137 bits
  Distance between Person 10 and Person 14: 133 bits
  Distance between Person 10 and Person 15: 117 bits
  Distance between Person 10 and Person 16: 130 bits
  Distance between Person 10 and Person 17: 125 bits
  Distance between Person 10 and Person 18: 112 bits
  Distance between Person 10 and Person 19: 124 bits
  Distance between Person 10 and Person 20: 128 bits
  Distance between Person 10 and Person 21: 123 bits
  Distance between Person 10 and Person 22: 133 bits
  Distance between Person 10 and Person 23: 143 bits
  Distance between Person 10 and Person 24: 127 bits
  Distance between Person 10 and Person 25: 125 bits
  Distance between Person 10 and Person 26: 127 bits
  Distance between Person 10 and Person 27: 131 bits
  Distance between Person 10 and Person 28: 142 bits
  Distance between Person 10 and Person 29: 128 bits
  Distance between Person 10 and Person 30: 122 bits
  Distance between Person 10 and Person 31: 129 bits
  Distance between Person 10 and Person 32: 133 bits
  Distance between Person 10 and Person 33: 124 bits
  Distance between Person 10 and Person 34: 130 bits
  Distance between Person 10 and Person 35: 133 bits
  Distance between Person 10 and Person 36: 111 bits
  Distance between Person 10 and Person 37: 120 bits
  Distance between Person 10 and Person 38: 131 bits
  Distance between Person 10 and Person 39: 121 bits
  Distance between Person 10 and Person 40: 131 bits
  Distance between Person 10 and Person 41: 115 bits
  Distance between Person 10 and Person 42: 111 bits
  Distance between Person 10 and Person 43: 130 bits
  Distance between Person 10 and Person 44: 121 bits
  Distance between Person 10 and Person 45: 137 bits
  Distance between Person 10 and Person 46: 126 bits
  Distance between Person 10 and Person 47: 111 bits
  Distance between Person 10 and Person 48: 122 bits
  Distance between Person 10 and Person 49: 128 bits
  Distance between Person 10 and Person 50: 126 bits
  Distance between Person 10 and Person 51: 112 bits
  Distance between Person 10 and Person 52: 129 bits
  Distance between Person 10 and Person 53: 138 bits
  Distance between Person 10 and Person 54: 123 bits
  Distance between Person 10 and Person 55: 137 bits
  Distance between Person 10 and Person 56: 123 bits
  Distance between Person 10 and Person 57: 124 bits
  Distance between Person 10 and Person 58: 128 bits
  Distance between Person 10 and Person 59: 147 bits
  Distance between Person 10 and Person 60: 134 bits
  Distance between Person 10 and Person 61: 134 bits
  Distance between Person 10 and Person 62: 126 bits
  Distance between Person 10 and Person 63: 116 bits
  Distance between Person 10 and Person 64: 101 bits
  Distance between Person 10 and Person 65: 119 bits
  Distance between Person 10 and Person 66: 109 bits
  Distance between Person 10 and Person 67: 126 bits
  Distance between Person 10 and Person 68: 116 bits
  Distance between Person 10 and Person 69: 126 bits
  Distance between Person 10 and Person 70: 128 bits
  Distance between Person 10 and Person 71: 133 bits
  Distance between Person 10 and Person 72: 142 bits
  Distance between Person 10 and Person 73: 136 bits
  Distance between Person 10 and Person 74: 126 bits
  Distance between Person 10 and Person 75: 95 bits
  Distance between Person 10 and Person 76: 117 bits
  Distance between Person 10 and Person 77: 128 bits
  Distance between Person 10 and Person 78: 130 bits
  Distance between Person 10 and Person 79: 103 bits
  Distance between Person 10 and Person 80: 138 bits
  Distance between Person 10 and Person 81: 120 bits
  Distance between Person 10 and Person 82: 93 bits
  Distance between Person 10 and Person 83: 142 bits
  Distance between Person 10 and Person 84: 120 bits
  Distance between Person 10 and Person 85: 141 bits
  Distance between Person 10 and Person 86: 146 bits
  Distance between Person 10 and Person 87: 134 bits
  Distance between Person 10 and Person 88: 125 bits
  Distance between Person 10 and Person 89: 130 bits
  Distance between Person 11 and Person 12: 127 bits
  Distance between Person 11 and Person 13: 134 bits
  Distance between Person 11 and Person 14: 130 bits
  Distance between Person 11 and Person 15: 112 bits
  Distance between Person 11 and Person 16: 125 bits
  Distance between Person 11 and Person 17: 114 bits
  Distance between Person 11 and Person 18: 125 bits
  Distance between Person 11 and Person 19: 119 bits
  Distance between Person 11 and Person 20: 121 bits
  Distance between Person 11 and Person 21: 122 bits
  Distance between Person 11 and Person 22: 136 bits
  Distance between Person 11 and Person 23: 126 bits
  Distance between Person 11 and Person 24: 118 bits
  Distance between Person 11 and Person 25: 120 bits
  Distance between Person 11 and Person 26: 142 bits
  Distance between Person 11 and Person 27: 146 bits
  Distance between Person 11 and Person 28: 109 bits
  Distance between Person 11 and Person 29: 123 bits
  Distance between Person 11 and Person 30: 131 bits
  Distance between Person 11 and Person 31: 116 bits
  Distance between Person 11 and Person 32: 122 bits
  Distance between Person 11 and Person 33: 137 bits
  Distance between Person 11 and Person 34: 137 bits
  Distance between Person 11 and Person 35: 124 bits
  Distance between Person 11 and Person 36: 124 bits
  Distance between Person 11 and Person 37: 125 bits
  Distance between Person 11 and Person 38: 116 bits
  Distance between Person 11 and Person 39: 110 bits
  Distance between Person 11 and Person 40: 116 bits
  Distance between Person 11 and Person 41: 132 bits
  Distance between Person 11 and Person 42: 116 bits
  Distance between Person 11 and Person 43: 147 bits
  Distance between Person 11 and Person 44: 144 bits
  Distance between Person 11 and Person 45: 120 bits
  Distance between Person 11 and Person 46: 117 bits
  Distance between Person 11 and Person 47: 134 bits
  Distance between Person 11 and Person 48: 135 bits
  Distance between Person 11 and Person 49: 135 bits
  Distance between Person 11 and Person 50: 129 bits
  Distance between Person 11 and Person 51: 159 bits
  Distance between Person 11 and Person 52: 150 bits
  Distance between Person 11 and Person 53: 135 bits
  Distance between Person 11 and Person 54: 134 bits
  Distance between Person 11 and Person 55: 128 bits
  Distance between Person 11 and Person 56: 126 bits
  Distance between Person 11 and Person 57: 107 bits
  Distance between Person 11 and Person 58: 127 bits
  Distance between Person 11 and Person 59: 118 bits
  Distance between Person 11 and Person 60: 123 bits
  Distance between Person 11 and Person 61: 95 bits
  Distance between Person 11 and Person 62: 137 bits
  Distance between Person 11 and Person 63: 135 bits
  Distance between Person 11 and Person 64: 134 bits
  Distance between Person 11 and Person 65: 108 bits
  Distance between Person 11 and Person 66: 140 bits
  Distance between Person 11 and Person 67: 129 bits
  Distance between Person 11 and Person 68: 109 bits
  Distance between Person 11 and Person 69: 141 bits
  Distance between Person 11 and Person 70: 137 bits
  Distance between Person 11 and Person 71: 104 bits
  Distance between Person 11 and Person 72: 123 bits
  Distance between Person 11 and Person 73: 111 bits
  Distance between Person 11 and Person 74: 135 bits
  Distance between Person 11 and Person 75: 146 bits
  Distance between Person 11 and Person 76: 126 bits
  Distance between Person 11 and Person 77: 115 bits
  Distance between Person 11 and Person 78: 111 bits
  Distance between Person 11 and Person 79: 134 bits
  Distance between Person 11 and Person 80: 131 bits
  Distance between Person 11 and Person 81: 135 bits
  Distance between Person 11 and Person 82: 134 bits
  Distance between Person 11 and Person 83: 145 bits
  Distance between Person 11 and Person 84: 119 bits
  Distance between Person 11 and Person 85: 140 bits
  Distance between Person 11 and Person 86: 111 bits
  Distance between Person 11 and Person 87: 123 bits
  Distance between Person 11 and Person 88: 120 bits
  Distance between Person 11 and Person 89: 131 bits
  Distance between Person 12 and Person 13: 151 bits
  Distance between Person 12 and Person 14: 133 bits
  Distance between Person 12 and Person 15: 129 bits
  Distance between Person 12 and Person 16: 120 bits
  Distance between Person 12 and Person 17: 105 bits
  Distance between Person 12 and Person 18: 120 bits
  Distance between Person 12 and Person 19: 126 bits
  Distance between Person 12 and Person 20: 128 bits
  Distance between Person 12 and Person 21: 139 bits
  Distance between Person 12 and Person 22: 127 bits
  Distance between Person 12 and Person 23: 133 bits
  Distance between Person 12 and Person 24: 141 bits
  Distance between Person 12 and Person 25: 131 bits
  Distance between Person 12 and Person 26: 127 bits
  Distance between Person 12 and Person 27: 133 bits
  Distance between Person 12 and Person 28: 120 bits
  Distance between Person 12 and Person 29: 136 bits
  Distance between Person 12 and Person 30: 126 bits
  Distance between Person 12 and Person 31: 125 bits
  Distance between Person 12 and Person 32: 133 bits
  Distance between Person 12 and Person 33: 144 bits
  Distance between Person 12 and Person 34: 138 bits
  Distance between Person 12 and Person 35: 131 bits
  Distance between Person 12 and Person 36: 123 bits
  Distance between Person 12 and Person 37: 144 bits
  Distance between Person 12 and Person 38: 117 bits
  Distance between Person 12 and Person 39: 147 bits
  Distance between Person 12 and Person 40: 127 bits
  Distance between Person 12 and Person 41: 131 bits
  Distance between Person 12 and Person 42: 121 bits
  Distance between Person 12 and Person 43: 134 bits
  Distance between Person 12 and Person 44: 149 bits
  Distance between Person 12 and Person 45: 135 bits
  Distance between Person 12 and Person 46: 120 bits
  Distance between Person 12 and Person 47: 141 bits
  Distance between Person 12 and Person 48: 120 bits
  Distance between Person 12 and Person 49: 134 bits
  Distance between Person 12 and Person 50: 136 bits
  Distance between Person 12 and Person 51: 116 bits
  Distance between Person 12 and Person 52: 137 bits
  Distance between Person 12 and Person 53: 122 bits
  Distance between Person 12 and Person 54: 135 bits
  Distance between Person 12 and Person 55: 131 bits
  Distance between Person 12 and Person 56: 125 bits
  Distance between Person 12 and Person 57: 114 bits
  Distance between Person 12 and Person 58: 118 bits
  Distance between Person 12 and Person 59: 129 bits
  Distance between Person 12 and Person 60: 134 bits
  Distance between Person 12 and Person 61: 114 bits
  Distance between Person 12 and Person 62: 124 bits
  Distance between Person 12 and Person 63: 132 bits
  Distance between Person 12 and Person 64: 125 bits
  Distance between Person 12 and Person 65: 139 bits
  Distance between Person 12 and Person 66: 147 bits
  Distance between Person 12 and Person 67: 144 bits
  Distance between Person 12 and Person 68: 136 bits
  Distance between Person 12 and Person 69: 140 bits
  Distance between Person 12 and Person 70: 126 bits
  Distance between Person 12 and Person 71: 137 bits
  Distance between Person 12 and Person 72: 124 bits
  Distance between Person 12 and Person 73: 116 bits
  Distance between Person 12 and Person 74: 146 bits
  Distance between Person 12 and Person 75: 131 bits
  Distance between Person 12 and Person 76: 129 bits
  Distance between Person 12 and Person 77: 126 bits
  Distance between Person 12 and Person 78: 122 bits
  Distance between Person 12 and Person 79: 123 bits
  Distance between Person 12 and Person 80: 122 bits
  Distance between Person 12 and Person 81: 118 bits
  Distance between Person 12 and Person 82: 141 bits
  Distance between Person 12 and Person 83: 138 bits
  Distance between Person 12 and Person 84: 114 bits
  Distance between Person 12 and Person 85: 117 bits
  Distance between Person 12 and Person 86: 124 bits
  Distance between Person 12 and Person 87: 118 bits
  Distance between Person 12 and Person 88: 117 bits
  Distance between Person 12 and Person 89: 118 bits
  Distance between Person 13 and Person 14: 116 bits
  Distance between Person 13 and Person 15: 122 bits
  Distance between Person 13 and Person 16: 117 bits
  Distance between Person 13 and Person 17: 128 bits
  Distance between Person 13 and Person 18: 133 bits
  Distance between Person 13 and Person 19: 141 bits
  Distance between Person 13 and Person 20: 111 bits
  Distance between Person 13 and Person 21: 110 bits
  Distance between Person 13 and Person 22: 112 bits
  Distance between Person 13 and Person 23: 142 bits
  Distance between Person 13 and Person 24: 120 bits
  Distance between Person 13 and Person 25: 118 bits
  Distance between Person 13 and Person 26: 110 bits
  Distance between Person 13 and Person 27: 128 bits
  Distance between Person 13 and Person 28: 127 bits
  Distance between Person 13 and Person 29: 155 bits
  Distance between Person 13 and Person 30: 139 bits
  Distance between Person 13 and Person 31: 132 bits
  Distance between Person 13 and Person 32: 96 bits
  Distance between Person 13 and Person 33: 125 bits
  Distance between Person 13 and Person 34: 135 bits
  Distance between Person 13 and Person 35: 112 bits
  Distance between Person 13 and Person 36: 122 bits
  Distance between Person 13 and Person 37: 115 bits
  Distance between Person 13 and Person 38: 116 bits
  Distance between Person 13 and Person 39: 108 bits
  Distance between Person 13 and Person 40: 148 bits
  Distance between Person 13 and Person 41: 142 bits
  Distance between Person 13 and Person 42: 142 bits
  Distance between Person 13 and Person 43: 127 bits
  Distance between Person 13 and Person 44: 110 bits
  Distance between Person 13 and Person 45: 106 bits
  Distance between Person 13 and Person 46: 147 bits
  Distance between Person 13 and Person 47: 136 bits
  Distance between Person 13 and Person 48: 135 bits
  Distance between Person 13 and Person 49: 121 bits
  Distance between Person 13 and Person 50: 133 bits
  Distance between Person 13 and Person 51: 119 bits
  Distance between Person 13 and Person 52: 114 bits
  Distance between Person 13 and Person 53: 131 bits
  Distance between Person 13 and Person 54: 112 bits
  Distance between Person 13 and Person 55: 128 bits
  Distance between Person 13 and Person 56: 114 bits
  Distance between Person 13 and Person 57: 153 bits
  Distance between Person 13 and Person 58: 141 bits
  Distance between Person 13 and Person 59: 132 bits
  Distance between Person 13 and Person 60: 115 bits
  Distance between Person 13 and Person 61: 137 bits
  Distance between Person 13 and Person 62: 123 bits
  Distance between Person 13 and Person 63: 143 bits
  Distance between Person 13 and Person 64: 106 bits
  Distance between Person 13 and Person 65: 128 bits
  Distance between Person 13 and Person 66: 116 bits
  Distance between Person 13 and Person 67: 131 bits
  Distance between Person 13 and Person 68: 121 bits
  Distance between Person 13 and Person 69: 133 bits
  Distance between Person 13 and Person 70: 101 bits
  Distance between Person 13 and Person 71: 146 bits
  Distance between Person 13 and Person 72: 113 bits
  Distance between Person 13 and Person 73: 121 bits
  Distance between Person 13 and Person 74: 107 bits
  Distance between Person 13 and Person 75: 138 bits
  Distance between Person 13 and Person 76: 138 bits
  Distance between Person 13 and Person 77: 139 bits
  Distance between Person 13 and Person 78: 115 bits
  Distance between Person 13 and Person 79: 126 bits
  Distance between Person 13 and Person 80: 97 bits
  Distance between Person 13 and Person 81: 145 bits
  Distance between Person 13 and Person 82: 136 bits
  Distance between Person 13 and Person 83: 135 bits
  Distance between Person 13 and Person 84: 135 bits
  Distance between Person 13 and Person 85: 130 bits
  Distance between Person 13 and Person 86: 107 bits
  Distance between Person 13 and Person 87: 135 bits
  Distance between Person 13 and Person 88: 140 bits
  Distance between Person 13 and Person 89: 155 bits
  Distance between Person 14 and Person 15: 116 bits
  Distance between Person 14 and Person 16: 99 bits
  Distance between Person 14 and Person 17: 122 bits
  Distance between Person 14 and Person 18: 149 bits
  Distance between Person 14 and Person 19: 111 bits
  Distance between Person 14 and Person 20: 119 bits
  Distance between Person 14 and Person 21: 124 bits
  Distance between Person 14 and Person 22: 130 bits
  Distance between Person 14 and Person 23: 146 bits
  Distance between Person 14 and Person 24: 138 bits
  Distance between Person 14 and Person 25: 110 bits
  Distance between Person 14 and Person 26: 132 bits
  Distance between Person 14 and Person 27: 114 bits
  Distance between Person 14 and Person 28: 131 bits
  Distance between Person 14 and Person 29: 131 bits
  Distance between Person 14 and Person 30: 125 bits
  Distance between Person 14 and Person 31: 128 bits
  Distance between Person 14 and Person 32: 140 bits
  Distance between Person 14 and Person 33: 129 bits
  Distance between Person 14 and Person 34: 119 bits
  Distance between Person 14 and Person 35: 120 bits
  Distance between Person 14 and Person 36: 152 bits
  Distance between Person 14 and Person 37: 121 bits
  Distance between Person 14 and Person 38: 98 bits
  Distance between Person 14 and Person 39: 142 bits
  Distance between Person 14 and Person 40: 110 bits
  Distance between Person 14 and Person 41: 140 bits
  Distance between Person 14 and Person 42: 136 bits
  Distance between Person 14 and Person 43: 115 bits
  Distance between Person 14 and Person 44: 120 bits
  Distance between Person 14 and Person 45: 114 bits
  Distance between Person 14 and Person 46: 119 bits
  Distance between Person 14 and Person 47: 118 bits
  Distance between Person 14 and Person 48: 151 bits
  Distance between Person 14 and Person 49: 129 bits
  Distance between Person 14 and Person 50: 121 bits
  Distance between Person 14 and Person 51: 113 bits
  Distance between Person 14 and Person 52: 100 bits
  Distance between Person 14 and Person 53: 133 bits
  Distance between Person 14 and Person 54: 134 bits
  Distance between Person 14 and Person 55: 100 bits
  Distance between Person 14 and Person 56: 124 bits
  Distance between Person 14 and Person 57: 129 bits
  Distance between Person 14 and Person 58: 113 bits
  Distance between Person 14 and Person 59: 132 bits
  Distance between Person 14 and Person 60: 123 bits
  Distance between Person 14 and Person 61: 113 bits
  Distance between Person 14 and Person 62: 149 bits
  Distance between Person 14 and Person 63: 133 bits
  Distance between Person 14 and Person 64: 144 bits
  Distance between Person 14 and Person 65: 118 bits
  Distance between Person 14 and Person 66: 114 bits
  Distance between Person 14 and Person 67: 129 bits
  Distance between Person 14 and Person 68: 107 bits
  Distance between Person 14 and Person 69: 127 bits
  Distance between Person 14 and Person 70: 121 bits
  Distance between Person 14 and Person 71: 122 bits
  Distance between Person 14 and Person 72: 125 bits
  Distance between Person 14 and Person 73: 117 bits
  Distance between Person 14 and Person 74: 123 bits
  Distance between Person 14 and Person 75: 132 bits
  Distance between Person 14 and Person 76: 118 bits
  Distance between Person 14 and Person 77: 121 bits
  Distance between Person 14 and Person 78: 115 bits
  Distance between Person 14 and Person 79: 134 bits
  Distance between Person 14 and Person 80: 83 bits
  Distance between Person 14 and Person 81: 145 bits
  Distance between Person 14 and Person 82: 114 bits
  Distance between Person 14 and Person 83: 119 bits
  Distance between Person 14 and Person 84: 127 bits
  Distance between Person 14 and Person 85: 122 bits
  Distance between Person 14 and Person 86: 107 bits
  Distance between Person 14 and Person 87: 137 bits
  Distance between Person 14 and Person 88: 124 bits
  Distance between Person 14 and Person 89: 109 bits
  Distance between Person 15 and Person 16: 131 bits
  Distance between Person 15 and Person 17: 116 bits
  Distance between Person 15 and Person 18: 131 bits
  Distance between Person 15 and Person 19: 131 bits
  Distance between Person 15 and Person 20: 135 bits
  Distance between Person 15 and Person 21: 126 bits
  Distance between Person 15 and Person 22: 144 bits
  Distance between Person 15 and Person 23: 122 bits
  Distance between Person 15 and Person 24: 132 bits
  Distance between Person 15 and Person 25: 134 bits
  Distance between Person 15 and Person 26: 134 bits
  Distance between Person 15 and Person 27: 162 bits
  Distance between Person 15 and Person 28: 131 bits
  Distance between Person 15 and Person 29: 125 bits
  Distance between Person 15 and Person 30: 137 bits
  Distance between Person 15 and Person 31: 130 bits
  Distance between Person 15 and Person 32: 136 bits
  Distance between Person 15 and Person 33: 153 bits
  Distance between Person 15 and Person 34: 135 bits
  Distance between Person 15 and Person 35: 132 bits
  Distance between Person 15 and Person 36: 114 bits
  Distance between Person 15 and Person 37: 119 bits
  Distance between Person 15 and Person 38: 110 bits
  Distance between Person 15 and Person 39: 130 bits
  Distance between Person 15 and Person 40: 128 bits
  Distance between Person 15 and Person 41: 102 bits
  Distance between Person 15 and Person 42: 144 bits
  Distance between Person 15 and Person 43: 127 bits
  Distance between Person 15 and Person 44: 122 bits
  Distance between Person 15 and Person 45: 110 bits
  Distance between Person 15 and Person 46: 135 bits
  Distance between Person 15 and Person 47: 130 bits
  Distance between Person 15 and Person 48: 123 bits
  Distance between Person 15 and Person 49: 111 bits
  Distance between Person 15 and Person 50: 123 bits
  Distance between Person 15 and Person 51: 125 bits
  Distance between Person 15 and Person 52: 122 bits
  Distance between Person 15 and Person 53: 147 bits
  Distance between Person 15 and Person 54: 124 bits
  Distance between Person 15 and Person 55: 118 bits
  Distance between Person 15 and Person 56: 132 bits
  Distance between Person 15 and Person 57: 135 bits
  Distance between Person 15 and Person 58: 123 bits
  Distance between Person 15 and Person 59: 140 bits
  Distance between Person 15 and Person 60: 107 bits
  Distance between Person 15 and Person 61: 141 bits
  Distance between Person 15 and Person 62: 131 bits
  Distance between Person 15 and Person 63: 129 bits
  Distance between Person 15 and Person 64: 114 bits
  Distance between Person 15 and Person 65: 130 bits
  Distance between Person 15 and Person 66: 140 bits
  Distance between Person 15 and Person 67: 159 bits
  Distance between Person 15 and Person 68: 83 bits
  Distance between Person 15 and Person 69: 147 bits
  Distance between Person 15 and Person 70: 133 bits
  Distance between Person 15 and Person 71: 132 bits
  Distance between Person 15 and Person 72: 117 bits
  Distance between Person 15 and Person 73: 123 bits
  Distance between Person 15 and Person 74: 123 bits
  Distance between Person 15 and Person 75: 122 bits
  Distance between Person 15 and Person 76: 136 bits
  Distance between Person 15 and Person 77: 129 bits
  Distance between Person 15 and Person 78: 141 bits
  Distance between Person 15 and Person 79: 128 bits
  Distance between Person 15 and Person 80: 103 bits
  Distance between Person 15 and Person 81: 135 bits
  Distance between Person 15 and Person 82: 138 bits
  Distance between Person 15 and Person 83: 121 bits
  Distance between Person 15 and Person 84: 127 bits
  Distance between Person 15 and Person 85: 118 bits
  Distance between Person 15 and Person 86: 139 bits
  Distance between Person 15 and Person 87: 113 bits
  Distance between Person 15 and Person 88: 144 bits
  Distance between Person 15 and Person 89: 127 bits
  Distance between Person 16 and Person 17: 125 bits
  Distance between Person 16 and Person 18: 124 bits
  Distance between Person 16 and Person 19: 112 bits
  Distance between Person 16 and Person 20: 138 bits
  Distance between Person 16 and Person 21: 137 bits
  Distance between Person 16 and Person 22: 141 bits
  Distance between Person 16 and Person 23: 153 bits
  Distance between Person 16 and Person 24: 111 bits
  Distance between Person 16 and Person 25: 99 bits
  Distance between Person 16 and Person 26: 123 bits
  Distance between Person 16 and Person 27: 101 bits
  Distance between Person 16 and Person 28: 134 bits
  Distance between Person 16 and Person 29: 138 bits
  Distance between Person 16 and Person 30: 148 bits
  Distance between Person 16 and Person 31: 135 bits
  Distance between Person 16 and Person 32: 117 bits
  Distance between Person 16 and Person 33: 130 bits
  Distance between Person 16 and Person 34: 118 bits
  Distance between Person 16 and Person 35: 125 bits
  Distance between Person 16 and Person 36: 125 bits
  Distance between Person 16 and Person 37: 122 bits
  Distance between Person 16 and Person 38: 141 bits
  Distance between Person 16 and Person 39: 121 bits
  Distance between Person 16 and Person 40: 113 bits
  Distance between Person 16 and Person 41: 105 bits
  Distance between Person 16 and Person 42: 121 bits
  Distance between Person 16 and Person 43: 124 bits
  Distance between Person 16 and Person 44: 137 bits
  Distance between Person 16 and Person 45: 115 bits
  Distance between Person 16 and Person 46: 118 bits
  Distance between Person 16 and Person 47: 135 bits
  Distance between Person 16 and Person 48: 130 bits
  Distance between Person 16 and Person 49: 116 bits
  Distance between Person 16 and Person 50: 134 bits
  Distance between Person 16 and Person 51: 120 bits
  Distance between Person 16 and Person 52: 101 bits
  Distance between Person 16 and Person 53: 138 bits
  Distance between Person 16 and Person 54: 141 bits
  Distance between Person 16 and Person 55: 111 bits
  Distance between Person 16 and Person 56: 109 bits
  Distance between Person 16 and Person 57: 120 bits
  Distance between Person 16 and Person 58: 112 bits
  Distance between Person 16 and Person 59: 117 bits
  Distance between Person 16 and Person 60: 132 bits
  Distance between Person 16 and Person 61: 112 bits
  Distance between Person 16 and Person 62: 116 bits
  Distance between Person 16 and Person 63: 134 bits
  Distance between Person 16 and Person 64: 137 bits
  Distance between Person 16 and Person 65: 117 bits
  Distance between Person 16 and Person 66: 123 bits
  Distance between Person 16 and Person 67: 124 bits
  Distance between Person 16 and Person 68: 112 bits
  Distance between Person 16 and Person 69: 130 bits
  Distance between Person 16 and Person 70: 136 bits
  Distance between Person 16 and Person 71: 127 bits
  Distance between Person 16 and Person 72: 128 bits
  Distance between Person 16 and Person 73: 138 bits
  Distance between Person 16 and Person 74: 124 bits
  Distance between Person 16 and Person 75: 139 bits
  Distance between Person 16 and Person 76: 119 bits
  Distance between Person 16 and Person 77: 124 bits
  Distance between Person 16 and Person 78: 112 bits
  Distance between Person 16 and Person 79: 103 bits
  Distance between Person 16 and Person 80: 96 bits
  Distance between Person 16 and Person 81: 132 bits
  Distance between Person 16 and Person 82: 137 bits
  Distance between Person 16 and Person 83: 128 bits
  Distance between Person 16 and Person 84: 64 bits
  Distance between Person 16 and Person 85: 133 bits
  Distance between Person 16 and Person 86: 108 bits
  Distance between Person 16 and Person 87: 146 bits
  Distance between Person 16 and Person 88: 125 bits
  Distance between Person 16 and Person 89: 136 bits
  Distance between Person 17 and Person 18: 123 bits
  Distance between Person 17 and Person 19: 135 bits
  Distance between Person 17 and Person 20: 123 bits
  Distance between Person 17 and Person 21: 142 bits
  Distance between Person 17 and Person 22: 138 bits
  Distance between Person 17 and Person 23: 120 bits
  Distance between Person 17 and Person 24: 130 bits
  Distance between Person 17 and Person 25: 122 bits
  Distance between Person 17 and Person 26: 136 bits
  Distance between Person 17 and Person 27: 116 bits
  Distance between Person 17 and Person 28: 143 bits
  Distance between Person 17 and Person 29: 147 bits
  Distance between Person 17 and Person 30: 99 bits
  Distance between Person 17 and Person 31: 94 bits
  Distance between Person 17 and Person 32: 128 bits
  Distance between Person 17 and Person 33: 127 bits
  Distance between Person 17 and Person 34: 125 bits
  Distance between Person 17 and Person 35: 134 bits
  Distance between Person 17 and Person 36: 118 bits
  Distance between Person 17 and Person 37: 121 bits
  Distance between Person 17 and Person 38: 112 bits
  Distance between Person 17 and Person 39: 138 bits
  Distance between Person 17 and Person 40: 116 bits
  Distance between Person 17 and Person 41: 144 bits
  Distance between Person 17 and Person 42: 122 bits
  Distance between Person 17 and Person 43: 139 bits
  Distance between Person 17 and Person 44: 148 bits
  Distance between Person 17 and Person 45: 114 bits
  Distance between Person 17 and Person 46: 135 bits
  Distance between Person 17 and Person 47: 144 bits
  Distance between Person 17 and Person 48: 125 bits
  Distance between Person 17 and Person 49: 109 bits
  Distance between Person 17 and Person 50: 133 bits
  Distance between Person 17 and Person 51: 127 bits
  Distance between Person 17 and Person 52: 134 bits
  Distance between Person 17 and Person 53: 143 bits
  Distance between Person 17 and Person 54: 164 bits
  Distance between Person 17 and Person 55: 138 bits
  Distance between Person 17 and Person 56: 138 bits
  Distance between Person 17 and Person 57: 129 bits
  Distance between Person 17 and Person 58: 141 bits
  Distance between Person 17 and Person 59: 140 bits
  Distance between Person 17 and Person 60: 143 bits
  Distance between Person 17 and Person 61: 129 bits
  Distance between Person 17 and Person 62: 123 bits
  Distance between Person 17 and Person 63: 137 bits
  Distance between Person 17 and Person 64: 128 bits
  Distance between Person 17 and Person 65: 118 bits
  Distance between Person 17 and Person 66: 140 bits
  Distance between Person 17 and Person 67: 125 bits
  Distance between Person 17 and Person 68: 121 bits
  Distance between Person 17 and Person 69: 135 bits
  Distance between Person 17 and Person 70: 127 bits
  Distance between Person 17 and Person 71: 116 bits
  Distance between Person 17 and Person 72: 127 bits
  Distance between Person 17 and Person 73: 127 bits
  Distance between Person 17 and Person 74: 129 bits
  Distance between Person 17 and Person 75: 140 bits
  Distance between Person 17 and Person 76: 154 bits
  Distance between Person 17 and Person 77: 123 bits
  Distance between Person 17 and Person 78: 111 bits
  Distance between Person 17 and Person 79: 132 bits
  Distance between Person 17 and Person 80: 119 bits
  Distance between Person 17 and Person 81: 127 bits
  Distance between Person 17 and Person 82: 140 bits
  Distance between Person 17 and Person 83: 137 bits
  Distance between Person 17 and Person 84: 127 bits
  Distance between Person 17 and Person 85: 136 bits
  Distance between Person 17 and Person 86: 115 bits
  Distance between Person 17 and Person 87: 119 bits
  Distance between Person 17 and Person 88: 134 bits
  Distance between Person 17 and Person 89: 139 bits
  Distance between Person 18 and Person 19: 130 bits
  Distance between Person 18 and Person 20: 134 bits
  Distance between Person 18 and Person 21: 119 bits
  Distance between Person 18 and Person 22: 129 bits
  Distance between Person 18 and Person 23: 123 bits
  Distance between Person 18 and Person 24: 149 bits
  Distance between Person 18 and Person 25: 119 bits
  Distance between Person 18 and Person 26: 115 bits
  Distance between Person 18 and Person 27: 129 bits
  Distance between Person 18 and Person 28: 136 bits
  Distance between Person 18 and Person 29: 114 bits
  Distance between Person 18 and Person 30: 144 bits
  Distance between Person 18 and Person 31: 135 bits
  Distance between Person 18 and Person 32: 141 bits
  Distance between Person 18 and Person 33: 130 bits
  Distance between Person 18 and Person 34: 130 bits
  Distance between Person 18 and Person 35: 135 bits
  Distance between Person 18 and Person 36: 109 bits
  Distance between Person 18 and Person 37: 128 bits
  Distance between Person 18 and Person 38: 131 bits
  Distance between Person 18 and Person 39: 151 bits
  Distance between Person 18 and Person 40: 103 bits
  Distance between Person 18 and Person 41: 115 bits
  Distance between Person 18 and Person 42: 109 bits
  Distance between Person 18 and Person 43: 126 bits
  Distance between Person 18 and Person 44: 145 bits
  Distance between Person 18 and Person 45: 127 bits
  Distance between Person 18 and Person 46: 120 bits
  Distance between Person 18 and Person 47: 147 bits
  Distance between Person 18 and Person 48: 126 bits
  Distance between Person 18 and Person 49: 134 bits
  Distance between Person 18 and Person 50: 138 bits
  Distance between Person 18 and Person 51: 122 bits
  Distance between Person 18 and Person 52: 147 bits
  Distance between Person 18 and Person 53: 140 bits
  Distance between Person 18 and Person 54: 115 bits
  Distance between Person 18 and Person 55: 131 bits
  Distance between Person 18 and Person 56: 97 bits
  Distance between Person 18 and Person 57: 120 bits
  Distance between Person 18 and Person 58: 138 bits
  Distance between Person 18 and Person 59: 117 bits
  Distance between Person 18 and Person 60: 108 bits
  Distance between Person 18 and Person 61: 140 bits
  Distance between Person 18 and Person 62: 118 bits
  Distance between Person 18 and Person 63: 116 bits
  Distance between Person 18 and Person 64: 119 bits
  Distance between Person 18 and Person 65: 121 bits
  Distance between Person 18 and Person 66: 133 bits
  Distance between Person 18 and Person 67: 132 bits
  Distance between Person 18 and Person 68: 124 bits
  Distance between Person 18 and Person 69: 132 bits
  Distance between Person 18 and Person 70: 130 bits
  Distance between Person 18 and Person 71: 99 bits
  Distance between Person 18 and Person 72: 162 bits
  Distance between Person 18 and Person 73: 144 bits
  Distance between Person 18 and Person 74: 140 bits
  Distance between Person 18 and Person 75: 127 bits
  Distance between Person 18 and Person 76: 145 bits
  Distance between Person 18 and Person 77: 118 bits
  Distance between Person 18 and Person 78: 98 bits
  Distance between Person 18 and Person 79: 113 bits
  Distance between Person 18 and Person 80: 136 bits
  Distance between Person 18 and Person 81: 112 bits
  Distance between Person 18 and Person 82: 113 bits
  Distance between Person 18 and Person 83: 136 bits
  Distance between Person 18 and Person 84: 120 bits
  Distance between Person 18 and Person 85: 131 bits
  Distance between Person 18 and Person 86: 142 bits
  Distance between Person 18 and Person 87: 156 bits
  Distance between Person 18 and Person 88: 105 bits
  Distance between Person 18 and Person 89: 156 bits
  Distance between Person 19 and Person 20: 102 bits
  Distance between Person 19 and Person 21: 139 bits
  Distance between Person 19 and Person 22: 141 bits
  Distance between Person 19 and Person 23: 135 bits
  Distance between Person 19 and Person 24: 123 bits
  Distance between Person 19 and Person 25: 139 bits
  Distance between Person 19 and Person 26: 127 bits
  Distance between Person 19 and Person 27: 109 bits
  Distance between Person 19 and Person 28: 134 bits
  Distance between Person 19 and Person 29: 100 bits
  Distance between Person 19 and Person 30: 136 bits
  Distance between Person 19 and Person 31: 133 bits
  Distance between Person 19 and Person 32: 123 bits
  Distance between Person 19 and Person 33: 132 bits
  Distance between Person 19 and Person 34: 134 bits
  Distance between Person 19 and Person 35: 121 bits
  Distance between Person 19 and Person 36: 141 bits
  Distance between Person 19 and Person 37: 120 bits
  Distance between Person 19 and Person 38: 135 bits
  Distance between Person 19 and Person 39: 113 bits
  Distance between Person 19 and Person 40: 97 bits
  Distance between Person 19 and Person 41: 117 bits
  Distance between Person 19 and Person 42: 121 bits
  Distance between Person 19 and Person 43: 122 bits
  Distance between Person 19 and Person 44: 141 bits
  Distance between Person 19 and Person 45: 125 bits
  Distance between Person 19 and Person 46: 150 bits
  Distance between Person 19 and Person 47: 101 bits
  Distance between Person 19 and Person 48: 118 bits
  Distance between Person 19 and Person 49: 118 bits
  Distance between Person 19 and Person 50: 120 bits
  Distance between Person 19 and Person 51: 132 bits
  Distance between Person 19 and Person 52: 119 bits
  Distance between Person 19 and Person 53: 148 bits
  Distance between Person 19 and Person 54: 135 bits
  Distance between Person 19 and Person 55: 91 bits
  Distance between Person 19 and Person 56: 129 bits
  Distance between Person 19 and Person 57: 122 bits
  Distance between Person 19 and Person 58: 116 bits
  Distance between Person 19 and Person 59: 129 bits
  Distance between Person 19 and Person 60: 114 bits
  Distance between Person 19 and Person 61: 104 bits
  Distance between Person 19 and Person 62: 134 bits
  Distance between Person 19 and Person 63: 148 bits
  Distance between Person 19 and Person 64: 119 bits
  Distance between Person 19 and Person 65: 131 bits
  Distance between Person 19 and Person 66: 107 bits
  Distance between Person 19 and Person 67: 126 bits
  Distance between Person 19 and Person 68: 132 bits
  Distance between Person 19 and Person 69: 114 bits
  Distance between Person 19 and Person 70: 144 bits
  Distance between Person 19 and Person 71: 101 bits
  Distance between Person 19 and Person 72: 128 bits
  Distance between Person 19 and Person 73: 134 bits
  Distance between Person 19 and Person 74: 140 bits
  Distance between Person 19 and Person 75: 119 bits
  Distance between Person 19 and Person 76: 133 bits
  Distance between Person 19 and Person 77: 114 bits
  Distance between Person 19 and Person 78: 112 bits
  Distance between Person 19 and Person 79: 123 bits
  Distance between Person 19 and Person 80: 108 bits
  Distance between Person 19 and Person 81: 130 bits
  Distance between Person 19 and Person 82: 123 bits
  Distance between Person 19 and Person 83: 128 bits
  Distance between Person 19 and Person 84: 138 bits
  Distance between Person 19 and Person 85: 115 bits
  Distance between Person 19 and Person 86: 100 bits
  Distance between Person 19 and Person 87: 140 bits
  Distance between Person 19 and Person 88: 143 bits
  Distance between Person 19 and Person 89: 126 bits
  Distance between Person 20 and Person 21: 125 bits
  Distance between Person 20 and Person 22: 115 bits
  Distance between Person 20 and Person 23: 115 bits
  Distance between Person 20 and Person 24: 139 bits
  Distance between Person 20 and Person 25: 129 bits
  Distance between Person 20 and Person 26: 131 bits
  Distance between Person 20 and Person 27: 119 bits
  Distance between Person 20 and Person 28: 130 bits
  Distance between Person 20 and Person 29: 128 bits
  Distance between Person 20 and Person 30: 128 bits
  Distance between Person 20 and Person 31: 125 bits
  Distance between Person 20 and Person 32: 117 bits
  Distance between Person 20 and Person 33: 128 bits
  Distance between Person 20 and Person 34: 140 bits
  Distance between Person 20 and Person 35: 125 bits
  Distance between Person 20 and Person 36: 121 bits
  Distance between Person 20 and Person 37: 124 bits
  Distance between Person 20 and Person 38: 129 bits
  Distance between Person 20 and Person 39: 119 bits
  Distance between Person 20 and Person 40: 129 bits
  Distance between Person 20 and Person 41: 115 bits
  Distance between Person 20 and Person 42: 113 bits
  Distance between Person 20 and Person 43: 130 bits
  Distance between Person 20 and Person 44: 125 bits
  Distance between Person 20 and Person 45: 103 bits
  Distance between Person 20 and Person 46: 124 bits
  Distance between Person 20 and Person 47: 111 bits
  Distance between Person 20 and Person 48: 146 bits
  Distance between Person 20 and Person 49: 122 bits
  Distance between Person 20 and Person 50: 122 bits
  Distance between Person 20 and Person 51: 122 bits
  Distance between Person 20 and Person 52: 147 bits
  Distance between Person 20 and Person 53: 136 bits
  Distance between Person 20 and Person 54: 123 bits
  Distance between Person 20 and Person 55: 135 bits
  Distance between Person 20 and Person 56: 119 bits
  Distance between Person 20 and Person 57: 132 bits
  Distance between Person 20 and Person 58: 124 bits
  Distance between Person 20 and Person 59: 123 bits
  Distance between Person 20 and Person 60: 128 bits
  Distance between Person 20 and Person 61: 120 bits
  Distance between Person 20 and Person 62: 138 bits
  Distance between Person 20 and Person 63: 134 bits
  Distance between Person 20 and Person 64: 133 bits
  Distance between Person 20 and Person 65: 137 bits
  Distance between Person 20 and Person 66: 133 bits
  Distance between Person 20 and Person 67: 122 bits
  Distance between Person 20 and Person 68: 130 bits
  Distance between Person 20 and Person 69: 126 bits
  Distance between Person 20 and Person 70: 130 bits
  Distance between Person 20 and Person 71: 145 bits
  Distance between Person 20 and Person 72: 120 bits
  Distance between Person 20 and Person 73: 110 bits
  Distance between Person 20 and Person 74: 124 bits
  Distance between Person 20 and Person 75: 123 bits
  Distance between Person 20 and Person 76: 127 bits
  Distance between Person 20 and Person 77: 130 bits
  Distance between Person 20 and Person 78: 98 bits
  Distance between Person 20 and Person 79: 131 bits
  Distance between Person 20 and Person 80: 126 bits
  Distance between Person 20 and Person 81: 132 bits
  Distance between Person 20 and Person 82: 99 bits
  Distance between Person 20 and Person 83: 134 bits
  Distance between Person 20 and Person 84: 146 bits
  Distance between Person 20 and Person 85: 141 bits
  Distance between Person 20 and Person 86: 120 bits
  Distance between Person 20 and Person 87: 122 bits
  Distance between Person 20 and Person 88: 131 bits
  Distance between Person 20 and Person 89: 114 bits
  Distance between Person 21 and Person 22: 140 bits
  Distance between Person 21 and Person 23: 124 bits
  Distance between Person 21 and Person 24: 122 bits
  Distance between Person 21 and Person 25: 130 bits
  Distance between Person 21 and Person 26: 118 bits
  Distance between Person 21 and Person 27: 130 bits
  Distance between Person 21 and Person 28: 107 bits
  Distance between Person 21 and Person 29: 121 bits
  Distance between Person 21 and Person 30: 155 bits
  Distance between Person 21 and Person 31: 130 bits
  Distance between Person 21 and Person 32: 112 bits
  Distance between Person 21 and Person 33: 127 bits
  Distance between Person 21 and Person 34: 113 bits
  Distance between Person 21 and Person 35: 144 bits
  Distance between Person 21 and Person 36: 132 bits
  Distance between Person 21 and Person 37: 119 bits
  Distance between Person 21 and Person 38: 86 bits
  Distance between Person 21 and Person 39: 120 bits
  Distance between Person 21 and Person 40: 112 bits
  Distance between Person 21 and Person 41: 132 bits
  Distance between Person 21 and Person 42: 126 bits
  Distance between Person 21 and Person 43: 131 bits
  Distance between Person 21 and Person 44: 108 bits
  Distance between Person 21 and Person 45: 130 bits
  Distance between Person 21 and Person 46: 111 bits
  Distance between Person 21 and Person 47: 130 bits
  Distance between Person 21 and Person 48: 125 bits
  Distance between Person 21 and Person 49: 137 bits
  Distance between Person 21 and Person 50: 125 bits
  Distance between Person 21 and Person 51: 123 bits
  Distance between Person 21 and Person 52: 140 bits
  Distance between Person 21 and Person 53: 123 bits
  Distance between Person 21 and Person 54: 100 bits
  Distance between Person 21 and Person 55: 126 bits
  Distance between Person 21 and Person 56: 130 bits
  Distance between Person 21 and Person 57: 121 bits
  Distance between Person 21 and Person 58: 125 bits
  Distance between Person 21 and Person 59: 120 bits
  Distance between Person 21 and Person 60: 111 bits
  Distance between Person 21 and Person 61: 117 bits
  Distance between Person 21 and Person 62: 97 bits
  Distance between Person 21 and Person 63: 139 bits
  Distance between Person 21 and Person 64: 112 bits
  Distance between Person 21 and Person 65: 112 bits
  Distance between Person 21 and Person 66: 128 bits
  Distance between Person 21 and Person 67: 135 bits
  Distance between Person 21 and Person 68: 137 bits
  Distance between Person 21 and Person 69: 135 bits
  Distance between Person 21 and Person 70: 39 bits
  Distance between Person 21 and Person 71: 132 bits
  Distance between Person 21 and Person 72: 123 bits
  Distance between Person 21 and Person 73: 127 bits
  Distance between Person 21 and Person 74: 133 bits
  Distance between Person 21 and Person 75: 106 bits
  Distance between Person 21 and Person 76: 120 bits
  Distance between Person 21 and Person 77: 117 bits
  Distance between Person 21 and Person 78: 117 bits
  Distance between Person 21 and Person 79: 140 bits
  Distance between Person 21 and Person 80: 113 bits
  Distance between Person 21 and Person 81: 155 bits
  Distance between Person 21 and Person 82: 122 bits
  Distance between Person 21 and Person 83: 133 bits
  Distance between Person 21 and Person 84: 129 bits
  Distance between Person 21 and Person 85: 130 bits
  Distance between Person 21 and Person 86: 119 bits
  Distance between Person 21 and Person 87: 139 bits
  Distance between Person 21 and Person 88: 108 bits
  Distance between Person 21 and Person 89: 133 bits
  Distance between Person 22 and Person 23: 124 bits
  Distance between Person 22 and Person 24: 122 bits
  Distance between Person 22 and Person 25: 134 bits
  Distance between Person 22 and Person 26: 116 bits
  Distance between Person 22 and Person 27: 120 bits
  Distance between Person 22 and Person 28: 129 bits
  Distance between Person 22 and Person 29: 125 bits
  Distance between Person 22 and Person 30: 107 bits
  Distance between Person 22 and Person 31: 132 bits
  Distance between Person 22 and Person 32: 112 bits
  Distance between Person 22 and Person 33: 149 bits
  Distance between Person 22 and Person 34: 125 bits
  Distance between Person 22 and Person 35: 132 bits
  Distance between Person 22 and Person 36: 106 bits
  Distance between Person 22 and Person 37: 155 bits
  Distance between Person 22 and Person 38: 152 bits
  Distance between Person 22 and Person 39: 120 bits
  Distance between Person 22 and Person 40: 120 bits
  Distance between Person 22 and Person 41: 136 bits
  Distance between Person 22 and Person 42: 136 bits
  Distance between Person 22 and Person 43: 99 bits
  Distance between Person 22 and Person 44: 140 bits
  Distance between Person 22 and Person 45: 150 bits
  Distance between Person 22 and Person 46: 125 bits
  Distance between Person 22 and Person 47: 128 bits
  Distance between Person 22 and Person 48: 133 bits
  Distance between Person 22 and Person 49: 137 bits
  Distance between Person 22 and Person 50: 119 bits
  Distance between Person 22 and Person 51: 113 bits
  Distance between Person 22 and Person 52: 120 bits
  Distance between Person 22 and Person 53: 127 bits
  Distance between Person 22 and Person 54: 118 bits
  Distance between Person 22 and Person 55: 134 bits
  Distance between Person 22 and Person 56: 88 bits
  Distance between Person 22 and Person 57: 119 bits
  Distance between Person 22 and Person 58: 137 bits
  Distance between Person 22 and Person 59: 122 bits
  Distance between Person 22 and Person 60: 137 bits
  Distance between Person 22 and Person 61: 133 bits
  Distance between Person 22 and Person 62: 127 bits
  Distance between Person 22 and Person 63: 141 bits
  Distance between Person 22 and Person 64: 126 bits
  Distance between Person 22 and Person 65: 142 bits
  Distance between Person 22 and Person 66: 144 bits
  Distance between Person 22 and Person 67: 131 bits
  Distance between Person 22 and Person 68: 121 bits
  Distance between Person 22 and Person 69: 129 bits
  Distance between Person 22 and Person 70: 143 bits
  Distance between Person 22 and Person 71: 140 bits
  Distance between Person 22 and Person 72: 149 bits
  Distance between Person 22 and Person 73: 103 bits
  Distance between Person 22 and Person 74: 121 bits
  Distance between Person 22 and Person 75: 140 bits
  Distance between Person 22 and Person 76: 128 bits
  Distance between Person 22 and Person 77: 139 bits
  Distance between Person 22 and Person 78: 135 bits
  Distance between Person 22 and Person 79: 128 bits
  Distance between Person 22 and Person 80: 147 bits
  Distance between Person 22 and Person 81: 129 bits
  Distance between Person 22 and Person 82: 132 bits
  Distance between Person 22 and Person 83: 127 bits
  Distance between Person 22 and Person 84: 113 bits
  Distance between Person 22 and Person 85: 126 bits
  Distance between Person 22 and Person 86: 149 bits
  Distance between Person 22 and Person 87: 133 bits
  Distance between Person 22 and Person 88: 138 bits
  Distance between Person 22 and Person 89: 113 bits
  Distance between Person 23 and Person 24: 152 bits
  Distance between Person 23 and Person 25: 124 bits
  Distance between Person 23 and Person 26: 162 bits
  Distance between Person 23 and Person 27: 136 bits
  Distance between Person 23 and Person 28: 133 bits
  Distance between Person 23 and Person 29: 103 bits
  Distance between Person 23 and Person 30: 123 bits
  Distance between Person 23 and Person 31: 116 bits
  Distance between Person 23 and Person 32: 132 bits
  Distance between Person 23 and Person 33: 137 bits
  Distance between Person 23 and Person 34: 141 bits
  Distance between Person 23 and Person 35: 134 bits
  Distance between Person 23 and Person 36: 134 bits
  Distance between Person 23 and Person 37: 125 bits
  Distance between Person 23 and Person 38: 128 bits
  Distance between Person 23 and Person 39: 132 bits
  Distance between Person 23 and Person 40: 132 bits
  Distance between Person 23 and Person 41: 122 bits
  Distance between Person 23 and Person 42: 120 bits
  Distance between Person 23 and Person 43: 123 bits
  Distance between Person 23 and Person 44: 130 bits
  Distance between Person 23 and Person 45: 122 bits
  Distance between Person 23 and Person 46: 117 bits
  Distance between Person 23 and Person 47: 134 bits
  Distance between Person 23 and Person 48: 123 bits
  Distance between Person 23 and Person 49: 131 bits
  Distance between Person 23 and Person 50: 121 bits
  Distance between Person 23 and Person 51: 133 bits
  Distance between Person 23 and Person 52: 160 bits
  Distance between Person 23 and Person 53: 119 bits
  Distance between Person 23 and Person 54: 138 bits
  Distance between Person 23 and Person 55: 144 bits
  Distance between Person 23 and Person 56: 118 bits
  Distance between Person 23 and Person 57: 141 bits
  Distance between Person 23 and Person 58: 133 bits
  Distance between Person 23 and Person 59: 114 bits
  Distance between Person 23 and Person 60: 117 bits
  Distance between Person 23 and Person 61: 121 bits
  Distance between Person 23 and Person 62: 127 bits
  Distance between Person 23 and Person 63: 115 bits
  Distance between Person 23 and Person 64: 148 bits
  Distance between Person 23 and Person 65: 140 bits
  Distance between Person 23 and Person 66: 144 bits
  Distance between Person 23 and Person 67: 133 bits
  Distance between Person 23 and Person 68: 137 bits
  Distance between Person 23 and Person 69: 145 bits
  Distance between Person 23 and Person 70: 127 bits
  Distance between Person 23 and Person 71: 130 bits
  Distance between Person 23 and Person 72: 147 bits
  Distance between Person 23 and Person 73: 139 bits
  Distance between Person 23 and Person 74: 107 bits
  Distance between Person 23 and Person 75: 148 bits
  Distance between Person 23 and Person 76: 138 bits
  Distance between Person 23 and Person 77: 115 bits
  Distance between Person 23 and Person 78: 141 bits
  Distance between Person 23 and Person 79: 160 bits
  Distance between Person 23 and Person 80: 131 bits
  Distance between Person 23 and Person 81: 125 bits
  Distance between Person 23 and Person 82: 138 bits
  Distance between Person 23 and Person 83: 91 bits
  Distance between Person 23 and Person 84: 135 bits
  Distance between Person 23 and Person 85: 126 bits
  Distance between Person 23 and Person 86: 129 bits
  Distance between Person 23 and Person 87: 133 bits
  Distance between Person 23 and Person 88: 142 bits
  Distance between Person 23 and Person 89: 133 bits
  Distance between Person 24 and Person 25: 134 bits
  Distance between Person 24 and Person 26: 106 bits
  Distance between Person 24 and Person 27: 128 bits
  Distance between Person 24 and Person 28: 111 bits
  Distance between Person 24 and Person 29: 131 bits
  Distance between Person 24 and Person 30: 127 bits
  Distance between Person 24 and Person 31: 136 bits
  Distance between Person 24 and Person 32: 100 bits
  Distance between Person 24 and Person 33: 123 bits
  Distance between Person 24 and Person 34: 117 bits
  Distance between Person 24 and Person 35: 140 bits
  Distance between Person 24 and Person 36: 120 bits
  Distance between Person 24 and Person 37: 129 bits
  Distance between Person 24 and Person 38: 120 bits
  Distance between Person 24 and Person 39: 82 bits
  Distance between Person 24 and Person 40: 98 bits
  Distance between Person 24 and Person 41: 124 bits
  Distance between Person 24 and Person 42: 120 bits
  Distance between Person 24 and Person 43: 105 bits
  Distance between Person 24 and Person 44: 128 bits
  Distance between Person 24 and Person 45: 110 bits
  Distance between Person 24 and Person 46: 141 bits
  Distance between Person 24 and Person 47: 104 bits
  Distance between Person 24 and Person 48: 99 bits
  Distance between Person 24 and Person 49: 129 bits
  Distance between Person 24 and Person 50: 135 bits
  Distance between Person 24 and Person 51: 121 bits
  Distance between Person 24 and Person 52: 102 bits
  Distance between Person 24 and Person 53: 129 bits
  Distance between Person 24 and Person 54: 150 bits
  Distance between Person 24 and Person 55: 108 bits
  Distance between Person 24 and Person 56: 116 bits
  Distance between Person 24 and Person 57: 107 bits
  Distance between Person 24 and Person 58: 135 bits
  Distance between Person 24 and Person 59: 120 bits
  Distance between Person 24 and Person 60: 151 bits
  Distance between Person 24 and Person 61: 127 bits
  Distance between Person 24 and Person 62: 117 bits
  Distance between Person 24 and Person 63: 129 bits
  Distance between Person 24 and Person 64: 114 bits
  Distance between Person 24 and Person 65: 118 bits
  Distance between Person 24 and Person 66: 102 bits
  Distance between Person 24 and Person 67: 117 bits
  Distance between Person 24 and Person 68: 137 bits
  Distance between Person 24 and Person 69: 143 bits
  Distance between Person 24 and Person 70: 123 bits
  Distance between Person 24 and Person 71: 136 bits
  Distance between Person 24 and Person 72: 113 bits
  Distance between Person 24 and Person 73: 127 bits
  Distance between Person 24 and Person 74: 137 bits
  Distance between Person 24 and Person 75: 130 bits
  Distance between Person 24 and Person 76: 130 bits
  Distance between Person 24 and Person 77: 111 bits
  Distance between Person 24 and Person 78: 131 bits
  Distance between Person 24 and Person 79: 112 bits
  Distance between Person 24 and Person 80: 133 bits
  Distance between Person 24 and Person 81: 145 bits
  Distance between Person 24 and Person 82: 124 bits
  Distance between Person 24 and Person 83: 127 bits
  Distance between Person 24 and Person 84: 123 bits
  Distance between Person 24 and Person 85: 132 bits
  Distance between Person 24 and Person 86: 117 bits
  Distance between Person 24 and Person 87: 113 bits
  Distance between Person 24 and Person 88: 146 bits
  Distance between Person 24 and Person 89: 131 bits
  Distance between Person 25 and Person 26: 128 bits
  Distance between Person 25 and Person 27: 120 bits
  Distance between Person 25 and Person 28: 147 bits
  Distance between Person 25 and Person 29: 143 bits
  Distance between Person 25 and Person 30: 143 bits
  Distance between Person 25 and Person 31: 104 bits
  Distance between Person 25 and Person 32: 124 bits
  Distance between Person 25 and Person 33: 87 bits
  Distance between Person 25 and Person 34: 125 bits
  Distance between Person 25 and Person 35: 124 bits
  Distance between Person 25 and Person 36: 140 bits
  Distance between Person 25 and Person 37: 143 bits
  Distance between Person 25 and Person 38: 140 bits
  Distance between Person 25 and Person 39: 138 bits
  Distance between Person 25 and Person 40: 136 bits
  Distance between Person 25 and Person 41: 120 bits
  Distance between Person 25 and Person 42: 104 bits
  Distance between Person 25 and Person 43: 129 bits
  Distance between Person 25 and Person 44: 114 bits
  Distance between Person 25 and Person 45: 108 bits
  Distance between Person 25 and Person 46: 109 bits
  Distance between Person 25 and Person 47: 120 bits
  Distance between Person 25 and Person 48: 145 bits
  Distance between Person 25 and Person 49: 143 bits
  Distance between Person 25 and Person 50: 131 bits
  Distance between Person 25 and Person 51: 139 bits
  Distance between Person 25 and Person 52: 140 bits
  Distance between Person 25 and Person 53: 123 bits
  Distance between Person 25 and Person 54: 118 bits
  Distance between Person 25 and Person 55: 138 bits
  Distance between Person 25 and Person 56: 90 bits
  Distance between Person 25 and Person 57: 131 bits
  Distance between Person 25 and Person 58: 97 bits
  Distance between Person 25 and Person 59: 114 bits
  Distance between Person 25 and Person 60: 133 bits
  Distance between Person 25 and Person 61: 149 bits
  Distance between Person 25 and Person 62: 121 bits
  Distance between Person 25 and Person 63: 127 bits
  Distance between Person 25 and Person 64: 144 bits
  Distance between Person 25 and Person 65: 128 bits
  Distance between Person 25 and Person 66: 140 bits
  Distance between Person 25 and Person 67: 81 bits
  Distance between Person 25 and Person 68: 135 bits
  Distance between Person 25 and Person 69: 133 bits
  Distance between Person 25 and Person 70: 129 bits
  Distance between Person 25 and Person 71: 136 bits
  Distance between Person 25 and Person 72: 141 bits
  Distance between Person 25 and Person 73: 119 bits
  Distance between Person 25 and Person 74: 137 bits
  Distance between Person 25 and Person 75: 132 bits
  Distance between Person 25 and Person 76: 124 bits
  Distance between Person 25 and Person 77: 123 bits
  Distance between Person 25 and Person 78: 121 bits
  Distance between Person 25 and Person 79: 146 bits
  Distance between Person 25 and Person 80: 115 bits
  Distance between Person 25 and Person 81: 121 bits
  Distance between Person 25 and Person 82: 122 bits
  Distance between Person 25 and Person 83: 145 bits
  Distance between Person 25 and Person 84: 101 bits
  Distance between Person 25 and Person 85: 120 bits
  Distance between Person 25 and Person 86: 127 bits
  Distance between Person 25 and Person 87: 147 bits
  Distance between Person 25 and Person 88: 138 bits
  Distance between Person 25 and Person 89: 139 bits
  Distance between Person 26 and Person 27: 118 bits
  Distance between Person 26 and Person 28: 125 bits
  Distance between Person 26 and Person 29: 133 bits
  Distance between Person 26 and Person 30: 117 bits
  Distance between Person 26 and Person 31: 118 bits
  Distance between Person 26 and Person 32: 128 bits
  Distance between Person 26 and Person 33: 115 bits
  Distance between Person 26 and Person 34: 105 bits
  Distance between Person 26 and Person 35: 142 bits
  Distance between Person 26 and Person 36: 98 bits
  Distance between Person 26 and Person 37: 135 bits
  Distance between Person 26 and Person 38: 122 bits
  Distance between Person 26 and Person 39: 128 bits
  Distance between Person 26 and Person 40: 108 bits
  Distance between Person 26 and Person 41: 124 bits
  Distance between Person 26 and Person 42: 132 bits
  Distance between Person 26 and Person 43: 91 bits
  Distance between Person 26 and Person 44: 104 bits
  Distance between Person 26 and Person 45: 128 bits
  Distance between Person 26 and Person 46: 135 bits
  Distance between Person 26 and Person 47: 126 bits
  Distance between Person 26 and Person 48: 129 bits
  Distance between Person 26 and Person 49: 131 bits
  Distance between Person 26 and Person 50: 123 bits
  Distance between Person 26 and Person 51: 103 bits
  Distance between Person 26 and Person 52: 104 bits
  Distance between Person 26 and Person 53: 135 bits
  Distance between Person 26 and Person 54: 110 bits
  Distance between Person 26 and Person 55: 108 bits
  Distance between Person 26 and Person 56: 100 bits
  Distance between Person 26 and Person 57: 135 bits
  Distance between Person 26 and Person 58: 129 bits
  Distance between Person 26 and Person 59: 118 bits
  Distance between Person 26 and Person 60: 137 bits
  Distance between Person 26 and Person 61: 133 bits
  Distance between Person 26 and Person 62: 113 bits
  Distance between Person 26 and Person 63: 145 bits
  Distance between Person 26 and Person 64: 96 bits
  Distance between Person 26 and Person 65: 138 bits
  Distance between Person 26 and Person 66: 142 bits
  Distance between Person 26 and Person 67: 123 bits
  Distance between Person 26 and Person 68: 135 bits
  Distance between Person 26 and Person 69: 111 bits
  Distance between Person 26 and Person 70: 117 bits
  Distance between Person 26 and Person 71: 132 bits
  Distance between Person 26 and Person 72: 133 bits
  Distance between Person 26 and Person 73: 141 bits
  Distance between Person 26 and Person 74: 131 bits
  Distance between Person 26 and Person 75: 128 bits
  Distance between Person 26 and Person 76: 128 bits
  Distance between Person 26 and Person 77: 127 bits
  Distance between Person 26 and Person 78: 125 bits
  Distance between Person 26 and Person 79: 120 bits
  Distance between Person 26 and Person 80: 125 bits
  Distance between Person 26 and Person 81: 141 bits
  Distance between Person 26 and Person 82: 114 bits
  Distance between Person 26 and Person 83: 157 bits
  Distance between Person 26 and Person 84: 127 bits
  Distance between Person 26 and Person 85: 130 bits
  Distance between Person 26 and Person 86: 141 bits
  Distance between Person 26 and Person 87: 131 bits
  Distance between Person 26 and Person 88: 118 bits
  Distance between Person 26 and Person 89: 127 bits
  Distance between Person 27 and Person 28: 129 bits
  Distance between Person 27 and Person 29: 133 bits
  Distance between Person 27 and Person 30: 129 bits
  Distance between Person 27 and Person 31: 148 bits
  Distance between Person 27 and Person 32: 122 bits
  Distance between Person 27 and Person 33: 103 bits
  Distance between Person 27 and Person 34: 115 bits
  Distance between Person 27 and Person 35: 118 bits
  Distance between Person 27 and Person 36: 130 bits
  Distance between Person 27 and Person 37: 129 bits
  Distance between Person 27 and Person 38: 152 bits
  Distance between Person 27 and Person 39: 124 bits
  Distance between Person 27 and Person 40: 106 bits
  Distance between Person 27 and Person 41: 130 bits
  Distance between Person 27 and Person 42: 112 bits
  Distance between Person 27 and Person 43: 111 bits
  Distance between Person 27 and Person 44: 138 bits
  Distance between Person 27 and Person 45: 132 bits
  Distance between Person 27 and Person 46: 125 bits
  Distance between Person 27 and Person 47: 130 bits
  Distance between Person 27 and Person 48: 127 bits
  Distance between Person 27 and Person 49: 141 bits
  Distance between Person 27 and Person 50: 139 bits
  Distance between Person 27 and Person 51: 123 bits
  Distance between Person 27 and Person 52: 126 bits
  Distance between Person 27 and Person 53: 149 bits
  Distance between Person 27 and Person 54: 144 bits
  Distance between Person 27 and Person 55: 132 bits
  Distance between Person 27 and Person 56: 124 bits
  Distance between Person 27 and Person 57: 137 bits
  Distance between Person 27 and Person 58: 139 bits
  Distance between Person 27 and Person 59: 96 bits
  Distance between Person 27 and Person 60: 125 bits
  Distance between Person 27 and Person 61: 137 bits
  Distance between Person 27 and Person 62: 117 bits
  Distance between Person 27 and Person 63: 127 bits
  Distance between Person 27 and Person 64: 130 bits
  Distance between Person 27 and Person 65: 118 bits
  Distance between Person 27 and Person 66: 120 bits
  Distance between Person 27 and Person 67: 101 bits
  Distance between Person 27 and Person 68: 103 bits
  Distance between Person 27 and Person 69: 135 bits
  Distance between Person 27 and Person 70: 129 bits
  Distance between Person 27 and Person 71: 138 bits
  Distance between Person 27 and Person 72: 121 bits
  Distance between Person 27 and Person 73: 135 bits
  Distance between Person 27 and Person 74: 149 bits
  Distance between Person 27 and Person 75: 132 bits
  Distance between Person 27 and Person 76: 136 bits
  Distance between Person 27 and Person 77: 129 bits
  Distance between Person 27 and Person 78: 113 bits
  Distance between Person 27 and Person 79: 142 bits
  Distance between Person 27 and Person 80: 133 bits
  Distance between Person 27 and Person 81: 109 bits
  Distance between Person 27 and Person 82: 126 bits
  Distance between Person 27 and Person 83: 131 bits
  Distance between Person 27 and Person 84: 117 bits
  Distance between Person 27 and Person 85: 130 bits
  Distance between Person 27 and Person 86: 119 bits
  Distance between Person 27 and Person 87: 131 bits
  Distance between Person 27 and Person 88: 134 bits
  Distance between Person 27 and Person 89: 139 bits
  Distance between Person 28 and Person 29: 132 bits
  Distance between Person 28 and Person 30: 158 bits
  Distance between Person 28 and Person 31: 133 bits
  Distance between Person 28 and Person 32: 121 bits
  Distance between Person 28 and Person 33: 138 bits
  Distance between Person 28 and Person 34: 132 bits
  Distance between Person 28 and Person 35: 139 bits
  Distance between Person 28 and Person 36: 137 bits
  Distance between Person 28 and Person 37: 110 bits
  Distance between Person 28 and Person 38: 121 bits
  Distance between Person 28 and Person 39: 103 bits
  Distance between Person 28 and Person 40: 119 bits
  Distance between Person 28 and Person 41: 129 bits
  Distance between Person 28 and Person 42: 135 bits
  Distance between Person 28 and Person 43: 114 bits
  Distance between Person 28 and Person 44: 139 bits
  Distance between Person 28 and Person 45: 119 bits
  Distance between Person 28 and Person 46: 114 bits
  Distance between Person 28 and Person 47: 125 bits
  Distance between Person 28 and Person 48: 132 bits
  Distance between Person 28 and Person 49: 132 bits
  Distance between Person 28 and Person 50: 150 bits
  Distance between Person 28 and Person 51: 122 bits
  Distance between Person 28 and Person 52: 125 bits
  Distance between Person 28 and Person 53: 124 bits
  Distance between Person 28 and Person 54: 111 bits
  Distance between Person 28 and Person 55: 125 bits
  Distance between Person 28 and Person 56: 129 bits
  Distance between Person 28 and Person 57: 144 bits
  Distance between Person 28 and Person 58: 136 bits
  Distance between Person 28 and Person 59: 123 bits
  Distance between Person 28 and Person 60: 138 bits
  Distance between Person 28 and Person 61: 74 bits
  Distance between Person 28 and Person 62: 130 bits
  Distance between Person 28 and Person 63: 112 bits
  Distance between Person 28 and Person 64: 123 bits
  Distance between Person 28 and Person 65: 141 bits
  Distance between Person 28 and Person 66: 137 bits
  Distance between Person 28 and Person 67: 136 bits
  Distance between Person 28 and Person 68: 120 bits
  Distance between Person 28 and Person 69: 136 bits
  Distance between Person 28 and Person 70: 108 bits
  Distance between Person 28 and Person 71: 123 bits
  Distance between Person 28 and Person 72: 124 bits
  Distance between Person 28 and Person 73: 128 bits
  Distance between Person 28 and Person 74: 122 bits
  Distance between Person 28 and Person 75: 127 bits
  Distance between Person 28 and Person 76: 119 bits
  Distance between Person 28 and Person 77: 116 bits
  Distance between Person 28 and Person 78: 126 bits
  Distance between Person 28 and Person 79: 151 bits
  Distance between Person 28 and Person 80: 126 bits
  Distance between Person 28 and Person 81: 144 bits
  Distance between Person 28 and Person 82: 137 bits
  Distance between Person 28 and Person 83: 136 bits
  Distance between Person 28 and Person 84: 126 bits
  Distance between Person 28 and Person 85: 115 bits
  Distance between Person 28 and Person 86: 92 bits
  Distance between Person 28 and Person 87: 128 bits
  Distance between Person 28 and Person 88: 113 bits
  Distance between Person 28 and Person 89: 118 bits
  Distance between Person 29 and Person 30: 128 bits
  Distance between Person 29 and Person 31: 137 bits
  Distance between Person 29 and Person 32: 125 bits
  Distance between Person 29 and Person 33: 128 bits
  Distance between Person 29 and Person 34: 124 bits
  Distance between Person 29 and Person 35: 133 bits
  Distance between Person 29 and Person 36: 143 bits
  Distance between Person 29 and Person 37: 84 bits
  Distance between Person 29 and Person 38: 147 bits
  Distance between Person 29 and Person 39: 117 bits
  Distance between Person 29 and Person 40: 105 bits
  Distance between Person 29 and Person 41: 115 bits
  Distance between Person 29 and Person 42: 129 bits
  Distance between Person 29 and Person 43: 122 bits
  Distance between Person 29 and Person 44: 121 bits
  Distance between Person 29 and Person 45: 131 bits
  Distance between Person 29 and Person 46: 120 bits
  Distance between Person 29 and Person 47: 113 bits
  Distance between Person 29 and Person 48: 132 bits
  Distance between Person 29 and Person 49: 144 bits
  Distance between Person 29 and Person 50: 96 bits
  Distance between Person 29 and Person 51: 114 bits
  Distance between Person 29 and Person 52: 129 bits
  Distance between Person 29 and Person 53: 126 bits
  Distance between Person 29 and Person 54: 127 bits
  Distance between Person 29 and Person 55: 121 bits
  Distance between Person 29 and Person 56: 119 bits
  Distance between Person 29 and Person 57: 146 bits
  Distance between Person 29 and Person 58: 118 bits
  Distance between Person 29 and Person 59: 127 bits
  Distance between Person 29 and Person 60: 124 bits
  Distance between Person 29 and Person 61: 110 bits
  Distance between Person 29 and Person 62: 144 bits
  Distance between Person 29 and Person 63: 130 bits
  Distance between Person 29 and Person 64: 135 bits
  Distance between Person 29 and Person 65: 131 bits
  Distance between Person 29 and Person 66: 123 bits
  Distance between Person 29 and Person 67: 128 bits
  Distance between Person 29 and Person 68: 130 bits
  Distance between Person 29 and Person 69: 124 bits
  Distance between Person 29 and Person 70: 142 bits
  Distance between Person 29 and Person 71: 115 bits
  Distance between Person 29 and Person 72: 144 bits
  Distance between Person 29 and Person 73: 124 bits
  Distance between Person 29 and Person 74: 132 bits
  Distance between Person 29 and Person 75: 111 bits
  Distance between Person 29 and Person 76: 129 bits
  Distance between Person 29 and Person 77: 106 bits
  Distance between Person 29 and Person 78: 136 bits
  Distance between Person 29 and Person 79: 117 bits
  Distance between Person 29 and Person 80: 122 bits
  Distance between Person 29 and Person 81: 142 bits
  Distance between Person 29 and Person 82: 105 bits
  Distance between Person 29 and Person 83: 128 bits
  Distance between Person 29 and Person 84: 124 bits
  Distance between Person 29 and Person 85: 127 bits
  Distance between Person 29 and Person 86: 140 bits
  Distance between Person 29 and Person 87: 134 bits
  Distance between Person 29 and Person 88: 125 bits
  Distance between Person 29 and Person 89: 116 bits
  Distance between Person 30 and Person 31: 121 bits
  Distance between Person 30 and Person 32: 137 bits
  Distance between Person 30 and Person 33: 136 bits
  Distance between Person 30 and Person 34: 134 bits
  Distance between Person 30 and Person 35: 123 bits
  Distance between Person 30 and Person 36: 113 bits
  Distance between Person 30 and Person 37: 132 bits
  Distance between Person 30 and Person 38: 133 bits
  Distance between Person 30 and Person 39: 143 bits
  Distance between Person 30 and Person 40: 139 bits
  Distance between Person 30 and Person 41: 151 bits
  Distance between Person 30 and Person 42: 135 bits
  Distance between Person 30 and Person 43: 114 bits
  Distance between Person 30 and Person 44: 137 bits
  Distance between Person 30 and Person 45: 145 bits
  Distance between Person 30 and Person 46: 130 bits
  Distance between Person 30 and Person 47: 117 bits
  Distance between Person 30 and Person 48: 120 bits
  Distance between Person 30 and Person 49: 110 bits
  Distance between Person 30 and Person 50: 104 bits
  Distance between Person 30 and Person 51: 118 bits
  Distance between Person 30 and Person 52: 129 bits
  Distance between Person 30 and Person 53: 142 bits
  Distance between Person 30 and Person 54: 155 bits
  Distance between Person 30 and Person 55: 133 bits
  Distance between Person 30 and Person 56: 123 bits
  Distance between Person 30 and Person 57: 134 bits
  Distance between Person 30 and Person 58: 124 bits
  Distance between Person 30 and Person 59: 131 bits
  Distance between Person 30 and Person 60: 148 bits
  Distance between Person 30 and Person 61: 130 bits
  Distance between Person 30 and Person 62: 126 bits
  Distance between Person 30 and Person 63: 146 bits
  Distance between Person 30 and Person 64: 139 bits
  Distance between Person 30 and Person 65: 133 bits
  Distance between Person 30 and Person 66: 139 bits
  Distance between Person 30 and Person 67: 138 bits
  Distance between Person 30 and Person 68: 122 bits
  Distance between Person 30 and Person 69: 104 bits
  Distance between Person 30 and Person 70: 138 bits
  Distance between Person 30 and Person 71: 111 bits
  Distance between Person 30 and Person 72: 124 bits
  Distance between Person 30 and Person 73: 126 bits
  Distance between Person 30 and Person 74: 128 bits
  Distance between Person 30 and Person 75: 135 bits
  Distance between Person 30 and Person 76: 133 bits
  Distance between Person 30 and Person 77: 134 bits
  Distance between Person 30 and Person 78: 144 bits
  Distance between Person 30 and Person 79: 117 bits
  Distance between Person 30 and Person 80: 142 bits
  Distance between Person 30 and Person 81: 118 bits
  Distance between Person 30 and Person 82: 131 bits
  Distance between Person 30 and Person 83: 116 bits
  Distance between Person 30 and Person 84: 130 bits
  Distance between Person 30 and Person 85: 123 bits
  Distance between Person 30 and Person 86: 146 bits
  Distance between Person 30 and Person 87: 94 bits
  Distance between Person 30 and Person 88: 147 bits
  Distance between Person 30 and Person 89: 122 bits
  Distance between Person 31 and Person 32: 146 bits
  Distance between Person 31 and Person 33: 113 bits
  Distance between Person 31 and Person 34: 125 bits
  Distance between Person 31 and Person 35: 142 bits
  Distance between Person 31 and Person 36: 128 bits
  Distance between Person 31 and Person 37: 131 bits
  Distance between Person 31 and Person 38: 112 bits
  Distance between Person 31 and Person 39: 154 bits
  Distance between Person 31 and Person 40: 132 bits
  Distance between Person 31 and Person 41: 138 bits
  Distance between Person 31 and Person 42: 146 bits
  Distance between Person 31 and Person 43: 137 bits
  Distance between Person 31 and Person 44: 108 bits
  Distance between Person 31 and Person 45: 116 bits
  Distance between Person 31 and Person 46: 117 bits
  Distance between Person 31 and Person 47: 136 bits
  Distance between Person 31 and Person 48: 127 bits
  Distance between Person 31 and Person 49: 115 bits
  Distance between Person 31 and Person 50: 127 bits
  Distance between Person 31 and Person 51: 133 bits
  Distance between Person 31 and Person 52: 146 bits
  Distance between Person 31 and Person 53: 117 bits
  Distance between Person 31 and Person 54: 120 bits
  Distance between Person 31 and Person 55: 128 bits
  Distance between Person 31 and Person 56: 132 bits
  Distance between Person 31 and Person 57: 137 bits
  Distance between Person 31 and Person 58: 117 bits
  Distance between Person 31 and Person 59: 156 bits
  Distance between Person 31 and Person 60: 139 bits
  Distance between Person 31 and Person 61: 123 bits
  Distance between Person 31 and Person 62: 121 bits
  Distance between Person 31 and Person 63: 143 bits
  Distance between Person 31 and Person 64: 136 bits
  Distance between Person 31 and Person 65: 116 bits
  Distance between Person 31 and Person 66: 154 bits
  Distance between Person 31 and Person 67: 127 bits
  Distance between Person 31 and Person 68: 145 bits
  Distance between Person 31 and Person 69: 107 bits
  Distance between Person 31 and Person 70: 117 bits
  Distance between Person 31 and Person 71: 128 bits
  Distance between Person 31 and Person 72: 115 bits
  Distance between Person 31 and Person 73: 115 bits
  Distance between Person 31 and Person 74: 121 bits
  Distance between Person 31 and Person 75: 134 bits
  Distance between Person 31 and Person 76: 120 bits
  Distance between Person 31 and Person 77: 133 bits
  Distance between Person 31 and Person 78: 119 bits
  Distance between Person 31 and Person 79: 140 bits
  Distance between Person 31 and Person 80: 131 bits
  Distance between Person 31 and Person 81: 137 bits
  Distance between Person 31 and Person 82: 124 bits
  Distance between Person 31 and Person 83: 143 bits
  Distance between Person 31 and Person 84: 133 bits
  Distance between Person 31 and Person 85: 132 bits
  Distance between Person 31 and Person 86: 131 bits
  Distance between Person 31 and Person 87: 105 bits
  Distance between Person 31 and Person 88: 108 bits
  Distance between Person 31 and Person 89: 107 bits
  Distance between Person 32 and Person 33: 139 bits
  Distance between Person 32 and Person 34: 123 bits
  Distance between Person 32 and Person 35: 130 bits
  Distance between Person 32 and Person 36: 128 bits
  Distance between Person 32 and Person 37: 115 bits
  Distance between Person 32 and Person 38: 128 bits
  Distance between Person 32 and Person 39: 48 bits
  Distance between Person 32 and Person 40: 108 bits
  Distance between Person 32 and Person 41: 148 bits
  Distance between Person 32 and Person 42: 134 bits
  Distance between Person 32 and Person 43: 107 bits
  Distance between Person 32 and Person 44: 138 bits
  Distance between Person 32 and Person 45: 114 bits
  Distance between Person 32 and Person 46: 137 bits
  Distance between Person 32 and Person 47: 112 bits
  Distance between Person 32 and Person 48: 123 bits
  Distance between Person 32 and Person 49: 137 bits
  Distance between Person 32 and Person 50: 113 bits
  Distance between Person 32 and Person 51: 131 bits
  Distance between Person 32 and Person 52: 134 bits
  Distance between Person 32 and Person 53: 137 bits
  Distance between Person 32 and Person 54: 130 bits
  Distance between Person 32 and Person 55: 148 bits
  Distance between Person 32 and Person 56: 112 bits
  Distance between Person 32 and Person 57: 133 bits
  Distance between Person 32 and Person 58: 123 bits
  Distance between Person 32 and Person 59: 120 bits
  Distance between Person 32 and Person 60: 143 bits
  Distance between Person 32 and Person 61: 107 bits
  Distance between Person 32 and Person 62: 99 bits
  Distance between Person 32 and Person 63: 143 bits
  Distance between Person 32 and Person 64: 98 bits
  Distance between Person 32 and Person 65: 138 bits
  Distance between Person 32 and Person 66: 116 bits
  Distance between Person 32 and Person 67: 143 bits
  Distance between Person 32 and Person 68: 125 bits
  Distance between Person 32 and Person 69: 141 bits
  Distance between Person 32 and Person 70: 123 bits
  Distance between Person 32 and Person 71: 118 bits
  Distance between Person 32 and Person 72: 105 bits
  Distance between Person 32 and Person 73: 109 bits
  Distance between Person 32 and Person 74: 119 bits
  Distance between Person 32 and Person 75: 136 bits
  Distance between Person 32 and Person 76: 124 bits
  Distance between Person 32 and Person 77: 137 bits
  Distance between Person 32 and Person 78: 123 bits
  Distance between Person 32 and Person 79: 140 bits
  Distance between Person 32 and Person 80: 117 bits
  Distance between Person 32 and Person 81: 143 bits
  Distance between Person 32 and Person 82: 120 bits
  Distance between Person 32 and Person 83: 129 bits
  Distance between Person 32 and Person 84: 113 bits
  Distance between Person 32 and Person 85: 142 bits
  Distance between Person 32 and Person 86: 93 bits
  Distance between Person 32 and Person 87: 119 bits
  Distance between Person 32 and Person 88: 142 bits
  Distance between Person 32 and Person 89: 149 bits
  Distance between Person 33 and Person 34: 110 bits
  Distance between Person 33 and Person 35: 135 bits
  Distance between Person 33 and Person 36: 149 bits
  Distance between Person 33 and Person 37: 128 bits
  Distance between Person 33 and Person 38: 147 bits
  Distance between Person 33 and Person 39: 133 bits
  Distance between Person 33 and Person 40: 139 bits
  Distance between Person 33 and Person 41: 131 bits
  Distance between Person 33 and Person 42: 117 bits
  Distance between Person 33 and Person 43: 124 bits
  Distance between Person 33 and Person 44: 91 bits
  Distance between Person 33 and Person 45: 131 bits
  Distance between Person 33 and Person 46: 130 bits
  Distance between Person 33 and Person 47: 125 bits
  Distance between Person 33 and Person 48: 144 bits
  Distance between Person 33 and Person 49: 130 bits
  Distance between Person 33 and Person 50: 130 bits
  Distance between Person 33 and Person 51: 130 bits
  Distance between Person 33 and Person 52: 135 bits
  Distance between Person 33 and Person 53: 110 bits
  Distance between Person 33 and Person 54: 123 bits
  Distance between Person 33 and Person 55: 125 bits
  Distance between Person 33 and Person 56: 129 bits
  Distance between Person 33 and Person 57: 124 bits
  Distance between Person 33 and Person 58: 128 bits
  Distance between Person 33 and Person 59: 119 bits
  Distance between Person 33 and Person 60: 128 bits
  Distance between Person 33 and Person 61: 160 bits
  Distance between Person 33 and Person 62: 140 bits
  Distance between Person 33 and Person 63: 110 bits
  Distance between Person 33 and Person 64: 129 bits
  Distance between Person 33 and Person 65: 125 bits
  Distance between Person 33 and Person 66: 111 bits
  Distance between Person 33 and Person 67: 36 bits
  Distance between Person 33 and Person 68: 154 bits
  Distance between Person 33 and Person 69: 108 bits
  Distance between Person 33 and Person 70: 128 bits
  Distance between Person 33 and Person 71: 139 bits
  Distance between Person 33 and Person 72: 126 bits
  Distance between Person 33 and Person 73: 120 bits
  Distance between Person 33 and Person 74: 132 bits
  Distance between Person 33 and Person 75: 137 bits
  Distance between Person 33 and Person 76: 125 bits
  Distance between Person 33 and Person 77: 132 bits
  Distance between Person 33 and Person 78: 122 bits
  Distance between Person 33 and Person 79: 139 bits
  Distance between Person 33 and Person 80: 130 bits
  Distance between Person 33 and Person 81: 114 bits
  Distance between Person 33 and Person 82: 121 bits
  Distance between Person 33 and Person 83: 140 bits
  Distance between Person 33 and Person 84: 142 bits
  Distance between Person 33 and Person 85: 127 bits
  Distance between Person 33 and Person 86: 128 bits
  Distance between Person 33 and Person 87: 132 bits
  Distance between Person 33 and Person 88: 115 bits
  Distance between Person 33 and Person 89: 134 bits
  Distance between Person 34 and Person 35: 141 bits
  Distance between Person 34 and Person 36: 145 bits
  Distance between Person 34 and Person 37: 104 bits
  Distance between Person 34 and Person 38: 125 bits
  Distance between Person 34 and Person 39: 117 bits
  Distance between Person 34 and Person 40: 79 bits
  Distance between Person 34 and Person 41: 117 bits
  Distance between Person 34 and Person 42: 119 bits
  Distance between Person 34 and Person 43: 112 bits
  Distance between Person 34 and Person 44: 129 bits
  Distance between Person 34 and Person 45: 119 bits
  Distance between Person 34 and Person 46: 108 bits
  Distance between Person 34 and Person 47: 131 bits
  Distance between Person 34 and Person 48: 116 bits
  Distance between Person 34 and Person 49: 128 bits
  Distance between Person 34 and Person 50: 124 bits
  Distance between Person 34 and Person 51: 110 bits
  Distance between Person 34 and Person 52: 105 bits
  Distance between Person 34 and Person 53: 110 bits
  Distance between Person 34 and Person 54: 127 bits
  Distance between Person 34 and Person 55: 119 bits
  Distance between Person 34 and Person 56: 133 bits
  Distance between Person 34 and Person 57: 124 bits
  Distance between Person 34 and Person 58: 128 bits
  Distance between Person 34 and Person 59: 125 bits
  Distance between Person 34 and Person 60: 132 bits
  Distance between Person 34 and Person 61: 128 bits
  Distance between Person 34 and Person 62: 120 bits
  Distance between Person 34 and Person 63: 130 bits
  Distance between Person 34 and Person 64: 129 bits
  Distance between Person 34 and Person 65: 137 bits
  Distance between Person 34 and Person 66: 145 bits
  Distance between Person 34 and Person 67: 114 bits
  Distance between Person 34 and Person 68: 138 bits
  Distance between Person 34 and Person 69: 134 bits
  Distance between Person 34 and Person 70: 118 bits
  Distance between Person 34 and Person 71: 135 bits
  Distance between Person 34 and Person 72: 136 bits
  Distance between Person 34 and Person 73: 112 bits
  Distance between Person 34 and Person 74: 104 bits
  Distance between Person 34 and Person 75: 145 bits
  Distance between Person 34 and Person 76: 107 bits
  Distance between Person 34 and Person 77: 136 bits
  Distance between Person 34 and Person 78: 126 bits
  Distance between Person 34 and Person 79: 131 bits
  Distance between Person 34 and Person 80: 122 bits
  Distance between Person 34 and Person 81: 132 bits
  Distance between Person 34 and Person 82: 125 bits
  Distance between Person 34 and Person 83: 150 bits
  Distance between Person 34 and Person 84: 120 bits
  Distance between Person 34 and Person 85: 131 bits
  Distance between Person 34 and Person 86: 136 bits
  Distance between Person 34 and Person 87: 152 bits
  Distance between Person 34 and Person 88: 129 bits
  Distance between Person 34 and Person 89: 124 bits
  Distance between Person 35 and Person 36: 118 bits
  Distance between Person 35 and Person 37: 117 bits
  Distance between Person 35 and Person 38: 108 bits
  Distance between Person 35 and Person 39: 112 bits
  Distance between Person 35 and Person 40: 130 bits
  Distance between Person 35 and Person 41: 140 bits
  Distance between Person 35 and Person 42: 128 bits
  Distance between Person 35 and Person 43: 131 bits
  Distance between Person 35 and Person 44: 132 bits
  Distance between Person 35 and Person 45: 132 bits
  Distance between Person 35 and Person 46: 145 bits
  Distance between Person 35 and Person 47: 124 bits
  Distance between Person 35 and Person 48: 121 bits
  Distance between Person 35 and Person 49: 115 bits
  Distance between Person 35 and Person 50: 111 bits
  Distance between Person 35 and Person 51: 143 bits
  Distance between Person 35 and Person 52: 122 bits
  Distance between Person 35 and Person 53: 119 bits
  Distance between Person 35 and Person 54: 120 bits
  Distance between Person 35 and Person 55: 124 bits
  Distance between Person 35 and Person 56: 126 bits
  Distance between Person 35 and Person 57: 119 bits
  Distance between Person 35 and Person 58: 145 bits
  Distance between Person 35 and Person 59: 122 bits
  Distance between Person 35 and Person 60: 113 bits
  Distance between Person 35 and Person 61: 129 bits
  Distance between Person 35 and Person 62: 145 bits
  Distance between Person 35 and Person 63: 143 bits
  Distance between Person 35 and Person 64: 122 bits
  Distance between Person 35 and Person 65: 130 bits
  Distance between Person 35 and Person 66: 126 bits
  Distance between Person 35 and Person 67: 125 bits
  Distance between Person 35 and Person 68: 123 bits
  Distance between Person 35 and Person 69: 125 bits
  Distance between Person 35 and Person 70: 127 bits
  Distance between Person 35 and Person 71: 122 bits
  Distance between Person 35 and Person 72: 121 bits
  Distance between Person 35 and Person 73: 139 bits
  Distance between Person 35 and Person 74: 131 bits
  Distance between Person 35 and Person 75: 134 bits
  Distance between Person 35 and Person 76: 122 bits
  Distance between Person 35 and Person 77: 107 bits
  Distance between Person 35 and Person 78: 141 bits
  Distance between Person 35 and Person 79: 120 bits
  Distance between Person 35 and Person 80: 127 bits
  Distance between Person 35 and Person 81: 99 bits
  Distance between Person 35 and Person 82: 140 bits
  Distance between Person 35 and Person 83: 135 bits
  Distance between Person 35 and Person 84: 135 bits
  Distance between Person 35 and Person 85: 114 bits
  Distance between Person 35 and Person 86: 121 bits
  Distance between Person 35 and Person 87: 113 bits
  Distance between Person 35 and Person 88: 118 bits
  Distance between Person 35 and Person 89: 143 bits
  Distance between Person 36 and Person 37: 129 bits
  Distance between Person 36 and Person 38: 140 bits
  Distance between Person 36 and Person 39: 128 bits
  Distance between Person 36 and Person 40: 128 bits
  Distance between Person 36 and Person 41: 142 bits
  Distance between Person 36 and Person 42: 130 bits
  Distance between Person 36 and Person 43: 117 bits
  Distance between Person 36 and Person 44: 134 bits
  Distance between Person 36 and Person 45: 144 bits
  Distance between Person 36 and Person 46: 117 bits
  Distance between Person 36 and Person 47: 144 bits
  Distance between Person 36 and Person 48: 125 bits
  Distance between Person 36 and Person 49: 117 bits
  Distance between Person 36 and Person 50: 115 bits
  Distance between Person 36 and Person 51: 123 bits
  Distance between Person 36 and Person 52: 124 bits
  Distance between Person 36 and Person 53: 137 bits
  Distance between Person 36 and Person 54: 128 bits
  Distance between Person 36 and Person 55: 132 bits
  Distance between Person 36 and Person 56: 116 bits
  Distance between Person 36 and Person 57: 125 bits
  Distance between Person 36 and Person 58: 133 bits
  Distance between Person 36 and Person 59: 124 bits
  Distance between Person 36 and Person 60: 125 bits
  Distance between Person 36 and Person 61: 143 bits
  Distance between Person 36 and Person 62: 131 bits
  Distance between Person 36 and Person 63: 135 bits
  Distance between Person 36 and Person 64: 124 bits
  Distance between Person 36 and Person 65: 104 bits
  Distance between Person 36 and Person 66: 148 bits
  Distance between Person 36 and Person 67: 147 bits
  Distance between Person 36 and Person 68: 121 bits
  Distance between Person 36 and Person 69: 149 bits
  Distance between Person 36 and Person 70: 135 bits
  Distance between Person 36 and Person 71: 124 bits
  Distance between Person 36 and Person 72: 125 bits
  Distance between Person 36 and Person 73: 133 bits
  Distance between Person 36 and Person 74: 147 bits
  Distance between Person 36 and Person 75: 122 bits
  Distance between Person 36 and Person 76: 118 bits
  Distance between Person 36 and Person 77: 123 bits
  Distance between Person 36 and Person 78: 135 bits
  Distance between Person 36 and Person 79: 114 bits
  Distance between Person 36 and Person 80: 141 bits
  Distance between Person 36 and Person 81: 135 bits
  Distance between Person 36 and Person 82: 128 bits
  Distance between Person 36 and Person 83: 149 bits
  Distance between Person 36 and Person 84: 117 bits
  Distance between Person 36 and Person 85: 130 bits
  Distance between Person 36 and Person 86: 147 bits
  Distance between Person 36 and Person 87: 109 bits
  Distance between Person 36 and Person 88: 116 bits
  Distance between Person 36 and Person 89: 147 bits
  Distance between Person 37 and Person 38: 137 bits
  Distance between Person 37 and Person 39: 91 bits
  Distance between Person 37 and Person 40: 125 bits
  Distance between Person 37 and Person 41: 131 bits
  Distance between Person 37 and Person 42: 133 bits
  Distance between Person 37 and Person 43: 142 bits
  Distance between Person 37 and Person 44: 117 bits
  Distance between Person 37 and Person 45: 111 bits
  Distance between Person 37 and Person 46: 112 bits
  Distance between Person 37 and Person 47: 135 bits
  Distance between Person 37 and Person 48: 126 bits
  Distance between Person 37 and Person 49: 118 bits
  Distance between Person 37 and Person 50: 122 bits
  Distance between Person 37 and Person 51: 108 bits
  Distance between Person 37 and Person 52: 109 bits
  Distance between Person 37 and Person 53: 128 bits
  Distance between Person 37 and Person 54: 131 bits
  Distance between Person 37 and Person 55: 125 bits
  Distance between Person 37 and Person 56: 133 bits
  Distance between Person 37 and Person 57: 164 bits
  Distance between Person 37 and Person 58: 132 bits
  Distance between Person 37 and Person 59: 145 bits
  Distance between Person 37 and Person 60: 134 bits
  Distance between Person 37 and Person 61: 110 bits
  Distance between Person 37 and Person 62: 136 bits
  Distance between Person 37 and Person 63: 118 bits
  Distance between Person 37 and Person 64: 117 bits
  Distance between Person 37 and Person 65: 123 bits
  Distance between Person 37 and Person 66: 133 bits
  Distance between Person 37 and Person 67: 138 bits
  Distance between Person 37 and Person 68: 126 bits
  Distance between Person 37 and Person 69: 126 bits
  Distance between Person 37 and Person 70: 132 bits
  Distance between Person 37 and Person 71: 115 bits
  Distance between Person 37 and Person 72: 122 bits
  Distance between Person 37 and Person 73: 122 bits
  Distance between Person 37 and Person 74: 114 bits
  Distance between Person 37 and Person 75: 133 bits
  Distance between Person 37 and Person 76: 121 bits
  Distance between Person 37 and Person 77: 98 bits
  Distance between Person 37 and Person 78: 132 bits
  Distance between Person 37 and Person 79: 129 bits
  Distance between Person 37 and Person 80: 98 bits
  Distance between Person 37 and Person 81: 144 bits
  Distance between Person 37 and Person 82: 111 bits
  Distance between Person 37 and Person 83: 134 bits
  Distance between Person 37 and Person 84: 132 bits
  Distance between Person 37 and Person 85: 149 bits
  Distance between Person 37 and Person 86: 116 bits
  Distance between Person 37 and Person 87: 132 bits
  Distance between Person 37 and Person 88: 137 bits
  Distance between Person 37 and Person 89: 140 bits
  Distance between Person 38 and Person 39: 132 bits
  Distance between Person 38 and Person 40: 122 bits
  Distance between Person 38 and Person 41: 136 bits
  Distance between Person 38 and Person 42: 130 bits
  Distance between Person 38 and Person 43: 125 bits
  Distance between Person 38 and Person 44: 130 bits
  Distance between Person 38 and Person 45: 122 bits
  Distance between Person 38 and Person 46: 145 bits
  Distance between Person 38 and Person 47: 138 bits
  Distance between Person 38 and Person 48: 117 bits
  Distance between Person 38 and Person 49: 113 bits
  Distance between Person 38 and Person 50: 131 bits
  Distance between Person 38 and Person 51: 145 bits
  Distance between Person 38 and Person 52: 130 bits
  Distance between Person 38 and Person 53: 117 bits
  Distance between Person 38 and Person 54: 124 bits
  Distance between Person 38 and Person 55: 116 bits
  Distance between Person 38 and Person 56: 138 bits
  Distance between Person 38 and Person 57: 113 bits
  Distance between Person 38 and Person 58: 157 bits
  Distance between Person 38 and Person 59: 138 bits
  Distance between Person 38 and Person 60: 119 bits
  Distance between Person 38 and Person 61: 117 bits
  Distance between Person 38 and Person 62: 117 bits
  Distance between Person 38 and Person 63: 139 bits
  Distance between Person 38 and Person 64: 124 bits
  Distance between Person 38 and Person 65: 132 bits
  Distance between Person 38 and Person 66: 126 bits
  Distance between Person 38 and Person 67: 147 bits
  Distance between Person 38 and Person 68: 137 bits
  Distance between Person 38 and Person 69: 135 bits
  Distance between Person 38 and Person 70: 73 bits
  Distance between Person 38 and Person 71: 124 bits
  Distance between Person 38 and Person 72: 121 bits
  Distance between Person 38 and Person 73: 133 bits
  Distance between Person 38 and Person 74: 123 bits
  Distance between Person 38 and Person 75: 116 bits
  Distance between Person 38 and Person 76: 126 bits
  Distance between Person 38 and Person 77: 123 bits
  Distance between Person 38 and Person 78: 115 bits
  Distance between Person 38 and Person 79: 136 bits
  Distance between Person 38 and Person 80: 113 bits
  Distance between Person 38 and Person 81: 121 bits
  Distance between Person 38 and Person 82: 138 bits
  Distance between Person 38 and Person 83: 137 bits
  Distance between Person 38 and Person 84: 155 bits
  Distance between Person 38 and Person 85: 126 bits
  Distance between Person 38 and Person 86: 103 bits
  Distance between Person 38 and Person 87: 135 bits
  Distance between Person 38 and Person 88: 118 bits
  Distance between Person 38 and Person 89: 115 bits
  Distance between Person 39 and Person 40: 108 bits
  Distance between Person 39 and Person 41: 142 bits
  Distance between Person 39 and Person 42: 130 bits
  Distance between Person 39 and Person 43: 109 bits
  Distance between Person 39 and Person 44: 140 bits
  Distance between Person 39 and Person 45: 112 bits
  Distance between Person 39 and Person 46: 147 bits
  Distance between Person 39 and Person 47: 102 bits
  Distance between Person 39 and Person 48: 115 bits
  Distance between Person 39 and Person 49: 133 bits
  Distance between Person 39 and Person 50: 119 bits
  Distance between Person 39 and Person 51: 135 bits
  Distance between Person 39 and Person 52: 120 bits
  Distance between Person 39 and Person 53: 127 bits
  Distance between Person 39 and Person 54: 122 bits
  Distance between Person 39 and Person 55: 134 bits
  Distance between Person 39 and Person 56: 116 bits
  Distance between Person 39 and Person 57: 127 bits
  Distance between Person 39 and Person 58: 135 bits
  Distance between Person 39 and Person 59: 122 bits
  Distance between Person 39 and Person 60: 143 bits
  Distance between Person 39 and Person 61: 113 bits
  Distance between Person 39 and Person 62: 115 bits
  Distance between Person 39 and Person 63: 133 bits
  Distance between Person 39 and Person 64: 90 bits
  Distance between Person 39 and Person 65: 142 bits
  Distance between Person 39 and Person 66: 120 bits
  Distance between Person 39 and Person 67: 127 bits
  Distance between Person 39 and Person 68: 133 bits
  Distance between Person 39 and Person 69: 143 bits
  Distance between Person 39 and Person 70: 133 bits
  Distance between Person 39 and Person 71: 124 bits
  Distance between Person 39 and Person 72: 113 bits
  Distance between Person 39 and Person 73: 117 bits
  Distance between Person 39 and Person 74: 113 bits
  Distance between Person 39 and Person 75: 136 bits
  Distance between Person 39 and Person 76: 112 bits
  Distance between Person 39 and Person 77: 119 bits
  Distance between Person 39 and Person 78: 133 bits
  Distance between Person 39 and Person 79: 136 bits
  Distance between Person 39 and Person 80: 119 bits
  Distance between Person 39 and Person 81: 135 bits
  Distance between Person 39 and Person 82: 128 bits
  Distance between Person 39 and Person 83: 141 bits
  Distance between Person 39 and Person 84: 121 bits
  Distance between Person 39 and Person 85: 142 bits
  Distance between Person 39 and Person 86: 97 bits
  Distance between Person 39 and Person 87: 125 bits
  Distance between Person 39 and Person 88: 150 bits
  Distance between Person 39 and Person 89: 147 bits
  Distance between Person 40 and Person 41: 122 bits
  Distance between Person 40 and Person 42: 116 bits
  Distance between Person 40 and Person 43: 97 bits
  Distance between Person 40 and Person 44: 134 bits
  Distance between Person 40 and Person 45: 116 bits
  Distance between Person 40 and Person 46: 133 bits
  Distance between Person 40 and Person 47: 122 bits
  Distance between Person 40 and Person 48: 119 bits
  Distance between Person 40 and Person 49: 141 bits
  Distance between Person 40 and Person 50: 129 bits
  Distance between Person 40 and Person 51: 135 bits
  Distance between Person 40 and Person 52: 108 bits
  Distance between Person 40 and Person 53: 135 bits
  Distance between Person 40 and Person 54: 130 bits
  Distance between Person 40 and Person 55: 98 bits
  Distance between Person 40 and Person 56: 120 bits
  Distance between Person 40 and Person 57: 111 bits
  Distance between Person 40 and Person 58: 125 bits
  Distance between Person 40 and Person 59: 116 bits
  Distance between Person 40 and Person 60: 131 bits
  Distance between Person 40 and Person 61: 117 bits
  Distance between Person 40 and Person 62: 125 bits
  Distance between Person 40 and Person 63: 139 bits
  Distance between Person 40 and Person 64: 120 bits
  Distance between Person 40 and Person 65: 120 bits
  Distance between Person 40 and Person 66: 130 bits
  Distance between Person 40 and Person 67: 145 bits
  Distance between Person 40 and Person 68: 121 bits
  Distance between Person 40 and Person 69: 141 bits
  Distance between Person 40 and Person 70: 133 bits
  Distance between Person 40 and Person 71: 104 bits
  Distance between Person 40 and Person 72: 149 bits
  Distance between Person 40 and Person 73: 123 bits
  Distance between Person 40 and Person 74: 125 bits
  Distance between Person 40 and Person 75: 130 bits
  Distance between Person 40 and Person 76: 126 bits
  Distance between Person 40 and Person 77: 127 bits
  Distance between Person 40 and Person 78: 115 bits
  Distance between Person 40 and Person 79: 136 bits
  Distance between Person 40 and Person 80: 133 bits
  Distance between Person 40 and Person 81: 129 bits
  Distance between Person 40 and Person 82: 114 bits
  Distance between Person 40 and Person 83: 131 bits
  Distance between Person 40 and Person 84: 123 bits
  Distance between Person 40 and Person 85: 136 bits
  Distance between Person 40 and Person 86: 113 bits
  Distance between Person 40 and Person 87: 129 bits
  Distance between Person 40 and Person 88: 116 bits
  Distance between Person 40 and Person 89: 131 bits
  Distance between Person 41 and Person 42: 124 bits
  Distance between Person 41 and Person 43: 139 bits
  Distance between Person 41 and Person 44: 124 bits
  Distance between Person 41 and Person 45: 94 bits
  Distance between Person 41 and Person 46: 145 bits
  Distance between Person 41 and Person 47: 122 bits
  Distance between Person 41 and Person 48: 125 bits
  Distance between Person 41 and Person 49: 121 bits
  Distance between Person 41 and Person 50: 139 bits
  Distance between Person 41 and Person 51: 129 bits
  Distance between Person 41 and Person 52: 106 bits
  Distance between Person 41 and Person 53: 117 bits
  Distance between Person 41 and Person 54: 130 bits
  Distance between Person 41 and Person 55: 96 bits
  Distance between Person 41 and Person 56: 120 bits
  Distance between Person 41 and Person 57: 115 bits
  Distance between Person 41 and Person 58: 127 bits
  Distance between Person 41 and Person 59: 128 bits
  Distance between Person 41 and Person 60: 119 bits
  Distance between Person 41 and Person 61: 133 bits
  Distance between Person 41 and Person 62: 121 bits
  Distance between Person 41 and Person 63: 111 bits
  Distance between Person 41 and Person 64: 132 bits
  Distance between Person 41 and Person 65: 124 bits
  Distance between Person 41 and Person 66: 108 bits
  Distance between Person 41 and Person 67: 123 bits
  Distance between Person 41 and Person 68: 129 bits
  Distance between Person 41 and Person 69: 119 bits
  Distance between Person 41 and Person 70: 135 bits
  Distance between Person 41 and Person 71: 138 bits
  Distance between Person 41 and Person 72: 127 bits
  Distance between Person 41 and Person 73: 137 bits
  Distance between Person 41 and Person 74: 123 bits
  Distance between Person 41 and Person 75: 122 bits
  Distance between Person 41 and Person 76: 128 bits
  Distance between Person 41 and Person 77: 125 bits
  Distance between Person 41 and Person 78: 117 bits
  Distance between Person 41 and Person 79: 110 bits
  Distance between Person 41 and Person 80: 123 bits
  Distance between Person 41 and Person 81: 121 bits
  Distance between Person 41 and Person 82: 118 bits
  Distance between Person 41 and Person 83: 107 bits
  Distance between Person 41 and Person 84: 103 bits
  Distance between Person 41 and Person 85: 120 bits
  Distance between Person 41 and Person 86: 133 bits
  Distance between Person 41 and Person 87: 141 bits
  Distance between Person 41 and Person 88: 134 bits
  Distance between Person 41 and Person 89: 121 bits
  Distance between Person 42 and Person 43: 119 bits
  Distance between Person 42 and Person 44: 136 bits
  Distance between Person 42 and Person 45: 126 bits
  Distance between Person 42 and Person 46: 113 bits
  Distance between Person 42 and Person 47: 104 bits
  Distance between Person 42 and Person 48: 121 bits
  Distance between Person 42 and Person 49: 123 bits
  Distance between Person 42 and Person 50: 145 bits
  Distance between Person 42 and Person 51: 127 bits
  Distance between Person 42 and Person 52: 144 bits
  Distance between Person 42 and Person 53: 141 bits
  Distance between Person 42 and Person 54: 128 bits
  Distance between Person 42 and Person 55: 136 bits
  Distance between Person 42 and Person 56: 106 bits
  Distance between Person 42 and Person 57: 103 bits
  Distance between Person 42 and Person 58: 131 bits
  Distance between Person 42 and Person 59: 64 bits
  Distance between Person 42 and Person 60: 115 bits
  Distance between Person 42 and Person 61: 151 bits
  Distance between Person 42 and Person 62: 129 bits
  Distance between Person 42 and Person 63: 129 bits
  Distance between Person 42 and Person 64: 128 bits
  Distance between Person 42 and Person 65: 122 bits
  Distance between Person 42 and Person 66: 120 bits
  Distance between Person 42 and Person 67: 103 bits
  Distance between Person 42 and Person 68: 135 bits
  Distance between Person 42 and Person 69: 125 bits
  Distance between Person 42 and Person 70: 123 bits
  Distance between Person 42 and Person 71: 140 bits
  Distance between Person 42 and Person 72: 127 bits
  Distance between Person 42 and Person 73: 149 bits
  Distance between Person 42 and Person 74: 103 bits
  Distance between Person 42 and Person 75: 104 bits
  Distance between Person 42 and Person 76: 132 bits
  Distance between Person 42 and Person 77: 109 bits
  Distance between Person 42 and Person 78: 117 bits
  Distance between Person 42 and Person 79: 122 bits
  Distance between Person 42 and Person 80: 135 bits
  Distance between Person 42 and Person 81: 115 bits
  Distance between Person 42 and Person 82: 128 bits
  Distance between Person 42 and Person 83: 153 bits
  Distance between Person 42 and Person 84: 129 bits
  Distance between Person 42 and Person 85: 136 bits
  Distance between Person 42 and Person 86: 145 bits
  Distance between Person 42 and Person 87: 133 bits
  Distance between Person 42 and Person 88: 130 bits
  Distance between Person 42 and Person 89: 151 bits
  Distance between Person 43 and Person 44: 139 bits
  Distance between Person 43 and Person 45: 145 bits
  Distance between Person 43 and Person 46: 134 bits
  Distance between Person 43 and Person 47: 103 bits
  Distance between Person 43 and Person 48: 128 bits
  Distance between Person 43 and Person 49: 128 bits
  Distance between Person 43 and Person 50: 112 bits
  Distance between Person 43 and Person 51: 108 bits
  Distance between Person 43 and Person 52: 133 bits
  Distance between Person 43 and Person 53: 158 bits
  Distance between Person 43 and Person 54: 119 bits
  Distance between Person 43 and Person 55: 141 bits
  Distance between Person 43 and Person 56: 85 bits
  Distance between Person 43 and Person 57: 134 bits
  Distance between Person 43 and Person 58: 138 bits
  Distance between Person 43 and Person 59: 107 bits
  Distance between Person 43 and Person 60: 144 bits
  Distance between Person 43 and Person 61: 140 bits
  Distance between Person 43 and Person 62: 124 bits
  Distance between Person 43 and Person 63: 120 bits
  Distance between Person 43 and Person 64: 131 bits
  Distance between Person 43 and Person 65: 147 bits
  Distance between Person 43 and Person 66: 141 bits
  Distance between Person 43 and Person 67: 120 bits
  Distance between Person 43 and Person 68: 112 bits
  Distance between Person 43 and Person 69: 130 bits
  Distance between Person 43 and Person 70: 128 bits
  Distance between Person 43 and Person 71: 129 bits
  Distance between Person 43 and Person 72: 144 bits
  Distance between Person 43 and Person 73: 132 bits
  Distance between Person 43 and Person 74: 118 bits
  Distance between Person 43 and Person 75: 113 bits
  Distance between Person 43 and Person 76: 129 bits
  Distance between Person 43 and Person 77: 120 bits
  Distance between Person 43 and Person 78: 140 bits
  Distance between Person 43 and Person 79: 147 bits
  Distance between Person 43 and Person 80: 126 bits
  Distance between Person 43 and Person 81: 122 bits
  Distance between Person 43 and Person 82: 131 bits
  Distance between Person 43 and Person 83: 128 bits
  Distance between Person 43 and Person 84: 122 bits
  Distance between Person 43 and Person 85: 121 bits
  Distance between Person 43 and Person 86: 144 bits
  Distance between Person 43 and Person 87: 128 bits
  Distance between Person 43 and Person 88: 125 bits
  Distance between Person 43 and Person 89: 118 bits
  Distance between Person 44 and Person 45: 104 bits
  Distance between Person 44 and Person 46: 111 bits
  Distance between Person 44 and Person 47: 110 bits
  Distance between Person 44 and Person 48: 137 bits
  Distance between Person 44 and Person 49: 131 bits
  Distance between Person 44 and Person 50: 123 bits
  Distance between Person 44 and Person 51: 117 bits
  Distance between Person 44 and Person 52: 114 bits
  Distance between Person 44 and Person 53: 91 bits
  Distance between Person 44 and Person 54: 84 bits
  Distance between Person 44 and Person 55: 108 bits
  Distance between Person 44 and Person 56: 132 bits
  Distance between Person 44 and Person 57: 141 bits
  Distance between Person 44 and Person 58: 103 bits
  Distance between Person 44 and Person 59: 132 bits
  Distance between Person 44 and Person 60: 129 bits
  Distance between Person 44 and Person 61: 147 bits
  Distance between Person 44 and Person 62: 135 bits
  Distance between Person 44 and Person 63: 127 bits
  Distance between Person 44 and Person 64: 88 bits
  Distance between Person 44 and Person 65: 108 bits
  Distance between Person 44 and Person 66: 112 bits
  Distance between Person 44 and Person 67: 123 bits
  Distance between Person 44 and Person 68: 141 bits
  Distance between Person 44 and Person 69: 111 bits
  Distance between Person 44 and Person 70: 113 bits
  Distance between Person 44 and Person 71: 148 bits
  Distance between Person 44 and Person 72: 117 bits
  Distance between Person 44 and Person 73: 113 bits
  Distance between Person 44 and Person 74: 133 bits
  Distance between Person 44 and Person 75: 116 bits
  Distance between Person 44 and Person 76: 118 bits
  Distance between Person 44 and Person 77: 123 bits
  Distance between Person 44 and Person 78: 135 bits
  Distance between Person 44 and Person 79: 138 bits
  Distance between Person 44 and Person 80: 123 bits
  Distance between Person 44 and Person 81: 143 bits
  Distance between Person 44 and Person 82: 98 bits
  Distance between Person 44 and Person 83: 123 bits
  Distance between Person 44 and Person 84: 151 bits
  Distance between Person 44 and Person 85: 136 bits
  Distance between Person 44 and Person 86: 135 bits
  Distance between Person 44 and Person 87: 121 bits
  Distance between Person 44 and Person 88: 124 bits
  Distance between Person 44 and Person 89: 123 bits
  Distance between Person 45 and Person 46: 129 bits
  Distance between Person 45 and Person 47: 100 bits
  Distance between Person 45 and Person 48: 143 bits
  Distance between Person 45 and Person 49: 139 bits
  Distance between Person 45 and Person 50: 135 bits
  Distance between Person 45 and Person 51: 119 bits
  Distance between Person 45 and Person 52: 104 bits
  Distance between Person 45 and Person 53: 97 bits
  Distance between Person 45 and Person 54: 124 bits
  Distance between Person 45 and Person 55: 86 bits
  Distance between Person 45 and Person 56: 104 bits
  Distance between Person 45 and Person 57: 135 bits
  Distance between Person 45 and Person 58: 127 bits
  Distance between Person 45 and Person 59: 130 bits
  Distance between Person 45 and Person 60: 123 bits
  Distance between Person 45 and Person 61: 117 bits
  Distance between Person 45 and Person 62: 113 bits
  Distance between Person 45 and Person 63: 119 bits
  Distance between Person 45 and Person 64: 112 bits
  Distance between Person 45 and Person 65: 128 bits
  Distance between Person 45 and Person 66: 130 bits
  Distance between Person 45 and Person 67: 137 bits
  Distance between Person 45 and Person 68: 129 bits
  Distance between Person 45 and Person 69: 129 bits
  Distance between Person 45 and Person 70: 127 bits
  Distance between Person 45 and Person 71: 126 bits
  Distance between Person 45 and Person 72: 121 bits
  Distance between Person 45 and Person 73: 125 bits
  Distance between Person 45 and Person 74: 123 bits
  Distance between Person 45 and Person 75: 120 bits
  Distance between Person 45 and Person 76: 130 bits
  Distance between Person 45 and Person 77: 127 bits
  Distance between Person 45 and Person 78: 117 bits
  Distance between Person 45 and Person 79: 134 bits
  Distance between Person 45 and Person 80: 97 bits
  Distance between Person 45 and Person 81: 145 bits
  Distance between Person 45 and Person 82: 100 bits
  Distance between Person 45 and Person 83: 117 bits
  Distance between Person 45 and Person 84: 127 bits
  Distance between Person 45 and Person 85: 142 bits
  Distance between Person 45 and Person 86: 95 bits
  Distance between Person 45 and Person 87: 127 bits
  Distance between Person 45 and Person 88: 136 bits
  Distance between Person 45 and Person 89: 129 bits
  Distance between Person 46 and Person 47: 127 bits
  Distance between Person 46 and Person 48: 126 bits
  Distance between Person 46 and Person 49: 126 bits
  Distance between Person 46 and Person 50: 126 bits
  Distance between Person 46 and Person 51: 122 bits
  Distance between Person 46 and Person 52: 155 bits
  Distance between Person 46 and Person 53: 118 bits
  Distance between Person 46 and Person 54: 119 bits
  Distance between Person 46 and Person 55: 165 bits
  Distance between Person 46 and Person 56: 131 bits
  Distance between Person 46 and Person 57: 130 bits
  Distance between Person 46 and Person 58: 84 bits
  Distance between Person 46 and Person 59: 117 bits
  Distance between Person 46 and Person 60: 120 bits
  Distance between Person 46 and Person 61: 110 bits
  Distance between Person 46 and Person 62: 136 bits
  Distance between Person 46 and Person 63: 112 bits
  Distance between Person 46 and Person 64: 147 bits
  Distance between Person 46 and Person 65: 107 bits
  Distance between Person 46 and Person 66: 159 bits
  Distance between Person 46 and Person 67: 134 bits
  Distance between Person 46 and Person 68: 128 bits
  Distance between Person 46 and Person 69: 140 bits
  Distance between Person 46 and Person 70: 124 bits
  Distance between Person 46 and Person 71: 137 bits
  Distance between Person 46 and Person 72: 120 bits
  Distance between Person 46 and Person 73: 104 bits
  Distance between Person 46 and Person 74: 128 bits
  Distance between Person 46 and Person 75: 133 bits
  Distance between Person 46 and Person 76: 113 bits
  Distance between Person 46 and Person 77: 124 bits
  Distance between Person 46 and Person 78: 120 bits
  Distance between Person 46 and Person 79: 133 bits
  Distance between Person 46 and Person 80: 126 bits
  Distance between Person 46 and Person 81: 140 bits
  Distance between Person 46 and Person 82: 113 bits
  Distance between Person 46 and Person 83: 148 bits
  Distance between Person 46 and Person 84: 102 bits
  Distance between Person 46 and Person 85: 125 bits
  Distance between Person 46 and Person 86: 136 bits
  Distance between Person 46 and Person 87: 134 bits
  Distance between Person 46 and Person 88: 137 bits
  Distance between Person 46 and Person 89: 124 bits
  Distance between Person 47 and Person 48: 119 bits
  Distance between Person 47 and Person 49: 127 bits
  Distance between Person 47 and Person 50: 107 bits
  Distance between Person 47 and Person 51: 131 bits
  Distance between Person 47 and Person 52: 132 bits
  Distance between Person 47 and Person 53: 139 bits
  Distance between Person 47 and Person 54: 114 bits
  Distance between Person 47 and Person 55: 116 bits
  Distance between Person 47 and Person 56: 104 bits
  Distance between Person 47 and Person 57: 123 bits
  Distance between Person 47 and Person 58: 105 bits
  Distance between Person 47 and Person 59: 124 bits
  Distance between Person 47 and Person 60: 139 bits
  Distance between Person 47 and Person 61: 125 bits
  Distance between Person 47 and Person 62: 139 bits
  Distance between Person 47 and Person 63: 139 bits
  Distance between Person 47 and Person 64: 116 bits
  Distance between Person 47 and Person 65: 134 bits
  Distance between Person 47 and Person 66: 106 bits
  Distance between Person 47 and Person 67: 115 bits
  Distance between Person 47 and Person 68: 133 bits
  Distance between Person 47 and Person 69: 115 bits
  Distance between Person 47 and Person 70: 127 bits
  Distance between Person 47 and Person 71: 122 bits
  Distance between Person 47 and Person 72: 111 bits
  Distance between Person 47 and Person 73: 119 bits
  Distance between Person 47 and Person 74: 119 bits
  Distance between Person 47 and Person 75: 82 bits
  Distance between Person 47 and Person 76: 134 bits
  Distance between Person 47 and Person 77: 115 bits
  Distance between Person 47 and Person 78: 125 bits
  Distance between Person 47 and Person 79: 120 bits
  Distance between Person 47 and Person 80: 129 bits
  Distance between Person 47 and Person 81: 133 bits
  Distance between Person 47 and Person 82: 110 bits
  Distance between Person 47 and Person 83: 115 bits
  Distance between Person 47 and Person 84: 141 bits
  Distance between Person 47 and Person 85: 120 bits
  Distance between Person 47 and Person 86: 121 bits
  Distance between Person 47 and Person 87: 109 bits
  Distance between Person 47 and Person 88: 132 bits
  Distance between Person 47 and Person 89: 131 bits
  Distance between Person 48 and Person 49: 114 bits
  Distance between Person 48 and Person 50: 128 bits
  Distance between Person 48 and Person 51: 130 bits
  Distance between Person 48 and Person 52: 117 bits
  Distance between Person 48 and Person 53: 136 bits
  Distance between Person 48 and Person 54: 129 bits
  Distance between Person 48 and Person 55: 121 bits
  Distance between Person 48 and Person 56: 137 bits
  Distance between Person 48 and Person 57: 126 bits
  Distance between Person 48 and Person 58: 120 bits
  Distance between Person 48 and Person 59: 117 bits
  Distance between Person 48 and Person 60: 122 bits
  Distance between Person 48 and Person 61: 132 bits
  Distance between Person 48 and Person 62: 104 bits
  Distance between Person 48 and Person 63: 130 bits
  Distance between Person 48 and Person 64: 105 bits
  Distance between Person 48 and Person 65: 131 bits
  Distance between Person 48 and Person 66: 109 bits
  Distance between Person 48 and Person 67: 142 bits
  Distance between Person 48 and Person 68: 134 bits
  Distance between Person 48 and Person 69: 136 bits
  Distance between Person 48 and Person 70: 128 bits
  Distance between Person 48 and Person 71: 135 bits
  Distance between Person 48 and Person 72: 116 bits
  Distance between Person 48 and Person 73: 134 bits
  Distance between Person 48 and Person 74: 118 bits
  Distance between Person 48 and Person 75: 139 bits
  Distance between Person 48 and Person 76: 109 bits
  Distance between Person 48 and Person 77: 106 bits
  Distance between Person 48 and Person 78: 118 bits
  Distance between Person 48 and Person 79: 123 bits
  Distance between Person 48 and Person 80: 144 bits
  Distance between Person 48 and Person 81: 106 bits
  Distance between Person 48 and Person 82: 149 bits
  Distance between Person 48 and Person 83: 138 bits
  Distance between Person 48 and Person 84: 142 bits
  Distance between Person 48 and Person 85: 121 bits
  Distance between Person 48 and Person 86: 132 bits
  Distance between Person 48 and Person 87: 120 bits
  Distance between Person 48 and Person 88: 167 bits
  Distance between Person 48 and Person 89: 128 bits
  Distance between Person 49 and Person 50: 132 bits
  Distance between Person 49 and Person 51: 132 bits
  Distance between Person 49 and Person 52: 137 bits
  Distance between Person 49 and Person 53: 144 bits
  Distance between Person 49 and Person 54: 131 bits
  Distance between Person 49 and Person 55: 129 bits
  Distance between Person 49 and Person 56: 133 bits
  Distance between Person 49 and Person 57: 114 bits
  Distance between Person 49 and Person 58: 128 bits
  Distance between Person 49 and Person 59: 117 bits
  Distance between Person 49 and Person 60: 112 bits
  Distance between Person 49 and Person 61: 120 bits
  Distance between Person 49 and Person 62: 126 bits
  Distance between Person 49 and Person 63: 140 bits
  Distance between Person 49 and Person 64: 133 bits
  Distance between Person 49 and Person 65: 127 bits
  Distance between Person 49 and Person 66: 137 bits
  Distance between Person 49 and Person 67: 122 bits
  Distance between Person 49 and Person 68: 138 bits
  Distance between Person 49 and Person 69: 120 bits
  Distance between Person 49 and Person 70: 124 bits
  Distance between Person 49 and Person 71: 105 bits
  Distance between Person 49 and Person 72: 104 bits
  Distance between Person 49 and Person 73: 152 bits
  Distance between Person 49 and Person 74: 124 bits
  Distance between Person 49 and Person 75: 123 bits
  Distance between Person 49 and Person 76: 125 bits
  Distance between Person 49 and Person 77: 128 bits
  Distance between Person 49 and Person 78: 138 bits
  Distance between Person 49 and Person 79: 107 bits
  Distance between Person 49 and Person 80: 106 bits
  Distance between Person 49 and Person 81: 138 bits
  Distance between Person 49 and Person 82: 153 bits
  Distance between Person 49 and Person 83: 124 bits
  Distance between Person 49 and Person 84: 134 bits
  Distance between Person 49 and Person 85: 127 bits
  Distance between Person 49 and Person 86: 120 bits
  Distance between Person 49 and Person 87: 120 bits
  Distance between Person 49 and Person 88: 137 bits
  Distance between Person 49 and Person 89: 118 bits
  Distance between Person 50 and Person 51: 120 bits
  Distance between Person 50 and Person 52: 131 bits
  Distance between Person 50 and Person 53: 126 bits
  Distance between Person 50 and Person 54: 123 bits
  Distance between Person 50 and Person 55: 125 bits
  Distance between Person 50 and Person 56: 117 bits
  Distance between Person 50 and Person 57: 132 bits
  Distance between Person 50 and Person 58: 136 bits
  Distance between Person 50 and Person 59: 127 bits
  Distance between Person 50 and Person 60: 130 bits
  Distance between Person 50 and Person 61: 122 bits
  Distance between Person 50 and Person 62: 124 bits
  Distance between Person 50 and Person 63: 132 bits
  Distance between Person 50 and Person 64: 125 bits
  Distance between Person 50 and Person 65: 129 bits
  Distance between Person 50 and Person 66: 141 bits
  Distance between Person 50 and Person 67: 138 bits
  Distance between Person 50 and Person 68: 126 bits
  Distance between Person 50 and Person 69: 128 bits
  Distance between Person 50 and Person 70: 128 bits
  Distance between Person 50 and Person 71: 117 bits
  Distance between Person 50 and Person 72: 128 bits
  Distance between Person 50 and Person 73: 140 bits
  Distance between Person 50 and Person 74: 136 bits
  Distance between Person 50 and Person 75: 113 bits
  Distance between Person 50 and Person 76: 105 bits
  Distance between Person 50 and Person 77: 114 bits
  Distance between Person 50 and Person 78: 124 bits
  Distance between Person 50 and Person 79: 115 bits
  Distance between Person 50 and Person 80: 128 bits
  Distance between Person 50 and Person 81: 136 bits
  Distance between Person 50 and Person 82: 129 bits
  Distance between Person 50 and Person 83: 138 bits
  Distance between Person 50 and Person 84: 122 bits
  Distance between Person 50 and Person 85: 123 bits
  Distance between Person 50 and Person 86: 136 bits
  Distance between Person 50 and Person 87: 116 bits
  Distance between Person 50 and Person 88: 125 bits
  Distance between Person 50 and Person 89: 128 bits
  Distance between Person 51 and Person 52: 107 bits
  Distance between Person 51 and Person 53: 134 bits
  Distance between Person 51 and Person 54: 119 bits
  Distance between Person 51 and Person 55: 121 bits
  Distance between Person 51 and Person 56: 113 bits
  Distance between Person 51 and Person 57: 156 bits
  Distance between Person 51 and Person 58: 134 bits
  Distance between Person 51 and Person 59: 121 bits
  Distance between Person 51 and Person 60: 146 bits
  Distance between Person 51 and Person 61: 118 bits
  Distance between Person 51 and Person 62: 116 bits
  Distance between Person 51 and Person 63: 132 bits
  Distance between Person 51 and Person 64: 107 bits
  Distance between Person 51 and Person 65: 139 bits
  Distance between Person 51 and Person 66: 137 bits
  Distance between Person 51 and Person 67: 136 bits
  Distance between Person 51 and Person 68: 116 bits
  Distance between Person 51 and Person 69: 118 bits
  Distance between Person 51 and Person 70: 122 bits
  Distance between Person 51 and Person 71: 147 bits
  Distance between Person 51 and Person 72: 132 bits
  Distance between Person 51 and Person 73: 122 bits
  Distance between Person 51 and Person 74: 114 bits
  Distance between Person 51 and Person 75: 131 bits
  Distance between Person 51 and Person 76: 123 bits
  Distance between Person 51 and Person 77: 126 bits
  Distance between Person 51 and Person 78: 124 bits
  Distance between Person 51 and Person 79: 115 bits
  Distance between Person 51 and Person 80: 116 bits
  Distance between Person 51 and Person 81: 148 bits
  Distance between Person 51 and Person 82: 93 bits
  Distance between Person 51 and Person 83: 120 bits
  Distance between Person 51 and Person 84: 114 bits
  Distance between Person 51 and Person 85: 123 bits
  Distance between Person 51 and Person 86: 148 bits
  Distance between Person 51 and Person 87: 124 bits
  Distance between Person 51 and Person 88: 125 bits
  Distance between Person 51 and Person 89: 114 bits
  Distance between Person 52 and Person 53: 105 bits
  Distance between Person 52 and Person 54: 120 bits
  Distance between Person 52 and Person 55: 48 bits
  Distance between Person 52 and Person 56: 134 bits
  Distance between Person 52 and Person 57: 131 bits
  Distance between Person 52 and Person 58: 123 bits
  Distance between Person 52 and Person 59: 150 bits
  Distance between Person 52 and Person 60: 133 bits
  Distance between Person 52 and Person 61: 119 bits
  Distance between Person 52 and Person 62: 127 bits
  Distance between Person 52 and Person 63: 129 bits
  Distance between Person 52 and Person 64: 116 bits
  Distance between Person 52 and Person 65: 132 bits
  Distance between Person 52 and Person 66: 112 bits
  Distance between Person 52 and Person 67: 139 bits
  Distance between Person 52 and Person 68: 135 bits
  Distance between Person 52 and Person 69: 127 bits
  Distance between Person 52 and Person 70: 137 bits
  Distance between Person 52 and Person 71: 136 bits
  Distance between Person 52 and Person 72: 129 bits
  Distance between Person 52 and Person 73: 135 bits
  Distance between Person 52 and Person 74: 113 bits
  Distance between Person 52 and Person 75: 136 bits
  Distance between Person 52 and Person 76: 110 bits
  Distance between Person 52 and Person 77: 121 bits
  Distance between Person 52 and Person 78: 135 bits
  Distance between Person 52 and Person 79: 112 bits
  Distance between Person 52 and Person 80: 113 bits
  Distance between Person 52 and Person 81: 147 bits
  Distance between Person 52 and Person 82: 126 bits
  Distance between Person 52 and Person 83: 125 bits
  Distance between Person 52 and Person 84: 117 bits
  Distance between Person 52 and Person 85: 130 bits
  Distance between Person 52 and Person 86: 123 bits
  Distance between Person 52 and Person 87: 125 bits
  Distance between Person 52 and Person 88: 132 bits
  Distance between Person 52 and Person 89: 115 bits
  Distance between Person 53 and Person 54: 109 bits
  Distance between Person 53 and Person 55: 97 bits
  Distance between Person 53 and Person 56: 129 bits
  Distance between Person 53 and Person 57: 116 bits
  Distance between Person 53 and Person 58: 130 bits
  Distance between Person 53 and Person 59: 151 bits
  Distance between Person 53 and Person 60: 120 bits
  Distance between Person 53 and Person 61: 126 bits
  Distance between Person 53 and Person 62: 138 bits
  Distance between Person 53 and Person 63: 114 bits
  Distance between Person 53 and Person 64: 125 bits
  Distance between Person 53 and Person 65: 125 bits
  Distance between Person 53 and Person 66: 131 bits
  Distance between Person 53 and Person 67: 120 bits
  Distance between Person 53 and Person 68: 156 bits
  Distance between Person 53 and Person 69: 122 bits
  Distance between Person 53 and Person 70: 120 bits
  Distance between Person 53 and Person 71: 147 bits
  Distance between Person 53 and Person 72: 144 bits
  Distance between Person 53 and Person 73: 120 bits
  Distance between Person 53 and Person 74: 130 bits
  Distance between Person 53 and Person 75: 131 bits
  Distance between Person 53 and Person 76: 107 bits
  Distance between Person 53 and Person 77: 128 bits
  Distance between Person 53 and Person 78: 148 bits
  Distance between Person 53 and Person 79: 135 bits
  Distance between Person 53 and Person 80: 134 bits
  Distance between Person 53 and Person 81: 134 bits
  Distance between Person 53 and Person 82: 119 bits
  Distance between Person 53 and Person 83: 130 bits
  Distance between Person 53 and Person 84: 132 bits
  Distance between Person 53 and Person 85: 129 bits
  Distance between Person 53 and Person 86: 126 bits
  Distance between Person 53 and Person 87: 138 bits
  Distance between Person 53 and Person 88: 113 bits
  Distance between Person 53 and Person 89: 112 bits
  Distance between Person 54 and Person 55: 116 bits
  Distance between Person 54 and Person 56: 116 bits
  Distance between Person 54 and Person 57: 131 bits
  Distance between Person 54 and Person 58: 97 bits
  Distance between Person 54 and Person 59: 120 bits
  Distance between Person 54 and Person 60: 119 bits
  Distance between Person 54 and Person 61: 141 bits
  Distance between Person 54 and Person 62: 121 bits
  Distance between Person 54 and Person 63: 129 bits
  Distance between Person 54 and Person 64: 88 bits
  Distance between Person 54 and Person 65: 130 bits
  Distance between Person 54 and Person 66: 140 bits
  Distance between Person 54 and Person 67: 129 bits
  Distance between Person 54 and Person 68: 133 bits
  Distance between Person 54 and Person 69: 111 bits
  Distance between Person 54 and Person 70: 113 bits
  Distance between Person 54 and Person 71: 148 bits
  Distance between Person 54 and Person 72: 141 bits
  Distance between Person 54 and Person 73: 123 bits
  Distance between Person 54 and Person 74: 117 bits
  Distance between Person 54 and Person 75: 106 bits
  Distance between Person 54 and Person 76: 90 bits
  Distance between Person 54 and Person 77: 123 bits
  Distance between Person 54 and Person 78: 129 bits
  Distance between Person 54 and Person 79: 138 bits
  Distance between Person 54 and Person 80: 119 bits
  Distance between Person 54 and Person 81: 129 bits
  Distance between Person 54 and Person 82: 120 bits
  Distance between Person 54 and Person 83: 145 bits
  Distance between Person 54 and Person 84: 141 bits
  Distance between Person 54 and Person 85: 130 bits
  Distance between Person 54 and Person 86: 137 bits
  Distance between Person 54 and Person 87: 133 bits
  Distance between Person 54 and Person 88: 110 bits
  Distance between Person 54 and Person 89: 127 bits
  Distance between Person 55 and Person 56: 132 bits
  Distance between Person 55 and Person 57: 115 bits
  Distance between Person 55 and Person 58: 127 bits
  Distance between Person 55 and Person 59: 138 bits
  Distance between Person 55 and Person 60: 111 bits
  Distance between Person 55 and Person 61: 115 bits
  Distance between Person 55 and Person 62: 129 bits
  Distance between Person 55 and Person 63: 131 bits
  Distance between Person 55 and Person 64: 116 bits
  Distance between Person 55 and Person 65: 128 bits
  Distance between Person 55 and Person 66: 108 bits
  Distance between Person 55 and Person 67: 133 bits
  Distance between Person 55 and Person 68: 139 bits
  Distance between Person 55 and Person 69: 115 bits
  Distance between Person 55 and Person 70: 127 bits
  Distance between Person 55 and Person 71: 124 bits
  Distance between Person 55 and Person 72: 137 bits
  Distance between Person 55 and Person 73: 143 bits
  Distance between Person 55 and Person 74: 123 bits
  Distance between Person 55 and Person 75: 120 bits
  Distance between Person 55 and Person 76: 110 bits
  Distance between Person 55 and Person 77: 113 bits
  Distance between Person 55 and Person 78: 127 bits
  Distance between Person 55 and Person 79: 130 bits
  Distance between Person 55 and Person 80: 105 bits
  Distance between Person 55 and Person 81: 143 bits
  Distance between Person 55 and Person 82: 124 bits
  Distance between Person 55 and Person 83: 117 bits
  Distance between Person 55 and Person 84: 141 bits
  Distance between Person 55 and Person 85: 134 bits
  Distance between Person 55 and Person 86: 113 bits
  Distance between Person 55 and Person 87: 135 bits
  Distance between Person 55 and Person 88: 124 bits
  Distance between Person 55 and Person 89: 115 bits
  Distance between Person 56 and Person 57: 109 bits
  Distance between Person 56 and Person 58: 137 bits
  Distance between Person 56 and Person 59: 108 bits
  Distance between Person 56 and Person 60: 129 bits
  Distance between Person 56 and Person 61: 135 bits
  Distance between Person 56 and Person 62: 129 bits
  Distance between Person 56 and Person 63: 137 bits
  Distance between Person 56 and Person 64: 122 bits
  Distance between Person 56 and Person 65: 140 bits
  Distance between Person 56 and Person 66: 150 bits
  Distance between Person 56 and Person 67: 119 bits
  Distance between Person 56 and Person 68: 123 bits
  Distance between Person 56 and Person 69: 135 bits
  Distance between Person 56 and Person 70: 137 bits
  Distance between Person 56 and Person 71: 118 bits
  Distance between Person 56 and Person 72: 141 bits
  Distance between Person 56 and Person 73: 143 bits
  Distance between Person 56 and Person 74: 133 bits
  Distance between Person 56 and Person 75: 122 bits
  Distance between Person 56 and Person 76: 138 bits
  Distance between Person 56 and Person 77: 117 bits
  Distance between Person 56 and Person 78: 137 bits
  Distance between Person 56 and Person 79: 128 bits
  Distance between Person 56 and Person 80: 109 bits
  Distance between Person 56 and Person 81: 139 bits
  Distance between Person 56 and Person 82: 114 bits
  Distance between Person 56 and Person 83: 127 bits
  Distance between Person 56 and Person 84: 101 bits
  Distance between Person 56 and Person 85: 128 bits
  Distance between Person 56 and Person 86: 143 bits
  Distance between Person 56 and Person 87: 135 bits
  Distance between Person 56 and Person 88: 124 bits
  Distance between Person 56 and Person 89: 125 bits
  Distance between Person 57 and Person 58: 138 bits
  Distance between Person 57 and Person 59: 117 bits
  Distance between Person 57 and Person 60: 104 bits
  Distance between Person 57 and Person 61: 142 bits
  Distance between Person 57 and Person 62: 136 bits
  Distance between Person 57 and Person 63: 130 bits
  Distance between Person 57 and Person 64: 139 bits
  Distance between Person 57 and Person 65: 129 bits
  Distance between Person 57 and Person 66: 127 bits
  Distance between Person 57 and Person 67: 114 bits
  Distance between Person 57 and Person 68: 142 bits
  Distance between Person 57 and Person 69: 138 bits
  Distance between Person 57 and Person 70: 132 bits
  Distance between Person 57 and Person 71: 125 bits
  Distance between Person 57 and Person 72: 130 bits
  Distance between Person 57 and Person 73: 138 bits
  Distance between Person 57 and Person 74: 134 bits
  Distance between Person 57 and Person 75: 141 bits
  Distance between Person 57 and Person 76: 127 bits
  Distance between Person 57 and Person 77: 126 bits
  Distance between Person 57 and Person 78: 124 bits
  Distance between Person 57 and Person 79: 121 bits
  Distance between Person 57 and Person 80: 136 bits
  Distance between Person 57 and Person 81: 124 bits
  Distance between Person 57 and Person 82: 141 bits
  Distance between Person 57 and Person 83: 122 bits
  Distance between Person 57 and Person 84: 112 bits
  Distance between Person 57 and Person 85: 129 bits
  Distance between Person 57 and Person 86: 140 bits
  Distance between Person 57 and Person 87: 126 bits
  Distance between Person 57 and Person 88: 123 bits
  Distance between Person 57 and Person 89: 128 bits
  Distance between Person 58 and Person 59: 127 bits
  Distance between Person 58 and Person 60: 138 bits
  Distance between Person 58 and Person 61: 124 bits
  Distance between Person 58 and Person 62: 122 bits
  Distance between Person 58 and Person 63: 118 bits
  Distance between Person 58 and Person 64: 135 bits
  Distance between Person 58 and Person 65: 117 bits
  Distance between Person 58 and Person 66: 133 bits
  Distance between Person 58 and Person 67: 122 bits
  Distance between Person 58 and Person 68: 142 bits
  Distance between Person 58 and Person 69: 112 bits
  Distance between Person 58 and Person 70: 134 bits
  Distance between Person 58 and Person 71: 135 bits
  Distance between Person 58 and Person 72: 108 bits
  Distance between Person 58 and Person 73: 100 bits
  Distance between Person 58 and Person 74: 130 bits
  Distance between Person 58 and Person 75: 115 bits
  Distance between Person 58 and Person 76: 115 bits
  Distance between Person 58 and Person 77: 130 bits
  Distance between Person 58 and Person 78: 124 bits
  Distance between Person 58 and Person 79: 133 bits
  Distance between Person 58 and Person 80: 118 bits
  Distance between Person 58 and Person 81: 126 bits
  Distance between Person 58 and Person 82: 129 bits
  Distance between Person 58 and Person 83: 134 bits
  Distance between Person 58 and Person 84: 118 bits
  Distance between Person 58 and Person 85: 115 bits
  Distance between Person 58 and Person 86: 120 bits
  Distance between Person 58 and Person 87: 114 bits
  Distance between Person 58 and Person 88: 145 bits
  Distance between Person 58 and Person 89: 124 bits
  Distance between Person 59 and Person 60: 101 bits
  Distance between Person 59 and Person 61: 145 bits
  Distance between Person 59 and Person 62: 101 bits
  Distance between Person 59 and Person 63: 133 bits
  Distance between Person 59 and Person 64: 122 bits
  Distance between Person 59 and Person 65: 130 bits
  Distance between Person 59 and Person 66: 126 bits
  Distance between Person 59 and Person 67: 109 bits
  Distance between Person 59 and Person 68: 119 bits
  Distance between Person 59 and Person 69: 129 bits
  Distance between Person 59 and Person 70: 119 bits
  Distance between Person 59 and Person 71: 130 bits
  Distance between Person 59 and Person 72: 119 bits
  Distance between Person 59 and Person 73: 149 bits
  Distance between Person 59 and Person 74: 125 bits
  Distance between Person 59 and Person 75: 128 bits
  Distance between Person 59 and Person 76: 120 bits
  Distance between Person 59 and Person 77: 107 bits
  Distance between Person 59 and Person 78: 125 bits
  Distance between Person 59 and Person 79: 126 bits
  Distance between Person 59 and Person 80: 139 bits
  Distance between Person 59 and Person 81: 129 bits
  Distance between Person 59 and Person 82: 148 bits
  Distance between Person 59 and Person 83: 139 bits
  Distance between Person 59 and Person 84: 125 bits
  Distance between Person 59 and Person 85: 134 bits
  Distance between Person 59 and Person 86: 143 bits
  Distance between Person 59 and Person 87: 121 bits
  Distance between Person 59 and Person 88: 134 bits
  Distance between Person 59 and Person 89: 147 bits
  Distance between Person 60 and Person 61: 136 bits
  Distance between Person 60 and Person 62: 122 bits
  Distance between Person 60 and Person 63: 136 bits
  Distance between Person 60 and Person 64: 131 bits
  Distance between Person 60 and Person 65: 133 bits
  Distance between Person 60 and Person 66: 135 bits
  Distance between Person 60 and Person 67: 124 bits
  Distance between Person 60 and Person 68: 112 bits
  Distance between Person 60 and Person 69: 128 bits
  Distance between Person 60 and Person 70: 110 bits
  Distance between Person 60 and Person 71: 121 bits
  Distance between Person 60 and Person 72: 120 bits
  Distance between Person 60 and Person 73: 144 bits
  Distance between Person 60 and Person 74: 130 bits
  Distance between Person 60 and Person 75: 133 bits
  Distance between Person 60 and Person 76: 135 bits
  Distance between Person 60 and Person 77: 122 bits
  Distance between Person 60 and Person 78: 122 bits
  Distance between Person 60 and Person 79: 129 bits
  Distance between Person 60 and Person 80: 122 bits
  Distance between Person 60 and Person 81: 142 bits
  Distance between Person 60 and Person 82: 131 bits
  Distance between Person 60 and Person 83: 130 bits
  Distance between Person 60 and Person 84: 136 bits
  Distance between Person 60 and Person 85: 117 bits
  Distance between Person 60 and Person 86: 134 bits
  Distance between Person 60 and Person 87: 134 bits
  Distance between Person 60 and Person 88: 139 bits
  Distance between Person 60 and Person 89: 136 bits
  Distance between Person 61 and Person 62: 138 bits
  Distance between Person 61 and Person 63: 150 bits
  Distance between Person 61 and Person 64: 125 bits
  Distance between Person 61 and Person 65: 131 bits
  Distance between Person 61 and Person 66: 127 bits
  Distance between Person 61 and Person 67: 160 bits
  Distance between Person 61 and Person 68: 128 bits
  Distance between Person 61 and Person 69: 132 bits
  Distance between Person 61 and Person 70: 122 bits
  Distance between Person 61 and Person 71: 97 bits
  Distance between Person 61 and Person 72: 116 bits
  Distance between Person 61 and Person 73: 134 bits
  Distance between Person 61 and Person 74: 118 bits
  Distance between Person 61 and Person 75: 145 bits
  Distance between Person 61 and Person 76: 123 bits
  Distance between Person 61 and Person 77: 130 bits
  Distance between Person 61 and Person 78: 130 bits
  Distance between Person 61 and Person 79: 125 bits
  Distance between Person 61 and Person 80: 112 bits
  Distance between Person 61 and Person 81: 158 bits
  Distance between Person 61 and Person 82: 123 bits
  Distance between Person 61 and Person 83: 130 bits
  Distance between Person 61 and Person 84: 100 bits
  Distance between Person 61 and Person 85: 111 bits
  Distance between Person 61 and Person 86: 74 bits
  Distance between Person 61 and Person 87: 138 bits
  Distance between Person 61 and Person 88: 117 bits
  Distance between Person 61 and Person 89: 114 bits
  Distance between Person 62 and Person 63: 118 bits
  Distance between Person 62 and Person 64: 109 bits
  Distance between Person 62 and Person 65: 133 bits
  Distance between Person 62 and Person 66: 149 bits
  Distance between Person 62 and Person 67: 138 bits
  Distance between Person 62 and Person 68: 128 bits
  Distance between Person 62 and Person 69: 126 bits
  Distance between Person 62 and Person 70: 88 bits
  Distance between Person 62 and Person 71: 115 bits
  Distance between Person 62 and Person 72: 132 bits
  Distance between Person 62 and Person 73: 130 bits
  Distance between Person 62 and Person 74: 128 bits
  Distance between Person 62 and Person 75: 125 bits
  Distance between Person 62 and Person 76: 113 bits
  Distance between Person 62 and Person 77: 132 bits
  Distance between Person 62 and Person 78: 108 bits
  Distance between Person 62 and Person 79: 133 bits
  Distance between Person 62 and Person 80: 144 bits
  Distance between Person 62 and Person 81: 124 bits
  Distance between Person 62 and Person 82: 139 bits
  Distance between Person 62 and Person 83: 128 bits
  Distance between Person 62 and Person 84: 120 bits
  Distance between Person 62 and Person 85: 131 bits
  Distance between Person 62 and Person 86: 128 bits
  Distance between Person 62 and Person 87: 130 bits
  Distance between Person 62 and Person 88: 137 bits
  Distance between Person 62 and Person 89: 104 bits
  Distance between Person 63 and Person 64: 145 bits
  Distance between Person 63 and Person 65: 121 bits
  Distance between Person 63 and Person 66: 123 bits
  Distance between Person 63 and Person 67: 108 bits
  Distance between Person 63 and Person 68: 128 bits
  Distance between Person 63 and Person 69: 134 bits
  Distance between Person 63 and Person 70: 140 bits
  Distance between Person 63 and Person 71: 143 bits
  Distance between Person 63 and Person 72: 152 bits
  Distance between Person 63 and Person 73: 132 bits
  Distance between Person 63 and Person 74: 126 bits
  Distance between Person 63 and Person 75: 113 bits
  Distance between Person 63 and Person 76: 101 bits
  Distance between Person 63 and Person 77: 132 bits
  Distance between Person 63 and Person 78: 120 bits
  Distance between Person 63 and Person 79: 145 bits
  Distance between Person 63 and Person 80: 150 bits
  Distance between Person 63 and Person 81: 92 bits
  Distance between Person 63 and Person 82: 115 bits
  Distance between Person 63 and Person 83: 116 bits
  Distance between Person 63 and Person 84: 138 bits
  Distance between Person 63 and Person 85: 135 bits
  Distance between Person 63 and Person 86: 132 bits
  Distance between Person 63 and Person 87: 122 bits
  Distance between Person 63 and Person 88: 133 bits
  Distance between Person 63 and Person 89: 116 bits
  Distance between Person 64 and Person 65: 116 bits
  Distance between Person 64 and Person 66: 100 bits
  Distance between Person 64 and Person 67: 149 bits
  Distance between Person 64 and Person 68: 121 bits
  Distance between Person 64 and Person 69: 121 bits
  Distance between Person 64 and Person 70: 115 bits
  Distance between Person 64 and Person 71: 128 bits
  Distance between Person 64 and Person 72: 121 bits
  Distance between Person 64 and Person 73: 133 bits
  Distance between Person 64 and Person 74: 121 bits
  Distance between Person 64 and Person 75: 120 bits
  Distance between Person 64 and Person 76: 120 bits
  Distance between Person 64 and Person 77: 115 bits
  Distance between Person 64 and Person 78: 133 bits
  Distance between Person 64 and Person 79: 114 bits
  Distance between Person 64 and Person 80: 129 bits
  Distance between Person 64 and Person 81: 139 bits
  Distance between Person 64 and Person 82: 112 bits
  Distance between Person 64 and Person 83: 139 bits
  Distance between Person 64 and Person 84: 139 bits
  Distance between Person 64 and Person 85: 130 bits
  Distance between Person 64 and Person 86: 123 bits
  Distance between Person 64 and Person 87: 129 bits
  Distance between Person 64 and Person 88: 132 bits
  Distance between Person 64 and Person 89: 141 bits
  Distance between Person 65 and Person 66: 98 bits
  Distance between Person 65 and Person 67: 139 bits
  Distance between Person 65 and Person 68: 111 bits
  Distance between Person 65 and Person 69: 123 bits
  Distance between Person 65 and Person 70: 115 bits
  Distance between Person 65 and Person 71: 124 bits
  Distance between Person 65 and Person 72: 115 bits
  Distance between Person 65 and Person 73: 125 bits
  Distance between Person 65 and Person 74: 137 bits
  Distance between Person 65 and Person 75: 120 bits
  Distance between Person 65 and Person 76: 122 bits
  Distance between Person 65 and Person 77: 123 bits
  Distance between Person 65 and Person 78: 123 bits
  Distance between Person 65 and Person 79: 114 bits
  Distance between Person 65 and Person 80: 145 bits
  Distance between Person 65 and Person 81: 157 bits
  Distance between Person 65 and Person 82: 122 bits
  Distance between Person 65 and Person 83: 123 bits
  Distance between Person 65 and Person 84: 115 bits
  Distance between Person 65 and Person 85: 114 bits
  Distance between Person 65 and Person 86: 141 bits
  Distance between Person 65 and Person 87: 115 bits
  Distance between Person 65 and Person 88: 120 bits
  Distance between Person 65 and Person 89: 139 bits
  Distance between Person 66 and Person 67: 125 bits
  Distance between Person 66 and Person 68: 123 bits
  Distance between Person 66 and Person 69: 117 bits
  Distance between Person 66 and Person 70: 121 bits
  Distance between Person 66 and Person 71: 138 bits
  Distance between Person 66 and Person 72: 119 bits
  Distance between Person 66 and Person 73: 143 bits
  Distance between Person 66 and Person 74: 127 bits
  Distance between Person 66 and Person 75: 124 bits
  Distance between Person 66 and Person 76: 122 bits
  Distance between Person 66 and Person 77: 123 bits
  Distance between Person 66 and Person 78: 125 bits
  Distance between Person 66 and Person 79: 108 bits
  Distance between Person 66 and Person 80: 129 bits
  Distance between Person 66 and Person 81: 121 bits
  Distance between Person 66 and Person 82: 114 bits
  Distance between Person 66 and Person 83: 111 bits
  Distance between Person 66 and Person 84: 141 bits
  Distance between Person 66 and Person 85: 114 bits
  Distance between Person 66 and Person 86: 113 bits
  Distance between Person 66 and Person 87: 135 bits
  Distance between Person 66 and Person 88: 134 bits
  Distance between Person 66 and Person 89: 151 bits
  Distance between Person 67 and Person 68: 156 bits
  Distance between Person 67 and Person 69: 118 bits
  Distance between Person 67 and Person 70: 134 bits
  Distance between Person 67 and Person 71: 139 bits
  Distance between Person 67 and Person 72: 132 bits
  Distance between Person 67 and Person 73: 122 bits
  Distance between Person 67 and Person 74: 130 bits
  Distance between Person 67 and Person 75: 135 bits
  Distance between Person 67 and Person 76: 129 bits
  Distance between Person 67 and Person 77: 126 bits
  Distance between Person 67 and Person 78: 116 bits
  Distance between Person 67 and Person 79: 137 bits
  Distance between Person 67 and Person 80: 128 bits
  Distance between Person 67 and Person 81: 106 bits
  Distance between Person 67 and Person 82: 127 bits
  Distance between Person 67 and Person 83: 148 bits
  Distance between Person 67 and Person 84: 132 bits
  Distance between Person 67 and Person 85: 119 bits
  Distance between Person 67 and Person 86: 128 bits
  Distance between Person 67 and Person 87: 138 bits
  Distance between Person 67 and Person 88: 131 bits
  Distance between Person 67 and Person 89: 140 bits
  Distance between Person 68 and Person 69: 144 bits
  Distance between Person 68 and Person 70: 134 bits
  Distance between Person 68 and Person 71: 123 bits
  Distance between Person 68 and Person 72: 128 bits
  Distance between Person 68 and Person 73: 130 bits
  Distance between Person 68 and Person 74: 130 bits
  Distance between Person 68 and Person 75: 135 bits
  Distance between Person 68 and Person 76: 121 bits
  Distance between Person 68 and Person 77: 128 bits
  Distance between Person 68 and Person 78: 128 bits
  Distance between Person 68 and Person 79: 117 bits
  Distance between Person 68 and Person 80: 138 bits
  Distance between Person 68 and Person 81: 130 bits
  Distance between Person 68 and Person 82: 121 bits
  Distance between Person 68 and Person 83: 126 bits
  Distance between Person 68 and Person 84: 104 bits
  Distance between Person 68 and Person 85: 127 bits
  Distance between Person 68 and Person 86: 140 bits
  Distance between Person 68 and Person 87: 118 bits
  Distance between Person 68 and Person 88: 127 bits
  Distance between Person 68 and Person 89: 124 bits
  Distance between Person 69 and Person 70: 128 bits
  Distance between Person 69 and Person 71: 113 bits
  Distance between Person 69 and Person 72: 116 bits
  Distance between Person 69 and Person 73: 110 bits
  Distance between Person 69 and Person 74: 114 bits
  Distance between Person 69 and Person 75: 101 bits
  Distance between Person 69 and Person 76: 121 bits
  Distance between Person 69 and Person 77: 132 bits
  Distance between Person 69 and Person 78: 114 bits
  Distance between Person 69 and Person 79: 121 bits
  Distance between Person 69 and Person 80: 120 bits
  Distance between Person 69 and Person 81: 112 bits
  Distance between Person 69 and Person 82: 119 bits
  Distance between Person 69 and Person 83: 124 bits
  Distance between Person 69 and Person 84: 144 bits
  Distance between Person 69 and Person 85: 127 bits
  Distance between Person 69 and Person 86: 126 bits
  Distance between Person 69 and Person 87: 118 bits
  Distance between Person 69 and Person 88: 123 bits
  Distance between Person 69 and Person 89: 128 bits
  Distance between Person 70 and Person 71: 137 bits
  Distance between Person 70 and Person 72: 118 bits
  Distance between Person 70 and Person 73: 130 bits
  Distance between Person 70 and Person 74: 126 bits
  Distance between Person 70 and Person 75: 105 bits
  Distance between Person 70 and Person 76: 127 bits
  Distance between Person 70 and Person 77: 116 bits
  Distance between Person 70 and Person 78: 112 bits
  Distance between Person 70 and Person 79: 131 bits
  Distance between Person 70 and Person 80: 112 bits
  Distance between Person 70 and Person 81: 142 bits
  Distance between Person 70 and Person 82: 137 bits
  Distance between Person 70 and Person 83: 128 bits
  Distance between Person 70 and Person 84: 132 bits
  Distance between Person 70 and Person 85: 119 bits
  Distance between Person 70 and Person 86: 122 bits
  Distance between Person 70 and Person 87: 134 bits
  Distance between Person 70 and Person 88: 105 bits
  Distance between Person 70 and Person 89: 126 bits
  Distance between Person 71 and Person 72: 137 bits
  Distance between Person 71 and Person 73: 145 bits
  Distance between Person 71 and Person 74: 139 bits
  Distance between Person 71 and Person 75: 128 bits
  Distance between Person 71 and Person 76: 134 bits
  Distance between Person 71 and Person 77: 129 bits
  Distance between Person 71 and Person 78: 129 bits
  Distance between Person 71 and Person 79: 112 bits
  Distance between Person 71 and Person 80: 121 bits
  Distance between Person 71 and Person 81: 137 bits
  Distance between Person 71 and Person 82: 134 bits
  Distance between Person 71 and Person 83: 115 bits
  Distance between Person 71 and Person 84: 123 bits
  Distance between Person 71 and Person 85: 118 bits
  Distance between Person 71 and Person 86: 107 bits
  Distance between Person 71 and Person 87: 139 bits
  Distance between Person 71 and Person 88: 118 bits
  Distance between Person 71 and Person 89: 127 bits
  Distance between Person 72 and Person 73: 124 bits
  Distance between Person 72 and Person 74: 120 bits
  Distance between Person 72 and Person 75: 133 bits
  Distance between Person 72 and Person 76: 135 bits
  Distance between Person 72 and Person 77: 134 bits
  Distance between Person 72 and Person 78: 126 bits
  Distance between Person 72 and Person 79: 121 bits
  Distance between Person 72 and Person 80: 114 bits
  Distance between Person 72 and Person 81: 138 bits
  Distance between Person 72 and Person 82: 133 bits
  Distance between Person 72 and Person 83: 130 bits
  Distance between Person 72 and Person 84: 124 bits
  Distance between Person 72 and Person 85: 113 bits
  Distance between Person 72 and Person 86: 108 bits
  Distance between Person 72 and Person 87: 68 bits
  Distance between Person 72 and Person 88: 127 bits
  Distance between Person 72 and Person 89: 136 bits
  Distance between Person 73 and Person 74: 140 bits
  Distance between Person 73 and Person 75: 125 bits
  Distance between Person 73 and Person 76: 137 bits
  Distance between Person 73 and Person 77: 138 bits
  Distance between Person 73 and Person 78: 108 bits
  Distance between Person 73 and Person 79: 149 bits
  Distance between Person 73 and Person 80: 122 bits
  Distance between Person 73 and Person 81: 120 bits
  Distance between Person 73 and Person 82: 119 bits
  Distance between Person 73 and Person 83: 144 bits
  Distance between Person 73 and Person 84: 136 bits
  Distance between Person 73 and Person 85: 135 bits
  Distance between Person 73 and Person 86: 120 bits
  Distance between Person 73 and Person 87: 128 bits
  Distance between Person 73 and Person 88: 137 bits
  Distance between Person 73 and Person 89: 116 bits
  Distance between Person 74 and Person 75: 141 bits
  Distance between Person 74 and Person 76: 119 bits
  Distance between Person 74 and Person 77: 134 bits
  Distance between Person 74 and Person 78: 126 bits
  Distance between Person 74 and Person 79: 121 bits
  Distance between Person 74 and Person 80: 128 bits
  Distance between Person 74 and Person 81: 136 bits
  Distance between Person 74 and Person 82: 137 bits
  Distance between Person 74 and Person 83: 124 bits
  Distance between Person 74 and Person 84: 116 bits
  Distance between Person 74 and Person 85: 117 bits
  Distance between Person 74 and Person 86: 138 bits
  Distance between Person 74 and Person 87: 128 bits
  Distance between Person 74 and Person 88: 131 bits
  Distance between Person 74 and Person 89: 130 bits
  Distance between Person 75 and Person 76: 122 bits
  Distance between Person 75 and Person 77: 119 bits
  Distance between Person 75 and Person 78: 133 bits
  Distance between Person 75 and Person 79: 114 bits
  Distance between Person 75 and Person 80: 125 bits
  Distance between Person 75 and Person 81: 115 bits
  Distance between Person 75 and Person 82: 124 bits
  Distance between Person 75 and Person 83: 139 bits
  Distance between Person 75 and Person 84: 157 bits
  Distance between Person 75 and Person 85: 148 bits
  Distance between Person 75 and Person 86: 135 bits
  Distance between Person 75 and Person 87: 125 bits
  Distance between Person 75 and Person 88: 110 bits
  Distance between Person 75 and Person 89: 113 bits
  Distance between Person 76 and Person 77: 129 bits
  Distance between Person 76 and Person 78: 121 bits
  Distance between Person 76 and Person 79: 130 bits
  Distance between Person 76 and Person 80: 137 bits
  Distance between Person 76 and Person 81: 121 bits
  Distance between Person 76 and Person 82: 122 bits
  Distance between Person 76 and Person 83: 151 bits
  Distance between Person 76 and Person 84: 125 bits
  Distance between Person 76 and Person 85: 130 bits
  Distance between Person 76 and Person 86: 145 bits
  Distance between Person 76 and Person 87: 125 bits
  Distance between Person 76 and Person 88: 124 bits
  Distance between Person 76 and Person 89: 121 bits
  Distance between Person 77 and Person 78: 112 bits
  Distance between Person 77 and Person 79: 115 bits
  Distance between Person 77 and Person 80: 120 bits
  Distance between Person 77 and Person 81: 134 bits
  Distance between Person 77 and Person 82: 141 bits
  Distance between Person 77 and Person 83: 134 bits
  Distance between Person 77 and Person 84: 138 bits
  Distance between Person 77 and Person 85: 137 bits
  Distance between Person 77 and Person 86: 134 bits
  Distance between Person 77 and Person 87: 136 bits
  Distance between Person 77 and Person 88: 131 bits
  Distance between Person 77 and Person 89: 142 bits
  Distance between Person 78 and Person 79: 119 bits
  Distance between Person 78 and Person 80: 128 bits
  Distance between Person 78 and Person 81: 114 bits
  Distance between Person 78 and Person 82: 109 bits
  Distance between Person 78 and Person 83: 138 bits
  Distance between Person 78 and Person 84: 142 bits
  Distance between Person 78 and Person 85: 135 bits
  Distance between Person 78 and Person 86: 110 bits
  Distance between Person 78 and Person 87: 132 bits
  Distance between Person 78 and Person 88: 123 bits
  Distance between Person 78 and Person 89: 146 bits
  Distance between Person 79 and Person 80: 137 bits
  Distance between Person 79 and Person 81: 139 bits
  Distance between Person 79 and Person 82: 128 bits
  Distance between Person 79 and Person 83: 123 bits
  Distance between Person 79 and Person 84: 101 bits
  Distance between Person 79 and Person 85: 126 bits
  Distance between Person 79 and Person 86: 151 bits
  Distance between Person 79 and Person 87: 129 bits
  Distance between Person 79 and Person 88: 116 bits
  Distance between Person 79 and Person 89: 129 bits
  Distance between Person 80 and Person 81: 140 bits
  Distance between Person 80 and Person 82: 135 bits
  Distance between Person 80 and Person 83: 126 bits
  Distance between Person 80 and Person 84: 118 bits
  Distance between Person 80 and Person 85: 131 bits
  Distance between Person 80 and Person 86: 90 bits
  Distance between Person 80 and Person 87: 150 bits
  Distance between Person 80 and Person 88: 137 bits
  Distance between Person 80 and Person 89: 138 bits
  Distance between Person 81 and Person 82: 135 bits
  Distance between Person 81 and Person 83: 136 bits
  Distance between Person 81 and Person 84: 144 bits
  Distance between Person 81 and Person 85: 149 bits
  Distance between Person 81 and Person 86: 120 bits
  Distance between Person 81 and Person 87: 128 bits
  Distance between Person 81 and Person 88: 129 bits
  Distance between Person 81 and Person 89: 128 bits
  Distance between Person 82 and Person 83: 117 bits
  Distance between Person 82 and Person 84: 133 bits
  Distance between Person 82 and Person 85: 148 bits
  Distance between Person 82 and Person 86: 133 bits
  Distance between Person 82 and Person 87: 127 bits
  Distance between Person 82 and Person 88: 120 bits
  Distance between Person 82 and Person 89: 119 bits
  Distance between Person 83 and Person 84: 124 bits
  Distance between Person 83 and Person 85: 125 bits
  Distance between Person 83 and Person 86: 128 bits
  Distance between Person 83 and Person 87: 118 bits
  Distance between Person 83 and Person 88: 127 bits
  Distance between Person 83 and Person 89: 116 bits
  Distance between Person 84 and Person 85: 107 bits
  Distance between Person 84 and Person 86: 140 bits
  Distance between Person 84 and Person 87: 134 bits
  Distance between Person 84 and Person 88: 121 bits
  Distance between Person 84 and Person 89: 130 bits
  Distance between Person 85 and Person 86: 119 bits
  Distance between Person 85 and Person 87: 133 bits
  Distance between Person 85 and Person 88: 138 bits
  Distance between Person 85 and Person 89: 143 bits
  Distance between Person 86 and Person 87: 138 bits
  Distance between Person 86 and Person 88: 133 bits
  Distance between Person 86 and Person 89: 138 bits
  Distance between Person 87 and Person 88: 107 bits
  Distance between Person 87 and Person 89: 118 bits
  Distance between Person 88 and Person 89: 121 bits

Process finished with exit code 0

/opt/anaconda3/envs/ECG_resnet/bin/python /Users/lucianomaldonado/ECG-PV-GENERATION-GROUND-KEY/VIT/1dconv-vit.py 
Initializing system...
Loaded 25 segments from Person_01
Loaded 28 segments from Person_02
Loaded 24 segments from Person_03
Loaded 25 segments from Person_04
Loaded 19 segments from Person_05
Loaded 24 segments from Person_06
Invalid ECG format in segment_23.csv
Loaded 22 segments from Person_07
Loaded 30 segments from Person_08
Loaded 20 segments from Person_09
Loaded 27 segments from Person_10
Loaded 26 segments from Person_11
Loaded 30 segments from Person_12
Loaded 15 segments from Person_13
Loaded 20 segments from Person_14
Invalid ECG format in segment_26.csv
Loaded 25 segments from Person_15
Loaded 29 segments from Person_16
Loaded 22 segments from Person_17
Loaded 24 segments from Person_18
Loaded 28 segments from Person_19
Loaded 30 segments from Person_20
Invalid ECG format in segment_28.csv
Loaded 27 segments from Person_21
Loaded 27 segments from Person_22
Loaded 20 segments from Person_23
Loaded 25 segments from Person_24
Loaded 24 segments from Person_25
Invalid ECG format in segment_25.csv
Loaded 24 segments from Person_26
Loaded 25 segments from Person_27
Invalid ECG format in segment_25.csv
Loaded 24 segments from Person_28
Loaded 28 segments from Person_29
Loaded 24 segments from Person_30
Loaded 30 segments from Person_31
Loaded 22 segments from Person_32
Loaded 25 segments from Person_33
Invalid ECG format in segment_30.csv
Loaded 29 segments from Person_34
Loaded 27 segments from Person_35
Loaded 20 segments from Person_36
Loaded 24 segments from Person_37
Loaded 24 segments from Person_38
Loaded 20 segments from Person_39
Loaded 29 segments from Person_40
Loaded 23 segments from Person_41
Loaded 28 segments from Person_42
Loaded 23 segments from Person_43
Invalid ECG format in segment_25.csv
Loaded 24 segments from Person_44
Loaded 28 segments from Person_45
Invalid ECG format in segment_28.csv
Loaded 27 segments from Person_46
Loaded 28 segments from Person_47
Invalid ECG format in segment_28.csv
Loaded 27 segments from Person_48
Loaded 31 segments from Person_49
Loaded 31 segments from Person_50
Loaded 19 segments from Person_51
Loaded 26 segments from Person_52
Loaded 25 segments from Person_53
Loaded 25 segments from Person_54
Loaded 25 segments from Person_55
Invalid ECG format in segment_23.csv
Loaded 22 segments from Person_56
Loaded 30 segments from Person_57
Loaded 20 segments from Person_58
Loaded 27 segments from Person_59
Invalid ECG format in segment_29.csv
Loaded 28 segments from Person_60
Loaded 30 segments from Person_61
Loaded 24 segments from Person_62
Loaded 27 segments from Person_63
Loaded 22 segments from Person_64
Invalid ECG format in segment_30.csv
Loaded 29 segments from Person_65
Invalid ECG format in segment_30.csv
Loaded 29 segments from Person_66
Invalid ECG format in segment_22.csv
Loaded 21 segments from Person_67
Loaded 29 segments from Person_68
Loaded 26 segments from Person_69
Invalid ECG format in segment_24.csv
Loaded 23 segments from Person_70
Loaded 27 segments from Person_71
Loaded 25 segments from Person_72
Loaded 24 segments from Person_73
Loaded 36 segments from Person_74
Invalid ECG format in segment_27.csv
Loaded 26 segments from Person_75
Invalid ECG format in segment_36.csv
Loaded 35 segments from Person_76
Loaded 25 segments from Person_77
Invalid ECG format in segment_23.csv
Loaded 22 segments from Person_78
Loaded 33 segments from Person_79
Loaded 25 segments from Person_80
Loaded 23 segments from Person_81
Loaded 30 segments from Person_82
Invalid ECG format in segment_33.csv
Loaded 32 segments from Person_83
Invalid ECG format in segment_26.csv
Loaded 25 segments from Person_84
Loaded 33 segments from Person_85
Loaded 30 segments from Person_86
Loaded 24 segments from Person_87
Loaded 21 segments from Person_88
Loaded 26 segments from Person_89
Dataset contains 89 persons with 2290 total segments

Starting training...
X_train shape: (1832, 170, 1)
Epoch 1/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 5s 26ms/step - accuracy: 0.0022 - loss: 0.6977 - val_accuracy: 0.0000e+00 - val_loss: 0.6906
Epoch 2/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 1s 24ms/step - accuracy: 0.0000e+00 - loss: 0.6905 - val_accuracy: 0.0000e+00 - val_loss: 0.6882
Epoch 3/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 27ms/step - accuracy: 5.8776e-04 - loss: 0.6885 - val_accuracy: 0.0000e+00 - val_loss: 0.6863
Epoch 4/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 2.1738e-04 - loss: 0.6866 - val_accuracy: 0.0000e+00 - val_loss: 0.6843
Epoch 5/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0000e+00 - loss: 0.6846 - val_accuracy: 0.0000e+00 - val_loss: 0.6823
Epoch 6/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - accuracy: 0.0000e+00 - loss: 0.6819 - val_accuracy: 0.0000e+00 - val_loss: 0.6804
Epoch 7/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 30ms/step - accuracy: 0.0000e+00 - loss: 0.6801 - val_accuracy: 0.0000e+00 - val_loss: 0.6786
Epoch 8/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0000e+00 - loss: 0.6788 - val_accuracy: 0.0000e+00 - val_loss: 0.6767
Epoch 9/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 29ms/step - accuracy: 0.0000e+00 - loss: 0.6775 - val_accuracy: 0.0000e+00 - val_loss: 0.6749
Epoch 10/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0000e+00 - loss: 0.6747 - val_accuracy: 0.0000e+00 - val_loss: 0.6734
Epoch 11/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 30ms/step - accuracy: 0.0000e+00 - loss: 0.6739 - val_accuracy: 0.0000e+00 - val_loss: 0.6716
Epoch 12/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0000e+00 - loss: 0.6716 - val_accuracy: 0.0000e+00 - val_loss: 0.6696
Epoch 13/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 30ms/step - accuracy: 0.0000e+00 - loss: 0.6696 - val_accuracy: 0.0000e+00 - val_loss: 0.6673
Epoch 14/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - accuracy: 0.0000e+00 - loss: 0.6670 - val_accuracy: 0.0000e+00 - val_loss: 0.6654
Epoch 15/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0000e+00 - loss: 0.6655 - val_accuracy: 0.0000e+00 - val_loss: 0.6617
Epoch 16/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - accuracy: 0.0000e+00 - loss: 0.6615 - val_accuracy: 0.0000e+00 - val_loss: 0.6585
Epoch 17/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0000e+00 - loss: 0.6589 - val_accuracy: 0.0000e+00 - val_loss: 0.6548
Epoch 18/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 30ms/step - accuracy: 0.0000e+00 - loss: 0.6541 - val_accuracy: 0.0000e+00 - val_loss: 0.6518
Epoch 19/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0000e+00 - loss: 0.6519 - val_accuracy: 0.0000e+00 - val_loss: 0.6481
Epoch 20/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0000e+00 - loss: 0.6481 - val_accuracy: 0.0000e+00 - val_loss: 0.6444
Epoch 21/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0000e+00 - loss: 0.6453 - val_accuracy: 0.0022 - val_loss: 0.6407
Epoch 22/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0020 - loss: 0.6399 - val_accuracy: 0.0000e+00 - val_loss: 0.6373
Epoch 23/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 1.9122e-04 - loss: 0.6367 - val_accuracy: 0.0000e+00 - val_loss: 0.6332
Epoch 24/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0029 - loss: 0.6321 - val_accuracy: 0.0000e+00 - val_loss: 0.6295
Epoch 25/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - accuracy: 0.0024 - loss: 0.6295 - val_accuracy: 0.0022 - val_loss: 0.6265
Epoch 26/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0041 - loss: 0.6254 - val_accuracy: 0.0044 - val_loss: 0.6221
Epoch 27/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0049 - loss: 0.6222 - val_accuracy: 0.0022 - val_loss: 0.6190
Epoch 28/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0066 - loss: 0.6192 - val_accuracy: 0.0175 - val_loss: 0.6160
Epoch 29/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0100 - loss: 0.6158 - val_accuracy: 0.0022 - val_loss: 0.6121
Epoch 30/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 38ms/step - accuracy: 0.0059 - loss: 0.6120 - val_accuracy: 0.0044 - val_loss: 0.6083
Epoch 31/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0075 - loss: 0.6064 - val_accuracy: 0.0087 - val_loss: 0.6044
Epoch 32/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0144 - loss: 0.6054 - val_accuracy: 0.0066 - val_loss: 0.6021
Epoch 33/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0101 - loss: 0.5997 - val_accuracy: 0.0218 - val_loss: 0.5980
Epoch 34/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0147 - loss: 0.5955 - val_accuracy: 0.0153 - val_loss: 0.5939
Epoch 35/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0177 - loss: 0.5927 - val_accuracy: 0.0153 - val_loss: 0.5917
Epoch 36/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 3s 44ms/step - accuracy: 0.0135 - loss: 0.5908 - val_accuracy: 0.0066 - val_loss: 0.5873
Epoch 37/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 40ms/step - accuracy: 0.0098 - loss: 0.5853 - val_accuracy: 0.0109 - val_loss: 0.5829
Epoch 38/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0122 - loss: 0.5836 - val_accuracy: 0.0175 - val_loss: 0.5799
Epoch 39/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0179 - loss: 0.5792 - val_accuracy: 0.0087 - val_loss: 0.5779
Epoch 40/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step - accuracy: 0.0165 - loss: 0.5749 - val_accuracy: 0.0197 - val_loss: 0.5736
Epoch 41/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0123 - loss: 0.5726 - val_accuracy: 0.0022 - val_loss: 0.5713
Epoch 42/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 39ms/step - accuracy: 0.0133 - loss: 0.5685 - val_accuracy: 0.0109 - val_loss: 0.5670
Epoch 43/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0134 - loss: 0.5632 - val_accuracy: 0.0066 - val_loss: 0.5629
Epoch 44/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0153 - loss: 0.5645 - val_accuracy: 0.0153 - val_loss: 0.5598
Epoch 45/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 37ms/step - accuracy: 0.0122 - loss: 0.5581 - val_accuracy: 0.0087 - val_loss: 0.5564
Epoch 46/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0090 - loss: 0.5542 - val_accuracy: 0.0087 - val_loss: 0.5531
Epoch 47/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0134 - loss: 0.5497 - val_accuracy: 0.0022 - val_loss: 0.5496
Epoch 48/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0035 - loss: 0.5495 - val_accuracy: 0.0044 - val_loss: 0.5467
Epoch 49/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0029 - loss: 0.5411 - val_accuracy: 0.0022 - val_loss: 0.5431
Epoch 50/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0065 - loss: 0.5372 - val_accuracy: 0.0000e+00 - val_loss: 0.5390
Epoch 51/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0043 - loss: 0.5368 - val_accuracy: 0.0022 - val_loss: 0.5363
Epoch 52/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 40ms/step - accuracy: 0.0046 - loss: 0.5348 - val_accuracy: 0.0000e+00 - val_loss: 0.5317
Epoch 53/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0026 - loss: 0.5258 - val_accuracy: 0.0022 - val_loss: 0.5319
Epoch 54/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0055 - loss: 0.5271 - val_accuracy: 0.0022 - val_loss: 0.5252
Epoch 55/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 37ms/step - accuracy: 0.0041 - loss: 0.5216 - val_accuracy: 0.0022 - val_loss: 0.5222
Epoch 56/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0015 - loss: 0.5195 - val_accuracy: 0.0000e+00 - val_loss: 0.5195
Epoch 57/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0055 - loss: 0.5167 - val_accuracy: 0.0000e+00 - val_loss: 0.5157
Epoch 58/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0042 - loss: 0.5119 - val_accuracy: 0.0022 - val_loss: 0.5133
Epoch 59/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0049 - loss: 0.5103 - val_accuracy: 0.0044 - val_loss: 0.5095
Epoch 60/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 38ms/step - accuracy: 0.0053 - loss: 0.5030 - val_accuracy: 0.0022 - val_loss: 0.5054
Epoch 61/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0039 - loss: 0.5021 - val_accuracy: 0.0000e+00 - val_loss: 0.5024
Epoch 62/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0013 - loss: 0.4976 - val_accuracy: 0.0044 - val_loss: 0.4997
Epoch 63/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 9.9202e-04 - loss: 0.4960 - val_accuracy: 0.0022 - val_loss: 0.4990
Epoch 64/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0034 - loss: 0.4938 - val_accuracy: 0.0000e+00 - val_loss: 0.4946
Epoch 65/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step - accuracy: 0.0025 - loss: 0.4909 - val_accuracy: 0.0000e+00 - val_loss: 0.4914
Epoch 66/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0019 - loss: 0.4862 - val_accuracy: 0.0000e+00 - val_loss: 0.4862
Epoch 67/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0037 - loss: 0.4820 - val_accuracy: 0.0000e+00 - val_loss: 0.4862
Epoch 68/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0025 - loss: 0.4814 - val_accuracy: 0.0022 - val_loss: 0.4821
Epoch 69/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0022 - loss: 0.4717 - val_accuracy: 0.0000e+00 - val_loss: 0.4780
Epoch 70/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 37ms/step - accuracy: 0.0019 - loss: 0.4712 - val_accuracy: 0.0022 - val_loss: 0.4742
Epoch 71/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0025 - loss: 0.4674 - val_accuracy: 0.0022 - val_loss: 0.4727
Epoch 72/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0014 - loss: 0.4632 - val_accuracy: 0.0022 - val_loss: 0.4678
Epoch 73/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 5.8754e-04 - loss: 0.4610 - val_accuracy: 0.0022 - val_loss: 0.4658
Epoch 74/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0012 - loss: 0.4587 - val_accuracy: 0.0000e+00 - val_loss: 0.4628
Epoch 75/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0040 - loss: 0.4492 - val_accuracy: 0.0000e+00 - val_loss: 0.4591
Epoch 76/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step - accuracy: 0.0042 - loss: 0.4510 - val_accuracy: 0.0022 - val_loss: 0.4556
Epoch 77/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0038 - loss: 0.4498 - val_accuracy: 0.0000e+00 - val_loss: 0.4525
Epoch 78/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 6.0922e-04 - loss: 0.4455 - val_accuracy: 0.0022 - val_loss: 0.4517
Epoch 79/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 35ms/step - accuracy: 0.0043 - loss: 0.4415 - val_accuracy: 0.0022 - val_loss: 0.4475
Epoch 80/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0032 - loss: 0.4346 - val_accuracy: 0.0022 - val_loss: 0.4431
Epoch 81/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 36ms/step - accuracy: 0.0047 - loss: 0.4323 - val_accuracy: 0.0000e+00 - val_loss: 0.4424
Epoch 82/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0024 - loss: 0.4332 - val_accuracy: 0.0022 - val_loss: 0.4391
Epoch 83/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0012 - loss: 0.4256 - val_accuracy: 0.0066 - val_loss: 0.4357
Epoch 84/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0065 - loss: 0.4262 - val_accuracy: 0.0000e+00 - val_loss: 0.4336
Epoch 85/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0037 - loss: 0.4210 - val_accuracy: 0.0044 - val_loss: 0.4292
Epoch 86/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0070 - loss: 0.4245 - val_accuracy: 0.0066 - val_loss: 0.4285
Epoch 87/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0036 - loss: 0.4131 - val_accuracy: 0.0022 - val_loss: 0.4239
Epoch 88/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0048 - loss: 0.4142 - val_accuracy: 0.0066 - val_loss: 0.4213
Epoch 89/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0052 - loss: 0.4091 - val_accuracy: 0.0109 - val_loss: 0.4190
Epoch 90/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0057 - loss: 0.4056 - val_accuracy: 0.0066 - val_loss: 0.4171
Epoch 91/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0079 - loss: 0.4067 - val_accuracy: 0.0087 - val_loss: 0.4174
Epoch 92/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0088 - loss: 0.4039 - val_accuracy: 0.0022 - val_loss: 0.4099
Epoch 93/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0063 - loss: 0.3990 - val_accuracy: 0.0087 - val_loss: 0.4096
Epoch 94/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0078 - loss: 0.3985 - val_accuracy: 0.0066 - val_loss: 0.4054
Epoch 95/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0099 - loss: 0.3888 - val_accuracy: 0.0087 - val_loss: 0.4021
Epoch 96/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0098 - loss: 0.3895 - val_accuracy: 0.0087 - val_loss: 0.4003
Epoch 97/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0078 - loss: 0.3837 - val_accuracy: 0.0066 - val_loss: 0.4016
Epoch 98/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 34ms/step - accuracy: 0.0039 - loss: 0.3810 - val_accuracy: 0.0044 - val_loss: 0.3976
Epoch 99/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 33ms/step - accuracy: 0.0018 - loss: 0.3843 - val_accuracy: 0.0066 - val_loss: 0.3940
Epoch 100/100
58/58 ━━━━━━━━━━━━━━━━━━━━ 2s 32ms/step - accuracy: 0.0080 - loss: 0.3814 - val_accuracy: 0.0022 - val_loss: 0.3909

Testing key generation for all persons:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 144ms/step

Person 1:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1]...
  Ground Truth:   [1 1 0 1 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 24.97 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 148ms/step

Person 2:
  Aggregated Key Accuracy: 85.16%
  Aggregated Key: [1 1 0 0 1 1 1 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 56.09 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 3:
  Aggregated Key Accuracy: 85.16%
  Aggregated Key: [1 0 1 1 1 1 1 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 0 1 1 0 1 0 0 1 1 0 1 0 1 1 0 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 28.43 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 4:
  Aggregated Key Accuracy: 76.17%
  Aggregated Key: [1 1 0 1 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 0]...
  Ground Truth:   [1 1 1 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 38.89 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 5:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 0]...
  Ground Truth:   [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 22.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 6:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 0 1 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 44.86 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 7:
  Aggregated Key Accuracy: 83.98%
  Aggregated Key: [0 0 1 0 0 1 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 1]...
  Ground Truth:   [0 1 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 56.90 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 8:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 13.26 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 9:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [1 0 1 1 1 0 0 1 1 1 0 0 1 1 0 1 1 1 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 1 0 0 1 1 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 30.07 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 10:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 24.83 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 11:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [1 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
  Ground Truth:   [0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 39.89 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 12:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
  Ground Truth:   [0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 8.94 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 13:
  Aggregated Key Accuracy: 79.69%
  Aggregated Key: [0 1 0 1 1 0 1 0 1 1 0 0 0 1 0 1 1 1 0 1 0 1 0 0]...
  Ground Truth:   [0 1 0 1 1 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 42.53 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 14:
  Aggregated Key Accuracy: 78.12%
  Aggregated Key: [1 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
  Ground Truth:   [1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 65.86 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 15:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [1 1 0 1 0 0 0 1 1 1 0 1 0 1 0 1 0 1 1 1 0 0 0 0]...
  Ground Truth:   [1 1 0 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 23.57 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 16:
  Aggregated Key Accuracy: 75.78%
  Aggregated Key: [1 1 0 1 0 1 0 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1]...
  Ground Truth:   [0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 71.42 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 17:
  Aggregated Key Accuracy: 91.80%
  Aggregated Key: [1 0 0 0 1 0 1 0 1 1 1 0 1 1 0 1 0 1 1 0 0 0 1 1]...
  Ground Truth:   [1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 24.72 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 18:
  Aggregated Key Accuracy: 87.89%
  Aggregated Key: [1 0 0 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 1 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 43.87 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 19:
  Aggregated Key Accuracy: 85.94%
  Aggregated Key: [1 0 1 0 0 1 1 1 1 1 1 1 0 1 0 1 0 1 0 0 0 1 0 0]...
  Ground Truth:   [1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 0 0 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 55.45 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 20:
  Aggregated Key Accuracy: 99.22%
  Aggregated Key: [1 0 1 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1]...
  Ground Truth:   [1 0 1 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 26.44 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 21:
  Aggregated Key Accuracy: 81.64%
  Aggregated Key: [1 0 0 0 1 1 1 1 1 0 0 0 0 0 1 1 1 0 1 1 0 1 1 0]...
  Ground Truth:   [1 0 0 0 0 1 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 42.59 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 22:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
  Ground Truth:   [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 17.25 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 23:
  Aggregated Key Accuracy: 84.38%
  Aggregated Key: [1 0 0 0 0 1 0 1 0 1 0 1 1 0 1 1 1 0 1 1 1 0 0 1]...
  Ground Truth:   [1 0 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 19.77 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 24:
  Aggregated Key Accuracy: 77.73%
  Aggregated Key: [0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0]...
  Ground Truth:   [0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 62.37 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 25:
  Aggregated Key Accuracy: 78.12%
  Aggregated Key: [1 0 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 68.47 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 26:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 0 0 0]...
  Ground Truth:   [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 19.25 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 27:
  Aggregated Key Accuracy: 89.84%
  Aggregated Key: [1 0 0 0 1 1 1 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 32.59 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 28:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 1 0 1 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 38.90 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 29:
  Aggregated Key Accuracy: 95.70%
  Aggregated Key: [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 21.20 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 30:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 0]...
  Ground Truth:   [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 27.83 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 31:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [0 1 1 0 0 0 0 1 1 1 0 0 1 1 1 1 0 1 0 0 1 0 0 1]...
  Ground Truth:   [0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 13.57 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 32:
  Aggregated Key Accuracy: 85.94%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0]...
  Ground Truth:   [1 1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 45.72 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 33:
  Aggregated Key Accuracy: 79.69%
  Aggregated Key: [1 0 1 0 1 1 1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 1 0 0]...
  Ground Truth:   [1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 40.60 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 34:
  Aggregated Key Accuracy: 88.28%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 0 1 1 0 0 1 0 1 0 1 1 1 0 0]...
  Ground Truth:   [1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 1 0 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 27.79 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 35:
  Aggregated Key Accuracy: 96.48%
  Aggregated Key: [1 0 0 0 1 0 1 1 1 0 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
  Ground Truth:   [1 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 14.74 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 36:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 1 0 1 1 0 0 0 1 0 1 0 0 0 1 0 1 1 1 1 0 0 1 1]...
  Ground Truth:   [0 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 22.29 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 37:
  Aggregated Key Accuracy: 82.42%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 1 1 0 0 1 0 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 48.41 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 38:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 47.39 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 39:
  Aggregated Key Accuracy: 69.53%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 1 1 0 0 0 1 0 1 0 0 0 1 1 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 64.91 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 40:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [1 0 0 0 1 1 0 1 1 1 1 1 1 0 0 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 51.39 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 41:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0]...
  Ground Truth:   [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
  Intra-person average Hamming distance: 9.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 42:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [0 1 0 0 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1 0 0 1 0 1]...
  Ground Truth:   [0 1 1 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 50.09 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 43:
  Aggregated Key Accuracy: 86.33%
  Aggregated Key: [1 1 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 1 1 1 0 0 0 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 50.66 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 44:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 0 1 0 1 0 1 1 1 1 0 1 0 1 1 0 1 0 0 1 1 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 45.97 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 45:
  Aggregated Key Accuracy: 86.72%
  Aggregated Key: [0 1 0 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 0 0 1 0 1 0]...
  Ground Truth:   [0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 1 1 0 1 0 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 54.21 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 46:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 1 1 1]...
  Ground Truth:   [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 19.82 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 47:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [0 0 1 0 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 1 0]...
  Ground Truth:   [0 0 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 58.99 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 48:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [1 0 1 1 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
  Ground Truth:   [1 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 7.03 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 49:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
  Ground Truth:   [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
  Intra-person average Hamming distance: 17.78 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 50:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 11.52 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 51:
  Aggregated Key Accuracy: 76.95%
  Aggregated Key: [0 1 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 0 1 1]...
  Ground Truth:   [0 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 24.60 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 52:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 0 1 1 1 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 19.54 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 53:
  Aggregated Key Accuracy: 94.14%
  Aggregated Key: [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
  Ground Truth:   [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 20.45 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 54:
  Aggregated Key Accuracy: 86.33%
  Aggregated Key: [0 0 0 1 1 0 0 1 1 0 0 0 1 0 1 1 0 0 0 1 1 1 0 0]...
  Ground Truth:   [0 0 0 0 1 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 29.85 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 55:
  Aggregated Key Accuracy: 74.22%
  Aggregated Key: [0 1 0 1 0 0 0 1 0 1 1 0 0 0 0 1 0 1 1 1 1 1 0 0]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 1 1 1 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 64.76 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 56:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [0 0 0 0 1 1 0 0 1 0 0 1 0 1 1 1 1 0 0 0 0 0 1 0]...
  Ground Truth:   [0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 50.48 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 57:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
  Ground Truth:   [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 10.23 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step

Person 58:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 0 1 1 0 1 0 1 1 0 1 1 1 1 0 1 1 0 1 0 1 0 0 1]...
  Ground Truth:   [0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 20.53 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 59:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 0 0 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
  Intra-person average Hamming distance: 15.99 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 60:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [0 0 0 1 1 0 0 1 1 0 0 1 0 0 1 0 1 1 0 1 0 1 1 1]...
  Ground Truth:   [0 0 0 1 1 0 0 1 0 0 0 1 0 0 1 0 1 1 0 1 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 26.32 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 61:
  Aggregated Key Accuracy: 83.98%
  Aggregated Key: [0 0 1 0 0 0 0 1 0 1 0 0 0 1 1 1 0 0 0 0 1 1 1 1]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 47.05 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step

Person 62:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0]...
  Ground Truth:   [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 30.22 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 63:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 18.25 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 64:
  Aggregated Key Accuracy: 86.72%
  Aggregated Key: [0 1 0 1 1 0 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 28.85 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 65:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
  Ground Truth:   [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 19.88 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step

Person 66:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 41.73 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 67:
  Aggregated Key Accuracy: 80.47%
  Aggregated Key: [1 0 1 0 1 1 1 0 1 1 0 0 1 0 0 1 1 1 0 0 1 1 0 1]...
  Ground Truth:   [0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 21.60 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Person 68:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [1 1 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 1 1 0 0 1 1]...
  Ground Truth:   [0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
  Intra-person average Hamming distance: 20.27 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step

Person 69:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [0 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 27.40 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 70:
  Aggregated Key Accuracy: 73.05%
  Aggregated Key: [0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 1 1 1 1 1 0 1 1 0]...
  Ground Truth:   [0 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 44.40 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step

Person 71:
  Aggregated Key Accuracy: 89.84%
  Aggregated Key: [1 0 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
  Ground Truth:   [1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 28.91 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 72:
  Aggregated Key Accuracy: 90.23%
  Aggregated Key: [0 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
  Ground Truth:   [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 15.27 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Person 73:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [1 0 1 1 0 1 1 1 0 1 1 1 1 1 0 1 0 1 0 0 0 1 0 0]...
  Ground Truth:   [1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 44.79 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

Person 74:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
  Intra-person average Hamming distance: 0.33 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 75:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 1 0 0 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 0 0 1 0]...
  Ground Truth:   [0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
  Intra-person average Hamming distance: 55.93 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 

Person 76:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
  Ground Truth:   [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 
  Intra-person average Hamming distance: 1.67 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 77:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 14.35 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 78:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [1 1 0 1 1 0 1 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 24.50 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 

Person 79:
  Aggregated Key Accuracy: 96.48%
  Aggregated Key: [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 0]...
  Ground Truth:   [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
  Intra-person average Hamming distance: 16.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 80:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 0 0 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 1 1 0 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step
  Intra-person average Hamming distance: 30.49 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 81:
  Aggregated Key Accuracy: 87.50%
  Aggregated Key: [1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 0 1]...
  Ground Truth:   [1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
  Intra-person average Hamming distance: 35.49 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step

Person 82:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
  Intra-person average Hamming distance: 9.48 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step

Person 83:
  Aggregated Key Accuracy: 97.27%
  Aggregated Key: [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
  Intra-person average Hamming distance: 10.16 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step

Person 84:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
  Intra-person average Hamming distance: 44.09 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 

Person 85:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
  Intra-person average Hamming distance: 12.05 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Person 86:
  Aggregated Key Accuracy: 77.73%
  Aggregated Key: [0 0 1 0 0 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1]...
  Ground Truth:   [0 0 0 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
  Intra-person average Hamming distance: 59.82 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 87:
  Aggregated Key Accuracy: 87.11%
  Aggregated Key: [0 1 0 1 0 0 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 0 0 1]...
  Ground Truth:   [0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
  Intra-person average Hamming distance: 21.70 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step

Person 88:
  Aggregated Key Accuracy: 88.28%
  Aggregated Key: [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1]...
  Ground Truth:   [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
  Intra-person average Hamming distance: 32.59 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step

Person 89:
  Aggregated Key Accuracy: 89.06%
  Aggregated Key: [0 0 1 0 0 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 1]...
  Ground Truth:   [1 0 1 1 0 0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
  Intra-person average Hamming distance: 44.06 bits

Inter-person Hamming distances (aggregated keys):
  Distance between Person 1 and Person 2: 120 bits
  Distance between Person 1 and Person 3: 126 bits
  Distance between Person 1 and Person 4: 105 bits
  Distance between Person 1 and Person 5: 135 bits
  Distance between Person 1 and Person 6: 135 bits
  Distance between Person 1 and Person 7: 123 bits
  Distance between Person 1 and Person 8: 139 bits
  Distance between Person 1 and Person 9: 134 bits
  Distance between Person 1 and Person 10: 136 bits
  Distance between Person 1 and Person 11: 135 bits
  Distance between Person 1 and Person 12: 120 bits
  Distance between Person 1 and Person 13: 119 bits
  Distance between Person 1 and Person 14: 113 bits
  Distance between Person 1 and Person 15: 153 bits
  Distance between Person 1 and Person 16: 86 bits
  Distance between Person 1 and Person 17: 127 bits
  Distance between Person 1 and Person 18: 142 bits
  Distance between Person 1 and Person 19: 124 bits
  Distance between Person 1 and Person 20: 126 bits
  Distance between Person 1 and Person 21: 131 bits
  Distance between Person 1 and Person 22: 145 bits
  Distance between Person 1 and Person 23: 145 bits
  Distance between Person 1 and Person 24: 123 bits
  Distance between Person 1 and Person 25: 109 bits
  Distance between Person 1 and Person 26: 119 bits
  Distance between Person 1 and Person 27: 105 bits
  Distance between Person 1 and Person 28: 138 bits
  Distance between Person 1 and Person 29: 146 bits
  Distance between Person 1 and Person 30: 132 bits
  Distance between Person 1 and Person 31: 125 bits
  Distance between Person 1 and Person 32: 135 bits
  Distance between Person 1 and Person 33: 104 bits
  Distance between Person 1 and Person 34: 122 bits
  Distance between Person 1 and Person 35: 135 bits
  Distance between Person 1 and Person 36: 123 bits
  Distance between Person 1 and Person 37: 132 bits
  Distance between Person 1 and Person 38: 145 bits
  Distance between Person 1 and Person 39: 143 bits
  Distance between Person 1 and Person 40: 139 bits
  Distance between Person 1 and Person 41: 129 bits
  Distance between Person 1 and Person 42: 117 bits
  Distance between Person 1 and Person 43: 128 bits
  Distance between Person 1 and Person 44: 113 bits
  Distance between Person 1 and Person 45: 135 bits
  Distance between Person 1 and Person 46: 96 bits
  Distance between Person 1 and Person 47: 125 bits
  Distance between Person 1 and Person 48: 130 bits
  Distance between Person 1 and Person 49: 118 bits
  Distance between Person 1 and Person 50: 136 bits
  Distance between Person 1 and Person 51: 134 bits
  Distance between Person 1 and Person 52: 139 bits
  Distance between Person 1 and Person 53: 142 bits
  Distance between Person 1 and Person 54: 131 bits
  Distance between Person 1 and Person 55: 145 bits
  Distance between Person 1 and Person 56: 139 bits
  Distance between Person 1 and Person 57: 140 bits
  Distance between Person 1 and Person 58: 86 bits
  Distance between Person 1 and Person 59: 105 bits
  Distance between Person 1 and Person 60: 134 bits
  Distance between Person 1 and Person 61: 132 bits
  Distance between Person 1 and Person 62: 122 bits
  Distance between Person 1 and Person 63: 124 bits
  Distance between Person 1 and Person 64: 137 bits
  Distance between Person 1 and Person 65: 99 bits
  Distance between Person 1 and Person 66: 115 bits
  Distance between Person 1 and Person 67: 102 bits
  Distance between Person 1 and Person 68: 144 bits
  Distance between Person 1 and Person 69: 90 bits
  Distance between Person 1 and Person 70: 126 bits
  Distance between Person 1 and Person 71: 135 bits
  Distance between Person 1 and Person 72: 102 bits
  Distance between Person 1 and Person 73: 108 bits
  Distance between Person 1 and Person 74: 130 bits
  Distance between Person 1 and Person 75: 123 bits
  Distance between Person 1 and Person 76: 127 bits
  Distance between Person 1 and Person 77: 126 bits
  Distance between Person 1 and Person 78: 106 bits
  Distance between Person 1 and Person 79: 111 bits
  Distance between Person 1 and Person 80: 116 bits
  Distance between Person 1 and Person 81: 122 bits
  Distance between Person 1 and Person 82: 125 bits
  Distance between Person 1 and Person 83: 130 bits
  Distance between Person 1 and Person 84: 118 bits
  Distance between Person 1 and Person 85: 119 bits
  Distance between Person 1 and Person 86: 118 bits
  Distance between Person 1 and Person 87: 126 bits
  Distance between Person 1 and Person 88: 141 bits
  Distance between Person 1 and Person 89: 140 bits
  Distance between Person 2 and Person 3: 106 bits
  Distance between Person 2 and Person 4: 117 bits
  Distance between Person 2 and Person 5: 133 bits
  Distance between Person 2 and Person 6: 131 bits
  Distance between Person 2 and Person 7: 137 bits
  Distance between Person 2 and Person 8: 119 bits
  Distance between Person 2 and Person 9: 130 bits
  Distance between Person 2 and Person 10: 114 bits
  Distance between Person 2 and Person 11: 155 bits
  Distance between Person 2 and Person 12: 116 bits
  Distance between Person 2 and Person 13: 125 bits
  Distance between Person 2 and Person 14: 105 bits
  Distance between Person 2 and Person 15: 137 bits
  Distance between Person 2 and Person 16: 110 bits
  Distance between Person 2 and Person 17: 133 bits
  Distance between Person 2 and Person 18: 124 bits
  Distance between Person 2 and Person 19: 122 bits
  Distance between Person 2 and Person 20: 122 bits
  Distance between Person 2 and Person 21: 115 bits
  Distance between Person 2 and Person 22: 123 bits
  Distance between Person 2 and Person 23: 135 bits
  Distance between Person 2 and Person 24: 125 bits
  Distance between Person 2 and Person 25: 119 bits
  Distance between Person 2 and Person 26: 111 bits
  Distance between Person 2 and Person 27: 99 bits
  Distance between Person 2 and Person 28: 122 bits
  Distance between Person 2 and Person 29: 124 bits
  Distance between Person 2 and Person 30: 122 bits
  Distance between Person 2 and Person 31: 149 bits
  Distance between Person 2 and Person 32: 127 bits
  Distance between Person 2 and Person 33: 118 bits
  Distance between Person 2 and Person 34: 104 bits
  Distance between Person 2 and Person 35: 127 bits
  Distance between Person 2 and Person 36: 123 bits
  Distance between Person 2 and Person 37: 106 bits
  Distance between Person 2 and Person 38: 147 bits
  Distance between Person 2 and Person 39: 125 bits
  Distance between Person 2 and Person 40: 129 bits
  Distance between Person 2 and Person 41: 133 bits
  Distance between Person 2 and Person 42: 113 bits
  Distance between Person 2 and Person 43: 114 bits
  Distance between Person 2 and Person 44: 117 bits
  Distance between Person 2 and Person 45: 119 bits
  Distance between Person 2 and Person 46: 112 bits
  Distance between Person 2 and Person 47: 127 bits
  Distance between Person 2 and Person 48: 128 bits
  Distance between Person 2 and Person 49: 132 bits
  Distance between Person 2 and Person 50: 122 bits
  Distance between Person 2 and Person 51: 36 bits
  Distance between Person 2 and Person 52: 111 bits
  Distance between Person 2 and Person 53: 140 bits
  Distance between Person 2 and Person 54: 115 bits
  Distance between Person 2 and Person 55: 127 bits
  Distance between Person 2 and Person 56: 113 bits
  Distance between Person 2 and Person 57: 144 bits
  Distance between Person 2 and Person 58: 130 bits
  Distance between Person 2 and Person 59: 101 bits
  Distance between Person 2 and Person 60: 140 bits
  Distance between Person 2 and Person 61: 124 bits
  Distance between Person 2 and Person 62: 110 bits
  Distance between Person 2 and Person 63: 136 bits
  Distance between Person 2 and Person 64: 111 bits
  Distance between Person 2 and Person 65: 127 bits
  Distance between Person 2 and Person 66: 133 bits
  Distance between Person 2 and Person 67: 122 bits
  Distance between Person 2 and Person 68: 114 bits
  Distance between Person 2 and Person 69: 122 bits
  Distance between Person 2 and Person 70: 114 bits
  Distance between Person 2 and Person 71: 133 bits
  Distance between Person 2 and Person 72: 130 bits
  Distance between Person 2 and Person 73: 120 bits
  Distance between Person 2 and Person 74: 120 bits
  Distance between Person 2 and Person 75: 133 bits
  Distance between Person 2 and Person 76: 115 bits
  Distance between Person 2 and Person 77: 118 bits
  Distance between Person 2 and Person 78: 130 bits
  Distance between Person 2 and Person 79: 125 bits
  Distance between Person 2 and Person 80: 106 bits
  Distance between Person 2 and Person 81: 140 bits
  Distance between Person 2 and Person 82: 113 bits
  Distance between Person 2 and Person 83: 124 bits
  Distance between Person 2 and Person 84: 110 bits
  Distance between Person 2 and Person 85: 117 bits
  Distance between Person 2 and Person 86: 142 bits
  Distance between Person 2 and Person 87: 132 bits
  Distance between Person 2 and Person 88: 129 bits
  Distance between Person 2 and Person 89: 128 bits
  Distance between Person 3 and Person 4: 113 bits
  Distance between Person 3 and Person 5: 125 bits
  Distance between Person 3 and Person 6: 123 bits
  Distance between Person 3 and Person 7: 151 bits
  Distance between Person 3 and Person 8: 111 bits
  Distance between Person 3 and Person 9: 136 bits
  Distance between Person 3 and Person 10: 118 bits
  Distance between Person 3 and Person 11: 123 bits
  Distance between Person 3 and Person 12: 150 bits
  Distance between Person 3 and Person 13: 117 bits
  Distance between Person 3 and Person 14: 143 bits
  Distance between Person 3 and Person 15: 141 bits
  Distance between Person 3 and Person 16: 126 bits
  Distance between Person 3 and Person 17: 135 bits
  Distance between Person 3 and Person 18: 136 bits
  Distance between Person 3 and Person 19: 138 bits
  Distance between Person 3 and Person 20: 144 bits
  Distance between Person 3 and Person 21: 127 bits
  Distance between Person 3 and Person 22: 117 bits
  Distance between Person 3 and Person 23: 135 bits
  Distance between Person 3 and Person 24: 87 bits
  Distance between Person 3 and Person 25: 139 bits
  Distance between Person 3 and Person 26: 123 bits
  Distance between Person 3 and Person 27: 111 bits
  Distance between Person 3 and Person 28: 138 bits
  Distance between Person 3 and Person 29: 112 bits
  Distance between Person 3 and Person 30: 106 bits
  Distance between Person 3 and Person 31: 141 bits
  Distance between Person 3 and Person 32: 107 bits
  Distance between Person 3 and Person 33: 122 bits
  Distance between Person 3 and Person 34: 96 bits
  Distance between Person 3 and Person 35: 113 bits
  Distance between Person 3 and Person 36: 121 bits
  Distance between Person 3 and Person 37: 90 bits
  Distance between Person 3 and Person 38: 145 bits
  Distance between Person 3 and Person 39: 81 bits
  Distance between Person 3 and Person 40: 117 bits
  Distance between Person 3 and Person 41: 129 bits
  Distance between Person 3 and Person 42: 141 bits
  Distance between Person 3 and Person 43: 128 bits
  Distance between Person 3 and Person 44: 127 bits
  Distance between Person 3 and Person 45: 119 bits
  Distance between Person 3 and Person 46: 130 bits
  Distance between Person 3 and Person 47: 123 bits
  Distance between Person 3 and Person 48: 112 bits
  Distance between Person 3 and Person 49: 134 bits
  Distance between Person 3 and Person 50: 114 bits
  Distance between Person 3 and Person 51: 112 bits
  Distance between Person 3 and Person 52: 119 bits
  Distance between Person 3 and Person 53: 128 bits
  Distance between Person 3 and Person 54: 145 bits
  Distance between Person 3 and Person 55: 129 bits
  Distance between Person 3 and Person 56: 127 bits
  Distance between Person 3 and Person 57: 130 bits
  Distance between Person 3 and Person 58: 144 bits
  Distance between Person 3 and Person 59: 127 bits
  Distance between Person 3 and Person 60: 142 bits
  Distance between Person 3 and Person 61: 130 bits
  Distance between Person 3 and Person 62: 120 bits
  Distance between Person 3 and Person 63: 136 bits
  Distance between Person 3 and Person 64: 111 bits
  Distance between Person 3 and Person 65: 127 bits
  Distance between Person 3 and Person 66: 111 bits
  Distance between Person 3 and Person 67: 126 bits
  Distance between Person 3 and Person 68: 118 bits
  Distance between Person 3 and Person 69: 134 bits
  Distance between Person 3 and Person 70: 132 bits
  Distance between Person 3 and Person 71: 117 bits
  Distance between Person 3 and Person 72: 128 bits
  Distance between Person 3 and Person 73: 126 bits
  Distance between Person 3 and Person 74: 124 bits
  Distance between Person 3 and Person 75: 147 bits
  Distance between Person 3 and Person 76: 119 bits
  Distance between Person 3 and Person 77: 124 bits
  Distance between Person 3 and Person 78: 138 bits
  Distance between Person 3 and Person 79: 107 bits
  Distance between Person 3 and Person 80: 128 bits
  Distance between Person 3 and Person 81: 130 bits
  Distance between Person 3 and Person 82: 117 bits
  Distance between Person 3 and Person 83: 120 bits
  Distance between Person 3 and Person 84: 110 bits
  Distance between Person 3 and Person 85: 131 bits
  Distance between Person 3 and Person 86: 136 bits
  Distance between Person 3 and Person 87: 128 bits
  Distance between Person 3 and Person 88: 149 bits
  Distance between Person 3 and Person 89: 148 bits
  Distance between Person 4 and Person 5: 126 bits
  Distance between Person 4 and Person 6: 96 bits
  Distance between Person 4 and Person 7: 144 bits
  Distance between Person 4 and Person 8: 126 bits
  Distance between Person 4 and Person 9: 139 bits
  Distance between Person 4 and Person 10: 127 bits
  Distance between Person 4 and Person 11: 146 bits
  Distance between Person 4 and Person 12: 135 bits
  Distance between Person 4 and Person 13: 130 bits
  Distance between Person 4 and Person 14: 102 bits
  Distance between Person 4 and Person 15: 116 bits
  Distance between Person 4 and Person 16: 105 bits
  Distance between Person 4 and Person 17: 124 bits
  Distance between Person 4 and Person 18: 113 bits
  Distance between Person 4 and Person 19: 123 bits
  Distance between Person 4 and Person 20: 155 bits
  Distance between Person 4 and Person 21: 118 bits
  Distance between Person 4 and Person 22: 142 bits
  Distance between Person 4 and Person 23: 140 bits
  Distance between Person 4 and Person 24: 116 bits
  Distance between Person 4 and Person 25: 112 bits
  Distance between Person 4 and Person 26: 112 bits
  Distance between Person 4 and Person 27: 104 bits
  Distance between Person 4 and Person 28: 139 bits
  Distance between Person 4 and Person 29: 135 bits
  Distance between Person 4 and Person 30: 141 bits
  Distance between Person 4 and Person 31: 136 bits
  Distance between Person 4 and Person 32: 130 bits
  Distance between Person 4 and Person 33: 123 bits
  Distance between Person 4 and Person 34: 53 bits
  Distance between Person 4 and Person 35: 126 bits
  Distance between Person 4 and Person 36: 142 bits
  Distance between Person 4 and Person 37: 121 bits
  Distance between Person 4 and Person 38: 120 bits
  Distance between Person 4 and Person 39: 136 bits
  Distance between Person 4 and Person 40: 74 bits
  Distance between Person 4 and Person 41: 100 bits
  Distance between Person 4 and Person 42: 122 bits
  Distance between Person 4 and Person 43: 115 bits
  Distance between Person 4 and Person 44: 128 bits
  Distance between Person 4 and Person 45: 116 bits
  Distance between Person 4 and Person 46: 127 bits
  Distance between Person 4 and Person 47: 138 bits
  Distance between Person 4 and Person 48: 119 bits
  Distance between Person 4 and Person 49: 131 bits
  Distance between Person 4 and Person 50: 139 bits
  Distance between Person 4 and Person 51: 127 bits
  Distance between Person 4 and Person 52: 106 bits
  Distance between Person 4 and Person 53: 129 bits
  Distance between Person 4 and Person 54: 144 bits
  Distance between Person 4 and Person 55: 114 bits
  Distance between Person 4 and Person 56: 138 bits
  Distance between Person 4 and Person 57: 129 bits
  Distance between Person 4 and Person 58: 117 bits
  Distance between Person 4 and Person 59: 116 bits
  Distance between Person 4 and Person 60: 121 bits
  Distance between Person 4 and Person 61: 141 bits
  Distance between Person 4 and Person 62: 121 bits
  Distance between Person 4 and Person 63: 129 bits
  Distance between Person 4 and Person 64: 132 bits
  Distance between Person 4 and Person 65: 116 bits
  Distance between Person 4 and Person 66: 120 bits
  Distance between Person 4 and Person 67: 123 bits
  Distance between Person 4 and Person 68: 123 bits
  Distance between Person 4 and Person 69: 127 bits
  Distance between Person 4 and Person 70: 113 bits
  Distance between Person 4 and Person 71: 122 bits
  Distance between Person 4 and Person 72: 129 bits
  Distance between Person 4 and Person 73: 123 bits
  Distance between Person 4 and Person 74: 121 bits
  Distance between Person 4 and Person 75: 142 bits
  Distance between Person 4 and Person 76: 134 bits
  Distance between Person 4 and Person 77: 135 bits
  Distance between Person 4 and Person 78: 117 bits
  Distance between Person 4 and Person 79: 118 bits
  Distance between Person 4 and Person 80: 113 bits
  Distance between Person 4 and Person 81: 119 bits
  Distance between Person 4 and Person 82: 132 bits
  Distance between Person 4 and Person 83: 127 bits
  Distance between Person 4 and Person 84: 117 bits
  Distance between Person 4 and Person 85: 114 bits
  Distance between Person 4 and Person 86: 123 bits
  Distance between Person 4 and Person 87: 147 bits
  Distance between Person 4 and Person 88: 138 bits
  Distance between Person 4 and Person 89: 145 bits
  Distance between Person 5 and Person 6: 144 bits
  Distance between Person 5 and Person 7: 132 bits
  Distance between Person 5 and Person 8: 122 bits
  Distance between Person 5 and Person 9: 113 bits
  Distance between Person 5 and Person 10: 131 bits
  Distance between Person 5 and Person 11: 112 bits
  Distance between Person 5 and Person 12: 145 bits
  Distance between Person 5 and Person 13: 130 bits
  Distance between Person 5 and Person 14: 130 bits
  Distance between Person 5 and Person 15: 108 bits
  Distance between Person 5 and Person 16: 113 bits
  Distance between Person 5 and Person 17: 122 bits
  Distance between Person 5 and Person 18: 127 bits
  Distance between Person 5 and Person 19: 127 bits
  Distance between Person 5 and Person 20: 141 bits
  Distance between Person 5 and Person 21: 136 bits
  Distance between Person 5 and Person 22: 100 bits
  Distance between Person 5 and Person 23: 132 bits
  Distance between Person 5 and Person 24: 118 bits
  Distance between Person 5 and Person 25: 132 bits
  Distance between Person 5 and Person 26: 128 bits
  Distance between Person 5 and Person 27: 120 bits
  Distance between Person 5 and Person 28: 113 bits
  Distance between Person 5 and Person 29: 141 bits
  Distance between Person 5 and Person 30: 123 bits
  Distance between Person 5 and Person 31: 126 bits
  Distance between Person 5 and Person 32: 130 bits
  Distance between Person 5 and Person 33: 137 bits
  Distance between Person 5 and Person 34: 133 bits
  Distance between Person 5 and Person 35: 132 bits
  Distance between Person 5 and Person 36: 112 bits
  Distance between Person 5 and Person 37: 129 bits
  Distance between Person 5 and Person 38: 128 bits
  Distance between Person 5 and Person 39: 120 bits
  Distance between Person 5 and Person 40: 124 bits
  Distance between Person 5 and Person 41: 114 bits
  Distance between Person 5 and Person 42: 144 bits
  Distance between Person 5 and Person 43: 131 bits
  Distance between Person 5 and Person 44: 136 bits
  Distance between Person 5 and Person 45: 132 bits
  Distance between Person 5 and Person 46: 133 bits
  Distance between Person 5 and Person 47: 140 bits
  Distance between Person 5 and Person 48: 109 bits
  Distance between Person 5 and Person 49: 97 bits
  Distance between Person 5 and Person 50: 135 bits
  Distance between Person 5 and Person 51: 125 bits
  Distance between Person 5 and Person 52: 112 bits
  Distance between Person 5 and Person 53: 135 bits
  Distance between Person 5 and Person 54: 124 bits
  Distance between Person 5 and Person 55: 106 bits
  Distance between Person 5 and Person 56: 122 bits
  Distance between Person 5 and Person 57: 125 bits
  Distance between Person 5 and Person 58: 119 bits
  Distance between Person 5 and Person 59: 122 bits
  Distance between Person 5 and Person 60: 125 bits
  Distance between Person 5 and Person 61: 125 bits
  Distance between Person 5 and Person 62: 115 bits
  Distance between Person 5 and Person 63: 137 bits
  Distance between Person 5 and Person 64: 128 bits
  Distance between Person 5 and Person 65: 134 bits
  Distance between Person 5 and Person 66: 144 bits
  Distance between Person 5 and Person 67: 125 bits
  Distance between Person 5 and Person 68: 121 bits
  Distance between Person 5 and Person 69: 133 bits
  Distance between Person 5 and Person 70: 139 bits
  Distance between Person 5 and Person 71: 130 bits
  Distance between Person 5 and Person 72: 115 bits
  Distance between Person 5 and Person 73: 129 bits
  Distance between Person 5 and Person 74: 135 bits
  Distance between Person 5 and Person 75: 142 bits
  Distance between Person 5 and Person 76: 140 bits
  Distance between Person 5 and Person 77: 119 bits
  Distance between Person 5 and Person 78: 129 bits
  Distance between Person 5 and Person 79: 126 bits
  Distance between Person 5 and Person 80: 115 bits
  Distance between Person 5 and Person 81: 119 bits
  Distance between Person 5 and Person 82: 156 bits
  Distance between Person 5 and Person 83: 121 bits
  Distance between Person 5 and Person 84: 109 bits
  Distance between Person 5 and Person 85: 138 bits
  Distance between Person 5 and Person 86: 119 bits
  Distance between Person 5 and Person 87: 127 bits
  Distance between Person 5 and Person 88: 136 bits
  Distance between Person 5 and Person 89: 117 bits
  Distance between Person 6 and Person 7: 146 bits
  Distance between Person 6 and Person 8: 124 bits
  Distance between Person 6 and Person 9: 145 bits
  Distance between Person 6 and Person 10: 111 bits
  Distance between Person 6 and Person 11: 136 bits
  Distance between Person 6 and Person 12: 141 bits
  Distance between Person 6 and Person 13: 132 bits
  Distance between Person 6 and Person 14: 110 bits
  Distance between Person 6 and Person 15: 134 bits
  Distance between Person 6 and Person 16: 131 bits
  Distance between Person 6 and Person 17: 128 bits
  Distance between Person 6 and Person 18: 101 bits
  Distance between Person 6 and Person 19: 117 bits
  Distance between Person 6 and Person 20: 137 bits
  Distance between Person 6 and Person 21: 122 bits
  Distance between Person 6 and Person 22: 132 bits
  Distance between Person 6 and Person 23: 104 bits
  Distance between Person 6 and Person 24: 140 bits
  Distance between Person 6 and Person 25: 128 bits
  Distance between Person 6 and Person 26: 144 bits
  Distance between Person 6 and Person 27: 124 bits
  Distance between Person 6 and Person 28: 147 bits
  Distance between Person 6 and Person 29: 123 bits
  Distance between Person 6 and Person 30: 123 bits
  Distance between Person 6 and Person 31: 146 bits
  Distance between Person 6 and Person 32: 114 bits
  Distance between Person 6 and Person 33: 131 bits
  Distance between Person 6 and Person 34: 115 bits
  Distance between Person 6 and Person 35: 136 bits
  Distance between Person 6 and Person 36: 138 bits
  Distance between Person 6 and Person 37: 119 bits
  Distance between Person 6 and Person 38: 130 bits
  Distance between Person 6 and Person 39: 118 bits
  Distance between Person 6 and Person 40: 94 bits
  Distance between Person 6 and Person 41: 124 bits
  Distance between Person 6 and Person 42: 118 bits
  Distance between Person 6 and Person 43: 123 bits
  Distance between Person 6 and Person 44: 120 bits
  Distance between Person 6 and Person 45: 106 bits
  Distance between Person 6 and Person 46: 127 bits
  Distance between Person 6 and Person 47: 120 bits
  Distance between Person 6 and Person 48: 117 bits
  Distance between Person 6 and Person 49: 137 bits
  Distance between Person 6 and Person 50: 127 bits
  Distance between Person 6 and Person 51: 143 bits
  Distance between Person 6 and Person 52: 120 bits
  Distance between Person 6 and Person 53: 127 bits
  Distance between Person 6 and Person 54: 126 bits
  Distance between Person 6 and Person 55: 108 bits
  Distance between Person 6 and Person 56: 122 bits
  Distance between Person 6 and Person 57: 127 bits
  Distance between Person 6 and Person 58: 129 bits
  Distance between Person 6 and Person 59: 112 bits
  Distance between Person 6 and Person 60: 123 bits
  Distance between Person 6 and Person 61: 143 bits
  Distance between Person 6 and Person 62: 125 bits
  Distance between Person 6 and Person 63: 109 bits
  Distance between Person 6 and Person 64: 116 bits
  Distance between Person 6 and Person 65: 132 bits
  Distance between Person 6 and Person 66: 116 bits
  Distance between Person 6 and Person 67: 127 bits
  Distance between Person 6 and Person 68: 131 bits
  Distance between Person 6 and Person 69: 125 bits
  Distance between Person 6 and Person 70: 135 bits
  Distance between Person 6 and Person 71: 98 bits
  Distance between Person 6 and Person 72: 153 bits
  Distance between Person 6 and Person 73: 121 bits
  Distance between Person 6 and Person 74: 123 bits
  Distance between Person 6 and Person 75: 120 bits
  Distance between Person 6 and Person 76: 116 bits
  Distance between Person 6 and Person 77: 137 bits
  Distance between Person 6 and Person 78: 123 bits
  Distance between Person 6 and Person 79: 146 bits
  Distance between Person 6 and Person 80: 113 bits
  Distance between Person 6 and Person 81: 117 bits
  Distance between Person 6 and Person 82: 116 bits
  Distance between Person 6 and Person 83: 119 bits
  Distance between Person 6 and Person 84: 143 bits
  Distance between Person 6 and Person 85: 152 bits
  Distance between Person 6 and Person 86: 115 bits
  Distance between Person 6 and Person 87: 151 bits
  Distance between Person 6 and Person 88: 144 bits
  Distance between Person 6 and Person 89: 147 bits
  Distance between Person 7 and Person 8: 140 bits
  Distance between Person 7 and Person 9: 123 bits
  Distance between Person 7 and Person 10: 139 bits
  Distance between Person 7 and Person 11: 134 bits
  Distance between Person 7 and Person 12: 131 bits
  Distance between Person 7 and Person 13: 132 bits
  Distance between Person 7 and Person 14: 120 bits
  Distance between Person 7 and Person 15: 140 bits
  Distance between Person 7 and Person 16: 123 bits
  Distance between Person 7 and Person 17: 128 bits
  Distance between Person 7 and Person 18: 131 bits
  Distance between Person 7 and Person 19: 109 bits
  Distance between Person 7 and Person 20: 123 bits
  Distance between Person 7 and Person 21: 102 bits
  Distance between Person 7 and Person 22: 132 bits
  Distance between Person 7 and Person 23: 110 bits
  Distance between Person 7 and Person 24: 122 bits
  Distance between Person 7 and Person 25: 134 bits
  Distance between Person 7 and Person 26: 128 bits
  Distance between Person 7 and Person 27: 122 bits
  Distance between Person 7 and Person 28: 119 bits
  Distance between Person 7 and Person 29: 109 bits
  Distance between Person 7 and Person 30: 139 bits
  Distance between Person 7 and Person 31: 108 bits
  Distance between Person 7 and Person 32: 158 bits
  Distance between Person 7 and Person 33: 115 bits
  Distance between Person 7 and Person 34: 139 bits
  Distance between Person 7 and Person 35: 142 bits
  Distance between Person 7 and Person 36: 132 bits
  Distance between Person 7 and Person 37: 119 bits
  Distance between Person 7 and Person 38: 122 bits
  Distance between Person 7 and Person 39: 150 bits
  Distance between Person 7 and Person 40: 128 bits
  Distance between Person 7 and Person 41: 120 bits
  Distance between Person 7 and Person 42: 128 bits
  Distance between Person 7 and Person 43: 123 bits
  Distance between Person 7 and Person 44: 122 bits
  Distance between Person 7 and Person 45: 124 bits
  Distance between Person 7 and Person 46: 121 bits
  Distance between Person 7 and Person 47: 122 bits
  Distance between Person 7 and Person 48: 147 bits
  Distance between Person 7 and Person 49: 115 bits
  Distance between Person 7 and Person 50: 141 bits
  Distance between Person 7 and Person 51: 131 bits
  Distance between Person 7 and Person 52: 122 bits
  Distance between Person 7 and Person 53: 111 bits
  Distance between Person 7 and Person 54: 130 bits
  Distance between Person 7 and Person 55: 104 bits
  Distance between Person 7 and Person 56: 124 bits
  Distance between Person 7 and Person 57: 113 bits
  Distance between Person 7 and Person 58: 133 bits
  Distance between Person 7 and Person 59: 142 bits
  Distance between Person 7 and Person 60: 101 bits
  Distance between Person 7 and Person 61: 115 bits
  Distance between Person 7 and Person 62: 145 bits
  Distance between Person 7 and Person 63: 125 bits
  Distance between Person 7 and Person 64: 152 bits
  Distance between Person 7 and Person 65: 114 bits
  Distance between Person 7 and Person 66: 136 bits
  Distance between Person 7 and Person 67: 113 bits
  Distance between Person 7 and Person 68: 143 bits
  Distance between Person 7 and Person 69: 117 bits
  Distance between Person 7 and Person 70: 105 bits
  Distance between Person 7 and Person 71: 138 bits
  Distance between Person 7 and Person 72: 131 bits
  Distance between Person 7 and Person 73: 133 bits
  Distance between Person 7 and Person 74: 133 bits
  Distance between Person 7 and Person 75: 108 bits
  Distance between Person 7 and Person 76: 140 bits
  Distance between Person 7 and Person 77: 113 bits
  Distance between Person 7 and Person 78: 143 bits
  Distance between Person 7 and Person 79: 132 bits
  Distance between Person 7 and Person 80: 113 bits
  Distance between Person 7 and Person 81: 151 bits
  Distance between Person 7 and Person 82: 122 bits
  Distance between Person 7 and Person 83: 107 bits
  Distance between Person 7 and Person 84: 127 bits
  Distance between Person 7 and Person 85: 124 bits
  Distance between Person 7 and Person 86: 125 bits
  Distance between Person 7 and Person 87: 131 bits
  Distance between Person 7 and Person 88: 112 bits
  Distance between Person 7 and Person 89: 111 bits
  Distance between Person 8 and Person 9: 113 bits
  Distance between Person 8 and Person 10: 125 bits
  Distance between Person 8 and Person 11: 124 bits
  Distance between Person 8 and Person 12: 127 bits
  Distance between Person 8 and Person 13: 122 bits
  Distance between Person 8 and Person 14: 122 bits
  Distance between Person 8 and Person 15: 118 bits
  Distance between Person 8 and Person 16: 123 bits
  Distance between Person 8 and Person 17: 122 bits
  Distance between Person 8 and Person 18: 109 bits
  Distance between Person 8 and Person 19: 153 bits
  Distance between Person 8 and Person 20: 133 bits
  Distance between Person 8 and Person 21: 134 bits
  Distance between Person 8 and Person 22: 114 bits
  Distance between Person 8 and Person 23: 148 bits
  Distance between Person 8 and Person 24: 122 bits
  Distance between Person 8 and Person 25: 110 bits
  Distance between Person 8 and Person 26: 140 bits
  Distance between Person 8 and Person 27: 138 bits
  Distance between Person 8 and Person 28: 121 bits
  Distance between Person 8 and Person 29: 117 bits
  Distance between Person 8 and Person 30: 137 bits
  Distance between Person 8 and Person 31: 144 bits
  Distance between Person 8 and Person 32: 132 bits
  Distance between Person 8 and Person 33: 131 bits
  Distance between Person 8 and Person 34: 115 bits
  Distance between Person 8 and Person 35: 114 bits
  Distance between Person 8 and Person 36: 130 bits
  Distance between Person 8 and Person 37: 107 bits
  Distance between Person 8 and Person 38: 136 bits
  Distance between Person 8 and Person 39: 116 bits
  Distance between Person 8 and Person 40: 112 bits
  Distance between Person 8 and Person 41: 134 bits
  Distance between Person 8 and Person 42: 118 bits
  Distance between Person 8 and Person 43: 119 bits
  Distance between Person 8 and Person 44: 124 bits
  Distance between Person 8 and Person 45: 116 bits
  Distance between Person 8 and Person 46: 129 bits
  Distance between Person 8 and Person 47: 116 bits
  Distance between Person 8 and Person 48: 125 bits
  Distance between Person 8 and Person 49: 137 bits
  Distance between Person 8 and Person 50: 137 bits
  Distance between Person 8 and Person 51: 131 bits
  Distance between Person 8 and Person 52: 116 bits
  Distance between Person 8 and Person 53: 125 bits
  Distance between Person 8 and Person 54: 108 bits
  Distance between Person 8 and Person 55: 128 bits
  Distance between Person 8 and Person 56: 110 bits
  Distance between Person 8 and Person 57: 123 bits
  Distance between Person 8 and Person 58: 119 bits
  Distance between Person 8 and Person 59: 130 bits
  Distance between Person 8 and Person 60: 121 bits
  Distance between Person 8 and Person 61: 151 bits
  Distance between Person 8 and Person 62: 137 bits
  Distance between Person 8 and Person 63: 117 bits
  Distance between Person 8 and Person 64: 122 bits
  Distance between Person 8 and Person 65: 126 bits
  Distance between Person 8 and Person 66: 138 bits
  Distance between Person 8 and Person 67: 121 bits
  Distance between Person 8 and Person 68: 107 bits
  Distance between Person 8 and Person 69: 133 bits
  Distance between Person 8 and Person 70: 141 bits
  Distance between Person 8 and Person 71: 130 bits
  Distance between Person 8 and Person 72: 147 bits
  Distance between Person 8 and Person 73: 111 bits
  Distance between Person 8 and Person 74: 121 bits
  Distance between Person 8 and Person 75: 114 bits
  Distance between Person 8 and Person 76: 132 bits
  Distance between Person 8 and Person 77: 107 bits
  Distance between Person 8 and Person 78: 131 bits
  Distance between Person 8 and Person 79: 122 bits
  Distance between Person 8 and Person 80: 125 bits
  Distance between Person 8 and Person 81: 127 bits
  Distance between Person 8 and Person 82: 132 bits
  Distance between Person 8 and Person 83: 135 bits
  Distance between Person 8 and Person 84: 119 bits
  Distance between Person 8 and Person 85: 132 bits
  Distance between Person 8 and Person 86: 145 bits
  Distance between Person 8 and Person 87: 131 bits
  Distance between Person 8 and Person 88: 122 bits
  Distance between Person 8 and Person 89: 141 bits
  Distance between Person 9 and Person 10: 132 bits
  Distance between Person 9 and Person 11: 121 bits
  Distance between Person 9 and Person 12: 122 bits
  Distance between Person 9 and Person 13: 139 bits
  Distance between Person 9 and Person 14: 151 bits
  Distance between Person 9 and Person 15: 115 bits
  Distance between Person 9 and Person 16: 138 bits
  Distance between Person 9 and Person 17: 125 bits
  Distance between Person 9 and Person 18: 122 bits
  Distance between Person 9 and Person 19: 136 bits
  Distance between Person 9 and Person 20: 118 bits
  Distance between Person 9 and Person 21: 125 bits
  Distance between Person 9 and Person 22: 131 bits
  Distance between Person 9 and Person 23: 105 bits
  Distance between Person 9 and Person 24: 135 bits
  Distance between Person 9 and Person 25: 121 bits
  Distance between Person 9 and Person 26: 145 bits
  Distance between Person 9 and Person 27: 135 bits
  Distance between Person 9 and Person 28: 104 bits
  Distance between Person 9 and Person 29: 114 bits
  Distance between Person 9 and Person 30: 148 bits
  Distance between Person 9 and Person 31: 113 bits
  Distance between Person 9 and Person 32: 131 bits
  Distance between Person 9 and Person 33: 108 bits
  Distance between Person 9 and Person 34: 132 bits
  Distance between Person 9 and Person 35: 145 bits
  Distance between Person 9 and Person 36: 143 bits
  Distance between Person 9 and Person 37: 124 bits
  Distance between Person 9 and Person 38: 135 bits
  Distance between Person 9 and Person 39: 129 bits
  Distance between Person 9 and Person 40: 139 bits
  Distance between Person 9 and Person 41: 111 bits
  Distance between Person 9 and Person 42: 125 bits
  Distance between Person 9 and Person 43: 116 bits
  Distance between Person 9 and Person 44: 129 bits
  Distance between Person 9 and Person 45: 129 bits
  Distance between Person 9 and Person 46: 126 bits
  Distance between Person 9 and Person 47: 139 bits
  Distance between Person 9 and Person 48: 136 bits
  Distance between Person 9 and Person 49: 148 bits
  Distance between Person 9 and Person 50: 132 bits
  Distance between Person 9 and Person 51: 114 bits
  Distance between Person 9 and Person 52: 151 bits
  Distance between Person 9 and Person 53: 126 bits
  Distance between Person 9 and Person 54: 107 bits
  Distance between Person 9 and Person 55: 147 bits
  Distance between Person 9 and Person 56: 133 bits
  Distance between Person 9 and Person 57: 124 bits
  Distance between Person 9 and Person 58: 124 bits
  Distance between Person 9 and Person 59: 131 bits
  Distance between Person 9 and Person 60: 124 bits
  Distance between Person 9 and Person 61: 136 bits
  Distance between Person 9 and Person 62: 136 bits
  Distance between Person 9 and Person 63: 106 bits
  Distance between Person 9 and Person 64: 143 bits
  Distance between Person 9 and Person 65: 157 bits
  Distance between Person 9 and Person 66: 141 bits
  Distance between Person 9 and Person 67: 104 bits
  Distance between Person 9 and Person 68: 124 bits
  Distance between Person 9 and Person 69: 124 bits
  Distance between Person 9 and Person 70: 136 bits
  Distance between Person 9 and Person 71: 151 bits
  Distance between Person 9 and Person 72: 126 bits
  Distance between Person 9 and Person 73: 116 bits
  Distance between Person 9 and Person 74: 128 bits
  Distance between Person 9 and Person 75: 131 bits
  Distance between Person 9 and Person 76: 129 bits
  Distance between Person 9 and Person 77: 120 bits
  Distance between Person 9 and Person 78: 98 bits
  Distance between Person 9 and Person 79: 149 bits
  Distance between Person 9 and Person 80: 126 bits
  Distance between Person 9 and Person 81: 106 bits
  Distance between Person 9 and Person 82: 123 bits
  Distance between Person 9 and Person 83: 128 bits
  Distance between Person 9 and Person 84: 128 bits
  Distance between Person 9 and Person 85: 131 bits
  Distance between Person 9 and Person 86: 126 bits
  Distance between Person 9 and Person 87: 120 bits
  Distance between Person 9 and Person 88: 113 bits
  Distance between Person 9 and Person 89: 134 bits
  Distance between Person 10 and Person 11: 139 bits
  Distance between Person 10 and Person 12: 136 bits
  Distance between Person 10 and Person 13: 137 bits
  Distance between Person 10 and Person 14: 133 bits
  Distance between Person 10 and Person 15: 117 bits
  Distance between Person 10 and Person 16: 130 bits
  Distance between Person 10 and Person 17: 125 bits
  Distance between Person 10 and Person 18: 112 bits
  Distance between Person 10 and Person 19: 124 bits
  Distance between Person 10 and Person 20: 128 bits
  Distance between Person 10 and Person 21: 123 bits
  Distance between Person 10 and Person 22: 133 bits
  Distance between Person 10 and Person 23: 143 bits
  Distance between Person 10 and Person 24: 127 bits
  Distance between Person 10 and Person 25: 125 bits
  Distance between Person 10 and Person 26: 127 bits
  Distance between Person 10 and Person 27: 131 bits
  Distance between Person 10 and Person 28: 142 bits
  Distance between Person 10 and Person 29: 128 bits
  Distance between Person 10 and Person 30: 122 bits
  Distance between Person 10 and Person 31: 129 bits
  Distance between Person 10 and Person 32: 133 bits
  Distance between Person 10 and Person 33: 124 bits
  Distance between Person 10 and Person 34: 130 bits
  Distance between Person 10 and Person 35: 133 bits
  Distance between Person 10 and Person 36: 111 bits
  Distance between Person 10 and Person 37: 120 bits
  Distance between Person 10 and Person 38: 131 bits
  Distance between Person 10 and Person 39: 121 bits
  Distance between Person 10 and Person 40: 131 bits
  Distance between Person 10 and Person 41: 115 bits
  Distance between Person 10 and Person 42: 111 bits
  Distance between Person 10 and Person 43: 130 bits
  Distance between Person 10 and Person 44: 121 bits
  Distance between Person 10 and Person 45: 137 bits
  Distance between Person 10 and Person 46: 126 bits
  Distance between Person 10 and Person 47: 111 bits
  Distance between Person 10 and Person 48: 122 bits
  Distance between Person 10 and Person 49: 128 bits
  Distance between Person 10 and Person 50: 126 bits
  Distance between Person 10 and Person 51: 112 bits
  Distance between Person 10 and Person 52: 129 bits
  Distance between Person 10 and Person 53: 138 bits
  Distance between Person 10 and Person 54: 123 bits
  Distance between Person 10 and Person 55: 137 bits
  Distance between Person 10 and Person 56: 123 bits
  Distance between Person 10 and Person 57: 124 bits
  Distance between Person 10 and Person 58: 128 bits
  Distance between Person 10 and Person 59: 147 bits
  Distance between Person 10 and Person 60: 134 bits
  Distance between Person 10 and Person 61: 134 bits
  Distance between Person 10 and Person 62: 126 bits
  Distance between Person 10 and Person 63: 116 bits
  Distance between Person 10 and Person 64: 101 bits
  Distance between Person 10 and Person 65: 119 bits
  Distance between Person 10 and Person 66: 109 bits
  Distance between Person 10 and Person 67: 126 bits
  Distance between Person 10 and Person 68: 116 bits
  Distance between Person 10 and Person 69: 126 bits
  Distance between Person 10 and Person 70: 128 bits
  Distance between Person 10 and Person 71: 133 bits
  Distance between Person 10 and Person 72: 142 bits
  Distance between Person 10 and Person 73: 136 bits
  Distance between Person 10 and Person 74: 126 bits
  Distance between Person 10 and Person 75: 95 bits
  Distance between Person 10 and Person 76: 117 bits
  Distance between Person 10 and Person 77: 128 bits
  Distance between Person 10 and Person 78: 130 bits
  Distance between Person 10 and Person 79: 103 bits
  Distance between Person 10 and Person 80: 138 bits
  Distance between Person 10 and Person 81: 120 bits
  Distance between Person 10 and Person 82: 93 bits
  Distance between Person 10 and Person 83: 142 bits
  Distance between Person 10 and Person 84: 120 bits
  Distance between Person 10 and Person 85: 141 bits
  Distance between Person 10 and Person 86: 146 bits
  Distance between Person 10 and Person 87: 134 bits
  Distance between Person 10 and Person 88: 125 bits
  Distance between Person 10 and Person 89: 130 bits
  Distance between Person 11 and Person 12: 127 bits
  Distance between Person 11 and Person 13: 134 bits
  Distance between Person 11 and Person 14: 130 bits
  Distance between Person 11 and Person 15: 112 bits
  Distance between Person 11 and Person 16: 125 bits
  Distance between Person 11 and Person 17: 114 bits
  Distance between Person 11 and Person 18: 125 bits
  Distance between Person 11 and Person 19: 119 bits
  Distance between Person 11 and Person 20: 121 bits
  Distance between Person 11 and Person 21: 122 bits
  Distance between Person 11 and Person 22: 136 bits
  Distance between Person 11 and Person 23: 126 bits
  Distance between Person 11 and Person 24: 118 bits
  Distance between Person 11 and Person 25: 120 bits
  Distance between Person 11 and Person 26: 142 bits
  Distance between Person 11 and Person 27: 146 bits
  Distance between Person 11 and Person 28: 109 bits
  Distance between Person 11 and Person 29: 123 bits
  Distance between Person 11 and Person 30: 131 bits
  Distance between Person 11 and Person 31: 116 bits
  Distance between Person 11 and Person 32: 122 bits
  Distance between Person 11 and Person 33: 137 bits
  Distance between Person 11 and Person 34: 137 bits
  Distance between Person 11 and Person 35: 124 bits
  Distance between Person 11 and Person 36: 124 bits
  Distance between Person 11 and Person 37: 125 bits
  Distance between Person 11 and Person 38: 116 bits
  Distance between Person 11 and Person 39: 110 bits
  Distance between Person 11 and Person 40: 116 bits
  Distance between Person 11 and Person 41: 132 bits
  Distance between Person 11 and Person 42: 116 bits
  Distance between Person 11 and Person 43: 147 bits
  Distance between Person 11 and Person 44: 144 bits
  Distance between Person 11 and Person 45: 120 bits
  Distance between Person 11 and Person 46: 117 bits
  Distance between Person 11 and Person 47: 134 bits
  Distance between Person 11 and Person 48: 135 bits
  Distance between Person 11 and Person 49: 135 bits
  Distance between Person 11 and Person 50: 129 bits
  Distance between Person 11 and Person 51: 159 bits
  Distance between Person 11 and Person 52: 150 bits
  Distance between Person 11 and Person 53: 135 bits
  Distance between Person 11 and Person 54: 134 bits
  Distance between Person 11 and Person 55: 128 bits
  Distance between Person 11 and Person 56: 126 bits
  Distance between Person 11 and Person 57: 107 bits
  Distance between Person 11 and Person 58: 127 bits
  Distance between Person 11 and Person 59: 118 bits
  Distance between Person 11 and Person 60: 123 bits
  Distance between Person 11 and Person 61: 95 bits
  Distance between Person 11 and Person 62: 137 bits
  Distance between Person 11 and Person 63: 135 bits
  Distance between Person 11 and Person 64: 134 bits
  Distance between Person 11 and Person 65: 108 bits
  Distance between Person 11 and Person 66: 140 bits
  Distance between Person 11 and Person 67: 129 bits
  Distance between Person 11 and Person 68: 109 bits
  Distance between Person 11 and Person 69: 141 bits
  Distance between Person 11 and Person 70: 137 bits
  Distance between Person 11 and Person 71: 104 bits
  Distance between Person 11 and Person 72: 123 bits
  Distance between Person 11 and Person 73: 111 bits
  Distance between Person 11 and Person 74: 135 bits
  Distance between Person 11 and Person 75: 146 bits
  Distance between Person 11 and Person 76: 126 bits
  Distance between Person 11 and Person 77: 115 bits
  Distance between Person 11 and Person 78: 111 bits
  Distance between Person 11 and Person 79: 134 bits
  Distance between Person 11 and Person 80: 131 bits
  Distance between Person 11 and Person 81: 135 bits
  Distance between Person 11 and Person 82: 134 bits
  Distance between Person 11 and Person 83: 145 bits
  Distance between Person 11 and Person 84: 119 bits
  Distance between Person 11 and Person 85: 140 bits
  Distance between Person 11 and Person 86: 111 bits
  Distance between Person 11 and Person 87: 123 bits
  Distance between Person 11 and Person 88: 120 bits
  Distance between Person 11 and Person 89: 131 bits
  Distance between Person 12 and Person 13: 151 bits
  Distance between Person 12 and Person 14: 133 bits
  Distance between Person 12 and Person 15: 129 bits
  Distance between Person 12 and Person 16: 120 bits
  Distance between Person 12 and Person 17: 105 bits
  Distance between Person 12 and Person 18: 120 bits
  Distance between Person 12 and Person 19: 126 bits
  Distance between Person 12 and Person 20: 128 bits
  Distance between Person 12 and Person 21: 139 bits
  Distance between Person 12 and Person 22: 127 bits
  Distance between Person 12 and Person 23: 133 bits
  Distance between Person 12 and Person 24: 141 bits
  Distance between Person 12 and Person 25: 131 bits
  Distance between Person 12 and Person 26: 127 bits
  Distance between Person 12 and Person 27: 133 bits
  Distance between Person 12 and Person 28: 120 bits
  Distance between Person 12 and Person 29: 136 bits
  Distance between Person 12 and Person 30: 126 bits
  Distance between Person 12 and Person 31: 125 bits
  Distance between Person 12 and Person 32: 133 bits
  Distance between Person 12 and Person 33: 144 bits
  Distance between Person 12 and Person 34: 138 bits
  Distance between Person 12 and Person 35: 131 bits
  Distance between Person 12 and Person 36: 123 bits
  Distance between Person 12 and Person 37: 144 bits
  Distance between Person 12 and Person 38: 117 bits
  Distance between Person 12 and Person 39: 147 bits
  Distance between Person 12 and Person 40: 127 bits
  Distance between Person 12 and Person 41: 131 bits
  Distance between Person 12 and Person 42: 121 bits
  Distance between Person 12 and Person 43: 134 bits
  Distance between Person 12 and Person 44: 149 bits
  Distance between Person 12 and Person 45: 135 bits
  Distance between Person 12 and Person 46: 120 bits
  Distance between Person 12 and Person 47: 141 bits
  Distance between Person 12 and Person 48: 120 bits
  Distance between Person 12 and Person 49: 134 bits
  Distance between Person 12 and Person 50: 136 bits
  Distance between Person 12 and Person 51: 116 bits
  Distance between Person 12 and Person 52: 137 bits
  Distance between Person 12 and Person 53: 122 bits
  Distance between Person 12 and Person 54: 135 bits
  Distance between Person 12 and Person 55: 131 bits
  Distance between Person 12 and Person 56: 125 bits
  Distance between Person 12 and Person 57: 114 bits
  Distance between Person 12 and Person 58: 118 bits
  Distance between Person 12 and Person 59: 129 bits
  Distance between Person 12 and Person 60: 134 bits
  Distance between Person 12 and Person 61: 114 bits
  Distance between Person 12 and Person 62: 124 bits
  Distance between Person 12 and Person 63: 132 bits
  Distance between Person 12 and Person 64: 125 bits
  Distance between Person 12 and Person 65: 139 bits
  Distance between Person 12 and Person 66: 147 bits
  Distance between Person 12 and Person 67: 144 bits
  Distance between Person 12 and Person 68: 136 bits
  Distance between Person 12 and Person 69: 140 bits
  Distance between Person 12 and Person 70: 126 bits
  Distance between Person 12 and Person 71: 137 bits
  Distance between Person 12 and Person 72: 124 bits
  Distance between Person 12 and Person 73: 116 bits
  Distance between Person 12 and Person 74: 146 bits
  Distance between Person 12 and Person 75: 131 bits
  Distance between Person 12 and Person 76: 129 bits
  Distance between Person 12 and Person 77: 126 bits
  Distance between Person 12 and Person 78: 122 bits
  Distance between Person 12 and Person 79: 123 bits
  Distance between Person 12 and Person 80: 122 bits
  Distance between Person 12 and Person 81: 118 bits
  Distance between Person 12 and Person 82: 141 bits
  Distance between Person 12 and Person 83: 138 bits
  Distance between Person 12 and Person 84: 114 bits
  Distance between Person 12 and Person 85: 117 bits
  Distance between Person 12 and Person 86: 124 bits
  Distance between Person 12 and Person 87: 118 bits
  Distance between Person 12 and Person 88: 117 bits
  Distance between Person 12 and Person 89: 118 bits
  Distance between Person 13 and Person 14: 116 bits
  Distance between Person 13 and Person 15: 122 bits
  Distance between Person 13 and Person 16: 117 bits
  Distance between Person 13 and Person 17: 128 bits
  Distance between Person 13 and Person 18: 133 bits
  Distance between Person 13 and Person 19: 141 bits
  Distance between Person 13 and Person 20: 111 bits
  Distance between Person 13 and Person 21: 110 bits
  Distance between Person 13 and Person 22: 112 bits
  Distance between Person 13 and Person 23: 142 bits
  Distance between Person 13 and Person 24: 120 bits
  Distance between Person 13 and Person 25: 118 bits
  Distance between Person 13 and Person 26: 110 bits
  Distance between Person 13 and Person 27: 128 bits
  Distance between Person 13 and Person 28: 127 bits
  Distance between Person 13 and Person 29: 155 bits
  Distance between Person 13 and Person 30: 139 bits
  Distance between Person 13 and Person 31: 132 bits
  Distance between Person 13 and Person 32: 96 bits
  Distance between Person 13 and Person 33: 125 bits
  Distance between Person 13 and Person 34: 135 bits
  Distance between Person 13 and Person 35: 112 bits
  Distance between Person 13 and Person 36: 122 bits
  Distance between Person 13 and Person 37: 115 bits
  Distance between Person 13 and Person 38: 116 bits
  Distance between Person 13 and Person 39: 108 bits
  Distance between Person 13 and Person 40: 148 bits
  Distance between Person 13 and Person 41: 142 bits
  Distance between Person 13 and Person 42: 142 bits
  Distance between Person 13 and Person 43: 127 bits
  Distance between Person 13 and Person 44: 110 bits
  Distance between Person 13 and Person 45: 106 bits
  Distance between Person 13 and Person 46: 147 bits
  Distance between Person 13 and Person 47: 136 bits
  Distance between Person 13 and Person 48: 135 bits
  Distance between Person 13 and Person 49: 121 bits
  Distance between Person 13 and Person 50: 133 bits
  Distance between Person 13 and Person 51: 119 bits
  Distance between Person 13 and Person 52: 114 bits
  Distance between Person 13 and Person 53: 131 bits
  Distance between Person 13 and Person 54: 112 bits
  Distance between Person 13 and Person 55: 128 bits
  Distance between Person 13 and Person 56: 114 bits
  Distance between Person 13 and Person 57: 153 bits
  Distance between Person 13 and Person 58: 141 bits
  Distance between Person 13 and Person 59: 132 bits
  Distance between Person 13 and Person 60: 115 bits
  Distance between Person 13 and Person 61: 137 bits
  Distance between Person 13 and Person 62: 123 bits
  Distance between Person 13 and Person 63: 143 bits
  Distance between Person 13 and Person 64: 106 bits
  Distance between Person 13 and Person 65: 128 bits
  Distance between Person 13 and Person 66: 116 bits
  Distance between Person 13 and Person 67: 131 bits
  Distance between Person 13 and Person 68: 121 bits
  Distance between Person 13 and Person 69: 133 bits
  Distance between Person 13 and Person 70: 101 bits
  Distance between Person 13 and Person 71: 146 bits
  Distance between Person 13 and Person 72: 113 bits
  Distance between Person 13 and Person 73: 121 bits
  Distance between Person 13 and Person 74: 107 bits
  Distance between Person 13 and Person 75: 138 bits
  Distance between Person 13 and Person 76: 138 bits
  Distance between Person 13 and Person 77: 139 bits
  Distance between Person 13 and Person 78: 115 bits
  Distance between Person 13 and Person 79: 126 bits
  Distance between Person 13 and Person 80: 97 bits
  Distance between Person 13 and Person 81: 145 bits
  Distance between Person 13 and Person 82: 136 bits
  Distance between Person 13 and Person 83: 135 bits
  Distance between Person 13 and Person 84: 135 bits
  Distance between Person 13 and Person 85: 130 bits
  Distance between Person 13 and Person 86: 107 bits
  Distance between Person 13 and Person 87: 135 bits
  Distance between Person 13 and Person 88: 140 bits
  Distance between Person 13 and Person 89: 155 bits
  Distance between Person 14 and Person 15: 116 bits
  Distance between Person 14 and Person 16: 99 bits
  Distance between Person 14 and Person 17: 122 bits
  Distance between Person 14 and Person 18: 149 bits
  Distance between Person 14 and Person 19: 111 bits
  Distance between Person 14 and Person 20: 119 bits
  Distance between Person 14 and Person 21: 124 bits
  Distance between Person 14 and Person 22: 130 bits
  Distance between Person 14 and Person 23: 146 bits
  Distance between Person 14 and Person 24: 138 bits
  Distance between Person 14 and Person 25: 110 bits
  Distance between Person 14 and Person 26: 132 bits
  Distance between Person 14 and Person 27: 114 bits
  Distance between Person 14 and Person 28: 131 bits
  Distance between Person 14 and Person 29: 131 bits
  Distance between Person 14 and Person 30: 125 bits
  Distance between Person 14 and Person 31: 128 bits
  Distance between Person 14 and Person 32: 140 bits
  Distance between Person 14 and Person 33: 129 bits
  Distance between Person 14 and Person 34: 119 bits
  Distance between Person 14 and Person 35: 120 bits
  Distance between Person 14 and Person 36: 152 bits
  Distance between Person 14 and Person 37: 121 bits
  Distance between Person 14 and Person 38: 98 bits
  Distance between Person 14 and Person 39: 142 bits
  Distance between Person 14 and Person 40: 110 bits
  Distance between Person 14 and Person 41: 140 bits
  Distance between Person 14 and Person 42: 136 bits
  Distance between Person 14 and Person 43: 115 bits
  Distance between Person 14 and Person 44: 120 bits
  Distance between Person 14 and Person 45: 114 bits
  Distance between Person 14 and Person 46: 119 bits
  Distance between Person 14 and Person 47: 118 bits
  Distance between Person 14 and Person 48: 151 bits
  Distance between Person 14 and Person 49: 129 bits
  Distance between Person 14 and Person 50: 121 bits
  Distance between Person 14 and Person 51: 113 bits
  Distance between Person 14 and Person 52: 100 bits
  Distance between Person 14 and Person 53: 133 bits
  Distance between Person 14 and Person 54: 134 bits
  Distance between Person 14 and Person 55: 100 bits
  Distance between Person 14 and Person 56: 124 bits
  Distance between Person 14 and Person 57: 129 bits
  Distance between Person 14 and Person 58: 113 bits
  Distance between Person 14 and Person 59: 132 bits
  Distance between Person 14 and Person 60: 123 bits
  Distance between Person 14 and Person 61: 113 bits
  Distance between Person 14 and Person 62: 149 bits
  Distance between Person 14 and Person 63: 133 bits
  Distance between Person 14 and Person 64: 144 bits
  Distance between Person 14 and Person 65: 118 bits
  Distance between Person 14 and Person 66: 114 bits
  Distance between Person 14 and Person 67: 129 bits
  Distance between Person 14 and Person 68: 107 bits
  Distance between Person 14 and Person 69: 127 bits
  Distance between Person 14 and Person 70: 121 bits
  Distance between Person 14 and Person 71: 122 bits
  Distance between Person 14 and Person 72: 125 bits
  Distance between Person 14 and Person 73: 117 bits
  Distance between Person 14 and Person 74: 123 bits
  Distance between Person 14 and Person 75: 132 bits
  Distance between Person 14 and Person 76: 118 bits
  Distance between Person 14 and Person 77: 121 bits
  Distance between Person 14 and Person 78: 115 bits
  Distance between Person 14 and Person 79: 134 bits
  Distance between Person 14 and Person 80: 83 bits
  Distance between Person 14 and Person 81: 145 bits
  Distance between Person 14 and Person 82: 114 bits
  Distance between Person 14 and Person 83: 119 bits
  Distance between Person 14 and Person 84: 127 bits
  Distance between Person 14 and Person 85: 122 bits
  Distance between Person 14 and Person 86: 107 bits
  Distance between Person 14 and Person 87: 137 bits
  Distance between Person 14 and Person 88: 124 bits
  Distance between Person 14 and Person 89: 109 bits
  Distance between Person 15 and Person 16: 131 bits
  Distance between Person 15 and Person 17: 116 bits
  Distance between Person 15 and Person 18: 131 bits
  Distance between Person 15 and Person 19: 131 bits
  Distance between Person 15 and Person 20: 135 bits
  Distance between Person 15 and Person 21: 126 bits
  Distance between Person 15 and Person 22: 144 bits
  Distance between Person 15 and Person 23: 122 bits
  Distance between Person 15 and Person 24: 132 bits
  Distance between Person 15 and Person 25: 134 bits
  Distance between Person 15 and Person 26: 134 bits
  Distance between Person 15 and Person 27: 162 bits
  Distance between Person 15 and Person 28: 131 bits
  Distance between Person 15 and Person 29: 125 bits
  Distance between Person 15 and Person 30: 137 bits
  Distance between Person 15 and Person 31: 130 bits
  Distance between Person 15 and Person 32: 136 bits
  Distance between Person 15 and Person 33: 153 bits
  Distance between Person 15 and Person 34: 135 bits
  Distance between Person 15 and Person 35: 132 bits
  Distance between Person 15 and Person 36: 114 bits
  Distance between Person 15 and Person 37: 119 bits
  Distance between Person 15 and Person 38: 110 bits
  Distance between Person 15 and Person 39: 130 bits
  Distance between Person 15 and Person 40: 128 bits
  Distance between Person 15 and Person 41: 102 bits
  Distance between Person 15 and Person 42: 144 bits
  Distance between Person 15 and Person 43: 127 bits
  Distance between Person 15 and Person 44: 122 bits
  Distance between Person 15 and Person 45: 110 bits
  Distance between Person 15 and Person 46: 135 bits
  Distance between Person 15 and Person 47: 130 bits
  Distance between Person 15 and Person 48: 123 bits
  Distance between Person 15 and Person 49: 111 bits
  Distance between Person 15 and Person 50: 123 bits
  Distance between Person 15 and Person 51: 125 bits
  Distance between Person 15 and Person 52: 122 bits
  Distance between Person 15 and Person 53: 147 bits
  Distance between Person 15 and Person 54: 124 bits
  Distance between Person 15 and Person 55: 118 bits
  Distance between Person 15 and Person 56: 132 bits
  Distance between Person 15 and Person 57: 135 bits
  Distance between Person 15 and Person 58: 123 bits
  Distance between Person 15 and Person 59: 140 bits
  Distance between Person 15 and Person 60: 107 bits
  Distance between Person 15 and Person 61: 141 bits
  Distance between Person 15 and Person 62: 131 bits
  Distance between Person 15 and Person 63: 129 bits
  Distance between Person 15 and Person 64: 114 bits
  Distance between Person 15 and Person 65: 130 bits
  Distance between Person 15 and Person 66: 140 bits
  Distance between Person 15 and Person 67: 159 bits
  Distance between Person 15 and Person 68: 83 bits
  Distance between Person 15 and Person 69: 147 bits
  Distance between Person 15 and Person 70: 133 bits
  Distance between Person 15 and Person 71: 132 bits
  Distance between Person 15 and Person 72: 117 bits
  Distance between Person 15 and Person 73: 123 bits
  Distance between Person 15 and Person 74: 123 bits
  Distance between Person 15 and Person 75: 122 bits
  Distance between Person 15 and Person 76: 136 bits
  Distance between Person 15 and Person 77: 129 bits
  Distance between Person 15 and Person 78: 141 bits
  Distance between Person 15 and Person 79: 128 bits
  Distance between Person 15 and Person 80: 103 bits
  Distance between Person 15 and Person 81: 135 bits
  Distance between Person 15 and Person 82: 138 bits
  Distance between Person 15 and Person 83: 121 bits
  Distance between Person 15 and Person 84: 127 bits
  Distance between Person 15 and Person 85: 118 bits
  Distance between Person 15 and Person 86: 139 bits
  Distance between Person 15 and Person 87: 113 bits
  Distance between Person 15 and Person 88: 144 bits
  Distance between Person 15 and Person 89: 127 bits
  Distance between Person 16 and Person 17: 125 bits
  Distance between Person 16 and Person 18: 124 bits
  Distance between Person 16 and Person 19: 112 bits
  Distance between Person 16 and Person 20: 138 bits
  Distance between Person 16 and Person 21: 137 bits
  Distance between Person 16 and Person 22: 141 bits
  Distance between Person 16 and Person 23: 153 bits
  Distance between Person 16 and Person 24: 111 bits
  Distance between Person 16 and Person 25: 99 bits
  Distance between Person 16 and Person 26: 123 bits
  Distance between Person 16 and Person 27: 101 bits
  Distance between Person 16 and Person 28: 134 bits
  Distance between Person 16 and Person 29: 138 bits
  Distance between Person 16 and Person 30: 148 bits
  Distance between Person 16 and Person 31: 135 bits
  Distance between Person 16 and Person 32: 117 bits
  Distance between Person 16 and Person 33: 130 bits
  Distance between Person 16 and Person 34: 118 bits
  Distance between Person 16 and Person 35: 125 bits
  Distance between Person 16 and Person 36: 125 bits
  Distance between Person 16 and Person 37: 122 bits
  Distance between Person 16 and Person 38: 141 bits
  Distance between Person 16 and Person 39: 121 bits
  Distance between Person 16 and Person 40: 113 bits
  Distance between Person 16 and Person 41: 105 bits
  Distance between Person 16 and Person 42: 121 bits
  Distance between Person 16 and Person 43: 124 bits
  Distance between Person 16 and Person 44: 137 bits
  Distance between Person 16 and Person 45: 115 bits
  Distance between Person 16 and Person 46: 118 bits
  Distance between Person 16 and Person 47: 135 bits
  Distance between Person 16 and Person 48: 130 bits
  Distance between Person 16 and Person 49: 116 bits
  Distance between Person 16 and Person 50: 134 bits
  Distance between Person 16 and Person 51: 120 bits
  Distance between Person 16 and Person 52: 101 bits
  Distance between Person 16 and Person 53: 138 bits
  Distance between Person 16 and Person 54: 141 bits
  Distance between Person 16 and Person 55: 111 bits
  Distance between Person 16 and Person 56: 109 bits
  Distance between Person 16 and Person 57: 120 bits
  Distance between Person 16 and Person 58: 112 bits
  Distance between Person 16 and Person 59: 117 bits
  Distance between Person 16 and Person 60: 132 bits
  Distance between Person 16 and Person 61: 112 bits
  Distance between Person 16 and Person 62: 116 bits
  Distance between Person 16 and Person 63: 134 bits
  Distance between Person 16 and Person 64: 137 bits
  Distance between Person 16 and Person 65: 117 bits
  Distance between Person 16 and Person 66: 123 bits
  Distance between Person 16 and Person 67: 124 bits
  Distance between Person 16 and Person 68: 112 bits
  Distance between Person 16 and Person 69: 130 bits
  Distance between Person 16 and Person 70: 136 bits
  Distance between Person 16 and Person 71: 127 bits
  Distance between Person 16 and Person 72: 128 bits
  Distance between Person 16 and Person 73: 138 bits
  Distance between Person 16 and Person 74: 124 bits
  Distance between Person 16 and Person 75: 139 bits
  Distance between Person 16 and Person 76: 119 bits
  Distance between Person 16 and Person 77: 124 bits
  Distance between Person 16 and Person 78: 112 bits
  Distance between Person 16 and Person 79: 103 bits
  Distance between Person 16 and Person 80: 96 bits
  Distance between Person 16 and Person 81: 132 bits
  Distance between Person 16 and Person 82: 137 bits
  Distance between Person 16 and Person 83: 128 bits
  Distance between Person 16 and Person 84: 64 bits
  Distance between Person 16 and Person 85: 133 bits
  Distance between Person 16 and Person 86: 108 bits
  Distance between Person 16 and Person 87: 146 bits
  Distance between Person 16 and Person 88: 125 bits
  Distance between Person 16 and Person 89: 136 bits
  Distance between Person 17 and Person 18: 123 bits
  Distance between Person 17 and Person 19: 135 bits
  Distance between Person 17 and Person 20: 123 bits
  Distance between Person 17 and Person 21: 142 bits
  Distance between Person 17 and Person 22: 138 bits
  Distance between Person 17 and Person 23: 120 bits
  Distance between Person 17 and Person 24: 130 bits
  Distance between Person 17 and Person 25: 122 bits
  Distance between Person 17 and Person 26: 136 bits
  Distance between Person 17 and Person 27: 116 bits
  Distance between Person 17 and Person 28: 143 bits
  Distance between Person 17 and Person 29: 147 bits
  Distance between Person 17 and Person 30: 99 bits
  Distance between Person 17 and Person 31: 94 bits
  Distance between Person 17 and Person 32: 128 bits
  Distance between Person 17 and Person 33: 127 bits
  Distance between Person 17 and Person 34: 125 bits
  Distance between Person 17 and Person 35: 134 bits
  Distance between Person 17 and Person 36: 118 bits
  Distance between Person 17 and Person 37: 121 bits
  Distance between Person 17 and Person 38: 112 bits
  Distance between Person 17 and Person 39: 138 bits
  Distance between Person 17 and Person 40: 116 bits
  Distance between Person 17 and Person 41: 144 bits
  Distance between Person 17 and Person 42: 122 bits
  Distance between Person 17 and Person 43: 139 bits
  Distance between Person 17 and Person 44: 148 bits
  Distance between Person 17 and Person 45: 114 bits
  Distance between Person 17 and Person 46: 135 bits
  Distance between Person 17 and Person 47: 144 bits
  Distance between Person 17 and Person 48: 125 bits
  Distance between Person 17 and Person 49: 109 bits
  Distance between Person 17 and Person 50: 133 bits
  Distance between Person 17 and Person 51: 127 bits
  Distance between Person 17 and Person 52: 134 bits
  Distance between Person 17 and Person 53: 143 bits
  Distance between Person 17 and Person 54: 164 bits
  Distance between Person 17 and Person 55: 138 bits
  Distance between Person 17 and Person 56: 138 bits
  Distance between Person 17 and Person 57: 129 bits
  Distance between Person 17 and Person 58: 141 bits
  Distance between Person 17 and Person 59: 140 bits
  Distance between Person 17 and Person 60: 143 bits
  Distance between Person 17 and Person 61: 129 bits
  Distance between Person 17 and Person 62: 123 bits
  Distance between Person 17 and Person 63: 137 bits
  Distance between Person 17 and Person 64: 128 bits
  Distance between Person 17 and Person 65: 118 bits
  Distance between Person 17 and Person 66: 140 bits
  Distance between Person 17 and Person 67: 125 bits
  Distance between Person 17 and Person 68: 121 bits
  Distance between Person 17 and Person 69: 135 bits
  Distance between Person 17 and Person 70: 127 bits
  Distance between Person 17 and Person 71: 116 bits
  Distance between Person 17 and Person 72: 127 bits
  Distance between Person 17 and Person 73: 127 bits
  Distance between Person 17 and Person 74: 129 bits
  Distance between Person 17 and Person 75: 140 bits
  Distance between Person 17 and Person 76: 154 bits
  Distance between Person 17 and Person 77: 123 bits
  Distance between Person 17 and Person 78: 111 bits
  Distance between Person 17 and Person 79: 132 bits
  Distance between Person 17 and Person 80: 119 bits
  Distance between Person 17 and Person 81: 127 bits
  Distance between Person 17 and Person 82: 140 bits
  Distance between Person 17 and Person 83: 137 bits
  Distance between Person 17 and Person 84: 127 bits
  Distance between Person 17 and Person 85: 136 bits
  Distance between Person 17 and Person 86: 115 bits
  Distance between Person 17 and Person 87: 119 bits
  Distance between Person 17 and Person 88: 134 bits
  Distance between Person 17 and Person 89: 139 bits
  Distance between Person 18 and Person 19: 130 bits
  Distance between Person 18 and Person 20: 134 bits
  Distance between Person 18 and Person 21: 119 bits
  Distance between Person 18 and Person 22: 129 bits
  Distance between Person 18 and Person 23: 123 bits
  Distance between Person 18 and Person 24: 149 bits
  Distance between Person 18 and Person 25: 119 bits
  Distance between Person 18 and Person 26: 115 bits
  Distance between Person 18 and Person 27: 129 bits
  Distance between Person 18 and Person 28: 136 bits
  Distance between Person 18 and Person 29: 114 bits
  Distance between Person 18 and Person 30: 144 bits
  Distance between Person 18 and Person 31: 135 bits
  Distance between Person 18 and Person 32: 141 bits
  Distance between Person 18 and Person 33: 130 bits
  Distance between Person 18 and Person 34: 130 bits
  Distance between Person 18 and Person 35: 135 bits
  Distance between Person 18 and Person 36: 109 bits
  Distance between Person 18 and Person 37: 128 bits
  Distance between Person 18 and Person 38: 131 bits
  Distance between Person 18 and Person 39: 151 bits
  Distance between Person 18 and Person 40: 103 bits
  Distance between Person 18 and Person 41: 115 bits
  Distance between Person 18 and Person 42: 109 bits
  Distance between Person 18 and Person 43: 126 bits
  Distance between Person 18 and Person 44: 145 bits
  Distance between Person 18 and Person 45: 127 bits
  Distance between Person 18 and Person 46: 120 bits
  Distance between Person 18 and Person 47: 147 bits
  Distance between Person 18 and Person 48: 126 bits
  Distance between Person 18 and Person 49: 134 bits
  Distance between Person 18 and Person 50: 138 bits
  Distance between Person 18 and Person 51: 122 bits
  Distance between Person 18 and Person 52: 147 bits
  Distance between Person 18 and Person 53: 140 bits
  Distance between Person 18 and Person 54: 115 bits
  Distance between Person 18 and Person 55: 131 bits
  Distance between Person 18 and Person 56: 97 bits
  Distance between Person 18 and Person 57: 120 bits
  Distance between Person 18 and Person 58: 138 bits
  Distance between Person 18 and Person 59: 117 bits
  Distance between Person 18 and Person 60: 108 bits
  Distance between Person 18 and Person 61: 140 bits
  Distance between Person 18 and Person 62: 118 bits
  Distance between Person 18 and Person 63: 116 bits
  Distance between Person 18 and Person 64: 119 bits
  Distance between Person 18 and Person 65: 121 bits
  Distance between Person 18 and Person 66: 133 bits
  Distance between Person 18 and Person 67: 132 bits
  Distance between Person 18 and Person 68: 124 bits
  Distance between Person 18 and Person 69: 132 bits
  Distance between Person 18 and Person 70: 130 bits
  Distance between Person 18 and Person 71: 99 bits
  Distance between Person 18 and Person 72: 162 bits
  Distance between Person 18 and Person 73: 144 bits
  Distance between Person 18 and Person 74: 140 bits
  Distance between Person 18 and Person 75: 127 bits
  Distance between Person 18 and Person 76: 145 bits
  Distance between Person 18 and Person 77: 118 bits
  Distance between Person 18 and Person 78: 98 bits
  Distance between Person 18 and Person 79: 113 bits
  Distance between Person 18 and Person 80: 136 bits
  Distance between Person 18 and Person 81: 112 bits
  Distance between Person 18 and Person 82: 113 bits
  Distance between Person 18 and Person 83: 136 bits
  Distance between Person 18 and Person 84: 120 bits
  Distance between Person 18 and Person 85: 131 bits
  Distance between Person 18 and Person 86: 142 bits
  Distance between Person 18 and Person 87: 156 bits
  Distance between Person 18 and Person 88: 105 bits
  Distance between Person 18 and Person 89: 156 bits
  Distance between Person 19 and Person 20: 102 bits
  Distance between Person 19 and Person 21: 139 bits
  Distance between Person 19 and Person 22: 141 bits
  Distance between Person 19 and Person 23: 135 bits
  Distance between Person 19 and Person 24: 123 bits
  Distance between Person 19 and Person 25: 139 bits
  Distance between Person 19 and Person 26: 127 bits
  Distance between Person 19 and Person 27: 109 bits
  Distance between Person 19 and Person 28: 134 bits
  Distance between Person 19 and Person 29: 100 bits
  Distance between Person 19 and Person 30: 136 bits
  Distance between Person 19 and Person 31: 133 bits
  Distance between Person 19 and Person 32: 123 bits
  Distance between Person 19 and Person 33: 132 bits
  Distance between Person 19 and Person 34: 134 bits
  Distance between Person 19 and Person 35: 121 bits
  Distance between Person 19 and Person 36: 141 bits
  Distance between Person 19 and Person 37: 120 bits
  Distance between Person 19 and Person 38: 135 bits
  Distance between Person 19 and Person 39: 113 bits
  Distance between Person 19 and Person 40: 97 bits
  Distance between Person 19 and Person 41: 117 bits
  Distance between Person 19 and Person 42: 121 bits
  Distance between Person 19 and Person 43: 122 bits
  Distance between Person 19 and Person 44: 141 bits
  Distance between Person 19 and Person 45: 125 bits
  Distance between Person 19 and Person 46: 150 bits
  Distance between Person 19 and Person 47: 101 bits
  Distance between Person 19 and Person 48: 118 bits
  Distance between Person 19 and Person 49: 118 bits
  Distance between Person 19 and Person 50: 120 bits
  Distance between Person 19 and Person 51: 132 bits
  Distance between Person 19 and Person 52: 119 bits
  Distance between Person 19 and Person 53: 148 bits
  Distance between Person 19 and Person 54: 135 bits
  Distance between Person 19 and Person 55: 91 bits
  Distance between Person 19 and Person 56: 129 bits
  Distance between Person 19 and Person 57: 122 bits
  Distance between Person 19 and Person 58: 116 bits
  Distance between Person 19 and Person 59: 129 bits
  Distance between Person 19 and Person 60: 114 bits
  Distance between Person 19 and Person 61: 104 bits
  Distance between Person 19 and Person 62: 134 bits
  Distance between Person 19 and Person 63: 148 bits
  Distance between Person 19 and Person 64: 119 bits
  Distance between Person 19 and Person 65: 131 bits
  Distance between Person 19 and Person 66: 107 bits
  Distance between Person 19 and Person 67: 126 bits
  Distance between Person 19 and Person 68: 132 bits
  Distance between Person 19 and Person 69: 114 bits
  Distance between Person 19 and Person 70: 144 bits
  Distance between Person 19 and Person 71: 101 bits
  Distance between Person 19 and Person 72: 128 bits
  Distance between Person 19 and Person 73: 134 bits
  Distance between Person 19 and Person 74: 140 bits
  Distance between Person 19 and Person 75: 119 bits
  Distance between Person 19 and Person 76: 133 bits
  Distance between Person 19 and Person 77: 114 bits
  Distance between Person 19 and Person 78: 112 bits
  Distance between Person 19 and Person 79: 123 bits
  Distance between Person 19 and Person 80: 108 bits
  Distance between Person 19 and Person 81: 130 bits
  Distance between Person 19 and Person 82: 123 bits
  Distance between Person 19 and Person 83: 128 bits
  Distance between Person 19 and Person 84: 138 bits
  Distance between Person 19 and Person 85: 115 bits
  Distance between Person 19 and Person 86: 100 bits
  Distance between Person 19 and Person 87: 140 bits
  Distance between Person 19 and Person 88: 143 bits
  Distance between Person 19 and Person 89: 126 bits
  Distance between Person 20 and Person 21: 125 bits
  Distance between Person 20 and Person 22: 115 bits
  Distance between Person 20 and Person 23: 115 bits
  Distance between Person 20 and Person 24: 139 bits
  Distance between Person 20 and Person 25: 129 bits
  Distance between Person 20 and Person 26: 131 bits
  Distance between Person 20 and Person 27: 119 bits
  Distance between Person 20 and Person 28: 130 bits
  Distance between Person 20 and Person 29: 128 bits
  Distance between Person 20 and Person 30: 128 bits
  Distance between Person 20 and Person 31: 125 bits
  Distance between Person 20 and Person 32: 117 bits
  Distance between Person 20 and Person 33: 128 bits
  Distance between Person 20 and Person 34: 140 bits
  Distance between Person 20 and Person 35: 125 bits
  Distance between Person 20 and Person 36: 121 bits
  Distance between Person 20 and Person 37: 124 bits
  Distance between Person 20 and Person 38: 129 bits
  Distance between Person 20 and Person 39: 119 bits
  Distance between Person 20 and Person 40: 129 bits
  Distance between Person 20 and Person 41: 115 bits
  Distance between Person 20 and Person 42: 113 bits
  Distance between Person 20 and Person 43: 130 bits
  Distance between Person 20 and Person 44: 125 bits
  Distance between Person 20 and Person 45: 103 bits
  Distance between Person 20 and Person 46: 124 bits
  Distance between Person 20 and Person 47: 111 bits
  Distance between Person 20 and Person 48: 146 bits
  Distance between Person 20 and Person 49: 122 bits
  Distance between Person 20 and Person 50: 122 bits
  Distance between Person 20 and Person 51: 122 bits
  Distance between Person 20 and Person 52: 147 bits
  Distance between Person 20 and Person 53: 136 bits
  Distance between Person 20 and Person 54: 123 bits
  Distance between Person 20 and Person 55: 135 bits
  Distance between Person 20 and Person 56: 119 bits
  Distance between Person 20 and Person 57: 132 bits
  Distance between Person 20 and Person 58: 124 bits
  Distance between Person 20 and Person 59: 123 bits
  Distance between Person 20 and Person 60: 128 bits
  Distance between Person 20 and Person 61: 120 bits
  Distance between Person 20 and Person 62: 138 bits
  Distance between Person 20 and Person 63: 134 bits
  Distance between Person 20 and Person 64: 133 bits
  Distance between Person 20 and Person 65: 137 bits
  Distance between Person 20 and Person 66: 133 bits
  Distance between Person 20 and Person 67: 122 bits
  Distance between Person 20 and Person 68: 130 bits
  Distance between Person 20 and Person 69: 126 bits
  Distance between Person 20 and Person 70: 130 bits
  Distance between Person 20 and Person 71: 145 bits
  Distance between Person 20 and Person 72: 120 bits
  Distance between Person 20 and Person 73: 110 bits
  Distance between Person 20 and Person 74: 124 bits
  Distance between Person 20 and Person 75: 123 bits
  Distance between Person 20 and Person 76: 127 bits
  Distance between Person 20 and Person 77: 130 bits
  Distance between Person 20 and Person 78: 98 bits
  Distance between Person 20 and Person 79: 131 bits
  Distance between Person 20 and Person 80: 126 bits
  Distance between Person 20 and Person 81: 132 bits
  Distance between Person 20 and Person 82: 99 bits
  Distance between Person 20 and Person 83: 134 bits
  Distance between Person 20 and Person 84: 146 bits
  Distance between Person 20 and Person 85: 141 bits
  Distance between Person 20 and Person 86: 120 bits
  Distance between Person 20 and Person 87: 122 bits
  Distance between Person 20 and Person 88: 131 bits
  Distance between Person 20 and Person 89: 114 bits
  Distance between Person 21 and Person 22: 140 bits
  Distance between Person 21 and Person 23: 124 bits
  Distance between Person 21 and Person 24: 122 bits
  Distance between Person 21 and Person 25: 130 bits
  Distance between Person 21 and Person 26: 118 bits
  Distance between Person 21 and Person 27: 130 bits
  Distance between Person 21 and Person 28: 107 bits
  Distance between Person 21 and Person 29: 121 bits
  Distance between Person 21 and Person 30: 155 bits
  Distance between Person 21 and Person 31: 130 bits
  Distance between Person 21 and Person 32: 112 bits
  Distance between Person 21 and Person 33: 127 bits
  Distance between Person 21 and Person 34: 113 bits
  Distance between Person 21 and Person 35: 144 bits
  Distance between Person 21 and Person 36: 132 bits
  Distance between Person 21 and Person 37: 119 bits
  Distance between Person 21 and Person 38: 86 bits
  Distance between Person 21 and Person 39: 120 bits
  Distance between Person 21 and Person 40: 112 bits
  Distance between Person 21 and Person 41: 132 bits
  Distance between Person 21 and Person 42: 126 bits
  Distance between Person 21 and Person 43: 131 bits
  Distance between Person 21 and Person 44: 108 bits
  Distance between Person 21 and Person 45: 130 bits
  Distance between Person 21 and Person 46: 111 bits
  Distance between Person 21 and Person 47: 130 bits
  Distance between Person 21 and Person 48: 125 bits
  Distance between Person 21 and Person 49: 137 bits
  Distance between Person 21 and Person 50: 125 bits
  Distance between Person 21 and Person 51: 123 bits
  Distance between Person 21 and Person 52: 140 bits
  Distance between Person 21 and Person 53: 123 bits
  Distance between Person 21 and Person 54: 100 bits
  Distance between Person 21 and Person 55: 126 bits
  Distance between Person 21 and Person 56: 130 bits
  Distance between Person 21 and Person 57: 121 bits
  Distance between Person 21 and Person 58: 125 bits
  Distance between Person 21 and Person 59: 120 bits
  Distance between Person 21 and Person 60: 111 bits
  Distance between Person 21 and Person 61: 117 bits
  Distance between Person 21 and Person 62: 97 bits
  Distance between Person 21 and Person 63: 139 bits
  Distance between Person 21 and Person 64: 112 bits
  Distance between Person 21 and Person 65: 112 bits
  Distance between Person 21 and Person 66: 128 bits
  Distance between Person 21 and Person 67: 135 bits
  Distance between Person 21 and Person 68: 137 bits
  Distance between Person 21 and Person 69: 135 bits
  Distance between Person 21 and Person 70: 39 bits
  Distance between Person 21 and Person 71: 132 bits
  Distance between Person 21 and Person 72: 123 bits
  Distance between Person 21 and Person 73: 127 bits
  Distance between Person 21 and Person 74: 133 bits
  Distance between Person 21 and Person 75: 106 bits
  Distance between Person 21 and Person 76: 120 bits
  Distance between Person 21 and Person 77: 117 bits
  Distance between Person 21 and Person 78: 117 bits
  Distance between Person 21 and Person 79: 140 bits
  Distance between Person 21 and Person 80: 113 bits
  Distance between Person 21 and Person 81: 155 bits
  Distance between Person 21 and Person 82: 122 bits
  Distance between Person 21 and Person 83: 133 bits
  Distance between Person 21 and Person 84: 129 bits
  Distance between Person 21 and Person 85: 130 bits
  Distance between Person 21 and Person 86: 119 bits
  Distance between Person 21 and Person 87: 139 bits
  Distance between Person 21 and Person 88: 108 bits
  Distance between Person 21 and Person 89: 133 bits
  Distance between Person 22 and Person 23: 124 bits
  Distance between Person 22 and Person 24: 122 bits
  Distance between Person 22 and Person 25: 134 bits
  Distance between Person 22 and Person 26: 116 bits
  Distance between Person 22 and Person 27: 120 bits
  Distance between Person 22 and Person 28: 129 bits
  Distance between Person 22 and Person 29: 125 bits
  Distance between Person 22 and Person 30: 107 bits
  Distance between Person 22 and Person 31: 132 bits
  Distance between Person 22 and Person 32: 112 bits
  Distance between Person 22 and Person 33: 149 bits
  Distance between Person 22 and Person 34: 125 bits
  Distance between Person 22 and Person 35: 132 bits
  Distance between Person 22 and Person 36: 106 bits
  Distance between Person 22 and Person 37: 155 bits
  Distance between Person 22 and Person 38: 152 bits
  Distance between Person 22 and Person 39: 120 bits
  Distance between Person 22 and Person 40: 120 bits
  Distance between Person 22 and Person 41: 136 bits
  Distance between Person 22 and Person 42: 136 bits
  Distance between Person 22 and Person 43: 99 bits
  Distance between Person 22 and Person 44: 140 bits
  Distance between Person 22 and Person 45: 150 bits
  Distance between Person 22 and Person 46: 125 bits
  Distance between Person 22 and Person 47: 128 bits
  Distance between Person 22 and Person 48: 133 bits
  Distance between Person 22 and Person 49: 137 bits
  Distance between Person 22 and Person 50: 119 bits
  Distance between Person 22 and Person 51: 113 bits
  Distance between Person 22 and Person 52: 120 bits
  Distance between Person 22 and Person 53: 127 bits
  Distance between Person 22 and Person 54: 118 bits
  Distance between Person 22 and Person 55: 134 bits
  Distance between Person 22 and Person 56: 88 bits
  Distance between Person 22 and Person 57: 119 bits
  Distance between Person 22 and Person 58: 137 bits
  Distance between Person 22 and Person 59: 122 bits
  Distance between Person 22 and Person 60: 137 bits
  Distance between Person 22 and Person 61: 133 bits
  Distance between Person 22 and Person 62: 127 bits
  Distance between Person 22 and Person 63: 141 bits
  Distance between Person 22 and Person 64: 126 bits
  Distance between Person 22 and Person 65: 142 bits
  Distance between Person 22 and Person 66: 144 bits
  Distance between Person 22 and Person 67: 131 bits
  Distance between Person 22 and Person 68: 121 bits
  Distance between Person 22 and Person 69: 129 bits
  Distance between Person 22 and Person 70: 143 bits
  Distance between Person 22 and Person 71: 140 bits
  Distance between Person 22 and Person 72: 149 bits
  Distance between Person 22 and Person 73: 103 bits
  Distance between Person 22 and Person 74: 121 bits
  Distance between Person 22 and Person 75: 140 bits
  Distance between Person 22 and Person 76: 128 bits
  Distance between Person 22 and Person 77: 139 bits
  Distance between Person 22 and Person 78: 135 bits
  Distance between Person 22 and Person 79: 128 bits
  Distance between Person 22 and Person 80: 147 bits
  Distance between Person 22 and Person 81: 129 bits
  Distance between Person 22 and Person 82: 132 bits
  Distance between Person 22 and Person 83: 127 bits
  Distance between Person 22 and Person 84: 113 bits
  Distance between Person 22 and Person 85: 126 bits
  Distance between Person 22 and Person 86: 149 bits
  Distance between Person 22 and Person 87: 133 bits
  Distance between Person 22 and Person 88: 138 bits
  Distance between Person 22 and Person 89: 113 bits
  Distance between Person 23 and Person 24: 152 bits
  Distance between Person 23 and Person 25: 124 bits
  Distance between Person 23 and Person 26: 162 bits
  Distance between Person 23 and Person 27: 136 bits
  Distance between Person 23 and Person 28: 133 bits
  Distance between Person 23 and Person 29: 103 bits
  Distance between Person 23 and Person 30: 123 bits
  Distance between Person 23 and Person 31: 116 bits
  Distance between Person 23 and Person 32: 132 bits
  Distance between Person 23 and Person 33: 137 bits
  Distance between Person 23 and Person 34: 141 bits
  Distance between Person 23 and Person 35: 134 bits
  Distance between Person 23 and Person 36: 134 bits
  Distance between Person 23 and Person 37: 125 bits
  Distance between Person 23 and Person 38: 128 bits
  Distance between Person 23 and Person 39: 132 bits
  Distance between Person 23 and Person 40: 132 bits
  Distance between Person 23 and Person 41: 122 bits
  Distance between Person 23 and Person 42: 120 bits
  Distance between Person 23 and Person 43: 123 bits
  Distance between Person 23 and Person 44: 130 bits
  Distance between Person 23 and Person 45: 122 bits
  Distance between Person 23 and Person 46: 117 bits
  Distance between Person 23 and Person 47: 134 bits
  Distance between Person 23 and Person 48: 123 bits
  Distance between Person 23 and Person 49: 131 bits
  Distance between Person 23 and Person 50: 121 bits
  Distance between Person 23 and Person 51: 133 bits
  Distance between Person 23 and Person 52: 160 bits
  Distance between Person 23 and Person 53: 119 bits
  Distance between Person 23 and Person 54: 138 bits
  Distance between Person 23 and Person 55: 144 bits
  Distance between Person 23 and Person 56: 118 bits
  Distance between Person 23 and Person 57: 141 bits
  Distance between Person 23 and Person 58: 133 bits
  Distance between Person 23 and Person 59: 114 bits
  Distance between Person 23 and Person 60: 117 bits
  Distance between Person 23 and Person 61: 121 bits
  Distance between Person 23 and Person 62: 127 bits
  Distance between Person 23 and Person 63: 115 bits
  Distance between Person 23 and Person 64: 148 bits
  Distance between Person 23 and Person 65: 140 bits
  Distance between Person 23 and Person 66: 144 bits
  Distance between Person 23 and Person 67: 133 bits
  Distance between Person 23 and Person 68: 137 bits
  Distance between Person 23 and Person 69: 145 bits
  Distance between Person 23 and Person 70: 127 bits
  Distance between Person 23 and Person 71: 130 bits
  Distance between Person 23 and Person 72: 147 bits
  Distance between Person 23 and Person 73: 139 bits
  Distance between Person 23 and Person 74: 107 bits
  Distance between Person 23 and Person 75: 148 bits
  Distance between Person 23 and Person 76: 138 bits
  Distance between Person 23 and Person 77: 115 bits
  Distance between Person 23 and Person 78: 141 bits
  Distance between Person 23 and Person 79: 160 bits
  Distance between Person 23 and Person 80: 131 bits
  Distance between Person 23 and Person 81: 125 bits
  Distance between Person 23 and Person 82: 138 bits
  Distance between Person 23 and Person 83: 91 bits
  Distance between Person 23 and Person 84: 135 bits
  Distance between Person 23 and Person 85: 126 bits
  Distance between Person 23 and Person 86: 129 bits
  Distance between Person 23 and Person 87: 133 bits
  Distance between Person 23 and Person 88: 142 bits
  Distance between Person 23 and Person 89: 133 bits
  Distance between Person 24 and Person 25: 134 bits
  Distance between Person 24 and Person 26: 106 bits
  Distance between Person 24 and Person 27: 128 bits
  Distance between Person 24 and Person 28: 111 bits
  Distance between Person 24 and Person 29: 131 bits
  Distance between Person 24 and Person 30: 127 bits
  Distance between Person 24 and Person 31: 136 bits
  Distance between Person 24 and Person 32: 100 bits
  Distance between Person 24 and Person 33: 123 bits
  Distance between Person 24 and Person 34: 117 bits
  Distance between Person 24 and Person 35: 140 bits
  Distance between Person 24 and Person 36: 120 bits
  Distance between Person 24 and Person 37: 129 bits
  Distance between Person 24 and Person 38: 120 bits
  Distance between Person 24 and Person 39: 82 bits
  Distance between Person 24 and Person 40: 98 bits
  Distance between Person 24 and Person 41: 124 bits
  Distance between Person 24 and Person 42: 120 bits
  Distance between Person 24 and Person 43: 105 bits
  Distance between Person 24 and Person 44: 128 bits
  Distance between Person 24 and Person 45: 110 bits
  Distance between Person 24 and Person 46: 141 bits
  Distance between Person 24 and Person 47: 104 bits
  Distance between Person 24 and Person 48: 99 bits
  Distance between Person 24 and Person 49: 129 bits
  Distance between Person 24 and Person 50: 135 bits
  Distance between Person 24 and Person 51: 121 bits
  Distance between Person 24 and Person 52: 102 bits
  Distance between Person 24 and Person 53: 129 bits
  Distance between Person 24 and Person 54: 150 bits
  Distance between Person 24 and Person 55: 108 bits
  Distance between Person 24 and Person 56: 116 bits
  Distance between Person 24 and Person 57: 107 bits
  Distance between Person 24 and Person 58: 135 bits
  Distance between Person 24 and Person 59: 120 bits
  Distance between Person 24 and Person 60: 151 bits
  Distance between Person 24 and Person 61: 127 bits
  Distance between Person 24 and Person 62: 117 bits
  Distance between Person 24 and Person 63: 129 bits
  Distance between Person 24 and Person 64: 114 bits
  Distance between Person 24 and Person 65: 118 bits
  Distance between Person 24 and Person 66: 102 bits
  Distance between Person 24 and Person 67: 117 bits
  Distance between Person 24 and Person 68: 137 bits
  Distance between Person 24 and Person 69: 143 bits
  Distance between Person 24 and Person 70: 123 bits
  Distance between Person 24 and Person 71: 136 bits
  Distance between Person 24 and Person 72: 113 bits
  Distance between Person 24 and Person 73: 127 bits
  Distance between Person 24 and Person 74: 137 bits
  Distance between Person 24 and Person 75: 130 bits
  Distance between Person 24 and Person 76: 130 bits
  Distance between Person 24 and Person 77: 111 bits
  Distance between Person 24 and Person 78: 131 bits
  Distance between Person 24 and Person 79: 112 bits
  Distance between Person 24 and Person 80: 133 bits
  Distance between Person 24 and Person 81: 145 bits
  Distance between Person 24 and Person 82: 124 bits
  Distance between Person 24 and Person 83: 127 bits
  Distance between Person 24 and Person 84: 123 bits
  Distance between Person 24 and Person 85: 132 bits
  Distance between Person 24 and Person 86: 117 bits
  Distance between Person 24 and Person 87: 113 bits
  Distance between Person 24 and Person 88: 146 bits
  Distance between Person 24 and Person 89: 131 bits
  Distance between Person 25 and Person 26: 128 bits
  Distance between Person 25 and Person 27: 120 bits
  Distance between Person 25 and Person 28: 147 bits
  Distance between Person 25 and Person 29: 143 bits
  Distance between Person 25 and Person 30: 143 bits
  Distance between Person 25 and Person 31: 104 bits
  Distance between Person 25 and Person 32: 124 bits
  Distance between Person 25 and Person 33: 87 bits
  Distance between Person 25 and Person 34: 125 bits
  Distance between Person 25 and Person 35: 124 bits
  Distance between Person 25 and Person 36: 140 bits
  Distance between Person 25 and Person 37: 143 bits
  Distance between Person 25 and Person 38: 140 bits
  Distance between Person 25 and Person 39: 138 bits
  Distance between Person 25 and Person 40: 136 bits
  Distance between Person 25 and Person 41: 120 bits
  Distance between Person 25 and Person 42: 104 bits
  Distance between Person 25 and Person 43: 129 bits
  Distance between Person 25 and Person 44: 114 bits
  Distance between Person 25 and Person 45: 108 bits
  Distance between Person 25 and Person 46: 109 bits
  Distance between Person 25 and Person 47: 120 bits
  Distance between Person 25 and Person 48: 145 bits
  Distance between Person 25 and Person 49: 143 bits
  Distance between Person 25 and Person 50: 131 bits
  Distance between Person 25 and Person 51: 139 bits
  Distance between Person 25 and Person 52: 140 bits
  Distance between Person 25 and Person 53: 123 bits
  Distance between Person 25 and Person 54: 118 bits
  Distance between Person 25 and Person 55: 138 bits
  Distance between Person 25 and Person 56: 90 bits
  Distance between Person 25 and Person 57: 131 bits
  Distance between Person 25 and Person 58: 97 bits
  Distance between Person 25 and Person 59: 114 bits
  Distance between Person 25 and Person 60: 133 bits
  Distance between Person 25 and Person 61: 149 bits
  Distance between Person 25 and Person 62: 121 bits
  Distance between Person 25 and Person 63: 127 bits
  Distance between Person 25 and Person 64: 144 bits
  Distance between Person 25 and Person 65: 128 bits
  Distance between Person 25 and Person 66: 140 bits
  Distance between Person 25 and Person 67: 81 bits
  Distance between Person 25 and Person 68: 135 bits
  Distance between Person 25 and Person 69: 133 bits
  Distance between Person 25 and Person 70: 129 bits
  Distance between Person 25 and Person 71: 136 bits
  Distance between Person 25 and Person 72: 141 bits
  Distance between Person 25 and Person 73: 119 bits
  Distance between Person 25 and Person 74: 137 bits
  Distance between Person 25 and Person 75: 132 bits
  Distance between Person 25 and Person 76: 124 bits
  Distance between Person 25 and Person 77: 123 bits
  Distance between Person 25 and Person 78: 121 bits
  Distance between Person 25 and Person 79: 146 bits
  Distance between Person 25 and Person 80: 115 bits
  Distance between Person 25 and Person 81: 121 bits
  Distance between Person 25 and Person 82: 122 bits
  Distance between Person 25 and Person 83: 145 bits
  Distance between Person 25 and Person 84: 101 bits
  Distance between Person 25 and Person 85: 120 bits
  Distance between Person 25 and Person 86: 127 bits
  Distance between Person 25 and Person 87: 147 bits
  Distance between Person 25 and Person 88: 138 bits
  Distance between Person 25 and Person 89: 139 bits
  Distance between Person 26 and Person 27: 118 bits
  Distance between Person 26 and Person 28: 125 bits
  Distance between Person 26 and Person 29: 133 bits
  Distance between Person 26 and Person 30: 117 bits
  Distance between Person 26 and Person 31: 118 bits
  Distance between Person 26 and Person 32: 128 bits
  Distance between Person 26 and Person 33: 115 bits
  Distance between Person 26 and Person 34: 105 bits
  Distance between Person 26 and Person 35: 142 bits
  Distance between Person 26 and Person 36: 98 bits
  Distance between Person 26 and Person 37: 135 bits
  Distance between Person 26 and Person 38: 122 bits
  Distance between Person 26 and Person 39: 128 bits
  Distance between Person 26 and Person 40: 108 bits
  Distance between Person 26 and Person 41: 124 bits
  Distance between Person 26 and Person 42: 132 bits
  Distance between Person 26 and Person 43: 91 bits
  Distance between Person 26 and Person 44: 104 bits
  Distance between Person 26 and Person 45: 128 bits
  Distance between Person 26 and Person 46: 135 bits
  Distance between Person 26 and Person 47: 126 bits
  Distance between Person 26 and Person 48: 129 bits
  Distance between Person 26 and Person 49: 131 bits
  Distance between Person 26 and Person 50: 123 bits
  Distance between Person 26 and Person 51: 103 bits
  Distance between Person 26 and Person 52: 104 bits
  Distance between Person 26 and Person 53: 135 bits
  Distance between Person 26 and Person 54: 110 bits
  Distance between Person 26 and Person 55: 108 bits
  Distance between Person 26 and Person 56: 100 bits
  Distance between Person 26 and Person 57: 135 bits
  Distance between Person 26 and Person 58: 129 bits
  Distance between Person 26 and Person 59: 118 bits
  Distance between Person 26 and Person 60: 137 bits
  Distance between Person 26 and Person 61: 133 bits
  Distance between Person 26 and Person 62: 113 bits
  Distance between Person 26 and Person 63: 145 bits
  Distance between Person 26 and Person 64: 96 bits
  Distance between Person 26 and Person 65: 138 bits
  Distance between Person 26 and Person 66: 142 bits
  Distance between Person 26 and Person 67: 123 bits
  Distance between Person 26 and Person 68: 135 bits
  Distance between Person 26 and Person 69: 111 bits
  Distance between Person 26 and Person 70: 117 bits
  Distance between Person 26 and Person 71: 132 bits
  Distance between Person 26 and Person 72: 133 bits
  Distance between Person 26 and Person 73: 141 bits
  Distance between Person 26 and Person 74: 131 bits
  Distance between Person 26 and Person 75: 128 bits
  Distance between Person 26 and Person 76: 128 bits
  Distance between Person 26 and Person 77: 127 bits
  Distance between Person 26 and Person 78: 125 bits
  Distance between Person 26 and Person 79: 120 bits
  Distance between Person 26 and Person 80: 125 bits
  Distance between Person 26 and Person 81: 141 bits
  Distance between Person 26 and Person 82: 114 bits
  Distance between Person 26 and Person 83: 157 bits
  Distance between Person 26 and Person 84: 127 bits
  Distance between Person 26 and Person 85: 130 bits
  Distance between Person 26 and Person 86: 141 bits
  Distance between Person 26 and Person 87: 131 bits
  Distance between Person 26 and Person 88: 118 bits
  Distance between Person 26 and Person 89: 127 bits
  Distance between Person 27 and Person 28: 129 bits
  Distance between Person 27 and Person 29: 133 bits
  Distance between Person 27 and Person 30: 129 bits
  Distance between Person 27 and Person 31: 148 bits
  Distance between Person 27 and Person 32: 122 bits
  Distance between Person 27 and Person 33: 103 bits
  Distance between Person 27 and Person 34: 115 bits
  Distance between Person 27 and Person 35: 118 bits
  Distance between Person 27 and Person 36: 130 bits
  Distance between Person 27 and Person 37: 129 bits
  Distance between Person 27 and Person 38: 152 bits
  Distance between Person 27 and Person 39: 124 bits
  Distance between Person 27 and Person 40: 106 bits
  Distance between Person 27 and Person 41: 130 bits
  Distance between Person 27 and Person 42: 112 bits
  Distance between Person 27 and Person 43: 111 bits
  Distance between Person 27 and Person 44: 138 bits
  Distance between Person 27 and Person 45: 132 bits
  Distance between Person 27 and Person 46: 125 bits
  Distance between Person 27 and Person 47: 130 bits
  Distance between Person 27 and Person 48: 127 bits
  Distance between Person 27 and Person 49: 141 bits
  Distance between Person 27 and Person 50: 139 bits
  Distance between Person 27 and Person 51: 123 bits
  Distance between Person 27 and Person 52: 126 bits
  Distance between Person 27 and Person 53: 149 bits
  Distance between Person 27 and Person 54: 144 bits
  Distance between Person 27 and Person 55: 132 bits
  Distance between Person 27 and Person 56: 124 bits
  Distance between Person 27 and Person 57: 137 bits
  Distance between Person 27 and Person 58: 139 bits
  Distance between Person 27 and Person 59: 96 bits
  Distance between Person 27 and Person 60: 125 bits
  Distance between Person 27 and Person 61: 137 bits
  Distance between Person 27 and Person 62: 117 bits
  Distance between Person 27 and Person 63: 127 bits
  Distance between Person 27 and Person 64: 130 bits
  Distance between Person 27 and Person 65: 118 bits
  Distance between Person 27 and Person 66: 120 bits
  Distance between Person 27 and Person 67: 101 bits
  Distance between Person 27 and Person 68: 103 bits
  Distance between Person 27 and Person 69: 135 bits
  Distance between Person 27 and Person 70: 129 bits
  Distance between Person 27 and Person 71: 138 bits
  Distance between Person 27 and Person 72: 121 bits
  Distance between Person 27 and Person 73: 135 bits
  Distance between Person 27 and Person 74: 149 bits
  Distance between Person 27 and Person 75: 132 bits
  Distance between Person 27 and Person 76: 136 bits
  Distance between Person 27 and Person 77: 129 bits
  Distance between Person 27 and Person 78: 113 bits
  Distance between Person 27 and Person 79: 142 bits
  Distance between Person 27 and Person 80: 133 bits
  Distance between Person 27 and Person 81: 109 bits
  Distance between Person 27 and Person 82: 126 bits
  Distance between Person 27 and Person 83: 131 bits
  Distance between Person 27 and Person 84: 117 bits
  Distance between Person 27 and Person 85: 130 bits
  Distance between Person 27 and Person 86: 119 bits
  Distance between Person 27 and Person 87: 131 bits
  Distance between Person 27 and Person 88: 134 bits
  Distance between Person 27 and Person 89: 139 bits
  Distance between Person 28 and Person 29: 132 bits
  Distance between Person 28 and Person 30: 158 bits
  Distance between Person 28 and Person 31: 133 bits
  Distance between Person 28 and Person 32: 121 bits
  Distance between Person 28 and Person 33: 138 bits
  Distance between Person 28 and Person 34: 132 bits
  Distance between Person 28 and Person 35: 139 bits
  Distance between Person 28 and Person 36: 137 bits
  Distance between Person 28 and Person 37: 110 bits
  Distance between Person 28 and Person 38: 121 bits
  Distance between Person 28 and Person 39: 103 bits
  Distance between Person 28 and Person 40: 119 bits
  Distance between Person 28 and Person 41: 129 bits
  Distance between Person 28 and Person 42: 135 bits
  Distance between Person 28 and Person 43: 114 bits
  Distance between Person 28 and Person 44: 139 bits
  Distance between Person 28 and Person 45: 119 bits
  Distance between Person 28 and Person 46: 114 bits
  Distance between Person 28 and Person 47: 125 bits
  Distance between Person 28 and Person 48: 132 bits
  Distance between Person 28 and Person 49: 132 bits
  Distance between Person 28 and Person 50: 150 bits
  Distance between Person 28 and Person 51: 122 bits
  Distance between Person 28 and Person 52: 125 bits
  Distance between Person 28 and Person 53: 124 bits
  Distance between Person 28 and Person 54: 111 bits
  Distance between Person 28 and Person 55: 125 bits
  Distance between Person 28 and Person 56: 129 bits
  Distance between Person 28 and Person 57: 144 bits
  Distance between Person 28 and Person 58: 136 bits
  Distance between Person 28 and Person 59: 123 bits
  Distance between Person 28 and Person 60: 138 bits
  Distance between Person 28 and Person 61: 74 bits
  Distance between Person 28 and Person 62: 130 bits
  Distance between Person 28 and Person 63: 112 bits
  Distance between Person 28 and Person 64: 123 bits
  Distance between Person 28 and Person 65: 141 bits
  Distance between Person 28 and Person 66: 137 bits
  Distance between Person 28 and Person 67: 136 bits
  Distance between Person 28 and Person 68: 120 bits
  Distance between Person 28 and Person 69: 136 bits
  Distance between Person 28 and Person 70: 108 bits
  Distance between Person 28 and Person 71: 123 bits
  Distance between Person 28 and Person 72: 124 bits
  Distance between Person 28 and Person 73: 128 bits
  Distance between Person 28 and Person 74: 122 bits
  Distance between Person 28 and Person 75: 127 bits
  Distance between Person 28 and Person 76: 119 bits
  Distance between Person 28 and Person 77: 116 bits
  Distance between Person 28 and Person 78: 126 bits
  Distance between Person 28 and Person 79: 151 bits
  Distance between Person 28 and Person 80: 126 bits
  Distance between Person 28 and Person 81: 144 bits
  Distance between Person 28 and Person 82: 137 bits
  Distance between Person 28 and Person 83: 136 bits
  Distance between Person 28 and Person 84: 126 bits
  Distance between Person 28 and Person 85: 115 bits
  Distance between Person 28 and Person 86: 92 bits
  Distance between Person 28 and Person 87: 128 bits
  Distance between Person 28 and Person 88: 113 bits
  Distance between Person 28 and Person 89: 118 bits
  Distance between Person 29 and Person 30: 128 bits
  Distance between Person 29 and Person 31: 137 bits
  Distance between Person 29 and Person 32: 125 bits
  Distance between Person 29 and Person 33: 128 bits
  Distance between Person 29 and Person 34: 124 bits
  Distance between Person 29 and Person 35: 133 bits
  Distance between Person 29 and Person 36: 143 bits
  Distance between Person 29 and Person 37: 84 bits
  Distance between Person 29 and Person 38: 147 bits
  Distance between Person 29 and Person 39: 117 bits
  Distance between Person 29 and Person 40: 105 bits
  Distance between Person 29 and Person 41: 115 bits
  Distance between Person 29 and Person 42: 129 bits
  Distance between Person 29 and Person 43: 122 bits
  Distance between Person 29 and Person 44: 121 bits
  Distance between Person 29 and Person 45: 131 bits
  Distance between Person 29 and Person 46: 120 bits
  Distance between Person 29 and Person 47: 113 bits
  Distance between Person 29 and Person 48: 132 bits
  Distance between Person 29 and Person 49: 144 bits
  Distance between Person 29 and Person 50: 96 bits
  Distance between Person 29 and Person 51: 114 bits
  Distance between Person 29 and Person 52: 129 bits
  Distance between Person 29 and Person 53: 126 bits
  Distance between Person 29 and Person 54: 127 bits
  Distance between Person 29 and Person 55: 121 bits
  Distance between Person 29 and Person 56: 119 bits
  Distance between Person 29 and Person 57: 146 bits
  Distance between Person 29 and Person 58: 118 bits
  Distance between Person 29 and Person 59: 127 bits
  Distance between Person 29 and Person 60: 124 bits
  Distance between Person 29 and Person 61: 110 bits
  Distance between Person 29 and Person 62: 144 bits
  Distance between Person 29 and Person 63: 130 bits
  Distance between Person 29 and Person 64: 135 bits
  Distance between Person 29 and Person 65: 131 bits
  Distance between Person 29 and Person 66: 123 bits
  Distance between Person 29 and Person 67: 128 bits
  Distance between Person 29 and Person 68: 130 bits
  Distance between Person 29 and Person 69: 124 bits
  Distance between Person 29 and Person 70: 142 bits
  Distance between Person 29 and Person 71: 115 bits
  Distance between Person 29 and Person 72: 144 bits
  Distance between Person 29 and Person 73: 124 bits
  Distance between Person 29 and Person 74: 132 bits
  Distance between Person 29 and Person 75: 111 bits
  Distance between Person 29 and Person 76: 129 bits
  Distance between Person 29 and Person 77: 106 bits
  Distance between Person 29 and Person 78: 136 bits
  Distance between Person 29 and Person 79: 117 bits
  Distance between Person 29 and Person 80: 122 bits
  Distance between Person 29 and Person 81: 142 bits
  Distance between Person 29 and Person 82: 105 bits
  Distance between Person 29 and Person 83: 128 bits
  Distance between Person 29 and Person 84: 124 bits
  Distance between Person 29 and Person 85: 127 bits
  Distance between Person 29 and Person 86: 140 bits
  Distance between Person 29 and Person 87: 134 bits
  Distance between Person 29 and Person 88: 125 bits
  Distance between Person 29 and Person 89: 116 bits
  Distance between Person 30 and Person 31: 121 bits
  Distance between Person 30 and Person 32: 137 bits
  Distance between Person 30 and Person 33: 136 bits
  Distance between Person 30 and Person 34: 134 bits
  Distance between Person 30 and Person 35: 123 bits
  Distance between Person 30 and Person 36: 113 bits
  Distance between Person 30 and Person 37: 132 bits
  Distance between Person 30 and Person 38: 133 bits
  Distance between Person 30 and Person 39: 143 bits
  Distance between Person 30 and Person 40: 139 bits
  Distance between Person 30 and Person 41: 151 bits
  Distance between Person 30 and Person 42: 135 bits
  Distance between Person 30 and Person 43: 114 bits
  Distance between Person 30 and Person 44: 137 bits
  Distance between Person 30 and Person 45: 145 bits
  Distance between Person 30 and Person 46: 130 bits
  Distance between Person 30 and Person 47: 117 bits
  Distance between Person 30 and Person 48: 120 bits
  Distance between Person 30 and Person 49: 110 bits
  Distance between Person 30 and Person 50: 104 bits
  Distance between Person 30 and Person 51: 118 bits
  Distance between Person 30 and Person 52: 129 bits
  Distance between Person 30 and Person 53: 142 bits
  Distance between Person 30 and Person 54: 155 bits
  Distance between Person 30 and Person 55: 133 bits
  Distance between Person 30 and Person 56: 123 bits
  Distance between Person 30 and Person 57: 134 bits
  Distance between Person 30 and Person 58: 124 bits
  Distance between Person 30 and Person 59: 131 bits
  Distance between Person 30 and Person 60: 148 bits
  Distance between Person 30 and Person 61: 130 bits
  Distance between Person 30 and Person 62: 126 bits
  Distance between Person 30 and Person 63: 146 bits
  Distance between Person 30 and Person 64: 139 bits
  Distance between Person 30 and Person 65: 133 bits
  Distance between Person 30 and Person 66: 139 bits
  Distance between Person 30 and Person 67: 138 bits
  Distance between Person 30 and Person 68: 122 bits
  Distance between Person 30 and Person 69: 104 bits
  Distance between Person 30 and Person 70: 138 bits
  Distance between Person 30 and Person 71: 111 bits
  Distance between Person 30 and Person 72: 124 bits
  Distance between Person 30 and Person 73: 126 bits
  Distance between Person 30 and Person 74: 128 bits
  Distance between Person 30 and Person 75: 135 bits
  Distance between Person 30 and Person 76: 133 bits
  Distance between Person 30 and Person 77: 134 bits
  Distance between Person 30 and Person 78: 144 bits
  Distance between Person 30 and Person 79: 117 bits
  Distance between Person 30 and Person 80: 142 bits
  Distance between Person 30 and Person 81: 118 bits
  Distance between Person 30 and Person 82: 131 bits
  Distance between Person 30 and Person 83: 116 bits
  Distance between Person 30 and Person 84: 130 bits
  Distance between Person 30 and Person 85: 123 bits
  Distance between Person 30 and Person 86: 146 bits
  Distance between Person 30 and Person 87: 94 bits
  Distance between Person 30 and Person 88: 147 bits
  Distance between Person 30 and Person 89: 122 bits
  Distance between Person 31 and Person 32: 146 bits
  Distance between Person 31 and Person 33: 113 bits
  Distance between Person 31 and Person 34: 125 bits
  Distance between Person 31 and Person 35: 142 bits
  Distance between Person 31 and Person 36: 128 bits
  Distance between Person 31 and Person 37: 131 bits
  Distance between Person 31 and Person 38: 112 bits
  Distance between Person 31 and Person 39: 154 bits
  Distance between Person 31 and Person 40: 132 bits
  Distance between Person 31 and Person 41: 138 bits
  Distance between Person 31 and Person 42: 146 bits
  Distance between Person 31 and Person 43: 137 bits
  Distance between Person 31 and Person 44: 108 bits
  Distance between Person 31 and Person 45: 116 bits
  Distance between Person 31 and Person 46: 117 bits
  Distance between Person 31 and Person 47: 136 bits
  Distance between Person 31 and Person 48: 127 bits
  Distance between Person 31 and Person 49: 115 bits
  Distance between Person 31 and Person 50: 127 bits
  Distance between Person 31 and Person 51: 133 bits
  Distance between Person 31 and Person 52: 146 bits
  Distance between Person 31 and Person 53: 117 bits
  Distance between Person 31 and Person 54: 120 bits
  Distance between Person 31 and Person 55: 128 bits
  Distance between Person 31 and Person 56: 132 bits
  Distance between Person 31 and Person 57: 137 bits
  Distance between Person 31 and Person 58: 117 bits
  Distance between Person 31 and Person 59: 156 bits
  Distance between Person 31 and Person 60: 139 bits
  Distance between Person 31 and Person 61: 123 bits
  Distance between Person 31 and Person 62: 121 bits
  Distance between Person 31 and Person 63: 143 bits
  Distance between Person 31 and Person 64: 136 bits
  Distance between Person 31 and Person 65: 116 bits
  Distance between Person 31 and Person 66: 154 bits
  Distance between Person 31 and Person 67: 127 bits
  Distance between Person 31 and Person 68: 145 bits
  Distance between Person 31 and Person 69: 107 bits
  Distance between Person 31 and Person 70: 117 bits
  Distance between Person 31 and Person 71: 128 bits
  Distance between Person 31 and Person 72: 115 bits
  Distance between Person 31 and Person 73: 115 bits
  Distance between Person 31 and Person 74: 121 bits
  Distance between Person 31 and Person 75: 134 bits
  Distance between Person 31 and Person 76: 120 bits
  Distance between Person 31 and Person 77: 133 bits
  Distance between Person 31 and Person 78: 119 bits
  Distance between Person 31 and Person 79: 140 bits
  Distance between Person 31 and Person 80: 131 bits
  Distance between Person 31 and Person 81: 137 bits
  Distance between Person 31 and Person 82: 124 bits
  Distance between Person 31 and Person 83: 143 bits
  Distance between Person 31 and Person 84: 133 bits
  Distance between Person 31 and Person 85: 132 bits
  Distance between Person 31 and Person 86: 131 bits
  Distance between Person 31 and Person 87: 105 bits
  Distance between Person 31 and Person 88: 108 bits
  Distance between Person 31 and Person 89: 107 bits
  Distance between Person 32 and Person 33: 139 bits
  Distance between Person 32 and Person 34: 123 bits
  Distance between Person 32 and Person 35: 130 bits
  Distance between Person 32 and Person 36: 128 bits
  Distance between Person 32 and Person 37: 115 bits
  Distance between Person 32 and Person 38: 128 bits
  Distance between Person 32 and Person 39: 48 bits
  Distance between Person 32 and Person 40: 108 bits
  Distance between Person 32 and Person 41: 148 bits
  Distance between Person 32 and Person 42: 134 bits
  Distance between Person 32 and Person 43: 107 bits
  Distance between Person 32 and Person 44: 138 bits
  Distance between Person 32 and Person 45: 114 bits
  Distance between Person 32 and Person 46: 137 bits
  Distance between Person 32 and Person 47: 112 bits
  Distance between Person 32 and Person 48: 123 bits
  Distance between Person 32 and Person 49: 137 bits
  Distance between Person 32 and Person 50: 113 bits
  Distance between Person 32 and Person 51: 131 bits
  Distance between Person 32 and Person 52: 134 bits
  Distance between Person 32 and Person 53: 137 bits
  Distance between Person 32 and Person 54: 130 bits
  Distance between Person 32 and Person 55: 148 bits
  Distance between Person 32 and Person 56: 112 bits
  Distance between Person 32 and Person 57: 133 bits
  Distance between Person 32 and Person 58: 123 bits
  Distance between Person 32 and Person 59: 120 bits
  Distance between Person 32 and Person 60: 143 bits
  Distance between Person 32 and Person 61: 107 bits
  Distance between Person 32 and Person 62: 99 bits
  Distance between Person 32 and Person 63: 143 bits
  Distance between Person 32 and Person 64: 98 bits
  Distance between Person 32 and Person 65: 138 bits
  Distance between Person 32 and Person 66: 116 bits
  Distance between Person 32 and Person 67: 143 bits
  Distance between Person 32 and Person 68: 125 bits
  Distance between Person 32 and Person 69: 141 bits
  Distance between Person 32 and Person 70: 123 bits
  Distance between Person 32 and Person 71: 118 bits
  Distance between Person 32 and Person 72: 105 bits
  Distance between Person 32 and Person 73: 109 bits
  Distance between Person 32 and Person 74: 119 bits
  Distance between Person 32 and Person 75: 136 bits
  Distance between Person 32 and Person 76: 124 bits
  Distance between Person 32 and Person 77: 137 bits
  Distance between Person 32 and Person 78: 123 bits
  Distance between Person 32 and Person 79: 140 bits
  Distance between Person 32 and Person 80: 117 bits
  Distance between Person 32 and Person 81: 143 bits
  Distance between Person 32 and Person 82: 120 bits
  Distance between Person 32 and Person 83: 129 bits
  Distance between Person 32 and Person 84: 113 bits
  Distance between Person 32 and Person 85: 142 bits
  Distance between Person 32 and Person 86: 93 bits
  Distance between Person 32 and Person 87: 119 bits
  Distance between Person 32 and Person 88: 142 bits
  Distance between Person 32 and Person 89: 149 bits
  Distance between Person 33 and Person 34: 110 bits
  Distance between Person 33 and Person 35: 135 bits
  Distance between Person 33 and Person 36: 149 bits
  Distance between Person 33 and Person 37: 128 bits
  Distance between Person 33 and Person 38: 147 bits
  Distance between Person 33 and Person 39: 133 bits
  Distance between Person 33 and Person 40: 139 bits
  Distance between Person 33 and Person 41: 131 bits
  Distance between Person 33 and Person 42: 117 bits
  Distance between Person 33 and Person 43: 124 bits
  Distance between Person 33 and Person 44: 91 bits
  Distance between Person 33 and Person 45: 131 bits
  Distance between Person 33 and Person 46: 130 bits
  Distance between Person 33 and Person 47: 125 bits
  Distance between Person 33 and Person 48: 144 bits
  Distance between Person 33 and Person 49: 130 bits
  Distance between Person 33 and Person 50: 130 bits
  Distance between Person 33 and Person 51: 130 bits
  Distance between Person 33 and Person 52: 135 bits
  Distance between Person 33 and Person 53: 110 bits
  Distance between Person 33 and Person 54: 123 bits
  Distance between Person 33 and Person 55: 125 bits
  Distance between Person 33 and Person 56: 129 bits
  Distance between Person 33 and Person 57: 124 bits
  Distance between Person 33 and Person 58: 128 bits
  Distance between Person 33 and Person 59: 119 bits
  Distance between Person 33 and Person 60: 128 bits
  Distance between Person 33 and Person 61: 160 bits
  Distance between Person 33 and Person 62: 140 bits
  Distance between Person 33 and Person 63: 110 bits
  Distance between Person 33 and Person 64: 129 bits
  Distance between Person 33 and Person 65: 125 bits
  Distance between Person 33 and Person 66: 111 bits
  Distance between Person 33 and Person 67: 36 bits
  Distance between Person 33 and Person 68: 154 bits
  Distance between Person 33 and Person 69: 108 bits
  Distance between Person 33 and Person 70: 128 bits
  Distance between Person 33 and Person 71: 139 bits
  Distance between Person 33 and Person 72: 126 bits
  Distance between Person 33 and Person 73: 120 bits
  Distance between Person 33 and Person 74: 132 bits
  Distance between Person 33 and Person 75: 137 bits
  Distance between Person 33 and Person 76: 125 bits
  Distance between Person 33 and Person 77: 132 bits
  Distance between Person 33 and Person 78: 122 bits
  Distance between Person 33 and Person 79: 139 bits
  Distance between Person 33 and Person 80: 130 bits
  Distance between Person 33 and Person 81: 114 bits
  Distance between Person 33 and Person 82: 121 bits
  Distance between Person 33 and Person 83: 140 bits
  Distance between Person 33 and Person 84: 142 bits
  Distance between Person 33 and Person 85: 127 bits
  Distance between Person 33 and Person 86: 128 bits
  Distance between Person 33 and Person 87: 132 bits
  Distance between Person 33 and Person 88: 115 bits
  Distance between Person 33 and Person 89: 134 bits
  Distance between Person 34 and Person 35: 141 bits
  Distance between Person 34 and Person 36: 145 bits
  Distance between Person 34 and Person 37: 104 bits
  Distance between Person 34 and Person 38: 125 bits
  Distance between Person 34 and Person 39: 117 bits
  Distance between Person 34 and Person 40: 79 bits
  Distance between Person 34 and Person 41: 117 bits
  Distance between Person 34 and Person 42: 119 bits
  Distance between Person 34 and Person 43: 112 bits
  Distance between Person 34 and Person 44: 129 bits
  Distance between Person 34 and Person 45: 119 bits
  Distance between Person 34 and Person 46: 108 bits
  Distance between Person 34 and Person 47: 131 bits
  Distance between Person 34 and Person 48: 116 bits
  Distance between Person 34 and Person 49: 128 bits
  Distance between Person 34 and Person 50: 124 bits
  Distance between Person 34 and Person 51: 110 bits
  Distance between Person 34 and Person 52: 105 bits
  Distance between Person 34 and Person 53: 110 bits
  Distance between Person 34 and Person 54: 127 bits
  Distance between Person 34 and Person 55: 119 bits
  Distance between Person 34 and Person 56: 133 bits
  Distance between Person 34 and Person 57: 124 bits
  Distance between Person 34 and Person 58: 128 bits
  Distance between Person 34 and Person 59: 125 bits
  Distance between Person 34 and Person 60: 132 bits
  Distance between Person 34 and Person 61: 128 bits
  Distance between Person 34 and Person 62: 120 bits
  Distance between Person 34 and Person 63: 130 bits
  Distance between Person 34 and Person 64: 129 bits
  Distance between Person 34 and Person 65: 137 bits
  Distance between Person 34 and Person 66: 145 bits
  Distance between Person 34 and Person 67: 114 bits
  Distance between Person 34 and Person 68: 138 bits
  Distance between Person 34 and Person 69: 134 bits
  Distance between Person 34 and Person 70: 118 bits
  Distance between Person 34 and Person 71: 135 bits
  Distance between Person 34 and Person 72: 136 bits
  Distance between Person 34 and Person 73: 112 bits
  Distance between Person 34 and Person 74: 104 bits
  Distance between Person 34 and Person 75: 145 bits
  Distance between Person 34 and Person 76: 107 bits
  Distance between Person 34 and Person 77: 136 bits
  Distance between Person 34 and Person 78: 126 bits
  Distance between Person 34 and Person 79: 131 bits
  Distance between Person 34 and Person 80: 122 bits
  Distance between Person 34 and Person 81: 132 bits
  Distance between Person 34 and Person 82: 125 bits
  Distance between Person 34 and Person 83: 150 bits
  Distance between Person 34 and Person 84: 120 bits
  Distance between Person 34 and Person 85: 131 bits
  Distance between Person 34 and Person 86: 136 bits
  Distance between Person 34 and Person 87: 152 bits
  Distance between Person 34 and Person 88: 129 bits
  Distance between Person 34 and Person 89: 124 bits
  Distance between Person 35 and Person 36: 118 bits
  Distance between Person 35 and Person 37: 117 bits
  Distance between Person 35 and Person 38: 108 bits
  Distance between Person 35 and Person 39: 112 bits
  Distance between Person 35 and Person 40: 130 bits
  Distance between Person 35 and Person 41: 140 bits
  Distance between Person 35 and Person 42: 128 bits
  Distance between Person 35 and Person 43: 131 bits
  Distance between Person 35 and Person 44: 132 bits
  Distance between Person 35 and Person 45: 132 bits
  Distance between Person 35 and Person 46: 145 bits
  Distance between Person 35 and Person 47: 124 bits
  Distance between Person 35 and Person 48: 121 bits
  Distance between Person 35 and Person 49: 115 bits
  Distance between Person 35 and Person 50: 111 bits
  Distance between Person 35 and Person 51: 143 bits
  Distance between Person 35 and Person 52: 122 bits
  Distance between Person 35 and Person 53: 119 bits
  Distance between Person 35 and Person 54: 120 bits
  Distance between Person 35 and Person 55: 124 bits
  Distance between Person 35 and Person 56: 126 bits
  Distance between Person 35 and Person 57: 119 bits
  Distance between Person 35 and Person 58: 145 bits
  Distance between Person 35 and Person 59: 122 bits
  Distance between Person 35 and Person 60: 113 bits
  Distance between Person 35 and Person 61: 129 bits
  Distance between Person 35 and Person 62: 145 bits
  Distance between Person 35 and Person 63: 143 bits
  Distance between Person 35 and Person 64: 122 bits
  Distance between Person 35 and Person 65: 130 bits
  Distance between Person 35 and Person 66: 126 bits
  Distance between Person 35 and Person 67: 125 bits
  Distance between Person 35 and Person 68: 123 bits
  Distance between Person 35 and Person 69: 125 bits
  Distance between Person 35 and Person 70: 127 bits
  Distance between Person 35 and Person 71: 122 bits
  Distance between Person 35 and Person 72: 121 bits
  Distance between Person 35 and Person 73: 139 bits
  Distance between Person 35 and Person 74: 131 bits
  Distance between Person 35 and Person 75: 134 bits
  Distance between Person 35 and Person 76: 122 bits
  Distance between Person 35 and Person 77: 107 bits
  Distance between Person 35 and Person 78: 141 bits
  Distance between Person 35 and Person 79: 120 bits
  Distance between Person 35 and Person 80: 127 bits
  Distance between Person 35 and Person 81: 99 bits
  Distance between Person 35 and Person 82: 140 bits
  Distance between Person 35 and Person 83: 135 bits
  Distance between Person 35 and Person 84: 135 bits
  Distance between Person 35 and Person 85: 114 bits
  Distance between Person 35 and Person 86: 121 bits
  Distance between Person 35 and Person 87: 113 bits
  Distance between Person 35 and Person 88: 118 bits
  Distance between Person 35 and Person 89: 143 bits
  Distance between Person 36 and Person 37: 129 bits
  Distance between Person 36 and Person 38: 140 bits
  Distance between Person 36 and Person 39: 128 bits
  Distance between Person 36 and Person 40: 128 bits
  Distance between Person 36 and Person 41: 142 bits
  Distance between Person 36 and Person 42: 130 bits
  Distance between Person 36 and Person 43: 117 bits
  Distance between Person 36 and Person 44: 134 bits
  Distance between Person 36 and Person 45: 144 bits
  Distance between Person 36 and Person 46: 117 bits
  Distance between Person 36 and Person 47: 144 bits
  Distance between Person 36 and Person 48: 125 bits
  Distance between Person 36 and Person 49: 117 bits
  Distance between Person 36 and Person 50: 115 bits
  Distance between Person 36 and Person 51: 123 bits
  Distance between Person 36 and Person 52: 124 bits
  Distance between Person 36 and Person 53: 137 bits
  Distance between Person 36 and Person 54: 128 bits
  Distance between Person 36 and Person 55: 132 bits
  Distance between Person 36 and Person 56: 116 bits
  Distance between Person 36 and Person 57: 125 bits
  Distance between Person 36 and Person 58: 133 bits
  Distance between Person 36 and Person 59: 124 bits
  Distance between Person 36 and Person 60: 125 bits
  Distance between Person 36 and Person 61: 143 bits
  Distance between Person 36 and Person 62: 131 bits
  Distance between Person 36 and Person 63: 135 bits
  Distance between Person 36 and Person 64: 124 bits
  Distance between Person 36 and Person 65: 104 bits
  Distance between Person 36 and Person 66: 148 bits
  Distance between Person 36 and Person 67: 147 bits
  Distance between Person 36 and Person 68: 121 bits
  Distance between Person 36 and Person 69: 149 bits
  Distance between Person 36 and Person 70: 135 bits
  Distance between Person 36 and Person 71: 124 bits
  Distance between Person 36 and Person 72: 125 bits
  Distance between Person 36 and Person 73: 133 bits
  Distance between Person 36 and Person 74: 147 bits
  Distance between Person 36 and Person 75: 122 bits
  Distance between Person 36 and Person 76: 118 bits
  Distance between Person 36 and Person 77: 123 bits
  Distance between Person 36 and Person 78: 135 bits
  Distance between Person 36 and Person 79: 114 bits
  Distance between Person 36 and Person 80: 141 bits
  Distance between Person 36 and Person 81: 135 bits
  Distance between Person 36 and Person 82: 128 bits
  Distance between Person 36 and Person 83: 149 bits
  Distance between Person 36 and Person 84: 117 bits
  Distance between Person 36 and Person 85: 130 bits
  Distance between Person 36 and Person 86: 147 bits
  Distance between Person 36 and Person 87: 109 bits
  Distance between Person 36 and Person 88: 116 bits
  Distance between Person 36 and Person 89: 147 bits
  Distance between Person 37 and Person 38: 137 bits
  Distance between Person 37 and Person 39: 91 bits
  Distance between Person 37 and Person 40: 125 bits
  Distance between Person 37 and Person 41: 131 bits
  Distance between Person 37 and Person 42: 133 bits
  Distance between Person 37 and Person 43: 142 bits
  Distance between Person 37 and Person 44: 117 bits
  Distance between Person 37 and Person 45: 111 bits
  Distance between Person 37 and Person 46: 112 bits
  Distance between Person 37 and Person 47: 135 bits
  Distance between Person 37 and Person 48: 126 bits
  Distance between Person 37 and Person 49: 118 bits
  Distance between Person 37 and Person 50: 122 bits
  Distance between Person 37 and Person 51: 108 bits
  Distance between Person 37 and Person 52: 109 bits
  Distance between Person 37 and Person 53: 128 bits
  Distance between Person 37 and Person 54: 131 bits
  Distance between Person 37 and Person 55: 125 bits
  Distance between Person 37 and Person 56: 133 bits
  Distance between Person 37 and Person 57: 164 bits
  Distance between Person 37 and Person 58: 132 bits
  Distance between Person 37 and Person 59: 145 bits
  Distance between Person 37 and Person 60: 134 bits
  Distance between Person 37 and Person 61: 110 bits
  Distance between Person 37 and Person 62: 136 bits
  Distance between Person 37 and Person 63: 118 bits
  Distance between Person 37 and Person 64: 117 bits
  Distance between Person 37 and Person 65: 123 bits
  Distance between Person 37 and Person 66: 133 bits
  Distance between Person 37 and Person 67: 138 bits
  Distance between Person 37 and Person 68: 126 bits
  Distance between Person 37 and Person 69: 126 bits
  Distance between Person 37 and Person 70: 132 bits
  Distance between Person 37 and Person 71: 115 bits
  Distance between Person 37 and Person 72: 122 bits
  Distance between Person 37 and Person 73: 122 bits
  Distance between Person 37 and Person 74: 114 bits
  Distance between Person 37 and Person 75: 133 bits
  Distance between Person 37 and Person 76: 121 bits
  Distance between Person 37 and Person 77: 98 bits
  Distance between Person 37 and Person 78: 132 bits
  Distance between Person 37 and Person 79: 129 bits
  Distance between Person 37 and Person 80: 98 bits
  Distance between Person 37 and Person 81: 144 bits
  Distance between Person 37 and Person 82: 111 bits
  Distance between Person 37 and Person 83: 134 bits
  Distance between Person 37 and Person 84: 132 bits
  Distance between Person 37 and Person 85: 149 bits
  Distance between Person 37 and Person 86: 116 bits
  Distance between Person 37 and Person 87: 132 bits
  Distance between Person 37 and Person 88: 137 bits
  Distance between Person 37 and Person 89: 140 bits
  Distance between Person 38 and Person 39: 132 bits
  Distance between Person 38 and Person 40: 122 bits
  Distance between Person 38 and Person 41: 136 bits
  Distance between Person 38 and Person 42: 130 bits
  Distance between Person 38 and Person 43: 125 bits
  Distance between Person 38 and Person 44: 130 bits
  Distance between Person 38 and Person 45: 122 bits
  Distance between Person 38 and Person 46: 145 bits
  Distance between Person 38 and Person 47: 138 bits
  Distance between Person 38 and Person 48: 117 bits
  Distance between Person 38 and Person 49: 113 bits
  Distance between Person 38 and Person 50: 131 bits
  Distance between Person 38 and Person 51: 145 bits
  Distance between Person 38 and Person 52: 130 bits
  Distance between Person 38 and Person 53: 117 bits
  Distance between Person 38 and Person 54: 124 bits
  Distance between Person 38 and Person 55: 116 bits
  Distance between Person 38 and Person 56: 138 bits
  Distance between Person 38 and Person 57: 113 bits
  Distance between Person 38 and Person 58: 157 bits
  Distance between Person 38 and Person 59: 138 bits
  Distance between Person 38 and Person 60: 119 bits
  Distance between Person 38 and Person 61: 117 bits
  Distance between Person 38 and Person 62: 117 bits
  Distance between Person 38 and Person 63: 139 bits
  Distance between Person 38 and Person 64: 124 bits
  Distance between Person 38 and Person 65: 132 bits
  Distance between Person 38 and Person 66: 126 bits
  Distance between Person 38 and Person 67: 147 bits
  Distance between Person 38 and Person 68: 137 bits
  Distance between Person 38 and Person 69: 135 bits
  Distance between Person 38 and Person 70: 73 bits
  Distance between Person 38 and Person 71: 124 bits
  Distance between Person 38 and Person 72: 121 bits
  Distance between Person 38 and Person 73: 133 bits
  Distance between Person 38 and Person 74: 123 bits
  Distance between Person 38 and Person 75: 116 bits
  Distance between Person 38 and Person 76: 126 bits
  Distance between Person 38 and Person 77: 123 bits
  Distance between Person 38 and Person 78: 115 bits
  Distance between Person 38 and Person 79: 136 bits
  Distance between Person 38 and Person 80: 113 bits
  Distance between Person 38 and Person 81: 121 bits
  Distance between Person 38 and Person 82: 138 bits
  Distance between Person 38 and Person 83: 137 bits
  Distance between Person 38 and Person 84: 155 bits
  Distance between Person 38 and Person 85: 126 bits
  Distance between Person 38 and Person 86: 103 bits
  Distance between Person 38 and Person 87: 135 bits
  Distance between Person 38 and Person 88: 118 bits
  Distance between Person 38 and Person 89: 115 bits
  Distance between Person 39 and Person 40: 108 bits
  Distance between Person 39 and Person 41: 142 bits
  Distance between Person 39 and Person 42: 130 bits
  Distance between Person 39 and Person 43: 109 bits
  Distance between Person 39 and Person 44: 140 bits
  Distance between Person 39 and Person 45: 112 bits
  Distance between Person 39 and Person 46: 147 bits
  Distance between Person 39 and Person 47: 102 bits
  Distance between Person 39 and Person 48: 115 bits
  Distance between Person 39 and Person 49: 133 bits
  Distance between Person 39 and Person 50: 119 bits
  Distance between Person 39 and Person 51: 135 bits
  Distance between Person 39 and Person 52: 120 bits
  Distance between Person 39 and Person 53: 127 bits
  Distance between Person 39 and Person 54: 122 bits
  Distance between Person 39 and Person 55: 134 bits
  Distance between Person 39 and Person 56: 116 bits
  Distance between Person 39 and Person 57: 127 bits
  Distance between Person 39 and Person 58: 135 bits
  Distance between Person 39 and Person 59: 122 bits
  Distance between Person 39 and Person 60: 143 bits
  Distance between Person 39 and Person 61: 113 bits
  Distance between Person 39 and Person 62: 115 bits
  Distance between Person 39 and Person 63: 133 bits
  Distance between Person 39 and Person 64: 90 bits
  Distance between Person 39 and Person 65: 142 bits
  Distance between Person 39 and Person 66: 120 bits
  Distance between Person 39 and Person 67: 127 bits
  Distance between Person 39 and Person 68: 133 bits
  Distance between Person 39 and Person 69: 143 bits
  Distance between Person 39 and Person 70: 133 bits
  Distance between Person 39 and Person 71: 124 bits
  Distance between Person 39 and Person 72: 113 bits
  Distance between Person 39 and Person 73: 117 bits
  Distance between Person 39 and Person 74: 113 bits
  Distance between Person 39 and Person 75: 136 bits
  Distance between Person 39 and Person 76: 112 bits
  Distance between Person 39 and Person 77: 119 bits
  Distance between Person 39 and Person 78: 133 bits
  Distance between Person 39 and Person 79: 136 bits
  Distance between Person 39 and Person 80: 119 bits
  Distance between Person 39 and Person 81: 135 bits
  Distance between Person 39 and Person 82: 128 bits
  Distance between Person 39 and Person 83: 141 bits
  Distance between Person 39 and Person 84: 121 bits
  Distance between Person 39 and Person 85: 142 bits
  Distance between Person 39 and Person 86: 97 bits
  Distance between Person 39 and Person 87: 125 bits
  Distance between Person 39 and Person 88: 150 bits
  Distance between Person 39 and Person 89: 147 bits
  Distance between Person 40 and Person 41: 122 bits
  Distance between Person 40 and Person 42: 116 bits
  Distance between Person 40 and Person 43: 97 bits
  Distance between Person 40 and Person 44: 134 bits
  Distance between Person 40 and Person 45: 116 bits
  Distance between Person 40 and Person 46: 133 bits
  Distance between Person 40 and Person 47: 122 bits
  Distance between Person 40 and Person 48: 119 bits
  Distance between Person 40 and Person 49: 141 bits
  Distance between Person 40 and Person 50: 129 bits
  Distance between Person 40 and Person 51: 135 bits
  Distance between Person 40 and Person 52: 108 bits
  Distance between Person 40 and Person 53: 135 bits
  Distance between Person 40 and Person 54: 130 bits
  Distance between Person 40 and Person 55: 98 bits
  Distance between Person 40 and Person 56: 120 bits
  Distance between Person 40 and Person 57: 111 bits
  Distance between Person 40 and Person 58: 125 bits
  Distance between Person 40 and Person 59: 116 bits
  Distance between Person 40 and Person 60: 131 bits
  Distance between Person 40 and Person 61: 117 bits
  Distance between Person 40 and Person 62: 125 bits
  Distance between Person 40 and Person 63: 139 bits
  Distance between Person 40 and Person 64: 120 bits
  Distance between Person 40 and Person 65: 120 bits
  Distance between Person 40 and Person 66: 130 bits
  Distance between Person 40 and Person 67: 145 bits
  Distance between Person 40 and Person 68: 121 bits
  Distance between Person 40 and Person 69: 141 bits
  Distance between Person 40 and Person 70: 133 bits
  Distance between Person 40 and Person 71: 104 bits
  Distance between Person 40 and Person 72: 149 bits
  Distance between Person 40 and Person 73: 123 bits
  Distance between Person 40 and Person 74: 125 bits
  Distance between Person 40 and Person 75: 130 bits
  Distance between Person 40 and Person 76: 126 bits
  Distance between Person 40 and Person 77: 127 bits
  Distance between Person 40 and Person 78: 115 bits
  Distance between Person 40 and Person 79: 136 bits
  Distance between Person 40 and Person 80: 133 bits
  Distance between Person 40 and Person 81: 129 bits
  Distance between Person 40 and Person 82: 114 bits
  Distance between Person 40 and Person 83: 131 bits
  Distance between Person 40 and Person 84: 123 bits
  Distance between Person 40 and Person 85: 136 bits
  Distance between Person 40 and Person 86: 113 bits
  Distance between Person 40 and Person 87: 129 bits
  Distance between Person 40 and Person 88: 116 bits
  Distance between Person 40 and Person 89: 131 bits
  Distance between Person 41 and Person 42: 124 bits
  Distance between Person 41 and Person 43: 139 bits
  Distance between Person 41 and Person 44: 124 bits
  Distance between Person 41 and Person 45: 94 bits
  Distance between Person 41 and Person 46: 145 bits
  Distance between Person 41 and Person 47: 122 bits
  Distance between Person 41 and Person 48: 125 bits
  Distance between Person 41 and Person 49: 121 bits
  Distance between Person 41 and Person 50: 139 bits
  Distance between Person 41 and Person 51: 129 bits
  Distance between Person 41 and Person 52: 106 bits
  Distance between Person 41 and Person 53: 117 bits
  Distance between Person 41 and Person 54: 130 bits
  Distance between Person 41 and Person 55: 96 bits
  Distance between Person 41 and Person 56: 120 bits
  Distance between Person 41 and Person 57: 115 bits
  Distance between Person 41 and Person 58: 127 bits
  Distance between Person 41 and Person 59: 128 bits
  Distance between Person 41 and Person 60: 119 bits
  Distance between Person 41 and Person 61: 133 bits
  Distance between Person 41 and Person 62: 121 bits
  Distance between Person 41 and Person 63: 111 bits
  Distance between Person 41 and Person 64: 132 bits
  Distance between Person 41 and Person 65: 124 bits
  Distance between Person 41 and Person 66: 108 bits
  Distance between Person 41 and Person 67: 123 bits
  Distance between Person 41 and Person 68: 129 bits
  Distance between Person 41 and Person 69: 119 bits
  Distance between Person 41 and Person 70: 135 bits
  Distance between Person 41 and Person 71: 138 bits
  Distance between Person 41 and Person 72: 127 bits
  Distance between Person 41 and Person 73: 137 bits
  Distance between Person 41 and Person 74: 123 bits
  Distance between Person 41 and Person 75: 122 bits
  Distance between Person 41 and Person 76: 128 bits
  Distance between Person 41 and Person 77: 125 bits
  Distance between Person 41 and Person 78: 117 bits
  Distance between Person 41 and Person 79: 110 bits
  Distance between Person 41 and Person 80: 123 bits
  Distance between Person 41 and Person 81: 121 bits
  Distance between Person 41 and Person 82: 118 bits
  Distance between Person 41 and Person 83: 107 bits
  Distance between Person 41 and Person 84: 103 bits
  Distance between Person 41 and Person 85: 120 bits
  Distance between Person 41 and Person 86: 133 bits
  Distance between Person 41 and Person 87: 141 bits
  Distance between Person 41 and Person 88: 134 bits
  Distance between Person 41 and Person 89: 121 bits
  Distance between Person 42 and Person 43: 119 bits
  Distance between Person 42 and Person 44: 136 bits
  Distance between Person 42 and Person 45: 126 bits
  Distance between Person 42 and Person 46: 113 bits
  Distance between Person 42 and Person 47: 104 bits
  Distance between Person 42 and Person 48: 121 bits
  Distance between Person 42 and Person 49: 123 bits
  Distance between Person 42 and Person 50: 145 bits
  Distance between Person 42 and Person 51: 127 bits
  Distance between Person 42 and Person 52: 144 bits
  Distance between Person 42 and Person 53: 141 bits
  Distance between Person 42 and Person 54: 128 bits
  Distance between Person 42 and Person 55: 136 bits
  Distance between Person 42 and Person 56: 106 bits
  Distance between Person 42 and Person 57: 103 bits
  Distance between Person 42 and Person 58: 131 bits
  Distance between Person 42 and Person 59: 64 bits
  Distance between Person 42 and Person 60: 115 bits
  Distance between Person 42 and Person 61: 151 bits
  Distance between Person 42 and Person 62: 129 bits
  Distance between Person 42 and Person 63: 129 bits
  Distance between Person 42 and Person 64: 128 bits
  Distance between Person 42 and Person 65: 122 bits
  Distance between Person 42 and Person 66: 120 bits
  Distance between Person 42 and Person 67: 103 bits
  Distance between Person 42 and Person 68: 135 bits
  Distance between Person 42 and Person 69: 125 bits
  Distance between Person 42 and Person 70: 123 bits
  Distance between Person 42 and Person 71: 140 bits
  Distance between Person 42 and Person 72: 127 bits
  Distance between Person 42 and Person 73: 149 bits
  Distance between Person 42 and Person 74: 103 bits
  Distance between Person 42 and Person 75: 104 bits
  Distance between Person 42 and Person 76: 132 bits
  Distance between Person 42 and Person 77: 109 bits
  Distance between Person 42 and Person 78: 117 bits
  Distance between Person 42 and Person 79: 122 bits
  Distance between Person 42 and Person 80: 135 bits
  Distance between Person 42 and Person 81: 115 bits
  Distance between Person 42 and Person 82: 128 bits
  Distance between Person 42 and Person 83: 153 bits
  Distance between Person 42 and Person 84: 129 bits
  Distance between Person 42 and Person 85: 136 bits
  Distance between Person 42 and Person 86: 145 bits
  Distance between Person 42 and Person 87: 133 bits
  Distance between Person 42 and Person 88: 130 bits
  Distance between Person 42 and Person 89: 151 bits
  Distance between Person 43 and Person 44: 139 bits
  Distance between Person 43 and Person 45: 145 bits
  Distance between Person 43 and Person 46: 134 bits
  Distance between Person 43 and Person 47: 103 bits
  Distance between Person 43 and Person 48: 128 bits
  Distance between Person 43 and Person 49: 128 bits
  Distance between Person 43 and Person 50: 112 bits
  Distance between Person 43 and Person 51: 108 bits
  Distance between Person 43 and Person 52: 133 bits
  Distance between Person 43 and Person 53: 158 bits
  Distance between Person 43 and Person 54: 119 bits
  Distance between Person 43 and Person 55: 141 bits
  Distance between Person 43 and Person 56: 85 bits
  Distance between Person 43 and Person 57: 134 bits
  Distance between Person 43 and Person 58: 138 bits
  Distance between Person 43 and Person 59: 107 bits
  Distance between Person 43 and Person 60: 144 bits
  Distance between Person 43 and Person 61: 140 bits
  Distance between Person 43 and Person 62: 124 bits
  Distance between Person 43 and Person 63: 120 bits
  Distance between Person 43 and Person 64: 131 bits
  Distance between Person 43 and Person 65: 147 bits
  Distance between Person 43 and Person 66: 141 bits
  Distance between Person 43 and Person 67: 120 bits
  Distance between Person 43 and Person 68: 112 bits
  Distance between Person 43 and Person 69: 130 bits
  Distance between Person 43 and Person 70: 128 bits
  Distance between Person 43 and Person 71: 129 bits
  Distance between Person 43 and Person 72: 144 bits
  Distance between Person 43 and Person 73: 132 bits
  Distance between Person 43 and Person 74: 118 bits
  Distance between Person 43 and Person 75: 113 bits
  Distance between Person 43 and Person 76: 129 bits
  Distance between Person 43 and Person 77: 120 bits
  Distance between Person 43 and Person 78: 140 bits
  Distance between Person 43 and Person 79: 147 bits
  Distance between Person 43 and Person 80: 126 bits
  Distance between Person 43 and Person 81: 122 bits
  Distance between Person 43 and Person 82: 131 bits
  Distance between Person 43 and Person 83: 128 bits
  Distance between Person 43 and Person 84: 122 bits
  Distance between Person 43 and Person 85: 121 bits
  Distance between Person 43 and Person 86: 144 bits
  Distance between Person 43 and Person 87: 128 bits
  Distance between Person 43 and Person 88: 125 bits
  Distance between Person 43 and Person 89: 118 bits
  Distance between Person 44 and Person 45: 104 bits
  Distance between Person 44 and Person 46: 111 bits
  Distance between Person 44 and Person 47: 110 bits
  Distance between Person 44 and Person 48: 137 bits
  Distance between Person 44 and Person 49: 131 bits
  Distance between Person 44 and Person 50: 123 bits
  Distance between Person 44 and Person 51: 117 bits
  Distance between Person 44 and Person 52: 114 bits
  Distance between Person 44 and Person 53: 91 bits
  Distance between Person 44 and Person 54: 84 bits
  Distance between Person 44 and Person 55: 108 bits
  Distance between Person 44 and Person 56: 132 bits
  Distance between Person 44 and Person 57: 141 bits
  Distance between Person 44 and Person 58: 103 bits
  Distance between Person 44 and Person 59: 132 bits
  Distance between Person 44 and Person 60: 129 bits
  Distance between Person 44 and Person 61: 147 bits
  Distance between Person 44 and Person 62: 135 bits
  Distance between Person 44 and Person 63: 127 bits
  Distance between Person 44 and Person 64: 88 bits
  Distance between Person 44 and Person 65: 108 bits
  Distance between Person 44 and Person 66: 112 bits
  Distance between Person 44 and Person 67: 123 bits
  Distance between Person 44 and Person 68: 141 bits
  Distance between Person 44 and Person 69: 111 bits
  Distance between Person 44 and Person 70: 113 bits
  Distance between Person 44 and Person 71: 148 bits
  Distance between Person 44 and Person 72: 117 bits
  Distance between Person 44 and Person 73: 113 bits
  Distance between Person 44 and Person 74: 133 bits
  Distance between Person 44 and Person 75: 116 bits
  Distance between Person 44 and Person 76: 118 bits
  Distance between Person 44 and Person 77: 123 bits
  Distance between Person 44 and Person 78: 135 bits
  Distance between Person 44 and Person 79: 138 bits
  Distance between Person 44 and Person 80: 123 bits
  Distance between Person 44 and Person 81: 143 bits
  Distance between Person 44 and Person 82: 98 bits
  Distance between Person 44 and Person 83: 123 bits
  Distance between Person 44 and Person 84: 151 bits
  Distance between Person 44 and Person 85: 136 bits
  Distance between Person 44 and Person 86: 135 bits
  Distance between Person 44 and Person 87: 121 bits
  Distance between Person 44 and Person 88: 124 bits
  Distance between Person 44 and Person 89: 123 bits
  Distance between Person 45 and Person 46: 129 bits
  Distance between Person 45 and Person 47: 100 bits
  Distance between Person 45 and Person 48: 143 bits
  Distance between Person 45 and Person 49: 139 bits
  Distance between Person 45 and Person 50: 135 bits
  Distance between Person 45 and Person 51: 119 bits
  Distance between Person 45 and Person 52: 104 bits
  Distance between Person 45 and Person 53: 97 bits
  Distance between Person 45 and Person 54: 124 bits
  Distance between Person 45 and Person 55: 86 bits
  Distance between Person 45 and Person 56: 104 bits
  Distance between Person 45 and Person 57: 135 bits
  Distance between Person 45 and Person 58: 127 bits
  Distance between Person 45 and Person 59: 130 bits
  Distance between Person 45 and Person 60: 123 bits
  Distance between Person 45 and Person 61: 117 bits
  Distance between Person 45 and Person 62: 113 bits
  Distance between Person 45 and Person 63: 119 bits
  Distance between Person 45 and Person 64: 112 bits
  Distance between Person 45 and Person 65: 128 bits
  Distance between Person 45 and Person 66: 130 bits
  Distance between Person 45 and Person 67: 137 bits
  Distance between Person 45 and Person 68: 129 bits
  Distance between Person 45 and Person 69: 129 bits
  Distance between Person 45 and Person 70: 127 bits
  Distance between Person 45 and Person 71: 126 bits
  Distance between Person 45 and Person 72: 121 bits
  Distance between Person 45 and Person 73: 125 bits
  Distance between Person 45 and Person 74: 123 bits
  Distance between Person 45 and Person 75: 120 bits
  Distance between Person 45 and Person 76: 130 bits
  Distance between Person 45 and Person 77: 127 bits
  Distance between Person 45 and Person 78: 117 bits
  Distance between Person 45 and Person 79: 134 bits
  Distance between Person 45 and Person 80: 97 bits
  Distance between Person 45 and Person 81: 145 bits
  Distance between Person 45 and Person 82: 100 bits
  Distance between Person 45 and Person 83: 117 bits
  Distance between Person 45 and Person 84: 127 bits
  Distance between Person 45 and Person 85: 142 bits
  Distance between Person 45 and Person 86: 95 bits
  Distance between Person 45 and Person 87: 127 bits
  Distance between Person 45 and Person 88: 136 bits
  Distance between Person 45 and Person 89: 129 bits
  Distance between Person 46 and Person 47: 127 bits
  Distance between Person 46 and Person 48: 126 bits
  Distance between Person 46 and Person 49: 126 bits
  Distance between Person 46 and Person 50: 126 bits
  Distance between Person 46 and Person 51: 122 bits
  Distance between Person 46 and Person 52: 155 bits
  Distance between Person 46 and Person 53: 118 bits
  Distance between Person 46 and Person 54: 119 bits
  Distance between Person 46 and Person 55: 165 bits
  Distance between Person 46 and Person 56: 131 bits
  Distance between Person 46 and Person 57: 130 bits
  Distance between Person 46 and Person 58: 84 bits
  Distance between Person 46 and Person 59: 117 bits
  Distance between Person 46 and Person 60: 120 bits
  Distance between Person 46 and Person 61: 110 bits
  Distance between Person 46 and Person 62: 136 bits
  Distance between Person 46 and Person 63: 112 bits
  Distance between Person 46 and Person 64: 147 bits
  Distance between Person 46 and Person 65: 107 bits
  Distance between Person 46 and Person 66: 159 bits
  Distance between Person 46 and Person 67: 134 bits
  Distance between Person 46 and Person 68: 128 bits
  Distance between Person 46 and Person 69: 140 bits
  Distance between Person 46 and Person 70: 124 bits
  Distance between Person 46 and Person 71: 137 bits
  Distance between Person 46 and Person 72: 120 bits
  Distance between Person 46 and Person 73: 104 bits
  Distance between Person 46 and Person 74: 128 bits
  Distance between Person 46 and Person 75: 133 bits
  Distance between Person 46 and Person 76: 113 bits
  Distance between Person 46 and Person 77: 124 bits
  Distance between Person 46 and Person 78: 120 bits
  Distance between Person 46 and Person 79: 133 bits
  Distance between Person 46 and Person 80: 126 bits
  Distance between Person 46 and Person 81: 140 bits
  Distance between Person 46 and Person 82: 113 bits
  Distance between Person 46 and Person 83: 148 bits
  Distance between Person 46 and Person 84: 102 bits
  Distance between Person 46 and Person 85: 125 bits
  Distance between Person 46 and Person 86: 136 bits
  Distance between Person 46 and Person 87: 134 bits
  Distance between Person 46 and Person 88: 137 bits
  Distance between Person 46 and Person 89: 124 bits
  Distance between Person 47 and Person 48: 119 bits
  Distance between Person 47 and Person 49: 127 bits
  Distance between Person 47 and Person 50: 107 bits
  Distance between Person 47 and Person 51: 131 bits
  Distance between Person 47 and Person 52: 132 bits
  Distance between Person 47 and Person 53: 139 bits
  Distance between Person 47 and Person 54: 114 bits
  Distance between Person 47 and Person 55: 116 bits
  Distance between Person 47 and Person 56: 104 bits
  Distance between Person 47 and Person 57: 123 bits
  Distance between Person 47 and Person 58: 105 bits
  Distance between Person 47 and Person 59: 124 bits
  Distance between Person 47 and Person 60: 139 bits
  Distance between Person 47 and Person 61: 125 bits
  Distance between Person 47 and Person 62: 139 bits
  Distance between Person 47 and Person 63: 139 bits
  Distance between Person 47 and Person 64: 116 bits
  Distance between Person 47 and Person 65: 134 bits
  Distance between Person 47 and Person 66: 106 bits
  Distance between Person 47 and Person 67: 115 bits
  Distance between Person 47 and Person 68: 133 bits
  Distance between Person 47 and Person 69: 115 bits
  Distance between Person 47 and Person 70: 127 bits
  Distance between Person 47 and Person 71: 122 bits
  Distance between Person 47 and Person 72: 111 bits
  Distance between Person 47 and Person 73: 119 bits
  Distance between Person 47 and Person 74: 119 bits
  Distance between Person 47 and Person 75: 82 bits
  Distance between Person 47 and Person 76: 134 bits
  Distance between Person 47 and Person 77: 115 bits
  Distance between Person 47 and Person 78: 125 bits
  Distance between Person 47 and Person 79: 120 bits
  Distance between Person 47 and Person 80: 129 bits
  Distance between Person 47 and Person 81: 133 bits
  Distance between Person 47 and Person 82: 110 bits
  Distance between Person 47 and Person 83: 115 bits
  Distance between Person 47 and Person 84: 141 bits
  Distance between Person 47 and Person 85: 120 bits
  Distance between Person 47 and Person 86: 121 bits
  Distance between Person 47 and Person 87: 109 bits
  Distance between Person 47 and Person 88: 132 bits
  Distance between Person 47 and Person 89: 131 bits
  Distance between Person 48 and Person 49: 114 bits
  Distance between Person 48 and Person 50: 128 bits
  Distance between Person 48 and Person 51: 130 bits
  Distance between Person 48 and Person 52: 117 bits
  Distance between Person 48 and Person 53: 136 bits
  Distance between Person 48 and Person 54: 129 bits
  Distance between Person 48 and Person 55: 121 bits
  Distance between Person 48 and Person 56: 137 bits
  Distance between Person 48 and Person 57: 126 bits
  Distance between Person 48 and Person 58: 120 bits
  Distance between Person 48 and Person 59: 117 bits
  Distance between Person 48 and Person 60: 122 bits
  Distance between Person 48 and Person 61: 132 bits
  Distance between Person 48 and Person 62: 104 bits
  Distance between Person 48 and Person 63: 130 bits
  Distance between Person 48 and Person 64: 105 bits
  Distance between Person 48 and Person 65: 131 bits
  Distance between Person 48 and Person 66: 109 bits
  Distance between Person 48 and Person 67: 142 bits
  Distance between Person 48 and Person 68: 134 bits
  Distance between Person 48 and Person 69: 136 bits
  Distance between Person 48 and Person 70: 128 bits
  Distance between Person 48 and Person 71: 135 bits
  Distance between Person 48 and Person 72: 116 bits
  Distance between Person 48 and Person 73: 134 bits
  Distance between Person 48 and Person 74: 118 bits
  Distance between Person 48 and Person 75: 139 bits
  Distance between Person 48 and Person 76: 109 bits
  Distance between Person 48 and Person 77: 106 bits
  Distance between Person 48 and Person 78: 118 bits
  Distance between Person 48 and Person 79: 123 bits
  Distance between Person 48 and Person 80: 144 bits
  Distance between Person 48 and Person 81: 106 bits
  Distance between Person 48 and Person 82: 149 bits
  Distance between Person 48 and Person 83: 138 bits
  Distance between Person 48 and Person 84: 142 bits
  Distance between Person 48 and Person 85: 121 bits
  Distance between Person 48 and Person 86: 132 bits
  Distance between Person 48 and Person 87: 120 bits
  Distance between Person 48 and Person 88: 167 bits
  Distance between Person 48 and Person 89: 128 bits
  Distance between Person 49 and Person 50: 132 bits
  Distance between Person 49 and Person 51: 132 bits
  Distance between Person 49 and Person 52: 137 bits
  Distance between Person 49 and Person 53: 144 bits
  Distance between Person 49 and Person 54: 131 bits
  Distance between Person 49 and Person 55: 129 bits
  Distance between Person 49 and Person 56: 133 bits
  Distance between Person 49 and Person 57: 114 bits
  Distance between Person 49 and Person 58: 128 bits
  Distance between Person 49 and Person 59: 117 bits
  Distance between Person 49 and Person 60: 112 bits
  Distance between Person 49 and Person 61: 120 bits
  Distance between Person 49 and Person 62: 126 bits
  Distance between Person 49 and Person 63: 140 bits
  Distance between Person 49 and Person 64: 133 bits
  Distance between Person 49 and Person 65: 127 bits
  Distance between Person 49 and Person 66: 137 bits
  Distance between Person 49 and Person 67: 122 bits
  Distance between Person 49 and Person 68: 138 bits
  Distance between Person 49 and Person 69: 120 bits
  Distance between Person 49 and Person 70: 124 bits
  Distance between Person 49 and Person 71: 105 bits
  Distance between Person 49 and Person 72: 104 bits
  Distance between Person 49 and Person 73: 152 bits
  Distance between Person 49 and Person 74: 124 bits
  Distance between Person 49 and Person 75: 123 bits
  Distance between Person 49 and Person 76: 125 bits
  Distance between Person 49 and Person 77: 128 bits
  Distance between Person 49 and Person 78: 138 bits
  Distance between Person 49 and Person 79: 107 bits
  Distance between Person 49 and Person 80: 106 bits
  Distance between Person 49 and Person 81: 138 bits
  Distance between Person 49 and Person 82: 153 bits
  Distance between Person 49 and Person 83: 124 bits
  Distance between Person 49 and Person 84: 134 bits
  Distance between Person 49 and Person 85: 127 bits
  Distance between Person 49 and Person 86: 120 bits
  Distance between Person 49 and Person 87: 120 bits
  Distance between Person 49 and Person 88: 137 bits
  Distance between Person 49 and Person 89: 118 bits
  Distance between Person 50 and Person 51: 120 bits
  Distance between Person 50 and Person 52: 131 bits
  Distance between Person 50 and Person 53: 126 bits
  Distance between Person 50 and Person 54: 123 bits
  Distance between Person 50 and Person 55: 125 bits
  Distance between Person 50 and Person 56: 117 bits
  Distance between Person 50 and Person 57: 132 bits
  Distance between Person 50 and Person 58: 136 bits
  Distance between Person 50 and Person 59: 127 bits
  Distance between Person 50 and Person 60: 130 bits
  Distance between Person 50 and Person 61: 122 bits
  Distance between Person 50 and Person 62: 124 bits
  Distance between Person 50 and Person 63: 132 bits
  Distance between Person 50 and Person 64: 125 bits
  Distance between Person 50 and Person 65: 129 bits
  Distance between Person 50 and Person 66: 141 bits
  Distance between Person 50 and Person 67: 138 bits
  Distance between Person 50 and Person 68: 126 bits
  Distance between Person 50 and Person 69: 128 bits
  Distance between Person 50 and Person 70: 128 bits
  Distance between Person 50 and Person 71: 117 bits
  Distance between Person 50 and Person 72: 128 bits
  Distance between Person 50 and Person 73: 140 bits
  Distance between Person 50 and Person 74: 136 bits
  Distance between Person 50 and Person 75: 113 bits
  Distance between Person 50 and Person 76: 105 bits
  Distance between Person 50 and Person 77: 114 bits
  Distance between Person 50 and Person 78: 124 bits
  Distance between Person 50 and Person 79: 115 bits
  Distance between Person 50 and Person 80: 128 bits
  Distance between Person 50 and Person 81: 136 bits
  Distance between Person 50 and Person 82: 129 bits
  Distance between Person 50 and Person 83: 138 bits
  Distance between Person 50 and Person 84: 122 bits
  Distance between Person 50 and Person 85: 123 bits
  Distance between Person 50 and Person 86: 136 bits
  Distance between Person 50 and Person 87: 116 bits
  Distance between Person 50 and Person 88: 125 bits
  Distance between Person 50 and Person 89: 128 bits
  Distance between Person 51 and Person 52: 107 bits
  Distance between Person 51 and Person 53: 134 bits
  Distance between Person 51 and Person 54: 119 bits
  Distance between Person 51 and Person 55: 121 bits
  Distance between Person 51 and Person 56: 113 bits
  Distance between Person 51 and Person 57: 156 bits
  Distance between Person 51 and Person 58: 134 bits
  Distance between Person 51 and Person 59: 121 bits
  Distance between Person 51 and Person 60: 146 bits
  Distance between Person 51 and Person 61: 118 bits
  Distance between Person 51 and Person 62: 116 bits
  Distance between Person 51 and Person 63: 132 bits
  Distance between Person 51 and Person 64: 107 bits
  Distance between Person 51 and Person 65: 139 bits
  Distance between Person 51 and Person 66: 137 bits
  Distance between Person 51 and Person 67: 136 bits
  Distance between Person 51 and Person 68: 116 bits
  Distance between Person 51 and Person 69: 118 bits
  Distance between Person 51 and Person 70: 122 bits
  Distance between Person 51 and Person 71: 147 bits
  Distance between Person 51 and Person 72: 132 bits
  Distance between Person 51 and Person 73: 122 bits
  Distance between Person 51 and Person 74: 114 bits
  Distance between Person 51 and Person 75: 131 bits
  Distance between Person 51 and Person 76: 123 bits
  Distance between Person 51 and Person 77: 126 bits
  Distance between Person 51 and Person 78: 124 bits
  Distance between Person 51 and Person 79: 115 bits
  Distance between Person 51 and Person 80: 116 bits
  Distance between Person 51 and Person 81: 148 bits
  Distance between Person 51 and Person 82: 93 bits
  Distance between Person 51 and Person 83: 120 bits
  Distance between Person 51 and Person 84: 114 bits
  Distance between Person 51 and Person 85: 123 bits
  Distance between Person 51 and Person 86: 148 bits
  Distance between Person 51 and Person 87: 124 bits
  Distance between Person 51 and Person 88: 125 bits
  Distance between Person 51 and Person 89: 114 bits
  Distance between Person 52 and Person 53: 105 bits
  Distance between Person 52 and Person 54: 120 bits
  Distance between Person 52 and Person 55: 48 bits
  Distance between Person 52 and Person 56: 134 bits
  Distance between Person 52 and Person 57: 131 bits
  Distance between Person 52 and Person 58: 123 bits
  Distance between Person 52 and Person 59: 150 bits
  Distance between Person 52 and Person 60: 133 bits
  Distance between Person 52 and Person 61: 119 bits
  Distance between Person 52 and Person 62: 127 bits
  Distance between Person 52 and Person 63: 129 bits
  Distance between Person 52 and Person 64: 116 bits
  Distance between Person 52 and Person 65: 132 bits
  Distance between Person 52 and Person 66: 112 bits
  Distance between Person 52 and Person 67: 139 bits
  Distance between Person 52 and Person 68: 135 bits
  Distance between Person 52 and Person 69: 127 bits
  Distance between Person 52 and Person 70: 137 bits
  Distance between Person 52 and Person 71: 136 bits
  Distance between Person 52 and Person 72: 129 bits
  Distance between Person 52 and Person 73: 135 bits
  Distance between Person 52 and Person 74: 113 bits
  Distance between Person 52 and Person 75: 136 bits
  Distance between Person 52 and Person 76: 110 bits
  Distance between Person 52 and Person 77: 121 bits
  Distance between Person 52 and Person 78: 135 bits
  Distance between Person 52 and Person 79: 112 bits
  Distance between Person 52 and Person 80: 113 bits
  Distance between Person 52 and Person 81: 147 bits
  Distance between Person 52 and Person 82: 126 bits
  Distance between Person 52 and Person 83: 125 bits
  Distance between Person 52 and Person 84: 117 bits
  Distance between Person 52 and Person 85: 130 bits
  Distance between Person 52 and Person 86: 123 bits
  Distance between Person 52 and Person 87: 125 bits
  Distance between Person 52 and Person 88: 132 bits
  Distance between Person 52 and Person 89: 115 bits
  Distance between Person 53 and Person 54: 109 bits
  Distance between Person 53 and Person 55: 97 bits
  Distance between Person 53 and Person 56: 129 bits
  Distance between Person 53 and Person 57: 116 bits
  Distance between Person 53 and Person 58: 130 bits
  Distance between Person 53 and Person 59: 151 bits
  Distance between Person 53 and Person 60: 120 bits
  Distance between Person 53 and Person 61: 126 bits
  Distance between Person 53 and Person 62: 138 bits
  Distance between Person 53 and Person 63: 114 bits
  Distance between Person 53 and Person 64: 125 bits
  Distance between Person 53 and Person 65: 125 bits
  Distance between Person 53 and Person 66: 131 bits
  Distance between Person 53 and Person 67: 120 bits
  Distance between Person 53 and Person 68: 156 bits
  Distance between Person 53 and Person 69: 122 bits
  Distance between Person 53 and Person 70: 120 bits
  Distance between Person 53 and Person 71: 147 bits
  Distance between Person 53 and Person 72: 144 bits
  Distance between Person 53 and Person 73: 120 bits
  Distance between Person 53 and Person 74: 130 bits
  Distance between Person 53 and Person 75: 131 bits
  Distance between Person 53 and Person 76: 107 bits
  Distance between Person 53 and Person 77: 128 bits
  Distance between Person 53 and Person 78: 148 bits
  Distance between Person 53 and Person 79: 135 bits
  Distance between Person 53 and Person 80: 134 bits
  Distance between Person 53 and Person 81: 134 bits
  Distance between Person 53 and Person 82: 119 bits
  Distance between Person 53 and Person 83: 130 bits
  Distance between Person 53 and Person 84: 132 bits
  Distance between Person 53 and Person 85: 129 bits
  Distance between Person 53 and Person 86: 126 bits
  Distance between Person 53 and Person 87: 138 bits
  Distance between Person 53 and Person 88: 113 bits
  Distance between Person 53 and Person 89: 112 bits
  Distance between Person 54 and Person 55: 116 bits
  Distance between Person 54 and Person 56: 116 bits
  Distance between Person 54 and Person 57: 131 bits
  Distance between Person 54 and Person 58: 97 bits
  Distance between Person 54 and Person 59: 120 bits
  Distance between Person 54 and Person 60: 119 bits
  Distance between Person 54 and Person 61: 141 bits
  Distance between Person 54 and Person 62: 121 bits
  Distance between Person 54 and Person 63: 129 bits
  Distance between Person 54 and Person 64: 88 bits
  Distance between Person 54 and Person 65: 130 bits
  Distance between Person 54 and Person 66: 140 bits
  Distance between Person 54 and Person 67: 129 bits
  Distance between Person 54 and Person 68: 133 bits
  Distance between Person 54 and Person 69: 111 bits
  Distance between Person 54 and Person 70: 113 bits
  Distance between Person 54 and Person 71: 148 bits
  Distance between Person 54 and Person 72: 141 bits
  Distance between Person 54 and Person 73: 123 bits
  Distance between Person 54 and Person 74: 117 bits
  Distance between Person 54 and Person 75: 106 bits
  Distance between Person 54 and Person 76: 90 bits
  Distance between Person 54 and Person 77: 123 bits
  Distance between Person 54 and Person 78: 129 bits
  Distance between Person 54 and Person 79: 138 bits
  Distance between Person 54 and Person 80: 119 bits
  Distance between Person 54 and Person 81: 129 bits
  Distance between Person 54 and Person 82: 120 bits
  Distance between Person 54 and Person 83: 145 bits
  Distance between Person 54 and Person 84: 141 bits
  Distance between Person 54 and Person 85: 130 bits
  Distance between Person 54 and Person 86: 137 bits
  Distance between Person 54 and Person 87: 133 bits
  Distance between Person 54 and Person 88: 110 bits
  Distance between Person 54 and Person 89: 127 bits
  Distance between Person 55 and Person 56: 132 bits
  Distance between Person 55 and Person 57: 115 bits
  Distance between Person 55 and Person 58: 127 bits
  Distance between Person 55 and Person 59: 138 bits
  Distance between Person 55 and Person 60: 111 bits
  Distance between Person 55 and Person 61: 115 bits
  Distance between Person 55 and Person 62: 129 bits
  Distance between Person 55 and Person 63: 131 bits
  Distance between Person 55 and Person 64: 116 bits
  Distance between Person 55 and Person 65: 128 bits
  Distance between Person 55 and Person 66: 108 bits
  Distance between Person 55 and Person 67: 133 bits
  Distance between Person 55 and Person 68: 139 bits
  Distance between Person 55 and Person 69: 115 bits
  Distance between Person 55 and Person 70: 127 bits
  Distance between Person 55 and Person 71: 124 bits
  Distance between Person 55 and Person 72: 137 bits
  Distance between Person 55 and Person 73: 143 bits
  Distance between Person 55 and Person 74: 123 bits
  Distance between Person 55 and Person 75: 120 bits
  Distance between Person 55 and Person 76: 110 bits
  Distance between Person 55 and Person 77: 113 bits
  Distance between Person 55 and Person 78: 127 bits
  Distance between Person 55 and Person 79: 130 bits
  Distance between Person 55 and Person 80: 105 bits
  Distance between Person 55 and Person 81: 143 bits
  Distance between Person 55 and Person 82: 124 bits
  Distance between Person 55 and Person 83: 117 bits
  Distance between Person 55 and Person 84: 141 bits
  Distance between Person 55 and Person 85: 134 bits
  Distance between Person 55 and Person 86: 113 bits
  Distance between Person 55 and Person 87: 135 bits
  Distance between Person 55 and Person 88: 124 bits
  Distance between Person 55 and Person 89: 115 bits
  Distance between Person 56 and Person 57: 109 bits
  Distance between Person 56 and Person 58: 137 bits
  Distance between Person 56 and Person 59: 108 bits
  Distance between Person 56 and Person 60: 129 bits
  Distance between Person 56 and Person 61: 135 bits
  Distance between Person 56 and Person 62: 129 bits
  Distance between Person 56 and Person 63: 137 bits
  Distance between Person 56 and Person 64: 122 bits
  Distance between Person 56 and Person 65: 140 bits
  Distance between Person 56 and Person 66: 150 bits
  Distance between Person 56 and Person 67: 119 bits
  Distance between Person 56 and Person 68: 123 bits
  Distance between Person 56 and Person 69: 135 bits
  Distance between Person 56 and Person 70: 137 bits
  Distance between Person 56 and Person 71: 118 bits
  Distance between Person 56 and Person 72: 141 bits
  Distance between Person 56 and Person 73: 143 bits
  Distance between Person 56 and Person 74: 133 bits
  Distance between Person 56 and Person 75: 122 bits
  Distance between Person 56 and Person 76: 138 bits
  Distance between Person 56 and Person 77: 117 bits
  Distance between Person 56 and Person 78: 137 bits
  Distance between Person 56 and Person 79: 128 bits
  Distance between Person 56 and Person 80: 109 bits
  Distance between Person 56 and Person 81: 139 bits
  Distance between Person 56 and Person 82: 114 bits
  Distance between Person 56 and Person 83: 127 bits
  Distance between Person 56 and Person 84: 101 bits
  Distance between Person 56 and Person 85: 128 bits
  Distance between Person 56 and Person 86: 143 bits
  Distance between Person 56 and Person 87: 135 bits
  Distance between Person 56 and Person 88: 124 bits
  Distance between Person 56 and Person 89: 125 bits
  Distance between Person 57 and Person 58: 138 bits
  Distance between Person 57 and Person 59: 117 bits
  Distance between Person 57 and Person 60: 104 bits
  Distance between Person 57 and Person 61: 142 bits
  Distance between Person 57 and Person 62: 136 bits
  Distance between Person 57 and Person 63: 130 bits
  Distance between Person 57 and Person 64: 139 bits
  Distance between Person 57 and Person 65: 129 bits
  Distance between Person 57 and Person 66: 127 bits
  Distance between Person 57 and Person 67: 114 bits
  Distance between Person 57 and Person 68: 142 bits
  Distance between Person 57 and Person 69: 138 bits
  Distance between Person 57 and Person 70: 132 bits
  Distance between Person 57 and Person 71: 125 bits
  Distance between Person 57 and Person 72: 130 bits
  Distance between Person 57 and Person 73: 138 bits
  Distance between Person 57 and Person 74: 134 bits
  Distance between Person 57 and Person 75: 141 bits
  Distance between Person 57 and Person 76: 127 bits
  Distance between Person 57 and Person 77: 126 bits
  Distance between Person 57 and Person 78: 124 bits
  Distance between Person 57 and Person 79: 121 bits
  Distance between Person 57 and Person 80: 136 bits
  Distance between Person 57 and Person 81: 124 bits
  Distance between Person 57 and Person 82: 141 bits
  Distance between Person 57 and Person 83: 122 bits
  Distance between Person 57 and Person 84: 112 bits
  Distance between Person 57 and Person 85: 129 bits
  Distance between Person 57 and Person 86: 140 bits
  Distance between Person 57 and Person 87: 126 bits
  Distance between Person 57 and Person 88: 123 bits
  Distance between Person 57 and Person 89: 128 bits
  Distance between Person 58 and Person 59: 127 bits
  Distance between Person 58 and Person 60: 138 bits
  Distance between Person 58 and Person 61: 124 bits
  Distance between Person 58 and Person 62: 122 bits
  Distance between Person 58 and Person 63: 118 bits
  Distance between Person 58 and Person 64: 135 bits
  Distance between Person 58 and Person 65: 117 bits
  Distance between Person 58 and Person 66: 133 bits
  Distance between Person 58 and Person 67: 122 bits
  Distance between Person 58 and Person 68: 142 bits
  Distance between Person 58 and Person 69: 112 bits
  Distance between Person 58 and Person 70: 134 bits
  Distance between Person 58 and Person 71: 135 bits
  Distance between Person 58 and Person 72: 108 bits
  Distance between Person 58 and Person 73: 100 bits
  Distance between Person 58 and Person 74: 130 bits
  Distance between Person 58 and Person 75: 115 bits
  Distance between Person 58 and Person 76: 115 bits
  Distance between Person 58 and Person 77: 130 bits
  Distance between Person 58 and Person 78: 124 bits
  Distance between Person 58 and Person 79: 133 bits
  Distance between Person 58 and Person 80: 118 bits
  Distance between Person 58 and Person 81: 126 bits
  Distance between Person 58 and Person 82: 129 bits
  Distance between Person 58 and Person 83: 134 bits
  Distance between Person 58 and Person 84: 118 bits
  Distance between Person 58 and Person 85: 115 bits
  Distance between Person 58 and Person 86: 120 bits
  Distance between Person 58 and Person 87: 114 bits
  Distance between Person 58 and Person 88: 145 bits
  Distance between Person 58 and Person 89: 124 bits
  Distance between Person 59 and Person 60: 101 bits
  Distance between Person 59 and Person 61: 145 bits
  Distance between Person 59 and Person 62: 101 bits
  Distance between Person 59 and Person 63: 133 bits
  Distance between Person 59 and Person 64: 122 bits
  Distance between Person 59 and Person 65: 130 bits
  Distance between Person 59 and Person 66: 126 bits
  Distance between Person 59 and Person 67: 109 bits
  Distance between Person 59 and Person 68: 119 bits
  Distance between Person 59 and Person 69: 129 bits
  Distance between Person 59 and Person 70: 119 bits
  Distance between Person 59 and Person 71: 130 bits
  Distance between Person 59 and Person 72: 119 bits
  Distance between Person 59 and Person 73: 149 bits
  Distance between Person 59 and Person 74: 125 bits
  Distance between Person 59 and Person 75: 128 bits
  Distance between Person 59 and Person 76: 120 bits
  Distance between Person 59 and Person 77: 107 bits
  Distance between Person 59 and Person 78: 125 bits
  Distance between Person 59 and Person 79: 126 bits
  Distance between Person 59 and Person 80: 139 bits
  Distance between Person 59 and Person 81: 129 bits
  Distance between Person 59 and Person 82: 148 bits
  Distance between Person 59 and Person 83: 139 bits
  Distance between Person 59 and Person 84: 125 bits
  Distance between Person 59 and Person 85: 134 bits
  Distance between Person 59 and Person 86: 143 bits
  Distance between Person 59 and Person 87: 121 bits
  Distance between Person 59 and Person 88: 134 bits
  Distance between Person 59 and Person 89: 147 bits
  Distance between Person 60 and Person 61: 136 bits
  Distance between Person 60 and Person 62: 122 bits
  Distance between Person 60 and Person 63: 136 bits
  Distance between Person 60 and Person 64: 131 bits
  Distance between Person 60 and Person 65: 133 bits
  Distance between Person 60 and Person 66: 135 bits
  Distance between Person 60 and Person 67: 124 bits
  Distance between Person 60 and Person 68: 112 bits
  Distance between Person 60 and Person 69: 128 bits
  Distance between Person 60 and Person 70: 110 bits
  Distance between Person 60 and Person 71: 121 bits
  Distance between Person 60 and Person 72: 120 bits
  Distance between Person 60 and Person 73: 144 bits
  Distance between Person 60 and Person 74: 130 bits
  Distance between Person 60 and Person 75: 133 bits
  Distance between Person 60 and Person 76: 135 bits
  Distance between Person 60 and Person 77: 122 bits
  Distance between Person 60 and Person 78: 122 bits
  Distance between Person 60 and Person 79: 129 bits
  Distance between Person 60 and Person 80: 122 bits
  Distance between Person 60 and Person 81: 142 bits
  Distance between Person 60 and Person 82: 131 bits
  Distance between Person 60 and Person 83: 130 bits
  Distance between Person 60 and Person 84: 136 bits
  Distance between Person 60 and Person 85: 117 bits
  Distance between Person 60 and Person 86: 134 bits
  Distance between Person 60 and Person 87: 134 bits
  Distance between Person 60 and Person 88: 139 bits
  Distance between Person 60 and Person 89: 136 bits
  Distance between Person 61 and Person 62: 138 bits
  Distance between Person 61 and Person 63: 150 bits
  Distance between Person 61 and Person 64: 125 bits
  Distance between Person 61 and Person 65: 131 bits
  Distance between Person 61 and Person 66: 127 bits
  Distance between Person 61 and Person 67: 160 bits
  Distance between Person 61 and Person 68: 128 bits
  Distance between Person 61 and Person 69: 132 bits
  Distance between Person 61 and Person 70: 122 bits
  Distance between Person 61 and Person 71: 97 bits
  Distance between Person 61 and Person 72: 116 bits
  Distance between Person 61 and Person 73: 134 bits
  Distance between Person 61 and Person 74: 118 bits
  Distance between Person 61 and Person 75: 145 bits
  Distance between Person 61 and Person 76: 123 bits
  Distance between Person 61 and Person 77: 130 bits
  Distance between Person 61 and Person 78: 130 bits
  Distance between Person 61 and Person 79: 125 bits
  Distance between Person 61 and Person 80: 112 bits
  Distance between Person 61 and Person 81: 158 bits
  Distance between Person 61 and Person 82: 123 bits
  Distance between Person 61 and Person 83: 130 bits
  Distance between Person 61 and Person 84: 100 bits
  Distance between Person 61 and Person 85: 111 bits
  Distance between Person 61 and Person 86: 74 bits
  Distance between Person 61 and Person 87: 138 bits
  Distance between Person 61 and Person 88: 117 bits
  Distance between Person 61 and Person 89: 114 bits
  Distance between Person 62 and Person 63: 118 bits
  Distance between Person 62 and Person 64: 109 bits
  Distance between Person 62 and Person 65: 133 bits
  Distance between Person 62 and Person 66: 149 bits
  Distance between Person 62 and Person 67: 138 bits
  Distance between Person 62 and Person 68: 128 bits
  Distance between Person 62 and Person 69: 126 bits
  Distance between Person 62 and Person 70: 88 bits
  Distance between Person 62 and Person 71: 115 bits
  Distance between Person 62 and Person 72: 132 bits
  Distance between Person 62 and Person 73: 130 bits
  Distance between Person 62 and Person 74: 128 bits
  Distance between Person 62 and Person 75: 125 bits
  Distance between Person 62 and Person 76: 113 bits
  Distance between Person 62 and Person 77: 132 bits
  Distance between Person 62 and Person 78: 108 bits
  Distance between Person 62 and Person 79: 133 bits
  Distance between Person 62 and Person 80: 144 bits
  Distance between Person 62 and Person 81: 124 bits
  Distance between Person 62 and Person 82: 139 bits
  Distance between Person 62 and Person 83: 128 bits
  Distance between Person 62 and Person 84: 120 bits
  Distance between Person 62 and Person 85: 131 bits
  Distance between Person 62 and Person 86: 128 bits
  Distance between Person 62 and Person 87: 130 bits
  Distance between Person 62 and Person 88: 137 bits
  Distance between Person 62 and Person 89: 104 bits
  Distance between Person 63 and Person 64: 145 bits
  Distance between Person 63 and Person 65: 121 bits
  Distance between Person 63 and Person 66: 123 bits
  Distance between Person 63 and Person 67: 108 bits
  Distance between Person 63 and Person 68: 128 bits
  Distance between Person 63 and Person 69: 134 bits
  Distance between Person 63 and Person 70: 140 bits
  Distance between Person 63 and Person 71: 143 bits
  Distance between Person 63 and Person 72: 152 bits
  Distance between Person 63 and Person 73: 132 bits
  Distance between Person 63 and Person 74: 126 bits
  Distance between Person 63 and Person 75: 113 bits
  Distance between Person 63 and Person 76: 101 bits
  Distance between Person 63 and Person 77: 132 bits
  Distance between Person 63 and Person 78: 120 bits
  Distance between Person 63 and Person 79: 145 bits
  Distance between Person 63 and Person 80: 150 bits
  Distance between Person 63 and Person 81: 92 bits
  Distance between Person 63 and Person 82: 115 bits
  Distance between Person 63 and Person 83: 116 bits
  Distance between Person 63 and Person 84: 138 bits
  Distance between Person 63 and Person 85: 135 bits
  Distance between Person 63 and Person 86: 132 bits
  Distance between Person 63 and Person 87: 122 bits
  Distance between Person 63 and Person 88: 133 bits
  Distance between Person 63 and Person 89: 116 bits
  Distance between Person 64 and Person 65: 116 bits
  Distance between Person 64 and Person 66: 100 bits
  Distance between Person 64 and Person 67: 149 bits
  Distance between Person 64 and Person 68: 121 bits
  Distance between Person 64 and Person 69: 121 bits
  Distance between Person 64 and Person 70: 115 bits
  Distance between Person 64 and Person 71: 128 bits
  Distance between Person 64 and Person 72: 121 bits
  Distance between Person 64 and Person 73: 133 bits
  Distance between Person 64 and Person 74: 121 bits
  Distance between Person 64 and Person 75: 120 bits
  Distance between Person 64 and Person 76: 120 bits
  Distance between Person 64 and Person 77: 115 bits
  Distance between Person 64 and Person 78: 133 bits
  Distance between Person 64 and Person 79: 114 bits
  Distance between Person 64 and Person 80: 129 bits
  Distance between Person 64 and Person 81: 139 bits
  Distance between Person 64 and Person 82: 112 bits
  Distance between Person 64 and Person 83: 139 bits
  Distance between Person 64 and Person 84: 139 bits
  Distance between Person 64 and Person 85: 130 bits
  Distance between Person 64 and Person 86: 123 bits
  Distance between Person 64 and Person 87: 129 bits
  Distance between Person 64 and Person 88: 132 bits
  Distance between Person 64 and Person 89: 141 bits
  Distance between Person 65 and Person 66: 98 bits
  Distance between Person 65 and Person 67: 139 bits
  Distance between Person 65 and Person 68: 111 bits
  Distance between Person 65 and Person 69: 123 bits
  Distance between Person 65 and Person 70: 115 bits
  Distance between Person 65 and Person 71: 124 bits
  Distance between Person 65 and Person 72: 115 bits
  Distance between Person 65 and Person 73: 125 bits
  Distance between Person 65 and Person 74: 137 bits
  Distance between Person 65 and Person 75: 120 bits
  Distance between Person 65 and Person 76: 122 bits
  Distance between Person 65 and Person 77: 123 bits
  Distance between Person 65 and Person 78: 123 bits
  Distance between Person 65 and Person 79: 114 bits
  Distance between Person 65 and Person 80: 145 bits
  Distance between Person 65 and Person 81: 157 bits
  Distance between Person 65 and Person 82: 122 bits
  Distance between Person 65 and Person 83: 123 bits
  Distance between Person 65 and Person 84: 115 bits
  Distance between Person 65 and Person 85: 114 bits
  Distance between Person 65 and Person 86: 141 bits
  Distance between Person 65 and Person 87: 115 bits
  Distance between Person 65 and Person 88: 120 bits
  Distance between Person 65 and Person 89: 139 bits
  Distance between Person 66 and Person 67: 125 bits
  Distance between Person 66 and Person 68: 123 bits
  Distance between Person 66 and Person 69: 117 bits
  Distance between Person 66 and Person 70: 121 bits
  Distance between Person 66 and Person 71: 138 bits
  Distance between Person 66 and Person 72: 119 bits
  Distance between Person 66 and Person 73: 143 bits
  Distance between Person 66 and Person 74: 127 bits
  Distance between Person 66 and Person 75: 124 bits
  Distance between Person 66 and Person 76: 122 bits
  Distance between Person 66 and Person 77: 123 bits
  Distance between Person 66 and Person 78: 125 bits
  Distance between Person 66 and Person 79: 108 bits
  Distance between Person 66 and Person 80: 129 bits
  Distance between Person 66 and Person 81: 121 bits
  Distance between Person 66 and Person 82: 114 bits
  Distance between Person 66 and Person 83: 111 bits
  Distance between Person 66 and Person 84: 141 bits
  Distance between Person 66 and Person 85: 114 bits
  Distance between Person 66 and Person 86: 113 bits
  Distance between Person 66 and Person 87: 135 bits
  Distance between Person 66 and Person 88: 134 bits
  Distance between Person 66 and Person 89: 151 bits
  Distance between Person 67 and Person 68: 156 bits
  Distance between Person 67 and Person 69: 118 bits
  Distance between Person 67 and Person 70: 134 bits
  Distance between Person 67 and Person 71: 139 bits
  Distance between Person 67 and Person 72: 132 bits
  Distance between Person 67 and Person 73: 122 bits
  Distance between Person 67 and Person 74: 130 bits
  Distance between Person 67 and Person 75: 135 bits
  Distance between Person 67 and Person 76: 129 bits
  Distance between Person 67 and Person 77: 126 bits
  Distance between Person 67 and Person 78: 116 bits
  Distance between Person 67 and Person 79: 137 bits
  Distance between Person 67 and Person 80: 128 bits
  Distance between Person 67 and Person 81: 106 bits
  Distance between Person 67 and Person 82: 127 bits
  Distance between Person 67 and Person 83: 148 bits
  Distance between Person 67 and Person 84: 132 bits
  Distance between Person 67 and Person 85: 119 bits
  Distance between Person 67 and Person 86: 128 bits
  Distance between Person 67 and Person 87: 138 bits
  Distance between Person 67 and Person 88: 131 bits
  Distance between Person 67 and Person 89: 140 bits
  Distance between Person 68 and Person 69: 144 bits
  Distance between Person 68 and Person 70: 134 bits
  Distance between Person 68 and Person 71: 123 bits
  Distance between Person 68 and Person 72: 128 bits
  Distance between Person 68 and Person 73: 130 bits
  Distance between Person 68 and Person 74: 130 bits
  Distance between Person 68 and Person 75: 135 bits
  Distance between Person 68 and Person 76: 121 bits
  Distance between Person 68 and Person 77: 128 bits
  Distance between Person 68 and Person 78: 128 bits
  Distance between Person 68 and Person 79: 117 bits
  Distance between Person 68 and Person 80: 138 bits
  Distance between Person 68 and Person 81: 130 bits
  Distance between Person 68 and Person 82: 121 bits
  Distance between Person 68 and Person 83: 126 bits
  Distance between Person 68 and Person 84: 104 bits
  Distance between Person 68 and Person 85: 127 bits
  Distance between Person 68 and Person 86: 140 bits
  Distance between Person 68 and Person 87: 118 bits
  Distance between Person 68 and Person 88: 127 bits
  Distance between Person 68 and Person 89: 124 bits
  Distance between Person 69 and Person 70: 128 bits
  Distance between Person 69 and Person 71: 113 bits
  Distance between Person 69 and Person 72: 116 bits
  Distance between Person 69 and Person 73: 110 bits
  Distance between Person 69 and Person 74: 114 bits
  Distance between Person 69 and Person 75: 101 bits
  Distance between Person 69 and Person 76: 121 bits
  Distance between Person 69 and Person 77: 132 bits
  Distance between Person 69 and Person 78: 114 bits
  Distance between Person 69 and Person 79: 121 bits
  Distance between Person 69 and Person 80: 120 bits
  Distance between Person 69 and Person 81: 112 bits
  Distance between Person 69 and Person 82: 119 bits
  Distance between Person 69 and Person 83: 124 bits
  Distance between Person 69 and Person 84: 144 bits
  Distance between Person 69 and Person 85: 127 bits
  Distance between Person 69 and Person 86: 126 bits
  Distance between Person 69 and Person 87: 118 bits
  Distance between Person 69 and Person 88: 123 bits
  Distance between Person 69 and Person 89: 128 bits
  Distance between Person 70 and Person 71: 137 bits
  Distance between Person 70 and Person 72: 118 bits
  Distance between Person 70 and Person 73: 130 bits
  Distance between Person 70 and Person 74: 126 bits
  Distance between Person 70 and Person 75: 105 bits
  Distance between Person 70 and Person 76: 127 bits
  Distance between Person 70 and Person 77: 116 bits
  Distance between Person 70 and Person 78: 112 bits
  Distance between Person 70 and Person 79: 131 bits
  Distance between Person 70 and Person 80: 112 bits
  Distance between Person 70 and Person 81: 142 bits
  Distance between Person 70 and Person 82: 137 bits
  Distance between Person 70 and Person 83: 128 bits
  Distance between Person 70 and Person 84: 132 bits
  Distance between Person 70 and Person 85: 119 bits
  Distance between Person 70 and Person 86: 122 bits
  Distance between Person 70 and Person 87: 134 bits
  Distance between Person 70 and Person 88: 105 bits
  Distance between Person 70 and Person 89: 126 bits
  Distance between Person 71 and Person 72: 137 bits
  Distance between Person 71 and Person 73: 145 bits
  Distance between Person 71 and Person 74: 139 bits
  Distance between Person 71 and Person 75: 128 bits
  Distance between Person 71 and Person 76: 134 bits
  Distance between Person 71 and Person 77: 129 bits
  Distance between Person 71 and Person 78: 129 bits
  Distance between Person 71 and Person 79: 112 bits
  Distance between Person 71 and Person 80: 121 bits
  Distance between Person 71 and Person 81: 137 bits
  Distance between Person 71 and Person 82: 134 bits
  Distance between Person 71 and Person 83: 115 bits
  Distance between Person 71 and Person 84: 123 bits
  Distance between Person 71 and Person 85: 118 bits
  Distance between Person 71 and Person 86: 107 bits
  Distance between Person 71 and Person 87: 139 bits
  Distance between Person 71 and Person 88: 118 bits
  Distance between Person 71 and Person 89: 127 bits
  Distance between Person 72 and Person 73: 124 bits
  Distance between Person 72 and Person 74: 120 bits
  Distance between Person 72 and Person 75: 133 bits
  Distance between Person 72 and Person 76: 135 bits
  Distance between Person 72 and Person 77: 134 bits
  Distance between Person 72 and Person 78: 126 bits
  Distance between Person 72 and Person 79: 121 bits
  Distance between Person 72 and Person 80: 114 bits
  Distance between Person 72 and Person 81: 138 bits
  Distance between Person 72 and Person 82: 133 bits
  Distance between Person 72 and Person 83: 130 bits
  Distance between Person 72 and Person 84: 124 bits
  Distance between Person 72 and Person 85: 113 bits
  Distance between Person 72 and Person 86: 108 bits
  Distance between Person 72 and Person 87: 68 bits
  Distance between Person 72 and Person 88: 127 bits
  Distance between Person 72 and Person 89: 136 bits
  Distance between Person 73 and Person 74: 140 bits
  Distance between Person 73 and Person 75: 125 bits
  Distance between Person 73 and Person 76: 137 bits
  Distance between Person 73 and Person 77: 138 bits
  Distance between Person 73 and Person 78: 108 bits
  Distance between Person 73 and Person 79: 149 bits
  Distance between Person 73 and Person 80: 122 bits
  Distance between Person 73 and Person 81: 120 bits
  Distance between Person 73 and Person 82: 119 bits
  Distance between Person 73 and Person 83: 144 bits
  Distance between Person 73 and Person 84: 136 bits
  Distance between Person 73 and Person 85: 135 bits
  Distance between Person 73 and Person 86: 120 bits
  Distance between Person 73 and Person 87: 128 bits
  Distance between Person 73 and Person 88: 137 bits
  Distance between Person 73 and Person 89: 116 bits
  Distance between Person 74 and Person 75: 141 bits
  Distance between Person 74 and Person 76: 119 bits
  Distance between Person 74 and Person 77: 134 bits
  Distance between Person 74 and Person 78: 126 bits
  Distance between Person 74 and Person 79: 121 bits
  Distance between Person 74 and Person 80: 128 bits
  Distance between Person 74 and Person 81: 136 bits
  Distance between Person 74 and Person 82: 137 bits
  Distance between Person 74 and Person 83: 124 bits
  Distance between Person 74 and Person 84: 116 bits
  Distance between Person 74 and Person 85: 117 bits
  Distance between Person 74 and Person 86: 138 bits
  Distance between Person 74 and Person 87: 128 bits
  Distance between Person 74 and Person 88: 131 bits
  Distance between Person 74 and Person 89: 130 bits
  Distance between Person 75 and Person 76: 122 bits
  Distance between Person 75 and Person 77: 119 bits
  Distance between Person 75 and Person 78: 133 bits
  Distance between Person 75 and Person 79: 114 bits
  Distance between Person 75 and Person 80: 125 bits
  Distance between Person 75 and Person 81: 115 bits
  Distance between Person 75 and Person 82: 124 bits
  Distance between Person 75 and Person 83: 139 bits
  Distance between Person 75 and Person 84: 157 bits
  Distance between Person 75 and Person 85: 148 bits
  Distance between Person 75 and Person 86: 135 bits
  Distance between Person 75 and Person 87: 125 bits
  Distance between Person 75 and Person 88: 110 bits
  Distance between Person 75 and Person 89: 113 bits
  Distance between Person 76 and Person 77: 129 bits
  Distance between Person 76 and Person 78: 121 bits
  Distance between Person 76 and Person 79: 130 bits
  Distance between Person 76 and Person 80: 137 bits
  Distance between Person 76 and Person 81: 121 bits
  Distance between Person 76 and Person 82: 122 bits
  Distance between Person 76 and Person 83: 151 bits
  Distance between Person 76 and Person 84: 125 bits
  Distance between Person 76 and Person 85: 130 bits
  Distance between Person 76 and Person 86: 145 bits
  Distance between Person 76 and Person 87: 125 bits
  Distance between Person 76 and Person 88: 124 bits
  Distance between Person 76 and Person 89: 121 bits
  Distance between Person 77 and Person 78: 112 bits
  Distance between Person 77 and Person 79: 115 bits
  Distance between Person 77 and Person 80: 120 bits
  Distance between Person 77 and Person 81: 134 bits
  Distance between Person 77 and Person 82: 141 bits
  Distance between Person 77 and Person 83: 134 bits
  Distance between Person 77 and Person 84: 138 bits
  Distance between Person 77 and Person 85: 137 bits
  Distance between Person 77 and Person 86: 134 bits
  Distance between Person 77 and Person 87: 136 bits
  Distance between Person 77 and Person 88: 131 bits
  Distance between Person 77 and Person 89: 142 bits
  Distance between Person 78 and Person 79: 119 bits
  Distance between Person 78 and Person 80: 128 bits
  Distance between Person 78 and Person 81: 114 bits
  Distance between Person 78 and Person 82: 109 bits
  Distance between Person 78 and Person 83: 138 bits
  Distance between Person 78 and Person 84: 142 bits
  Distance between Person 78 and Person 85: 135 bits
  Distance between Person 78 and Person 86: 110 bits
  Distance between Person 78 and Person 87: 132 bits
  Distance between Person 78 and Person 88: 123 bits
  Distance between Person 78 and Person 89: 146 bits
  Distance between Person 79 and Person 80: 137 bits
  Distance between Person 79 and Person 81: 139 bits
  Distance between Person 79 and Person 82: 128 bits
  Distance between Person 79 and Person 83: 123 bits
  Distance between Person 79 and Person 84: 101 bits
  Distance between Person 79 and Person 85: 126 bits
  Distance between Person 79 and Person 86: 151 bits
  Distance between Person 79 and Person 87: 129 bits
  Distance between Person 79 and Person 88: 116 bits
  Distance between Person 79 and Person 89: 129 bits
  Distance between Person 80 and Person 81: 140 bits
  Distance between Person 80 and Person 82: 135 bits
  Distance between Person 80 and Person 83: 126 bits
  Distance between Person 80 and Person 84: 118 bits
  Distance between Person 80 and Person 85: 131 bits
  Distance between Person 80 and Person 86: 90 bits
  Distance between Person 80 and Person 87: 150 bits
  Distance between Person 80 and Person 88: 137 bits
  Distance between Person 80 and Person 89: 138 bits
  Distance between Person 81 and Person 82: 135 bits
  Distance between Person 81 and Person 83: 136 bits
  Distance between Person 81 and Person 84: 144 bits
  Distance between Person 81 and Person 85: 149 bits
  Distance between Person 81 and Person 86: 120 bits
  Distance between Person 81 and Person 87: 128 bits
  Distance between Person 81 and Person 88: 129 bits
  Distance between Person 81 and Person 89: 128 bits
  Distance between Person 82 and Person 83: 117 bits
  Distance between Person 82 and Person 84: 133 bits
  Distance between Person 82 and Person 85: 148 bits
  Distance between Person 82 and Person 86: 133 bits
  Distance between Person 82 and Person 87: 127 bits
  Distance between Person 82 and Person 88: 120 bits
  Distance between Person 82 and Person 89: 119 bits
  Distance between Person 83 and Person 84: 124 bits
  Distance between Person 83 and Person 85: 125 bits
  Distance between Person 83 and Person 86: 128 bits
  Distance between Person 83 and Person 87: 118 bits
  Distance between Person 83 and Person 88: 127 bits
  Distance between Person 83 and Person 89: 116 bits
  Distance between Person 84 and Person 85: 107 bits
  Distance between Person 84 and Person 86: 140 bits
  Distance between Person 84 and Person 87: 134 bits
  Distance between Person 84 and Person 88: 121 bits
  Distance between Person 84 and Person 89: 130 bits
  Distance between Person 85 and Person 86: 119 bits
  Distance between Person 85 and Person 87: 133 bits
  Distance between Person 85 and Person 88: 138 bits
  Distance between Person 85 and Person 89: 143 bits
  Distance between Person 86 and Person 87: 138 bits
  Distance between Person 86 and Person 88: 133 bits
  Distance between Person 86 and Person 89: 138 bits
  Distance between Person 87 and Person 88: 107 bits
  Distance between Person 87 and Person 89: 118 bits
  Distance between Person 88 and Person 89: 121 bits
"""

# -----------------------------

# -----------------------------
# Extract Intra-person Hamming Distances
intra_pattern = r"Intra-person average Hamming distance:\s*([\d\.]+) bits"
intra_matches = re.findall(intra_pattern, text)
intra_hd = [float(x) for x in intra_matches]

# Compute mean and standard deviation for intra-person distances
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


# -----------------------------
# Extract Inter-person Hamming Distances and group by Person
pattern = r"Distance between Person (\d+) and Person (\d+):\s*(\d+) bits"
matches = re.findall(pattern, text)
# Create a dictionary for persons 1 to 89
person_dists = {p: [] for p in range(1, 90)}
for p1, p2, d in matches:
    p1, p2, d = int(p1), int(p2), int(d)
    person_dists[p1].append(d)
    person_dists[p2].append(d)
# Order the inter-person data by person number:
inter_data = [person_dists[p] for p in sorted(person_dists.keys())]

# -----------------------------
# Divide persons into two groups:
group1 = inter_data[:44]   # Persons 1 to 44 (44 persons)
group2 = inter_data[44:]   # Persons 45 to 89 (45 persons)

# -----------------------------
# Plot Inter-person Boxplot for Group 1 (Persons 1-44)
plt.figure(figsize=(12, 6))
bp1 = plt.boxplot(group1, patch_artist=True, showfliers=True)
# Create an array of colors from a colormap (using hsv for distinct colors)
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

# -----------------------------
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