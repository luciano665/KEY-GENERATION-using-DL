import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paste the full text into a multi-line string:
text = r"""
Person 1:
  Aggregated Key Accuracy: 82.81%
  Aggregated Key: [1 0 0 1 1 0 0 0 1 0 1 0 1 1 1 1 0 0 1 0 1 1 1 1]...
  Ground Truth:   [1 1 0 1 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step
  Intra-person average Hamming distance: 37.99 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step

Person 2:
  Aggregated Key Accuracy: 73.83%
  Aggregated Key: [1 0 1 1 1 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0 1 1 1 1]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
  Intra-person average Hamming distance: 76.95 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 3:
  Aggregated Key Accuracy: 79.30%
  Aggregated Key: [1 0 1 1 1 1 1 0 1 0 1 1 1 0 0 1 1 1 0 0 1 0 1 0]...
  Ground Truth:   [1 0 1 1 0 1 1 0 1 0 0 1 1 0 1 0 1 1 0 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step
  Intra-person average Hamming distance: 52.07 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 4:
  Aggregated Key Accuracy: 74.61%
  Aggregated Key: [1 0 1 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 1 1 1 0 0]...
  Ground Truth:   [1 1 1 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step
  Intra-person average Hamming distance: 70.11 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step

Person 5:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 1 0 0 0]...
  Ground Truth:   [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 35.45 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step

Person 6:
  Aggregated Key Accuracy: 70.70%
  Aggregated Key: [1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step
  Intra-person average Hamming distance: 74.08 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step

Person 7:
  Aggregated Key Accuracy: 73.05%
  Aggregated Key: [1 0 0 0 1 0 1 1 1 1 0 1 0 0 1 1 1 1 0 0 0 1 0 0]...
  Ground Truth:   [0 1 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 79.77 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step

Person 8:
  Aggregated Key Accuracy: 94.53%
  Aggregated Key: [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step
  Intra-person average Hamming distance: 39.08 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step

Person 9:
  Aggregated Key Accuracy: 84.38%
  Aggregated Key: [1 0 1 1 1 0 1 1 1 1 0 0 1 1 0 1 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 1 0 0 1 1 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step
  Intra-person average Hamming distance: 57.78 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step

Person 10:
  Aggregated Key Accuracy: 93.75%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
  Intra-person average Hamming distance: 44.19 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step

Person 11:
  Aggregated Key Accuracy: 82.81%
  Aggregated Key: [0 0 0 1 1 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 1 0 1]...
  Ground Truth:   [0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step
  Intra-person average Hamming distance: 63.95 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step

Person 12:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
  Ground Truth:   [0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step
  Intra-person average Hamming distance: 14.55 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step

Person 13:
  Aggregated Key Accuracy: 67.58%
  Aggregated Key: [0 0 1 1 1 0 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 1 0 1]...
  Ground Truth:   [0 1 0 1 1 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 37ms/step
  Intra-person average Hamming distance: 56.38 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step

Person 14:
  Aggregated Key Accuracy: 73.44%
  Aggregated Key: [1 0 0 0 0 0 0 1 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 1]...
  Ground Truth:   [1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
  Intra-person average Hamming distance: 68.96 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 15:
  Aggregated Key Accuracy: 86.72%
  Aggregated Key: [1 0 0 1 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 0 0 0]...
  Ground Truth:   [1 1 0 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step
  Intra-person average Hamming distance: 47.13 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step

Person 16:
  Aggregated Key Accuracy: 71.09%
  Aggregated Key: [1 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 0 1 1]...
  Ground Truth:   [0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step
  Intra-person average Hamming distance: 85.67 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step

Person 17:
  Aggregated Key Accuracy: 81.25%
  Aggregated Key: [1 0 0 1 1 0 0 1 1 0 1 0 1 1 0 0 1 0 1 0 1 0 1 1]...
  Ground Truth:   [1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step
  Intra-person average Hamming distance: 54.61 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step

Person 18:
  Aggregated Key Accuracy: 81.25%
  Aggregated Key: [1 0 0 0 1 0 1 1 1 1 0 0 1 0 1 1 0 1 1 0 1 0 1 1]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step
  Intra-person average Hamming distance: 66.21 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 81ms/step

Person 19:
  Aggregated Key Accuracy: 76.56%
  Aggregated Key: [1 0 0 1 0 1 0 1 1 1 1 1 0 1 0 1 0 1 0 0 0 1 1 0]...
  Ground Truth:   [1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 0 0 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step
  Intra-person average Hamming distance: 76.38 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step

Person 20:
  Aggregated Key Accuracy: 81.25%
  Aggregated Key: [1 0 1 1 1 0 0 1 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1]...
  Ground Truth:   [1 0 1 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 71ms/step
  Intra-person average Hamming distance: 55.78 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step

Person 21:
  Aggregated Key Accuracy: 75.39%
  Aggregated Key: [0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 1 1 1 0 1 1 0]...
  Ground Truth:   [1 0 0 0 0 1 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step
  Intra-person average Hamming distance: 59.48 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step

Person 22:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
  Ground Truth:   [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
  Intra-person average Hamming distance: 5.52 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step

Person 23:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1]...
  Ground Truth:   [1 0 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step
  Intra-person average Hamming distance: 8.15 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step

Person 24:
  Aggregated Key Accuracy: 72.27%
  Aggregated Key: [1 1 0 1 1 1 0 0 1 0 1 0 0 0 1 1 1 1 1 0 1 1 1 0]...
  Ground Truth:   [0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 75.27 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step

Person 25:
  Aggregated Key Accuracy: 85.94%
  Aggregated Key: [1 1 1 0 1 1 0 0 0 0 1 0 1 1 0 1 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step
  Intra-person average Hamming distance: 59.92 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 26:
  Aggregated Key Accuracy: 85.16%
  Aggregated Key: [0 1 0 1 1 0 0 0 1 1 0 1 0 0 1 1 1 1 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step
  Intra-person average Hamming distance: 39.22 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step

Person 27:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
  Intra-person average Hamming distance: 23.23 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step

Person 28:
  Aggregated Key Accuracy: 82.81%
  Aggregated Key: [0 0 1 0 1 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step
  Intra-person average Hamming distance: 52.36 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step

Person 29:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step
  Intra-person average Hamming distance: 23.32 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 30:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1]...
  Ground Truth:   [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step
  Intra-person average Hamming distance: 21.42 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step

Person 31:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1]...
  Ground Truth:   [0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step
  Intra-person average Hamming distance: 32.34 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step

Person 32:
  Aggregated Key Accuracy: 73.05%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 1 1 1 0 0 1 0 1 1 0 1 1 1 0]...
  Ground Truth:   [1 1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step
  Intra-person average Hamming distance: 70.97 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step

Person 33:
  Aggregated Key Accuracy: 90.23%
  Aggregated Key: [1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 0 0 0 1 1 0 0]...
  Ground Truth:   [1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step
  Intra-person average Hamming distance: 42.87 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step

Person 34:
  Aggregated Key Accuracy: 81.64%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 1 1 0 1]...
  Ground Truth:   [1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 1 0 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step
  Intra-person average Hamming distance: 48.35 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 35:
  Aggregated Key Accuracy: 85.16%
  Aggregated Key: [1 0 1 1 1 0 0 1 1 0 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
  Ground Truth:   [1 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step
  Intra-person average Hamming distance: 37.51 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step

Person 36:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [0 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0]...
  Ground Truth:   [0 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
  Intra-person average Hamming distance: 18.70 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 37:
  Aggregated Key Accuracy: 88.28%
  Aggregated Key: [1 0 0 1 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0 1 0 0 1]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step
  Intra-person average Hamming distance: 62.46 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step

Person 38:
  Aggregated Key Accuracy: 83.59%
  Aggregated Key: [1 0 0 0 0 0 0 1 1 0 0 0 0 1 0 1 0 1 1 1 0 1 0 1]...
  Ground Truth:   [1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 65.88 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step

Person 39:
  Aggregated Key Accuracy: 79.30%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 1 1 0 0 0 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step
  Intra-person average Hamming distance: 64.37 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step

Person 40:
  Aggregated Key Accuracy: 76.56%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 1 1 0 0 0 1 1 1 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step
  Intra-person average Hamming distance: 74.77 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step

Person 41:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [1 1 0 0 0 1 0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 1]...
  Ground Truth:   [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 39.66 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 42:
  Aggregated Key Accuracy: 87.50%
  Aggregated Key: [0 1 1 1 1 0 0 0 1 1 1 0 0 0 0 1 1 0 1 1 0 0 0 1]...
  Ground Truth:   [0 1 1 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 57.61 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step

Person 43:
  Aggregated Key Accuracy: 77.73%
  Aggregated Key: [0 1 0 0 1 0 0 0 1 0 0 0 0 1 1 1 1 1 1 0 1 0 1 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step
  Intra-person average Hamming distance: 75.79 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step

Person 44:
  Aggregated Key Accuracy: 82.03%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 1 1 0 1 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step
  Intra-person average Hamming distance: 65.06 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step

Person 45:
  Aggregated Key Accuracy: 76.95%
  Aggregated Key: [1 1 1 0 0 1 0 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 1 1]...
  Ground Truth:   [0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 1 1 0 1 0 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
  Intra-person average Hamming distance: 81.44 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step

Person 46:
  Aggregated Key Accuracy: 95.70%
  Aggregated Key: [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step
  Intra-person average Hamming distance: 32.84 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step

Person 47:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [0 1 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 0 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
  Intra-person average Hamming distance: 79.52 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step

Person 48:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 1 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
  Ground Truth:   [1 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step
  Intra-person average Hamming distance: 15.92 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step

Person 49:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 0 1 0 1]...
  Ground Truth:   [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step
  Intra-person average Hamming distance: 23.22 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step

Person 50:
  Aggregated Key Accuracy: 80.08%
  Aggregated Key: [1 0 0 1 1 0 0 1 1 0 1 0 0 0 1 1 0 0 0 1 0 0 1 1]...
  Ground Truth:   [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step
  Intra-person average Hamming distance: 66.83 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step

Person 51:
  Aggregated Key Accuracy: 79.69%
  Aggregated Key: [1 0 1 1 1 0 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step
  Intra-person average Hamming distance: 43.79 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step

Person 52:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [0 1 0 1 1 1 0 1 1 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step
  Intra-person average Hamming distance: 27.12 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 53:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
  Ground Truth:   [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 31.29 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step

Person 54:
  Aggregated Key Accuracy: 93.75%
  Aggregated Key: [0 0 0 1 1 0 0 1 1 0 0 0 1 0 1 1 1 0 0 1 1 0 0 1]...
  Ground Truth:   [0 0 0 0 1 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step
  Intra-person average Hamming distance: 31.13 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step

Person 55:
  Aggregated Key Accuracy: 70.70%
  Aggregated Key: [0 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 0]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 1 1 1 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 87.02 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step

Person 56:
  Aggregated Key Accuracy: 95.70%
  Aggregated Key: [0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 0 0 0 1 1 0 0]...
  Ground Truth:   [0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 49ms/step
  Intra-person average Hamming distance: 43.19 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step

Person 57:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
  Ground Truth:   [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step
  Intra-person average Hamming distance: 30.56 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 50ms/step

Person 58:
  Aggregated Key Accuracy: 93.75%
  Aggregated Key: [0 0 1 1 0 1 0 0 0 0 1 1 1 1 1 1 0 0 1 0 1 1 0 0]...
  Ground Truth:   [0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step
  Intra-person average Hamming distance: 23.21 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step

Person 59:
  Aggregated Key Accuracy: 87.11%
  Aggregated Key: [0 1 0 1 1 0 0 0 0 0 1 1 0 0 0 1 0 0 1 1 0 0 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 0 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
  Intra-person average Hamming distance: 30.48 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step

Person 60:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [1 0 0 0 1 0 0 1 1 0 0 0 0 0 1 1 1 1 0 1 0 1 0 0]...
  Ground Truth:   [0 0 0 1 1 0 0 1 0 0 0 1 0 0 1 0 1 1 0 1 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step
  Intra-person average Hamming distance: 52.37 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step

Person 61:
  Aggregated Key Accuracy: 82.42%
  Aggregated Key: [0 0 1 1 1 0 0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
  Intra-person average Hamming distance: 67.37 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step

Person 62:
  Aggregated Key Accuracy: 81.64%
  Aggregated Key: [1 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 1 0 1 0 0 0 1 0]...
  Ground Truth:   [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step
  Intra-person average Hamming distance: 51.45 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step

Person 63:
  Aggregated Key Accuracy: 95.31%
  Aggregated Key: [1 1 0 1 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step
  Intra-person average Hamming distance: 22.40 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step

Person 64:
  Aggregated Key Accuracy: 93.75%
  Aggregated Key: [0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step
  Intra-person average Hamming distance: 33.40 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 83ms/step

Person 65:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
  Ground Truth:   [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
  Intra-person average Hamming distance: 17.96 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step

Person 66:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
  Intra-person average Hamming distance: 43.38 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step

Person 67:
  Aggregated Key Accuracy: 82.81%
  Aggregated Key: [0 1 1 1 1 0 0 0 1 1 0 0 1 1 0 1 0 1 0 0 1 1 0 1]...
  Ground Truth:   [0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step
  Intra-person average Hamming distance: 44.27 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 62ms/step

Person 68:
  Aggregated Key Accuracy: 92.58%
  Aggregated Key: [0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 1 1]...
  Ground Truth:   [0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step
  Intra-person average Hamming distance: 36.11 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 69:
  Aggregated Key Accuracy: 82.42%
  Aggregated Key: [1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 65.22 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step

Person 70:
  Aggregated Key Accuracy: 78.52%
  Aggregated Key: [0 1 0 0 1 0 0 1 1 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0]...
  Ground Truth:   [0 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step
  Intra-person average Hamming distance: 59.19 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step

Person 71:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 1]...
  Ground Truth:   [1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step
  Intra-person average Hamming distance: 29.79 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 72:
  Aggregated Key Accuracy: 95.31%
  Aggregated Key: [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 0 1 0 0 0 0 0]...
  Ground Truth:   [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 18.47 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step

Person 73:
  Aggregated Key Accuracy: 77.34%
  Aggregated Key: [1 0 0 1 1 0 0 1 0 1 0 0 1 1 0 1 0 1 0 0 0 1 0 1]...
  Ground Truth:   [1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step
  Intra-person average Hamming distance: 64.17 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step

Person 74:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 0.39 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step

Person 75:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 1 1 1 0 0 1 0]...
  Ground Truth:   [0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 1 1 1 0 0 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 47.05 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 76:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [1 1 0 1 1 0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 1 0 1 1]...
  Ground Truth:   [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 19.75 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step

Person 77:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 10.53 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step

Person 78:
  Aggregated Key Accuracy: 78.12%
  Aggregated Key: [1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 0 1 0 1 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 39.36 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 79:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [0 1 1 0 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 1]...
  Ground Truth:   [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 55.06 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 80:
  Aggregated Key Accuracy: 87.50%
  Aggregated Key: [1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 0 1 1 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 0 1 1 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step
  Intra-person average Hamming distance: 64.68 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step

Person 81:
  Aggregated Key Accuracy: 80.08%
  Aggregated Key: [1 0 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 0 1 0 1 0 1 0]...
  Ground Truth:   [1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step
  Intra-person average Hamming distance: 57.81 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step

Person 82:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [1 0 1 0 1 0 0 1 1 1 1 0 1 0 1 0 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step
  Intra-person average Hamming distance: 43.31 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step

Person 83:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 68ms/step
  Intra-person average Hamming distance: 7.32 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step

Person 84:
  Aggregated Key Accuracy: 73.83%
  Aggregated Key: [1 0 0 1 1 1 0 0 0 0 1 1 1 0 1 1 1 1 0 0 0 0 1 1]...
  Ground Truth:   [1 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step
  Intra-person average Hamming distance: 74.46 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 

Person 85:
  Aggregated Key Accuracy: 91.41%
  Aggregated Key: [1 1 1 0 1 1 1 1 0 1 1 0 0 1 1 1 1 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
  Intra-person average Hamming distance: 56.89 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step

Person 86:
  Aggregated Key Accuracy: 85.55%
  Aggregated Key: [1 0 0 1 0 1 0 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 1 1]...
  Ground Truth:   [0 0 0 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step
  Intra-person average Hamming distance: 73.67 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step

Person 87:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1]...
  Ground Truth:   [0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
  Intra-person average Hamming distance: 16.04 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step

Person 88:
  Aggregated Key Accuracy: 81.25%
  Aggregated Key: [0 1 0 0 1 0 0 1 0 1 0 0 0 0 1 1 0 1 0 1 1 1 1 1]...
  Ground Truth:   [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
  Intra-person average Hamming distance: 46.17 bits
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 64ms/step

Person 89:
  Aggregated Key Accuracy: 86.33%
  Aggregated Key: [1 0 1 0 1 0 0 1 0 1 1 0 0 0 1 1 0 1 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 1 0 0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 0 0 1]...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step
  Intra-person average Hamming distance: 61.95 bits

Inter-person Hamming distances (aggregated keys):
  Distance between Person 1 and Person 2: 108 bits
  Distance between Person 1 and Person 3: 110 bits
  Distance between Person 1 and Person 4: 132 bits
  Distance between Person 1 and Person 5: 123 bits
  Distance between Person 1 and Person 6: 123 bits
  Distance between Person 1 and Person 7: 148 bits
  Distance between Person 1 and Person 8: 126 bits
  Distance between Person 1 and Person 9: 124 bits
  Distance between Person 1 and Person 10: 134 bits
  Distance between Person 1 and Person 11: 109 bits
  Distance between Person 1 and Person 12: 123 bits
  Distance between Person 1 and Person 13: 133 bits
  Distance between Person 1 and Person 14: 132 bits
  Distance between Person 1 and Person 15: 152 bits
  Distance between Person 1 and Person 16: 81 bits
  Distance between Person 1 and Person 17: 109 bits
  Distance between Person 1 and Person 18: 114 bits
  Distance between Person 1 and Person 19: 127 bits
  Distance between Person 1 and Person 20: 123 bits
  Distance between Person 1 and Person 21: 116 bits
  Distance between Person 1 and Person 22: 146 bits
  Distance between Person 1 and Person 23: 130 bits
  Distance between Person 1 and Person 24: 100 bits
  Distance between Person 1 and Person 25: 126 bits
  Distance between Person 1 and Person 26: 106 bits
  Distance between Person 1 and Person 27: 107 bits
  Distance between Person 1 and Person 28: 103 bits
  Distance between Person 1 and Person 29: 128 bits
  Distance between Person 1 and Person 30: 119 bits
  Distance between Person 1 and Person 31: 128 bits
  Distance between Person 1 and Person 32: 95 bits
  Distance between Person 1 and Person 33: 114 bits
  Distance between Person 1 and Person 34: 136 bits
  Distance between Person 1 and Person 35: 121 bits
  Distance between Person 1 and Person 36: 134 bits
  Distance between Person 1 and Person 37: 124 bits
  Distance between Person 1 and Person 38: 120 bits
  Distance between Person 1 and Person 39: 131 bits
  Distance between Person 1 and Person 40: 117 bits
  Distance between Person 1 and Person 41: 139 bits
  Distance between Person 1 and Person 42: 123 bits
  Distance between Person 1 and Person 43: 111 bits
  Distance between Person 1 and Person 44: 104 bits
  Distance between Person 1 and Person 45: 127 bits
  Distance between Person 1 and Person 46: 97 bits
  Distance between Person 1 and Person 47: 108 bits
  Distance between Person 1 and Person 48: 113 bits
  Distance between Person 1 and Person 49: 129 bits
  Distance between Person 1 and Person 50: 122 bits
  Distance between Person 1 and Person 51: 130 bits
  Distance between Person 1 and Person 52: 146 bits
  Distance between Person 1 and Person 53: 123 bits
  Distance between Person 1 and Person 54: 117 bits
  Distance between Person 1 and Person 55: 103 bits
  Distance between Person 1 and Person 56: 138 bits
  Distance between Person 1 and Person 57: 145 bits
  Distance between Person 1 and Person 58: 95 bits
  Distance between Person 1 and Person 59: 124 bits
  Distance between Person 1 and Person 60: 124 bits
  Distance between Person 1 and Person 61: 91 bits
  Distance between Person 1 and Person 62: 107 bits
  Distance between Person 1 and Person 63: 131 bits
  Distance between Person 1 and Person 64: 124 bits
  Distance between Person 1 and Person 65: 117 bits
  Distance between Person 1 and Person 66: 111 bits
  Distance between Person 1 and Person 67: 125 bits
  Distance between Person 1 and Person 68: 125 bits
  Distance between Person 1 and Person 69: 99 bits
  Distance between Person 1 and Person 70: 121 bits
  Distance between Person 1 and Person 71: 116 bits
  Distance between Person 1 and Person 72: 114 bits
  Distance between Person 1 and Person 73: 114 bits
  Distance between Person 1 and Person 74: 125 bits
  Distance between Person 1 and Person 75: 129 bits
  Distance between Person 1 and Person 76: 117 bits
  Distance between Person 1 and Person 77: 117 bits
  Distance between Person 1 and Person 78: 113 bits
  Distance between Person 1 and Person 79: 107 bits
  Distance between Person 1 and Person 80: 119 bits
  Distance between Person 1 and Person 81: 128 bits
  Distance between Person 1 and Person 82: 104 bits
  Distance between Person 1 and Person 83: 140 bits
  Distance between Person 1 and Person 84: 103 bits
  Distance between Person 1 and Person 85: 122 bits
  Distance between Person 1 and Person 86: 89 bits
  Distance between Person 1 and Person 87: 131 bits
  Distance between Person 1 and Person 88: 142 bits
  Distance between Person 1 and Person 89: 112 bits
  Distance between Person 2 and Person 3: 122 bits
  Distance between Person 2 and Person 4: 128 bits
  Distance between Person 2 and Person 5: 143 bits
  Distance between Person 2 and Person 6: 141 bits
  Distance between Person 2 and Person 7: 132 bits
  Distance between Person 2 and Person 8: 104 bits
  Distance between Person 2 and Person 9: 110 bits
  Distance between Person 2 and Person 10: 114 bits
  Distance between Person 2 and Person 11: 143 bits
  Distance between Person 2 and Person 12: 107 bits
  Distance between Person 2 and Person 13: 121 bits
  Distance between Person 2 and Person 14: 136 bits
  Distance between Person 2 and Person 15: 130 bits
  Distance between Person 2 and Person 16: 113 bits
  Distance between Person 2 and Person 17: 117 bits
  Distance between Person 2 and Person 18: 108 bits
  Distance between Person 2 and Person 19: 139 bits
  Distance between Person 2 and Person 20: 131 bits
  Distance between Person 2 and Person 21: 122 bits
  Distance between Person 2 and Person 22: 122 bits
  Distance between Person 2 and Person 23: 122 bits
  Distance between Person 2 and Person 24: 116 bits
  Distance between Person 2 and Person 25: 106 bits
  Distance between Person 2 and Person 26: 116 bits
  Distance between Person 2 and Person 27: 119 bits
  Distance between Person 2 and Person 28: 123 bits
  Distance between Person 2 and Person 29: 104 bits
  Distance between Person 2 and Person 30: 119 bits
  Distance between Person 2 and Person 31: 130 bits
  Distance between Person 2 and Person 32: 129 bits
  Distance between Person 2 and Person 33: 110 bits
  Distance between Person 2 and Person 34: 122 bits
  Distance between Person 2 and Person 35: 123 bits
  Distance between Person 2 and Person 36: 130 bits
  Distance between Person 2 and Person 37: 116 bits
  Distance between Person 2 and Person 38: 128 bits
  Distance between Person 2 and Person 39: 125 bits
  Distance between Person 2 and Person 40: 125 bits
  Distance between Person 2 and Person 41: 145 bits
  Distance between Person 2 and Person 42: 93 bits
  Distance between Person 2 and Person 43: 119 bits
  Distance between Person 2 and Person 44: 106 bits
  Distance between Person 2 and Person 45: 121 bits
  Distance between Person 2 and Person 46: 101 bits
  Distance between Person 2 and Person 47: 134 bits
  Distance between Person 2 and Person 48: 121 bits
  Distance between Person 2 and Person 49: 135 bits
  Distance between Person 2 and Person 50: 114 bits
  Distance between Person 2 and Person 51: 40 bits
  Distance between Person 2 and Person 52: 132 bits
  Distance between Person 2 and Person 53: 103 bits
  Distance between Person 2 and Person 54: 103 bits
  Distance between Person 2 and Person 55: 131 bits
  Distance between Person 2 and Person 56: 128 bits
  Distance between Person 2 and Person 57: 139 bits
  Distance between Person 2 and Person 58: 115 bits
  Distance between Person 2 and Person 59: 102 bits
  Distance between Person 2 and Person 60: 126 bits
  Distance between Person 2 and Person 61: 99 bits
  Distance between Person 2 and Person 62: 119 bits
  Distance between Person 2 and Person 63: 123 bits
  Distance between Person 2 and Person 64: 108 bits
  Distance between Person 2 and Person 65: 131 bits
  Distance between Person 2 and Person 66: 139 bits
  Distance between Person 2 and Person 67: 127 bits
  Distance between Person 2 and Person 68: 123 bits
  Distance between Person 2 and Person 69: 95 bits
  Distance between Person 2 and Person 70: 121 bits
  Distance between Person 2 and Person 71: 142 bits
  Distance between Person 2 and Person 72: 136 bits
  Distance between Person 2 and Person 73: 100 bits
  Distance between Person 2 and Person 74: 115 bits
  Distance between Person 2 and Person 75: 127 bits
  Distance between Person 2 and Person 76: 111 bits
  Distance between Person 2 and Person 77: 111 bits
  Distance between Person 2 and Person 78: 123 bits
  Distance between Person 2 and Person 79: 119 bits
  Distance between Person 2 and Person 80: 99 bits
  Distance between Person 2 and Person 81: 122 bits
  Distance between Person 2 and Person 82: 96 bits
  Distance between Person 2 and Person 83: 134 bits
  Distance between Person 2 and Person 84: 111 bits
  Distance between Person 2 and Person 85: 126 bits
  Distance between Person 2 and Person 86: 127 bits
  Distance between Person 2 and Person 87: 129 bits
  Distance between Person 2 and Person 88: 126 bits
  Distance between Person 2 and Person 89: 114 bits
  Distance between Person 3 and Person 4: 106 bits
  Distance between Person 3 and Person 5: 121 bits
  Distance between Person 3 and Person 6: 133 bits
  Distance between Person 3 and Person 7: 136 bits
  Distance between Person 3 and Person 8: 114 bits
  Distance between Person 3 and Person 9: 122 bits
  Distance between Person 3 and Person 10: 130 bits
  Distance between Person 3 and Person 11: 129 bits
  Distance between Person 3 and Person 12: 139 bits
  Distance between Person 3 and Person 13: 117 bits
  Distance between Person 3 and Person 14: 130 bits
  Distance between Person 3 and Person 15: 122 bits
  Distance between Person 3 and Person 16: 103 bits
  Distance between Person 3 and Person 17: 133 bits
  Distance between Person 3 and Person 18: 148 bits
  Distance between Person 3 and Person 19: 125 bits
  Distance between Person 3 and Person 20: 121 bits
  Distance between Person 3 and Person 21: 108 bits
  Distance between Person 3 and Person 22: 130 bits
  Distance between Person 3 and Person 23: 120 bits
  Distance between Person 3 and Person 24: 70 bits
  Distance between Person 3 and Person 25: 134 bits
  Distance between Person 3 and Person 26: 126 bits
  Distance between Person 3 and Person 27: 115 bits
  Distance between Person 3 and Person 28: 117 bits
  Distance between Person 3 and Person 29: 108 bits
  Distance between Person 3 and Person 30: 119 bits
  Distance between Person 3 and Person 31: 142 bits
  Distance between Person 3 and Person 32: 75 bits
  Distance between Person 3 and Person 33: 114 bits
  Distance between Person 3 and Person 34: 88 bits
  Distance between Person 3 and Person 35: 101 bits
  Distance between Person 3 and Person 36: 128 bits
  Distance between Person 3 and Person 37: 94 bits
  Distance between Person 3 and Person 38: 126 bits
  Distance between Person 3 and Person 39: 83 bits
  Distance between Person 3 and Person 40: 101 bits
  Distance between Person 3 and Person 41: 121 bits
  Distance between Person 3 and Person 42: 135 bits
  Distance between Person 3 and Person 43: 125 bits
  Distance between Person 3 and Person 44: 126 bits
  Distance between Person 3 and Person 45: 123 bits
  Distance between Person 3 and Person 46: 123 bits
  Distance between Person 3 and Person 47: 132 bits
  Distance between Person 3 and Person 48: 109 bits
  Distance between Person 3 and Person 49: 127 bits
  Distance between Person 3 and Person 50: 114 bits
  Distance between Person 3 and Person 51: 130 bits
  Distance between Person 3 and Person 52: 114 bits
  Distance between Person 3 and Person 53: 115 bits
  Distance between Person 3 and Person 54: 135 bits
  Distance between Person 3 and Person 55: 125 bits
  Distance between Person 3 and Person 56: 134 bits
  Distance between Person 3 and Person 57: 127 bits
  Distance between Person 3 and Person 58: 123 bits
  Distance between Person 3 and Person 59: 130 bits
  Distance between Person 3 and Person 60: 136 bits
  Distance between Person 3 and Person 61: 111 bits
  Distance between Person 3 and Person 62: 115 bits
  Distance between Person 3 and Person 63: 127 bits
  Distance between Person 3 and Person 64: 116 bits
  Distance between Person 3 and Person 65: 129 bits
  Distance between Person 3 and Person 66: 107 bits
  Distance between Person 3 and Person 67: 143 bits
  Distance between Person 3 and Person 68: 127 bits
  Distance between Person 3 and Person 69: 107 bits
  Distance between Person 3 and Person 70: 121 bits
  Distance between Person 3 and Person 71: 136 bits
  Distance between Person 3 and Person 72: 114 bits
  Distance between Person 3 and Person 73: 116 bits
  Distance between Person 3 and Person 74: 127 bits
  Distance between Person 3 and Person 75: 131 bits
  Distance between Person 3 and Person 76: 117 bits
  Distance between Person 3 and Person 77: 117 bits
  Distance between Person 3 and Person 78: 125 bits
  Distance between Person 3 and Person 79: 135 bits
  Distance between Person 3 and Person 80: 109 bits
  Distance between Person 3 and Person 81: 124 bits
  Distance between Person 3 and Person 82: 122 bits
  Distance between Person 3 and Person 83: 134 bits
  Distance between Person 3 and Person 84: 93 bits
  Distance between Person 3 and Person 85: 140 bits
  Distance between Person 3 and Person 86: 117 bits
  Distance between Person 3 and Person 87: 135 bits
  Distance between Person 3 and Person 88: 158 bits
  Distance between Person 3 and Person 89: 126 bits
  Distance between Person 4 and Person 5: 125 bits
  Distance between Person 4 and Person 6: 101 bits
  Distance between Person 4 and Person 7: 98 bits
  Distance between Person 4 and Person 8: 108 bits
  Distance between Person 4 and Person 9: 120 bits
  Distance between Person 4 and Person 10: 130 bits
  Distance between Person 4 and Person 11: 137 bits
  Distance between Person 4 and Person 12: 117 bits
  Distance between Person 4 and Person 13: 127 bits
  Distance between Person 4 and Person 14: 116 bits
  Distance between Person 4 and Person 15: 94 bits
  Distance between Person 4 and Person 16: 97 bits
  Distance between Person 4 and Person 17: 137 bits
  Distance between Person 4 and Person 18: 126 bits
  Distance between Person 4 and Person 19: 97 bits
  Distance between Person 4 and Person 20: 129 bits
  Distance between Person 4 and Person 21: 106 bits
  Distance between Person 4 and Person 22: 122 bits
  Distance between Person 4 and Person 23: 130 bits
  Distance between Person 4 and Person 24: 96 bits
  Distance between Person 4 and Person 25: 126 bits
  Distance between Person 4 and Person 26: 116 bits
  Distance between Person 4 and Person 27: 111 bits
  Distance between Person 4 and Person 28: 119 bits
  Distance between Person 4 and Person 29: 108 bits
  Distance between Person 4 and Person 30: 135 bits
  Distance between Person 4 and Person 31: 136 bits
  Distance between Person 4 and Person 32: 103 bits
  Distance between Person 4 and Person 33: 130 bits
  Distance between Person 4 and Person 34: 76 bits
  Distance between Person 4 and Person 35: 115 bits
  Distance between Person 4 and Person 36: 134 bits
  Distance between Person 4 and Person 37: 132 bits
  Distance between Person 4 and Person 38: 124 bits
  Distance between Person 4 and Person 39: 113 bits
  Distance between Person 4 and Person 40: 49 bits
  Distance between Person 4 and Person 41: 121 bits
  Distance between Person 4 and Person 42: 111 bits
  Distance between Person 4 and Person 43: 95 bits
  Distance between Person 4 and Person 44: 146 bits
  Distance between Person 4 and Person 45: 107 bits
  Distance between Person 4 and Person 46: 125 bits
  Distance between Person 4 and Person 47: 112 bits
  Distance between Person 4 and Person 48: 105 bits
  Distance between Person 4 and Person 49: 133 bits
  Distance between Person 4 and Person 50: 132 bits
  Distance between Person 4 and Person 51: 130 bits
  Distance between Person 4 and Person 52: 116 bits
  Distance between Person 4 and Person 53: 121 bits
  Distance between Person 4 and Person 54: 143 bits
  Distance between Person 4 and Person 55: 95 bits
  Distance between Person 4 and Person 56: 122 bits
  Distance between Person 4 and Person 57: 107 bits
  Distance between Person 4 and Person 58: 129 bits
  Distance between Person 4 and Person 59: 112 bits
  Distance between Person 4 and Person 60: 108 bits
  Distance between Person 4 and Person 61: 123 bits
  Distance between Person 4 and Person 62: 119 bits
  Distance between Person 4 and Person 63: 133 bits
  Distance between Person 4 and Person 64: 134 bits
  Distance between Person 4 and Person 65: 135 bits
  Distance between Person 4 and Person 66: 137 bits
  Distance between Person 4 and Person 67: 121 bits
  Distance between Person 4 and Person 68: 125 bits
  Distance between Person 4 and Person 69: 147 bits
  Distance between Person 4 and Person 70: 107 bits
  Distance between Person 4 and Person 71: 132 bits
  Distance between Person 4 and Person 72: 124 bits
  Distance between Person 4 and Person 73: 132 bits
  Distance between Person 4 and Person 74: 113 bits
  Distance between Person 4 and Person 75: 131 bits
  Distance between Person 4 and Person 76: 129 bits
  Distance between Person 4 and Person 77: 115 bits
  Distance between Person 4 and Person 78: 111 bits
  Distance between Person 4 and Person 79: 125 bits
  Distance between Person 4 and Person 80: 119 bits
  Distance between Person 4 and Person 81: 118 bits
  Distance between Person 4 and Person 82: 142 bits
  Distance between Person 4 and Person 83: 132 bits
  Distance between Person 4 and Person 84: 109 bits
  Distance between Person 4 and Person 85: 116 bits
  Distance between Person 4 and Person 86: 117 bits
  Distance between Person 4 and Person 87: 145 bits
  Distance between Person 4 and Person 88: 140 bits
  Distance between Person 4 and Person 89: 122 bits
  Distance between Person 5 and Person 6: 122 bits
  Distance between Person 5 and Person 7: 135 bits
  Distance between Person 5 and Person 8: 131 bits
  Distance between Person 5 and Person 9: 119 bits
  Distance between Person 5 and Person 10: 127 bits
  Distance between Person 5 and Person 11: 118 bits
  Distance between Person 5 and Person 12: 140 bits
  Distance between Person 5 and Person 13: 124 bits
  Distance between Person 5 and Person 14: 115 bits
  Distance between Person 5 and Person 15: 119 bits
  Distance between Person 5 and Person 16: 122 bits
  Distance between Person 5 and Person 17: 132 bits
  Distance between Person 5 and Person 18: 121 bits
  Distance between Person 5 and Person 19: 134 bits
  Distance between Person 5 and Person 20: 128 bits
  Distance between Person 5 and Person 21: 131 bits
  Distance between Person 5 and Person 22: 115 bits
  Distance between Person 5 and Person 23: 123 bits
  Distance between Person 5 and Person 24: 129 bits
  Distance between Person 5 and Person 25: 119 bits
  Distance between Person 5 and Person 26: 139 bits
  Distance between Person 5 and Person 27: 116 bits
  Distance between Person 5 and Person 28: 110 bits
  Distance between Person 5 and Person 29: 143 bits
  Distance between Person 5 and Person 30: 120 bits
  Distance between Person 5 and Person 31: 127 bits
  Distance between Person 5 and Person 32: 116 bits
  Distance between Person 5 and Person 33: 135 bits
  Distance between Person 5 and Person 34: 127 bits
  Distance between Person 5 and Person 35: 124 bits
  Distance between Person 5 and Person 36: 121 bits
  Distance between Person 5 and Person 37: 127 bits
  Distance between Person 5 and Person 38: 133 bits
  Distance between Person 5 and Person 39: 114 bits
  Distance between Person 5 and Person 40: 126 bits
  Distance between Person 5 and Person 41: 116 bits
  Distance between Person 5 and Person 42: 142 bits
  Distance between Person 5 and Person 43: 136 bits
  Distance between Person 5 and Person 44: 133 bits
  Distance between Person 5 and Person 45: 146 bits
  Distance between Person 5 and Person 46: 122 bits
  Distance between Person 5 and Person 47: 123 bits
  Distance between Person 5 and Person 48: 110 bits
  Distance between Person 5 and Person 49: 120 bits
  Distance between Person 5 and Person 50: 139 bits
  Distance between Person 5 and Person 51: 135 bits
  Distance between Person 5 and Person 52: 121 bits
  Distance between Person 5 and Person 53: 136 bits
  Distance between Person 5 and Person 54: 134 bits
  Distance between Person 5 and Person 55: 134 bits
  Distance between Person 5 and Person 56: 125 bits
  Distance between Person 5 and Person 57: 128 bits
  Distance between Person 5 and Person 58: 128 bits
  Distance between Person 5 and Person 59: 119 bits
  Distance between Person 5 and Person 60: 129 bits
  Distance between Person 5 and Person 61: 130 bits
  Distance between Person 5 and Person 62: 124 bits
  Distance between Person 5 and Person 63: 132 bits
  Distance between Person 5 and Person 64: 131 bits
  Distance between Person 5 and Person 65: 134 bits
  Distance between Person 5 and Person 66: 128 bits
  Distance between Person 5 and Person 67: 136 bits
  Distance between Person 5 and Person 68: 114 bits
  Distance between Person 5 and Person 69: 150 bits
  Distance between Person 5 and Person 70: 132 bits
  Distance between Person 5 and Person 71: 127 bits
  Distance between Person 5 and Person 72: 119 bits
  Distance between Person 5 and Person 73: 123 bits
  Distance between Person 5 and Person 74: 132 bits
  Distance between Person 5 and Person 75: 154 bits
  Distance between Person 5 and Person 76: 130 bits
  Distance between Person 5 and Person 77: 120 bits
  Distance between Person 5 and Person 78: 130 bits
  Distance between Person 5 and Person 79: 128 bits
  Distance between Person 5 and Person 80: 132 bits
  Distance between Person 5 and Person 81: 119 bits
  Distance between Person 5 and Person 82: 149 bits
  Distance between Person 5 and Person 83: 127 bits
  Distance between Person 5 and Person 84: 122 bits
  Distance between Person 5 and Person 85: 125 bits
  Distance between Person 5 and Person 86: 120 bits
  Distance between Person 5 and Person 87: 122 bits
  Distance between Person 5 and Person 88: 137 bits
  Distance between Person 5 and Person 89: 121 bits
  Distance between Person 6 and Person 7: 95 bits
  Distance between Person 6 and Person 8: 121 bits
  Distance between Person 6 and Person 9: 125 bits
  Distance between Person 6 and Person 10: 137 bits
  Distance between Person 6 and Person 11: 112 bits
  Distance between Person 6 and Person 12: 122 bits
  Distance between Person 6 and Person 13: 146 bits
  Distance between Person 6 and Person 14: 113 bits
  Distance between Person 6 and Person 15: 103 bits
  Distance between Person 6 and Person 16: 116 bits
  Distance between Person 6 and Person 17: 120 bits
  Distance between Person 6 and Person 18: 99 bits
  Distance between Person 6 and Person 19: 82 bits
  Distance between Person 6 and Person 20: 118 bits
  Distance between Person 6 and Person 21: 81 bits
  Distance between Person 6 and Person 22: 137 bits
  Distance between Person 6 and Person 23: 125 bits
  Distance between Person 6 and Person 24: 137 bits
  Distance between Person 6 and Person 25: 133 bits
  Distance between Person 6 and Person 26: 139 bits
  Distance between Person 6 and Person 27: 132 bits
  Distance between Person 6 and Person 28: 134 bits
  Distance between Person 6 and Person 29: 139 bits
  Distance between Person 6 and Person 30: 146 bits
  Distance between Person 6 and Person 31: 117 bits
  Distance between Person 6 and Person 32: 88 bits
  Distance between Person 6 and Person 33: 127 bits
  Distance between Person 6 and Person 34: 111 bits
  Distance between Person 6 and Person 35: 142 bits
  Distance between Person 6 and Person 36: 131 bits
  Distance between Person 6 and Person 37: 127 bits
  Distance between Person 6 and Person 38: 107 bits
  Distance between Person 6 and Person 39: 120 bits
  Distance between Person 6 and Person 40: 84 bits
  Distance between Person 6 and Person 41: 128 bits
  Distance between Person 6 and Person 42: 128 bits
  Distance between Person 6 and Person 43: 128 bits
  Distance between Person 6 and Person 44: 145 bits
  Distance between Person 6 and Person 45: 114 bits
  Distance between Person 6 and Person 46: 136 bits
  Distance between Person 6 and Person 47: 121 bits
  Distance between Person 6 and Person 48: 124 bits
  Distance between Person 6 and Person 49: 134 bits
  Distance between Person 6 and Person 50: 131 bits
  Distance between Person 6 and Person 51: 137 bits
  Distance between Person 6 and Person 52: 131 bits
  Distance between Person 6 and Person 53: 134 bits
  Distance between Person 6 and Person 54: 126 bits
  Distance between Person 6 and Person 55: 108 bits
  Distance between Person 6 and Person 56: 137 bits
  Distance between Person 6 and Person 57: 122 bits
  Distance between Person 6 and Person 58: 118 bits
  Distance between Person 6 and Person 59: 115 bits
  Distance between Person 6 and Person 60: 95 bits
  Distance between Person 6 and Person 61: 122 bits
  Distance between Person 6 and Person 62: 92 bits
  Distance between Person 6 and Person 63: 126 bits
  Distance between Person 6 and Person 64: 125 bits
  Distance between Person 6 and Person 65: 132 bits
  Distance between Person 6 and Person 66: 126 bits
  Distance between Person 6 and Person 67: 124 bits
  Distance between Person 6 and Person 68: 130 bits
  Distance between Person 6 and Person 69: 146 bits
  Distance between Person 6 and Person 70: 118 bits
  Distance between Person 6 and Person 71: 85 bits
  Distance between Person 6 and Person 72: 145 bits
  Distance between Person 6 and Person 73: 117 bits
  Distance between Person 6 and Person 74: 132 bits
  Distance between Person 6 and Person 75: 130 bits
  Distance between Person 6 and Person 76: 118 bits
  Distance between Person 6 and Person 77: 128 bits
  Distance between Person 6 and Person 78: 108 bits
  Distance between Person 6 and Person 79: 150 bits
  Distance between Person 6 and Person 80: 124 bits
  Distance between Person 6 and Person 81: 119 bits
  Distance between Person 6 and Person 82: 147 bits
  Distance between Person 6 and Person 83: 133 bits
  Distance between Person 6 and Person 84: 128 bits
  Distance between Person 6 and Person 85: 131 bits
  Distance between Person 6 and Person 86: 110 bits
  Distance between Person 6 and Person 87: 128 bits
  Distance between Person 6 and Person 88: 119 bits
  Distance between Person 6 and Person 89: 117 bits
  Distance between Person 7 and Person 8: 128 bits
  Distance between Person 7 and Person 9: 132 bits
  Distance between Person 7 and Person 10: 140 bits
  Distance between Person 7 and Person 11: 139 bits
  Distance between Person 7 and Person 12: 133 bits
  Distance between Person 7 and Person 13: 137 bits
  Distance between Person 7 and Person 14: 122 bits
  Distance between Person 7 and Person 15: 112 bits
  Distance between Person 7 and Person 16: 129 bits
  Distance between Person 7 and Person 17: 141 bits
  Distance between Person 7 and Person 18: 106 bits
  Distance between Person 7 and Person 19: 101 bits
  Distance between Person 7 and Person 20: 113 bits
  Distance between Person 7 and Person 21: 112 bits
  Distance between Person 7 and Person 22: 134 bits
  Distance between Person 7 and Person 23: 108 bits
  Distance between Person 7 and Person 24: 134 bits
  Distance between Person 7 and Person 25: 134 bits
  Distance between Person 7 and Person 26: 132 bits
  Distance between Person 7 and Person 27: 123 bits
  Distance between Person 7 and Person 28: 131 bits
  Distance between Person 7 and Person 29: 122 bits
  Distance between Person 7 and Person 30: 139 bits
  Distance between Person 7 and Person 31: 136 bits
  Distance between Person 7 and Person 32: 131 bits
  Distance between Person 7 and Person 33: 136 bits
  Distance between Person 7 and Person 34: 124 bits
  Distance between Person 7 and Person 35: 127 bits
  Distance between Person 7 and Person 36: 128 bits
  Distance between Person 7 and Person 37: 114 bits
  Distance between Person 7 and Person 38: 118 bits
  Distance between Person 7 and Person 39: 145 bits
  Distance between Person 7 and Person 40: 95 bits
  Distance between Person 7 and Person 41: 131 bits
  Distance between Person 7 and Person 42: 125 bits
  Distance between Person 7 and Person 43: 101 bits
  Distance between Person 7 and Person 44: 136 bits
  Distance between Person 7 and Person 45: 121 bits
  Distance between Person 7 and Person 46: 127 bits
  Distance between Person 7 and Person 47: 122 bits
  Distance between Person 7 and Person 48: 113 bits
  Distance between Person 7 and Person 49: 123 bits
  Distance between Person 7 and Person 50: 112 bits
  Distance between Person 7 and Person 51: 122 bits
  Distance between Person 7 and Person 52: 154 bits
  Distance between Person 7 and Person 53: 133 bits
  Distance between Person 7 and Person 54: 129 bits
  Distance between Person 7 and Person 55: 127 bits
  Distance between Person 7 and Person 56: 114 bits
  Distance between Person 7 and Person 57: 125 bits
  Distance between Person 7 and Person 58: 139 bits
  Distance between Person 7 and Person 59: 138 bits
  Distance between Person 7 and Person 60: 66 bits
  Distance between Person 7 and Person 61: 143 bits
  Distance between Person 7 and Person 62: 115 bits
  Distance between Person 7 and Person 63: 127 bits
  Distance between Person 7 and Person 64: 136 bits
  Distance between Person 7 and Person 65: 133 bits
  Distance between Person 7 and Person 66: 131 bits
  Distance between Person 7 and Person 67: 127 bits
  Distance between Person 7 and Person 68: 141 bits
  Distance between Person 7 and Person 69: 139 bits
  Distance between Person 7 and Person 70: 111 bits
  Distance between Person 7 and Person 71: 128 bits
  Distance between Person 7 and Person 72: 128 bits
  Distance between Person 7 and Person 73: 128 bits
  Distance between Person 7 and Person 74: 127 bits
  Distance between Person 7 and Person 75: 121 bits
  Distance between Person 7 and Person 76: 141 bits
  Distance between Person 7 and Person 77: 115 bits
  Distance between Person 7 and Person 78: 121 bits
  Distance between Person 7 and Person 79: 145 bits
  Distance between Person 7 and Person 80: 127 bits
  Distance between Person 7 and Person 81: 120 bits
  Distance between Person 7 and Person 82: 126 bits
  Distance between Person 7 and Person 83: 128 bits
  Distance between Person 7 and Person 84: 135 bits
  Distance between Person 7 and Person 85: 98 bits
  Distance between Person 7 and Person 86: 131 bits
  Distance between Person 7 and Person 87: 141 bits
  Distance between Person 7 and Person 88: 144 bits
  Distance between Person 7 and Person 89: 124 bits
  Distance between Person 8 and Person 9: 128 bits
  Distance between Person 8 and Person 10: 122 bits
  Distance between Person 8 and Person 11: 119 bits
  Distance between Person 8 and Person 12: 105 bits
  Distance between Person 8 and Person 13: 129 bits
  Distance between Person 8 and Person 14: 118 bits
  Distance between Person 8 and Person 15: 114 bits
  Distance between Person 8 and Person 16: 115 bits
  Distance between Person 8 and Person 17: 131 bits
  Distance between Person 8 and Person 18: 128 bits
  Distance between Person 8 and Person 19: 137 bits
  Distance between Person 8 and Person 20: 125 bits
  Distance between Person 8 and Person 21: 124 bits
  Distance between Person 8 and Person 22: 116 bits
  Distance between Person 8 and Person 23: 134 bits
  Distance between Person 8 and Person 24: 114 bits
  Distance between Person 8 and Person 25: 134 bits
  Distance between Person 8 and Person 26: 122 bits
  Distance between Person 8 and Person 27: 129 bits
  Distance between Person 8 and Person 28: 117 bits
  Distance between Person 8 and Person 29: 104 bits
  Distance between Person 8 and Person 30: 137 bits
  Distance between Person 8 and Person 31: 136 bits
  Distance between Person 8 and Person 32: 111 bits
  Distance between Person 8 and Person 33: 114 bits
  Distance between Person 8 and Person 34: 110 bits
  Distance between Person 8 and Person 35: 123 bits
  Distance between Person 8 and Person 36: 122 bits
  Distance between Person 8 and Person 37: 118 bits
  Distance between Person 8 and Person 38: 120 bits
  Distance between Person 8 and Person 39: 113 bits
  Distance between Person 8 and Person 40: 89 bits
  Distance between Person 8 and Person 41: 147 bits
  Distance between Person 8 and Person 42: 127 bits
  Distance between Person 8 and Person 43: 121 bits
  Distance between Person 8 and Person 44: 114 bits
  Distance between Person 8 and Person 45: 131 bits
  Distance between Person 8 and Person 46: 121 bits
  Distance between Person 8 and Person 47: 136 bits
  Distance between Person 8 and Person 48: 125 bits
  Distance between Person 8 and Person 49: 145 bits
  Distance between Person 8 and Person 50: 130 bits
  Distance between Person 8 and Person 51: 110 bits
  Distance between Person 8 and Person 52: 126 bits
  Distance between Person 8 and Person 53: 119 bits
  Distance between Person 8 and Person 54: 111 bits
  Distance between Person 8 and Person 55: 141 bits
  Distance between Person 8 and Person 56: 118 bits
  Distance between Person 8 and Person 57: 133 bits
  Distance between Person 8 and Person 58: 125 bits
  Distance between Person 8 and Person 59: 126 bits
  Distance between Person 8 and Person 60: 122 bits
  Distance between Person 8 and Person 61: 137 bits
  Distance between Person 8 and Person 62: 139 bits
  Distance between Person 8 and Person 63: 121 bits
  Distance between Person 8 and Person 64: 138 bits
  Distance between Person 8 and Person 65: 125 bits
  Distance between Person 8 and Person 66: 133 bits
  Distance between Person 8 and Person 67: 109 bits
  Distance between Person 8 and Person 68: 103 bits
  Distance between Person 8 and Person 69: 127 bits
  Distance between Person 8 and Person 70: 127 bits
  Distance between Person 8 and Person 71: 136 bits
  Distance between Person 8 and Person 72: 138 bits
  Distance between Person 8 and Person 73: 96 bits
  Distance between Person 8 and Person 74: 127 bits
  Distance between Person 8 and Person 75: 125 bits
  Distance between Person 8 and Person 76: 117 bits
  Distance between Person 8 and Person 77: 115 bits
  Distance between Person 8 and Person 78: 143 bits
  Distance between Person 8 and Person 79: 131 bits
  Distance between Person 8 and Person 80: 125 bits
  Distance between Person 8 and Person 81: 118 bits
  Distance between Person 8 and Person 82: 126 bits
  Distance between Person 8 and Person 83: 144 bits
  Distance between Person 8 and Person 84: 111 bits
  Distance between Person 8 and Person 85: 116 bits
  Distance between Person 8 and Person 86: 131 bits
  Distance between Person 8 and Person 87: 131 bits
  Distance between Person 8 and Person 88: 134 bits
  Distance between Person 8 and Person 89: 122 bits
  Distance between Person 9 and Person 10: 104 bits
  Distance between Person 9 and Person 11: 115 bits
  Distance between Person 9 and Person 12: 117 bits
  Distance between Person 9 and Person 13: 127 bits
  Distance between Person 9 and Person 14: 126 bits
  Distance between Person 9 and Person 15: 120 bits
  Distance between Person 9 and Person 16: 119 bits
  Distance between Person 9 and Person 17: 117 bits
  Distance between Person 9 and Person 18: 102 bits
  Distance between Person 9 and Person 19: 123 bits
  Distance between Person 9 and Person 20: 117 bits
  Distance between Person 9 and Person 21: 130 bits
  Distance between Person 9 and Person 22: 128 bits
  Distance between Person 9 and Person 23: 108 bits
  Distance between Person 9 and Person 24: 118 bits
  Distance between Person 9 and Person 25: 98 bits
  Distance between Person 9 and Person 26: 116 bits
  Distance between Person 9 and Person 27: 133 bits
  Distance between Person 9 and Person 28: 121 bits
  Distance between Person 9 and Person 29: 116 bits
  Distance between Person 9 and Person 30: 127 bits
  Distance between Person 9 and Person 31: 128 bits
  Distance between Person 9 and Person 32: 111 bits
  Distance between Person 9 and Person 33: 116 bits
  Distance between Person 9 and Person 34: 118 bits
  Distance between Person 9 and Person 35: 131 bits
  Distance between Person 9 and Person 36: 144 bits
  Distance between Person 9 and Person 37: 114 bits
  Distance between Person 9 and Person 38: 124 bits
  Distance between Person 9 and Person 39: 107 bits
  Distance between Person 9 and Person 40: 117 bits
  Distance between Person 9 and Person 41: 117 bits
  Distance between Person 9 and Person 42: 103 bits
  Distance between Person 9 and Person 43: 109 bits
  Distance between Person 9 and Person 44: 132 bits
  Distance between Person 9 and Person 45: 119 bits
  Distance between Person 9 and Person 46: 119 bits
  Distance between Person 9 and Person 47: 134 bits
  Distance between Person 9 and Person 48: 115 bits
  Distance between Person 9 and Person 49: 137 bits
  Distance between Person 9 and Person 50: 128 bits
  Distance between Person 9 and Person 51: 104 bits
  Distance between Person 9 and Person 52: 140 bits
  Distance between Person 9 and Person 53: 137 bits
  Distance between Person 9 and Person 54: 111 bits
  Distance between Person 9 and Person 55: 123 bits
  Distance between Person 9 and Person 56: 150 bits
  Distance between Person 9 and Person 57: 105 bits
  Distance between Person 9 and Person 58: 131 bits
  Distance between Person 9 and Person 59: 94 bits
  Distance between Person 9 and Person 60: 130 bits
  Distance between Person 9 and Person 61: 115 bits
  Distance between Person 9 and Person 62: 101 bits
  Distance between Person 9 and Person 63: 105 bits
  Distance between Person 9 and Person 64: 124 bits
  Distance between Person 9 and Person 65: 151 bits
  Distance between Person 9 and Person 66: 131 bits
  Distance between Person 9 and Person 67: 115 bits
  Distance between Person 9 and Person 68: 133 bits
  Distance between Person 9 and Person 69: 127 bits
  Distance between Person 9 and Person 70: 119 bits
  Distance between Person 9 and Person 71: 138 bits
  Distance between Person 9 and Person 72: 120 bits
  Distance between Person 9 and Person 73: 114 bits
  Distance between Person 9 and Person 74: 127 bits
  Distance between Person 9 and Person 75: 141 bits
  Distance between Person 9 and Person 76: 123 bits
  Distance between Person 9 and Person 77: 111 bits
  Distance between Person 9 and Person 78: 65 bits
  Distance between Person 9 and Person 79: 139 bits
  Distance between Person 9 and Person 80: 129 bits
  Distance between Person 9 and Person 81: 92 bits
  Distance between Person 9 and Person 82: 120 bits
  Distance between Person 9 and Person 83: 126 bits
  Distance between Person 9 and Person 84: 123 bits
  Distance between Person 9 and Person 85: 118 bits
  Distance between Person 9 and Person 86: 119 bits
  Distance between Person 9 and Person 87: 127 bits
  Distance between Person 9 and Person 88: 120 bits
  Distance between Person 9 and Person 89: 128 bits
  Distance between Person 10 and Person 11: 143 bits
  Distance between Person 10 and Person 12: 139 bits
  Distance between Person 10 and Person 13: 139 bits
  Distance between Person 10 and Person 14: 138 bits
  Distance between Person 10 and Person 15: 128 bits
  Distance between Person 10 and Person 16: 129 bits
  Distance between Person 10 and Person 17: 133 bits
  Distance between Person 10 and Person 18: 122 bits
  Distance between Person 10 and Person 19: 119 bits
  Distance between Person 10 and Person 20: 151 bits
  Distance between Person 10 and Person 21: 150 bits
  Distance between Person 10 and Person 22: 120 bits
  Distance between Person 10 and Person 23: 128 bits
  Distance between Person 10 and Person 24: 144 bits
  Distance between Person 10 and Person 25: 108 bits
  Distance between Person 10 and Person 26: 130 bits
  Distance between Person 10 and Person 27: 135 bits
  Distance between Person 10 and Person 28: 145 bits
  Distance between Person 10 and Person 29: 116 bits
  Distance between Person 10 and Person 30: 121 bits
  Distance between Person 10 and Person 31: 134 bits
  Distance between Person 10 and Person 32: 143 bits
  Distance between Person 10 and Person 33: 124 bits
  Distance between Person 10 and Person 34: 132 bits
  Distance between Person 10 and Person 35: 127 bits
  Distance between Person 10 and Person 36: 120 bits
  Distance between Person 10 and Person 37: 118 bits
  Distance between Person 10 and Person 38: 138 bits
  Distance between Person 10 and Person 39: 109 bits
  Distance between Person 10 and Person 40: 129 bits
  Distance between Person 10 and Person 41: 125 bits
  Distance between Person 10 and Person 42: 123 bits
  Distance between Person 10 and Person 43: 137 bits
  Distance between Person 10 and Person 44: 128 bits
  Distance between Person 10 and Person 45: 125 bits
  Distance between Person 10 and Person 46: 127 bits
  Distance between Person 10 and Person 47: 120 bits
  Distance between Person 10 and Person 48: 129 bits
  Distance between Person 10 and Person 49: 123 bits
  Distance between Person 10 and Person 50: 134 bits
  Distance between Person 10 and Person 51: 112 bits
  Distance between Person 10 and Person 52: 146 bits
  Distance between Person 10 and Person 53: 149 bits
  Distance between Person 10 and Person 54: 137 bits
  Distance between Person 10 and Person 55: 137 bits
  Distance between Person 10 and Person 56: 126 bits
  Distance between Person 10 and Person 57: 127 bits
  Distance between Person 10 and Person 58: 137 bits
  Distance between Person 10 and Person 59: 122 bits
  Distance between Person 10 and Person 60: 144 bits
  Distance between Person 10 and Person 61: 143 bits
  Distance between Person 10 and Person 62: 137 bits
  Distance between Person 10 and Person 63: 117 bits
  Distance between Person 10 and Person 64: 112 bits
  Distance between Person 10 and Person 65: 123 bits
  Distance between Person 10 and Person 66: 131 bits
  Distance between Person 10 and Person 67: 125 bits
  Distance between Person 10 and Person 68: 115 bits
  Distance between Person 10 and Person 69: 121 bits
  Distance between Person 10 and Person 70: 139 bits
  Distance between Person 10 and Person 71: 138 bits
  Distance between Person 10 and Person 72: 134 bits
  Distance between Person 10 and Person 73: 132 bits
  Distance between Person 10 and Person 74: 131 bits
  Distance between Person 10 and Person 75: 121 bits
  Distance between Person 10 and Person 76: 129 bits
  Distance between Person 10 and Person 77: 121 bits
  Distance between Person 10 and Person 78: 145 bits
  Distance between Person 10 and Person 79: 99 bits
  Distance between Person 10 and Person 80: 133 bits
  Distance between Person 10 and Person 81: 114 bits
  Distance between Person 10 and Person 82: 110 bits
  Distance between Person 10 and Person 83: 140 bits
  Distance between Person 10 and Person 84: 129 bits
  Distance between Person 10 and Person 85: 138 bits
  Distance between Person 10 and Person 86: 143 bits
  Distance between Person 10 and Person 87: 125 bits
  Distance between Person 10 and Person 88: 132 bits
  Distance between Person 10 and Person 89: 128 bits
  Distance between Person 11 and Person 12: 130 bits
  Distance between Person 11 and Person 13: 78 bits
  Distance between Person 11 and Person 14: 89 bits
  Distance between Person 11 and Person 15: 121 bits
  Distance between Person 11 and Person 16: 126 bits
  Distance between Person 11 and Person 17: 108 bits
  Distance between Person 11 and Person 18: 137 bits
  Distance between Person 11 and Person 19: 120 bits
  Distance between Person 11 and Person 20: 90 bits
  Distance between Person 11 and Person 21: 123 bits
  Distance between Person 11 and Person 22: 131 bits
  Distance between Person 11 and Person 23: 131 bits
  Distance between Person 11 and Person 24: 123 bits
  Distance between Person 11 and Person 25: 131 bits
  Distance between Person 11 and Person 26: 129 bits
  Distance between Person 11 and Person 27: 136 bits
  Distance between Person 11 and Person 28: 112 bits
  Distance between Person 11 and Person 29: 129 bits
  Distance between Person 11 and Person 30: 116 bits
  Distance between Person 11 and Person 31: 115 bits
  Distance between Person 11 and Person 32: 108 bits
  Distance between Person 11 and Person 33: 125 bits
  Distance between Person 11 and Person 34: 127 bits
  Distance between Person 11 and Person 35: 112 bits
  Distance between Person 11 and Person 36: 141 bits
  Distance between Person 11 and Person 37: 119 bits
  Distance between Person 11 and Person 38: 105 bits
  Distance between Person 11 and Person 39: 112 bits
  Distance between Person 11 and Person 40: 118 bits
  Distance between Person 11 and Person 41: 134 bits
  Distance between Person 11 and Person 42: 112 bits
  Distance between Person 11 and Person 43: 140 bits
  Distance between Person 11 and Person 44: 125 bits
  Distance between Person 11 and Person 45: 126 bits
  Distance between Person 11 and Person 46: 118 bits
  Distance between Person 11 and Person 47: 119 bits
  Distance between Person 11 and Person 48: 136 bits
  Distance between Person 11 and Person 49: 112 bits
  Distance between Person 11 and Person 50: 131 bits
  Distance between Person 11 and Person 51: 151 bits
  Distance between Person 11 and Person 52: 153 bits
  Distance between Person 11 and Person 53: 130 bits
  Distance between Person 11 and Person 54: 122 bits
  Distance between Person 11 and Person 55: 124 bits
  Distance between Person 11 and Person 56: 135 bits
  Distance between Person 11 and Person 57: 108 bits
  Distance between Person 11 and Person 58: 132 bits
  Distance between Person 11 and Person 59: 99 bits
  Distance between Person 11 and Person 60: 119 bits
  Distance between Person 11 and Person 61: 110 bits
  Distance between Person 11 and Person 62: 130 bits
  Distance between Person 11 and Person 63: 142 bits
  Distance between Person 11 and Person 64: 129 bits
  Distance between Person 11 and Person 65: 116 bits
  Distance between Person 11 and Person 66: 134 bits
  Distance between Person 11 and Person 67: 120 bits
  Distance between Person 11 and Person 68: 116 bits
  Distance between Person 11 and Person 69: 124 bits
  Distance between Person 11 and Person 70: 130 bits
  Distance between Person 11 and Person 71: 111 bits
  Distance between Person 11 and Person 72: 119 bits
  Distance between Person 11 and Person 73: 93 bits
  Distance between Person 11 and Person 74: 122 bits
  Distance between Person 11 and Person 75: 146 bits
  Distance between Person 11 and Person 76: 126 bits
  Distance between Person 11 and Person 77: 124 bits
  Distance between Person 11 and Person 78: 114 bits
  Distance between Person 11 and Person 79: 126 bits
  Distance between Person 11 and Person 80: 116 bits
  Distance between Person 11 and Person 81: 135 bits
  Distance between Person 11 and Person 82: 145 bits
  Distance between Person 11 and Person 83: 151 bits
  Distance between Person 11 and Person 84: 122 bits
  Distance between Person 11 and Person 85: 133 bits
  Distance between Person 11 and Person 86: 118 bits
  Distance between Person 11 and Person 87: 122 bits
  Distance between Person 11 and Person 88: 119 bits
  Distance between Person 11 and Person 89: 119 bits
  Distance between Person 12 and Person 13: 116 bits
  Distance between Person 12 and Person 14: 121 bits
  Distance between Person 12 and Person 15: 127 bits
  Distance between Person 12 and Person 16: 116 bits
  Distance between Person 12 and Person 17: 78 bits
  Distance between Person 12 and Person 18: 121 bits
  Distance between Person 12 and Person 19: 124 bits
  Distance between Person 12 and Person 20: 118 bits
  Distance between Person 12 and Person 21: 133 bits
  Distance between Person 12 and Person 22: 123 bits
  Distance between Person 12 and Person 23: 131 bits
  Distance between Person 12 and Person 24: 121 bits
  Distance between Person 12 and Person 25: 139 bits
  Distance between Person 12 and Person 26: 115 bits
  Distance between Person 12 and Person 27: 130 bits
  Distance between Person 12 and Person 28: 128 bits
  Distance between Person 12 and Person 29: 139 bits
  Distance between Person 12 and Person 30: 124 bits
  Distance between Person 12 and Person 31: 113 bits
  Distance between Person 12 and Person 32: 130 bits
  Distance between Person 12 and Person 33: 137 bits
  Distance between Person 12 and Person 34: 131 bits
  Distance between Person 12 and Person 35: 140 bits
  Distance between Person 12 and Person 36: 129 bits
  Distance between Person 12 and Person 37: 143 bits
  Distance between Person 12 and Person 38: 125 bits
  Distance between Person 12 and Person 39: 146 bits
  Distance between Person 12 and Person 40: 118 bits
  Distance between Person 12 and Person 41: 134 bits
  Distance between Person 12 and Person 42: 126 bits
  Distance between Person 12 and Person 43: 126 bits
  Distance between Person 12 and Person 44: 127 bits
  Distance between Person 12 and Person 45: 118 bits
  Distance between Person 12 and Person 46: 124 bits
  Distance between Person 12 and Person 47: 131 bits
  Distance between Person 12 and Person 48: 114 bits
  Distance between Person 12 and Person 49: 134 bits
  Distance between Person 12 and Person 50: 127 bits
  Distance between Person 12 and Person 51: 111 bits
  Distance between Person 12 and Person 52: 129 bits
  Distance between Person 12 and Person 53: 116 bits
  Distance between Person 12 and Person 54: 116 bits
  Distance between Person 12 and Person 55: 118 bits
  Distance between Person 12 and Person 56: 129 bits
  Distance between Person 12 and Person 57: 120 bits
  Distance between Person 12 and Person 58: 126 bits
  Distance between Person 12 and Person 59: 133 bits
  Distance between Person 12 and Person 60: 137 bits
  Distance between Person 12 and Person 61: 112 bits
  Distance between Person 12 and Person 62: 122 bits
  Distance between Person 12 and Person 63: 124 bits
  Distance between Person 12 and Person 64: 125 bits
  Distance between Person 12 and Person 65: 142 bits
  Distance between Person 12 and Person 66: 140 bits
  Distance between Person 12 and Person 67: 126 bits
  Distance between Person 12 and Person 68: 134 bits
  Distance between Person 12 and Person 69: 122 bits
  Distance between Person 12 and Person 70: 118 bits
  Distance between Person 12 and Person 71: 131 bits
  Distance between Person 12 and Person 72: 125 bits
  Distance between Person 12 and Person 73: 115 bits
  Distance between Person 12 and Person 74: 142 bits
  Distance between Person 12 and Person 75: 132 bits
  Distance between Person 12 and Person 76: 124 bits
  Distance between Person 12 and Person 77: 130 bits
  Distance between Person 12 and Person 78: 120 bits
  Distance between Person 12 and Person 79: 132 bits
  Distance between Person 12 and Person 80: 120 bits
  Distance between Person 12 and Person 81: 125 bits
  Distance between Person 12 and Person 82: 131 bits
  Distance between Person 12 and Person 83: 133 bits
  Distance between Person 12 and Person 84: 120 bits
  Distance between Person 12 and Person 85: 115 bits
  Distance between Person 12 and Person 86: 122 bits
  Distance between Person 12 and Person 87: 126 bits
  Distance between Person 12 and Person 88: 125 bits
  Distance between Person 12 and Person 89: 107 bits
  Distance between Person 13 and Person 14: 91 bits
  Distance between Person 13 and Person 15: 111 bits
  Distance between Person 13 and Person 16: 144 bits
  Distance between Person 13 and Person 17: 102 bits
  Distance between Person 13 and Person 18: 153 bits
  Distance between Person 13 and Person 19: 134 bits
  Distance between Person 13 and Person 20: 110 bits
  Distance between Person 13 and Person 21: 125 bits
  Distance between Person 13 and Person 22: 117 bits
  Distance between Person 13 and Person 23: 137 bits
  Distance between Person 13 and Person 24: 105 bits
  Distance between Person 13 and Person 25: 133 bits
  Distance between Person 13 and Person 26: 109 bits
  Distance between Person 13 and Person 27: 142 bits
  Distance between Person 13 and Person 28: 112 bits
  Distance between Person 13 and Person 29: 143 bits
  Distance between Person 13 and Person 30: 112 bits
  Distance between Person 13 and Person 31: 117 bits
  Distance between Person 13 and Person 32: 132 bits
  Distance between Person 13 and Person 33: 125 bits
  Distance between Person 13 and Person 34: 133 bits
  Distance between Person 13 and Person 35: 144 bits
  Distance between Person 13 and Person 36: 137 bits
  Distance between Person 13 and Person 37: 129 bits
  Distance between Person 13 and Person 38: 99 bits
  Distance between Person 13 and Person 39: 124 bits
  Distance between Person 13 and Person 40: 138 bits
  Distance between Person 13 and Person 41: 124 bits
  Distance between Person 13 and Person 42: 118 bits
  Distance between Person 13 and Person 43: 114 bits
  Distance between Person 13 and Person 44: 125 bits
  Distance between Person 13 and Person 45: 124 bits
  Distance between Person 13 and Person 46: 132 bits
  Distance between Person 13 and Person 47: 127 bits
  Distance between Person 13 and Person 48: 132 bits
  Distance between Person 13 and Person 49: 112 bits
  Distance between Person 13 and Person 50: 151 bits
  Distance between Person 13 and Person 51: 111 bits
  Distance between Person 13 and Person 52: 123 bits
  Distance between Person 13 and Person 53: 106 bits
  Distance between Person 13 and Person 54: 118 bits
  Distance between Person 13 and Person 55: 112 bits
  Distance between Person 13 and Person 56: 125 bits
  Distance between Person 13 and Person 57: 134 bits
  Distance between Person 13 and Person 58: 138 bits
  Distance between Person 13 and Person 59: 141 bits
  Distance between Person 13 and Person 60: 143 bits
  Distance between Person 13 and Person 61: 112 bits
  Distance between Person 13 and Person 62: 146 bits
  Distance between Person 13 and Person 63: 152 bits
  Distance between Person 13 and Person 64: 129 bits
  Distance between Person 13 and Person 65: 138 bits
  Distance between Person 13 and Person 66: 122 bits
  Distance between Person 13 and Person 67: 128 bits
  Distance between Person 13 and Person 68: 132 bits
  Distance between Person 13 and Person 69: 128 bits
  Distance between Person 13 and Person 70: 116 bits
  Distance between Person 13 and Person 71: 157 bits
  Distance between Person 13 and Person 72: 113 bits
  Distance between Person 13 and Person 73: 85 bits
  Distance between Person 13 and Person 74: 106 bits
  Distance between Person 13 and Person 75: 144 bits
  Distance between Person 13 and Person 76: 150 bits
  Distance between Person 13 and Person 77: 144 bits
  Distance between Person 13 and Person 78: 124 bits
  Distance between Person 13 and Person 79: 136 bits
  Distance between Person 13 and Person 80: 96 bits
  Distance between Person 13 and Person 81: 145 bits
  Distance between Person 13 and Person 82: 129 bits
  Distance between Person 13 and Person 83: 137 bits
  Distance between Person 13 and Person 84: 142 bits
  Distance between Person 13 and Person 85: 139 bits
  Distance between Person 13 and Person 86: 130 bits
  Distance between Person 13 and Person 87: 136 bits
  Distance between Person 13 and Person 88: 137 bits
  Distance between Person 13 and Person 89: 127 bits
  Distance between Person 14 and Person 15: 92 bits
  Distance between Person 14 and Person 16: 117 bits
  Distance between Person 14 and Person 17: 109 bits
  Distance between Person 14 and Person 18: 130 bits
  Distance between Person 14 and Person 19: 131 bits
  Distance between Person 14 and Person 20: 93 bits
  Distance between Person 14 and Person 21: 126 bits
  Distance between Person 14 and Person 22: 128 bits
  Distance between Person 14 and Person 23: 130 bits
  Distance between Person 14 and Person 24: 132 bits
  Distance between Person 14 and Person 25: 120 bits
  Distance between Person 14 and Person 26: 150 bits
  Distance between Person 14 and Person 27: 117 bits
  Distance between Person 14 and Person 28: 107 bits
  Distance between Person 14 and Person 29: 136 bits
  Distance between Person 14 and Person 30: 121 bits
  Distance between Person 14 and Person 31: 120 bits
  Distance between Person 14 and Person 32: 127 bits
  Distance between Person 14 and Person 33: 126 bits
  Distance between Person 14 and Person 34: 116 bits
  Distance between Person 14 and Person 35: 119 bits
  Distance between Person 14 and Person 36: 150 bits
  Distance between Person 14 and Person 37: 132 bits
  Distance between Person 14 and Person 38: 86 bits
  Distance between Person 14 and Person 39: 135 bits
  Distance between Person 14 and Person 40: 129 bits
  Distance between Person 14 and Person 41: 125 bits
  Distance between Person 14 and Person 42: 119 bits
  Distance between Person 14 and Person 43: 137 bits
  Distance between Person 14 and Person 44: 130 bits
  Distance between Person 14 and Person 45: 111 bits
  Distance between Person 14 and Person 46: 113 bits
  Distance between Person 14 and Person 47: 124 bits
  Distance between Person 14 and Person 48: 139 bits
  Distance between Person 14 and Person 49: 139 bits
  Distance between Person 14 and Person 50: 126 bits
  Distance between Person 14 and Person 51: 138 bits
  Distance between Person 14 and Person 52: 130 bits
  Distance between Person 14 and Person 53: 109 bits
  Distance between Person 14 and Person 54: 115 bits
  Distance between Person 14 and Person 55: 135 bits
  Distance between Person 14 and Person 56: 126 bits
  Distance between Person 14 and Person 57: 131 bits
  Distance between Person 14 and Person 58: 129 bits
  Distance between Person 14 and Person 59: 116 bits
  Distance between Person 14 and Person 60: 118 bits
  Distance between Person 14 and Person 61: 123 bits
  Distance between Person 14 and Person 62: 121 bits
  Distance between Person 14 and Person 63: 117 bits
  Distance between Person 14 and Person 64: 150 bits
  Distance between Person 14 and Person 65: 129 bits
  Distance between Person 14 and Person 66: 133 bits
  Distance between Person 14 and Person 67: 125 bits
  Distance between Person 14 and Person 68: 99 bits
  Distance between Person 14 and Person 69: 135 bits
  Distance between Person 14 and Person 70: 109 bits
  Distance between Person 14 and Person 71: 144 bits
  Distance between Person 14 and Person 72: 106 bits
  Distance between Person 14 and Person 73: 94 bits
  Distance between Person 14 and Person 74: 119 bits
  Distance between Person 14 and Person 75: 131 bits
  Distance between Person 14 and Person 76: 123 bits
  Distance between Person 14 and Person 77: 145 bits
  Distance between Person 14 and Person 78: 127 bits
  Distance between Person 14 and Person 79: 137 bits
  Distance between Person 14 and Person 80: 133 bits
  Distance between Person 14 and Person 81: 114 bits
  Distance between Person 14 and Person 82: 138 bits
  Distance between Person 14 and Person 83: 128 bits
  Distance between Person 14 and Person 84: 115 bits
  Distance between Person 14 and Person 85: 138 bits
  Distance between Person 14 and Person 86: 117 bits
  Distance between Person 14 and Person 87: 117 bits
  Distance between Person 14 and Person 88: 126 bits
  Distance between Person 14 and Person 89: 76 bits
  Distance between Person 15 and Person 16: 133 bits
  Distance between Person 15 and Person 17: 119 bits
  Distance between Person 15 and Person 18: 140 bits
  Distance between Person 15 and Person 19: 115 bits
  Distance between Person 15 and Person 20: 111 bits
  Distance between Person 15 and Person 21: 106 bits
  Distance between Person 15 and Person 22: 130 bits
  Distance between Person 15 and Person 23: 114 bits
  Distance between Person 15 and Person 24: 136 bits
  Distance between Person 15 and Person 25: 132 bits
  Distance between Person 15 and Person 26: 154 bits
  Distance between Person 15 and Person 27: 131 bits
  Distance between Person 15 and Person 28: 133 bits
  Distance between Person 15 and Person 29: 126 bits
  Distance between Person 15 and Person 30: 133 bits
  Distance between Person 15 and Person 31: 128 bits
  Distance between Person 15 and Person 32: 131 bits
  Distance between Person 15 and Person 33: 154 bits
  Distance between Person 15 and Person 34: 128 bits
  Distance between Person 15 and Person 35: 137 bits
  Distance between Person 15 and Person 36: 126 bits
  Distance between Person 15 and Person 37: 122 bits
  Distance between Person 15 and Person 38: 104 bits
  Distance between Person 15 and Person 39: 123 bits
  Distance between Person 15 and Person 40: 107 bits
  Distance between Person 15 and Person 41: 119 bits
  Distance between Person 15 and Person 42: 139 bits
  Distance between Person 15 and Person 43: 135 bits
  Distance between Person 15 and Person 44: 136 bits
  Distance between Person 15 and Person 45: 111 bits
  Distance between Person 15 and Person 46: 123 bits
  Distance between Person 15 and Person 47: 142 bits
  Distance between Person 15 and Person 48: 125 bits
  Distance between Person 15 and Person 49: 129 bits
  Distance between Person 15 and Person 50: 132 bits
  Distance between Person 15 and Person 51: 124 bits
  Distance between Person 15 and Person 52: 118 bits
  Distance between Person 15 and Person 53: 141 bits
  Distance between Person 15 and Person 54: 127 bits
  Distance between Person 15 and Person 55: 115 bits
  Distance between Person 15 and Person 56: 126 bits
  Distance between Person 15 and Person 57: 131 bits
  Distance between Person 15 and Person 58: 119 bits
  Distance between Person 15 and Person 59: 124 bits
  Distance between Person 15 and Person 60: 116 bits
  Distance between Person 15 and Person 61: 129 bits
  Distance between Person 15 and Person 62: 135 bits
  Distance between Person 15 and Person 63: 121 bits
  Distance between Person 15 and Person 64: 134 bits
  Distance between Person 15 and Person 65: 133 bits
  Distance between Person 15 and Person 66: 131 bits
  Distance between Person 15 and Person 67: 141 bits
  Distance between Person 15 and Person 68: 85 bits
  Distance between Person 15 and Person 69: 153 bits
  Distance between Person 15 and Person 70: 119 bits
  Distance between Person 15 and Person 71: 152 bits
  Distance between Person 15 and Person 72: 108 bits
  Distance between Person 15 and Person 73: 98 bits
  Distance between Person 15 and Person 74: 121 bits
  Distance between Person 15 and Person 75: 137 bits
  Distance between Person 15 and Person 76: 137 bits
  Distance between Person 15 and Person 77: 131 bits
  Distance between Person 15 and Person 78: 129 bits
  Distance between Person 15 and Person 79: 145 bits
  Distance between Person 15 and Person 80: 91 bits
  Distance between Person 15 and Person 81: 130 bits
  Distance between Person 15 and Person 82: 162 bits
  Distance between Person 15 and Person 83: 124 bits
  Distance between Person 15 and Person 84: 121 bits
  Distance between Person 15 and Person 85: 122 bits
  Distance between Person 15 and Person 86: 119 bits
  Distance between Person 15 and Person 87: 127 bits
  Distance between Person 15 and Person 88: 144 bits
  Distance between Person 15 and Person 89: 132 bits
  Distance between Person 16 and Person 17: 134 bits
  Distance between Person 16 and Person 18: 95 bits
  Distance between Person 16 and Person 19: 114 bits
  Distance between Person 16 and Person 20: 116 bits
  Distance between Person 16 and Person 21: 117 bits
  Distance between Person 16 and Person 22: 127 bits
  Distance between Person 16 and Person 23: 135 bits
  Distance between Person 16 and Person 24: 95 bits
  Distance between Person 16 and Person 25: 107 bits
  Distance between Person 16 and Person 26: 107 bits
  Distance between Person 16 and Person 27: 114 bits
  Distance between Person 16 and Person 28: 110 bits
  Distance between Person 16 and Person 29: 127 bits
  Distance between Person 16 and Person 30: 148 bits
  Distance between Person 16 and Person 31: 117 bits
  Distance between Person 16 and Person 32: 86 bits
  Distance between Person 16 and Person 33: 115 bits
  Distance between Person 16 and Person 34: 99 bits
  Distance between Person 16 and Person 35: 120 bits
  Distance between Person 16 and Person 36: 139 bits
  Distance between Person 16 and Person 37: 129 bits
  Distance between Person 16 and Person 38: 139 bits
  Distance between Person 16 and Person 39: 120 bits
  Distance between Person 16 and Person 40: 88 bits
  Distance between Person 16 and Person 41: 116 bits
  Distance between Person 16 and Person 42: 110 bits
  Distance between Person 16 and Person 43: 98 bits
  Distance between Person 16 and Person 44: 111 bits
  Distance between Person 16 and Person 45: 118 bits
  Distance between Person 16 and Person 46: 96 bits
  Distance between Person 16 and Person 47: 113 bits
  Distance between Person 16 and Person 48: 108 bits
  Distance between Person 16 and Person 49: 132 bits
  Distance between Person 16 and Person 50: 103 bits
  Distance between Person 16 and Person 51: 131 bits
  Distance between Person 16 and Person 52: 131 bits
  Distance between Person 16 and Person 53: 108 bits
  Distance between Person 16 and Person 54: 128 bits
  Distance between Person 16 and Person 55: 122 bits
  Distance between Person 16 and Person 56: 113 bits
  Distance between Person 16 and Person 57: 130 bits
  Distance between Person 16 and Person 58: 104 bits
  Distance between Person 16 and Person 59: 113 bits
  Distance between Person 16 and Person 60: 123 bits
  Distance between Person 16 and Person 61: 104 bits
  Distance between Person 16 and Person 62: 98 bits
  Distance between Person 16 and Person 63: 120 bits
  Distance between Person 16 and Person 64: 127 bits
  Distance between Person 16 and Person 65: 116 bits
  Distance between Person 16 and Person 66: 132 bits
  Distance between Person 16 and Person 67: 118 bits
  Distance between Person 16 and Person 68: 114 bits
  Distance between Person 16 and Person 69: 118 bits
  Distance between Person 16 and Person 70: 118 bits
  Distance between Person 16 and Person 71: 119 bits
  Distance between Person 16 and Person 72: 127 bits
  Distance between Person 16 and Person 73: 121 bits
  Distance between Person 16 and Person 74: 112 bits
  Distance between Person 16 and Person 75: 124 bits
  Distance between Person 16 and Person 76: 96 bits
  Distance between Person 16 and Person 77: 122 bits
  Distance between Person 16 and Person 78: 106 bits
  Distance between Person 16 and Person 79: 116 bits
  Distance between Person 16 and Person 80: 122 bits
  Distance between Person 16 and Person 81: 115 bits
  Distance between Person 16 and Person 82: 107 bits
  Distance between Person 16 and Person 83: 133 bits
  Distance between Person 16 and Person 84: 46 bits
  Distance between Person 16 and Person 85: 113 bits
  Distance between Person 16 and Person 86: 102 bits
  Distance between Person 16 and Person 87: 140 bits
  Distance between Person 16 and Person 88: 127 bits
  Distance between Person 16 and Person 89: 101 bits
  Distance between Person 17 and Person 18: 131 bits
  Distance between Person 17 and Person 19: 128 bits
  Distance between Person 17 and Person 20: 120 bits
  Distance between Person 17 and Person 21: 115 bits
  Distance between Person 17 and Person 22: 131 bits
  Distance between Person 17 and Person 23: 117 bits
  Distance between Person 17 and Person 24: 125 bits
  Distance between Person 17 and Person 25: 115 bits
  Distance between Person 17 and Person 26: 133 bits
  Distance between Person 17 and Person 27: 122 bits
  Distance between Person 17 and Person 28: 138 bits
  Distance between Person 17 and Person 29: 145 bits
  Distance between Person 17 and Person 30: 86 bits
  Distance between Person 17 and Person 31: 117 bits
  Distance between Person 17 and Person 32: 120 bits
  Distance between Person 17 and Person 33: 125 bits
  Distance between Person 17 and Person 34: 137 bits
  Distance between Person 17 and Person 35: 142 bits
  Distance between Person 17 and Person 36: 129 bits
  Distance between Person 17 and Person 37: 125 bits
  Distance between Person 17 and Person 38: 107 bits
  Distance between Person 17 and Person 39: 140 bits
  Distance between Person 17 and Person 40: 134 bits
  Distance between Person 17 and Person 41: 108 bits
  Distance between Person 17 and Person 42: 120 bits
  Distance between Person 17 and Person 43: 126 bits
  Distance between Person 17 and Person 44: 133 bits
  Distance between Person 17 and Person 45: 120 bits
  Distance between Person 17 and Person 46: 110 bits
  Distance between Person 17 and Person 47: 141 bits
  Distance between Person 17 and Person 48: 122 bits
  Distance between Person 17 and Person 49: 122 bits
  Distance between Person 17 and Person 50: 131 bits
  Distance between Person 17 and Person 51: 119 bits
  Distance between Person 17 and Person 52: 119 bits
  Distance between Person 17 and Person 53: 136 bits
  Distance between Person 17 and Person 54: 120 bits
  Distance between Person 17 and Person 55: 110 bits
  Distance between Person 17 and Person 56: 155 bits
  Distance between Person 17 and Person 57: 128 bits
  Distance between Person 17 and Person 58: 114 bits
  Distance between Person 17 and Person 59: 123 bits
  Distance between Person 17 and Person 60: 125 bits
  Distance between Person 17 and Person 61: 116 bits
  Distance between Person 17 and Person 62: 112 bits
  Distance between Person 17 and Person 63: 138 bits
  Distance between Person 17 and Person 64: 135 bits
  Distance between Person 17 and Person 65: 138 bits
  Distance between Person 17 and Person 66: 136 bits
  Distance between Person 17 and Person 67: 132 bits
  Distance between Person 17 and Person 68: 132 bits
  Distance between Person 17 and Person 69: 108 bits
  Distance between Person 17 and Person 70: 106 bits
  Distance between Person 17 and Person 71: 129 bits
  Distance between Person 17 and Person 72: 113 bits
  Distance between Person 17 and Person 73: 107 bits
  Distance between Person 17 and Person 74: 126 bits
  Distance between Person 17 and Person 75: 138 bits
  Distance between Person 17 and Person 76: 132 bits
  Distance between Person 17 and Person 77: 120 bits
  Distance between Person 17 and Person 78: 112 bits
  Distance between Person 17 and Person 79: 124 bits
  Distance between Person 17 and Person 80: 120 bits
  Distance between Person 17 and Person 81: 131 bits
  Distance between Person 17 and Person 82: 141 bits
  Distance between Person 17 and Person 83: 131 bits
  Distance between Person 17 and Person 84: 136 bits
  Distance between Person 17 and Person 85: 119 bits
  Distance between Person 17 and Person 86: 112 bits
  Distance between Person 17 and Person 87: 120 bits
  Distance between Person 17 and Person 88: 137 bits
  Distance between Person 17 and Person 89: 117 bits
  Distance between Person 18 and Person 19: 105 bits
  Distance between Person 18 and Person 20: 133 bits
  Distance between Person 18 and Person 21: 116 bits
  Distance between Person 18 and Person 22: 138 bits
  Distance between Person 18 and Person 23: 114 bits
  Distance between Person 18 and Person 24: 138 bits
  Distance between Person 18 and Person 25: 118 bits
  Distance between Person 18 and Person 26: 106 bits
  Distance between Person 18 and Person 27: 115 bits
  Distance between Person 18 and Person 28: 131 bits
  Distance between Person 18 and Person 29: 96 bits
  Distance between Person 18 and Person 30: 145 bits
  Distance between Person 18 and Person 31: 126 bits
  Distance between Person 18 and Person 32: 107 bits
  Distance between Person 18 and Person 33: 110 bits
  Distance between Person 18 and Person 34: 128 bits
  Distance between Person 18 and Person 35: 133 bits
  Distance between Person 18 and Person 36: 126 bits
  Distance between Person 18 and Person 37: 104 bits
  Distance between Person 18 and Person 38: 126 bits
  Distance between Person 18 and Person 39: 131 bits
  Distance between Person 18 and Person 40: 105 bits
  Distance between Person 18 and Person 41: 121 bits
  Distance between Person 18 and Person 42: 119 bits
  Distance between Person 18 and Person 43: 117 bits
  Distance between Person 18 and Person 44: 118 bits
  Distance between Person 18 and Person 45: 115 bits
  Distance between Person 18 and Person 46: 109 bits
  Distance between Person 18 and Person 47: 124 bits
  Distance between Person 18 and Person 48: 129 bits
  Distance between Person 18 and Person 49: 147 bits
  Distance between Person 18 and Person 50: 122 bits
  Distance between Person 18 and Person 51: 104 bits
  Distance between Person 18 and Person 52: 142 bits
  Distance between Person 18 and Person 53: 125 bits
  Distance between Person 18 and Person 54: 113 bits
  Distance between Person 18 and Person 55: 115 bits
  Distance between Person 18 and Person 56: 132 bits
  Distance between Person 18 and Person 57: 141 bits
  Distance between Person 18 and Person 58: 127 bits
  Distance between Person 18 and Person 59: 118 bits
  Distance between Person 18 and Person 60: 110 bits
  Distance between Person 18 and Person 61: 129 bits
  Distance between Person 18 and Person 62: 99 bits
  Distance between Person 18 and Person 63: 111 bits
  Distance between Person 18 and Person 64: 116 bits
  Distance between Person 18 and Person 65: 123 bits
  Distance between Person 18 and Person 66: 137 bits
  Distance between Person 18 and Person 67: 123 bits
  Distance between Person 18 and Person 68: 139 bits
  Distance between Person 18 and Person 69: 103 bits
  Distance between Person 18 and Person 70: 125 bits
  Distance between Person 18 and Person 71: 114 bits
  Distance between Person 18 and Person 72: 148 bits
  Distance between Person 18 and Person 73: 122 bits
  Distance between Person 18 and Person 74: 123 bits
  Distance between Person 18 and Person 75: 133 bits
  Distance between Person 18 and Person 76: 129 bits
  Distance between Person 18 and Person 77: 121 bits
  Distance between Person 18 and Person 78: 101 bits
  Distance between Person 18 and Person 79: 131 bits
  Distance between Person 18 and Person 80: 135 bits
  Distance between Person 18 and Person 81: 114 bits
  Distance between Person 18 and Person 82: 102 bits
  Distance between Person 18 and Person 83: 136 bits
  Distance between Person 18 and Person 84: 101 bits
  Distance between Person 18 and Person 85: 106 bits
  Distance between Person 18 and Person 86: 97 bits
  Distance between Person 18 and Person 87: 139 bits
  Distance between Person 18 and Person 88: 92 bits
  Distance between Person 18 and Person 89: 112 bits
  Distance between Person 19 and Person 20: 100 bits
  Distance between Person 19 and Person 21: 119 bits
  Distance between Person 19 and Person 22: 139 bits
  Distance between Person 19 and Person 23: 131 bits
  Distance between Person 19 and Person 24: 129 bits
  Distance between Person 19 and Person 25: 137 bits
  Distance between Person 19 and Person 26: 119 bits
  Distance between Person 19 and Person 27: 122 bits
  Distance between Person 19 and Person 28: 130 bits
  Distance between Person 19 and Person 29: 115 bits
  Distance between Person 19 and Person 30: 140 bits
  Distance between Person 19 and Person 31: 129 bits
  Distance between Person 19 and Person 32: 108 bits
  Distance between Person 19 and Person 33: 135 bits
  Distance between Person 19 and Person 34: 105 bits
  Distance between Person 19 and Person 35: 136 bits
  Distance between Person 19 and Person 36: 139 bits
  Distance between Person 19 and Person 37: 123 bits
  Distance between Person 19 and Person 38: 127 bits
  Distance between Person 19 and Person 39: 112 bits
  Distance between Person 19 and Person 40: 82 bits
  Distance between Person 19 and Person 41: 114 bits
  Distance between Person 19 and Person 42: 108 bits
  Distance between Person 19 and Person 43: 130 bits
  Distance between Person 19 and Person 44: 133 bits
  Distance between Person 19 and Person 45: 82 bits
  Distance between Person 19 and Person 46: 126 bits
  Distance between Person 19 and Person 47: 83 bits
  Distance between Person 19 and Person 48: 118 bits
  Distance between Person 19 and Person 49: 128 bits
  Distance between Person 19 and Person 50: 115 bits
  Distance between Person 19 and Person 51: 139 bits
  Distance between Person 19 and Person 52: 123 bits
  Distance between Person 19 and Person 53: 126 bits
  Distance between Person 19 and Person 54: 138 bits
  Distance between Person 19 and Person 55: 66 bits
  Distance between Person 19 and Person 56: 125 bits
  Distance between Person 19 and Person 57: 102 bits
  Distance between Person 19 and Person 58: 128 bits
  Distance between Person 19 and Person 59: 117 bits
  Distance between Person 19 and Person 60: 111 bits
  Distance between Person 19 and Person 61: 118 bits
  Distance between Person 19 and Person 62: 120 bits
  Distance between Person 19 and Person 63: 128 bits
  Distance between Person 19 and Person 64: 109 bits
  Distance between Person 19 and Person 65: 144 bits
  Distance between Person 19 and Person 66: 120 bits
  Distance between Person 19 and Person 67: 118 bits
  Distance between Person 19 and Person 68: 146 bits
  Distance between Person 19 and Person 69: 112 bits
  Distance between Person 19 and Person 70: 134 bits
  Distance between Person 19 and Person 71: 101 bits
  Distance between Person 19 and Person 72: 129 bits
  Distance between Person 19 and Person 73: 131 bits
  Distance between Person 19 and Person 74: 124 bits
  Distance between Person 19 and Person 75: 124 bits
  Distance between Person 19 and Person 76: 114 bits
  Distance between Person 19 and Person 77: 128 bits
  Distance between Person 19 and Person 78: 116 bits
  Distance between Person 19 and Person 79: 126 bits
  Distance between Person 19 and Person 80: 112 bits
  Distance between Person 19 and Person 81: 121 bits
  Distance between Person 19 and Person 82: 131 bits
  Distance between Person 19 and Person 83: 151 bits
  Distance between Person 19 and Person 84: 118 bits
  Distance between Person 19 and Person 85: 115 bits
  Distance between Person 19 and Person 86: 92 bits
  Distance between Person 19 and Person 87: 142 bits
  Distance between Person 19 and Person 88: 121 bits
  Distance between Person 19 and Person 89: 127 bits
  Distance between Person 20 and Person 21: 125 bits
  Distance between Person 20 and Person 22: 121 bits
  Distance between Person 20 and Person 23: 127 bits
  Distance between Person 20 and Person 24: 129 bits
  Distance between Person 20 and Person 25: 131 bits
  Distance between Person 20 and Person 26: 143 bits
  Distance between Person 20 and Person 27: 136 bits
  Distance between Person 20 and Person 28: 124 bits
  Distance between Person 20 and Person 29: 123 bits
  Distance between Person 20 and Person 30: 136 bits
  Distance between Person 20 and Person 31: 115 bits
  Distance between Person 20 and Person 32: 112 bits
  Distance between Person 20 and Person 33: 129 bits
  Distance between Person 20 and Person 34: 121 bits
  Distance between Person 20 and Person 35: 108 bits
  Distance between Person 20 and Person 36: 129 bits
  Distance between Person 20 and Person 37: 111 bits
  Distance between Person 20 and Person 38: 127 bits
  Distance between Person 20 and Person 39: 120 bits
  Distance between Person 20 and Person 40: 122 bits
  Distance between Person 20 and Person 41: 122 bits
  Distance between Person 20 and Person 42: 114 bits
  Distance between Person 20 and Person 43: 148 bits
  Distance between Person 20 and Person 44: 125 bits
  Distance between Person 20 and Person 45: 98 bits
  Distance between Person 20 and Person 46: 110 bits
  Distance between Person 20 and Person 47: 101 bits
  Distance between Person 20 and Person 48: 122 bits
  Distance between Person 20 and Person 49: 120 bits
  Distance between Person 20 and Person 50: 77 bits
  Distance between Person 20 and Person 51: 141 bits
  Distance between Person 20 and Person 52: 133 bits
  Distance between Person 20 and Person 53: 114 bits
  Distance between Person 20 and Person 54: 108 bits
  Distance between Person 20 and Person 55: 124 bits
  Distance between Person 20 and Person 56: 123 bits
  Distance between Person 20 and Person 57: 114 bits
  Distance between Person 20 and Person 58: 122 bits
  Distance between Person 20 and Person 59: 117 bits
  Distance between Person 20 and Person 60: 103 bits
  Distance between Person 20 and Person 61: 116 bits
  Distance between Person 20 and Person 62: 122 bits
  Distance between Person 20 and Person 63: 136 bits
  Distance between Person 20 and Person 64: 139 bits
  Distance between Person 20 and Person 65: 142 bits
  Distance between Person 20 and Person 66: 130 bits
  Distance between Person 20 and Person 67: 122 bits
  Distance between Person 20 and Person 68: 126 bits
  Distance between Person 20 and Person 69: 114 bits
  Distance between Person 20 and Person 70: 128 bits
  Distance between Person 20 and Person 71: 131 bits
  Distance between Person 20 and Person 72: 109 bits
  Distance between Person 20 and Person 73: 95 bits
  Distance between Person 20 and Person 74: 122 bits
  Distance between Person 20 and Person 75: 130 bits
  Distance between Person 20 and Person 76: 88 bits
  Distance between Person 20 and Person 77: 132 bits
  Distance between Person 20 and Person 78: 116 bits
  Distance between Person 20 and Person 79: 124 bits
  Distance between Person 20 and Person 80: 104 bits
  Distance between Person 20 and Person 81: 119 bits
  Distance between Person 20 and Person 82: 123 bits
  Distance between Person 20 and Person 83: 145 bits
  Distance between Person 20 and Person 84: 106 bits
  Distance between Person 20 and Person 85: 141 bits
  Distance between Person 20 and Person 86: 114 bits
  Distance between Person 20 and Person 87: 122 bits
  Distance between Person 20 and Person 88: 121 bits
  Distance between Person 20 and Person 89: 103 bits
  Distance between Person 21 and Person 22: 132 bits
  Distance between Person 21 and Person 23: 126 bits
  Distance between Person 21 and Person 24: 102 bits
  Distance between Person 21 and Person 25: 134 bits
  Distance between Person 21 and Person 26: 112 bits
  Distance between Person 21 and Person 27: 139 bits
  Distance between Person 21 and Person 28: 125 bits
  Distance between Person 21 and Person 29: 128 bits
  Distance between Person 21 and Person 30: 147 bits
  Distance between Person 21 and Person 31: 128 bits
  Distance between Person 21 and Person 32: 85 bits
  Distance between Person 21 and Person 33: 134 bits
  Distance between Person 21 and Person 34: 122 bits
  Distance between Person 21 and Person 35: 135 bits
  Distance between Person 21 and Person 36: 120 bits
  Distance between Person 21 and Person 37: 122 bits
  Distance between Person 21 and Person 38: 72 bits
  Distance between Person 21 and Person 39: 115 bits
  Distance between Person 21 and Person 40: 101 bits
  Distance between Person 21 and Person 41: 121 bits
  Distance between Person 21 and Person 42: 133 bits
  Distance between Person 21 and Person 43: 93 bits
  Distance between Person 21 and Person 44: 114 bits
  Distance between Person 21 and Person 45: 123 bits
  Distance between Person 21 and Person 46: 125 bits
  Distance between Person 21 and Person 47: 128 bits
  Distance between Person 21 and Person 48: 115 bits
  Distance between Person 21 and Person 49: 131 bits
  Distance between Person 21 and Person 50: 116 bits
  Distance between Person 21 and Person 51: 126 bits
  Distance between Person 21 and Person 52: 128 bits
  Distance between Person 21 and Person 53: 121 bits
  Distance between Person 21 and Person 54: 115 bits
  Distance between Person 21 and Person 55: 111 bits
  Distance between Person 21 and Person 56: 126 bits
  Distance between Person 21 and Person 57: 135 bits
  Distance between Person 21 and Person 58: 127 bits
  Distance between Person 21 and Person 59: 114 bits
  Distance between Person 21 and Person 60: 74 bits
  Distance between Person 21 and Person 61: 111 bits
  Distance between Person 21 and Person 62: 69 bits
  Distance between Person 21 and Person 63: 151 bits
  Distance between Person 21 and Person 64: 120 bits
  Distance between Person 21 and Person 65: 125 bits
  Distance between Person 21 and Person 66: 133 bits
  Distance between Person 21 and Person 67: 151 bits
  Distance between Person 21 and Person 68: 103 bits
  Distance between Person 21 and Person 69: 139 bits
  Distance between Person 21 and Person 70: 65 bits
  Distance between Person 21 and Person 71: 118 bits
  Distance between Person 21 and Person 72: 128 bits
  Distance between Person 21 and Person 73: 116 bits
  Distance between Person 21 and Person 74: 123 bits
  Distance between Person 21 and Person 75: 113 bits
  Distance between Person 21 and Person 76: 115 bits
  Distance between Person 21 and Person 77: 103 bits
  Distance between Person 21 and Person 78: 93 bits
  Distance between Person 21 and Person 79: 143 bits
  Distance between Person 21 and Person 80: 109 bits
  Distance between Person 21 and Person 81: 154 bits
  Distance between Person 21 and Person 82: 146 bits
  Distance between Person 21 and Person 83: 140 bits
  Distance between Person 21 and Person 84: 121 bits
  Distance between Person 21 and Person 85: 132 bits
  Distance between Person 21 and Person 86: 117 bits
  Distance between Person 21 and Person 87: 127 bits
  Distance between Person 21 and Person 88: 118 bits
  Distance between Person 21 and Person 89: 120 bits
  Distance between Person 22 and Person 23: 130 bits
  Distance between Person 22 and Person 24: 130 bits
  Distance between Person 22 and Person 25: 128 bits
  Distance between Person 22 and Person 26: 120 bits
  Distance between Person 22 and Person 27: 117 bits
  Distance between Person 22 and Person 28: 117 bits
  Distance between Person 22 and Person 29: 126 bits
  Distance between Person 22 and Person 30: 121 bits
  Distance between Person 22 and Person 31: 130 bits
  Distance between Person 22 and Person 32: 131 bits
  Distance between Person 22 and Person 33: 144 bits
  Distance between Person 22 and Person 34: 122 bits
  Distance between Person 22 and Person 35: 131 bits
  Distance between Person 22 and Person 36: 110 bits
  Distance between Person 22 and Person 37: 148 bits
  Distance between Person 22 and Person 38: 144 bits
  Distance between Person 22 and Person 39: 129 bits
  Distance between Person 22 and Person 40: 125 bits
  Distance between Person 22 and Person 41: 131 bits
  Distance between Person 22 and Person 42: 133 bits
  Distance between Person 22 and Person 43: 121 bits
  Distance between Person 22 and Person 44: 140 bits
  Distance between Person 22 and Person 45: 145 bits
  Distance between Person 22 and Person 46: 125 bits
  Distance between Person 22 and Person 47: 136 bits
  Distance between Person 22 and Person 48: 131 bits
  Distance between Person 22 and Person 49: 135 bits
  Distance between Person 22 and Person 50: 124 bits
  Distance between Person 22 and Person 51: 112 bits
  Distance between Person 22 and Person 52: 118 bits
  Distance between Person 22 and Person 53: 129 bits
  Distance between Person 22 and Person 54: 129 bits
  Distance between Person 22 and Person 55: 149 bits
  Distance between Person 22 and Person 56: 112 bits
  Distance between Person 22 and Person 57: 123 bits
  Distance between Person 22 and Person 58: 143 bits
  Distance between Person 22 and Person 59: 126 bits
  Distance between Person 22 and Person 60: 132 bits
  Distance between Person 22 and Person 61: 129 bits
  Distance between Person 22 and Person 62: 125 bits
  Distance between Person 22 and Person 63: 149 bits
  Distance between Person 22 and Person 64: 124 bits
  Distance between Person 22 and Person 65: 139 bits
  Distance between Person 22 and Person 66: 141 bits
  Distance between Person 22 and Person 67: 137 bits
  Distance between Person 22 and Person 68: 121 bits
  Distance between Person 22 and Person 69: 123 bits
  Distance between Person 22 and Person 70: 127 bits
  Distance between Person 22 and Person 71: 150 bits
  Distance between Person 22 and Person 72: 136 bits
  Distance between Person 22 and Person 73: 114 bits
  Distance between Person 22 and Person 74: 127 bits
  Distance between Person 22 and Person 75: 133 bits
  Distance between Person 22 and Person 76: 127 bits
  Distance between Person 22 and Person 77: 137 bits
  Distance between Person 22 and Person 78: 127 bits
  Distance between Person 22 and Person 79: 129 bits
  Distance between Person 22 and Person 80: 135 bits
  Distance between Person 22 and Person 81: 132 bits
  Distance between Person 22 and Person 82: 126 bits
  Distance between Person 22 and Person 83: 124 bits
  Distance between Person 22 and Person 84: 121 bits
  Distance between Person 22 and Person 85: 126 bits
  Distance between Person 22 and Person 86: 141 bits
  Distance between Person 22 and Person 87: 127 bits
  Distance between Person 22 and Person 88: 136 bits
  Distance between Person 22 and Person 89: 116 bits
  Distance between Person 23 and Person 24: 142 bits
  Distance between Person 23 and Person 25: 126 bits
  Distance between Person 23 and Person 26: 150 bits
  Distance between Person 23 and Person 27: 137 bits
  Distance between Person 23 and Person 28: 123 bits
  Distance between Person 23 and Person 29: 118 bits
  Distance between Person 23 and Person 30: 125 bits
  Distance between Person 23 and Person 31: 128 bits
  Distance between Person 23 and Person 32: 127 bits
  Distance between Person 23 and Person 33: 128 bits
  Distance between Person 23 and Person 34: 138 bits
  Distance between Person 23 and Person 35: 127 bits
  Distance between Person 23 and Person 36: 132 bits
  Distance between Person 23 and Person 37: 110 bits
  Distance between Person 23 and Person 38: 126 bits
  Distance between Person 23 and Person 39: 129 bits
  Distance between Person 23 and Person 40: 137 bits
  Distance between Person 23 and Person 41: 127 bits
  Distance between Person 23 and Person 42: 129 bits
  Distance between Person 23 and Person 43: 133 bits
  Distance between Person 23 and Person 44: 138 bits
  Distance between Person 23 and Person 45: 111 bits
  Distance between Person 23 and Person 46: 121 bits
  Distance between Person 23 and Person 47: 140 bits
  Distance between Person 23 and Person 48: 123 bits
  Distance between Person 23 and Person 49: 121 bits
  Distance between Person 23 and Person 50: 126 bits
  Distance between Person 23 and Person 51: 120 bits
  Distance between Person 23 and Person 52: 154 bits
  Distance between Person 23 and Person 53: 125 bits
  Distance between Person 23 and Person 54: 131 bits
  Distance between Person 23 and Person 55: 125 bits
  Distance between Person 23 and Person 56: 132 bits
  Distance between Person 23 and Person 57: 135 bits
  Distance between Person 23 and Person 58: 137 bits
  Distance between Person 23 and Person 59: 128 bits
  Distance between Person 23 and Person 60: 116 bits
  Distance between Person 23 and Person 61: 127 bits
  Distance between Person 23 and Person 62: 127 bits
  Distance between Person 23 and Person 63: 117 bits
  Distance between Person 23 and Person 64: 138 bits
  Distance between Person 23 and Person 65: 129 bits
  Distance between Person 23 and Person 66: 143 bits
  Distance between Person 23 and Person 67: 139 bits
  Distance between Person 23 and Person 68: 135 bits
  Distance between Person 23 and Person 69: 139 bits
  Distance between Person 23 and Person 70: 127 bits
  Distance between Person 23 and Person 71: 132 bits
  Distance between Person 23 and Person 72: 130 bits
  Distance between Person 23 and Person 73: 118 bits
  Distance between Person 23 and Person 74: 119 bits
  Distance between Person 23 and Person 75: 137 bits
  Distance between Person 23 and Person 76: 129 bits
  Distance between Person 23 and Person 77: 119 bits
  Distance between Person 23 and Person 78: 129 bits
  Distance between Person 23 and Person 79: 145 bits
  Distance between Person 23 and Person 80: 119 bits
  Distance between Person 23 and Person 81: 124 bits
  Distance between Person 23 and Person 82: 144 bits
  Distance between Person 23 and Person 83: 124 bits
  Distance between Person 23 and Person 84: 135 bits
  Distance between Person 23 and Person 85: 124 bits
  Distance between Person 23 and Person 86: 123 bits
  Distance between Person 23 and Person 87: 147 bits
  Distance between Person 23 and Person 88: 140 bits
  Distance between Person 23 and Person 89: 132 bits
  Distance between Person 24 and Person 25: 118 bits
  Distance between Person 24 and Person 26: 82 bits
  Distance between Person 24 and Person 27: 121 bits
  Distance between Person 24 and Person 28: 97 bits
  Distance between Person 24 and Person 29: 118 bits
  Distance between Person 24 and Person 30: 125 bits
  Distance between Person 24 and Person 31: 130 bits
  Distance between Person 24 and Person 32: 79 bits
  Distance between Person 24 and Person 33: 112 bits
  Distance between Person 24 and Person 34: 94 bits
  Distance between Person 24 and Person 35: 119 bits
  Distance between Person 24 and Person 36: 122 bits
  Distance between Person 24 and Person 37: 128 bits
  Distance between Person 24 and Person 38: 108 bits
  Distance between Person 24 and Person 39: 103 bits
  Distance between Person 24 and Person 40: 91 bits
  Distance between Person 24 and Person 41: 125 bits
  Distance between Person 24 and Person 42: 137 bits
  Distance between Person 24 and Person 43: 85 bits
  Distance between Person 24 and Person 44: 108 bits
  Distance between Person 24 and Person 45: 123 bits
  Distance between Person 24 and Person 46: 121 bits
  Distance between Person 24 and Person 47: 118 bits
  Distance between Person 24 and Person 48: 91 bits
  Distance between Person 24 and Person 49: 133 bits
  Distance between Person 24 and Person 50: 112 bits
  Distance between Person 24 and Person 51: 116 bits
  Distance between Person 24 and Person 52: 116 bits
  Distance between Person 24 and Person 53: 97 bits
  Distance between Person 24 and Person 54: 129 bits
  Distance between Person 24 and Person 55: 105 bits
  Distance between Person 24 and Person 56: 126 bits
  Distance between Person 24 and Person 57: 113 bits
  Distance between Person 24 and Person 58: 127 bits
  Distance between Person 24 and Person 59: 130 bits
  Distance between Person 24 and Person 60: 138 bits
  Distance between Person 24 and Person 61: 99 bits
  Distance between Person 24 and Person 62: 113 bits
  Distance between Person 24 and Person 63: 131 bits
  Distance between Person 24 and Person 64: 122 bits
  Distance between Person 24 and Person 65: 123 bits
  Distance between Person 24 and Person 66: 103 bits
  Distance between Person 24 and Person 67: 135 bits
  Distance between Person 24 and Person 68: 139 bits
  Distance between Person 24 and Person 69: 123 bits
  Distance between Person 24 and Person 70: 101 bits
  Distance between Person 24 and Person 71: 132 bits
  Distance between Person 24 and Person 72: 116 bits
  Distance between Person 24 and Person 73: 126 bits
  Distance between Person 24 and Person 74: 133 bits
  Distance between Person 24 and Person 75: 119 bits
  Distance between Person 24 and Person 76: 101 bits
  Distance between Person 24 and Person 77: 107 bits
  Distance between Person 24 and Person 78: 101 bits
  Distance between Person 24 and Person 79: 139 bits
  Distance between Person 24 and Person 80: 113 bits
  Distance between Person 24 and Person 81: 128 bits
  Distance between Person 24 and Person 82: 110 bits
  Distance between Person 24 and Person 83: 146 bits
  Distance between Person 24 and Person 84: 97 bits
  Distance between Person 24 and Person 85: 118 bits
  Distance between Person 24 and Person 86: 113 bits
  Distance between Person 24 and Person 87: 119 bits
  Distance between Person 24 and Person 88: 136 bits
  Distance between Person 24 and Person 89: 126 bits
  Distance between Person 25 and Person 26: 136 bits
  Distance between Person 25 and Person 27: 117 bits
  Distance between Person 25 and Person 28: 113 bits
  Distance between Person 25 and Person 29: 126 bits
  Distance between Person 25 and Person 30: 139 bits
  Distance between Person 25 and Person 31: 124 bits
  Distance between Person 25 and Person 32: 135 bits
  Distance between Person 25 and Person 33: 108 bits
  Distance between Person 25 and Person 34: 116 bits
  Distance between Person 25 and Person 35: 127 bits
  Distance between Person 25 and Person 36: 138 bits
  Distance between Person 25 and Person 37: 132 bits
  Distance between Person 25 and Person 38: 130 bits
  Distance between Person 25 and Person 39: 119 bits
  Distance between Person 25 and Person 40: 131 bits
  Distance between Person 25 and Person 41: 103 bits
  Distance between Person 25 and Person 42: 111 bits
  Distance between Person 25 and Person 43: 117 bits
  Distance between Person 25 and Person 44: 106 bits
  Distance between Person 25 and Person 45: 115 bits
  Distance between Person 25 and Person 46: 97 bits
  Distance between Person 25 and Person 47: 110 bits
  Distance between Person 25 and Person 48: 135 bits
  Distance between Person 25 and Person 49: 137 bits
  Distance between Person 25 and Person 50: 114 bits
  Distance between Person 25 and Person 51: 116 bits
  Distance between Person 25 and Person 52: 136 bits
  Distance between Person 25 and Person 53: 109 bits
  Distance between Person 25 and Person 54: 127 bits
  Distance between Person 25 and Person 55: 139 bits
  Distance between Person 25 and Person 56: 122 bits
  Distance between Person 25 and Person 57: 135 bits
  Distance between Person 25 and Person 58: 87 bits
  Distance between Person 25 and Person 59: 108 bits
  Distance between Person 25 and Person 60: 122 bits
  Distance between Person 25 and Person 61: 115 bits
  Distance between Person 25 and Person 62: 113 bits
  Distance between Person 25 and Person 63: 119 bits
  Distance between Person 25 and Person 64: 124 bits
  Distance between Person 25 and Person 65: 131 bits
  Distance between Person 25 and Person 66: 131 bits
  Distance between Person 25 and Person 67: 99 bits
  Distance between Person 25 and Person 68: 125 bits
  Distance between Person 25 and Person 69: 121 bits
  Distance between Person 25 and Person 70: 127 bits
  Distance between Person 25 and Person 71: 124 bits
  Distance between Person 25 and Person 72: 112 bits
  Distance between Person 25 and Person 73: 118 bits
  Distance between Person 25 and Person 74: 141 bits
  Distance between Person 25 and Person 75: 113 bits
  Distance between Person 25 and Person 76: 109 bits
  Distance between Person 25 and Person 77: 113 bits
  Distance between Person 25 and Person 78: 111 bits
  Distance between Person 25 and Person 79: 131 bits
  Distance between Person 25 and Person 80: 115 bits
  Distance between Person 25 and Person 81: 104 bits
  Distance between Person 25 and Person 82: 108 bits
  Distance between Person 25 and Person 83: 138 bits
  Distance between Person 25 and Person 84: 121 bits
  Distance between Person 25 and Person 85: 120 bits
  Distance between Person 25 and Person 86: 121 bits
  Distance between Person 25 and Person 87: 119 bits
  Distance between Person 25 and Person 88: 124 bits
  Distance between Person 25 and Person 89: 118 bits
  Distance between Person 26 and Person 27: 129 bits
  Distance between Person 26 and Person 28: 125 bits
  Distance between Person 26 and Person 29: 124 bits
  Distance between Person 26 and Person 30: 139 bits
  Distance between Person 26 and Person 31: 122 bits
  Distance between Person 26 and Person 32: 105 bits
  Distance between Person 26 and Person 33: 124 bits
  Distance between Person 26 and Person 34: 122 bits
  Distance between Person 26 and Person 35: 145 bits
  Distance between Person 26 and Person 36: 100 bits
  Distance between Person 26 and Person 37: 124 bits
  Distance between Person 26 and Person 38: 128 bits
  Distance between Person 26 and Person 39: 125 bits
  Distance between Person 26 and Person 40: 107 bits
  Distance between Person 26 and Person 41: 123 bits
  Distance between Person 26 and Person 42: 117 bits
  Distance between Person 26 and Person 43: 77 bits
  Distance between Person 26 and Person 44: 118 bits
  Distance between Person 26 and Person 45: 135 bits
  Distance between Person 26 and Person 46: 123 bits
  Distance between Person 26 and Person 47: 114 bits
  Distance between Person 26 and Person 48: 127 bits
  Distance between Person 26 and Person 49: 149 bits
  Distance between Person 26 and Person 50: 132 bits
  Distance between Person 26 and Person 51: 104 bits
  Distance between Person 26 and Person 52: 118 bits
  Distance between Person 26 and Person 53: 109 bits
  Distance between Person 26 and Person 54: 99 bits
  Distance between Person 26 and Person 55: 103 bits
  Distance between Person 26 and Person 56: 124 bits
  Distance between Person 26 and Person 57: 119 bits
  Distance between Person 26 and Person 58: 129 bits
  Distance between Person 26 and Person 59: 134 bits
  Distance between Person 26 and Person 60: 148 bits
  Distance between Person 26 and Person 61: 125 bits
  Distance between Person 26 and Person 62: 125 bits
  Distance between Person 26 and Person 63: 139 bits
  Distance between Person 26 and Person 64: 118 bits
  Distance between Person 26 and Person 65: 121 bits
  Distance between Person 26 and Person 66: 129 bits
  Distance between Person 26 and Person 67: 129 bits
  Distance between Person 26 and Person 68: 145 bits
  Distance between Person 26 and Person 69: 117 bits
  Distance between Person 26 and Person 70: 115 bits
  Distance between Person 26 and Person 71: 138 bits
  Distance between Person 26 and Person 72: 134 bits
  Distance between Person 26 and Person 73: 124 bits
  Distance between Person 26 and Person 74: 127 bits
  Distance between Person 26 and Person 75: 125 bits
  Distance between Person 26 and Person 76: 119 bits
  Distance between Person 26 and Person 77: 117 bits
  Distance between Person 26 and Person 78: 113 bits
  Distance between Person 26 and Person 79: 135 bits
  Distance between Person 26 and Person 80: 127 bits
  Distance between Person 26 and Person 81: 138 bits
  Distance between Person 26 and Person 82: 92 bits
  Distance between Person 26 and Person 83: 158 bits
  Distance between Person 26 and Person 84: 123 bits
  Distance between Person 26 and Person 85: 124 bits
  Distance between Person 26 and Person 86: 135 bits
  Distance between Person 26 and Person 87: 135 bits
  Distance between Person 26 and Person 88: 126 bits
  Distance between Person 26 and Person 89: 132 bits
  Distance between Person 27 and Person 28: 124 bits
  Distance between Person 27 and Person 29: 121 bits
  Distance between Person 27 and Person 30: 124 bits
  Distance between Person 27 and Person 31: 141 bits
  Distance between Person 27 and Person 32: 126 bits
  Distance between Person 27 and Person 33: 115 bits
  Distance between Person 27 and Person 34: 119 bits
  Distance between Person 27 and Person 35: 126 bits
  Distance between Person 27 and Person 36: 121 bits
  Distance between Person 27 and Person 37: 127 bits
  Distance between Person 27 and Person 38: 131 bits
  Distance between Person 27 and Person 39: 122 bits
  Distance between Person 27 and Person 40: 114 bits
  Distance between Person 27 and Person 41: 136 bits
  Distance between Person 27 and Person 42: 118 bits
  Distance between Person 27 and Person 43: 112 bits
  Distance between Person 27 and Person 44: 145 bits
  Distance between Person 27 and Person 45: 144 bits
  Distance between Person 27 and Person 46: 128 bits
  Distance between Person 27 and Person 47: 131 bits
  Distance between Person 27 and Person 48: 126 bits
  Distance between Person 27 and Person 49: 138 bits
  Distance between Person 27 and Person 50: 137 bits
  Distance between Person 27 and Person 51: 129 bits
  Distance between Person 27 and Person 52: 111 bits
  Distance between Person 27 and Person 53: 142 bits
  Distance between Person 27 and Person 54: 138 bits
  Distance between Person 27 and Person 55: 142 bits
  Distance between Person 27 and Person 56: 125 bits
  Distance between Person 27 and Person 57: 140 bits
  Distance between Person 27 and Person 58: 130 bits
  Distance between Person 27 and Person 59: 123 bits
  Distance between Person 27 and Person 60: 129 bits
  Distance between Person 27 and Person 61: 132 bits
  Distance between Person 27 and Person 62: 120 bits
  Distance between Person 27 and Person 63: 124 bits
  Distance between Person 27 and Person 64: 137 bits
  Distance between Person 27 and Person 65: 122 bits
  Distance between Person 27 and Person 66: 128 bits
  Distance between Person 27 and Person 67: 116 bits
  Distance between Person 27 and Person 68: 118 bits
  Distance between Person 27 and Person 69: 114 bits
  Distance between Person 27 and Person 70: 128 bits
  Distance between Person 27 and Person 71: 129 bits
  Distance between Person 27 and Person 72: 109 bits
  Distance between Person 27 and Person 73: 129 bits
  Distance between Person 27 and Person 74: 134 bits
  Distance between Person 27 and Person 75: 120 bits
  Distance between Person 27 and Person 76: 134 bits
  Distance between Person 27 and Person 77: 132 bits
  Distance between Person 27 and Person 78: 124 bits
  Distance between Person 27 and Person 79: 130 bits
  Distance between Person 27 and Person 80: 116 bits
  Distance between Person 27 and Person 81: 109 bits
  Distance between Person 27 and Person 82: 121 bits
  Distance between Person 27 and Person 83: 129 bits
  Distance between Person 27 and Person 84: 116 bits
  Distance between Person 27 and Person 85: 125 bits
  Distance between Person 27 and Person 86: 122 bits
  Distance between Person 27 and Person 87: 138 bits
  Distance between Person 27 and Person 88: 129 bits
  Distance between Person 27 and Person 89: 125 bits
  Distance between Person 28 and Person 29: 137 bits
  Distance between Person 28 and Person 30: 138 bits
  Distance between Person 28 and Person 31: 121 bits
  Distance between Person 28 and Person 32: 110 bits
  Distance between Person 28 and Person 33: 119 bits
  Distance between Person 28 and Person 34: 109 bits
  Distance between Person 28 and Person 35: 116 bits
  Distance between Person 28 and Person 36: 133 bits
  Distance between Person 28 and Person 37: 121 bits
  Distance between Person 28 and Person 38: 113 bits
  Distance between Person 28 and Person 39: 112 bits
  Distance between Person 28 and Person 40: 112 bits
  Distance between Person 28 and Person 41: 132 bits
  Distance between Person 28 and Person 42: 126 bits
  Distance between Person 28 and Person 43: 110 bits
  Distance between Person 28 and Person 44: 101 bits
  Distance between Person 28 and Person 45: 114 bits
  Distance between Person 28 and Person 46: 108 bits
  Distance between Person 28 and Person 47: 117 bits
  Distance between Person 28 and Person 48: 112 bits
  Distance between Person 28 and Person 49: 124 bits
  Distance between Person 28 and Person 50: 123 bits
  Distance between Person 28 and Person 51: 127 bits
  Distance between Person 28 and Person 52: 117 bits
  Distance between Person 28 and Person 53: 86 bits
  Distance between Person 28 and Person 54: 122 bits
  Distance between Person 28 and Person 55: 114 bits
  Distance between Person 28 and Person 56: 137 bits
  Distance between Person 28 and Person 57: 130 bits
  Distance between Person 28 and Person 58: 126 bits
  Distance between Person 28 and Person 59: 137 bits
  Distance between Person 28 and Person 60: 131 bits
  Distance between Person 28 and Person 61: 76 bits
  Distance between Person 28 and Person 62: 118 bits
  Distance between Person 28 and Person 63: 126 bits
  Distance between Person 28 and Person 64: 121 bits
  Distance between Person 28 and Person 65: 122 bits
  Distance between Person 28 and Person 66: 128 bits
  Distance between Person 28 and Person 67: 118 bits
  Distance between Person 28 and Person 68: 124 bits
  Distance between Person 28 and Person 69: 130 bits
  Distance between Person 28 and Person 70: 114 bits
  Distance between Person 28 and Person 71: 125 bits
  Distance between Person 28 and Person 72: 99 bits
  Distance between Person 28 and Person 73: 113 bits
  Distance between Person 28 and Person 74: 130 bits
  Distance between Person 28 and Person 75: 146 bits
  Distance between Person 28 and Person 76: 118 bits
  Distance between Person 28 and Person 77: 132 bits
  Distance between Person 28 and Person 78: 96 bits
  Distance between Person 28 and Person 79: 136 bits
  Distance between Person 28 and Person 80: 132 bits
  Distance between Person 28 and Person 81: 117 bits
  Distance between Person 28 and Person 82: 127 bits
  Distance between Person 28 and Person 83: 141 bits
  Distance between Person 28 and Person 84: 118 bits
  Distance between Person 28 and Person 85: 97 bits
  Distance between Person 28 and Person 86: 104 bits
  Distance between Person 28 and Person 87: 126 bits
  Distance between Person 28 and Person 88: 117 bits
  Distance between Person 28 and Person 89: 101 bits
  Distance between Person 29 and Person 30: 125 bits
  Distance between Person 29 and Person 31: 134 bits
  Distance between Person 29 and Person 32: 129 bits
  Distance between Person 29 and Person 33: 116 bits
  Distance between Person 29 and Person 34: 114 bits
  Distance between Person 29 and Person 35: 107 bits
  Distance between Person 29 and Person 36: 130 bits
  Distance between Person 29 and Person 37: 102 bits
  Distance between Person 29 and Person 38: 134 bits
  Distance between Person 29 and Person 39: 117 bits
  Distance between Person 29 and Person 40: 101 bits
  Distance between Person 29 and Person 41: 121 bits
  Distance between Person 29 and Person 42: 129 bits
  Distance between Person 29 and Person 43: 127 bits
  Distance between Person 29 and Person 44: 118 bits
  Distance between Person 29 and Person 45: 115 bits
  Distance between Person 29 and Person 46: 119 bits
  Distance between Person 29 and Person 47: 118 bits
  Distance between Person 29 and Person 48: 135 bits
  Distance between Person 29 and Person 49: 131 bits
  Distance between Person 29 and Person 50: 120 bits
  Distance between Person 29 and Person 51: 112 bits
  Distance between Person 29 and Person 52: 136 bits
  Distance between Person 29 and Person 53: 121 bits
  Distance between Person 29 and Person 54: 123 bits
  Distance between Person 29 and Person 55: 121 bits
  Distance between Person 29 and Person 56: 122 bits
  Distance between Person 29 and Person 57: 137 bits
  Distance between Person 29 and Person 58: 129 bits
  Distance between Person 29 and Person 59: 122 bits
  Distance between Person 29 and Person 60: 126 bits
  Distance between Person 29 and Person 61: 127 bits
  Distance between Person 29 and Person 62: 139 bits
  Distance between Person 29 and Person 63: 121 bits
  Distance between Person 29 and Person 64: 134 bits
  Distance between Person 29 and Person 65: 129 bits
  Distance between Person 29 and Person 66: 125 bits
  Distance between Person 29 and Person 67: 123 bits
  Distance between Person 29 and Person 68: 123 bits
  Distance between Person 29 and Person 69: 109 bits
  Distance between Person 29 and Person 70: 129 bits
  Distance between Person 29 and Person 71: 124 bits
  Distance between Person 29 and Person 72: 144 bits
  Distance between Person 29 and Person 73: 134 bits
  Distance between Person 29 and Person 74: 121 bits
  Distance between Person 29 and Person 75: 123 bits
  Distance between Person 29 and Person 76: 121 bits
  Distance between Person 29 and Person 77: 115 bits
  Distance between Person 29 and Person 78: 133 bits
  Distance between Person 29 and Person 79: 119 bits
  Distance between Person 29 and Person 80: 121 bits
  Distance between Person 29 and Person 81: 130 bits
  Distance between Person 29 and Person 82: 98 bits
  Distance between Person 29 and Person 83: 132 bits
  Distance between Person 29 and Person 84: 123 bits
  Distance between Person 29 and Person 85: 136 bits
  Distance between Person 29 and Person 86: 123 bits
  Distance between Person 29 and Person 87: 133 bits
  Distance between Person 29 and Person 88: 108 bits
  Distance between Person 29 and Person 89: 120 bits
  Distance between Person 30 and Person 31: 135 bits
  Distance between Person 30 and Person 32: 148 bits
  Distance between Person 30 and Person 33: 129 bits
  Distance between Person 30 and Person 34: 137 bits
  Distance between Person 30 and Person 35: 106 bits
  Distance between Person 30 and Person 36: 115 bits
  Distance between Person 30 and Person 37: 127 bits
  Distance between Person 30 and Person 38: 139 bits
  Distance between Person 30 and Person 39: 140 bits
  Distance between Person 30 and Person 40: 144 bits
  Distance between Person 30 and Person 41: 150 bits
  Distance between Person 30 and Person 42: 112 bits
  Distance between Person 30 and Person 43: 130 bits
  Distance between Person 30 and Person 44: 143 bits
  Distance between Person 30 and Person 45: 136 bits
  Distance between Person 30 and Person 46: 124 bits
  Distance between Person 30 and Person 47: 137 bits
  Distance between Person 30 and Person 48: 118 bits
  Distance between Person 30 and Person 49: 112 bits
  Distance between Person 30 and Person 50: 131 bits
  Distance between Person 30 and Person 51: 121 bits
  Distance between Person 30 and Person 52: 141 bits
  Distance between Person 30 and Person 53: 136 bits
  Distance between Person 30 and Person 54: 142 bits
  Distance between Person 30 and Person 55: 120 bits
  Distance between Person 30 and Person 56: 139 bits
  Distance between Person 30 and Person 57: 130 bits
  Distance between Person 30 and Person 58: 128 bits
  Distance between Person 30 and Person 59: 121 bits
  Distance between Person 30 and Person 60: 141 bits
  Distance between Person 30 and Person 61: 126 bits
  Distance between Person 30 and Person 62: 140 bits
  Distance between Person 30 and Person 63: 126 bits
  Distance between Person 30 and Person 64: 137 bits
  Distance between Person 30 and Person 65: 134 bits
  Distance between Person 30 and Person 66: 134 bits
  Distance between Person 30 and Person 67: 140 bits
  Distance between Person 30 and Person 68: 124 bits
  Distance between Person 30 and Person 69: 106 bits
  Distance between Person 30 and Person 70: 132 bits
  Distance between Person 30 and Person 71: 133 bits
  Distance between Person 30 and Person 72: 125 bits
  Distance between Person 30 and Person 73: 135 bits
  Distance between Person 30 and Person 74: 124 bits
  Distance between Person 30 and Person 75: 144 bits
  Distance between Person 30 and Person 76: 138 bits
  Distance between Person 30 and Person 77: 138 bits
  Distance between Person 30 and Person 78: 146 bits
  Distance between Person 30 and Person 79: 114 bits
  Distance between Person 30 and Person 80: 140 bits
  Distance between Person 30 and Person 81: 119 bits
  Distance between Person 30 and Person 82: 125 bits
  Distance between Person 30 and Person 83: 119 bits
  Distance between Person 30 and Person 84: 138 bits
  Distance between Person 30 and Person 85: 121 bits
  Distance between Person 30 and Person 86: 130 bits
  Distance between Person 30 and Person 87: 116 bits
  Distance between Person 30 and Person 88: 147 bits
  Distance between Person 30 and Person 89: 131 bits
  Distance between Person 31 and Person 32: 137 bits
  Distance between Person 31 and Person 33: 124 bits
  Distance between Person 31 and Person 34: 136 bits
  Distance between Person 31 and Person 35: 139 bits
  Distance between Person 31 and Person 36: 132 bits
  Distance between Person 31 and Person 37: 132 bits
  Distance between Person 31 and Person 38: 128 bits
  Distance between Person 31 and Person 39: 131 bits
  Distance between Person 31 and Person 40: 125 bits
  Distance between Person 31 and Person 41: 131 bits
  Distance between Person 31 and Person 42: 147 bits
  Distance between Person 31 and Person 43: 135 bits
  Distance between Person 31 and Person 44: 110 bits
  Distance between Person 31 and Person 45: 127 bits
  Distance between Person 31 and Person 46: 127 bits
  Distance between Person 31 and Person 47: 122 bits
  Distance between Person 31 and Person 48: 125 bits
  Distance between Person 31 and Person 49: 123 bits
  Distance between Person 31 and Person 50: 118 bits
  Distance between Person 31 and Person 51: 122 bits
  Distance between Person 31 and Person 52: 138 bits
  Distance between Person 31 and Person 53: 117 bits
  Distance between Person 31 and Person 54: 119 bits
  Distance between Person 31 and Person 55: 123 bits
  Distance between Person 31 and Person 56: 122 bits
  Distance between Person 31 and Person 57: 131 bits
  Distance between Person 31 and Person 58: 119 bits
  Distance between Person 31 and Person 59: 134 bits
  Distance between Person 31 and Person 60: 130 bits
  Distance between Person 31 and Person 61: 119 bits
  Distance between Person 31 and Person 62: 123 bits
  Distance between Person 31 and Person 63: 129 bits
  Distance between Person 31 and Person 64: 124 bits
  Distance between Person 31 and Person 65: 117 bits
  Distance between Person 31 and Person 66: 145 bits
  Distance between Person 31 and Person 67: 97 bits
  Distance between Person 31 and Person 68: 135 bits
  Distance between Person 31 and Person 69: 121 bits
  Distance between Person 31 and Person 70: 115 bits
  Distance between Person 31 and Person 71: 126 bits
  Distance between Person 31 and Person 72: 128 bits
  Distance between Person 31 and Person 73: 122 bits
  Distance between Person 31 and Person 74: 121 bits
  Distance between Person 31 and Person 75: 133 bits
  Distance between Person 31 and Person 76: 121 bits
  Distance between Person 31 and Person 77: 133 bits
  Distance between Person 31 and Person 78: 119 bits
  Distance between Person 31 and Person 79: 139 bits
  Distance between Person 31 and Person 80: 125 bits
  Distance between Person 31 and Person 81: 138 bits
  Distance between Person 31 and Person 82: 124 bits
  Distance between Person 31 and Person 83: 140 bits
  Distance between Person 31 and Person 84: 127 bits
  Distance between Person 31 and Person 85: 128 bits
  Distance between Person 31 and Person 86: 131 bits
  Distance between Person 31 and Person 87: 113 bits
  Distance between Person 31 and Person 88: 108 bits
  Distance between Person 31 and Person 89: 118 bits
  Distance between Person 32 and Person 33: 117 bits
  Distance between Person 32 and Person 34: 97 bits
  Distance between Person 32 and Person 35: 136 bits
  Distance between Person 32 and Person 36: 119 bits
  Distance between Person 32 and Person 37: 111 bits
  Distance between Person 32 and Person 38: 111 bits
  Distance between Person 32 and Person 39: 90 bits
  Distance between Person 32 and Person 40: 76 bits
  Distance between Person 32 and Person 41: 128 bits
  Distance between Person 32 and Person 42: 130 bits
  Distance between Person 32 and Person 43: 114 bits
  Distance between Person 32 and Person 44: 123 bits
  Distance between Person 32 and Person 45: 114 bits
  Distance between Person 32 and Person 46: 118 bits
  Distance between Person 32 and Person 47: 117 bits
  Distance between Person 32 and Person 48: 122 bits
  Distance between Person 32 and Person 49: 146 bits
  Distance between Person 32 and Person 50: 119 bits
  Distance between Person 32 and Person 51: 135 bits
  Distance between Person 32 and Person 52: 117 bits
  Distance between Person 32 and Person 53: 122 bits
  Distance between Person 32 and Person 54: 122 bits
  Distance between Person 32 and Person 55: 108 bits
  Distance between Person 32 and Person 56: 133 bits
  Distance between Person 32 and Person 57: 122 bits
  Distance between Person 32 and Person 58: 114 bits
  Distance between Person 32 and Person 59: 121 bits
  Distance between Person 32 and Person 60: 123 bits
  Distance between Person 32 and Person 61: 112 bits
  Distance between Person 32 and Person 62: 90 bits
  Distance between Person 32 and Person 63: 148 bits
  Distance between Person 32 and Person 64: 109 bits
  Distance between Person 32 and Person 65: 122 bits
  Distance between Person 32 and Person 66: 124 bits
  Distance between Person 32 and Person 67: 142 bits
  Distance between Person 32 and Person 68: 134 bits
  Distance between Person 32 and Person 69: 124 bits
  Distance between Person 32 and Person 70: 116 bits
  Distance between Person 32 and Person 71: 111 bits
  Distance between Person 32 and Person 72: 127 bits
  Distance between Person 32 and Person 73: 99 bits
  Distance between Person 32 and Person 74: 116 bits
  Distance between Person 32 and Person 75: 132 bits
  Distance between Person 32 and Person 76: 110 bits
  Distance between Person 32 and Person 77: 126 bits
  Distance between Person 32 and Person 78: 84 bits
  Distance between Person 32 and Person 79: 138 bits
  Distance between Person 32 and Person 80: 114 bits
  Distance between Person 32 and Person 81: 137 bits
  Distance between Person 32 and Person 82: 127 bits
  Distance between Person 32 and Person 83: 135 bits
  Distance between Person 32 and Person 84: 76 bits
  Distance between Person 32 and Person 85: 127 bits
  Distance between Person 32 and Person 86: 94 bits
  Distance between Person 32 and Person 87: 130 bits
  Distance between Person 32 and Person 88: 123 bits
  Distance between Person 32 and Person 89: 125 bits
  Distance between Person 33 and Person 34: 116 bits
  Distance between Person 33 and Person 35: 123 bits
  Distance between Person 33 and Person 36: 148 bits
  Distance between Person 33 and Person 37: 126 bits
  Distance between Person 33 and Person 38: 132 bits
  Distance between Person 33 and Person 39: 113 bits
  Distance between Person 33 and Person 40: 123 bits
  Distance between Person 33 and Person 41: 133 bits
  Distance between Person 33 and Person 42: 119 bits
  Distance between Person 33 and Person 43: 125 bits
  Distance between Person 33 and Person 44: 110 bits
  Distance between Person 33 and Person 45: 129 bits
  Distance between Person 33 and Person 46: 113 bits
  Distance between Person 33 and Person 47: 112 bits
  Distance between Person 33 and Person 48: 123 bits
  Distance between Person 33 and Person 49: 141 bits
  Distance between Person 33 and Person 50: 126 bits
  Distance between Person 33 and Person 51: 110 bits
  Distance between Person 33 and Person 52: 116 bits
  Distance between Person 33 and Person 53: 111 bits
  Distance between Person 33 and Person 54: 131 bits
  Distance between Person 33 and Person 55: 137 bits
  Distance between Person 33 and Person 56: 138 bits
  Distance between Person 33 and Person 57: 133 bits
  Distance between Person 33 and Person 58: 123 bits
  Distance between Person 33 and Person 59: 128 bits
  Distance between Person 33 and Person 60: 134 bits
  Distance between Person 33 and Person 61: 129 bits
  Distance between Person 33 and Person 62: 127 bits
  Distance between Person 33 and Person 63: 123 bits
  Distance between Person 33 and Person 64: 116 bits
  Distance between Person 33 and Person 65: 121 bits
  Distance between Person 33 and Person 66: 117 bits
  Distance between Person 33 and Person 67: 71 bits
  Distance between Person 33 and Person 68: 143 bits
  Distance between Person 33 and Person 69: 115 bits
  Distance between Person 33 and Person 70: 139 bits
  Distance between Person 33 and Person 71: 120 bits
  Distance between Person 33 and Person 72: 134 bits
  Distance between Person 33 and Person 73: 114 bits
  Distance between Person 33 and Person 74: 125 bits
  Distance between Person 33 and Person 75: 145 bits
  Distance between Person 33 and Person 76: 107 bits
  Distance between Person 33 and Person 77: 133 bits
  Distance between Person 33 and Person 78: 123 bits
  Distance between Person 33 and Person 79: 121 bits
  Distance between Person 33 and Person 80: 121 bits
  Distance between Person 33 and Person 81: 104 bits
  Distance between Person 33 and Person 82: 102 bits
  Distance between Person 33 and Person 83: 130 bits
  Distance between Person 33 and Person 84: 125 bits
  Distance between Person 33 and Person 85: 128 bits
  Distance between Person 33 and Person 86: 129 bits
  Distance between Person 33 and Person 87: 139 bits
  Distance between Person 33 and Person 88: 126 bits
  Distance between Person 33 and Person 89: 122 bits
  Distance between Person 34 and Person 35: 85 bits
  Distance between Person 34 and Person 36: 136 bits
  Distance between Person 34 and Person 37: 106 bits
  Distance between Person 34 and Person 38: 120 bits
  Distance between Person 34 and Person 39: 63 bits
  Distance between Person 34 and Person 40: 83 bits
  Distance between Person 34 and Person 41: 139 bits
  Distance between Person 34 and Person 42: 105 bits
  Distance between Person 34 and Person 43: 123 bits
  Distance between Person 34 and Person 44: 118 bits
  Distance between Person 34 and Person 45: 111 bits
  Distance between Person 34 and Person 46: 109 bits
  Distance between Person 34 and Person 47: 126 bits
  Distance between Person 34 and Person 48: 117 bits
  Distance between Person 34 and Person 49: 131 bits
  Distance between Person 34 and Person 50: 102 bits
  Distance between Person 34 and Person 51: 126 bits
  Distance between Person 34 and Person 52: 90 bits
  Distance between Person 34 and Person 53: 107 bits
  Distance between Person 34 and Person 54: 125 bits
  Distance between Person 34 and Person 55: 121 bits
  Distance between Person 34 and Person 56: 122 bits
  Distance between Person 34 and Person 57: 95 bits
  Distance between Person 34 and Person 58: 123 bits
  Distance between Person 34 and Person 59: 110 bits
  Distance between Person 34 and Person 60: 126 bits
  Distance between Person 34 and Person 61: 115 bits
  Distance between Person 34 and Person 62: 123 bits
  Distance between Person 34 and Person 63: 123 bits
  Distance between Person 34 and Person 64: 132 bits
  Distance between Person 34 and Person 65: 131 bits
  Distance between Person 34 and Person 66: 135 bits
  Distance between Person 34 and Person 67: 119 bits
  Distance between Person 34 and Person 68: 133 bits
  Distance between Person 34 and Person 69: 149 bits
  Distance between Person 34 and Person 70: 125 bits
  Distance between Person 34 and Person 71: 120 bits
  Distance between Person 34 and Person 72: 134 bits
  Distance between Person 34 and Person 73: 126 bits
  Distance between Person 34 and Person 74: 121 bits
  Distance between Person 34 and Person 75: 131 bits
  Distance between Person 34 and Person 76: 87 bits
  Distance between Person 34 and Person 77: 117 bits
  Distance between Person 34 and Person 78: 115 bits
  Distance between Person 34 and Person 79: 133 bits
  Distance between Person 34 and Person 80: 131 bits
  Distance between Person 34 and Person 81: 110 bits
  Distance between Person 34 and Person 82: 122 bits
  Distance between Person 34 and Person 83: 154 bits
  Distance between Person 34 and Person 84: 103 bits
  Distance between Person 34 and Person 85: 124 bits
  Distance between Person 34 and Person 86: 111 bits
  Distance between Person 34 and Person 87: 137 bits
  Distance between Person 34 and Person 88: 116 bits
  Distance between Person 34 and Person 89: 112 bits
  Distance between Person 35 and Person 36: 127 bits
  Distance between Person 35 and Person 37: 99 bits
  Distance between Person 35 and Person 38: 129 bits
  Distance between Person 35 and Person 39: 94 bits
  Distance between Person 35 and Person 40: 124 bits
  Distance between Person 35 and Person 41: 140 bits
  Distance between Person 35 and Person 42: 108 bits
  Distance between Person 35 and Person 43: 132 bits
  Distance between Person 35 and Person 44: 145 bits
  Distance between Person 35 and Person 45: 136 bits
  Distance between Person 35 and Person 46: 126 bits
  Distance between Person 35 and Person 47: 127 bits
  Distance between Person 35 and Person 48: 120 bits
  Distance between Person 35 and Person 49: 112 bits
  Distance between Person 35 and Person 50: 87 bits
  Distance between Person 35 and Person 51: 131 bits
  Distance between Person 35 and Person 52: 127 bits
  Distance between Person 35 and Person 53: 118 bits
  Distance between Person 35 and Person 54: 130 bits
  Distance between Person 35 and Person 55: 150 bits
  Distance between Person 35 and Person 56: 129 bits
  Distance between Person 35 and Person 57: 118 bits
  Distance between Person 35 and Person 58: 142 bits
  Distance between Person 35 and Person 59: 109 bits
  Distance between Person 35 and Person 60: 99 bits
  Distance between Person 35 and Person 61: 110 bits
  Distance between Person 35 and Person 62: 126 bits
  Distance between Person 35 and Person 63: 122 bits
  Distance between Person 35 and Person 64: 143 bits
  Distance between Person 35 and Person 65: 138 bits
  Distance between Person 35 and Person 66: 132 bits
  Distance between Person 35 and Person 67: 130 bits
  Distance between Person 35 and Person 68: 130 bits
  Distance between Person 35 and Person 69: 118 bits
  Distance between Person 35 and Person 70: 138 bits
  Distance between Person 35 and Person 71: 113 bits
  Distance between Person 35 and Person 72: 121 bits
  Distance between Person 35 and Person 73: 137 bits
  Distance between Person 35 and Person 74: 122 bits
  Distance between Person 35 and Person 75: 136 bits
  Distance between Person 35 and Person 76: 112 bits
  Distance between Person 35 and Person 77: 110 bits
  Distance between Person 35 and Person 78: 134 bits
  Distance between Person 35 and Person 79: 106 bits
  Distance between Person 35 and Person 80: 130 bits
  Distance between Person 35 and Person 81: 107 bits
  Distance between Person 35 and Person 82: 135 bits
  Distance between Person 35 and Person 83: 141 bits
  Distance between Person 35 and Person 84: 120 bits
  Distance between Person 35 and Person 85: 119 bits
  Distance between Person 35 and Person 86: 114 bits
  Distance between Person 35 and Person 87: 132 bits
  Distance between Person 35 and Person 88: 119 bits
  Distance between Person 35 and Person 89: 107 bits
  Distance between Person 36 and Person 37: 122 bits
  Distance between Person 36 and Person 38: 124 bits
  Distance between Person 36 and Person 39: 131 bits
  Distance between Person 36 and Person 40: 129 bits
  Distance between Person 36 and Person 41: 143 bits
  Distance between Person 36 and Person 42: 135 bits
  Distance between Person 36 and Person 43: 131 bits
  Distance between Person 36 and Person 44: 122 bits
  Distance between Person 36 and Person 45: 141 bits
  Distance between Person 36 and Person 46: 117 bits
  Distance between Person 36 and Person 47: 132 bits
  Distance between Person 36 and Person 48: 127 bits
  Distance between Person 36 and Person 49: 129 bits
  Distance between Person 36 and Person 50: 116 bits
  Distance between Person 36 and Person 51: 116 bits
  Distance between Person 36 and Person 52: 122 bits
  Distance between Person 36 and Person 53: 133 bits
  Distance between Person 36 and Person 54: 117 bits
  Distance between Person 36 and Person 55: 125 bits
  Distance between Person 36 and Person 56: 114 bits
  Distance between Person 36 and Person 57: 131 bits
  Distance between Person 36 and Person 58: 133 bits
  Distance between Person 36 and Person 59: 136 bits
  Distance between Person 36 and Person 60: 130 bits
  Distance between Person 36 and Person 61: 143 bits
  Distance between Person 36 and Person 62: 141 bits
  Distance between Person 36 and Person 63: 139 bits
  Distance between Person 36 and Person 64: 138 bits
  Distance between Person 36 and Person 65: 111 bits
  Distance between Person 36 and Person 66: 149 bits
  Distance between Person 36 and Person 67: 157 bits
  Distance between Person 36 and Person 68: 125 bits
  Distance between Person 36 and Person 69: 133 bits
  Distance between Person 36 and Person 70: 129 bits
  Distance between Person 36 and Person 71: 134 bits
  Distance between Person 36 and Person 72: 126 bits
  Distance between Person 36 and Person 73: 124 bits
  Distance between Person 36 and Person 74: 137 bits
  Distance between Person 36 and Person 75: 119 bits
  Distance between Person 36 and Person 76: 127 bits
  Distance between Person 36 and Person 77: 123 bits
  Distance between Person 36 and Person 78: 135 bits
  Distance between Person 36 and Person 79: 125 bits
  Distance between Person 36 and Person 80: 125 bits
  Distance between Person 36 and Person 81: 140 bits
  Distance between Person 36 and Person 82: 126 bits
  Distance between Person 36 and Person 83: 144 bits
  Distance between Person 36 and Person 84: 129 bits
  Distance between Person 36 and Person 85: 122 bits
  Distance between Person 36 and Person 86: 137 bits
  Distance between Person 36 and Person 87: 107 bits
  Distance between Person 36 and Person 88: 126 bits
  Distance between Person 36 and Person 89: 130 bits
  Distance between Person 37 and Person 38: 134 bits
  Distance between Person 37 and Person 39: 109 bits
  Distance between Person 37 and Person 40: 121 bits
  Distance between Person 37 and Person 41: 115 bits
  Distance between Person 37 and Person 42: 123 bits
  Distance between Person 37 and Person 43: 141 bits
  Distance between Person 37 and Person 44: 110 bits
  Distance between Person 37 and Person 45: 115 bits
  Distance between Person 37 and Person 46: 125 bits
  Distance between Person 37 and Person 47: 144 bits
  Distance between Person 37 and Person 48: 131 bits
  Distance between Person 37 and Person 49: 141 bits
  Distance between Person 37 and Person 50: 104 bits
  Distance between Person 37 and Person 51: 124 bits
  Distance between Person 37 and Person 52: 132 bits
  Distance between Person 37 and Person 53: 127 bits
  Distance between Person 37 and Person 54: 109 bits
  Distance between Person 37 and Person 55: 135 bits
  Distance between Person 37 and Person 56: 144 bits
  Distance between Person 37 and Person 57: 143 bits
  Distance between Person 37 and Person 58: 131 bits
  Distance between Person 37 and Person 59: 130 bits
  Distance between Person 37 and Person 60: 122 bits
  Distance between Person 37 and Person 61: 119 bits
  Distance between Person 37 and Person 62: 133 bits
  Distance between Person 37 and Person 63: 115 bits
  Distance between Person 37 and Person 64: 122 bits
  Distance between Person 37 and Person 65: 115 bits
  Distance between Person 37 and Person 66: 127 bits
  Distance between Person 37 and Person 67: 135 bits
  Distance between Person 37 and Person 68: 147 bits
  Distance between Person 37 and Person 69: 99 bits
  Distance between Person 37 and Person 70: 141 bits
  Distance between Person 37 and Person 71: 132 bits
  Distance between Person 37 and Person 72: 120 bits
  Distance between Person 37 and Person 73: 120 bits
  Distance between Person 37 and Person 74: 117 bits
  Distance between Person 37 and Person 75: 139 bits
  Distance between Person 37 and Person 76: 127 bits
  Distance between Person 37 and Person 77: 117 bits
  Distance between Person 37 and Person 78: 125 bits
  Distance between Person 37 and Person 79: 123 bits
  Distance between Person 37 and Person 80: 113 bits
  Distance between Person 37 and Person 81: 120 bits
  Distance between Person 37 and Person 82: 116 bits
  Distance between Person 37 and Person 83: 138 bits
  Distance between Person 37 and Person 84: 119 bits
  Distance between Person 37 and Person 85: 106 bits
  Distance between Person 37 and Person 86: 129 bits
  Distance between Person 37 and Person 87: 133 bits
  Distance between Person 37 and Person 88: 130 bits
  Distance between Person 37 and Person 89: 132 bits
  Distance between Person 38 and Person 39: 113 bits
  Distance between Person 38 and Person 40: 127 bits
  Distance between Person 38 and Person 41: 133 bits
  Distance between Person 38 and Person 42: 131 bits
  Distance between Person 38 and Person 43: 113 bits
  Distance between Person 38 and Person 44: 122 bits
  Distance between Person 38 and Person 45: 123 bits
  Distance between Person 38 and Person 46: 121 bits
  Distance between Person 38 and Person 47: 136 bits
  Distance between Person 38 and Person 48: 125 bits
  Distance between Person 38 and Person 49: 125 bits
  Distance between Person 38 and Person 50: 126 bits
  Distance between Person 38 and Person 51: 124 bits
  Distance between Person 38 and Person 52: 130 bits
  Distance between Person 38 and Person 53: 109 bits
  Distance between Person 38 and Person 54: 105 bits
  Distance between Person 38 and Person 55: 107 bits
  Distance between Person 38 and Person 56: 126 bits
  Distance between Person 38 and Person 57: 105 bits
  Distance between Person 38 and Person 58: 127 bits
  Distance between Person 38 and Person 59: 134 bits
  Distance between Person 38 and Person 60: 96 bits
  Distance between Person 38 and Person 61: 105 bits
  Distance between Person 38 and Person 62: 95 bits
  Distance between Person 38 and Person 63: 127 bits
  Distance between Person 38 and Person 64: 136 bits
  Distance between Person 38 and Person 65: 113 bits
  Distance between Person 38 and Person 66: 127 bits
  Distance between Person 38 and Person 67: 141 bits
  Distance between Person 38 and Person 68: 113 bits
  Distance between Person 38 and Person 69: 137 bits
  Distance between Person 38 and Person 70: 71 bits
  Distance between Person 38 and Person 71: 122 bits
  Distance between Person 38 and Person 72: 116 bits
  Distance between Person 38 and Person 73: 106 bits
  Distance between Person 38 and Person 74: 125 bits
  Distance between Person 38 and Person 75: 105 bits
  Distance between Person 38 and Person 76: 117 bits
  Distance between Person 38 and Person 77: 125 bits
  Distance between Person 38 and Person 78: 109 bits
  Distance between Person 38 and Person 79: 147 bits
  Distance between Person 38 and Person 80: 111 bits
  Distance between Person 38 and Person 81: 136 bits
  Distance between Person 38 and Person 82: 146 bits
  Distance between Person 38 and Person 83: 142 bits
  Distance between Person 38 and Person 84: 141 bits
  Distance between Person 38 and Person 85: 120 bits
  Distance between Person 38 and Person 86: 97 bits
  Distance between Person 38 and Person 87: 125 bits
  Distance between Person 38 and Person 88: 118 bits
  Distance between Person 38 and Person 89: 102 bits
  Distance between Person 39 and Person 40: 100 bits
  Distance between Person 39 and Person 41: 130 bits
  Distance between Person 39 and Person 42: 120 bits
  Distance between Person 39 and Person 43: 124 bits
  Distance between Person 39 and Person 44: 109 bits
  Distance between Person 39 and Person 45: 120 bits
  Distance between Person 39 and Person 46: 116 bits
  Distance between Person 39 and Person 47: 109 bits
  Distance between Person 39 and Person 48: 112 bits
  Distance between Person 39 and Person 49: 126 bits
  Distance between Person 39 and Person 50: 121 bits
  Distance between Person 39 and Person 51: 125 bits
  Distance between Person 39 and Person 52: 101 bits
  Distance between Person 39 and Person 53: 122 bits
  Distance between Person 39 and Person 54: 128 bits
  Distance between Person 39 and Person 55: 126 bits
  Distance between Person 39 and Person 56: 119 bits
  Distance between Person 39 and Person 57: 104 bits
  Distance between Person 39 and Person 58: 128 bits
  Distance between Person 39 and Person 59: 111 bits
  Distance between Person 39 and Person 60: 133 bits
  Distance between Person 39 and Person 61: 126 bits
  Distance between Person 39 and Person 62: 112 bits
  Distance between Person 39 and Person 63: 124 bits
  Distance between Person 39 and Person 64: 99 bits
  Distance between Person 39 and Person 65: 140 bits
  Distance between Person 39 and Person 66: 134 bits
  Distance between Person 39 and Person 67: 116 bits
  Distance between Person 39 and Person 68: 130 bits
  Distance between Person 39 and Person 69: 140 bits
  Distance between Person 39 and Person 70: 124 bits
  Distance between Person 39 and Person 71: 115 bits
  Distance between Person 39 and Person 72: 133 bits
  Distance between Person 39 and Person 73: 113 bits
  Distance between Person 39 and Person 74: 118 bits
  Distance between Person 39 and Person 75: 140 bits
  Distance between Person 39 and Person 76: 106 bits
  Distance between Person 39 and Person 77: 96 bits
  Distance between Person 39 and Person 78: 100 bits
  Distance between Person 39 and Person 79: 132 bits
  Distance between Person 39 and Person 80: 122 bits
  Distance between Person 39 and Person 81: 117 bits
  Distance between Person 39 and Person 82: 131 bits
  Distance between Person 39 and Person 83: 147 bits
  Distance between Person 39 and Person 84: 120 bits
  Distance between Person 39 and Person 85: 135 bits
  Distance between Person 39 and Person 86: 114 bits
  Distance between Person 39 and Person 87: 142 bits
  Distance between Person 39 and Person 88: 115 bits
  Distance between Person 39 and Person 89: 125 bits
  Distance between Person 40 and Person 41: 134 bits
  Distance between Person 40 and Person 42: 124 bits
  Distance between Person 40 and Person 43: 100 bits
  Distance between Person 40 and Person 44: 133 bits
  Distance between Person 40 and Person 45: 106 bits
  Distance between Person 40 and Person 46: 114 bits
  Distance between Person 40 and Person 47: 105 bits
  Distance between Person 40 and Person 48: 110 bits
  Distance between Person 40 and Person 49: 136 bits
  Distance between Person 40 and Person 50: 129 bits
  Distance between Person 40 and Person 51: 131 bits
  Distance between Person 40 and Person 52: 117 bits
  Distance between Person 40 and Person 53: 140 bits
  Distance between Person 40 and Person 54: 138 bits
  Distance between Person 40 and Person 55: 106 bits
  Distance between Person 40 and Person 56: 121 bits
  Distance between Person 40 and Person 57: 112 bits
  Distance between Person 40 and Person 58: 122 bits
  Distance between Person 40 and Person 59: 121 bits
  Distance between Person 40 and Person 60: 107 bits
  Distance between Person 40 and Person 61: 118 bits
  Distance between Person 40 and Person 62: 114 bits
  Distance between Person 40 and Person 63: 136 bits
  Distance between Person 40 and Person 64: 129 bits
  Distance between Person 40 and Person 65: 132 bits
  Distance between Person 40 and Person 66: 140 bits
  Distance between Person 40 and Person 67: 118 bits
  Distance between Person 40 and Person 68: 128 bits
  Distance between Person 40 and Person 69: 134 bits
  Distance between Person 40 and Person 70: 126 bits
  Distance between Person 40 and Person 71: 101 bits
  Distance between Person 40 and Person 72: 135 bits
  Distance between Person 40 and Person 73: 125 bits
  Distance between Person 40 and Person 74: 116 bits
  Distance between Person 40 and Person 75: 126 bits
  Distance between Person 40 and Person 76: 122 bits
  Distance between Person 40 and Person 77: 114 bits
  Distance between Person 40 and Person 78: 106 bits
  Distance between Person 40 and Person 79: 130 bits
  Distance between Person 40 and Person 80: 126 bits
  Distance between Person 40 and Person 81: 115 bits
  Distance between Person 40 and Person 82: 127 bits
  Distance between Person 40 and Person 83: 135 bits
  Distance between Person 40 and Person 84: 102 bits
  Distance between Person 40 and Person 85: 105 bits
  Distance between Person 40 and Person 86: 116 bits
  Distance between Person 40 and Person 87: 140 bits
  Distance between Person 40 and Person 88: 125 bits
  Distance between Person 40 and Person 89: 117 bits
  Distance between Person 41 and Person 42: 134 bits
  Distance between Person 41 and Person 43: 124 bits
  Distance between Person 41 and Person 44: 139 bits
  Distance between Person 41 and Person 45: 106 bits
  Distance between Person 41 and Person 46: 136 bits
  Distance between Person 41 and Person 47: 119 bits
  Distance between Person 41 and Person 48: 118 bits
  Distance between Person 41 and Person 49: 128 bits
  Distance between Person 41 and Person 50: 125 bits
  Distance between Person 41 and Person 51: 141 bits
  Distance between Person 41 and Person 52: 115 bits
  Distance between Person 41 and Person 53: 120 bits
  Distance between Person 41 and Person 54: 130 bits
  Distance between Person 41 and Person 55: 108 bits
  Distance between Person 41 and Person 56: 129 bits
  Distance between Person 41 and Person 57: 124 bits
  Distance between Person 41 and Person 58: 126 bits
  Distance between Person 41 and Person 59: 119 bits
  Distance between Person 41 and Person 60: 133 bits
  Distance between Person 41 and Person 61: 120 bits
  Distance between Person 41 and Person 62: 118 bits
  Distance between Person 41 and Person 63: 116 bits
  Distance between Person 41 and Person 64: 121 bits
  Distance between Person 41 and Person 65: 112 bits
  Distance between Person 41 and Person 66: 116 bits
  Distance between Person 41 and Person 67: 134 bits
  Distance between Person 41 and Person 68: 134 bits
  Distance between Person 41 and Person 69: 126 bits
  Distance between Person 41 and Person 70: 116 bits
  Distance between Person 41 and Person 71: 129 bits
  Distance between Person 41 and Person 72: 125 bits
  Distance between Person 41 and Person 73: 137 bits
  Distance between Person 41 and Person 74: 130 bits
  Distance between Person 41 and Person 75: 126 bits
  Distance between Person 41 and Person 76: 126 bits
  Distance between Person 41 and Person 77: 102 bits
  Distance between Person 41 and Person 78: 118 bits
  Distance between Person 41 and Person 79: 120 bits
  Distance between Person 41 and Person 80: 136 bits
  Distance between Person 41 and Person 81: 137 bits
  Distance between Person 41 and Person 82: 127 bits
  Distance between Person 41 and Person 83: 111 bits
  Distance between Person 41 and Person 84: 122 bits
  Distance between Person 41 and Person 85: 127 bits
  Distance between Person 41 and Person 86: 124 bits
  Distance between Person 41 and Person 87: 132 bits
  Distance between Person 41 and Person 88: 131 bits
  Distance between Person 41 and Person 89: 129 bits
  Distance between Person 42 and Person 43: 108 bits
  Distance between Person 42 and Person 44: 125 bits
  Distance between Person 42 and Person 45: 96 bits
  Distance between Person 42 and Person 46: 108 bits
  Distance between Person 42 and Person 47: 103 bits
  Distance between Person 42 and Person 48: 130 bits
  Distance between Person 42 and Person 49: 130 bits
  Distance between Person 42 and Person 50: 123 bits
  Distance between Person 42 and Person 51: 109 bits
  Distance between Person 42 and Person 52: 141 bits
  Distance between Person 42 and Person 53: 114 bits
  Distance between Person 42 and Person 54: 114 bits
  Distance between Person 42 and Person 55: 108 bits
  Distance between Person 42 and Person 56: 127 bits
  Distance between Person 42 and Person 57: 106 bits
  Distance between Person 42 and Person 58: 126 bits
  Distance between Person 42 and Person 59: 63 bits
  Distance between Person 42 and Person 60: 117 bits
  Distance between Person 42 and Person 61: 124 bits
  Distance between Person 42 and Person 62: 128 bits
  Distance between Person 42 and Person 63: 122 bits
  Distance between Person 42 and Person 64: 139 bits
  Distance between Person 42 and Person 65: 134 bits
  Distance between Person 42 and Person 66: 134 bits
  Distance between Person 42 and Person 67: 112 bits
  Distance between Person 42 and Person 68: 130 bits
  Distance between Person 42 and Person 69: 114 bits
  Distance between Person 42 and Person 70: 116 bits
  Distance between Person 42 and Person 71: 131 bits
  Distance between Person 42 and Person 72: 129 bits
  Distance between Person 42 and Person 73: 139 bits
  Distance between Person 42 and Person 74: 104 bits
  Distance between Person 42 and Person 75: 128 bits
  Distance between Person 42 and Person 76: 118 bits
  Distance between Person 42 and Person 77: 126 bits
  Distance between Person 42 and Person 78: 114 bits
  Distance between Person 42 and Person 79: 106 bits
  Distance between Person 42 and Person 80: 128 bits
  Distance between Person 42 and Person 81: 105 bits
  Distance between Person 42 and Person 82: 121 bits
  Distance between Person 42 and Person 83: 133 bits
  Distance between Person 42 and Person 84: 124 bits
  Distance between Person 42 and Person 85: 133 bits
  Distance between Person 42 and Person 86: 122 bits
  Distance between Person 42 and Person 87: 136 bits
  Distance between Person 42 and Person 88: 127 bits
  Distance between Person 42 and Person 89: 127 bits
  Distance between Person 43 and Person 44: 133 bits
  Distance between Person 43 and Person 45: 130 bits
  Distance between Person 43 and Person 46: 120 bits
  Distance between Person 43 and Person 47: 117 bits
  Distance between Person 43 and Person 48: 110 bits
  Distance between Person 43 and Person 49: 128 bits
  Distance between Person 43 and Person 50: 135 bits
  Distance between Person 43 and Person 51: 113 bits
  Distance between Person 43 and Person 52: 131 bits
  Distance between Person 43 and Person 53: 120 bits
  Distance between Person 43 and Person 54: 118 bits
  Distance between Person 43 and Person 55: 112 bits
  Distance between Person 43 and Person 56: 115 bits
  Distance between Person 43 and Person 57: 144 bits
  Distance between Person 43 and Person 58: 132 bits
  Distance between Person 43 and Person 59: 125 bits
  Distance between Person 43 and Person 60: 107 bits
  Distance between Person 43 and Person 61: 118 bits
  Distance between Person 43 and Person 62: 88 bits
  Distance between Person 43 and Person 63: 128 bits
  Distance between Person 43 and Person 64: 123 bits
  Distance between Person 43 and Person 65: 146 bits
  Distance between Person 43 and Person 66: 140 bits
  Distance between Person 43 and Person 67: 122 bits
  Distance between Person 43 and Person 68: 138 bits
  Distance between Person 43 and Person 69: 130 bits
  Distance between Person 43 and Person 70: 76 bits
  Distance between Person 43 and Person 71: 129 bits
  Distance between Person 43 and Person 72: 129 bits
  Distance between Person 43 and Person 73: 135 bits
  Distance between Person 43 and Person 74: 118 bits
  Distance between Person 43 and Person 75: 108 bits
  Distance between Person 43 and Person 76: 122 bits
  Distance between Person 43 and Person 77: 114 bits
  Distance between Person 43 and Person 78: 94 bits
  Distance between Person 43 and Person 79: 132 bits
  Distance between Person 43 and Person 80: 124 bits
  Distance between Person 43 and Person 81: 101 bits
  Distance between Person 43 and Person 82: 113 bits
  Distance between Person 43 and Person 83: 143 bits
  Distance between Person 43 and Person 84: 132 bits
  Distance between Person 43 and Person 85: 117 bits
  Distance between Person 43 and Person 86: 130 bits
  Distance between Person 43 and Person 87: 146 bits
  Distance between Person 43 and Person 88: 129 bits
  Distance between Person 43 and Person 89: 125 bits
  Distance between Person 44 and Person 45: 103 bits
  Distance between Person 44 and Person 46: 83 bits
  Distance between Person 44 and Person 47: 94 bits
  Distance between Person 44 and Person 48: 131 bits
  Distance between Person 44 and Person 49: 151 bits
  Distance between Person 44 and Person 50: 108 bits
  Distance between Person 44 and Person 51: 112 bits
  Distance between Person 44 and Person 52: 134 bits
  Distance between Person 44 and Person 53: 97 bits
  Distance between Person 44 and Person 54: 105 bits
  Distance between Person 44 and Person 55: 107 bits
  Distance between Person 44 and Person 56: 118 bits
  Distance between Person 44 and Person 57: 141 bits
  Distance between Person 44 and Person 58: 97 bits
  Distance between Person 44 and Person 59: 126 bits
  Distance between Person 44 and Person 60: 136 bits
  Distance between Person 44 and Person 61: 115 bits
  Distance between Person 44 and Person 62: 137 bits
  Distance between Person 44 and Person 63: 127 bits
  Distance between Person 44 and Person 64: 108 bits
  Distance between Person 44 and Person 65: 107 bits
  Distance between Person 44 and Person 66: 125 bits
  Distance between Person 44 and Person 67: 109 bits
  Distance between Person 44 and Person 68: 121 bits
  Distance between Person 44 and Person 69: 121 bits
  Distance between Person 44 and Person 70: 111 bits
  Distance between Person 44 and Person 71: 150 bits
  Distance between Person 44 and Person 72: 124 bits
  Distance between Person 44 and Person 73: 120 bits
  Distance between Person 44 and Person 74: 115 bits
  Distance between Person 44 and Person 75: 129 bits
  Distance between Person 44 and Person 76: 115 bits
  Distance between Person 44 and Person 77: 111 bits
  Distance between Person 44 and Person 78: 123 bits
  Distance between Person 44 and Person 79: 129 bits
  Distance between Person 44 and Person 80: 125 bits
  Distance between Person 44 and Person 81: 140 bits
  Distance between Person 44 and Person 82: 94 bits
  Distance between Person 44 and Person 83: 140 bits
  Distance between Person 44 and Person 84: 113 bits
  Distance between Person 44 and Person 85: 116 bits
  Distance between Person 44 and Person 86: 117 bits
  Distance between Person 44 and Person 87: 125 bits
  Distance between Person 44 and Person 88: 136 bits
  Distance between Person 44 and Person 89: 128 bits
  Distance between Person 45 and Person 46: 74 bits
  Distance between Person 45 and Person 47: 85 bits
  Distance between Person 45 and Person 48: 128 bits
  Distance between Person 45 and Person 49: 140 bits
  Distance between Person 45 and Person 50: 111 bits
  Distance between Person 45 and Person 51: 127 bits
  Distance between Person 45 and Person 52: 125 bits
  Distance between Person 45 and Person 53: 118 bits
  Distance between Person 45 and Person 54: 134 bits
  Distance between Person 45 and Person 55: 80 bits
  Distance between Person 45 and Person 56: 111 bits
  Distance between Person 45 and Person 57: 118 bits
  Distance between Person 45 and Person 58: 108 bits
  Distance between Person 45 and Person 59: 117 bits
  Distance between Person 45 and Person 60: 129 bits
  Distance between Person 45 and Person 61: 98 bits
  Distance between Person 45 and Person 62: 130 bits
  Distance between Person 45 and Person 63: 110 bits
  Distance between Person 45 and Person 64: 143 bits
  Distance between Person 45 and Person 65: 126 bits
  Distance between Person 45 and Person 66: 140 bits
  Distance between Person 45 and Person 67: 128 bits
  Distance between Person 45 and Person 68: 144 bits
  Distance between Person 45 and Person 69: 128 bits
  Distance between Person 45 and Person 70: 116 bits
  Distance between Person 45 and Person 71: 127 bits
  Distance between Person 45 and Person 72: 131 bits
  Distance between Person 45 and Person 73: 125 bits
  Distance between Person 45 and Person 74: 116 bits
  Distance between Person 45 and Person 75: 114 bits
  Distance between Person 45 and Person 76: 112 bits
  Distance between Person 45 and Person 77: 128 bits
  Distance between Person 45 and Person 78: 112 bits
  Distance between Person 45 and Person 79: 130 bits
  Distance between Person 45 and Person 80: 120 bits
  Distance between Person 45 and Person 81: 125 bits
  Distance between Person 45 and Person 82: 117 bits
  Distance between Person 45 and Person 83: 133 bits
  Distance between Person 45 and Person 84: 112 bits
  Distance between Person 45 and Person 85: 115 bits
  Distance between Person 45 and Person 86: 96 bits
  Distance between Person 45 and Person 87: 148 bits
  Distance between Person 45 and Person 88: 125 bits
  Distance between Person 45 and Person 89: 105 bits
  Distance between Person 46 and Person 47: 111 bits
  Distance between Person 46 and Person 48: 124 bits
  Distance between Person 46 and Person 49: 128 bits
  Distance between Person 46 and Person 50: 115 bits
  Distance between Person 46 and Person 51: 107 bits
  Distance between Person 46 and Person 52: 129 bits
  Distance between Person 46 and Person 53: 120 bits
  Distance between Person 46 and Person 54: 122 bits
  Distance between Person 46 and Person 55: 114 bits
  Distance between Person 46 and Person 56: 123 bits
  Distance between Person 46 and Person 57: 124 bits
  Distance between Person 46 and Person 58: 80 bits
  Distance between Person 46 and Person 59: 113 bits
  Distance between Person 46 and Person 60: 123 bits
  Distance between Person 46 and Person 61: 98 bits
  Distance between Person 46 and Person 62: 124 bits
  Distance between Person 46 and Person 63: 124 bits
  Distance between Person 46 and Person 64: 149 bits
  Distance between Person 46 and Person 65: 122 bits
  Distance between Person 46 and Person 66: 144 bits
  Distance between Person 46 and Person 67: 126 bits
  Distance between Person 46 and Person 68: 130 bits
  Distance between Person 46 and Person 69: 128 bits
  Distance between Person 46 and Person 70: 122 bits
  Distance between Person 46 and Person 71: 135 bits
  Distance between Person 46 and Person 72: 121 bits
  Distance between Person 46 and Person 73: 115 bits
  Distance between Person 46 and Person 74: 128 bits
  Distance between Person 46 and Person 75: 134 bits
  Distance between Person 46 and Person 76: 108 bits
  Distance between Person 46 and Person 77: 120 bits
  Distance between Person 46 and Person 78: 120 bits
  Distance between Person 46 and Person 79: 132 bits
  Distance between Person 46 and Person 80: 124 bits
  Distance between Person 46 and Person 81: 113 bits
  Distance between Person 46 and Person 82: 107 bits
  Distance between Person 46 and Person 83: 145 bits
  Distance between Person 46 and Person 84: 108 bits
  Distance between Person 46 and Person 85: 117 bits
  Distance between Person 46 and Person 86: 102 bits
  Distance between Person 46 and Person 87: 144 bits
  Distance between Person 46 and Person 88: 139 bits
  Distance between Person 46 and Person 89: 99 bits
  Distance between Person 47 and Person 48: 121 bits
  Distance between Person 47 and Person 49: 135 bits
  Distance between Person 47 and Person 50: 108 bits
  Distance between Person 47 and Person 51: 144 bits
  Distance between Person 47 and Person 52: 138 bits
  Distance between Person 47 and Person 53: 105 bits
  Distance between Person 47 and Person 54: 121 bits
  Distance between Person 47 and Person 55: 85 bits
  Distance between Person 47 and Person 56: 106 bits
  Distance between Person 47 and Person 57: 121 bits
  Distance between Person 47 and Person 58: 109 bits
  Distance between Person 47 and Person 59: 110 bits
  Distance between Person 47 and Person 60: 116 bits
  Distance between Person 47 and Person 61: 117 bits
  Distance between Person 47 and Person 62: 131 bits
  Distance between Person 47 and Person 63: 135 bits
  Distance between Person 47 and Person 64: 112 bits
  Distance between Person 47 and Person 65: 121 bits
  Distance between Person 47 and Person 66: 119 bits
  Distance between Person 47 and Person 67: 111 bits
  Distance between Person 47 and Person 68: 137 bits
  Distance between Person 47 and Person 69: 115 bits
  Distance between Person 47 and Person 70: 119 bits
  Distance between Person 47 and Person 71: 110 bits
  Distance between Person 47 and Person 72: 118 bits
  Distance between Person 47 and Person 73: 136 bits
  Distance between Person 47 and Person 74: 117 bits
  Distance between Person 47 and Person 75: 123 bits
  Distance between Person 47 and Person 76: 107 bits
  Distance between Person 47 and Person 77: 109 bits
  Distance between Person 47 and Person 78: 125 bits
  Distance between Person 47 and Person 79: 91 bits
  Distance between Person 47 and Person 80: 131 bits
  Distance between Person 47 and Person 81: 126 bits
  Distance between Person 47 and Person 82: 112 bits
  Distance between Person 47 and Person 83: 122 bits
  Distance between Person 47 and Person 84: 119 bits
  Distance between Person 47 and Person 85: 130 bits
  Distance between Person 47 and Person 86: 117 bits
  Distance between Person 47 and Person 87: 127 bits
  Distance between Person 47 and Person 88: 116 bits
  Distance between Person 47 and Person 89: 118 bits
  Distance between Person 48 and Person 49: 116 bits
  Distance between Person 48 and Person 50: 119 bits
  Distance between Person 48 and Person 51: 115 bits
  Distance between Person 48 and Person 52: 129 bits
  Distance between Person 48 and Person 53: 132 bits
  Distance between Person 48 and Person 54: 132 bits
  Distance between Person 48 and Person 55: 116 bits
  Distance between Person 48 and Person 56: 133 bits
  Distance between Person 48 and Person 57: 122 bits
  Distance between Person 48 and Person 58: 128 bits
  Distance between Person 48 and Person 59: 115 bits
  Distance between Person 48 and Person 60: 119 bits
  Distance between Person 48 and Person 61: 112 bits
  Distance between Person 48 and Person 62: 108 bits
  Distance between Person 48 and Person 63: 120 bits
  Distance between Person 48 and Person 64: 119 bits
  Distance between Person 48 and Person 65: 124 bits
  Distance between Person 48 and Person 66: 112 bits
  Distance between Person 48 and Person 67: 136 bits
  Distance between Person 48 and Person 68: 126 bits
  Distance between Person 48 and Person 69: 126 bits
  Distance between Person 48 and Person 70: 106 bits
  Distance between Person 48 and Person 71: 129 bits
  Distance between Person 48 and Person 72: 107 bits
  Distance between Person 48 and Person 73: 141 bits
  Distance between Person 48 and Person 74: 120 bits
  Distance between Person 48 and Person 75: 124 bits
  Distance between Person 48 and Person 76: 106 bits
  Distance between Person 48 and Person 77: 112 bits
  Distance between Person 48 and Person 78: 106 bits
  Distance between Person 48 and Person 79: 130 bits
  Distance between Person 48 and Person 80: 132 bits
  Distance between Person 48 and Person 81: 111 bits
  Distance between Person 48 and Person 82: 153 bits
  Distance between Person 48 and Person 83: 139 bits
  Distance between Person 48 and Person 84: 104 bits
  Distance between Person 48 and Person 85: 103 bits
  Distance between Person 48 and Person 86: 118 bits
  Distance between Person 48 and Person 87: 122 bits
  Distance between Person 48 and Person 88: 143 bits
  Distance between Person 48 and Person 89: 119 bits
  Distance between Person 49 and Person 50: 129 bits
  Distance between Person 49 and Person 51: 139 bits
  Distance between Person 49 and Person 52: 131 bits
  Distance between Person 49 and Person 53: 138 bits
  Distance between Person 49 and Person 54: 126 bits
  Distance between Person 49 and Person 55: 130 bits
  Distance between Person 49 and Person 56: 131 bits
  Distance between Person 49 and Person 57: 124 bits
  Distance between Person 49 and Person 58: 126 bits
  Distance between Person 49 and Person 59: 119 bits
  Distance between Person 49 and Person 60: 107 bits
  Distance between Person 49 and Person 61: 122 bits
  Distance between Person 49 and Person 62: 134 bits
  Distance between Person 49 and Person 63: 136 bits
  Distance between Person 49 and Person 64: 125 bits
  Distance between Person 49 and Person 65: 130 bits
  Distance between Person 49 and Person 66: 136 bits
  Distance between Person 49 and Person 67: 134 bits
  Distance between Person 49 and Person 68: 138 bits
  Distance between Person 49 and Person 69: 136 bits
  Distance between Person 49 and Person 70: 128 bits
  Distance between Person 49 and Person 71: 111 bits
  Distance between Person 49 and Person 72: 103 bits
  Distance between Person 49 and Person 73: 153 bits
  Distance between Person 49 and Person 74: 124 bits
  Distance between Person 49 and Person 75: 128 bits
  Distance between Person 49 and Person 76: 130 bits
  Distance between Person 49 and Person 77: 128 bits
  Distance between Person 49 and Person 78: 134 bits
  Distance between Person 49 and Person 79: 118 bits
  Distance between Person 49 and Person 80: 112 bits
  Distance between Person 49 and Person 81: 125 bits
  Distance between Person 49 and Person 82: 159 bits
  Distance between Person 49 and Person 83: 127 bits
  Distance between Person 49 and Person 84: 144 bits
  Distance between Person 49 and Person 85: 131 bits
  Distance between Person 49 and Person 86: 124 bits
  Distance between Person 49 and Person 87: 126 bits
  Distance between Person 49 and Person 88: 125 bits
  Distance between Person 49 and Person 89: 131 bits
  Distance between Person 50 and Person 51: 124 bits
  Distance between Person 50 and Person 52: 132 bits
  Distance between Person 50 and Person 53: 105 bits
  Distance between Person 50 and Person 54: 109 bits
  Distance between Person 50 and Person 55: 129 bits
  Distance between Person 50 and Person 56: 116 bits
  Distance between Person 50 and Person 57: 117 bits
  Distance between Person 50 and Person 58: 133 bits
  Distance between Person 50 and Person 59: 118 bits
  Distance between Person 50 and Person 60: 98 bits
  Distance between Person 50 and Person 61: 107 bits
  Distance between Person 50 and Person 62: 113 bits
  Distance between Person 50 and Person 63: 135 bits
  Distance between Person 50 and Person 64: 132 bits
  Distance between Person 50 and Person 65: 123 bits
  Distance between Person 50 and Person 66: 137 bits
  Distance between Person 50 and Person 67: 137 bits
  Distance between Person 50 and Person 68: 127 bits
  Distance between Person 50 and Person 69: 103 bits
  Distance between Person 50 and Person 70: 117 bits
  Distance between Person 50 and Person 71: 132 bits
  Distance between Person 50 and Person 72: 114 bits
  Distance between Person 50 and Person 73: 124 bits
  Distance between Person 50 and Person 74: 125 bits
  Distance between Person 50 and Person 75: 109 bits
  Distance between Person 50 and Person 76: 75 bits
  Distance between Person 50 and Person 77: 115 bits
  Distance between Person 50 and Person 78: 125 bits
  Distance between Person 50 and Person 79: 115 bits
  Distance between Person 50 and Person 80: 109 bits
  Distance between Person 50 and Person 81: 132 bits
  Distance between Person 50 and Person 82: 98 bits
  Distance between Person 50 and Person 83: 146 bits
  Distance between Person 50 and Person 84: 95 bits
  Distance between Person 50 and Person 85: 118 bits
  Distance between Person 50 and Person 86: 121 bits
  Distance between Person 50 and Person 87: 115 bits
  Distance between Person 50 and Person 88: 116 bits
  Distance between Person 50 and Person 89: 116 bits
  Distance between Person 51 and Person 52: 128 bits
  Distance between Person 51 and Person 53: 111 bits
  Distance between Person 51 and Person 54: 107 bits
  Distance between Person 51 and Person 55: 131 bits
  Distance between Person 51 and Person 56: 118 bits
  Distance between Person 51 and Person 57: 139 bits
  Distance between Person 51 and Person 58: 125 bits
  Distance between Person 51 and Person 59: 118 bits
  Distance between Person 51 and Person 60: 140 bits
  Distance between Person 51 and Person 61: 121 bits
  Distance between Person 51 and Person 62: 121 bits
  Distance between Person 51 and Person 63: 119 bits
  Distance between Person 51 and Person 64: 112 bits
  Distance between Person 51 and Person 65: 145 bits
  Distance between Person 51 and Person 66: 143 bits
  Distance between Person 51 and Person 67: 131 bits
  Distance between Person 51 and Person 68: 135 bits
  Distance between Person 51 and Person 69: 109 bits
  Distance between Person 51 and Person 70: 119 bits
  Distance between Person 51 and Person 71: 156 bits
  Distance between Person 51 and Person 72: 142 bits
  Distance between Person 51 and Person 73: 100 bits
  Distance between Person 51 and Person 74: 117 bits
  Distance between Person 51 and Person 75: 137 bits
  Distance between Person 51 and Person 76: 127 bits
  Distance between Person 51 and Person 77: 117 bits
  Distance between Person 51 and Person 78: 125 bits
  Distance between Person 51 and Person 79: 135 bits
  Distance between Person 51 and Person 80: 103 bits
  Distance between Person 51 and Person 81: 120 bits
  Distance between Person 51 and Person 82: 96 bits
  Distance between Person 51 and Person 83: 130 bits
  Distance between Person 51 and Person 84: 131 bits
  Distance between Person 51 and Person 85: 124 bits
  Distance between Person 51 and Person 86: 139 bits
  Distance between Person 51 and Person 87: 131 bits
  Distance between Person 51 and Person 88: 136 bits
  Distance between Person 51 and Person 89: 118 bits
  Distance between Person 52 and Person 53: 127 bits
  Distance between Person 52 and Person 54: 121 bits
  Distance between Person 52 and Person 55: 121 bits
  Distance between Person 52 and Person 56: 138 bits
  Distance between Person 52 and Person 57: 119 bits
  Distance between Person 52 and Person 58: 127 bits
  Distance between Person 52 and Person 59: 140 bits
  Distance between Person 52 and Person 60: 152 bits
  Distance between Person 52 and Person 61: 117 bits
  Distance between Person 52 and Person 62: 125 bits
  Distance between Person 52 and Person 63: 137 bits
  Distance between Person 52 and Person 64: 122 bits
  Distance between Person 52 and Person 65: 139 bits
  Distance between Person 52 and Person 66: 115 bits
  Distance between Person 52 and Person 67: 127 bits
  Distance between Person 52 and Person 68: 131 bits
  Distance between Person 52 and Person 69: 139 bits
  Distance between Person 52 and Person 70: 125 bits
  Distance between Person 52 and Person 71: 130 bits
  Distance between Person 52 and Person 72: 132 bits
  Distance between Person 52 and Person 73: 132 bits
  Distance between Person 52 and Person 74: 119 bits
  Distance between Person 52 and Person 75: 137 bits
  Distance between Person 52 and Person 76: 107 bits
  Distance between Person 52 and Person 77: 129 bits
  Distance between Person 52 and Person 78: 125 bits
  Distance between Person 52 and Person 79: 131 bits
  Distance between Person 52 and Person 80: 139 bits
  Distance between Person 52 and Person 81: 136 bits
  Distance between Person 52 and Person 82: 138 bits
  Distance between Person 52 and Person 83: 134 bits
  Distance between Person 52 and Person 84: 131 bits
  Distance between Person 52 and Person 85: 132 bits
  Distance between Person 52 and Person 86: 121 bits
  Distance between Person 52 and Person 87: 127 bits
  Distance between Person 52 and Person 88: 116 bits
  Distance between Person 52 and Person 89: 120 bits
  Distance between Person 53 and Person 54: 98 bits
  Distance between Person 53 and Person 55: 102 bits
  Distance between Person 53 and Person 56: 117 bits
  Distance between Person 53 and Person 57: 124 bits
  Distance between Person 53 and Person 58: 126 bits
  Distance between Person 53 and Person 59: 133 bits
  Distance between Person 53 and Person 60: 129 bits
  Distance between Person 53 and Person 61: 94 bits
  Distance between Person 53 and Person 62: 134 bits
  Distance between Person 53 and Person 63: 120 bits
  Distance between Person 53 and Person 64: 127 bits
  Distance between Person 53 and Person 65: 118 bits
  Distance between Person 53 and Person 66: 122 bits
  Distance between Person 53 and Person 67: 112 bits
  Distance between Person 53 and Person 68: 148 bits
  Distance between Person 53 and Person 69: 118 bits
  Distance between Person 53 and Person 70: 110 bits
  Distance between Person 53 and Person 71: 139 bits
  Distance between Person 53 and Person 72: 129 bits
  Distance between Person 53 and Person 73: 121 bits
  Distance between Person 53 and Person 74: 128 bits
  Distance between Person 53 and Person 75: 128 bits
  Distance between Person 53 and Person 76: 104 bits
  Distance between Person 53 and Person 77: 128 bits
  Distance between Person 53 and Person 78: 126 bits
  Distance between Person 53 and Person 79: 128 bits
  Distance between Person 53 and Person 80: 120 bits
  Distance between Person 53 and Person 81: 127 bits
  Distance between Person 53 and Person 82: 111 bits
  Distance between Person 53 and Person 83: 137 bits
  Distance between Person 53 and Person 84: 112 bits
  Distance between Person 53 and Person 85: 127 bits
  Distance between Person 53 and Person 86: 114 bits
  Distance between Person 53 and Person 87: 132 bits
  Distance between Person 53 and Person 88: 111 bits
  Distance between Person 53 and Person 89: 103 bits
  Distance between Person 54 and Person 55: 122 bits
  Distance between Person 54 and Person 56: 119 bits
  Distance between Person 54 and Person 57: 130 bits
  Distance between Person 54 and Person 58: 104 bits
  Distance between Person 54 and Person 59: 119 bits
  Distance between Person 54 and Person 60: 123 bits
  Distance between Person 54 and Person 61: 136 bits
  Distance between Person 54 and Person 62: 120 bits
  Distance between Person 54 and Person 63: 118 bits
  Distance between Person 54 and Person 64: 123 bits
  Distance between Person 54 and Person 65: 130 bits
  Distance between Person 54 and Person 66: 134 bits
  Distance between Person 54 and Person 67: 132 bits
  Distance between Person 54 and Person 68: 132 bits
  Distance between Person 54 and Person 69: 114 bits
  Distance between Person 54 and Person 70: 118 bits
  Distance between Person 54 and Person 71: 141 bits
  Distance between Person 54 and Person 72: 113 bits
  Distance between Person 54 and Person 73: 117 bits
  Distance between Person 54 and Person 74: 116 bits
  Distance between Person 54 and Person 75: 126 bits
  Distance between Person 54 and Person 76: 94 bits
  Distance between Person 54 and Person 77: 134 bits
  Distance between Person 54 and Person 78: 124 bits
  Distance between Person 54 and Person 79: 138 bits
  Distance between Person 54 and Person 80: 126 bits
  Distance between Person 54 and Person 81: 125 bits
  Distance between Person 54 and Person 82: 119 bits
  Distance between Person 54 and Person 83: 143 bits
  Distance between Person 54 and Person 84: 134 bits
  Distance between Person 54 and Person 85: 131 bits
  Distance between Person 54 and Person 86: 128 bits
  Distance between Person 54 and Person 87: 124 bits
  Distance between Person 54 and Person 88: 113 bits
  Distance between Person 54 and Person 89: 115 bits
  Distance between Person 55 and Person 56: 129 bits
  Distance between Person 55 and Person 57: 98 bits
  Distance between Person 55 and Person 58: 114 bits
  Distance between Person 55 and Person 59: 135 bits
  Distance between Person 55 and Person 60: 129 bits
  Distance between Person 55 and Person 61: 98 bits
  Distance between Person 55 and Person 62: 130 bits
  Distance between Person 55 and Person 63: 132 bits
  Distance between Person 55 and Person 64: 125 bits
  Distance between Person 55 and Person 65: 122 bits
  Distance between Person 55 and Person 66: 112 bits
  Distance between Person 55 and Person 67: 132 bits
  Distance between Person 55 and Person 68: 154 bits
  Distance between Person 55 and Person 69: 120 bits
  Distance between Person 55 and Person 70: 102 bits
  Distance between Person 55 and Person 71: 127 bits
  Distance between Person 55 and Person 72: 125 bits
  Distance between Person 55 and Person 73: 139 bits
  Distance between Person 55 and Person 74: 124 bits
  Distance between Person 55 and Person 75: 126 bits
  Distance between Person 55 and Person 76: 122 bits
  Distance between Person 55 and Person 77: 116 bits
  Distance between Person 55 and Person 78: 112 bits
  Distance between Person 55 and Person 79: 122 bits
  Distance between Person 55 and Person 80: 116 bits
  Distance between Person 55 and Person 81: 139 bits
  Distance between Person 55 and Person 82: 127 bits
  Distance between Person 55 and Person 83: 133 bits
  Distance between Person 55 and Person 84: 128 bits
  Distance between Person 55 and Person 85: 115 bits
  Distance between Person 55 and Person 86: 80 bits
  Distance between Person 55 and Person 87: 140 bits
  Distance between Person 55 and Person 88: 135 bits
  Distance between Person 55 and Person 89: 129 bits
  Distance between Person 56 and Person 57: 133 bits
  Distance between Person 56 and Person 58: 119 bits
  Distance between Person 56 and Person 59: 134 bits
  Distance between Person 56 and Person 60: 132 bits
  Distance between Person 56 and Person 61: 139 bits
  Distance between Person 56 and Person 62: 129 bits
  Distance between Person 56 and Person 63: 129 bits
  Distance between Person 56 and Person 64: 132 bits
  Distance between Person 56 and Person 65: 127 bits
  Distance between Person 56 and Person 66: 147 bits
  Distance between Person 56 and Person 67: 139 bits
  Distance between Person 56 and Person 68: 127 bits
  Distance between Person 56 and Person 69: 137 bits
  Distance between Person 56 and Person 70: 133 bits
  Distance between Person 56 and Person 71: 134 bits
  Distance between Person 56 and Person 72: 144 bits
  Distance between Person 56 and Person 73: 132 bits
  Distance between Person 56 and Person 74: 127 bits
  Distance between Person 56 and Person 75: 123 bits
  Distance between Person 56 and Person 76: 129 bits
  Distance between Person 56 and Person 77: 121 bits
  Distance between Person 56 and Person 78: 143 bits
  Distance between Person 56 and Person 79: 125 bits
  Distance between Person 56 and Person 80: 115 bits
  Distance between Person 56 and Person 81: 130 bits
  Distance between Person 56 and Person 82: 114 bits
  Distance between Person 56 and Person 83: 130 bits
  Distance between Person 56 and Person 84: 123 bits
  Distance between Person 56 and Person 85: 128 bits
  Distance between Person 56 and Person 86: 133 bits
  Distance between Person 56 and Person 87: 131 bits
  Distance between Person 56 and Person 88: 118 bits
  Distance between Person 56 and Person 89: 126 bits
  Distance between Person 57 and Person 58: 138 bits
  Distance between Person 57 and Person 59: 121 bits
  Distance between Person 57 and Person 60: 121 bits
  Distance between Person 57 and Person 61: 126 bits
  Distance between Person 57 and Person 62: 132 bits
  Distance between Person 57 and Person 63: 126 bits
  Distance between Person 57 and Person 64: 131 bits
  Distance between Person 57 and Person 65: 126 bits
  Distance between Person 57 and Person 66: 130 bits
  Distance between Person 57 and Person 67: 126 bits
  Distance between Person 57 and Person 68: 146 bits
  Distance between Person 57 and Person 69: 134 bits
  Distance between Person 57 and Person 70: 124 bits
  Distance between Person 57 and Person 71: 119 bits
  Distance between Person 57 and Person 72: 125 bits
  Distance between Person 57 and Person 73: 139 bits
  Distance between Person 57 and Person 74: 134 bits
  Distance between Person 57 and Person 75: 138 bits
  Distance between Person 57 and Person 76: 120 bits
  Distance between Person 57 and Person 77: 128 bits
  Distance between Person 57 and Person 78: 112 bits
  Distance between Person 57 and Person 79: 130 bits
  Distance between Person 57 and Person 80: 134 bits
  Distance between Person 57 and Person 81: 123 bits
  Distance between Person 57 and Person 82: 137 bits
  Distance between Person 57 and Person 83: 131 bits
  Distance between Person 57 and Person 84: 130 bits
  Distance between Person 57 and Person 85: 119 bits
  Distance between Person 57 and Person 86: 120 bits
  Distance between Person 57 and Person 87: 132 bits
  Distance between Person 57 and Person 88: 125 bits
  Distance between Person 57 and Person 89: 125 bits
  Distance between Person 58 and Person 59: 115 bits
  Distance between Person 58 and Person 60: 131 bits
  Distance between Person 58 and Person 61: 114 bits
  Distance between Person 58 and Person 62: 120 bits
  Distance between Person 58 and Person 63: 128 bits
  Distance between Person 58 and Person 64: 137 bits
  Distance between Person 58 and Person 65: 114 bits
  Distance between Person 58 and Person 66: 118 bits
  Distance between Person 58 and Person 67: 122 bits
  Distance between Person 58 and Person 68: 128 bits
  Distance between Person 58 and Person 69: 126 bits
  Distance between Person 58 and Person 70: 136 bits
  Distance between Person 58 and Person 71: 121 bits
  Distance between Person 58 and Person 72: 107 bits
  Distance between Person 58 and Person 73: 123 bits
  Distance between Person 58 and Person 74: 126 bits
  Distance between Person 58 and Person 75: 124 bits
  Distance between Person 58 and Person 76: 116 bits
  Distance between Person 58 and Person 77: 142 bits
  Distance between Person 58 and Person 78: 130 bits
  Distance between Person 58 and Person 79: 128 bits
  Distance between Person 58 and Person 80: 116 bits
  Distance between Person 58 and Person 81: 117 bits
  Distance between Person 58 and Person 82: 121 bits
  Distance between Person 58 and Person 83: 129 bits
  Distance between Person 58 and Person 84: 116 bits
  Distance between Person 58 and Person 85: 119 bits
  Distance between Person 58 and Person 86: 108 bits
  Distance between Person 58 and Person 87: 120 bits
  Distance between Person 58 and Person 88: 137 bits
  Distance between Person 58 and Person 89: 125 bits
  Distance between Person 59 and Person 60: 106 bits
  Distance between Person 59 and Person 61: 137 bits
  Distance between Person 59 and Person 62: 109 bits
  Distance between Person 59 and Person 63: 115 bits
  Distance between Person 59 and Person 64: 124 bits
  Distance between Person 59 and Person 65: 129 bits
  Distance between Person 59 and Person 66: 135 bits
  Distance between Person 59 and Person 67: 121 bits
  Distance between Person 59 and Person 68: 105 bits
  Distance between Person 59 and Person 69: 131 bits
  Distance between Person 59 and Person 70: 121 bits
  Distance between Person 59 and Person 71: 118 bits
  Distance between Person 59 and Person 72: 134 bits
  Distance between Person 59 and Person 73: 146 bits
  Distance between Person 59 and Person 74: 129 bits
  Distance between Person 59 and Person 75: 139 bits
  Distance between Person 59 and Person 76: 109 bits
  Distance between Person 59 and Person 77: 107 bits
  Distance between Person 59 and Person 78: 115 bits
  Distance between Person 59 and Person 79: 111 bits
  Distance between Person 59 and Person 80: 155 bits
  Distance between Person 59 and Person 81: 116 bits
  Distance between Person 59 and Person 82: 144 bits
  Distance between Person 59 and Person 83: 132 bits
  Distance between Person 59 and Person 84: 115 bits
  Distance between Person 59 and Person 85: 136 bits
  Distance between Person 59 and Person 86: 135 bits
  Distance between Person 59 and Person 87: 125 bits
  Distance between Person 59 and Person 88: 128 bits
  Distance between Person 59 and Person 89: 136 bits
  Distance between Person 60 and Person 61: 121 bits
  Distance between Person 60 and Person 62: 85 bits
  Distance between Person 60 and Person 63: 137 bits
  Distance between Person 60 and Person 64: 130 bits
  Distance between Person 60 and Person 65: 127 bits
  Distance between Person 60 and Person 66: 143 bits
  Distance between Person 60 and Person 67: 125 bits
  Distance between Person 60 and Person 68: 115 bits
  Distance between Person 60 and Person 69: 127 bits
  Distance between Person 60 and Person 70: 87 bits
  Distance between Person 60 and Person 71: 108 bits
  Distance between Person 60 and Person 72: 126 bits
  Distance between Person 60 and Person 73: 130 bits
  Distance between Person 60 and Person 74: 125 bits
  Distance between Person 60 and Person 75: 111 bits
  Distance between Person 60 and Person 76: 125 bits
  Distance between Person 60 and Person 77: 111 bits
  Distance between Person 60 and Person 78: 109 bits
  Distance between Person 60 and Person 79: 119 bits
  Distance between Person 60 and Person 80: 127 bits
  Distance between Person 60 and Person 81: 128 bits
  Distance between Person 60 and Person 82: 152 bits
  Distance between Person 60 and Person 83: 134 bits
  Distance between Person 60 and Person 84: 123 bits
  Distance between Person 60 and Person 85: 112 bits
  Distance between Person 60 and Person 86: 123 bits
  Distance between Person 60 and Person 87: 141 bits
  Distance between Person 60 and Person 88: 114 bits
  Distance between Person 60 and Person 89: 110 bits
  Distance between Person 61 and Person 62: 112 bits
  Distance between Person 61 and Person 63: 140 bits
  Distance between Person 61 and Person 64: 125 bits
  Distance between Person 61 and Person 65: 120 bits
  Distance between Person 61 and Person 66: 116 bits
  Distance between Person 61 and Person 67: 132 bits
  Distance between Person 61 and Person 68: 132 bits
  Distance between Person 61 and Person 69: 114 bits
  Distance between Person 61 and Person 70: 112 bits
  Distance between Person 61 and Person 71: 113 bits
  Distance between Person 61 and Person 72: 105 bits
  Distance between Person 61 and Person 73: 113 bits
  Distance between Person 61 and Person 74: 120 bits
  Distance between Person 61 and Person 75: 126 bits
  Distance between Person 61 and Person 76: 114 bits
  Distance between Person 61 and Person 77: 126 bits
  Distance between Person 61 and Person 78: 92 bits
  Distance between Person 61 and Person 79: 116 bits
  Distance between Person 61 and Person 80: 104 bits
  Distance between Person 61 and Person 81: 137 bits
  Distance between Person 61 and Person 82: 125 bits
  Distance between Person 61 and Person 83: 139 bits
  Distance between Person 61 and Person 84: 104 bits
  Distance between Person 61 and Person 85: 105 bits
  Distance between Person 61 and Person 86: 88 bits
  Distance between Person 61 and Person 87: 132 bits
  Distance between Person 61 and Person 88: 111 bits
  Distance between Person 61 and Person 89: 105 bits
  Distance between Person 62 and Person 63: 126 bits
  Distance between Person 62 and Person 64: 111 bits
  Distance between Person 62 and Person 65: 136 bits
  Distance between Person 62 and Person 66: 146 bits
  Distance between Person 62 and Person 67: 140 bits
  Distance between Person 62 and Person 68: 118 bits
  Distance between Person 62 and Person 69: 114 bits
  Distance between Person 62 and Person 70: 66 bits
  Distance between Person 62 and Person 71: 111 bits
  Distance between Person 62 and Person 72: 135 bits
  Distance between Person 62 and Person 73: 121 bits
  Distance between Person 62 and Person 74: 132 bits
  Distance between Person 62 and Person 75: 118 bits
  Distance between Person 62 and Person 76: 106 bits
  Distance between Person 62 and Person 77: 122 bits
  Distance between Person 62 and Person 78: 70 bits
  Distance between Person 62 and Person 79: 138 bits
  Distance between Person 62 and Person 80: 136 bits
  Distance between Person 62 and Person 81: 115 bits
  Distance between Person 62 and Person 82: 127 bits
  Distance between Person 62 and Person 83: 135 bits
  Distance between Person 62 and Person 84: 100 bits
  Distance between Person 62 and Person 85: 121 bits
  Distance between Person 62 and Person 86: 118 bits
  Distance between Person 62 and Person 87: 118 bits
  Distance between Person 62 and Person 88: 109 bits
  Distance between Person 62 and Person 89: 91 bits
  Distance between Person 63 and Person 64: 135 bits
  Distance between Person 63 and Person 65: 124 bits
  Distance between Person 63 and Person 66: 120 bits
  Distance between Person 63 and Person 67: 112 bits
  Distance between Person 63 and Person 68: 136 bits
  Distance between Person 63 and Person 69: 130 bits
  Distance between Person 63 and Person 70: 132 bits
  Distance between Person 63 and Person 71: 137 bits
  Distance between Person 63 and Person 72: 151 bits
  Distance between Person 63 and Person 73: 135 bits
  Distance between Person 63 and Person 74: 126 bits
  Distance between Person 63 and Person 75: 116 bits
  Distance between Person 63 and Person 76: 106 bits
  Distance between Person 63 and Person 77: 132 bits
  Distance between Person 63 and Person 78: 126 bits
  Distance between Person 63 and Person 79: 148 bits
  Distance between Person 63 and Person 80: 150 bits
  Distance between Person 63 and Person 81: 67 bits
  Distance between Person 63 and Person 82: 119 bits
  Distance between Person 63 and Person 83: 123 bits
  Distance between Person 63 and Person 84: 142 bits
  Distance between Person 63 and Person 85: 125 bits
  Distance between Person 63 and Person 86: 114 bits
  Distance between Person 63 and Person 87: 130 bits
  Distance between Person 63 and Person 88: 129 bits
  Distance between Person 63 and Person 89: 111 bits
  Distance between Person 64 and Person 65: 123 bits
  Distance between Person 64 and Person 66: 123 bits
  Distance between Person 64 and Person 67: 125 bits
  Distance between Person 64 and Person 68: 133 bits
  Distance between Person 64 and Person 69: 87 bits
  Distance between Person 64 and Person 70: 123 bits
  Distance between Person 64 and Person 71: 122 bits
  Distance between Person 64 and Person 72: 126 bits
  Distance between Person 64 and Person 73: 126 bits
  Distance between Person 64 and Person 74: 119 bits
  Distance between Person 64 and Person 75: 119 bits
  Distance between Person 64 and Person 76: 123 bits
  Distance between Person 64 and Person 77: 119 bits
  Distance between Person 64 and Person 78: 117 bits
  Distance between Person 64 and Person 79: 121 bits
  Distance between Person 64 and Person 80: 117 bits
  Distance between Person 64 and Person 81: 136 bits
  Distance between Person 64 and Person 82: 124 bits
  Distance between Person 64 and Person 83: 128 bits
  Distance between Person 64 and Person 84: 113 bits
  Distance between Person 64 and Person 85: 130 bits
  Distance between Person 64 and Person 86: 135 bits
  Distance between Person 64 and Person 87: 121 bits
  Distance between Person 64 and Person 88: 126 bits
  Distance between Person 64 and Person 89: 142 bits
  Distance between Person 65 and Person 66: 112 bits
  Distance between Person 65 and Person 67: 144 bits
  Distance between Person 65 and Person 68: 120 bits
  Distance between Person 65 and Person 69: 120 bits
  Distance between Person 65 and Person 70: 132 bits
  Distance between Person 65 and Person 71: 129 bits
  Distance between Person 65 and Person 72: 127 bits
  Distance between Person 65 and Person 73: 125 bits
  Distance between Person 65 and Person 74: 132 bits
  Distance between Person 65 and Person 75: 114 bits
  Distance between Person 65 and Person 76: 126 bits
  Distance between Person 65 and Person 77: 124 bits
  Distance between Person 65 and Person 78: 136 bits
  Distance between Person 65 and Person 79: 116 bits
  Distance between Person 65 and Person 80: 140 bits
  Distance between Person 65 and Person 81: 153 bits
  Distance between Person 65 and Person 82: 115 bits
  Distance between Person 65 and Person 83: 125 bits
  Distance between Person 65 and Person 84: 104 bits
  Distance between Person 65 and Person 85: 105 bits
  Distance between Person 65 and Person 86: 134 bits
  Distance between Person 65 and Person 87: 114 bits
  Distance between Person 65 and Person 88: 131 bits
  Distance between Person 65 and Person 89: 127 bits
  Distance between Person 66 and Person 67: 132 bits
  Distance between Person 66 and Person 68: 140 bits
  Distance between Person 66 and Person 69: 122 bits
  Distance between Person 66 and Person 70: 140 bits
  Distance between Person 66 and Person 71: 131 bits
  Distance between Person 66 and Person 72: 119 bits
  Distance between Person 66 and Person 73: 147 bits
  Distance between Person 66 and Person 74: 128 bits
  Distance between Person 66 and Person 75: 128 bits
  Distance between Person 66 and Person 76: 126 bits
  Distance between Person 66 and Person 77: 130 bits
  Distance between Person 66 and Person 78: 124 bits
  Distance between Person 66 and Person 79: 122 bits
  Distance between Person 66 and Person 80: 124 bits
  Distance between Person 66 and Person 81: 111 bits
  Distance between Person 66 and Person 82: 127 bits
  Distance between Person 66 and Person 83: 129 bits
  Distance between Person 66 and Person 84: 138 bits
  Distance between Person 66 and Person 85: 123 bits
  Distance between Person 66 and Person 86: 112 bits
  Distance between Person 66 and Person 87: 130 bits
  Distance between Person 66 and Person 88: 153 bits
  Distance between Person 66 and Person 89: 153 bits
  Distance between Person 67 and Person 68: 134 bits
  Distance between Person 67 and Person 69: 132 bits
  Distance between Person 67 and Person 70: 134 bits
  Distance between Person 67 and Person 71: 123 bits
  Distance between Person 67 and Person 72: 135 bits
  Distance between Person 67 and Person 73: 129 bits
  Distance between Person 67 and Person 74: 116 bits
  Distance between Person 67 and Person 75: 134 bits
  Distance between Person 67 and Person 76: 114 bits
  Distance between Person 67 and Person 77: 132 bits
  Distance between Person 67 and Person 78: 126 bits
  Distance between Person 67 and Person 79: 130 bits
  Distance between Person 67 and Person 80: 124 bits
  Distance between Person 67 and Person 81: 105 bits
  Distance between Person 67 and Person 82: 117 bits
  Distance between Person 67 and Person 83: 149 bits
  Distance between Person 67 and Person 84: 134 bits
  Distance between Person 67 and Person 85: 125 bits
  Distance between Person 67 and Person 86: 126 bits
  Distance between Person 67 and Person 87: 140 bits
  Distance between Person 67 and Person 88: 119 bits
  Distance between Person 67 and Person 89: 125 bits
  Distance between Person 68 and Person 69: 138 bits
  Distance between Person 68 and Person 70: 110 bits
  Distance between Person 68 and Person 71: 145 bits
  Distance between Person 68 and Person 72: 115 bits
  Distance between Person 68 and Person 73: 99 bits
  Distance between Person 68 and Person 74: 126 bits
  Distance between Person 68 and Person 75: 132 bits
  Distance between Person 68 and Person 76: 116 bits
  Distance between Person 68 and Person 77: 128 bits
  Distance between Person 68 and Person 78: 138 bits
  Distance between Person 68 and Person 79: 122 bits
  Distance between Person 68 and Person 80: 134 bits
  Distance between Person 68 and Person 81: 139 bits
  Distance between Person 68 and Person 82: 131 bits
  Distance between Person 68 and Person 83: 131 bits
  Distance between Person 68 and Person 84: 108 bits
  Distance between Person 68 and Person 85: 133 bits
  Distance between Person 68 and Person 86: 144 bits
  Distance between Person 68 and Person 87: 110 bits
  Distance between Person 68 and Person 88: 131 bits
  Distance between Person 68 and Person 89: 119 bits
  Distance between Person 69 and Person 70: 130 bits
  Distance between Person 69 and Person 71: 127 bits
  Distance between Person 69 and Person 72: 119 bits
  Distance between Person 69 and Person 73: 117 bits
  Distance between Person 69 and Person 74: 122 bits
  Distance between Person 69 and Person 75: 110 bits
  Distance between Person 69 and Person 76: 124 bits
  Distance between Person 69 and Person 77: 132 bits
  Distance between Person 69 and Person 78: 130 bits
  Distance between Person 69 and Person 79: 104 bits
  Distance between Person 69 and Person 80: 114 bits
  Distance between Person 69 and Person 81: 123 bits
  Distance between Person 69 and Person 82: 103 bits
  Distance between Person 69 and Person 83: 133 bits
  Distance between Person 69 and Person 84: 106 bits
  Distance between Person 69 and Person 85: 117 bits
  Distance between Person 69 and Person 86: 112 bits
  Distance between Person 69 and Person 87: 120 bits
  Distance between Person 69 and Person 88: 129 bits
  Distance between Person 69 and Person 89: 121 bits
  Distance between Person 70 and Person 71: 139 bits
  Distance between Person 70 and Person 72: 121 bits
  Distance between Person 70 and Person 73: 135 bits
  Distance between Person 70 and Person 74: 116 bits
  Distance between Person 70 and Person 75: 98 bits
  Distance between Person 70 and Person 76: 118 bits
  Distance between Person 70 and Person 77: 108 bits
  Distance between Person 70 and Person 78: 92 bits
  Distance between Person 70 and Person 79: 134 bits
  Distance between Person 70 and Person 80: 126 bits
  Distance between Person 70 and Person 81: 133 bits
  Distance between Person 70 and Person 82: 139 bits
  Distance between Person 70 and Person 83: 143 bits
  Distance between Person 70 and Person 84: 122 bits
  Distance between Person 70 and Person 85: 121 bits
  Distance between Person 70 and Person 86: 118 bits
  Distance between Person 70 and Person 87: 118 bits
  Distance between Person 70 and Person 88: 123 bits
  Distance between Person 70 and Person 89: 105 bits
  Distance between Person 71 and Person 72: 142 bits
  Distance between Person 71 and Person 73: 148 bits
  Distance between Person 71 and Person 74: 139 bits
  Distance between Person 71 and Person 75: 127 bits
  Distance between Person 71 and Person 76: 115 bits
  Distance between Person 71 and Person 77: 121 bits
  Distance between Person 71 and Person 78: 125 bits
  Distance between Person 71 and Person 79: 111 bits
  Distance between Person 71 and Person 80: 137 bits
  Distance between Person 71 and Person 81: 124 bits
  Distance between Person 71 and Person 82: 146 bits
  Distance between Person 71 and Person 83: 124 bits
  Distance between Person 71 and Person 84: 131 bits
  Distance between Person 71 and Person 85: 116 bits
  Distance between Person 71 and Person 86: 121 bits
  Distance between Person 71 and Person 87: 135 bits
  Distance between Person 71 and Person 88: 78 bits
  Distance between Person 71 and Person 89: 122 bits
  Distance between Person 72 and Person 73: 120 bits
  Distance between Person 72 and Person 74: 127 bits
  Distance between Person 72 and Person 75: 125 bits
  Distance between Person 72 and Person 76: 129 bits
  Distance between Person 72 and Person 77: 133 bits
  Distance between Person 72 and Person 78: 107 bits
  Distance between Person 72 and Person 79: 123 bits
  Distance between Person 72 and Person 80: 99 bits
  Distance between Person 72 and Person 81: 142 bits
  Distance between Person 72 and Person 82: 146 bits
  Distance between Person 72 and Person 83: 138 bits
  Distance between Person 72 and Person 84: 121 bits
  Distance between Person 72 and Person 85: 118 bits
  Distance between Person 72 and Person 86: 121 bits
  Distance between Person 72 and Person 87: 105 bits
  Distance between Person 72 and Person 88: 140 bits
  Distance between Person 72 and Person 89: 128 bits
  Distance between Person 73 and Person 74: 129 bits
  Distance between Person 73 and Person 75: 145 bits
  Distance between Person 73 and Person 76: 133 bits
  Distance between Person 73 and Person 77: 137 bits
  Distance between Person 73 and Person 78: 113 bits
  Distance between Person 73 and Person 79: 157 bits
  Distance between Person 73 and Person 80: 83 bits
  Distance between Person 73 and Person 81: 136 bits
  Distance between Person 73 and Person 82: 114 bits
  Distance between Person 73 and Person 83: 140 bits
  Distance between Person 73 and Person 84: 113 bits
  Distance between Person 73 and Person 85: 134 bits
  Distance between Person 73 and Person 86: 115 bits
  Distance between Person 73 and Person 87: 133 bits
  Distance between Person 73 and Person 88: 128 bits
  Distance between Person 73 and Person 89: 100 bits
  Distance between Person 74 and Person 75: 126 bits
  Distance between Person 74 and Person 76: 118 bits
  Distance between Person 74 and Person 77: 136 bits
  Distance between Person 74 and Person 78: 112 bits
  Distance between Person 74 and Person 79: 108 bits
  Distance between Person 74 and Person 80: 128 bits
  Distance between Person 74 and Person 81: 137 bits
  Distance between Person 74 and Person 82: 141 bits
  Distance between Person 74 and Person 83: 125 bits
  Distance between Person 74 and Person 84: 114 bits
  Distance between Person 74 and Person 85: 123 bits
  Distance between Person 74 and Person 86: 126 bits
  Distance between Person 74 and Person 87: 132 bits
  Distance between Person 74 and Person 88: 111 bits
  Distance between Person 74 and Person 89: 127 bits
  Distance between Person 75 and Person 76: 108 bits
  Distance between Person 75 and Person 77: 118 bits
  Distance between Person 75 and Person 78: 134 bits
  Distance between Person 75 and Person 79: 132 bits
  Distance between Person 75 and Person 80: 112 bits
  Distance between Person 75 and Person 81: 131 bits
  Distance between Person 75 and Person 82: 135 bits
  Distance between Person 75 and Person 83: 141 bits
  Distance between Person 75 and Person 84: 118 bits
  Distance between Person 75 and Person 85: 131 bits
  Distance between Person 75 and Person 86: 124 bits
  Distance between Person 75 and Person 87: 112 bits
  Distance between Person 75 and Person 88: 117 bits
  Distance between Person 75 and Person 89: 111 bits
  Distance between Person 76 and Person 77: 124 bits
  Distance between Person 76 and Person 78: 126 bits
  Distance between Person 76 and Person 79: 122 bits
  Distance between Person 76 and Person 80: 132 bits
  Distance between Person 76 and Person 81: 111 bits
  Distance between Person 76 and Person 82: 111 bits
  Distance between Person 76 and Person 83: 151 bits
  Distance between Person 76 and Person 84: 100 bits
  Distance between Person 76 and Person 85: 141 bits
  Distance between Person 76 and Person 86: 130 bits
  Distance between Person 76 and Person 87: 116 bits
  Distance between Person 76 and Person 88: 115 bits
  Distance between Person 76 and Person 89: 107 bits
  Distance between Person 77 and Person 78: 116 bits
  Distance between Person 77 and Person 79: 122 bits
  Distance between Person 77 and Person 80: 128 bits
  Distance between Person 77 and Person 81: 135 bits
  Distance between Person 77 and Person 82: 137 bits
  Distance between Person 77 and Person 83: 131 bits
  Distance between Person 77 and Person 84: 128 bits
  Distance between Person 77 and Person 85: 127 bits
  Distance between Person 77 and Person 86: 116 bits
  Distance between Person 77 and Person 87: 126 bits
  Distance between Person 77 and Person 88: 131 bits
  Distance between Person 77 and Person 89: 129 bits
  Distance between Person 78 and Person 79: 146 bits
  Distance between Person 78 and Person 80: 122 bits
  Distance between Person 78 and Person 81: 111 bits
  Distance between Person 78 and Person 82: 133 bits
  Distance between Person 78 and Person 83: 133 bits
  Distance between Person 78 and Person 84: 120 bits
  Distance between Person 78 and Person 85: 101 bits
  Distance between Person 78 and Person 86: 98 bits
  Distance between Person 78 and Person 87: 128 bits
  Distance between Person 78 and Person 88: 101 bits
  Distance between Person 78 and Person 89: 121 bits
  Distance between Person 79 and Person 80: 140 bits
  Distance between Person 79 and Person 81: 139 bits
  Distance between Person 79 and Person 82: 125 bits
  Distance between Person 79 and Person 83: 123 bits
  Distance between Person 79 and Person 84: 112 bits
  Distance between Person 79 and Person 85: 121 bits
  Distance between Person 79 and Person 86: 132 bits
  Distance between Person 79 and Person 87: 122 bits
  Distance between Person 79 and Person 88: 109 bits
  Distance between Person 79 and Person 89: 131 bits
  Distance between Person 80 and Person 81: 137 bits
  Distance between Person 80 and Person 82: 131 bits
  Distance between Person 80 and Person 83: 147 bits
  Distance between Person 80 and Person 84: 124 bits
  Distance between Person 80 and Person 85: 133 bits
  Distance between Person 80 and Person 86: 112 bits
  Distance between Person 80 and Person 87: 146 bits
  Distance between Person 80 and Person 88: 133 bits
  Distance between Person 80 and Person 89: 141 bits
  Distance between Person 81 and Person 82: 120 bits
  Distance between Person 81 and Person 83: 132 bits
  Distance between Person 81 and Person 84: 145 bits
  Distance between Person 81 and Person 85: 128 bits
  Distance between Person 81 and Person 86: 119 bits
  Distance between Person 81 and Person 87: 141 bits
  Distance between Person 81 and Person 88: 140 bits
  Distance between Person 81 and Person 89: 112 bits
  Distance between Person 82 and Person 83: 122 bits
  Distance between Person 82 and Person 84: 113 bits
  Distance between Person 82 and Person 85: 130 bits
  Distance between Person 82 and Person 86: 141 bits
  Distance between Person 82 and Person 87: 125 bits
  Distance between Person 82 and Person 88: 132 bits
  Distance between Person 82 and Person 89: 116 bits
  Distance between Person 83 and Person 84: 127 bits
  Distance between Person 83 and Person 85: 128 bits
  Distance between Person 83 and Person 86: 145 bits
  Distance between Person 83 and Person 87: 125 bits
  Distance between Person 83 and Person 88: 134 bits
  Distance between Person 83 and Person 89: 132 bits
  Distance between Person 84 and Person 85: 113 bits
  Distance between Person 84 and Person 86: 110 bits
  Distance between Person 84 and Person 87: 118 bits
  Distance between Person 84 and Person 88: 121 bits
  Distance between Person 84 and Person 89: 99 bits
  Distance between Person 85 and Person 86: 109 bits
  Distance between Person 85 and Person 87: 137 bits
  Distance between Person 85 and Person 88: 118 bits
  Distance between Person 85 and Person 89: 118 bits
  Distance between Person 86 and Person 87: 136 bits
  Distance between Person 86 and Person 88: 119 bits
  Distance between Person 86 and Person 89: 115 bits
  Distance between Person 87 and Person 88: 115 bits
  Distance between Person 87 and Person 89: 119 bits
  Distance between Person 88 and Person 89: 100 bits
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