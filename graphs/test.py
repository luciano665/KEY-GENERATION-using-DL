import re
import numpy as np
import matplotlib.pyplot as plt


text = r"""

Person 1:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 1 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1]...
  Ground Truth:   [1 1 0 1 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1]...
16/16 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step
  Intra-person average Hamming distance: 4.73 bits
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step

Person 2:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 1 1 1 0]...
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
  Intra-person average Hamming distance: 5.56 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 3:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 1 1 0 1 1 0 1 0 0 1 1 0 1 0 1 1 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 0 1 1 0 1 0 0 1 1 0 1 0 1 1 0 1 1 0 1 0]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 20.74 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 4:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [1 1 1 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 0]...
  Ground Truth:   [1 1 1 1 0 1 1 1 0 0 1 1 0 1 0 1 1 1 1 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 41.09 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 

Person 5:
  Aggregated Key Accuracy: 96.48%
  Aggregated Key: [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 1 0 0]...
  Ground Truth:   [0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step 
  Intra-person average Hamming distance: 18.43 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 6:
  Aggregated Key Accuracy: 74.61%
  Aggregated Key: [1 0 0 1 0 1 1 0 1 0 1 0 1 1 0 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
  Intra-person average Hamming distance: 73.30 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 7:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [0 1 1 0 0 1 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0]...
  Ground Truth:   [0 1 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 1 0 0 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 76.66 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 8:
  Aggregated Key Accuracy: 99.22%
  Aggregated Key: [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 0 0 1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 20.72 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 9:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 1 1 0 0 1 1 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1]...
  Ground Truth:   [1 0 1 1 0 0 1 1 0 1 0 0 1 1 0 1 1 1 0 0 1 0 0 1]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 9.30 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 10:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 1 1 1]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 1.73 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 11:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
  Ground Truth:   [0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 6.33 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 12:
  Aggregated Key Accuracy: 94.92%
  Aggregated Key: [0 0 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
  Ground Truth:   [0 1 1 1 1 0 0 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 20.02 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 

Person 13:
  Aggregated Key Accuracy: 77.73%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 0 1 0 0 1 0 1 1 1 0 1 0 0 0 0]...
  Ground Truth:   [0 1 0 1 1 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step 
  Intra-person average Hamming distance: 38.99 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 14:
  Aggregated Key Accuracy: 87.50%
  Aggregated Key: [1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
  Ground Truth:   [1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 1 0 1 1 1 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 61.81 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 15:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [1 1 0 1 0 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 0 0 0]...
  Ground Truth:   [1 1 0 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 36.83 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 16:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 0 0 1]...
  Ground Truth:   [0 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 0 0 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 31.71 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 17:
  Aggregated Key Accuracy: 90.23%
  Aggregated Key: [1 1 0 1 1 0 0 0 1 1 1 0 1 1 0 0 0 1 1 0 0 0 1 1]...
  Ground Truth:   [1 0 0 0 1 0 0 0 1 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 39.80 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 18:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 0 0 1 1 0 1 0 1 0 0 0 1 0 1 0 0 1 1 1 1 0 1 0]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
  Intra-person average Hamming distance: 47.05 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 19:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 0 0 0 0 1 1 0]...
  Ground Truth:   [1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0 0 1 0 0 0 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 61.87 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 20:
  Aggregated Key Accuracy: 93.75%
  Aggregated Key: [1 0 1 1 1 0 0 1 1 1 1 1 0 1 0 1 1 0 0 0 0 0 0 1]...
  Ground Truth:   [1 0 1 0 1 0 0 1 1 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 46.72 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 21:
  Aggregated Key Accuracy: 96.09%
  Aggregated Key: [1 0 0 0 0 1 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 1 1]...
  Ground Truth:   [1 0 0 0 0 1 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 48.29 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 22:
  Aggregated Key Accuracy: 99.22%
  Aggregated Key: [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
  Ground Truth:   [0 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 25.62 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 

Person 23:
  Aggregated Key Accuracy: 97.27%
  Aggregated Key: [1 0 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1]...
  Ground Truth:   [1 0 0 0 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 1 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
  Intra-person average Hamming distance: 19.94 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 24:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 1 1 0]...
  Ground Truth:   [0 0 1 0 1 0 0 0 0 1 1 1 0 0 0 1 0 1 1 0 0 1 1 0]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 18.18 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 25:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 1 1 1 0 0 0 0 0 1 1 1 1 1 0 0 0 1 0 1 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 9.59 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 26:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0]...
  Ground Truth:   [0 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0 0 1 1 0 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 5.13 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 27:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 0 1 0 1 1 0 0 0 1 1 1 1 0 1 0 1 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 15.08 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 28:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 2.94 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 29:
  Aggregated Key Accuracy: 99.22%
  Aggregated Key: [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
  Ground Truth:   [1 0 1 1 1 1 0 1 1 1 0 1 1 0 0 0 0 0 0 1 1 0 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 18.47 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 30:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1]...
  Ground Truth:   [0 0 1 1 1 0 1 1 0 0 1 0 0 1 1 0 1 0 1 0 0 0 0 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 0.20 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 31:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1]...
  Ground Truth:   [0 1 1 1 0 0 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 7.54 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 32:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0]...
  Ground Truth:   [1 1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 1 0 0 0 1 0]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 6.06 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 33:
  Aggregated Key Accuracy: 83.98%
  Aggregated Key: [1 1 1 0 1 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0 1 0 0 0]...
  Ground Truth:   [1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 0 1 0 0 1 0 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 51.23 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 34:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 1 0 1 1 1 0 0]...
  Ground Truth:   [1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 0 0 1 0 1 1 1 0 0]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 5.21 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 35:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
  Ground Truth:   [1 0 0 0 1 0 1 1 1 1 1 0 0 0 1 1 0 1 0 1 0 0 0 1]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 2.39 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 36:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0]...
  Ground Truth:   [0 1 0 1 1 0 0 0 1 0 1 1 0 0 1 1 1 1 1 1 0 0 1 0]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
  Intra-person average Hamming distance: 6.83 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 37:
  Aggregated Key Accuracy: 84.77%
  Aggregated Key: [1 0 0 1 1 1 1 0 1 1 1 1 0 1 0 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 0 0 1 1 0 1 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 58.43 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 38:
  Aggregated Key Accuracy: 81.25%
  Aggregated Key: [1 0 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0]...
  Ground Truth:   [1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 55.75 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 39:
  Aggregated Key Accuracy: 80.47%
  Aggregated Key: [1 0 0 1 0 1 0 0 1 0 1 1 0 0 0 1 0 1 0 0 0 1 1 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 75.56 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 40:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0 0 1 0]...
  Ground Truth:   [1 0 0 0 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0 0 1 0]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 32.72 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 41:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 1]...
  Ground Truth:   [1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 21.08 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 42:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 1 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 0 0 1]...
  Ground Truth:   [0 1 1 1 1 0 0 0 0 1 0 0 1 0 0 0 1 0 1 1 0 0 0 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 15.01 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 

Person 43:
  Aggregated Key Accuracy: 87.50%
  Aggregated Key: [1 1 0 0 1 1 1 0 1 0 0 0 0 1 1 1 1 0 1 1 1 0 1 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
  Intra-person average Hamming distance: 43.34 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 44:
  Aggregated Key Accuracy: 89.45%
  Aggregated Key: [0 1 0 0 0 0 0 1 1 1 1 0 1 1 1 1 0 1 0 0 1 1 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 1 1 1 0 1 1 1 0 0 1 0 0 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 56.92 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 45:
  Aggregated Key Accuracy: 79.69%
  Aggregated Key: [0 1 0 0 0 1 0 1 1 1 1 1 0 1 0 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 1 1 0 1 0 1 0 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 75.97 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 46:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 1 0 1 1]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 8.74 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 47:
  Aggregated Key Accuracy: 91.02%
  Aggregated Key: [0 0 1 0 1 1 0 0 1 0 0 1 0 0 1 1 0 0 0 0 0 1 1 0]...
  Ground Truth:   [0 0 1 0 1 1 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 59.01 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 48:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
  Ground Truth:   [1 0 1 0 1 0 1 0 1 0 1 1 0 0 1 1 0 0 1 0 0 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 8.04 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step

Person 49:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
  Ground Truth:   [0 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0 0 1 1 1 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 16.55 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 50:
  Aggregated Key Accuracy: 96.48%
  Aggregated Key: [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
  Ground Truth:   [1 0 0 1 1 1 0 1 0 0 1 0 0 0 1 1 0 0 0 1 0 0 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 28.81 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 51:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1]...
  Ground Truth:   [0 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 0 0 1 0 1 0 1 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 9.03 bits
10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step

Person 52:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0]...
  Ground Truth:   [0 1 0 1 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 1 1 0 0 0]...
10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
  Intra-person average Hamming distance: 0.05 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 53:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
  Ground Truth:   [0 0 1 1 1 0 0 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 0.69 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 54:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [0 0 0 0 1 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 0 1]...
  Ground Truth:   [0 0 0 0 1 0 0 1 1 0 0 1 1 0 1 1 1 0 0 1 1 0 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 30.15 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 55:
  Aggregated Key Accuracy: 83.98%
  Aggregated Key: [0 0 1 1 0 0 0 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 0 0]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 1 1 1 1 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 57.57 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 56:
  Aggregated Key Accuracy: 90.62%
  Aggregated Key: [0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 1 0 0 0 0 1 1 0]...
  Ground Truth:   [0 0 0 0 0 1 0 0 1 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 48.26 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step

Person 57:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
  Ground Truth:   [1 0 1 1 0 1 0 1 1 0 1 1 0 0 1 0 1 0 0 1 0 1 0 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 7.60 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 

Person 58:
  Aggregated Key Accuracy: 95.70%
  Aggregated Key: [0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 0]...
  Ground Truth:   [0 0 1 1 0 1 0 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
  Intra-person average Hamming distance: 18.38 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 59:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 0 0 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 0 1 1 0 0 1 1 0 1 1 1 0 0 0 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
  Intra-person average Hamming distance: 5.92 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 60:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [0 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 1 1 0 1 0 1 1 0]...
  Ground Truth:   [0 0 0 1 1 0 0 1 0 0 0 1 0 0 1 0 1 1 0 1 0 1 1 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 34.61 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 61:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 1 1]...
  Ground Truth:   [0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 0 0 0 0 0 0 1 1 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 14.25 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 62:
  Aggregated Key Accuracy: 97.27%
  Aggregated Key: [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 0]...
  Ground Truth:   [0 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 0 0 0 1 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 21.02 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 63:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
  Ground Truth:   [1 1 0 0 0 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 8.64 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 64:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
  Ground Truth:   [0 1 0 1 1 0 1 1 0 1 1 0 0 1 1 1 0 0 0 0 1 1 1 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 6.82 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step

Person 65:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
  Ground Truth:   [1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
  Intra-person average Hamming distance: 8.75 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step

Person 66:
  Aggregated Key Accuracy: 98.83%
  Aggregated Key: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 0 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 22.46 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 67:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [1 0 1 0 1 1 0 0 0 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0]...
  Ground Truth:   [0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
  Intra-person average Hamming distance: 19.71 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step

Person 68:
  Aggregated Key Accuracy: 92.97%
  Aggregated Key: [0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 1 1 1 0 0 1 1]...
  Ground Truth:   [0 1 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
  Intra-person average Hamming distance: 29.16 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 69:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 26.55 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 70:
  Aggregated Key Accuracy: 97.66%
  Aggregated Key: [0 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0]...
  Ground Truth:   [0 0 0 0 1 0 1 1 1 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 23.45 bits
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 71:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
  Ground Truth:   [1 1 0 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 1 1 1 1 1 0]...
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 3.70 bits
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step

Person 72:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
  Ground Truth:   [1 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 0 0 0 0]...
7/7 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step
  Intra-person average Hamming distance: 1.65 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 73:
  Aggregated Key Accuracy: 74.22%
  Aggregated Key: [1 0 1 1 0 1 0 0 0 1 1 0 1 1 0 1 1 1 1 1 0 0 0 0]...
  Ground Truth:   [1 0 1 1 0 1 1 1 0 1 1 0 1 1 0 1 0 1 0 0 0 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 77.88 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 74:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
  Ground Truth:   [0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 0 1 1 1 0 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 0.14 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 75:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 1 1 1 0 0 1 0]...
  Ground Truth:   [0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 1 1 1 1 1 0 0 1 0]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 19.28 bits
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 76:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
  Ground Truth:   [1 1 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 1 1]...
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 15.38 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 77:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
  Ground Truth:   [1 1 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 22.15 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 

Person 78:
  Aggregated Key Accuracy: 84.38%
  Aggregated Key: [1 1 1 1 1 0 1 1 1 1 1 0 0 1 0 1 1 1 1 0 1 0 1 0]...
  Ground Truth:   [1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 0 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 52.69 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step

Person 79:
  Aggregated Key Accuracy: 99.61%
  Aggregated Key: [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0]...
  Ground Truth:   [0 1 1 1 1 0 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 21.41 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 80:
  Aggregated Key Accuracy: 90.23%
  Aggregated Key: [1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 0 1 1 0 0 1 0 0]...
  Ground Truth:   [1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 1 0 0 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 46.42 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 

Person 81:
  Aggregated Key Accuracy: 93.75%
  Aggregated Key: [1 0 1 1 1 1 1 1 1 0 1 0 1 1 0 0 1 1 1 1 0 0 0 0]...
  Ground Truth:   [1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
  Intra-person average Hamming distance: 37.03 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step

Person 82:
  Aggregated Key Accuracy: 98.44%
  Aggregated Key: [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
  Ground Truth:   [1 0 1 0 0 0 1 1 1 1 1 1 1 0 1 0 1 0 0 0 1 0 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 26.01 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 

Person 83:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
  Ground Truth:   [1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
  Intra-person average Hamming distance: 5.50 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 84:
  Aggregated Key Accuracy: 96.88%
  Aggregated Key: [1 0 0 0 1 1 0 0 0 0 0 1 1 1 0 1 1 0 0 0 0 1 1 1]...
  Ground Truth:   [1 0 0 0 1 1 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 1 1 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step 
  Intra-person average Hamming distance: 54.76 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step

Person 85:
  Aggregated Key Accuracy: 99.22%
  Aggregated Key: [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
  Ground Truth:   [0 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 0 0 1 1 1 1 0]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step
  Intra-person average Hamming distance: 34.46 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step

Person 86:
  Aggregated Key Accuracy: 92.19%
  Aggregated Key: [0 0 0 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1]...
  Ground Truth:   [0 0 0 1 0 1 1 1 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 66.07 bits
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 

Person 87:
  Aggregated Key Accuracy: 100.00%
  Aggregated Key: [0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1]...
  Ground Truth:   [0 1 0 1 0 0 0 1 1 1 0 0 1 0 1 0 0 1 1 0 0 0 1 1]...
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step 
  Intra-person average Hamming distance: 15.65 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 

Person 88:
  Aggregated Key Accuracy: 95.70%
  Aggregated Key: [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 0]...
  Ground Truth:   [0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
  Intra-person average Hamming distance: 34.66 bits
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 

Person 89:
  Aggregated Key Accuracy: 98.05%
  Aggregated Key: [1 0 1 1 0 0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 0 0 1]...
  Ground Truth:   [1 0 1 1 0 0 0 1 0 1 1 0 0 0 1 0 0 1 0 0 1 0 0 1]...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
  Intra-person average Hamming distance: 41.92 bits

Inter-person Hamming distances (aggregated keys):
  Distance between Person 1 and Person 2: 123 bits
  Distance between Person 1 and Person 3: 125 bits
  Distance between Person 1 and Person 4: 117 bits
  Distance between Person 1 and Person 5: 128 bits
  Distance between Person 1 and Person 6: 139 bits
  Distance between Person 1 and Person 7: 132 bits
  Distance between Person 1 and Person 8: 132 bits
  Distance between Person 1 and Person 9: 130 bits
  Distance between Person 1 and Person 10: 126 bits
  Distance between Person 1 and Person 11: 129 bits
  Distance between Person 1 and Person 12: 131 bits
  Distance between Person 1 and Person 13: 137 bits
  Distance between Person 1 and Person 14: 112 bits
  Distance between Person 1 and Person 15: 140 bits
  Distance between Person 1 and Person 16: 121 bits
  Distance between Person 1 and Person 17: 116 bits
  Distance between Person 1 and Person 18: 144 bits
  Distance between Person 1 and Person 19: 119 bits
  Distance between Person 1 and Person 20: 123 bits
  Distance between Person 1 and Person 21: 117 bits
  Distance between Person 1 and Person 22: 140 bits
  Distance between Person 1 and Person 23: 147 bits
  Distance between Person 1 and Person 24: 137 bits
  Distance between Person 1 and Person 25: 122 bits
  Distance between Person 1 and Person 26: 122 bits
  Distance between Person 1 and Person 27: 124 bits
  Distance between Person 1 and Person 28: 131 bits
  Distance between Person 1 and Person 29: 144 bits
  Distance between Person 1 and Person 30: 130 bits
  Distance between Person 1 and Person 31: 124 bits
  Distance between Person 1 and Person 32: 138 bits
  Distance between Person 1 and Person 33: 108 bits
  Distance between Person 1 and Person 34: 131 bits
  Distance between Person 1 and Person 35: 141 bits
  Distance between Person 1 and Person 36: 128 bits
  Distance between Person 1 and Person 37: 133 bits
  Distance between Person 1 and Person 38: 122 bits
  Distance between Person 1 and Person 39: 114 bits
  Distance between Person 1 and Person 40: 121 bits
  Distance between Person 1 and Person 41: 121 bits
  Distance between Person 1 and Person 42: 145 bits
  Distance between Person 1 and Person 43: 134 bits
  Distance between Person 1 and Person 44: 111 bits
  Distance between Person 1 and Person 45: 130 bits
  Distance between Person 1 and Person 46: 118 bits
  Distance between Person 1 and Person 47: 120 bits
  Distance between Person 1 and Person 48: 133 bits
  Distance between Person 1 and Person 49: 129 bits
  Distance between Person 1 and Person 50: 140 bits
  Distance between Person 1 and Person 51: 126 bits
  Distance between Person 1 and Person 52: 136 bits
  Distance between Person 1 and Person 53: 134 bits
  Distance between Person 1 and Person 54: 136 bits
  Distance between Person 1 and Person 55: 123 bits
  Distance between Person 1 and Person 56: 149 bits
  Distance between Person 1 and Person 57: 143 bits
  Distance between Person 1 and Person 58: 116 bits
  Distance between Person 1 and Person 59: 115 bits
  Distance between Person 1 and Person 60: 136 bits
  Distance between Person 1 and Person 61: 126 bits
  Distance between Person 1 and Person 62: 121 bits
  Distance between Person 1 and Person 63: 131 bits
  Distance between Person 1 and Person 64: 128 bits
  Distance between Person 1 and Person 65: 112 bits
  Distance between Person 1 and Person 66: 109 bits
  Distance between Person 1 and Person 67: 120 bits
  Distance between Person 1 and Person 68: 134 bits
  Distance between Person 1 and Person 69: 114 bits
  Distance between Person 1 and Person 70: 124 bits
  Distance between Person 1 and Person 71: 136 bits
  Distance between Person 1 and Person 72: 118 bits
  Distance between Person 1 and Person 73: 122 bits
  Distance between Person 1 and Person 74: 131 bits
  Distance between Person 1 and Person 75: 126 bits
  Distance between Person 1 and Person 76: 127 bits
  Distance between Person 1 and Person 77: 124 bits
  Distance between Person 1 and Person 78: 121 bits
  Distance between Person 1 and Person 79: 124 bits
  Distance between Person 1 and Person 80: 108 bits
  Distance between Person 1 and Person 81: 123 bits
  Distance between Person 1 and Person 82: 135 bits
  Distance between Person 1 and Person 83: 132 bits
  Distance between Person 1 and Person 84: 138 bits
  Distance between Person 1 and Person 85: 118 bits
  Distance between Person 1 and Person 86: 120 bits
  Distance between Person 1 and Person 87: 120 bits
  Distance between Person 1 and Person 88: 127 bits
  Distance between Person 1 and Person 89: 136 bits
  Distance between Person 2 and Person 3: 120 bits
  Distance between Person 2 and Person 4: 130 bits
  Distance between Person 2 and Person 5: 127 bits
  Distance between Person 2 and Person 6: 132 bits
  Distance between Person 2 and Person 7: 139 bits
  Distance between Person 2 and Person 8: 117 bits
  Distance between Person 2 and Person 9: 135 bits
  Distance between Person 2 and Person 10: 109 bits
  Distance between Person 2 and Person 11: 136 bits
  Distance between Person 2 and Person 12: 116 bits
  Distance between Person 2 and Person 13: 114 bits
  Distance between Person 2 and Person 14: 117 bits
  Distance between Person 2 and Person 15: 137 bits
  Distance between Person 2 and Person 16: 130 bits
  Distance between Person 2 and Person 17: 141 bits
  Distance between Person 2 and Person 18: 131 bits
  Distance between Person 2 and Person 19: 110 bits
  Distance between Person 2 and Person 20: 120 bits
  Distance between Person 2 and Person 21: 114 bits
  Distance between Person 2 and Person 22: 119 bits
  Distance between Person 2 and Person 23: 128 bits
  Distance between Person 2 and Person 24: 122 bits
  Distance between Person 2 and Person 25: 111 bits
  Distance between Person 2 and Person 26: 125 bits
  Distance between Person 2 and Person 27: 115 bits
  Distance between Person 2 and Person 28: 130 bits
  Distance between Person 2 and Person 29: 123 bits
  Distance between Person 2 and Person 30: 129 bits
  Distance between Person 2 and Person 31: 139 bits
  Distance between Person 2 and Person 32: 135 bits
  Distance between Person 2 and Person 33: 125 bits
  Distance between Person 2 and Person 34: 118 bits
  Distance between Person 2 and Person 35: 118 bits
  Distance between Person 2 and Person 36: 131 bits
  Distance between Person 2 and Person 37: 118 bits
  Distance between Person 2 and Person 38: 135 bits
  Distance between Person 2 and Person 39: 113 bits
  Distance between Person 2 and Person 40: 132 bits
  Distance between Person 2 and Person 41: 142 bits
  Distance between Person 2 and Person 42: 142 bits
  Distance between Person 2 and Person 43: 123 bits
  Distance between Person 2 and Person 44: 114 bits
  Distance between Person 2 and Person 45: 131 bits
  Distance between Person 2 and Person 46: 123 bits
  Distance between Person 2 and Person 47: 123 bits
  Distance between Person 2 and Person 48: 120 bits
  Distance between Person 2 and Person 49: 128 bits
  Distance between Person 2 and Person 50: 111 bits
  Distance between Person 2 and Person 51: 121 bits
  Distance between Person 2 and Person 52: 125 bits
  Distance between Person 2 and Person 53: 137 bits
  Distance between Person 2 and Person 54: 123 bits
  Distance between Person 2 and Person 55: 130 bits
  Distance between Person 2 and Person 56: 108 bits
  Distance between Person 2 and Person 57: 134 bits
  Distance between Person 2 and Person 58: 137 bits
  Distance between Person 2 and Person 59: 122 bits
  Distance between Person 2 and Person 60: 133 bits
  Distance between Person 2 and Person 61: 125 bits
  Distance between Person 2 and Person 62: 120 bits
  Distance between Person 2 and Person 63: 120 bits
  Distance between Person 2 and Person 64: 123 bits
  Distance between Person 2 and Person 65: 129 bits
  Distance between Person 2 and Person 66: 130 bits
  Distance between Person 2 and Person 67: 129 bits
  Distance between Person 2 and Person 68: 129 bits
  Distance between Person 2 and Person 69: 113 bits
  Distance between Person 2 and Person 70: 133 bits
  Distance between Person 2 and Person 71: 131 bits
  Distance between Person 2 and Person 72: 137 bits
  Distance between Person 2 and Person 73: 143 bits
  Distance between Person 2 and Person 74: 128 bits
  Distance between Person 2 and Person 75: 129 bits
  Distance between Person 2 and Person 76: 116 bits
  Distance between Person 2 and Person 77: 113 bits
  Distance between Person 2 and Person 78: 126 bits
  Distance between Person 2 and Person 79: 131 bits
  Distance between Person 2 and Person 80: 123 bits
  Distance between Person 2 and Person 81: 132 bits
  Distance between Person 2 and Person 82: 120 bits
  Distance between Person 2 and Person 83: 123 bits
  Distance between Person 2 and Person 84: 119 bits
  Distance between Person 2 and Person 85: 125 bits
  Distance between Person 2 and Person 86: 129 bits
  Distance between Person 2 and Person 87: 125 bits
  Distance between Person 2 and Person 88: 116 bits
  Distance between Person 2 and Person 89: 127 bits
  Distance between Person 3 and Person 4: 126 bits
  Distance between Person 3 and Person 5: 133 bits
  Distance between Person 3 and Person 6: 126 bits
  Distance between Person 3 and Person 7: 137 bits
  Distance between Person 3 and Person 8: 123 bits
  Distance between Person 3 and Person 9: 127 bits
  Distance between Person 3 and Person 10: 121 bits
  Distance between Person 3 and Person 11: 124 bits
  Distance between Person 3 and Person 12: 152 bits
  Distance between Person 3 and Person 13: 132 bits
  Distance between Person 3 and Person 14: 131 bits
  Distance between Person 3 and Person 15: 147 bits
  Distance between Person 3 and Person 16: 126 bits
  Distance between Person 3 and Person 17: 133 bits
  Distance between Person 3 and Person 18: 133 bits
  Distance between Person 3 and Person 19: 130 bits
  Distance between Person 3 and Person 20: 146 bits
  Distance between Person 3 and Person 21: 118 bits
  Distance between Person 3 and Person 22: 123 bits
  Distance between Person 3 and Person 23: 120 bits
  Distance between Person 3 and Person 24: 134 bits
  Distance between Person 3 and Person 25: 137 bits
  Distance between Person 3 and Person 26: 131 bits
  Distance between Person 3 and Person 27: 117 bits
  Distance between Person 3 and Person 28: 132 bits
  Distance between Person 3 and Person 29: 111 bits
  Distance between Person 3 and Person 30: 113 bits
  Distance between Person 3 and Person 31: 135 bits
  Distance between Person 3 and Person 32: 127 bits
  Distance between Person 3 and Person 33: 115 bits
  Distance between Person 3 and Person 34: 118 bits
  Distance between Person 3 and Person 35: 116 bits
  Distance between Person 3 and Person 36: 121 bits
  Distance between Person 3 and Person 37: 124 bits
  Distance between Person 3 and Person 38: 133 bits
  Distance between Person 3 and Person 39: 91 bits
  Distance between Person 3 and Person 40: 122 bits
  Distance between Person 3 and Person 41: 124 bits
  Distance between Person 3 and Person 42: 130 bits
  Distance between Person 3 and Person 43: 133 bits
  Distance between Person 3 and Person 44: 142 bits
  Distance between Person 3 and Person 45: 131 bits
  Distance between Person 3 and Person 46: 123 bits
  Distance between Person 3 and Person 47: 115 bits
  Distance between Person 3 and Person 48: 118 bits
  Distance between Person 3 and Person 49: 124 bits
  Distance between Person 3 and Person 50: 121 bits
  Distance between Person 3 and Person 51: 133 bits
  Distance between Person 3 and Person 52: 125 bits
  Distance between Person 3 and Person 53: 133 bits
  Distance between Person 3 and Person 54: 139 bits
  Distance between Person 3 and Person 55: 140 bits
  Distance between Person 3 and Person 56: 134 bits
  Distance between Person 3 and Person 57: 130 bits
  Distance between Person 3 and Person 58: 135 bits
  Distance between Person 3 and Person 59: 126 bits
  Distance between Person 3 and Person 60: 129 bits
  Distance between Person 3 and Person 61: 133 bits
  Distance between Person 3 and Person 62: 128 bits
  Distance between Person 3 and Person 63: 140 bits
  Distance between Person 3 and Person 64: 125 bits
  Distance between Person 3 and Person 65: 129 bits
  Distance between Person 3 and Person 66: 116 bits
  Distance between Person 3 and Person 67: 139 bits
  Distance between Person 3 and Person 68: 119 bits
  Distance between Person 3 and Person 69: 129 bits
  Distance between Person 3 and Person 70: 139 bits
  Distance between Person 3 and Person 71: 133 bits
  Distance between Person 3 and Person 72: 137 bits
  Distance between Person 3 and Person 73: 141 bits
  Distance between Person 3 and Person 74: 128 bits
  Distance between Person 3 and Person 75: 133 bits
  Distance between Person 3 and Person 76: 128 bits
  Distance between Person 3 and Person 77: 127 bits
  Distance between Person 3 and Person 78: 140 bits
  Distance between Person 3 and Person 79: 119 bits
  Distance between Person 3 and Person 80: 129 bits
  Distance between Person 3 and Person 81: 118 bits
  Distance between Person 3 and Person 82: 122 bits
  Distance between Person 3 and Person 83: 117 bits
  Distance between Person 3 and Person 84: 121 bits
  Distance between Person 3 and Person 85: 131 bits
  Distance between Person 3 and Person 86: 131 bits
  Distance between Person 3 and Person 87: 135 bits
  Distance between Person 3 and Person 88: 146 bits
  Distance between Person 3 and Person 89: 129 bits
  Distance between Person 4 and Person 5: 129 bits
  Distance between Person 4 and Person 6: 108 bits
  Distance between Person 4 and Person 7: 129 bits
  Distance between Person 4 and Person 8: 117 bits
  Distance between Person 4 and Person 9: 131 bits
  Distance between Person 4 and Person 10: 119 bits
  Distance between Person 4 and Person 11: 126 bits
  Distance between Person 4 and Person 12: 126 bits
  Distance between Person 4 and Person 13: 136 bits
  Distance between Person 4 and Person 14: 111 bits
  Distance between Person 4 and Person 15: 97 bits
  Distance between Person 4 and Person 16: 116 bits
  Distance between Person 4 and Person 17: 133 bits
  Distance between Person 4 and Person 18: 125 bits
  Distance between Person 4 and Person 19: 120 bits
  Distance between Person 4 and Person 20: 152 bits
  Distance between Person 4 and Person 21: 140 bits
  Distance between Person 4 and Person 22: 131 bits
  Distance between Person 4 and Person 23: 152 bits
  Distance between Person 4 and Person 24: 128 bits
  Distance between Person 4 and Person 25: 135 bits
  Distance between Person 4 and Person 26: 127 bits
  Distance between Person 4 and Person 27: 127 bits
  Distance between Person 4 and Person 28: 122 bits
  Distance between Person 4 and Person 29: 125 bits
  Distance between Person 4 and Person 30: 129 bits
  Distance between Person 4 and Person 31: 127 bits
  Distance between Person 4 and Person 32: 137 bits
  Distance between Person 4 and Person 33: 129 bits
  Distance between Person 4 and Person 34: 120 bits
  Distance between Person 4 and Person 35: 128 bits
  Distance between Person 4 and Person 36: 123 bits
  Distance between Person 4 and Person 37: 118 bits
  Distance between Person 4 and Person 38: 129 bits
  Distance between Person 4 and Person 39: 145 bits
  Distance between Person 4 and Person 40: 118 bits
  Distance between Person 4 and Person 41: 126 bits
  Distance between Person 4 and Person 42: 116 bits
  Distance between Person 4 and Person 43: 127 bits
  Distance between Person 4 and Person 44: 130 bits
  Distance between Person 4 and Person 45: 133 bits
  Distance between Person 4 and Person 46: 127 bits
  Distance between Person 4 and Person 47: 129 bits
  Distance between Person 4 and Person 48: 124 bits
  Distance between Person 4 and Person 49: 144 bits
  Distance between Person 4 and Person 50: 133 bits
  Distance between Person 4 and Person 51: 127 bits
  Distance between Person 4 and Person 52: 111 bits
  Distance between Person 4 and Person 53: 127 bits
  Distance between Person 4 and Person 54: 129 bits
  Distance between Person 4 and Person 55: 100 bits
  Distance between Person 4 and Person 56: 122 bits
  Distance between Person 4 and Person 57: 134 bits
  Distance between Person 4 and Person 58: 119 bits
  Distance between Person 4 and Person 59: 118 bits
  Distance between Person 4 and Person 60: 129 bits
  Distance between Person 4 and Person 61: 129 bits
  Distance between Person 4 and Person 62: 130 bits
  Distance between Person 4 and Person 63: 136 bits
  Distance between Person 4 and Person 64: 111 bits
  Distance between Person 4 and Person 65: 119 bits
  Distance between Person 4 and Person 66: 122 bits
  Distance between Person 4 and Person 67: 129 bits
  Distance between Person 4 and Person 68: 119 bits
  Distance between Person 4 and Person 69: 125 bits
  Distance between Person 4 and Person 70: 111 bits
  Distance between Person 4 and Person 71: 135 bits
  Distance between Person 4 and Person 72: 137 bits
  Distance between Person 4 and Person 73: 119 bits
  Distance between Person 4 and Person 74: 138 bits
  Distance between Person 4 and Person 75: 133 bits
  Distance between Person 4 and Person 76: 136 bits
  Distance between Person 4 and Person 77: 125 bits
  Distance between Person 4 and Person 78: 134 bits
  Distance between Person 4 and Person 79: 109 bits
  Distance between Person 4 and Person 80: 123 bits
  Distance between Person 4 and Person 81: 120 bits
  Distance between Person 4 and Person 82: 132 bits
  Distance between Person 4 and Person 83: 119 bits
  Distance between Person 4 and Person 84: 115 bits
  Distance between Person 4 and Person 85: 113 bits
  Distance between Person 4 and Person 86: 133 bits
  Distance between Person 4 and Person 87: 135 bits
  Distance between Person 4 and Person 88: 134 bits
  Distance between Person 4 and Person 89: 127 bits
  Distance between Person 5 and Person 6: 125 bits
  Distance between Person 5 and Person 7: 124 bits
  Distance between Person 5 and Person 8: 128 bits
  Distance between Person 5 and Person 9: 120 bits
  Distance between Person 5 and Person 10: 120 bits
  Distance between Person 5 and Person 11: 117 bits
  Distance between Person 5 and Person 12: 141 bits
  Distance between Person 5 and Person 13: 117 bits
  Distance between Person 5 and Person 14: 138 bits
  Distance between Person 5 and Person 15: 122 bits
  Distance between Person 5 and Person 16: 129 bits
  Distance between Person 5 and Person 17: 132 bits
  Distance between Person 5 and Person 18: 106 bits
  Distance between Person 5 and Person 19: 133 bits
  Distance between Person 5 and Person 20: 147 bits
  Distance between Person 5 and Person 21: 145 bits
  Distance between Person 5 and Person 22: 120 bits
  Distance between Person 5 and Person 23: 123 bits
  Distance between Person 5 and Person 24: 123 bits
  Distance between Person 5 and Person 25: 132 bits
  Distance between Person 5 and Person 26: 136 bits
  Distance between Person 5 and Person 27: 122 bits
  Distance between Person 5 and Person 28: 113 bits
  Distance between Person 5 and Person 29: 144 bits
  Distance between Person 5 and Person 30: 120 bits
  Distance between Person 5 and Person 31: 128 bits
  Distance between Person 5 and Person 32: 150 bits
  Distance between Person 5 and Person 33: 140 bits
  Distance between Person 5 and Person 34: 125 bits
  Distance between Person 5 and Person 35: 135 bits
  Distance between Person 5 and Person 36: 124 bits
  Distance between Person 5 and Person 37: 129 bits
  Distance between Person 5 and Person 38: 128 bits
  Distance between Person 5 and Person 39: 128 bits
  Distance between Person 5 and Person 40: 133 bits
  Distance between Person 5 and Person 41: 121 bits
  Distance between Person 5 and Person 42: 139 bits
  Distance between Person 5 and Person 43: 144 bits
  Distance between Person 5 and Person 44: 131 bits
  Distance between Person 5 and Person 45: 146 bits
  Distance between Person 5 and Person 46: 124 bits
  Distance between Person 5 and Person 47: 132 bits
  Distance between Person 5 and Person 48: 111 bits
  Distance between Person 5 and Person 49: 113 bits
  Distance between Person 5 and Person 50: 134 bits
  Distance between Person 5 and Person 51: 138 bits
  Distance between Person 5 and Person 52: 120 bits
  Distance between Person 5 and Person 53: 140 bits
  Distance between Person 5 and Person 54: 136 bits
  Distance between Person 5 and Person 55: 111 bits
  Distance between Person 5 and Person 56: 131 bits
  Distance between Person 5 and Person 57: 131 bits
  Distance between Person 5 and Person 58: 122 bits
  Distance between Person 5 and Person 59: 131 bits
  Distance between Person 5 and Person 60: 126 bits
  Distance between Person 5 and Person 61: 132 bits
  Distance between Person 5 and Person 62: 121 bits
  Distance between Person 5 and Person 63: 127 bits
  Distance between Person 5 and Person 64: 124 bits
  Distance between Person 5 and Person 65: 140 bits
  Distance between Person 5 and Person 66: 133 bits
  Distance between Person 5 and Person 67: 120 bits
  Distance between Person 5 and Person 68: 112 bits
  Distance between Person 5 and Person 69: 144 bits
  Distance between Person 5 and Person 70: 126 bits
  Distance between Person 5 and Person 71: 128 bits
  Distance between Person 5 and Person 72: 124 bits
  Distance between Person 5 and Person 73: 114 bits
  Distance between Person 5 and Person 74: 133 bits
  Distance between Person 5 and Person 75: 148 bits
  Distance between Person 5 and Person 76: 139 bits
  Distance between Person 5 and Person 77: 118 bits
  Distance between Person 5 and Person 78: 117 bits
  Distance between Person 5 and Person 79: 124 bits
  Distance between Person 5 and Person 80: 122 bits
  Distance between Person 5 and Person 81: 109 bits
  Distance between Person 5 and Person 82: 153 bits
  Distance between Person 5 and Person 83: 126 bits
  Distance between Person 5 and Person 84: 118 bits
  Distance between Person 5 and Person 85: 134 bits
  Distance between Person 5 and Person 86: 120 bits
  Distance between Person 5 and Person 87: 122 bits
  Distance between Person 5 and Person 88: 141 bits
  Distance between Person 5 and Person 89: 130 bits
  Distance between Person 6 and Person 7: 123 bits
  Distance between Person 6 and Person 8: 121 bits
  Distance between Person 6 and Person 9: 139 bits
  Distance between Person 6 and Person 10: 109 bits
  Distance between Person 6 and Person 11: 134 bits
  Distance between Person 6 and Person 12: 122 bits
  Distance between Person 6 and Person 13: 126 bits
  Distance between Person 6 and Person 14: 123 bits
  Distance between Person 6 and Person 15: 109 bits
  Distance between Person 6 and Person 16: 128 bits
  Distance between Person 6 and Person 17: 129 bits
  Distance between Person 6 and Person 18: 95 bits
  Distance between Person 6 and Person 19: 118 bits
  Distance between Person 6 and Person 20: 132 bits
  Distance between Person 6 and Person 21: 92 bits
  Distance between Person 6 and Person 22: 143 bits
  Distance between Person 6 and Person 23: 122 bits
  Distance between Person 6 and Person 24: 130 bits
  Distance between Person 6 and Person 25: 135 bits
  Distance between Person 6 and Person 26: 149 bits
  Distance between Person 6 and Person 27: 145 bits
  Distance between Person 6 and Person 28: 150 bits
  Distance between Person 6 and Person 29: 111 bits
  Distance between Person 6 and Person 30: 117 bits
  Distance between Person 6 and Person 31: 143 bits
  Distance between Person 6 and Person 32: 149 bits
  Distance between Person 6 and Person 33: 127 bits
  Distance between Person 6 and Person 34: 130 bits
  Distance between Person 6 and Person 35: 128 bits
  Distance between Person 6 and Person 36: 137 bits
  Distance between Person 6 and Person 37: 120 bits
  Distance between Person 6 and Person 38: 103 bits
  Distance between Person 6 and Person 39: 125 bits
  Distance between Person 6 and Person 40: 114 bits
  Distance between Person 6 and Person 41: 122 bits
  Distance between Person 6 and Person 42: 126 bits
  Distance between Person 6 and Person 43: 115 bits
  Distance between Person 6 and Person 44: 120 bits
  Distance between Person 6 and Person 45: 109 bits
  Distance between Person 6 and Person 46: 125 bits
  Distance between Person 6 and Person 47: 113 bits
  Distance between Person 6 and Person 48: 122 bits
  Distance between Person 6 and Person 49: 130 bits
  Distance between Person 6 and Person 50: 123 bits
  Distance between Person 6 and Person 51: 135 bits
  Distance between Person 6 and Person 52: 125 bits
  Distance between Person 6 and Person 53: 135 bits
  Distance between Person 6 and Person 54: 129 bits
  Distance between Person 6 and Person 55: 114 bits
  Distance between Person 6 and Person 56: 106 bits
  Distance between Person 6 and Person 57: 138 bits
  Distance between Person 6 and Person 58: 123 bits
  Distance between Person 6 and Person 59: 122 bits
  Distance between Person 6 and Person 60: 127 bits
  Distance between Person 6 and Person 61: 133 bits
  Distance between Person 6 and Person 62: 134 bits
  Distance between Person 6 and Person 63: 118 bits
  Distance between Person 6 and Person 64: 127 bits
  Distance between Person 6 and Person 65: 129 bits
  Distance between Person 6 and Person 66: 118 bits
  Distance between Person 6 and Person 67: 127 bits
  Distance between Person 6 and Person 68: 127 bits
  Distance between Person 6 and Person 69: 135 bits
  Distance between Person 6 and Person 70: 129 bits
  Distance between Person 6 and Person 71: 111 bits
  Distance between Person 6 and Person 72: 141 bits
  Distance between Person 6 and Person 73: 115 bits
  Distance between Person 6 and Person 74: 136 bits
  Distance between Person 6 and Person 75: 117 bits
  Distance between Person 6 and Person 76: 130 bits
  Distance between Person 6 and Person 77: 123 bits
  Distance between Person 6 and Person 78: 130 bits
  Distance between Person 6 and Person 79: 125 bits
  Distance between Person 6 and Person 80: 105 bits
  Distance between Person 6 and Person 81: 118 bits
  Distance between Person 6 and Person 82: 124 bits
  Distance between Person 6 and Person 83: 129 bits
  Distance between Person 6 and Person 84: 141 bits
  Distance between Person 6 and Person 85: 143 bits
  Distance between Person 6 and Person 86: 117 bits
  Distance between Person 6 and Person 87: 141 bits
  Distance between Person 6 and Person 88: 128 bits
  Distance between Person 6 and Person 89: 127 bits
  Distance between Person 7 and Person 8: 162 bits
  Distance between Person 7 and Person 9: 124 bits
  Distance between Person 7 and Person 10: 138 bits
  Distance between Person 7 and Person 11: 131 bits
  Distance between Person 7 and Person 12: 143 bits
  Distance between Person 7 and Person 13: 123 bits
  Distance between Person 7 and Person 14: 128 bits
  Distance between Person 7 and Person 15: 132 bits
  Distance between Person 7 and Person 16: 139 bits
  Distance between Person 7 and Person 17: 130 bits
  Distance between Person 7 and Person 18: 126 bits
  Distance between Person 7 and Person 19: 135 bits
  Distance between Person 7 and Person 20: 141 bits
  Distance between Person 7 and Person 21: 121 bits
  Distance between Person 7 and Person 22: 132 bits
  Distance between Person 7 and Person 23: 119 bits
  Distance between Person 7 and Person 24: 121 bits
  Distance between Person 7 and Person 25: 128 bits
  Distance between Person 7 and Person 26: 120 bits
  Distance between Person 7 and Person 27: 122 bits
  Distance between Person 7 and Person 28: 123 bits
  Distance between Person 7 and Person 29: 126 bits
  Distance between Person 7 and Person 30: 128 bits
  Distance between Person 7 and Person 31: 118 bits
  Distance between Person 7 and Person 32: 136 bits
  Distance between Person 7 and Person 33: 120 bits
  Distance between Person 7 and Person 34: 139 bits
  Distance between Person 7 and Person 35: 139 bits
  Distance between Person 7 and Person 36: 126 bits
  Distance between Person 7 and Person 37: 127 bits
  Distance between Person 7 and Person 38: 116 bits
  Distance between Person 7 and Person 39: 130 bits
  Distance between Person 7 and Person 40: 127 bits
  Distance between Person 7 and Person 41: 111 bits
  Distance between Person 7 and Person 42: 129 bits
  Distance between Person 7 and Person 43: 130 bits
  Distance between Person 7 and Person 44: 125 bits
  Distance between Person 7 and Person 45: 116 bits
  Distance between Person 7 and Person 46: 126 bits
  Distance between Person 7 and Person 47: 134 bits
  Distance between Person 7 and Person 48: 135 bits
  Distance between Person 7 and Person 49: 115 bits
  Distance between Person 7 and Person 50: 136 bits
  Distance between Person 7 and Person 51: 118 bits
  Distance between Person 7 and Person 52: 124 bits
  Distance between Person 7 and Person 53: 126 bits
  Distance between Person 7 and Person 54: 128 bits
  Distance between Person 7 and Person 55: 119 bits
  Distance between Person 7 and Person 56: 107 bits
  Distance between Person 7 and Person 57: 113 bits
  Distance between Person 7 and Person 58: 136 bits
  Distance between Person 7 and Person 59: 147 bits
  Distance between Person 7 and Person 60: 106 bits
  Distance between Person 7 and Person 61: 124 bits
  Distance between Person 7 and Person 62: 135 bits
  Distance between Person 7 and Person 63: 115 bits
  Distance between Person 7 and Person 64: 144 bits
  Distance between Person 7 and Person 65: 114 bits
  Distance between Person 7 and Person 66: 137 bits
  Distance between Person 7 and Person 67: 134 bits
  Distance between Person 7 and Person 68: 138 bits
  Distance between Person 7 and Person 69: 124 bits
  Distance between Person 7 and Person 70: 128 bits
  Distance between Person 7 and Person 71: 140 bits
  Distance between Person 7 and Person 72: 116 bits
  Distance between Person 7 and Person 73: 128 bits
  Distance between Person 7 and Person 74: 133 bits
  Distance between Person 7 and Person 75: 110 bits
  Distance between Person 7 and Person 76: 141 bits
  Distance between Person 7 and Person 77: 114 bits
  Distance between Person 7 and Person 78: 117 bits
  Distance between Person 7 and Person 79: 132 bits
  Distance between Person 7 and Person 80: 118 bits
  Distance between Person 7 and Person 81: 145 bits
  Distance between Person 7 and Person 82: 113 bits
  Distance between Person 7 and Person 83: 120 bits
  Distance between Person 7 and Person 84: 128 bits
  Distance between Person 7 and Person 85: 114 bits
  Distance between Person 7 and Person 86: 122 bits
  Distance between Person 7 and Person 87: 126 bits
  Distance between Person 7 and Person 88: 131 bits
  Distance between Person 7 and Person 89: 126 bits
  Distance between Person 8 and Person 9: 132 bits
  Distance between Person 8 and Person 10: 122 bits
  Distance between Person 8 and Person 11: 125 bits
  Distance between Person 8 and Person 12: 111 bits
  Distance between Person 8 and Person 13: 123 bits
  Distance between Person 8 and Person 14: 126 bits
  Distance between Person 8 and Person 15: 126 bits
  Distance between Person 8 and Person 16: 131 bits
  Distance between Person 8 and Person 17: 126 bits
  Distance between Person 8 and Person 18: 120 bits
  Distance between Person 8 and Person 19: 143 bits
  Distance between Person 8 and Person 20: 129 bits
  Distance between Person 8 and Person 21: 139 bits
  Distance between Person 8 and Person 22: 120 bits
  Distance between Person 8 and Person 23: 133 bits
  Distance between Person 8 and Person 24: 127 bits
  Distance between Person 8 and Person 25: 140 bits
  Distance between Person 8 and Person 26: 130 bits
  Distance between Person 8 and Person 27: 130 bits
  Distance between Person 8 and Person 28: 125 bits
  Distance between Person 8 and Person 29: 116 bits
  Distance between Person 8 and Person 30: 136 bits
  Distance between Person 8 and Person 31: 136 bits
  Distance between Person 8 and Person 32: 128 bits
  Distance between Person 8 and Person 33: 126 bits
  Distance between Person 8 and Person 34: 115 bits
  Distance between Person 8 and Person 35: 123 bits
  Distance between Person 8 and Person 36: 130 bits
  Distance between Person 8 and Person 37: 97 bits
  Distance between Person 8 and Person 38: 130 bits
  Distance between Person 8 and Person 39: 120 bits
  Distance between Person 8 and Person 40: 131 bits
  Distance between Person 8 and Person 41: 143 bits
  Distance between Person 8 and Person 42: 127 bits
  Distance between Person 8 and Person 43: 122 bits
  Distance between Person 8 and Person 44: 125 bits
  Distance between Person 8 and Person 45: 136 bits
  Distance between Person 8 and Person 46: 126 bits
  Distance between Person 8 and Person 47: 126 bits
  Distance between Person 8 and Person 48: 127 bits
  Distance between Person 8 and Person 49: 135 bits
  Distance between Person 8 and Person 50: 138 bits
  Distance between Person 8 and Person 51: 140 bits
  Distance between Person 8 and Person 52: 124 bits
  Distance between Person 8 and Person 53: 126 bits
  Distance between Person 8 and Person 54: 128 bits
  Distance between Person 8 and Person 55: 121 bits
  Distance between Person 8 and Person 56: 119 bits
  Distance between Person 8 and Person 57: 127 bits
  Distance between Person 8 and Person 58: 124 bits
  Distance between Person 8 and Person 59: 131 bits
  Distance between Person 8 and Person 60: 120 bits
  Distance between Person 8 and Person 61: 146 bits
  Distance between Person 8 and Person 62: 143 bits
  Distance between Person 8 and Person 63: 125 bits
  Distance between Person 8 and Person 64: 138 bits
  Distance between Person 8 and Person 65: 128 bits
  Distance between Person 8 and Person 66: 133 bits
  Distance between Person 8 and Person 67: 108 bits
  Distance between Person 8 and Person 68: 118 bits
  Distance between Person 8 and Person 69: 128 bits
  Distance between Person 8 and Person 70: 120 bits
  Distance between Person 8 and Person 71: 130 bits
  Distance between Person 8 and Person 72: 138 bits
  Distance between Person 8 and Person 73: 112 bits
  Distance between Person 8 and Person 74: 123 bits
  Distance between Person 8 and Person 75: 128 bits
  Distance between Person 8 and Person 76: 127 bits
  Distance between Person 8 and Person 77: 122 bits
  Distance between Person 8 and Person 78: 131 bits
  Distance between Person 8 and Person 79: 124 bits
  Distance between Person 8 and Person 80: 128 bits
  Distance between Person 8 and Person 81: 129 bits
  Distance between Person 8 and Person 82: 131 bits
  Distance between Person 8 and Person 83: 136 bits
  Distance between Person 8 and Person 84: 120 bits
  Distance between Person 8 and Person 85: 130 bits
  Distance between Person 8 and Person 86: 142 bits
  Distance between Person 8 and Person 87: 130 bits
  Distance between Person 8 and Person 88: 115 bits
  Distance between Person 8 and Person 89: 136 bits
  Distance between Person 9 and Person 10: 132 bits
  Distance between Person 9 and Person 11: 119 bits
  Distance between Person 9 and Person 12: 135 bits
  Distance between Person 9 and Person 13: 133 bits
  Distance between Person 9 and Person 14: 144 bits
  Distance between Person 9 and Person 15: 110 bits
  Distance between Person 9 and Person 16: 127 bits
  Distance between Person 9 and Person 17: 132 bits
  Distance between Person 9 and Person 18: 136 bits
  Distance between Person 9 and Person 19: 135 bits
  Distance between Person 9 and Person 20: 125 bits
  Distance between Person 9 and Person 21: 129 bits
  Distance between Person 9 and Person 22: 130 bits
  Distance between Person 9 and Person 23: 111 bits
  Distance between Person 9 and Person 24: 137 bits
  Distance between Person 9 and Person 25: 110 bits
  Distance between Person 9 and Person 26: 134 bits
  Distance between Person 9 and Person 27: 134 bits
  Distance between Person 9 and Person 28: 111 bits
  Distance between Person 9 and Person 29: 118 bits
  Distance between Person 9 and Person 30: 128 bits
  Distance between Person 9 and Person 31: 118 bits
  Distance between Person 9 and Person 32: 120 bits
  Distance between Person 9 and Person 33: 124 bits
  Distance between Person 9 and Person 34: 145 bits
  Distance between Person 9 and Person 35: 131 bits
  Distance between Person 9 and Person 36: 136 bits
  Distance between Person 9 and Person 37: 129 bits
  Distance between Person 9 and Person 38: 122 bits
  Distance between Person 9 and Person 39: 126 bits
  Distance between Person 9 and Person 40: 129 bits
  Distance between Person 9 and Person 41: 119 bits
  Distance between Person 9 and Person 42: 115 bits
  Distance between Person 9 and Person 43: 124 bits
  Distance between Person 9 and Person 44: 137 bits
  Distance between Person 9 and Person 45: 142 bits
  Distance between Person 9 and Person 46: 136 bits
  Distance between Person 9 and Person 47: 136 bits
  Distance between Person 9 and Person 48: 145 bits
  Distance between Person 9 and Person 49: 145 bits
  Distance between Person 9 and Person 50: 142 bits
  Distance between Person 9 and Person 51: 122 bits
  Distance between Person 9 and Person 52: 132 bits
  Distance between Person 9 and Person 53: 130 bits
  Distance between Person 9 and Person 54: 120 bits
  Distance between Person 9 and Person 55: 131 bits
  Distance between Person 9 and Person 56: 147 bits
  Distance between Person 9 and Person 57: 107 bits
  Distance between Person 9 and Person 58: 128 bits
  Distance between Person 9 and Person 59: 135 bits
  Distance between Person 9 and Person 60: 126 bits
  Distance between Person 9 and Person 61: 134 bits
  Distance between Person 9 and Person 62: 135 bits
  Distance between Person 9 and Person 63: 115 bits
  Distance between Person 9 and Person 64: 140 bits
  Distance between Person 9 and Person 65: 142 bits
  Distance between Person 9 and Person 66: 133 bits
  Distance between Person 9 and Person 67: 134 bits
  Distance between Person 9 and Person 68: 128 bits
  Distance between Person 9 and Person 69: 134 bits
  Distance between Person 9 and Person 70: 118 bits
  Distance between Person 9 and Person 71: 140 bits
  Distance between Person 9 and Person 72: 128 bits
  Distance between Person 9 and Person 73: 118 bits
  Distance between Person 9 and Person 74: 135 bits
  Distance between Person 9 and Person 75: 136 bits
  Distance between Person 9 and Person 76: 135 bits
  Distance between Person 9 and Person 77: 124 bits
  Distance between Person 9 and Person 78: 105 bits
  Distance between Person 9 and Person 79: 136 bits
  Distance between Person 9 and Person 80: 130 bits
  Distance between Person 9 and Person 81: 103 bits
  Distance between Person 9 and Person 82: 127 bits
  Distance between Person 9 and Person 83: 118 bits
  Distance between Person 9 and Person 84: 132 bits
  Distance between Person 9 and Person 85: 128 bits
  Distance between Person 9 and Person 86: 134 bits
  Distance between Person 9 and Person 87: 126 bits
  Distance between Person 9 and Person 88: 135 bits
  Distance between Person 9 and Person 89: 130 bits
  Distance between Person 10 and Person 11: 131 bits
  Distance between Person 10 and Person 12: 141 bits
  Distance between Person 10 and Person 13: 147 bits
  Distance between Person 10 and Person 14: 134 bits
  Distance between Person 10 and Person 15: 116 bits
  Distance between Person 10 and Person 16: 119 bits
  Distance between Person 10 and Person 17: 118 bits
  Distance between Person 10 and Person 18: 120 bits
  Distance between Person 10 and Person 19: 117 bits
  Distance between Person 10 and Person 20: 129 bits
  Distance between Person 10 and Person 21: 127 bits
  Distance between Person 10 and Person 22: 128 bits
  Distance between Person 10 and Person 23: 127 bits
  Distance between Person 10 and Person 24: 129 bits
  Distance between Person 10 and Person 25: 124 bits
  Distance between Person 10 and Person 26: 120 bits
  Distance between Person 10 and Person 27: 132 bits
  Distance between Person 10 and Person 28: 143 bits
  Distance between Person 10 and Person 29: 124 bits
  Distance between Person 10 and Person 30: 120 bits
  Distance between Person 10 and Person 31: 130 bits
  Distance between Person 10 and Person 32: 144 bits
  Distance between Person 10 and Person 33: 130 bits
  Distance between Person 10 and Person 34: 135 bits
  Distance between Person 10 and Person 35: 123 bits
  Distance between Person 10 and Person 36: 128 bits
  Distance between Person 10 and Person 37: 123 bits
  Distance between Person 10 and Person 38: 126 bits
  Distance between Person 10 and Person 39: 122 bits
  Distance between Person 10 and Person 40: 133 bits
  Distance between Person 10 and Person 41: 119 bits
  Distance between Person 10 and Person 42: 135 bits
  Distance between Person 10 and Person 43: 116 bits
  Distance between Person 10 and Person 44: 125 bits
  Distance between Person 10 and Person 45: 120 bits
  Distance between Person 10 and Person 46: 130 bits
  Distance between Person 10 and Person 47: 120 bits
  Distance between Person 10 and Person 48: 123 bits
  Distance between Person 10 and Person 49: 123 bits
  Distance between Person 10 and Person 50: 124 bits
  Distance between Person 10 and Person 51: 116 bits
  Distance between Person 10 and Person 52: 138 bits
  Distance between Person 10 and Person 53: 140 bits
  Distance between Person 10 and Person 54: 130 bits
  Distance between Person 10 and Person 55: 133 bits
  Distance between Person 10 and Person 56: 121 bits
  Distance between Person 10 and Person 57: 127 bits
  Distance between Person 10 and Person 58: 130 bits
  Distance between Person 10 and Person 59: 143 bits
  Distance between Person 10 and Person 60: 140 bits
  Distance between Person 10 and Person 61: 130 bits
  Distance between Person 10 and Person 62: 123 bits
  Distance between Person 10 and Person 63: 119 bits
  Distance between Person 10 and Person 64: 108 bits
  Distance between Person 10 and Person 65: 122 bits
  Distance between Person 10 and Person 66: 123 bits
  Distance between Person 10 and Person 67: 116 bits
  Distance between Person 10 and Person 68: 110 bits
  Distance between Person 10 and Person 69: 120 bits
  Distance between Person 10 and Person 70: 132 bits
  Distance between Person 10 and Person 71: 136 bits
  Distance between Person 10 and Person 72: 134 bits
  Distance between Person 10 and Person 73: 138 bits
  Distance between Person 10 and Person 74: 129 bits
  Distance between Person 10 and Person 75: 126 bits
  Distance between Person 10 and Person 76: 125 bits
  Distance between Person 10 and Person 77: 128 bits
  Distance between Person 10 and Person 78: 143 bits
  Distance between Person 10 and Person 79: 114 bits
  Distance between Person 10 and Person 80: 122 bits
  Distance between Person 10 and Person 81: 119 bits
  Distance between Person 10 and Person 82: 109 bits
  Distance between Person 10 and Person 83: 136 bits
  Distance between Person 10 and Person 84: 122 bits
  Distance between Person 10 and Person 85: 138 bits
  Distance between Person 10 and Person 86: 138 bits
  Distance between Person 10 and Person 87: 128 bits
  Distance between Person 10 and Person 88: 121 bits
  Distance between Person 10 and Person 89: 128 bits
  Distance between Person 11 and Person 12: 122 bits
  Distance between Person 11 and Person 13: 152 bits
  Distance between Person 11 and Person 14: 121 bits
  Distance between Person 11 and Person 15: 125 bits
  Distance between Person 11 and Person 16: 138 bits
  Distance between Person 11 and Person 17: 119 bits
  Distance between Person 11 and Person 18: 137 bits
  Distance between Person 11 and Person 19: 122 bits
  Distance between Person 11 and Person 20: 132 bits
  Distance between Person 11 and Person 21: 138 bits
  Distance between Person 11 and Person 22: 135 bits
  Distance between Person 11 and Person 23: 128 bits
  Distance between Person 11 and Person 24: 120 bits
  Distance between Person 11 and Person 25: 131 bits
  Distance between Person 11 and Person 26: 137 bits
  Distance between Person 11 and Person 27: 147 bits
  Distance between Person 11 and Person 28: 120 bits
  Distance between Person 11 and Person 29: 117 bits
  Distance between Person 11 and Person 30: 119 bits
  Distance between Person 11 and Person 31: 115 bits
  Distance between Person 11 and Person 32: 127 bits
  Distance between Person 11 and Person 33: 125 bits
  Distance between Person 11 and Person 34: 136 bits
  Distance between Person 11 and Person 35: 126 bits
  Distance between Person 11 and Person 36: 125 bits
  Distance between Person 11 and Person 37: 124 bits
  Distance between Person 11 and Person 38: 137 bits
  Distance between Person 11 and Person 39: 135 bits
  Distance between Person 11 and Person 40: 126 bits
  Distance between Person 11 and Person 41: 124 bits
  Distance between Person 11 and Person 42: 122 bits
  Distance between Person 11 and Person 43: 149 bits
  Distance between Person 11 and Person 44: 126 bits
  Distance between Person 11 and Person 45: 125 bits
  Distance between Person 11 and Person 46: 131 bits
  Distance between Person 11 and Person 47: 125 bits
  Distance between Person 11 and Person 48: 132 bits
  Distance between Person 11 and Person 49: 136 bits
  Distance between Person 11 and Person 50: 127 bits
  Distance between Person 11 and Person 51: 145 bits
  Distance between Person 11 and Person 52: 145 bits
  Distance between Person 11 and Person 53: 135 bits
  Distance between Person 11 and Person 54: 125 bits
  Distance between Person 11 and Person 55: 122 bits
  Distance between Person 11 and Person 56: 122 bits
  Distance between Person 11 and Person 57: 116 bits
  Distance between Person 11 and Person 58: 131 bits
  Distance between Person 11 and Person 59: 120 bits
  Distance between Person 11 and Person 60: 129 bits
  Distance between Person 11 and Person 61: 111 bits
  Distance between Person 11 and Person 62: 140 bits
  Distance between Person 11 and Person 63: 142 bits
  Distance between Person 11 and Person 64: 127 bits
  Distance between Person 11 and Person 65: 115 bits
  Distance between Person 11 and Person 66: 130 bits
  Distance between Person 11 and Person 67: 131 bits
  Distance between Person 11 and Person 68: 117 bits
  Distance between Person 11 and Person 69: 139 bits
  Distance between Person 11 and Person 70: 139 bits
  Distance between Person 11 and Person 71: 127 bits
  Distance between Person 11 and Person 72: 137 bits
  Distance between Person 11 and Person 73: 135 bits
  Distance between Person 11 and Person 74: 136 bits
  Distance between Person 11 and Person 75: 141 bits
  Distance between Person 11 and Person 76: 122 bits
  Distance between Person 11 and Person 77: 119 bits
  Distance between Person 11 and Person 78: 130 bits
  Distance between Person 11 and Person 79: 131 bits
  Distance between Person 11 and Person 80: 139 bits
  Distance between Person 11 and Person 81: 142 bits
  Distance between Person 11 and Person 82: 138 bits
  Distance between Person 11 and Person 83: 139 bits
  Distance between Person 11 and Person 84: 119 bits
  Distance between Person 11 and Person 85: 133 bits
  Distance between Person 11 and Person 86: 133 bits
  Distance between Person 11 and Person 87: 125 bits
  Distance between Person 11 and Person 88: 120 bits
  Distance between Person 11 and Person 89: 129 bits
  Distance between Person 12 and Person 13: 144 bits
  Distance between Person 12 and Person 14: 123 bits
  Distance between Person 12 and Person 15: 119 bits
  Distance between Person 12 and Person 16: 114 bits
  Distance between Person 12 and Person 17: 97 bits
  Distance between Person 12 and Person 18: 125 bits
  Distance between Person 12 and Person 19: 126 bits
  Distance between Person 12 and Person 20: 116 bits
  Distance between Person 12 and Person 21: 136 bits
  Distance between Person 12 and Person 22: 121 bits
  Distance between Person 12 and Person 23: 128 bits
  Distance between Person 12 and Person 24: 134 bits
  Distance between Person 12 and Person 25: 145 bits
  Distance between Person 12 and Person 26: 115 bits
  Distance between Person 12 and Person 27: 129 bits
  Distance between Person 12 and Person 28: 132 bits
  Distance between Person 12 and Person 29: 137 bits
  Distance between Person 12 and Person 30: 131 bits
  Distance between Person 12 and Person 31: 111 bits
  Distance between Person 12 and Person 32: 129 bits
  Distance between Person 12 and Person 33: 139 bits
  Distance between Person 12 and Person 34: 132 bits
  Distance between Person 12 and Person 35: 142 bits
  Distance between Person 12 and Person 36: 127 bits
  Distance between Person 12 and Person 37: 132 bits
  Distance between Person 12 and Person 38: 129 bits
  Distance between Person 12 and Person 39: 143 bits
  Distance between Person 12 and Person 40: 122 bits
  Distance between Person 12 and Person 41: 126 bits
  Distance between Person 12 and Person 42: 122 bits
  Distance between Person 12 and Person 43: 115 bits
  Distance between Person 12 and Person 44: 138 bits
  Distance between Person 12 and Person 45: 127 bits
  Distance between Person 12 and Person 46: 123 bits
  Distance between Person 12 and Person 47: 129 bits
  Distance between Person 12 and Person 48: 114 bits
  Distance between Person 12 and Person 49: 134 bits
  Distance between Person 12 and Person 50: 135 bits
  Distance between Person 12 and Person 51: 113 bits
  Distance between Person 12 and Person 52: 127 bits
  Distance between Person 12 and Person 53: 123 bits
  Distance between Person 12 and Person 54: 123 bits
  Distance between Person 12 and Person 55: 120 bits
  Distance between Person 12 and Person 56: 116 bits
  Distance between Person 12 and Person 57: 126 bits
  Distance between Person 12 and Person 58: 121 bits
  Distance between Person 12 and Person 59: 132 bits
  Distance between Person 12 and Person 60: 135 bits
  Distance between Person 12 and Person 61: 117 bits
  Distance between Person 12 and Person 62: 126 bits
  Distance between Person 12 and Person 63: 128 bits
  Distance between Person 12 and Person 64: 127 bits
  Distance between Person 12 and Person 65: 141 bits
  Distance between Person 12 and Person 66: 138 bits
  Distance between Person 12 and Person 67: 133 bits
  Distance between Person 12 and Person 68: 135 bits
  Distance between Person 12 and Person 69: 135 bits
  Distance between Person 12 and Person 70: 117 bits
  Distance between Person 12 and Person 71: 133 bits
  Distance between Person 12 and Person 72: 129 bits
  Distance between Person 12 and Person 73: 117 bits
  Distance between Person 12 and Person 74: 144 bits
  Distance between Person 12 and Person 75: 133 bits
  Distance between Person 12 and Person 76: 122 bits
  Distance between Person 12 and Person 77: 125 bits
  Distance between Person 12 and Person 78: 130 bits
  Distance between Person 12 and Person 79: 131 bits
  Distance between Person 12 and Person 80: 117 bits
  Distance between Person 12 and Person 81: 128 bits
  Distance between Person 12 and Person 82: 132 bits
  Distance between Person 12 and Person 83: 133 bits
  Distance between Person 12 and Person 84: 123 bits
  Distance between Person 12 and Person 85: 117 bits
  Distance between Person 12 and Person 86: 123 bits
  Distance between Person 12 and Person 87: 123 bits
  Distance between Person 12 and Person 88: 116 bits
  Distance between Person 12 and Person 89: 113 bits
  Distance between Person 13 and Person 14: 129 bits
  Distance between Person 13 and Person 15: 115 bits
  Distance between Person 13 and Person 16: 136 bits
  Distance between Person 13 and Person 17: 125 bits
  Distance between Person 13 and Person 18: 129 bits
  Distance between Person 13 and Person 19: 138 bits
  Distance between Person 13 and Person 20: 104 bits
  Distance between Person 13 and Person 21: 112 bits
  Distance between Person 13 and Person 22: 103 bits
  Distance between Person 13 and Person 23: 126 bits
  Distance between Person 13 and Person 24: 130 bits
  Distance between Person 13 and Person 25: 111 bits
  Distance between Person 13 and Person 26: 133 bits
  Distance between Person 13 and Person 27: 119 bits
  Distance between Person 13 and Person 28: 140 bits
  Distance between Person 13 and Person 29: 143 bits
  Distance between Person 13 and Person 30: 131 bits
  Distance between Person 13 and Person 31: 135 bits
  Distance between Person 13 and Person 32: 99 bits
  Distance between Person 13 and Person 33: 125 bits
  Distance between Person 13 and Person 34: 122 bits
  Distance between Person 13 and Person 35: 120 bits
  Distance between Person 13 and Person 36: 123 bits
  Distance between Person 13 and Person 37: 118 bits
  Distance between Person 13 and Person 38: 115 bits
  Distance between Person 13 and Person 39: 117 bits
  Distance between Person 13 and Person 40: 150 bits
  Distance between Person 13 and Person 41: 142 bits
  Distance between Person 13 and Person 42: 144 bits
  Distance between Person 13 and Person 43: 133 bits
  Distance between Person 13 and Person 44: 126 bits
  Distance between Person 13 and Person 45: 127 bits
  Distance between Person 13 and Person 46: 125 bits
  Distance between Person 13 and Person 47: 139 bits
  Distance between Person 13 and Person 48: 130 bits
  Distance between Person 13 and Person 49: 116 bits
  Distance between Person 13 and Person 50: 127 bits
  Distance between Person 13 and Person 51: 137 bits
  Distance between Person 13 and Person 52: 123 bits
  Distance between Person 13 and Person 53: 133 bits
  Distance between Person 13 and Person 54: 135 bits
  Distance between Person 13 and Person 55: 134 bits
  Distance between Person 13 and Person 56: 118 bits
  Distance between Person 13 and Person 57: 138 bits
  Distance between Person 13 and Person 58: 143 bits
  Distance between Person 13 and Person 59: 136 bits
  Distance between Person 13 and Person 60: 115 bits
  Distance between Person 13 and Person 61: 131 bits
  Distance between Person 13 and Person 62: 120 bits
  Distance between Person 13 and Person 63: 140 bits
  Distance between Person 13 and Person 64: 117 bits
  Distance between Person 13 and Person 65: 133 bits
  Distance between Person 13 and Person 66: 128 bits
  Distance between Person 13 and Person 67: 125 bits
  Distance between Person 13 and Person 68: 147 bits
  Distance between Person 13 and Person 69: 131 bits
  Distance between Person 13 and Person 70: 111 bits
  Distance between Person 13 and Person 71: 143 bits
  Distance between Person 13 and Person 72: 119 bits
  Distance between Person 13 and Person 73: 75 bits
  Distance between Person 13 and Person 74: 116 bits
  Distance between Person 13 and Person 75: 121 bits
  Distance between Person 13 and Person 76: 144 bits
  Distance between Person 13 and Person 77: 141 bits
  Distance between Person 13 and Person 78: 104 bits
  Distance between Person 13 and Person 79: 131 bits
  Distance between Person 13 and Person 80: 83 bits
  Distance between Person 13 and Person 81: 130 bits
  Distance between Person 13 and Person 82: 130 bits
  Distance between Person 13 and Person 83: 127 bits
  Distance between Person 13 and Person 84: 117 bits
  Distance between Person 13 and Person 85: 131 bits
  Distance between Person 13 and Person 86: 117 bits
  Distance between Person 13 and Person 87: 139 bits
  Distance between Person 13 and Person 88: 138 bits
  Distance between Person 13 and Person 89: 157 bits
  Distance between Person 14 and Person 15: 122 bits
  Distance between Person 14 and Person 16: 109 bits
  Distance between Person 14 and Person 17: 110 bits
  Distance between Person 14 and Person 18: 140 bits
  Distance between Person 14 and Person 19: 123 bits
  Distance between Person 14 and Person 20: 109 bits
  Distance between Person 14 and Person 21: 145 bits
  Distance between Person 14 and Person 22: 132 bits
  Distance between Person 14 and Person 23: 143 bits
  Distance between Person 14 and Person 24: 133 bits
  Distance between Person 14 and Person 25: 136 bits
  Distance between Person 14 and Person 26: 138 bits
  Distance between Person 14 and Person 27: 110 bits
  Distance between Person 14 and Person 28: 119 bits
  Distance between Person 14 and Person 29: 132 bits
  Distance between Person 14 and Person 30: 124 bits
  Distance between Person 14 and Person 31: 136 bits
  Distance between Person 14 and Person 32: 130 bits
  Distance between Person 14 and Person 33: 126 bits
  Distance between Person 14 and Person 34: 125 bits
  Distance between Person 14 and Person 35: 135 bits
  Distance between Person 14 and Person 36: 156 bits
  Distance between Person 14 and Person 37: 127 bits
  Distance between Person 14 and Person 38: 134 bits
  Distance between Person 14 and Person 39: 138 bits
  Distance between Person 14 and Person 40: 133 bits
  Distance between Person 14 and Person 41: 133 bits
  Distance between Person 14 and Person 42: 129 bits
  Distance between Person 14 and Person 43: 120 bits
  Distance between Person 14 and Person 44: 123 bits
  Distance between Person 14 and Person 45: 102 bits
  Distance between Person 14 and Person 46: 116 bits
  Distance between Person 14 and Person 47: 120 bits
  Distance between Person 14 and Person 48: 147 bits
  Distance between Person 14 and Person 49: 127 bits
  Distance between Person 14 and Person 50: 122 bits
  Distance between Person 14 and Person 51: 114 bits
  Distance between Person 14 and Person 52: 124 bits
  Distance between Person 14 and Person 53: 128 bits
  Distance between Person 14 and Person 54: 132 bits
  Distance between Person 14 and Person 55: 137 bits
  Distance between Person 14 and Person 56: 131 bits
  Distance between Person 14 and Person 57: 145 bits
  Distance between Person 14 and Person 58: 120 bits
  Distance between Person 14 and Person 59: 123 bits
  Distance between Person 14 and Person 60: 138 bits
  Distance between Person 14 and Person 61: 134 bits
  Distance between Person 14 and Person 62: 141 bits
  Distance between Person 14 and Person 63: 131 bits
  Distance between Person 14 and Person 64: 142 bits
  Distance between Person 14 and Person 65: 128 bits
  Distance between Person 14 and Person 66: 131 bits
  Distance between Person 14 and Person 67: 128 bits
  Distance between Person 14 and Person 68: 100 bits
  Distance between Person 14 and Person 69: 118 bits
  Distance between Person 14 and Person 70: 148 bits
  Distance between Person 14 and Person 71: 124 bits
  Distance between Person 14 and Person 72: 114 bits
  Distance between Person 14 and Person 73: 104 bits
  Distance between Person 14 and Person 74: 121 bits
  Distance between Person 14 and Person 75: 124 bits
  Distance between Person 14 and Person 76: 119 bits
  Distance between Person 14 and Person 77: 126 bits
  Distance between Person 14 and Person 78: 125 bits
  Distance between Person 14 and Person 79: 130 bits
  Distance between Person 14 and Person 80: 112 bits
  Distance between Person 14 and Person 81: 127 bits
  Distance between Person 14 and Person 82: 121 bits
  Distance between Person 14 and Person 83: 120 bits
  Distance between Person 14 and Person 84: 112 bits
  Distance between Person 14 and Person 85: 132 bits
  Distance between Person 14 and Person 86: 110 bits
  Distance between Person 14 and Person 87: 136 bits
  Distance between Person 14 and Person 88: 135 bits
  Distance between Person 14 and Person 89: 104 bits
  Distance between Person 15 and Person 16: 121 bits
  Distance between Person 15 and Person 17: 114 bits
  Distance between Person 15 and Person 18: 126 bits
  Distance between Person 15 and Person 19: 121 bits
  Distance between Person 15 and Person 20: 119 bits
  Distance between Person 15 and Person 21: 137 bits
  Distance between Person 15 and Person 22: 134 bits
  Distance between Person 15 and Person 23: 119 bits
  Distance between Person 15 and Person 24: 133 bits
  Distance between Person 15 and Person 25: 136 bits
  Distance between Person 15 and Person 26: 142 bits
  Distance between Person 15 and Person 27: 152 bits
  Distance between Person 15 and Person 28: 125 bits
  Distance between Person 15 and Person 29: 128 bits
  Distance between Person 15 and Person 30: 138 bits
  Distance between Person 15 and Person 31: 116 bits
  Distance between Person 15 and Person 32: 132 bits
  Distance between Person 15 and Person 33: 156 bits
  Distance between Person 15 and Person 34: 127 bits
  Distance between Person 15 and Person 35: 121 bits
  Distance between Person 15 and Person 36: 116 bits
  Distance between Person 15 and Person 37: 109 bits
  Distance between Person 15 and Person 38: 118 bits
  Distance between Person 15 and Person 39: 138 bits
  Distance between Person 15 and Person 40: 123 bits
  Distance between Person 15 and Person 41: 121 bits
  Distance between Person 15 and Person 42: 135 bits
  Distance between Person 15 and Person 43: 128 bits
  Distance between Person 15 and Person 44: 121 bits
  Distance between Person 15 and Person 45: 118 bits
  Distance between Person 15 and Person 46: 126 bits
  Distance between Person 15 and Person 47: 146 bits
  Distance between Person 15 and Person 48: 133 bits
  Distance between Person 15 and Person 49: 127 bits
  Distance between Person 15 and Person 50: 124 bits
  Distance between Person 15 and Person 51: 118 bits
  Distance between Person 15 and Person 52: 120 bits
  Distance between Person 15 and Person 53: 146 bits
  Distance between Person 15 and Person 54: 122 bits
  Distance between Person 15 and Person 55: 129 bits
  Distance between Person 15 and Person 56: 127 bits
  Distance between Person 15 and Person 57: 129 bits
  Distance between Person 15 and Person 58: 116 bits
  Distance between Person 15 and Person 59: 125 bits
  Distance between Person 15 and Person 60: 116 bits
  Distance between Person 15 and Person 61: 128 bits
  Distance between Person 15 and Person 62: 127 bits
  Distance between Person 15 and Person 63: 129 bits
  Distance between Person 15 and Person 64: 116 bits
  Distance between Person 15 and Person 65: 130 bits
  Distance between Person 15 and Person 66: 131 bits
  Distance between Person 15 and Person 67: 148 bits
  Distance between Person 15 and Person 68: 96 bits
  Distance between Person 15 and Person 69: 144 bits
  Distance between Person 15 and Person 70: 126 bits
  Distance between Person 15 and Person 71: 126 bits
  Distance between Person 15 and Person 72: 122 bits
  Distance between Person 15 and Person 73: 98 bits
  Distance between Person 15 and Person 74: 127 bits
  Distance between Person 15 and Person 75: 130 bits
  Distance between Person 15 and Person 76: 131 bits
  Distance between Person 15 and Person 77: 130 bits
  Distance between Person 15 and Person 78: 133 bits
  Distance between Person 15 and Person 79: 126 bits
  Distance between Person 15 and Person 80: 106 bits
  Distance between Person 15 and Person 81: 125 bits
  Distance between Person 15 and Person 82: 131 bits
  Distance between Person 15 and Person 83: 130 bits
  Distance between Person 15 and Person 84: 128 bits
  Distance between Person 15 and Person 85: 124 bits
  Distance between Person 15 and Person 86: 138 bits
  Distance between Person 15 and Person 87: 124 bits
  Distance between Person 15 and Person 88: 149 bits
  Distance between Person 15 and Person 89: 128 bits
  Distance between Person 16 and Person 17: 125 bits
  Distance between Person 16 and Person 18: 125 bits
  Distance between Person 16 and Person 19: 124 bits
  Distance between Person 16 and Person 20: 126 bits
  Distance between Person 16 and Person 21: 120 bits
  Distance between Person 16 and Person 22: 135 bits
  Distance between Person 16 and Person 23: 130 bits
  Distance between Person 16 and Person 24: 144 bits
  Distance between Person 16 and Person 25: 131 bits
  Distance between Person 16 and Person 26: 127 bits
  Distance between Person 16 and Person 27: 119 bits
  Distance between Person 16 and Person 28: 140 bits
  Distance between Person 16 and Person 29: 137 bits
  Distance between Person 16 and Person 30: 141 bits
  Distance between Person 16 and Person 31: 119 bits
  Distance between Person 16 and Person 32: 125 bits
  Distance between Person 16 and Person 33: 119 bits
  Distance between Person 16 and Person 34: 134 bits
  Distance between Person 16 and Person 35: 128 bits
  Distance between Person 16 and Person 36: 137 bits
  Distance between Person 16 and Person 37: 126 bits
  Distance between Person 16 and Person 38: 129 bits
  Distance between Person 16 and Person 39: 155 bits
  Distance between Person 16 and Person 40: 134 bits
  Distance between Person 16 and Person 41: 126 bits
  Distance between Person 16 and Person 42: 122 bits
  Distance between Person 16 and Person 43: 121 bits
  Distance between Person 16 and Person 44: 130 bits
  Distance between Person 16 and Person 45: 131 bits
  Distance between Person 16 and Person 46: 125 bits
  Distance between Person 16 and Person 47: 135 bits
  Distance between Person 16 and Person 48: 140 bits
  Distance between Person 16 and Person 49: 112 bits
  Distance between Person 16 and Person 50: 149 bits
  Distance between Person 16 and Person 51: 123 bits
  Distance between Person 16 and Person 52: 127 bits
  Distance between Person 16 and Person 53: 129 bits
  Distance between Person 16 and Person 54: 135 bits
  Distance between Person 16 and Person 55: 120 bits
  Distance between Person 16 and Person 56: 144 bits
  Distance between Person 16 and Person 57: 128 bits
  Distance between Person 16 and Person 58: 111 bits
  Distance between Person 16 and Person 59: 122 bits
  Distance between Person 16 and Person 60: 127 bits
  Distance between Person 16 and Person 61: 131 bits
  Distance between Person 16 and Person 62: 116 bits
  Distance between Person 16 and Person 63: 130 bits
  Distance between Person 16 and Person 64: 127 bits
  Distance between Person 16 and Person 65: 123 bits
  Distance between Person 16 and Person 66: 136 bits
  Distance between Person 16 and Person 67: 131 bits
  Distance between Person 16 and Person 68: 117 bits
  Distance between Person 16 and Person 69: 117 bits
  Distance between Person 16 and Person 70: 129 bits
  Distance between Person 16 and Person 71: 131 bits
  Distance between Person 16 and Person 72: 133 bits
  Distance between Person 16 and Person 73: 123 bits
  Distance between Person 16 and Person 74: 132 bits
  Distance between Person 16 and Person 75: 133 bits
  Distance between Person 16 and Person 76: 130 bits
  Distance between Person 16 and Person 77: 119 bits
  Distance between Person 16 and Person 78: 106 bits
  Distance between Person 16 and Person 79: 115 bits
  Distance between Person 16 and Person 80: 105 bits
  Distance between Person 16 and Person 81: 118 bits
  Distance between Person 16 and Person 82: 126 bits
  Distance between Person 16 and Person 83: 119 bits
  Distance between Person 16 and Person 84: 127 bits
  Distance between Person 16 and Person 85: 133 bits
  Distance between Person 16 and Person 86: 119 bits
  Distance between Person 16 and Person 87: 135 bits
  Distance between Person 16 and Person 88: 134 bits
  Distance between Person 16 and Person 89: 131 bits
  Distance between Person 17 and Person 18: 128 bits
  Distance between Person 17 and Person 19: 141 bits
  Distance between Person 17 and Person 20: 125 bits
  Distance between Person 17 and Person 21: 137 bits
  Distance between Person 17 and Person 22: 144 bits
  Distance between Person 17 and Person 23: 113 bits
  Distance between Person 17 and Person 24: 133 bits
  Distance between Person 17 and Person 25: 128 bits
  Distance between Person 17 and Person 26: 132 bits
  Distance between Person 17 and Person 27: 118 bits
  Distance between Person 17 and Person 28: 139 bits
  Distance between Person 17 and Person 29: 140 bits
  Distance between Person 17 and Person 30: 102 bits
  Distance between Person 17 and Person 31: 112 bits
  Distance between Person 17 and Person 32: 124 bits
  Distance between Person 17 and Person 33: 128 bits
  Distance between Person 17 and Person 34: 123 bits
  Distance between Person 17 and Person 35: 135 bits
  Distance between Person 17 and Person 36: 132 bits
  Distance between Person 17 and Person 37: 131 bits
  Distance between Person 17 and Person 38: 128 bits
  Distance between Person 17 and Person 39: 134 bits
  Distance between Person 17 and Person 40: 113 bits
  Distance between Person 17 and Person 41: 133 bits
  Distance between Person 17 and Person 42: 123 bits
  Distance between Person 17 and Person 43: 116 bits
  Distance between Person 17 and Person 44: 143 bits
  Distance between Person 17 and Person 45: 130 bits
  Distance between Person 17 and Person 46: 126 bits
  Distance between Person 17 and Person 47: 142 bits
  Distance between Person 17 and Person 48: 137 bits
  Distance between Person 17 and Person 49: 117 bits
  Distance between Person 17 and Person 50: 136 bits
  Distance between Person 17 and Person 51: 126 bits
  Distance between Person 17 and Person 52: 126 bits
  Distance between Person 17 and Person 53: 142 bits
  Distance between Person 17 and Person 54: 146 bits
  Distance between Person 17 and Person 55: 141 bits
  Distance between Person 17 and Person 56: 149 bits
  Distance between Person 17 and Person 57: 131 bits
  Distance between Person 17 and Person 58: 128 bits
  Distance between Person 17 and Person 59: 133 bits
  Distance between Person 17 and Person 60: 150 bits
  Distance between Person 17 and Person 61: 130 bits
  Distance between Person 17 and Person 62: 117 bits
  Distance between Person 17 and Person 63: 141 bits
  Distance between Person 17 and Person 64: 128 bits
  Distance between Person 17 and Person 65: 136 bits
  Distance between Person 17 and Person 66: 139 bits
  Distance between Person 17 and Person 67: 130 bits
  Distance between Person 17 and Person 68: 120 bits
  Distance between Person 17 and Person 69: 130 bits
  Distance between Person 17 and Person 70: 124 bits
  Distance between Person 17 and Person 71: 124 bits
  Distance between Person 17 and Person 72: 122 bits
  Distance between Person 17 and Person 73: 118 bits
  Distance between Person 17 and Person 74: 125 bits
  Distance between Person 17 and Person 75: 140 bits
  Distance between Person 17 and Person 76: 135 bits
  Distance between Person 17 and Person 77: 120 bits
  Distance between Person 17 and Person 78: 121 bits
  Distance between Person 17 and Person 79: 138 bits
  Distance between Person 17 and Person 80: 112 bits
  Distance between Person 17 and Person 81: 113 bits
  Distance between Person 17 and Person 82: 143 bits
  Distance between Person 17 and Person 83: 134 bits
  Distance between Person 17 and Person 84: 134 bits
  Distance between Person 17 and Person 85: 140 bits
  Distance between Person 17 and Person 86: 120 bits
  Distance between Person 17 and Person 87: 124 bits
  Distance between Person 17 and Person 88: 135 bits
  Distance between Person 17 and Person 89: 124 bits
  Distance between Person 18 and Person 19: 119 bits
  Distance between Person 18 and Person 20: 149 bits
  Distance between Person 18 and Person 21: 133 bits
  Distance between Person 18 and Person 22: 138 bits
  Distance between Person 18 and Person 23: 129 bits
  Distance between Person 18 and Person 24: 137 bits
  Distance between Person 18 and Person 25: 122 bits
  Distance between Person 18 and Person 26: 132 bits
  Distance between Person 18 and Person 27: 128 bits
  Distance between Person 18 and Person 28: 127 bits
  Distance between Person 18 and Person 29: 128 bits
  Distance between Person 18 and Person 30: 138 bits
  Distance between Person 18 and Person 31: 132 bits
  Distance between Person 18 and Person 32: 156 bits
  Distance between Person 18 and Person 33: 120 bits
  Distance between Person 18 and Person 34: 115 bits
  Distance between Person 18 and Person 35: 131 bits
  Distance between Person 18 and Person 36: 120 bits
  Distance between Person 18 and Person 37: 129 bits
  Distance between Person 18 and Person 38: 122 bits
  Distance between Person 18 and Person 39: 132 bits
  Distance between Person 18 and Person 40: 125 bits
  Distance between Person 18 and Person 41: 125 bits
  Distance between Person 18 and Person 42: 99 bits
  Distance between Person 18 and Person 43: 120 bits
  Distance between Person 18 and Person 44: 135 bits
  Distance between Person 18 and Person 45: 134 bits
  Distance between Person 18 and Person 46: 104 bits
  Distance between Person 18 and Person 47: 134 bits
  Distance between Person 18 and Person 48: 115 bits
  Distance between Person 18 and Person 49: 133 bits
  Distance between Person 18 and Person 50: 130 bits
  Distance between Person 18 and Person 51: 118 bits
  Distance between Person 18 and Person 52: 136 bits
  Distance between Person 18 and Person 53: 134 bits
  Distance between Person 18 and Person 54: 128 bits
  Distance between Person 18 and Person 55: 123 bits
  Distance between Person 18 and Person 56: 117 bits
  Distance between Person 18 and Person 57: 131 bits
  Distance between Person 18 and Person 58: 126 bits
  Distance between Person 18 and Person 59: 115 bits
  Distance between Person 18 and Person 60: 114 bits
  Distance between Person 18 and Person 61: 158 bits
  Distance between Person 18 and Person 62: 131 bits
  Distance between Person 18 and Person 63: 117 bits
  Distance between Person 18 and Person 64: 126 bits
  Distance between Person 18 and Person 65: 130 bits
  Distance between Person 18 and Person 66: 135 bits
  Distance between Person 18 and Person 67: 118 bits
  Distance between Person 18 and Person 68: 116 bits
  Distance between Person 18 and Person 69: 132 bits
  Distance between Person 18 and Person 70: 104 bits
  Distance between Person 18 and Person 71: 120 bits
  Distance between Person 18 and Person 72: 152 bits
  Distance between Person 18 and Person 73: 126 bits
  Distance between Person 18 and Person 74: 133 bits
  Distance between Person 18 and Person 75: 134 bits
  Distance between Person 18 and Person 76: 127 bits
  Distance between Person 18 and Person 77: 120 bits
  Distance between Person 18 and Person 78: 89 bits
  Distance between Person 18 and Person 79: 124 bits
  Distance between Person 18 and Person 80: 126 bits
  Distance between Person 18 and Person 81: 99 bits
  Distance between Person 18 and Person 82: 121 bits
  Distance between Person 18 and Person 83: 144 bits
  Distance between Person 18 and Person 84: 142 bits
  Distance between Person 18 and Person 85: 140 bits
  Distance between Person 18 and Person 86: 104 bits
  Distance between Person 18 and Person 87: 136 bits
  Distance between Person 18 and Person 88: 125 bits
  Distance between Person 18 and Person 89: 142 bits
  Distance between Person 19 and Person 20: 114 bits
  Distance between Person 19 and Person 21: 134 bits
  Distance between Person 19 and Person 22: 137 bits
  Distance between Person 19 and Person 23: 136 bits
  Distance between Person 19 and Person 24: 122 bits
  Distance between Person 19 and Person 25: 125 bits
  Distance between Person 19 and Person 26: 131 bits
  Distance between Person 19 and Person 27: 125 bits
  Distance between Person 19 and Person 28: 132 bits
  Distance between Person 19 and Person 29: 117 bits
  Distance between Person 19 and Person 30: 137 bits
  Distance between Person 19 and Person 31: 139 bits
  Distance between Person 19 and Person 32: 129 bits
  Distance between Person 19 and Person 33: 129 bits
  Distance between Person 19 and Person 34: 132 bits
  Distance between Person 19 and Person 35: 128 bits
  Distance between Person 19 and Person 36: 125 bits
  Distance between Person 19 and Person 37: 120 bits
  Distance between Person 19 and Person 38: 137 bits
  Distance between Person 19 and Person 39: 97 bits
  Distance between Person 19 and Person 40: 110 bits
  Distance between Person 19 and Person 41: 110 bits
  Distance between Person 19 and Person 42: 126 bits
  Distance between Person 19 and Person 43: 117 bits
  Distance between Person 19 and Person 44: 128 bits
  Distance between Person 19 and Person 45: 99 bits
  Distance between Person 19 and Person 46: 145 bits
  Distance between Person 19 and Person 47: 91 bits
  Distance between Person 19 and Person 48: 120 bits
  Distance between Person 19 and Person 49: 122 bits
  Distance between Person 19 and Person 50: 121 bits
  Distance between Person 19 and Person 51: 135 bits
  Distance between Person 19 and Person 52: 131 bits
  Distance between Person 19 and Person 53: 147 bits
  Distance between Person 19 and Person 54: 125 bits
  Distance between Person 19 and Person 55: 90 bits
  Distance between Person 19 and Person 56: 114 bits
  Distance between Person 19 and Person 57: 122 bits
  Distance between Person 19 and Person 58: 115 bits
  Distance between Person 19 and Person 59: 114 bits
  Distance between Person 19 and Person 60: 113 bits
  Distance between Person 19 and Person 61: 125 bits
  Distance between Person 19 and Person 62: 144 bits
  Distance between Person 19 and Person 63: 138 bits
  Distance between Person 19 and Person 64: 121 bits
  Distance between Person 19 and Person 65: 133 bits
  Distance between Person 19 and Person 66: 116 bits
  Distance between Person 19 and Person 67: 115 bits
  Distance between Person 19 and Person 68: 125 bits
  Distance between Person 19 and Person 69: 117 bits
  Distance between Person 19 and Person 70: 133 bits
  Distance between Person 19 and Person 71: 107 bits
  Distance between Person 19 and Person 72: 135 bits
  Distance between Person 19 and Person 73: 133 bits
  Distance between Person 19 and Person 74: 136 bits
  Distance between Person 19 and Person 75: 137 bits
  Distance between Person 19 and Person 76: 120 bits
  Distance between Person 19 and Person 77: 117 bits
  Distance between Person 19 and Person 78: 122 bits
  Distance between Person 19 and Person 79: 121 bits
  Distance between Person 19 and Person 80: 117 bits
  Distance between Person 19 and Person 81: 130 bits
  Distance between Person 19 and Person 82: 122 bits
  Distance between Person 19 and Person 83: 119 bits
  Distance between Person 19 and Person 84: 141 bits
  Distance between Person 19 and Person 85: 119 bits
  Distance between Person 19 and Person 86: 115 bits
  Distance between Person 19 and Person 87: 135 bits
  Distance between Person 19 and Person 88: 144 bits
  Distance between Person 19 and Person 89: 133 bits
  Distance between Person 20 and Person 21: 130 bits
  Distance between Person 20 and Person 22: 115 bits
  Distance between Person 20 and Person 23: 122 bits
  Distance between Person 20 and Person 24: 126 bits
  Distance between Person 20 and Person 25: 139 bits
  Distance between Person 20 and Person 26: 131 bits
  Distance between Person 20 and Person 27: 133 bits
  Distance between Person 20 and Person 28: 146 bits
  Distance between Person 20 and Person 29: 129 bits
  Distance between Person 20 and Person 30: 129 bits
  Distance between Person 20 and Person 31: 127 bits
  Distance between Person 20 and Person 32: 109 bits
  Distance between Person 20 and Person 33: 133 bits
  Distance between Person 20 and Person 34: 136 bits
  Distance between Person 20 and Person 35: 128 bits
  Distance between Person 20 and Person 36: 129 bits
  Distance between Person 20 and Person 37: 120 bits
  Distance between Person 20 and Person 38: 133 bits
  Distance between Person 20 and Person 39: 121 bits
  Distance between Person 20 and Person 40: 124 bits
  Distance between Person 20 and Person 41: 116 bits
  Distance between Person 20 and Person 42: 124 bits
  Distance between Person 20 and Person 43: 131 bits
  Distance between Person 20 and Person 44: 126 bits
  Distance between Person 20 and Person 45: 103 bits
  Distance between Person 20 and Person 46: 119 bits
  Distance between Person 20 and Person 47: 119 bits
  Distance between Person 20 and Person 48: 148 bits
  Distance between Person 20 and Person 49: 120 bits
  Distance between Person 20 and Person 50: 117 bits
  Distance between Person 20 and Person 51: 131 bits
  Distance between Person 20 and Person 52: 139 bits
  Distance between Person 20 and Person 53: 127 bits
  Distance between Person 20 and Person 54: 127 bits
  Distance between Person 20 and Person 55: 144 bits
  Distance between Person 20 and Person 56: 120 bits
  Distance between Person 20 and Person 57: 128 bits
  Distance between Person 20 and Person 58: 127 bits
  Distance between Person 20 and Person 59: 126 bits
  Distance between Person 20 and Person 60: 127 bits
  Distance between Person 20 and Person 61: 123 bits
  Distance between Person 20 and Person 62: 134 bits
  Distance between Person 20 and Person 63: 136 bits
  Distance between Person 20 and Person 64: 139 bits
  Distance between Person 20 and Person 65: 141 bits
  Distance between Person 20 and Person 66: 136 bits
  Distance between Person 20 and Person 67: 127 bits
  Distance between Person 20 and Person 68: 141 bits
  Distance between Person 20 and Person 69: 129 bits
  Distance between Person 20 and Person 70: 129 bits
  Distance between Person 20 and Person 71: 129 bits
  Distance between Person 20 and Person 72: 121 bits
  Distance between Person 20 and Person 73: 103 bits
  Distance between Person 20 and Person 74: 118 bits
  Distance between Person 20 and Person 75: 125 bits
  Distance between Person 20 and Person 76: 128 bits
  Distance between Person 20 and Person 77: 133 bits
  Distance between Person 20 and Person 78: 110 bits
  Distance between Person 20 and Person 79: 127 bits
  Distance between Person 20 and Person 80: 127 bits
  Distance between Person 20 and Person 81: 122 bits
  Distance between Person 20 and Person 82: 106 bits
  Distance between Person 20 and Person 83: 133 bits
  Distance between Person 20 and Person 84: 135 bits
  Distance between Person 20 and Person 85: 139 bits
  Distance between Person 20 and Person 86: 109 bits
  Distance between Person 20 and Person 87: 125 bits
  Distance between Person 20 and Person 88: 142 bits
  Distance between Person 20 and Person 89: 107 bits
  Distance between Person 21 and Person 22: 131 bits
  Distance between Person 21 and Person 23: 116 bits
  Distance between Person 21 and Person 24: 128 bits
  Distance between Person 21 and Person 25: 117 bits
  Distance between Person 21 and Person 26: 135 bits
  Distance between Person 21 and Person 27: 125 bits
  Distance between Person 21 and Person 28: 126 bits
  Distance between Person 21 and Person 29: 127 bits
  Distance between Person 21 and Person 30: 149 bits
  Distance between Person 21 and Person 31: 127 bits
  Distance between Person 21 and Person 32: 119 bits
  Distance between Person 21 and Person 33: 129 bits
  Distance between Person 21 and Person 34: 126 bits
  Distance between Person 21 and Person 35: 138 bits
  Distance between Person 21 and Person 36: 133 bits
  Distance between Person 21 and Person 37: 126 bits
  Distance between Person 21 and Person 38: 75 bits
  Distance between Person 21 and Person 39: 121 bits
  Distance between Person 21 and Person 40: 120 bits
  Distance between Person 21 and Person 41: 126 bits
  Distance between Person 21 and Person 42: 130 bits
  Distance between Person 21 and Person 43: 129 bits
  Distance between Person 21 and Person 44: 116 bits
  Distance between Person 21 and Person 45: 119 bits
  Distance between Person 21 and Person 46: 127 bits
  Distance between Person 21 and Person 47: 129 bits
  Distance between Person 21 and Person 48: 128 bits
  Distance between Person 21 and Person 49: 120 bits
  Distance between Person 21 and Person 50: 117 bits
  Distance between Person 21 and Person 51: 129 bits
  Distance between Person 21 and Person 52: 135 bits
  Distance between Person 21 and Person 53: 131 bits
  Distance between Person 21 and Person 54: 121 bits
  Distance between Person 21 and Person 55: 126 bits
  Distance between Person 21 and Person 56: 124 bits
  Distance between Person 21 and Person 57: 126 bits
  Distance between Person 21 and Person 58: 129 bits
  Distance between Person 21 and Person 59: 120 bits
  Distance between Person 21 and Person 60: 117 bits
  Distance between Person 21 and Person 61: 105 bits
  Distance between Person 21 and Person 62: 94 bits
  Distance between Person 21 and Person 63: 130 bits
  Distance between Person 21 and Person 64: 129 bits
  Distance between Person 21 and Person 65: 121 bits
  Distance between Person 21 and Person 66: 124 bits
  Distance between Person 21 and Person 67: 151 bits
  Distance between Person 21 and Person 68: 131 bits
  Distance between Person 21 and Person 69: 131 bits
  Distance between Person 21 and Person 70: 123 bits
  Distance between Person 21 and Person 71: 113 bits
  Distance between Person 21 and Person 72: 131 bits
  Distance between Person 21 and Person 73: 125 bits
  Distance between Person 21 and Person 74: 132 bits
  Distance between Person 21 and Person 75: 123 bits
  Distance between Person 21 and Person 76: 120 bits
  Distance between Person 21 and Person 77: 115 bits
  Distance between Person 21 and Person 78: 116 bits
  Distance between Person 21 and Person 79: 143 bits
  Distance between Person 21 and Person 80: 117 bits
  Distance between Person 21 and Person 81: 138 bits
  Distance between Person 21 and Person 82: 124 bits
  Distance between Person 21 and Person 83: 127 bits
  Distance between Person 21 and Person 84: 135 bits
  Distance between Person 21 and Person 85: 129 bits
  Distance between Person 21 and Person 86: 131 bits
  Distance between Person 21 and Person 87: 137 bits
  Distance between Person 21 and Person 88: 116 bits
  Distance between Person 21 and Person 89: 137 bits
  Distance between Person 22 and Person 23: 127 bits
  Distance between Person 22 and Person 24: 121 bits
  Distance between Person 22 and Person 25: 130 bits
  Distance between Person 22 and Person 26: 120 bits
  Distance between Person 22 and Person 27: 116 bits
  Distance between Person 22 and Person 28: 135 bits
  Distance between Person 22 and Person 29: 122 bits
  Distance between Person 22 and Person 30: 122 bits
  Distance between Person 22 and Person 31: 132 bits
  Distance between Person 22 and Person 32: 134 bits
  Distance between Person 22 and Person 33: 136 bits
  Distance between Person 22 and Person 34: 125 bits
  Distance between Person 22 and Person 35: 133 bits
  Distance between Person 22 and Person 36: 110 bits
  Distance between Person 22 and Person 37: 151 bits
  Distance between Person 22 and Person 38: 142 bits
  Distance between Person 22 and Person 39: 114 bits
  Distance between Person 22 and Person 40: 127 bits
  Distance between Person 22 and Person 41: 127 bits
  Distance between Person 22 and Person 42: 141 bits
  Distance between Person 22 and Person 43: 120 bits
  Distance between Person 22 and Person 44: 161 bits
  Distance between Person 22 and Person 45: 150 bits
  Distance between Person 22 and Person 46: 126 bits
  Distance between Person 22 and Person 47: 130 bits
  Distance between Person 22 and Person 48: 131 bits
  Distance between Person 22 and Person 49: 137 bits
  Distance between Person 22 and Person 50: 122 bits
  Distance between Person 22 and Person 51: 116 bits
  Distance between Person 22 and Person 52: 118 bits
  Distance between Person 22 and Person 53: 128 bits
  Distance between Person 22 and Person 54: 126 bits
  Distance between Person 22 and Person 55: 141 bits
  Distance between Person 22 and Person 56: 103 bits
  Distance between Person 22 and Person 57: 121 bits
  Distance between Person 22 and Person 58: 134 bits
  Distance between Person 22 and Person 59: 117 bits
  Distance between Person 22 and Person 60: 136 bits
  Distance between Person 22 and Person 61: 122 bits
  Distance between Person 22 and Person 62: 125 bits
  Distance between Person 22 and Person 63: 143 bits
  Distance between Person 22 and Person 64: 120 bits
  Distance between Person 22 and Person 65: 140 bits
  Distance between Person 22 and Person 66: 141 bits
  Distance between Person 22 and Person 67: 128 bits
  Distance between Person 22 and Person 68: 130 bits
  Distance between Person 22 and Person 69: 132 bits
  Distance between Person 22 and Person 70: 136 bits
  Distance between Person 22 and Person 71: 144 bits
  Distance between Person 22 and Person 72: 140 bits
  Distance between Person 22 and Person 73: 108 bits
  Distance between Person 22 and Person 74: 125 bits
  Distance between Person 22 and Person 75: 134 bits
  Distance between Person 22 and Person 76: 131 bits
  Distance between Person 22 and Person 77: 136 bits
  Distance between Person 22 and Person 78: 133 bits
  Distance between Person 22 and Person 79: 126 bits
  Distance between Person 22 and Person 80: 142 bits
  Distance between Person 22 and Person 81: 129 bits
  Distance between Person 22 and Person 82: 135 bits
  Distance between Person 22 and Person 83: 124 bits
  Distance between Person 22 and Person 84: 110 bits
  Distance between Person 22 and Person 85: 122 bits
  Distance between Person 22 and Person 86: 140 bits
  Distance between Person 22 and Person 87: 128 bits
  Distance between Person 22 and Person 88: 123 bits
  Distance between Person 22 and Person 89: 118 bits
  Distance between Person 23 and Person 24: 130 bits
  Distance between Person 23 and Person 25: 123 bits
  Distance between Person 23 and Person 26: 161 bits
  Distance between Person 23 and Person 27: 135 bits
  Distance between Person 23 and Person 28: 134 bits
  Distance between Person 23 and Person 29: 119 bits
  Distance between Person 23 and Person 30: 121 bits
  Distance between Person 23 and Person 31: 123 bits
  Distance between Person 23 and Person 32: 119 bits
  Distance between Person 23 and Person 33: 137 bits
  Distance between Person 23 and Person 34: 128 bits
  Distance between Person 23 and Person 35: 132 bits
  Distance between Person 23 and Person 36: 133 bits
  Distance between Person 23 and Person 37: 132 bits
  Distance between Person 23 and Person 38: 125 bits
  Distance between Person 23 and Person 39: 137 bits
  Distance between Person 23 and Person 40: 124 bits
  Distance between Person 23 and Person 41: 130 bits
  Distance between Person 23 and Person 42: 124 bits
  Distance between Person 23 and Person 43: 127 bits
  Distance between Person 23 and Person 44: 144 bits
  Distance between Person 23 and Person 45: 119 bits
  Distance between Person 23 and Person 46: 119 bits
  Distance between Person 23 and Person 47: 139 bits
  Distance between Person 23 and Person 48: 114 bits
  Distance between Person 23 and Person 49: 122 bits
  Distance between Person 23 and Person 50: 131 bits
  Distance between Person 23 and Person 51: 125 bits
  Distance between Person 23 and Person 52: 161 bits
  Distance between Person 23 and Person 53: 115 bits
  Distance between Person 23 and Person 54: 135 bits
  Distance between Person 23 and Person 55: 138 bits
  Distance between Person 23 and Person 56: 132 bits
  Distance between Person 23 and Person 57: 136 bits
  Distance between Person 23 and Person 58: 129 bits
  Distance between Person 23 and Person 59: 142 bits
  Distance between Person 23 and Person 60: 113 bits
  Distance between Person 23 and Person 61: 119 bits
  Distance between Person 23 and Person 62: 124 bits
  Distance between Person 23 and Person 63: 118 bits
  Distance between Person 23 and Person 64: 141 bits
  Distance between Person 23 and Person 65: 131 bits
  Distance between Person 23 and Person 66: 144 bits
  Distance between Person 23 and Person 67: 147 bits
  Distance between Person 23 and Person 68: 127 bits
  Distance between Person 23 and Person 69: 139 bits
  Distance between Person 23 and Person 70: 141 bits
  Distance between Person 23 and Person 71: 129 bits
  Distance between Person 23 and Person 72: 139 bits
  Distance between Person 23 and Person 73: 123 bits
  Distance between Person 23 and Person 74: 114 bits
  Distance between Person 23 and Person 75: 135 bits
  Distance between Person 23 and Person 76: 132 bits
  Distance between Person 23 and Person 77: 123 bits
  Distance between Person 23 and Person 78: 136 bits
  Distance between Person 23 and Person 79: 157 bits
  Distance between Person 23 and Person 80: 127 bits
  Distance between Person 23 and Person 81: 120 bits
  Distance between Person 23 and Person 82: 134 bits
  Distance between Person 23 and Person 83: 117 bits
  Distance between Person 23 and Person 84: 135 bits
  Distance between Person 23 and Person 85: 121 bits
  Distance between Person 23 and Person 86: 121 bits
  Distance between Person 23 and Person 87: 151 bits
  Distance between Person 23 and Person 88: 140 bits
  Distance between Person 23 and Person 89: 139 bits
  Distance between Person 24 and Person 25: 141 bits
  Distance between Person 24 and Person 26: 145 bits
  Distance between Person 24 and Person 27: 131 bits
  Distance between Person 24 and Person 28: 108 bits
  Distance between Person 24 and Person 29: 117 bits
  Distance between Person 24 and Person 30: 133 bits
  Distance between Person 24 and Person 31: 133 bits
  Distance between Person 24 and Person 32: 133 bits
  Distance between Person 24 and Person 33: 119 bits
  Distance between Person 24 and Person 34: 138 bits
  Distance between Person 24 and Person 35: 138 bits
  Distance between Person 24 and Person 36: 129 bits
  Distance between Person 24 and Person 37: 122 bits
  Distance between Person 24 and Person 38: 125 bits
  Distance between Person 24 and Person 39: 121 bits
  Distance between Person 24 and Person 40: 118 bits
  Distance between Person 24 and Person 41: 134 bits
  Distance between Person 24 and Person 42: 132 bits
  Distance between Person 24 and Person 43: 137 bits
  Distance between Person 24 and Person 44: 130 bits
  Distance between Person 24 and Person 45: 121 bits
  Distance between Person 24 and Person 46: 129 bits
  Distance between Person 24 and Person 47: 121 bits
  Distance between Person 24 and Person 48: 110 bits
  Distance between Person 24 and Person 49: 130 bits
  Distance between Person 24 and Person 50: 129 bits
  Distance between Person 24 and Person 51: 127 bits
  Distance between Person 24 and Person 52: 127 bits
  Distance between Person 24 and Person 53: 121 bits
  Distance between Person 24 and Person 54: 125 bits
  Distance between Person 24 and Person 55: 118 bits
  Distance between Person 24 and Person 56: 132 bits
  Distance between Person 24 and Person 57: 126 bits
  Distance between Person 24 and Person 58: 123 bits
  Distance between Person 24 and Person 59: 124 bits
  Distance between Person 24 and Person 60: 139 bits
  Distance between Person 24 and Person 61: 117 bits
  Distance between Person 24 and Person 62: 128 bits
  Distance between Person 24 and Person 63: 124 bits
  Distance between Person 24 and Person 64: 123 bits
  Distance between Person 24 and Person 65: 125 bits
  Distance between Person 24 and Person 66: 108 bits
  Distance between Person 24 and Person 67: 119 bits
  Distance between Person 24 and Person 68: 139 bits
  Distance between Person 24 and Person 69: 133 bits
  Distance between Person 24 and Person 70: 145 bits
  Distance between Person 24 and Person 71: 137 bits
  Distance between Person 24 and Person 72: 117 bits
  Distance between Person 24 and Person 73: 117 bits
  Distance between Person 24 and Person 74: 134 bits
  Distance between Person 24 and Person 75: 125 bits
  Distance between Person 24 and Person 76: 124 bits
  Distance between Person 24 and Person 77: 131 bits
  Distance between Person 24 and Person 78: 128 bits
  Distance between Person 24 and Person 79: 127 bits
  Distance between Person 24 and Person 80: 127 bits
  Distance between Person 24 and Person 81: 148 bits
  Distance between Person 24 and Person 82: 132 bits
  Distance between Person 24 and Person 83: 135 bits
  Distance between Person 24 and Person 84: 125 bits
  Distance between Person 24 and Person 85: 131 bits
  Distance between Person 24 and Person 86: 131 bits
  Distance between Person 24 and Person 87: 113 bits
  Distance between Person 24 and Person 88: 134 bits
  Distance between Person 24 and Person 89: 127 bits
  Distance between Person 25 and Person 26: 138 bits
  Distance between Person 25 and Person 27: 122 bits
  Distance between Person 25 and Person 28: 129 bits
  Distance between Person 25 and Person 29: 146 bits
  Distance between Person 25 and Person 30: 130 bits
  Distance between Person 25 and Person 31: 126 bits
  Distance between Person 25 and Person 32: 128 bits
  Distance between Person 25 and Person 33: 114 bits
  Distance between Person 25 and Person 34: 127 bits
  Distance between Person 25 and Person 35: 115 bits
  Distance between Person 25 and Person 36: 132 bits
  Distance between Person 25 and Person 37: 141 bits
  Distance between Person 25 and Person 38: 130 bits
  Distance between Person 25 and Person 39: 128 bits
  Distance between Person 25 and Person 40: 145 bits
  Distance between Person 25 and Person 41: 127 bits
  Distance between Person 25 and Person 42: 121 bits
  Distance between Person 25 and Person 43: 120 bits
  Distance between Person 25 and Person 44: 115 bits
  Distance between Person 25 and Person 45: 144 bits
  Distance between Person 25 and Person 46: 122 bits
  Distance between Person 25 and Person 47: 124 bits
  Distance between Person 25 and Person 48: 141 bits
  Distance between Person 25 and Person 49: 129 bits
  Distance between Person 25 and Person 50: 136 bits
  Distance between Person 25 and Person 51: 130 bits
  Distance between Person 25 and Person 52: 132 bits
  Distance between Person 25 and Person 53: 122 bits
  Distance between Person 25 and Person 54: 126 bits
  Distance between Person 25 and Person 55: 141 bits
  Distance between Person 25 and Person 56: 113 bits
  Distance between Person 25 and Person 57: 131 bits
  Distance between Person 25 and Person 58: 128 bits
  Distance between Person 25 and Person 59: 121 bits
  Distance between Person 25 and Person 60: 132 bits
  Distance between Person 25 and Person 61: 146 bits
  Distance between Person 25 and Person 62: 127 bits
  Distance between Person 25 and Person 63: 125 bits
  Distance between Person 25 and Person 64: 120 bits
  Distance between Person 25 and Person 65: 128 bits
  Distance between Person 25 and Person 66: 131 bits
  Distance between Person 25 and Person 67: 114 bits
  Distance between Person 25 and Person 68: 146 bits
  Distance between Person 25 and Person 69: 134 bits
  Distance between Person 25 and Person 70: 128 bits
  Distance between Person 25 and Person 71: 126 bits
  Distance between Person 25 and Person 72: 120 bits
  Distance between Person 25 and Person 73: 126 bits
  Distance between Person 25 and Person 74: 137 bits
  Distance between Person 25 and Person 75: 126 bits
  Distance between Person 25 and Person 76: 129 bits
  Distance between Person 25 and Person 77: 116 bits
  Distance between Person 25 and Person 78: 129 bits
  Distance between Person 25 and Person 79: 128 bits
  Distance between Person 25 and Person 80: 126 bits
  Distance between Person 25 and Person 81: 121 bits
  Distance between Person 25 and Person 82: 125 bits
  Distance between Person 25 and Person 83: 138 bits
  Distance between Person 25 and Person 84: 130 bits
  Distance between Person 25 and Person 85: 120 bits
  Distance between Person 25 and Person 86: 124 bits
  Distance between Person 25 and Person 87: 130 bits
  Distance between Person 25 and Person 88: 131 bits
  Distance between Person 25 and Person 89: 134 bits
  Distance between Person 26 and Person 27: 118 bits
  Distance between Person 26 and Person 28: 129 bits
  Distance between Person 26 and Person 29: 136 bits
  Distance between Person 26 and Person 30: 136 bits
  Distance between Person 26 and Person 31: 122 bits
  Distance between Person 26 and Person 32: 128 bits
  Distance between Person 26 and Person 33: 134 bits
  Distance between Person 26 and Person 34: 113 bits
  Distance between Person 26 and Person 35: 135 bits
  Distance between Person 26 and Person 36: 106 bits
  Distance between Person 26 and Person 37: 125 bits
  Distance between Person 26 and Person 38: 126 bits
  Distance between Person 26 and Person 39: 126 bits
  Distance between Person 26 and Person 40: 117 bits
  Distance between Person 26 and Person 41: 115 bits
  Distance between Person 26 and Person 42: 129 bits
  Distance between Person 26 and Person 43: 88 bits
  Distance between Person 26 and Person 44: 125 bits
  Distance between Person 26 and Person 45: 134 bits
  Distance between Person 26 and Person 46: 136 bits
  Distance between Person 26 and Person 47: 116 bits
  Distance between Person 26 and Person 48: 133 bits
  Distance between Person 26 and Person 49: 131 bits
  Distance between Person 26 and Person 50: 128 bits
  Distance between Person 26 and Person 51: 110 bits
  Distance between Person 26 and Person 52: 122 bits
  Distance between Person 26 and Person 53: 126 bits
  Distance between Person 26 and Person 54: 114 bits
  Distance between Person 26 and Person 55: 119 bits
  Distance between Person 26 and Person 56: 117 bits
  Distance between Person 26 and Person 57: 131 bits
  Distance between Person 26 and Person 58: 136 bits
  Distance between Person 26 and Person 59: 125 bits
  Distance between Person 26 and Person 60: 138 bits
  Distance between Person 26 and Person 61: 130 bits
  Distance between Person 26 and Person 62: 121 bits
  Distance between Person 26 and Person 63: 141 bits
  Distance between Person 26 and Person 64: 118 bits
  Distance between Person 26 and Person 65: 130 bits
  Distance between Person 26 and Person 66: 139 bits
  Distance between Person 26 and Person 67: 126 bits
  Distance between Person 26 and Person 68: 134 bits
  Distance between Person 26 and Person 69: 118 bits
  Distance between Person 26 and Person 70: 124 bits
  Distance between Person 26 and Person 71: 128 bits
  Distance between Person 26 and Person 72: 132 bits
  Distance between Person 26 and Person 73: 144 bits
  Distance between Person 26 and Person 74: 135 bits
  Distance between Person 26 and Person 75: 134 bits
  Distance between Person 26 and Person 76: 131 bits
  Distance between Person 26 and Person 77: 120 bits
  Distance between Person 26 and Person 78: 127 bits
  Distance between Person 26 and Person 79: 122 bits
  Distance between Person 26 and Person 80: 122 bits
  Distance between Person 26 and Person 81: 143 bits
  Distance between Person 26 and Person 82: 115 bits
  Distance between Person 26 and Person 83: 148 bits
  Distance between Person 26 and Person 84: 130 bits
  Distance between Person 26 and Person 85: 132 bits
  Distance between Person 26 and Person 86: 140 bits
  Distance between Person 26 and Person 87: 130 bits
  Distance between Person 26 and Person 88: 117 bits
  Distance between Person 26 and Person 89: 124 bits
  Distance between Person 27 and Person 28: 137 bits
  Distance between Person 27 and Person 29: 122 bits
  Distance between Person 27 and Person 30: 128 bits
  Distance between Person 27 and Person 31: 138 bits
  Distance between Person 27 and Person 32: 118 bits
  Distance between Person 27 and Person 33: 114 bits
  Distance between Person 27 and Person 34: 123 bits
  Distance between Person 27 and Person 35: 123 bits
  Distance between Person 27 and Person 36: 122 bits
  Distance between Person 27 and Person 37: 125 bits
  Distance between Person 27 and Person 38: 122 bits
  Distance between Person 27 and Person 39: 112 bits
  Distance between Person 27 and Person 40: 119 bits
  Distance between Person 27 and Person 41: 135 bits
  Distance between Person 27 and Person 42: 135 bits
  Distance between Person 27 and Person 43: 120 bits
  Distance between Person 27 and Person 44: 143 bits
  Distance between Person 27 and Person 45: 128 bits
  Distance between Person 27 and Person 46: 126 bits
  Distance between Person 27 and Person 47: 142 bits
  Distance between Person 27 and Person 48: 127 bits
  Distance between Person 27 and Person 49: 137 bits
  Distance between Person 27 and Person 50: 130 bits
  Distance between Person 27 and Person 51: 130 bits
  Distance between Person 27 and Person 52: 128 bits
  Distance between Person 27 and Person 53: 134 bits
  Distance between Person 27 and Person 54: 134 bits
  Distance between Person 27 and Person 55: 125 bits
  Distance between Person 27 and Person 56: 129 bits
  Distance between Person 27 and Person 57: 141 bits
  Distance between Person 27 and Person 58: 132 bits
  Distance between Person 27 and Person 59: 115 bits
  Distance between Person 27 and Person 60: 130 bits
  Distance between Person 27 and Person 61: 138 bits
  Distance between Person 27 and Person 62: 115 bits
  Distance between Person 27 and Person 63: 123 bits
  Distance between Person 27 and Person 64: 134 bits
  Distance between Person 27 and Person 65: 122 bits
  Distance between Person 27 and Person 66: 133 bits
  Distance between Person 27 and Person 67: 120 bits
  Distance between Person 27 and Person 68: 120 bits
  Distance between Person 27 and Person 69: 124 bits
  Distance between Person 27 and Person 70: 136 bits
  Distance between Person 27 and Person 71: 132 bits
  Distance between Person 27 and Person 72: 108 bits
  Distance between Person 27 and Person 73: 132 bits
  Distance between Person 27 and Person 74: 135 bits
  Distance between Person 27 and Person 75: 120 bits
  Distance between Person 27 and Person 76: 137 bits
  Distance between Person 27 and Person 77: 132 bits
  Distance between Person 27 and Person 78: 111 bits
  Distance between Person 27 and Person 79: 130 bits
  Distance between Person 27 and Person 80: 140 bits
  Distance between Person 27 and Person 81: 109 bits
  Distance between Person 27 and Person 82: 127 bits
  Distance between Person 27 and Person 83: 126 bits
  Distance between Person 27 and Person 84: 122 bits
  Distance between Person 27 and Person 85: 126 bits
  Distance between Person 27 and Person 86: 124 bits
  Distance between Person 27 and Person 87: 136 bits
  Distance between Person 27 and Person 88: 125 bits
  Distance between Person 27 and Person 89: 128 bits
  Distance between Person 28 and Person 29: 133 bits
  Distance between Person 28 and Person 30: 149 bits
  Distance between Person 28 and Person 31: 121 bits
  Distance between Person 28 and Person 32: 127 bits
  Distance between Person 28 and Person 33: 135 bits
  Distance between Person 28 and Person 34: 126 bits
  Distance between Person 28 and Person 35: 138 bits
  Distance between Person 28 and Person 36: 129 bits
  Distance between Person 28 and Person 37: 108 bits
  Distance between Person 28 and Person 38: 127 bits
  Distance between Person 28 and Person 39: 123 bits
  Distance between Person 28 and Person 40: 126 bits
  Distance between Person 28 and Person 41: 138 bits
  Distance between Person 28 and Person 42: 130 bits
  Distance between Person 28 and Person 43: 131 bits
  Distance between Person 28 and Person 44: 126 bits
  Distance between Person 28 and Person 45: 129 bits
  Distance between Person 28 and Person 46: 125 bits
  Distance between Person 28 and Person 47: 127 bits
  Distance between Person 28 and Person 48: 130 bits
  Distance between Person 28 and Person 49: 124 bits
  Distance between Person 28 and Person 50: 125 bits
  Distance between Person 28 and Person 51: 119 bits
  Distance between Person 28 and Person 52: 125 bits
  Distance between Person 28 and Person 53: 123 bits
  Distance between Person 28 and Person 54: 115 bits
  Distance between Person 28 and Person 55: 134 bits
  Distance between Person 28 and Person 56: 128 bits
  Distance between Person 28 and Person 57: 140 bits
  Distance between Person 28 and Person 58: 135 bits
  Distance between Person 28 and Person 59: 124 bits
  Distance between Person 28 and Person 60: 127 bits
  Distance between Person 28 and Person 61: 125 bits
  Distance between Person 28 and Person 62: 114 bits
  Distance between Person 28 and Person 63: 126 bits
  Distance between Person 28 and Person 64: 121 bits
  Distance between Person 28 and Person 65: 137 bits
  Distance between Person 28 and Person 66: 138 bits
  Distance between Person 28 and Person 67: 129 bits
  Distance between Person 28 and Person 68: 113 bits
  Distance between Person 28 and Person 69: 137 bits
  Distance between Person 28 and Person 70: 119 bits
  Distance between Person 28 and Person 71: 133 bits
  Distance between Person 28 and Person 72: 121 bits
  Distance between Person 28 and Person 73: 121 bits
  Distance between Person 28 and Person 74: 132 bits
  Distance between Person 28 and Person 75: 137 bits
  Distance between Person 28 and Person 76: 122 bits
  Distance between Person 28 and Person 77: 127 bits
  Distance between Person 28 and Person 78: 114 bits
  Distance between Person 28 and Person 79: 133 bits
  Distance between Person 28 and Person 80: 127 bits
  Distance between Person 28 and Person 81: 140 bits
  Distance between Person 28 and Person 82: 124 bits
  Distance between Person 28 and Person 83: 135 bits
  Distance between Person 28 and Person 84: 131 bits
  Distance between Person 28 and Person 85: 127 bits
  Distance between Person 28 and Person 86: 119 bits
  Distance between Person 28 and Person 87: 119 bits
  Distance between Person 28 and Person 88: 120 bits
  Distance between Person 28 and Person 89: 119 bits
  Distance between Person 29 and Person 30: 120 bits
  Distance between Person 29 and Person 31: 132 bits
  Distance between Person 29 and Person 32: 128 bits
  Distance between Person 29 and Person 33: 130 bits
  Distance between Person 29 and Person 34: 127 bits
  Distance between Person 29 and Person 35: 125 bits
  Distance between Person 29 and Person 36: 134 bits
  Distance between Person 29 and Person 37: 81 bits
  Distance between Person 29 and Person 38: 142 bits
  Distance between Person 29 and Person 39: 126 bits
  Distance between Person 29 and Person 40: 123 bits
  Distance between Person 29 and Person 41: 119 bits
  Distance between Person 29 and Person 42: 129 bits
  Distance between Person 29 and Person 43: 122 bits
  Distance between Person 29 and Person 44: 131 bits
  Distance between Person 29 and Person 45: 122 bits
  Distance between Person 29 and Person 46: 124 bits
  Distance between Person 29 and Person 47: 138 bits
  Distance between Person 29 and Person 48: 131 bits
  Distance between Person 29 and Person 49: 133 bits
  Distance between Person 29 and Person 50: 106 bits
  Distance between Person 29 and Person 51: 130 bits
  Distance between Person 29 and Person 52: 126 bits
  Distance between Person 29 and Person 53: 122 bits
  Distance between Person 29 and Person 54: 124 bits
  Distance between Person 29 and Person 55: 119 bits
  Distance between Person 29 and Person 56: 133 bits
  Distance between Person 29 and Person 57: 137 bits
  Distance between Person 29 and Person 58: 122 bits
  Distance between Person 29 and Person 59: 125 bits
  Distance between Person 29 and Person 60: 122 bits
  Distance between Person 29 and Person 61: 112 bits
  Distance between Person 29 and Person 62: 145 bits
  Distance between Person 29 and Person 63: 125 bits
  Distance between Person 29 and Person 64: 138 bits
  Distance between Person 29 and Person 65: 128 bits
  Distance between Person 29 and Person 66: 125 bits
  Distance between Person 29 and Person 67: 126 bits
  Distance between Person 29 and Person 68: 132 bits
  Distance between Person 29 and Person 69: 124 bits
  Distance between Person 29 and Person 70: 134 bits
  Distance between Person 29 and Person 71: 128 bits
  Distance between Person 29 and Person 72: 140 bits
  Distance between Person 29 and Person 73: 128 bits
  Distance between Person 29 and Person 74: 125 bits
  Distance between Person 29 and Person 75: 126 bits
  Distance between Person 29 and Person 76: 129 bits
  Distance between Person 29 and Person 77: 114 bits
  Distance between Person 29 and Person 78: 137 bits
  Distance between Person 29 and Person 79: 120 bits
  Distance between Person 29 and Person 80: 136 bits
  Distance between Person 29 and Person 81: 131 bits
  Distance between Person 29 and Person 82: 113 bits
  Distance between Person 29 and Person 83: 130 bits
  Distance between Person 29 and Person 84: 116 bits
  Distance between Person 29 and Person 85: 130 bits
  Distance between Person 29 and Person 86: 136 bits
  Distance between Person 29 and Person 87: 132 bits
  Distance between Person 29 and Person 88: 117 bits
  Distance between Person 29 and Person 89: 118 bits
  Distance between Person 30 and Person 31: 140 bits
  Distance between Person 30 and Person 32: 134 bits
  Distance between Person 30 and Person 33: 136 bits
  Distance between Person 30 and Person 34: 143 bits
  Distance between Person 30 and Person 35: 115 bits
  Distance between Person 30 and Person 36: 120 bits
  Distance between Person 30 and Person 37: 137 bits
  Distance between Person 30 and Person 38: 136 bits
  Distance between Person 30 and Person 39: 124 bits
  Distance between Person 30 and Person 40: 137 bits
  Distance between Person 30 and Person 41: 149 bits
  Distance between Person 30 and Person 42: 119 bits
  Distance between Person 30 and Person 43: 120 bits
  Distance between Person 30 and Person 44: 153 bits
  Distance between Person 30 and Person 45: 148 bits
  Distance between Person 30 and Person 46: 126 bits
  Distance between Person 30 and Person 47: 126 bits
  Distance between Person 30 and Person 48: 123 bits
  Distance between Person 30 and Person 49: 111 bits
  Distance between Person 30 and Person 50: 122 bits
  Distance between Person 30 and Person 51: 124 bits
  Distance between Person 30 and Person 52: 138 bits
  Distance between Person 30 and Person 53: 128 bits
  Distance between Person 30 and Person 54: 146 bits
  Distance between Person 30 and Person 55: 129 bits
  Distance between Person 30 and Person 56: 131 bits
  Distance between Person 30 and Person 57: 125 bits
  Distance between Person 30 and Person 58: 130 bits
  Distance between Person 30 and Person 59: 125 bits
  Distance between Person 30 and Person 60: 146 bits
  Distance between Person 30 and Person 61: 118 bits
  Distance between Person 30 and Person 62: 137 bits
  Distance between Person 30 and Person 63: 131 bits
  Distance between Person 30 and Person 64: 136 bits
  Distance between Person 30 and Person 65: 134 bits
  Distance between Person 30 and Person 66: 137 bits
  Distance between Person 30 and Person 67: 132 bits
  Distance between Person 30 and Person 68: 126 bits
  Distance between Person 30 and Person 69: 128 bits
  Distance between Person 30 and Person 70: 134 bits
  Distance between Person 30 and Person 71: 132 bits
  Distance between Person 30 and Person 72: 126 bits
  Distance between Person 30 and Person 73: 150 bits
  Distance between Person 30 and Person 74: 129 bits
  Distance between Person 30 and Person 75: 144 bits
  Distance between Person 30 and Person 76: 139 bits
  Distance between Person 30 and Person 77: 140 bits
  Distance between Person 30 and Person 78: 149 bits
  Distance between Person 30 and Person 79: 120 bits
  Distance between Person 30 and Person 80: 150 bits
  Distance between Person 30 and Person 81: 111 bits
  Distance between Person 30 and Person 82: 133 bits
  Distance between Person 30 and Person 83: 122 bits
  Distance between Person 30 and Person 84: 128 bits
  Distance between Person 30 and Person 85: 122 bits
  Distance between Person 30 and Person 86: 132 bits
  Distance between Person 30 and Person 87: 120 bits
  Distance between Person 30 and Person 88: 141 bits
  Distance between Person 30 and Person 89: 126 bits
  Distance between Person 31 and Person 32: 136 bits
  Distance between Person 31 and Person 33: 124 bits
  Distance between Person 31 and Person 34: 123 bits
  Distance between Person 31 and Person 35: 139 bits
  Distance between Person 31 and Person 36: 128 bits
  Distance between Person 31 and Person 37: 133 bits
  Distance between Person 31 and Person 38: 126 bits
  Distance between Person 31 and Person 39: 142 bits
  Distance between Person 31 and Person 40: 133 bits
  Distance between Person 31 and Person 41: 137 bits
  Distance between Person 31 and Person 42: 133 bits
  Distance between Person 31 and Person 43: 142 bits
  Distance between Person 31 and Person 44: 105 bits
  Distance between Person 31 and Person 45: 120 bits
  Distance between Person 31 and Person 46: 126 bits
  Distance between Person 31 and Person 47: 122 bits
  Distance between Person 31 and Person 48: 125 bits
  Distance between Person 31 and Person 49: 121 bits
  Distance between Person 31 and Person 50: 118 bits
  Distance between Person 31 and Person 51: 126 bits
  Distance between Person 31 and Person 52: 134 bits
  Distance between Person 31 and Person 53: 124 bits
  Distance between Person 31 and Person 54: 124 bits
  Distance between Person 31 and Person 55: 121 bits
  Distance between Person 31 and Person 56: 117 bits
  Distance between Person 31 and Person 57: 133 bits
  Distance between Person 31 and Person 58: 118 bits
  Distance between Person 31 and Person 59: 143 bits
  Distance between Person 31 and Person 60: 128 bits
  Distance between Person 31 and Person 61: 130 bits
  Distance between Person 31 and Person 62: 119 bits
  Distance between Person 31 and Person 63: 135 bits
  Distance between Person 31 and Person 64: 132 bits
  Distance between Person 31 and Person 65: 116 bits
  Distance between Person 31 and Person 66: 145 bits
  Distance between Person 31 and Person 67: 150 bits
  Distance between Person 31 and Person 68: 132 bits
  Distance between Person 31 and Person 69: 116 bits
  Distance between Person 31 and Person 70: 126 bits
  Distance between Person 31 and Person 71: 120 bits
  Distance between Person 31 and Person 72: 120 bits
  Distance between Person 31 and Person 73: 106 bits
  Distance between Person 31 and Person 74: 125 bits
  Distance between Person 31 and Person 75: 138 bits
  Distance between Person 31 and Person 76: 123 bits
  Distance between Person 31 and Person 77: 134 bits
  Distance between Person 31 and Person 78: 111 bits
  Distance between Person 31 and Person 79: 132 bits
  Distance between Person 31 and Person 80: 116 bits
  Distance between Person 31 and Person 81: 135 bits
  Distance between Person 31 and Person 82: 121 bits
  Distance between Person 31 and Person 83: 140 bits
  Distance between Person 31 and Person 84: 128 bits
  Distance between Person 31 and Person 85: 134 bits
  Distance between Person 31 and Person 86: 136 bits
  Distance between Person 31 and Person 87: 118 bits
  Distance between Person 31 and Person 88: 109 bits
  Distance between Person 31 and Person 89: 112 bits
  Distance between Person 32 and Person 33: 130 bits
  Distance between Person 32 and Person 34: 127 bits
  Distance between Person 32 and Person 35: 139 bits
  Distance between Person 32 and Person 36: 124 bits
  Distance between Person 32 and Person 37: 125 bits
  Distance between Person 32 and Person 38: 134 bits
  Distance between Person 32 and Person 39: 114 bits
  Distance between Person 32 and Person 40: 115 bits
  Distance between Person 32 and Person 41: 149 bits
  Distance between Person 32 and Person 42: 137 bits
  Distance between Person 32 and Person 43: 110 bits
  Distance between Person 32 and Person 44: 139 bits
  Distance between Person 32 and Person 45: 122 bits
  Distance between Person 32 and Person 46: 132 bits
  Distance between Person 32 and Person 47: 120 bits
  Distance between Person 32 and Person 48: 125 bits
  Distance between Person 32 and Person 49: 139 bits
  Distance between Person 32 and Person 50: 134 bits
  Distance between Person 32 and Person 51: 142 bits
  Distance between Person 32 and Person 52: 134 bits
  Distance between Person 32 and Person 53: 132 bits
  Distance between Person 32 and Person 54: 138 bits
  Distance between Person 32 and Person 55: 155 bits
  Distance between Person 32 and Person 56: 131 bits
  Distance between Person 32 and Person 57: 127 bits
  Distance between Person 32 and Person 58: 128 bits
  Distance between Person 32 and Person 59: 129 bits
  Distance between Person 32 and Person 60: 126 bits
  Distance between Person 32 and Person 61: 132 bits
  Distance between Person 32 and Person 62: 107 bits
  Distance between Person 32 and Person 63: 135 bits
  Distance between Person 32 and Person 64: 124 bits
  Distance between Person 32 and Person 65: 130 bits
  Distance between Person 32 and Person 66: 117 bits
  Distance between Person 32 and Person 67: 144 bits
  Distance between Person 32 and Person 68: 134 bits
  Distance between Person 32 and Person 69: 142 bits
  Distance between Person 32 and Person 70: 140 bits
  Distance between Person 32 and Person 71: 114 bits
  Distance between Person 32 and Person 72: 118 bits
  Distance between Person 32 and Person 73: 130 bits
  Distance between Person 32 and Person 74: 127 bits
  Distance between Person 32 and Person 75: 122 bits
  Distance between Person 32 and Person 76: 119 bits
  Distance between Person 32 and Person 77: 138 bits
  Distance between Person 32 and Person 78: 119 bits
  Distance between Person 32 and Person 79: 134 bits
  Distance between Person 32 and Person 80: 126 bits
  Distance between Person 32 and Person 81: 137 bits
  Distance between Person 32 and Person 82: 123 bits
  Distance between Person 32 and Person 83: 134 bits
  Distance between Person 32 and Person 84: 104 bits
  Distance between Person 32 and Person 85: 130 bits
  Distance between Person 32 and Person 86: 136 bits
  Distance between Person 32 and Person 87: 128 bits
  Distance between Person 32 and Person 88: 129 bits
  Distance between Person 32 and Person 89: 138 bits
  Distance between Person 33 and Person 34: 125 bits
  Distance between Person 33 and Person 35: 135 bits
  Distance between Person 33 and Person 36: 138 bits
  Distance between Person 33 and Person 37: 131 bits
  Distance between Person 33 and Person 38: 140 bits
  Distance between Person 33 and Person 39: 136 bits
  Distance between Person 33 and Person 40: 129 bits
  Distance between Person 33 and Person 41: 131 bits
  Distance between Person 33 and Person 42: 137 bits
  Distance between Person 33 and Person 43: 126 bits
  Distance between Person 33 and Person 44: 107 bits
  Distance between Person 33 and Person 45: 120 bits
  Distance between Person 33 and Person 46: 122 bits
  Distance between Person 33 and Person 47: 132 bits
  Distance between Person 33 and Person 48: 139 bits
  Distance between Person 33 and Person 49: 137 bits
  Distance between Person 33 and Person 50: 138 bits
  Distance between Person 33 and Person 51: 132 bits
  Distance between Person 33 and Person 52: 116 bits
  Distance between Person 33 and Person 53: 114 bits
  Distance between Person 33 and Person 54: 132 bits
  Distance between Person 33 and Person 55: 129 bits
  Distance between Person 33 and Person 56: 149 bits
  Distance between Person 33 and Person 57: 127 bits
  Distance between Person 33 and Person 58: 132 bits
  Distance between Person 33 and Person 59: 123 bits
  Distance between Person 33 and Person 60: 132 bits
  Distance between Person 33 and Person 61: 140 bits
  Distance between Person 33 and Person 62: 137 bits
  Distance between Person 33 and Person 63: 127 bits
  Distance between Person 33 and Person 64: 126 bits
  Distance between Person 33 and Person 65: 116 bits
  Distance between Person 33 and Person 66: 113 bits
  Distance between Person 33 and Person 67: 74 bits
  Distance between Person 33 and Person 68: 146 bits
  Distance between Person 33 and Person 69: 114 bits
  Distance between Person 33 and Person 70: 134 bits
  Distance between Person 33 and Person 71: 122 bits
  Distance between Person 33 and Person 72: 134 bits
  Distance between Person 33 and Person 73: 124 bits
  Distance between Person 33 and Person 74: 139 bits
  Distance between Person 33 and Person 75: 134 bits
  Distance between Person 33 and Person 76: 113 bits
  Distance between Person 33 and Person 77: 136 bits
  Distance between Person 33 and Person 78: 119 bits
  Distance between Person 33 and Person 79: 146 bits
  Distance between Person 33 and Person 80: 126 bits
  Distance between Person 33 and Person 81: 107 bits
  Distance between Person 33 and Person 82: 121 bits
  Distance between Person 33 and Person 83: 130 bits
  Distance between Person 33 and Person 84: 138 bits
  Distance between Person 33 and Person 85: 124 bits
  Distance between Person 33 and Person 86: 136 bits
  Distance between Person 33 and Person 87: 140 bits
  Distance between Person 33 and Person 88: 117 bits
  Distance between Person 33 and Person 89: 136 bits
  Distance between Person 34 and Person 35: 136 bits
  Distance between Person 34 and Person 36: 139 bits
  Distance between Person 34 and Person 37: 112 bits
  Distance between Person 34 and Person 38: 133 bits
  Distance between Person 34 and Person 39: 113 bits
  Distance between Person 34 and Person 40: 112 bits
  Distance between Person 34 and Person 41: 124 bits
  Distance between Person 34 and Person 42: 128 bits
  Distance between Person 34 and Person 43: 111 bits
  Distance between Person 34 and Person 44: 130 bits
  Distance between Person 34 and Person 45: 123 bits
  Distance between Person 34 and Person 46: 117 bits
  Distance between Person 34 and Person 47: 125 bits
  Distance between Person 34 and Person 48: 118 bits
  Distance between Person 34 and Person 49: 126 bits
  Distance between Person 34 and Person 50: 125 bits
  Distance between Person 34 and Person 51: 133 bits
  Distance between Person 34 and Person 52: 123 bits
  Distance between Person 34 and Person 53: 119 bits
  Distance between Person 34 and Person 54: 133 bits
  Distance between Person 34 and Person 55: 148 bits
  Distance between Person 34 and Person 56: 128 bits
  Distance between Person 34 and Person 57: 130 bits
  Distance between Person 34 and Person 58: 127 bits
  Distance between Person 34 and Person 59: 126 bits
  Distance between Person 34 and Person 60: 127 bits
  Distance between Person 34 and Person 61: 141 bits
  Distance between Person 34 and Person 62: 128 bits
  Distance between Person 34 and Person 63: 126 bits
  Distance between Person 34 and Person 64: 129 bits
  Distance between Person 34 and Person 65: 131 bits
  Distance between Person 34 and Person 66: 134 bits
  Distance between Person 34 and Person 67: 119 bits
  Distance between Person 34 and Person 68: 135 bits
  Distance between Person 34 and Person 69: 129 bits
  Distance between Person 34 and Person 70: 131 bits
  Distance between Person 34 and Person 71: 129 bits
  Distance between Person 34 and Person 72: 133 bits
  Distance between Person 34 and Person 73: 115 bits
  Distance between Person 34 and Person 74: 108 bits
  Distance between Person 34 and Person 75: 135 bits
  Distance between Person 34 and Person 76: 96 bits
  Distance between Person 34 and Person 77: 131 bits
  Distance between Person 34 and Person 78: 136 bits
  Distance between Person 34 and Person 79: 135 bits
  Distance between Person 34 and Person 80: 125 bits
  Distance between Person 34 and Person 81: 126 bits
  Distance between Person 34 and Person 82: 124 bits
  Distance between Person 34 and Person 83: 147 bits
  Distance between Person 34 and Person 84: 121 bits
  Distance between Person 34 and Person 85: 133 bits
  Distance between Person 34 and Person 86: 121 bits
  Distance between Person 34 and Person 87: 137 bits
  Distance between Person 34 and Person 88: 126 bits
  Distance between Person 34 and Person 89: 125 bits
  Distance between Person 35 and Person 36: 127 bits
  Distance between Person 35 and Person 37: 118 bits
  Distance between Person 35 and Person 38: 123 bits
  Distance between Person 35 and Person 39: 133 bits
  Distance between Person 35 and Person 40: 130 bits
  Distance between Person 35 and Person 41: 142 bits
  Distance between Person 35 and Person 42: 130 bits
  Distance between Person 35 and Person 43: 135 bits
  Distance between Person 35 and Person 44: 134 bits
  Distance between Person 35 and Person 45: 139 bits
  Distance between Person 35 and Person 46: 143 bits
  Distance between Person 35 and Person 47: 127 bits
  Distance between Person 35 and Person 48: 124 bits
  Distance between Person 35 and Person 49: 120 bits
  Distance between Person 35 and Person 50: 113 bits
  Distance between Person 35 and Person 51: 147 bits
  Distance between Person 35 and Person 52: 125 bits
  Distance between Person 35 and Person 53: 117 bits
  Distance between Person 35 and Person 54: 127 bits
  Distance between Person 35 and Person 55: 136 bits
  Distance between Person 35 and Person 56: 130 bits
  Distance between Person 35 and Person 57: 128 bits
  Distance between Person 35 and Person 58: 147 bits
  Distance between Person 35 and Person 59: 124 bits
  Distance between Person 35 and Person 60: 123 bits
  Distance between Person 35 and Person 61: 123 bits
  Distance between Person 35 and Person 62: 134 bits
  Distance between Person 35 and Person 63: 130 bits
  Distance between Person 35 and Person 64: 121 bits
  Distance between Person 35 and Person 65: 127 bits
  Distance between Person 35 and Person 66: 130 bits
  Distance between Person 35 and Person 67: 135 bits
  Distance between Person 35 and Person 68: 133 bits
  Distance between Person 35 and Person 69: 135 bits
  Distance between Person 35 and Person 70: 129 bits
  Distance between Person 35 and Person 71: 125 bits
  Distance between Person 35 and Person 72: 127 bits
  Distance between Person 35 and Person 73: 139 bits
  Distance between Person 35 and Person 74: 134 bits
  Distance between Person 35 and Person 75: 133 bits
  Distance between Person 35 and Person 76: 120 bits
  Distance between Person 35 and Person 77: 111 bits
  Distance between Person 35 and Person 78: 126 bits
  Distance between Person 35 and Person 79: 121 bits
  Distance between Person 35 and Person 80: 137 bits
  Distance between Person 35 and Person 81: 124 bits
  Distance between Person 35 and Person 82: 138 bits
  Distance between Person 35 and Person 83: 133 bits
  Distance between Person 35 and Person 84: 129 bits
  Distance between Person 35 and Person 85: 119 bits
  Distance between Person 35 and Person 86: 121 bits
  Distance between Person 35 and Person 87: 123 bits
  Distance between Person 35 and Person 88: 114 bits
  Distance between Person 35 and Person 89: 133 bits
  Distance between Person 36 and Person 37: 119 bits
  Distance between Person 36 and Person 38: 122 bits
  Distance between Person 36 and Person 39: 114 bits
  Distance between Person 36 and Person 40: 127 bits
  Distance between Person 36 and Person 41: 143 bits
  Distance between Person 36 and Person 42: 133 bits
  Distance between Person 36 and Person 43: 132 bits
  Distance between Person 36 and Person 44: 135 bits
  Distance between Person 36 and Person 45: 144 bits
  Distance between Person 36 and Person 46: 120 bits
  Distance between Person 36 and Person 47: 126 bits
  Distance between Person 36 and Person 48: 133 bits
  Distance between Person 36 and Person 49: 133 bits
  Distance between Person 36 and Person 50: 112 bits
  Distance between Person 36 and Person 51: 126 bits
  Distance between Person 36 and Person 52: 126 bits
  Distance between Person 36 and Person 53: 132 bits
  Distance between Person 36 and Person 54: 114 bits
  Distance between Person 36 and Person 55: 121 bits
  Distance between Person 36 and Person 56: 115 bits
  Distance between Person 36 and Person 57: 131 bits
  Distance between Person 36 and Person 58: 132 bits
  Distance between Person 36 and Person 59: 121 bits
  Distance between Person 36 and Person 60: 128 bits
  Distance between Person 36 and Person 61: 136 bits
  Distance between Person 36 and Person 62: 133 bits
  Distance between Person 36 and Person 63: 135 bits
  Distance between Person 36 and Person 64: 134 bits
  Distance between Person 36 and Person 65: 114 bits
  Distance between Person 36 and Person 66: 141 bits
  Distance between Person 36 and Person 67: 144 bits
  Distance between Person 36 and Person 68: 126 bits
  Distance between Person 36 and Person 69: 142 bits
  Distance between Person 36 and Person 70: 132 bits
  Distance between Person 36 and Person 71: 130 bits
  Distance between Person 36 and Person 72: 124 bits
  Distance between Person 36 and Person 73: 112 bits
  Distance between Person 36 and Person 74: 139 bits
  Distance between Person 36 and Person 75: 126 bits
  Distance between Person 36 and Person 76: 127 bits
  Distance between Person 36 and Person 77: 126 bits
  Distance between Person 36 and Person 78: 123 bits
  Distance between Person 36 and Person 79: 116 bits
  Distance between Person 36 and Person 80: 128 bits
  Distance between Person 36 and Person 81: 139 bits
  Distance between Person 36 and Person 82: 127 bits
  Distance between Person 36 and Person 83: 142 bits
  Distance between Person 36 and Person 84: 128 bits
  Distance between Person 36 and Person 85: 122 bits
  Distance between Person 36 and Person 86: 144 bits
  Distance between Person 36 and Person 87: 114 bits
  Distance between Person 36 and Person 88: 131 bits
  Distance between Person 36 and Person 89: 132 bits
  Distance between Person 37 and Person 38: 129 bits
  Distance between Person 37 and Person 39: 115 bits
  Distance between Person 37 and Person 40: 126 bits
  Distance between Person 37 and Person 41: 124 bits
  Distance between Person 37 and Person 42: 144 bits
  Distance between Person 37 and Person 43: 131 bits
  Distance between Person 37 and Person 44: 102 bits
  Distance between Person 37 and Person 45: 109 bits
  Distance between Person 37 and Person 46: 121 bits
  Distance between Person 37 and Person 47: 147 bits
  Distance between Person 37 and Person 48: 132 bits
  Distance between Person 37 and Person 49: 118 bits
  Distance between Person 37 and Person 50: 117 bits
  Distance between Person 37 and Person 51: 115 bits
  Distance between Person 37 and Person 52: 121 bits
  Distance between Person 37 and Person 53: 129 bits
  Distance between Person 37 and Person 54: 125 bits
  Distance between Person 37 and Person 55: 126 bits
  Distance between Person 37 and Person 56: 132 bits
  Distance between Person 37 and Person 57: 154 bits
  Distance between Person 37 and Person 58: 129 bits
  Distance between Person 37 and Person 59: 132 bits
  Distance between Person 37 and Person 60: 115 bits
  Distance between Person 37 and Person 61: 113 bits
  Distance between Person 37 and Person 62: 146 bits
  Distance between Person 37 and Person 63: 116 bits
  Distance between Person 37 and Person 64: 131 bits
  Distance between Person 37 and Person 65: 123 bits
  Distance between Person 37 and Person 66: 122 bits
  Distance between Person 37 and Person 67: 129 bits
  Distance between Person 37 and Person 68: 137 bits
  Distance between Person 37 and Person 69: 127 bits
  Distance between Person 37 and Person 70: 119 bits
  Distance between Person 37 and Person 71: 129 bits
  Distance between Person 37 and Person 72: 113 bits
  Distance between Person 37 and Person 73: 117 bits
  Distance between Person 37 and Person 74: 126 bits
  Distance between Person 37 and Person 75: 131 bits
  Distance between Person 37 and Person 76: 140 bits
  Distance between Person 37 and Person 77: 103 bits
  Distance between Person 37 and Person 78: 126 bits
  Distance between Person 37 and Person 79: 131 bits
  Distance between Person 37 and Person 80: 105 bits
  Distance between Person 37 and Person 81: 144 bits
  Distance between Person 37 and Person 82: 108 bits
  Distance between Person 37 and Person 83: 135 bits
  Distance between Person 37 and Person 84: 117 bits
  Distance between Person 37 and Person 85: 133 bits
  Distance between Person 37 and Person 86: 117 bits
  Distance between Person 37 and Person 87: 133 bits
  Distance between Person 37 and Person 88: 132 bits
  Distance between Person 37 and Person 89: 135 bits
  Distance between Person 38 and Person 39: 126 bits
  Distance between Person 38 and Person 40: 137 bits
  Distance between Person 38 and Person 41: 135 bits
  Distance between Person 38 and Person 42: 131 bits
  Distance between Person 38 and Person 43: 120 bits
  Distance between Person 38 and Person 44: 115 bits
  Distance between Person 38 and Person 45: 128 bits
  Distance between Person 38 and Person 46: 124 bits
  Distance between Person 38 and Person 47: 132 bits
  Distance between Person 38 and Person 48: 125 bits
  Distance between Person 38 and Person 49: 127 bits
  Distance between Person 38 and Person 50: 130 bits
  Distance between Person 38 and Person 51: 128 bits
  Distance between Person 38 and Person 52: 130 bits
  Distance between Person 38 and Person 53: 126 bits
  Distance between Person 38 and Person 54: 112 bits
  Distance between Person 38 and Person 55: 121 bits
  Distance between Person 38 and Person 56: 125 bits
  Distance between Person 38 and Person 57: 115 bits
  Distance between Person 38 and Person 58: 136 bits
  Distance between Person 38 and Person 59: 139 bits
  Distance between Person 38 and Person 60: 110 bits
  Distance between Person 38 and Person 61: 106 bits
  Distance between Person 38 and Person 62: 119 bits
  Distance between Person 38 and Person 63: 131 bits
  Distance between Person 38 and Person 64: 148 bits
  Distance between Person 38 and Person 65: 126 bits
  Distance between Person 38 and Person 66: 119 bits
  Distance between Person 38 and Person 67: 144 bits
  Distance between Person 38 and Person 68: 124 bits
  Distance between Person 38 and Person 69: 128 bits
  Distance between Person 38 and Person 70: 74 bits
  Distance between Person 38 and Person 71: 134 bits
  Distance between Person 38 and Person 72: 116 bits
  Distance between Person 38 and Person 73: 116 bits
  Distance between Person 38 and Person 74: 125 bits
  Distance between Person 38 and Person 75: 120 bits
  Distance between Person 38 and Person 76: 127 bits
  Distance between Person 38 and Person 77: 116 bits
  Distance between Person 38 and Person 78: 109 bits
  Distance between Person 38 and Person 79: 144 bits
  Distance between Person 38 and Person 80: 112 bits
  Distance between Person 38 and Person 81: 125 bits
  Distance between Person 38 and Person 82: 141 bits
  Distance between Person 38 and Person 83: 138 bits
  Distance between Person 38 and Person 84: 142 bits
  Distance between Person 38 and Person 85: 118 bits
  Distance between Person 38 and Person 86: 114 bits
  Distance between Person 38 and Person 87: 142 bits
  Distance between Person 38 and Person 88: 121 bits
  Distance between Person 38 and Person 89: 124 bits
  Distance between Person 39 and Person 40: 117 bits
  Distance between Person 39 and Person 41: 137 bits
  Distance between Person 39 and Person 42: 141 bits
  Distance between Person 39 and Person 43: 138 bits
  Distance between Person 39 and Person 44: 127 bits
  Distance between Person 39 and Person 45: 108 bits
  Distance between Person 39 and Person 46: 138 bits
  Distance between Person 39 and Person 47: 104 bits
  Distance between Person 39 and Person 48: 131 bits
  Distance between Person 39 and Person 49: 141 bits
  Distance between Person 39 and Person 50: 112 bits
  Distance between Person 39 and Person 51: 128 bits
  Distance between Person 39 and Person 52: 120 bits
  Distance between Person 39 and Person 53: 142 bits
  Distance between Person 39 and Person 54: 142 bits
  Distance between Person 39 and Person 55: 119 bits
  Distance between Person 39 and Person 56: 107 bits
  Distance between Person 39 and Person 57: 111 bits
  Distance between Person 39 and Person 58: 130 bits
  Distance between Person 39 and Person 59: 123 bits
  Distance between Person 39 and Person 60: 132 bits
  Distance between Person 39 and Person 61: 128 bits
  Distance between Person 39 and Person 62: 123 bits
  Distance between Person 39 and Person 63: 131 bits
  Distance between Person 39 and Person 64: 130 bits
  Distance between Person 39 and Person 65: 142 bits
  Distance between Person 39 and Person 66: 131 bits
  Distance between Person 39 and Person 67: 116 bits
  Distance between Person 39 and Person 68: 130 bits
  Distance between Person 39 and Person 69: 136 bits
  Distance between Person 39 and Person 70: 138 bits
  Distance between Person 39 and Person 71: 128 bits
  Distance between Person 39 and Person 72: 138 bits
  Distance between Person 39 and Person 73: 124 bits
  Distance between Person 39 and Person 74: 119 bits
  Distance between Person 39 and Person 75: 118 bits
  Distance between Person 39 and Person 76: 121 bits
  Distance between Person 39 and Person 77: 116 bits
  Distance between Person 39 and Person 78: 127 bits
  Distance between Person 39 and Person 79: 124 bits
  Distance between Person 39 and Person 80: 116 bits
  Distance between Person 39 and Person 81: 127 bits
  Distance between Person 39 and Person 82: 137 bits
  Distance between Person 39 and Person 83: 126 bits
  Distance between Person 39 and Person 84: 114 bits
  Distance between Person 39 and Person 85: 134 bits
  Distance between Person 39 and Person 86: 116 bits
  Distance between Person 39 and Person 87: 122 bits
  Distance between Person 39 and Person 88: 137 bits
  Distance between Person 39 and Person 89: 136 bits
  Distance between Person 40 and Person 41: 118 bits
  Distance between Person 40 and Person 42: 134 bits
  Distance between Person 40 and Person 43: 101 bits
  Distance between Person 40 and Person 44: 140 bits
  Distance between Person 40 and Person 45: 111 bits
  Distance between Person 40 and Person 46: 135 bits
  Distance between Person 40 and Person 47: 111 bits
  Distance between Person 40 and Person 48: 116 bits
  Distance between Person 40 and Person 49: 138 bits
  Distance between Person 40 and Person 50: 123 bits
  Distance between Person 40 and Person 51: 129 bits
  Distance between Person 40 and Person 52: 129 bits
  Distance between Person 40 and Person 53: 139 bits
  Distance between Person 40 and Person 54: 137 bits
  Distance between Person 40 and Person 55: 122 bits
  Distance between Person 40 and Person 56: 128 bits
  Distance between Person 40 and Person 57: 124 bits
  Distance between Person 40 and Person 58: 113 bits
  Distance between Person 40 and Person 59: 132 bits
  Distance between Person 40 and Person 60: 137 bits
  Distance between Person 40 and Person 61: 129 bits
  Distance between Person 40 and Person 62: 126 bits
  Distance between Person 40 and Person 63: 134 bits
  Distance between Person 40 and Person 64: 125 bits
  Distance between Person 40 and Person 65: 125 bits
  Distance between Person 40 and Person 66: 126 bits
  Distance between Person 40 and Person 67: 143 bits
  Distance between Person 40 and Person 68: 135 bits
  Distance between Person 40 and Person 69: 129 bits
  Distance between Person 40 and Person 70: 139 bits
  Distance between Person 40 and Person 71: 109 bits
  Distance between Person 40 and Person 72: 127 bits
  Distance between Person 40 and Person 73: 123 bits
  Distance between Person 40 and Person 74: 120 bits
  Distance between Person 40 and Person 75: 127 bits
  Distance between Person 40 and Person 76: 134 bits
  Distance between Person 40 and Person 77: 123 bits
  Distance between Person 40 and Person 78: 124 bits
  Distance between Person 40 and Person 79: 137 bits
  Distance between Person 40 and Person 80: 133 bits
  Distance between Person 40 and Person 81: 126 bits
  Distance between Person 40 and Person 82: 124 bits
  Distance between Person 40 and Person 83: 131 bits
  Distance between Person 40 and Person 84: 135 bits
  Distance between Person 40 and Person 85: 131 bits
  Distance between Person 40 and Person 86: 131 bits
  Distance between Person 40 and Person 87: 131 bits
  Distance between Person 40 and Person 88: 128 bits
  Distance between Person 40 and Person 89: 129 bits
  Distance between Person 41 and Person 42: 130 bits
  Distance between Person 41 and Person 43: 123 bits
  Distance between Person 41 and Person 44: 126 bits
  Distance between Person 41 and Person 45: 99 bits
  Distance between Person 41 and Person 46: 139 bits
  Distance between Person 41 and Person 47: 123 bits
  Distance between Person 41 and Person 48: 126 bits
  Distance between Person 41 and Person 49: 124 bits
  Distance between Person 41 and Person 50: 139 bits
  Distance between Person 41 and Person 51: 129 bits
  Distance between Person 41 and Person 52: 111 bits
  Distance between Person 41 and Person 53: 123 bits
  Distance between Person 41 and Person 54: 139 bits
  Distance between Person 41 and Person 55: 124 bits
  Distance between Person 41 and Person 56: 118 bits
  Distance between Person 41 and Person 57: 122 bits
  Distance between Person 41 and Person 58: 127 bits
  Distance between Person 41 and Person 59: 120 bits
  Distance between Person 41 and Person 60: 123 bits
  Distance between Person 41 and Person 61: 123 bits
  Distance between Person 41 and Person 62: 136 bits
  Distance between Person 41 and Person 63: 114 bits
  Distance between Person 41 and Person 64: 123 bits
  Distance between Person 41 and Person 65: 125 bits
  Distance between Person 41 and Person 66: 110 bits
  Distance between Person 41 and Person 67: 131 bits
  Distance between Person 41 and Person 68: 139 bits
  Distance between Person 41 and Person 69: 115 bits
  Distance between Person 41 and Person 70: 127 bits
  Distance between Person 41 and Person 71: 133 bits
  Distance between Person 41 and Person 72: 123 bits
  Distance between Person 41 and Person 73: 139 bits
  Distance between Person 41 and Person 74: 128 bits
  Distance between Person 41 and Person 75: 131 bits
  Distance between Person 41 and Person 76: 132 bits
  Distance between Person 41 and Person 77: 117 bits
  Distance between Person 41 and Person 78: 132 bits
  Distance between Person 41 and Person 79: 115 bits
  Distance between Person 41 and Person 80: 137 bits
  Distance between Person 41 and Person 81: 138 bits
  Distance between Person 41 and Person 82: 124 bits
  Distance between Person 41 and Person 83: 115 bits
  Distance between Person 41 and Person 84: 117 bits
  Distance between Person 41 and Person 85: 125 bits
  Distance between Person 41 and Person 86: 113 bits
  Distance between Person 41 and Person 87: 133 bits
  Distance between Person 41 and Person 88: 138 bits
  Distance between Person 41 and Person 89: 121 bits
  Distance between Person 42 and Person 43: 129 bits
  Distance between Person 42 and Person 44: 152 bits
  Distance between Person 42 and Person 45: 139 bits
  Distance between Person 42 and Person 46: 119 bits
  Distance between Person 42 and Person 47: 119 bits
  Distance between Person 42 and Person 48: 124 bits
  Distance between Person 42 and Person 49: 128 bits
  Distance between Person 42 and Person 50: 131 bits
  Distance between Person 42 and Person 51: 113 bits
  Distance between Person 42 and Person 52: 137 bits
  Distance between Person 42 and Person 53: 129 bits
  Distance between Person 42 and Person 54: 123 bits
  Distance between Person 42 and Person 55: 134 bits
  Distance between Person 42 and Person 56: 134 bits
  Distance between Person 42 and Person 57: 110 bits
  Distance between Person 42 and Person 58: 125 bits
  Distance between Person 42 and Person 59: 110 bits
  Distance between Person 42 and Person 60: 123 bits
  Distance between Person 42 and Person 61: 133 bits
  Distance between Person 42 and Person 62: 136 bits
  Distance between Person 42 and Person 63: 132 bits
  Distance between Person 42 and Person 64: 131 bits
  Distance between Person 42 and Person 65: 131 bits
  Distance between Person 42 and Person 66: 122 bits
  Distance between Person 42 and Person 67: 127 bits
  Distance between Person 42 and Person 68: 133 bits
  Distance between Person 42 and Person 69: 123 bits
  Distance between Person 42 and Person 70: 115 bits
  Distance between Person 42 and Person 71: 133 bits
  Distance between Person 42 and Person 72: 137 bits
  Distance between Person 42 and Person 73: 135 bits
  Distance between Person 42 and Person 74: 118 bits
  Distance between Person 42 and Person 75: 119 bits
  Distance between Person 42 and Person 76: 134 bits
  Distance between Person 42 and Person 77: 125 bits
  Distance between Person 42 and Person 78: 126 bits
  Distance between Person 42 and Person 79: 119 bits
  Distance between Person 42 and Person 80: 143 bits
  Distance between Person 42 and Person 81: 118 bits
  Distance between Person 42 and Person 82: 130 bits
  Distance between Person 42 and Person 83: 143 bits
  Distance between Person 42 and Person 84: 137 bits
  Distance between Person 42 and Person 85: 133 bits
  Distance between Person 42 and Person 86: 137 bits
  Distance between Person 42 and Person 87: 135 bits
  Distance between Person 42 and Person 88: 132 bits
  Distance between Person 42 and Person 89: 135 bits
  Distance between Person 43 and Person 44: 147 bits
  Distance between Person 43 and Person 45: 130 bits
  Distance between Person 43 and Person 46: 132 bits
  Distance between Person 43 and Person 47: 112 bits
  Distance between Person 43 and Person 48: 121 bits
  Distance between Person 43 and Person 49: 113 bits
  Distance between Person 43 and Person 50: 142 bits
  Distance between Person 43 and Person 51: 116 bits
  Distance between Person 43 and Person 52: 126 bits
  Distance between Person 43 and Person 53: 152 bits
  Distance between Person 43 and Person 54: 114 bits
  Distance between Person 43 and Person 55: 139 bits
  Distance between Person 43 and Person 56: 125 bits
  Distance between Person 43 and Person 57: 141 bits
  Distance between Person 43 and Person 58: 116 bits
  Distance between Person 43 and Person 59: 127 bits
  Distance between Person 43 and Person 60: 140 bits
  Distance between Person 43 and Person 61: 140 bits
  Distance between Person 43 and Person 62: 129 bits
  Distance between Person 43 and Person 63: 113 bits
  Distance between Person 43 and Person 64: 130 bits
  Distance between Person 43 and Person 65: 146 bits
  Distance between Person 43 and Person 66: 133 bits
  Distance between Person 43 and Person 67: 118 bits
  Distance between Person 43 and Person 68: 134 bits
  Distance between Person 43 and Person 69: 118 bits
  Distance between Person 43 and Person 70: 118 bits
  Distance between Person 43 and Person 71: 114 bits
  Distance between Person 43 and Person 72: 150 bits
  Distance between Person 43 and Person 73: 130 bits
  Distance between Person 43 and Person 74: 117 bits
  Distance between Person 43 and Person 75: 124 bits
  Distance between Person 43 and Person 76: 113 bits
  Distance between Person 43 and Person 77: 116 bits
  Distance between Person 43 and Person 78: 117 bits
  Distance between Person 43 and Person 79: 132 bits
  Distance between Person 43 and Person 80: 126 bits
  Distance between Person 43 and Person 81: 123 bits
  Distance between Person 43 and Person 82: 125 bits
  Distance between Person 43 and Person 83: 122 bits
  Distance between Person 43 and Person 84: 134 bits
  Distance between Person 43 and Person 85: 120 bits
  Distance between Person 43 and Person 86: 134 bits
  Distance between Person 43 and Person 87: 142 bits
  Distance between Person 43 and Person 88: 133 bits
  Distance between Person 43 and Person 89: 126 bits
  Distance between Person 44 and Person 45: 81 bits
  Distance between Person 44 and Person 46: 123 bits
  Distance between Person 44 and Person 47: 111 bits
  Distance between Person 44 and Person 48: 136 bits
  Distance between Person 44 and Person 49: 138 bits
  Distance between Person 44 and Person 50: 125 bits
  Distance between Person 44 and Person 51: 125 bits
  Distance between Person 44 and Person 52: 125 bits
  Distance between Person 44 and Person 53: 113 bits
  Distance between Person 44 and Person 54: 115 bits
  Distance between Person 44 and Person 55: 112 bits
  Distance between Person 44 and Person 56: 120 bits
  Distance between Person 44 and Person 57: 144 bits
  Distance between Person 44 and Person 58: 119 bits
  Distance between Person 44 and Person 59: 134 bits
  Distance between Person 44 and Person 60: 137 bits
  Distance between Person 44 and Person 61: 135 bits
  Distance between Person 44 and Person 62: 128 bits
  Distance between Person 44 and Person 63: 130 bits
  Distance between Person 44 and Person 64: 119 bits
  Distance between Person 44 and Person 65: 103 bits
  Distance between Person 44 and Person 66: 116 bits
  Distance between Person 44 and Person 67: 127 bits
  Distance between Person 44 and Person 68: 133 bits
  Distance between Person 44 and Person 69: 109 bits
  Distance between Person 44 and Person 70: 109 bits
  Distance between Person 44 and Person 71: 143 bits
  Distance between Person 44 and Person 72: 113 bits
  Distance between Person 44 and Person 73: 123 bits
  Distance between Person 44 and Person 74: 126 bits
  Distance between Person 44 and Person 75: 125 bits
  Distance between Person 44 and Person 76: 124 bits
  Distance between Person 44 and Person 77: 119 bits
  Distance between Person 44 and Person 78: 126 bits
  Distance between Person 44 and Person 79: 133 bits
  Distance between Person 44 and Person 80: 125 bits
  Distance between Person 44 and Person 81: 140 bits
  Distance between Person 44 and Person 82: 110 bits
  Distance between Person 44 and Person 83: 123 bits
  Distance between Person 44 and Person 84: 129 bits
  Distance between Person 44 and Person 85: 129 bits
  Distance between Person 44 and Person 86: 127 bits
  Distance between Person 44 and Person 87: 113 bits
  Distance between Person 44 and Person 88: 120 bits
  Distance between Person 44 and Person 89: 131 bits
  Distance between Person 45 and Person 46: 126 bits
  Distance between Person 45 and Person 47: 98 bits
  Distance between Person 45 and Person 48: 129 bits
  Distance between Person 45 and Person 49: 141 bits
  Distance between Person 45 and Person 50: 134 bits
  Distance between Person 45 and Person 51: 124 bits
  Distance between Person 45 and Person 52: 120 bits
  Distance between Person 45 and Person 53: 120 bits
  Distance between Person 45 and Person 54: 136 bits
  Distance between Person 45 and Person 55: 105 bits
  Distance between Person 45 and Person 56: 121 bits
  Distance between Person 45 and Person 57: 133 bits
  Distance between Person 45 and Person 58: 120 bits
  Distance between Person 45 and Person 59: 133 bits
  Distance between Person 45 and Person 60: 116 bits
  Distance between Person 45 and Person 61: 118 bits
  Distance between Person 45 and Person 62: 131 bits
  Distance between Person 45 and Person 63: 111 bits
  Distance between Person 45 and Person 64: 124 bits
  Distance between Person 45 and Person 65: 122 bits
  Distance between Person 45 and Person 66: 123 bits
  Distance between Person 45 and Person 67: 130 bits
  Distance between Person 45 and Person 68: 126 bits
  Distance between Person 45 and Person 69: 122 bits
  Distance between Person 45 and Person 70: 144 bits
  Distance between Person 45 and Person 71: 130 bits
  Distance between Person 45 and Person 72: 128 bits
  Distance between Person 45 and Person 73: 116 bits
  Distance between Person 45 and Person 74: 119 bits
  Distance between Person 45 and Person 75: 120 bits
  Distance between Person 45 and Person 76: 121 bits
  Distance between Person 45 and Person 77: 128 bits
  Distance between Person 45 and Person 78: 135 bits
  Distance between Person 45 and Person 79: 152 bits
  Distance between Person 45 and Person 80: 118 bits
  Distance between Person 45 and Person 81: 145 bits
  Distance between Person 45 and Person 82: 111 bits
  Distance between Person 45 and Person 83: 110 bits
  Distance between Person 45 and Person 84: 124 bits
  Distance between Person 45 and Person 85: 126 bits
  Distance between Person 45 and Person 86: 110 bits
  Distance between Person 45 and Person 87: 140 bits
  Distance between Person 45 and Person 88: 123 bits
  Distance between Person 45 and Person 89: 122 bits
  Distance between Person 46 and Person 47: 128 bits
  Distance between Person 46 and Person 48: 127 bits
  Distance between Person 46 and Person 49: 125 bits
  Distance between Person 46 and Person 50: 134 bits
  Distance between Person 46 and Person 51: 118 bits
  Distance between Person 46 and Person 52: 142 bits
  Distance between Person 46 and Person 53: 118 bits
  Distance between Person 46 and Person 54: 120 bits
  Distance between Person 46 and Person 55: 151 bits
  Distance between Person 46 and Person 56: 127 bits
  Distance between Person 46 and Person 57: 127 bits
  Distance between Person 46 and Person 58: 92 bits
  Distance between Person 46 and Person 59: 123 bits
  Distance between Person 46 and Person 60: 128 bits
  Distance between Person 46 and Person 61: 120 bits
  Distance between Person 46 and Person 62: 133 bits
  Distance between Person 46 and Person 63: 125 bits
  Distance between Person 46 and Person 64: 150 bits
  Distance between Person 46 and Person 65: 114 bits
  Distance between Person 46 and Person 66: 143 bits
  Distance between Person 46 and Person 67: 120 bits
  Distance between Person 46 and Person 68: 132 bits
  Distance between Person 46 and Person 69: 142 bits
  Distance between Person 46 and Person 70: 116 bits
  Distance between Person 46 and Person 71: 134 bits
  Distance between Person 46 and Person 72: 124 bits
  Distance between Person 46 and Person 73: 118 bits
  Distance between Person 46 and Person 74: 129 bits
  Distance between Person 46 and Person 75: 136 bits
  Distance between Person 46 and Person 76: 115 bits
  Distance between Person 46 and Person 77: 126 bits
  Distance between Person 46 and Person 78: 129 bits
  Distance between Person 46 and Person 79: 136 bits
  Distance between Person 46 and Person 80: 118 bits
  Distance between Person 46 and Person 81: 121 bits
  Distance between Person 46 and Person 82: 113 bits
  Distance between Person 46 and Person 83: 142 bits
  Distance between Person 46 and Person 84: 118 bits
  Distance between Person 46 and Person 85: 122 bits
  Distance between Person 46 and Person 86: 126 bits
  Distance between Person 46 and Person 87: 144 bits
  Distance between Person 46 and Person 88: 139 bits
  Distance between Person 46 and Person 89: 114 bits
  Distance between Person 47 and Person 48: 115 bits
  Distance between Person 47 and Person 49: 129 bits
  Distance between Person 47 and Person 50: 120 bits
  Distance between Person 47 and Person 51: 132 bits
  Distance between Person 47 and Person 52: 142 bits
  Distance between Person 47 and Person 53: 136 bits
  Distance between Person 47 and Person 54: 124 bits
  Distance between Person 47 and Person 55: 125 bits
  Distance between Person 47 and Person 56: 101 bits
  Distance between Person 47 and Person 57: 125 bits
  Distance between Person 47 and Person 58: 110 bits
  Distance between Person 47 and Person 59: 115 bits
  Distance between Person 47 and Person 60: 138 bits
  Distance between Person 47 and Person 61: 138 bits
  Distance between Person 47 and Person 62: 141 bits
  Distance between Person 47 and Person 63: 149 bits
  Distance between Person 47 and Person 64: 132 bits
  Distance between Person 47 and Person 65: 132 bits
  Distance between Person 47 and Person 66: 123 bits
  Distance between Person 47 and Person 67: 122 bits
  Distance between Person 47 and Person 68: 138 bits
  Distance between Person 47 and Person 69: 122 bits
  Distance between Person 47 and Person 70: 132 bits
  Distance between Person 47 and Person 71: 124 bits
  Distance between Person 47 and Person 72: 122 bits
  Distance between Person 47 and Person 73: 126 bits
  Distance between Person 47 and Person 74: 117 bits
  Distance between Person 47 and Person 75: 128 bits
  Distance between Person 47 and Person 76: 123 bits
  Distance between Person 47 and Person 77: 128 bits
  Distance between Person 47 and Person 78: 135 bits
  Distance between Person 47 and Person 79: 114 bits
  Distance between Person 47 and Person 80: 120 bits
  Distance between Person 47 and Person 81: 145 bits
  Distance between Person 47 and Person 82: 127 bits
  Distance between Person 47 and Person 83: 118 bits
  Distance between Person 47 and Person 84: 114 bits
  Distance between Person 47 and Person 85: 122 bits
  Distance between Person 47 and Person 86: 128 bits
  Distance between Person 47 and Person 87: 120 bits
  Distance between Person 47 and Person 88: 117 bits
  Distance between Person 47 and Person 89: 126 bits
  Distance between Person 48 and Person 49: 120 bits
  Distance between Person 48 and Person 50: 125 bits
  Distance between Person 48 and Person 51: 139 bits
  Distance between Person 48 and Person 52: 131 bits
  Distance between Person 48 and Person 53: 135 bits
  Distance between Person 48 and Person 54: 127 bits
  Distance between Person 48 and Person 55: 120 bits
  Distance between Person 48 and Person 56: 128 bits
  Distance between Person 48 and Person 57: 130 bits
  Distance between Person 48 and Person 58: 131 bits
  Distance between Person 48 and Person 59: 132 bits
  Distance between Person 48 and Person 60: 125 bits
  Distance between Person 48 and Person 61: 127 bits
  Distance between Person 48 and Person 62: 114 bits
  Distance between Person 48 and Person 63: 128 bits
  Distance between Person 48 and Person 64: 117 bits
  Distance between Person 48 and Person 65: 125 bits
  Distance between Person 48 and Person 66: 114 bits
  Distance between Person 48 and Person 67: 137 bits
  Distance between Person 48 and Person 68: 127 bits
  Distance between Person 48 and Person 69: 133 bits
  Distance between Person 48 and Person 70: 119 bits
  Distance between Person 48 and Person 71: 131 bits
  Distance between Person 48 and Person 72: 121 bits
  Distance between Person 48 and Person 73: 141 bits
  Distance between Person 48 and Person 74: 120 bits
  Distance between Person 48 and Person 75: 133 bits
  Distance between Person 48 and Person 76: 108 bits
  Distance between Person 48 and Person 77: 115 bits
  Distance between Person 48 and Person 78: 130 bits
  Distance between Person 48 and Person 79: 125 bits
  Distance between Person 48 and Person 80: 135 bits
  Distance between Person 48 and Person 81: 114 bits
  Distance between Person 48 and Person 82: 146 bits
  Distance between Person 48 and Person 83: 137 bits
  Distance between Person 48 and Person 84: 129 bits
  Distance between Person 48 and Person 85: 117 bits
  Distance between Person 48 and Person 86: 137 bits
  Distance between Person 48 and Person 87: 127 bits
  Distance between Person 48 and Person 88: 148 bits
  Distance between Person 48 and Person 89: 131 bits
  Distance between Person 49 and Person 50: 131 bits
  Distance between Person 49 and Person 51: 131 bits
  Distance between Person 49 and Person 52: 133 bits
  Distance between Person 49 and Person 53: 139 bits
  Distance between Person 49 and Person 54: 137 bits
  Distance between Person 49 and Person 55: 138 bits
  Distance between Person 49 and Person 56: 132 bits
  Distance between Person 49 and Person 57: 124 bits
  Distance between Person 49 and Person 58: 131 bits
  Distance between Person 49 and Person 59: 120 bits
  Distance between Person 49 and Person 60: 115 bits
  Distance between Person 49 and Person 61: 123 bits
  Distance between Person 49 and Person 62: 128 bits
  Distance between Person 49 and Person 63: 134 bits
  Distance between Person 49 and Person 64: 133 bits
  Distance between Person 49 and Person 65: 131 bits
  Distance between Person 49 and Person 66: 134 bits
  Distance between Person 49 and Person 67: 129 bits
  Distance between Person 49 and Person 68: 141 bits
  Distance between Person 49 and Person 69: 129 bits
  Distance between Person 49 and Person 70: 125 bits
  Distance between Person 49 and Person 71: 109 bits
  Distance between Person 49 and Person 72: 105 bits
  Distance between Person 49 and Person 73: 135 bits
  Distance between Person 49 and Person 74: 122 bits
  Distance between Person 49 and Person 75: 131 bits
  Distance between Person 49 and Person 76: 126 bits
  Distance between Person 49 and Person 77: 123 bits
  Distance between Person 49 and Person 78: 132 bits
  Distance between Person 49 and Person 79: 119 bits
  Distance between Person 49 and Person 80: 115 bits
  Distance between Person 49 and Person 81: 128 bits
  Distance between Person 49 and Person 82: 148 bits
  Distance between Person 49 and Person 83: 127 bits
  Distance between Person 49 and Person 84: 153 bits
  Distance between Person 49 and Person 85: 129 bits
  Distance between Person 49 and Person 86: 121 bits
  Distance between Person 49 and Person 87: 129 bits
  Distance between Person 49 and Person 88: 130 bits
  Distance between Person 49 and Person 89: 117 bits
  Distance between Person 50 and Person 51: 124 bits
  Distance between Person 50 and Person 52: 136 bits
  Distance between Person 50 and Person 53: 122 bits
  Distance between Person 50 and Person 54: 122 bits
  Distance between Person 50 and Person 55: 121 bits
  Distance between Person 50 and Person 56: 123 bits
  Distance between Person 50 and Person 57: 127 bits
  Distance between Person 50 and Person 58: 142 bits
  Distance between Person 50 and Person 59: 117 bits
  Distance between Person 50 and Person 60: 122 bits
  Distance between Person 50 and Person 61: 126 bits
  Distance between Person 50 and Person 62: 119 bits
  Distance between Person 50 and Person 63: 135 bits
  Distance between Person 50 and Person 64: 126 bits
  Distance between Person 50 and Person 65: 132 bits
  Distance between Person 50 and Person 66: 145 bits
  Distance between Person 50 and Person 67: 138 bits
  Distance between Person 50 and Person 68: 126 bits
  Distance between Person 50 and Person 69: 126 bits
  Distance between Person 50 and Person 70: 136 bits
  Distance between Person 50 and Person 71: 130 bits
  Distance between Person 50 and Person 72: 126 bits
  Distance between Person 50 and Person 73: 120 bits
  Distance between Person 50 and Person 74: 133 bits
  Distance between Person 50 and Person 75: 112 bits
  Distance between Person 50 and Person 76: 105 bits
  Distance between Person 50 and Person 77: 112 bits
  Distance between Person 50 and Person 78: 121 bits
  Distance between Person 50 and Person 79: 120 bits
  Distance between Person 50 and Person 80: 128 bits
  Distance between Person 50 and Person 81: 123 bits
  Distance between Person 50 and Person 82: 131 bits
  Distance between Person 50 and Person 83: 136 bits
  Distance between Person 50 and Person 84: 114 bits
  Distance between Person 50 and Person 85: 134 bits
  Distance between Person 50 and Person 86: 140 bits
  Distance between Person 50 and Person 87: 124 bits
  Distance between Person 50 and Person 88: 123 bits
  Distance between Person 50 and Person 89: 122 bits
  Distance between Person 51 and Person 52: 132 bits
  Distance between Person 51 and Person 53: 122 bits
  Distance between Person 51 and Person 54: 122 bits
  Distance between Person 51 and Person 55: 131 bits
  Distance between Person 51 and Person 56: 135 bits
  Distance between Person 51 and Person 57: 141 bits
  Distance between Person 51 and Person 58: 122 bits
  Distance between Person 51 and Person 59: 129 bits
  Distance between Person 51 and Person 60: 144 bits
  Distance between Person 51 and Person 61: 126 bits
  Distance between Person 51 and Person 62: 131 bits
  Distance between Person 51 and Person 63: 127 bits
  Distance between Person 51 and Person 64: 104 bits
  Distance between Person 51 and Person 65: 134 bits
  Distance between Person 51 and Person 66: 137 bits
  Distance between Person 51 and Person 67: 142 bits
  Distance between Person 51 and Person 68: 128 bits
  Distance between Person 51 and Person 69: 132 bits
  Distance between Person 51 and Person 70: 120 bits
  Distance between Person 51 and Person 71: 138 bits
  Distance between Person 51 and Person 72: 136 bits
  Distance between Person 51 and Person 73: 120 bits
  Distance between Person 51 and Person 74: 119 bits
  Distance between Person 51 and Person 75: 132 bits
  Distance between Person 51 and Person 76: 135 bits
  Distance between Person 51 and Person 77: 120 bits
  Distance between Person 51 and Person 78: 127 bits
  Distance between Person 51 and Person 79: 132 bits
  Distance between Person 51 and Person 80: 116 bits
  Distance between Person 51 and Person 81: 133 bits
  Distance between Person 51 and Person 82: 111 bits
  Distance between Person 51 and Person 83: 128 bits
  Distance between Person 51 and Person 84: 120 bits
  Distance between Person 51 and Person 85: 118 bits
  Distance between Person 51 and Person 86: 128 bits
  Distance between Person 51 and Person 87: 130 bits
  Distance between Person 51 and Person 88: 135 bits
  Distance between Person 51 and Person 89: 120 bits
  Distance between Person 52 and Person 53: 132 bits
  Distance between Person 52 and Person 54: 120 bits
  Distance between Person 52 and Person 55: 123 bits
  Distance between Person 52 and Person 56: 141 bits
  Distance between Person 52 and Person 57: 131 bits
  Distance between Person 52 and Person 58: 126 bits
  Distance between Person 52 and Person 59: 145 bits
  Distance between Person 52 and Person 60: 130 bits
  Distance between Person 52 and Person 61: 126 bits
  Distance between Person 52 and Person 62: 139 bits
  Distance between Person 52 and Person 63: 131 bits
  Distance between Person 52 and Person 64: 120 bits
  Distance between Person 52 and Person 65: 138 bits
  Distance between Person 52 and Person 66: 117 bits
  Distance between Person 52 and Person 67: 134 bits
  Distance between Person 52 and Person 68: 132 bits
  Distance between Person 52 and Person 69: 134 bits
  Distance between Person 52 and Person 70: 126 bits
  Distance between Person 52 and Person 71: 130 bits
  Distance between Person 52 and Person 72: 128 bits
  Distance between Person 52 and Person 73: 130 bits
  Distance between Person 52 and Person 74: 125 bits
  Distance between Person 52 and Person 75: 130 bits
  Distance between Person 52 and Person 76: 113 bits
  Distance between Person 52 and Person 77: 124 bits
  Distance between Person 52 and Person 78: 135 bits
  Distance between Person 52 and Person 79: 122 bits
  Distance between Person 52 and Person 80: 134 bits
  Distance between Person 52 and Person 81: 139 bits
  Distance between Person 52 and Person 82: 129 bits
  Distance between Person 52 and Person 83: 126 bits
  Distance between Person 52 and Person 84: 120 bits
  Distance between Person 52 and Person 85: 136 bits
  Distance between Person 52 and Person 86: 126 bits
  Distance between Person 52 and Person 87: 122 bits
  Distance between Person 52 and Person 88: 137 bits
  Distance between Person 52 and Person 89: 114 bits
  Distance between Person 53 and Person 54: 124 bits
  Distance between Person 53 and Person 55: 107 bits
  Distance between Person 53 and Person 56: 129 bits
  Distance between Person 53 and Person 57: 123 bits
  Distance between Person 53 and Person 58: 134 bits
  Distance between Person 53 and Person 59: 139 bits
  Distance between Person 53 and Person 60: 118 bits
  Distance between Person 53 and Person 61: 128 bits
  Distance between Person 53 and Person 62: 127 bits
  Distance between Person 53 and Person 63: 123 bits
  Distance between Person 53 and Person 64: 120 bits
  Distance between Person 53 and Person 65: 122 bits
  Distance between Person 53 and Person 66: 123 bits
  Distance between Person 53 and Person 67: 118 bits
  Distance between Person 53 and Person 68: 144 bits
  Distance between Person 53 and Person 69: 120 bits
  Distance between Person 53 and Person 70: 132 bits
  Distance between Person 53 and Person 71: 144 bits
  Distance between Person 53 and Person 72: 132 bits
  Distance between Person 53 and Person 73: 122 bits
  Distance between Person 53 and Person 74: 129 bits
  Distance between Person 53 and Person 75: 134 bits
  Distance between Person 53 and Person 76: 113 bits
  Distance between Person 53 and Person 77: 134 bits
  Distance between Person 53 and Person 78: 139 bits
  Distance between Person 53 and Person 79: 132 bits
  Distance between Person 53 and Person 80: 134 bits
  Distance between Person 53 and Person 81: 135 bits
  Distance between Person 53 and Person 82: 119 bits
  Distance between Person 53 and Person 83: 132 bits
  Distance between Person 53 and Person 84: 130 bits
  Distance between Person 53 and Person 85: 130 bits
  Distance between Person 53 and Person 86: 124 bits
  Distance between Person 53 and Person 87: 136 bits
  Distance between Person 53 and Person 88: 113 bits
  Distance between Person 53 and Person 89: 122 bits
  Distance between Person 54 and Person 55: 133 bits
  Distance between Person 54 and Person 56: 117 bits
  Distance between Person 54 and Person 57: 131 bits
  Distance between Person 54 and Person 58: 108 bits
  Distance between Person 54 and Person 59: 131 bits
  Distance between Person 54 and Person 60: 120 bits
  Distance between Person 54 and Person 61: 140 bits
  Distance between Person 54 and Person 62: 137 bits
  Distance between Person 54 and Person 63: 127 bits
  Distance between Person 54 and Person 64: 124 bits
  Distance between Person 54 and Person 65: 124 bits
  Distance between Person 54 and Person 66: 135 bits
  Distance between Person 54 and Person 67: 134 bits
  Distance between Person 54 and Person 68: 126 bits
  Distance between Person 54 and Person 69: 118 bits
  Distance between Person 54 and Person 70: 110 bits
  Distance between Person 54 and Person 71: 140 bits
  Distance between Person 54 and Person 72: 136 bits
  Distance between Person 54 and Person 73: 110 bits
  Distance between Person 54 and Person 74: 119 bits
  Distance between Person 54 and Person 75: 124 bits
  Distance between Person 54 and Person 76: 103 bits
  Distance between Person 54 and Person 77: 136 bits
  Distance between Person 54 and Person 78: 139 bits
  Distance between Person 54 and Person 79: 140 bits
  Distance between Person 54 and Person 80: 126 bits
  Distance between Person 54 and Person 81: 141 bits
  Distance between Person 54 and Person 82: 113 bits
  Distance between Person 54 and Person 83: 142 bits
  Distance between Person 54 and Person 84: 142 bits
  Distance between Person 54 and Person 85: 120 bits
  Distance between Person 54 and Person 86: 130 bits
  Distance between Person 54 and Person 87: 130 bits
  Distance between Person 54 and Person 88: 129 bits
  Distance between Person 54 and Person 89: 116 bits
  Distance between Person 55 and Person 56: 130 bits
  Distance between Person 55 and Person 57: 108 bits
  Distance between Person 55 and Person 58: 131 bits
  Distance between Person 55 and Person 59: 132 bits
  Distance between Person 55 and Person 60: 117 bits
  Distance between Person 55 and Person 61: 115 bits
  Distance between Person 55 and Person 62: 120 bits
  Distance between Person 55 and Person 63: 122 bits
  Distance between Person 55 and Person 64: 123 bits
  Distance between Person 55 and Person 65: 127 bits
  Distance between Person 55 and Person 66: 114 bits
  Distance between Person 55 and Person 67: 113 bits
  Distance between Person 55 and Person 68: 125 bits
  Distance between Person 55 and Person 69: 103 bits
  Distance between Person 55 and Person 70: 121 bits
  Distance between Person 55 and Person 71: 129 bits
  Distance between Person 55 and Person 72: 147 bits
  Distance between Person 55 and Person 73: 127 bits
  Distance between Person 55 and Person 74: 144 bits
  Distance between Person 55 and Person 75: 125 bits
  Distance between Person 55 and Person 76: 126 bits
  Distance between Person 55 and Person 77: 111 bits
  Distance between Person 55 and Person 78: 112 bits
  Distance between Person 55 and Person 79: 129 bits
  Distance between Person 55 and Person 80: 115 bits
  Distance between Person 55 and Person 81: 130 bits
  Distance between Person 55 and Person 82: 138 bits
  Distance between Person 55 and Person 83: 121 bits
  Distance between Person 55 and Person 84: 135 bits
  Distance between Person 55 and Person 85: 127 bits
  Distance between Person 55 and Person 86: 127 bits
  Distance between Person 55 and Person 87: 123 bits
  Distance between Person 55 and Person 88: 128 bits
  Distance between Person 55 and Person 89: 137 bits
  Distance between Person 56 and Person 57: 122 bits
  Distance between Person 56 and Person 58: 135 bits
  Distance between Person 56 and Person 59: 114 bits
  Distance between Person 56 and Person 60: 115 bits
  Distance between Person 56 and Person 61: 139 bits
  Distance between Person 56 and Person 62: 142 bits
  Distance between Person 56 and Person 63: 126 bits
  Distance between Person 56 and Person 64: 139 bits
  Distance between Person 56 and Person 65: 137 bits
  Distance between Person 56 and Person 66: 148 bits
  Distance between Person 56 and Person 67: 129 bits
  Distance between Person 56 and Person 68: 129 bits
  Distance between Person 56 and Person 69: 131 bits
  Distance between Person 56 and Person 70: 133 bits
  Distance between Person 56 and Person 71: 125 bits
  Distance between Person 56 and Person 72: 147 bits
  Distance between Person 56 and Person 73: 123 bits
  Distance between Person 56 and Person 74: 132 bits
  Distance between Person 56 and Person 75: 135 bits
  Distance between Person 56 and Person 76: 140 bits
  Distance between Person 56 and Person 77: 135 bits
  Distance between Person 56 and Person 78: 138 bits
  Distance between Person 56 and Person 79: 121 bits
  Distance between Person 56 and Person 80: 121 bits
  Distance between Person 56 and Person 81: 150 bits
  Distance between Person 56 and Person 82: 110 bits
  Distance between Person 56 and Person 83: 123 bits
  Distance between Person 56 and Person 84: 117 bits
  Distance between Person 56 and Person 85: 125 bits
  Distance between Person 56 and Person 86: 133 bits
  Distance between Person 56 and Person 87: 125 bits
  Distance between Person 56 and Person 88: 122 bits
  Distance between Person 56 and Person 89: 125 bits
  Distance between Person 57 and Person 58: 133 bits
  Distance between Person 57 and Person 59: 124 bits
  Distance between Person 57 and Person 60: 111 bits
  Distance between Person 57 and Person 61: 135 bits
  Distance between Person 57 and Person 62: 130 bits
  Distance between Person 57 and Person 63: 130 bits
  Distance between Person 57 and Person 64: 133 bits
  Distance between Person 57 and Person 65: 125 bits
  Distance between Person 57 and Person 66: 134 bits
  Distance between Person 57 and Person 67: 117 bits
  Distance between Person 57 and Person 68: 149 bits
  Distance between Person 57 and Person 69: 135 bits
  Distance between Person 57 and Person 70: 131 bits
  Distance between Person 57 and Person 71: 127 bits
  Distance between Person 57 and Person 72: 125 bits
  Distance between Person 57 and Person 73: 135 bits
  Distance between Person 57 and Person 74: 136 bits
  Distance between Person 57 and Person 75: 137 bits
  Distance between Person 57 and Person 76: 130 bits
  Distance between Person 57 and Person 77: 127 bits
  Distance between Person 57 and Person 78: 132 bits
  Distance between Person 57 and Person 79: 123 bits
  Distance between Person 57 and Person 80: 137 bits
  Distance between Person 57 and Person 81: 114 bits
  Distance between Person 57 and Person 82: 138 bits
  Distance between Person 57 and Person 83: 131 bits
  Distance between Person 57 and Person 84: 121 bits
  Distance between Person 57 and Person 85: 129 bits
  Distance between Person 57 and Person 86: 137 bits
  Distance between Person 57 and Person 87: 129 bits
  Distance between Person 57 and Person 88: 124 bits
  Distance between Person 57 and Person 89: 133 bits
  Distance between Person 58 and Person 59: 123 bits
  Distance between Person 58 and Person 60: 128 bits
  Distance between Person 58 and Person 61: 122 bits
  Distance between Person 58 and Person 62: 127 bits
  Distance between Person 58 and Person 63: 127 bits
  Distance between Person 58 and Person 64: 138 bits
  Distance between Person 58 and Person 65: 114 bits
  Distance between Person 58 and Person 66: 125 bits
  Distance between Person 58 and Person 67: 122 bits
  Distance between Person 58 and Person 68: 132 bits
  Distance between Person 58 and Person 69: 120 bits
  Distance between Person 58 and Person 70: 130 bits
  Distance between Person 58 and Person 71: 114 bits
  Distance between Person 58 and Person 72: 128 bits
  Distance between Person 58 and Person 73: 102 bits
  Distance between Person 58 and Person 74: 123 bits
  Distance between Person 58 and Person 75: 124 bits
  Distance between Person 58 and Person 76: 125 bits
  Distance between Person 58 and Person 77: 140 bits
  Distance between Person 58 and Person 78: 131 bits
  Distance between Person 58 and Person 79: 130 bits
  Distance between Person 58 and Person 80: 110 bits
  Distance between Person 58 and Person 81: 115 bits
  Distance between Person 58 and Person 82: 127 bits
  Distance between Person 58 and Person 83: 124 bits
  Distance between Person 58 and Person 84: 122 bits
  Distance between Person 58 and Person 85: 112 bits
  Distance between Person 58 and Person 86: 118 bits
  Distance between Person 58 and Person 87: 120 bits
  Distance between Person 58 and Person 88: 133 bits
  Distance between Person 58 and Person 89: 116 bits
  Distance between Person 59 and Person 60: 117 bits
  Distance between Person 59 and Person 61: 139 bits
  Distance between Person 59 and Person 62: 112 bits
  Distance between Person 59 and Person 63: 128 bits
  Distance between Person 59 and Person 64: 119 bits
  Distance between Person 59 and Person 65: 133 bits
  Distance between Person 59 and Person 66: 126 bits
  Distance between Person 59 and Person 67: 121 bits
  Distance between Person 59 and Person 68: 115 bits
  Distance between Person 59 and Person 69: 137 bits
  Distance between Person 59 and Person 70: 133 bits
  Distance between Person 59 and Person 71: 125 bits
  Distance between Person 59 and Person 72: 127 bits
  Distance between Person 59 and Person 73: 143 bits
  Distance between Person 59 and Person 74: 126 bits
  Distance between Person 59 and Person 75: 135 bits
  Distance between Person 59 and Person 76: 120 bits
  Distance between Person 59 and Person 77: 109 bits
  Distance between Person 59 and Person 78: 126 bits
  Distance between Person 59 and Person 79: 123 bits
  Distance between Person 59 and Person 80: 137 bits
  Distance between Person 59 and Person 81: 126 bits
  Distance between Person 59 and Person 82: 150 bits
  Distance between Person 59 and Person 83: 129 bits
  Distance between Person 59 and Person 84: 127 bits
  Distance between Person 59 and Person 85: 129 bits
  Distance between Person 59 and Person 86: 149 bits
  Distance between Person 59 and Person 87: 121 bits
  Distance between Person 59 and Person 88: 124 bits
  Distance between Person 59 and Person 89: 141 bits
  Distance between Person 60 and Person 61: 124 bits
  Distance between Person 60 and Person 62: 121 bits
  Distance between Person 60 and Person 63: 127 bits
  Distance between Person 60 and Person 64: 136 bits
  Distance between Person 60 and Person 65: 134 bits
  Distance between Person 60 and Person 66: 139 bits
  Distance between Person 60 and Person 67: 128 bits
  Distance between Person 60 and Person 68: 110 bits
  Distance between Person 60 and Person 69: 128 bits
  Distance between Person 60 and Person 70: 126 bits
  Distance between Person 60 and Person 71: 118 bits
  Distance between Person 60 and Person 72: 120 bits
  Distance between Person 60 and Person 73: 126 bits
  Distance between Person 60 and Person 74: 131 bits
  Distance between Person 60 and Person 75: 120 bits
  Distance between Person 60 and Person 76: 131 bits
  Distance between Person 60 and Person 77: 132 bits
  Distance between Person 60 and Person 78: 123 bits
  Distance between Person 60 and Person 79: 130 bits
  Distance between Person 60 and Person 80: 118 bits
  Distance between Person 60 and Person 81: 149 bits
  Distance between Person 60 and Person 82: 131 bits
  Distance between Person 60 and Person 83: 130 bits
  Distance between Person 60 and Person 84: 136 bits
  Distance between Person 60 and Person 85: 118 bits
  Distance between Person 60 and Person 86: 130 bits
  Distance between Person 60 and Person 87: 138 bits
  Distance between Person 60 and Person 88: 133 bits
  Distance between Person 60 and Person 89: 140 bits
  Distance between Person 61 and Person 62: 137 bits
  Distance between Person 61 and Person 63: 135 bits
  Distance between Person 61 and Person 64: 124 bits
  Distance between Person 61 and Person 65: 128 bits
  Distance between Person 61 and Person 66: 117 bits
  Distance between Person 61 and Person 67: 140 bits
  Distance between Person 61 and Person 68: 132 bits
  Distance between Person 61 and Person 69: 136 bits
  Distance between Person 61 and Person 70: 122 bits
  Distance between Person 61 and Person 71: 118 bits
  Distance between Person 61 and Person 72: 124 bits
  Distance between Person 61 and Person 73: 138 bits
  Distance between Person 61 and Person 74: 129 bits
  Distance between Person 61 and Person 75: 124 bits
  Distance between Person 61 and Person 76: 121 bits
  Distance between Person 61 and Person 77: 126 bits
  Distance between Person 61 and Person 78: 143 bits
  Distance between Person 61 and Person 79: 136 bits
  Distance between Person 61 and Person 80: 118 bits
  Distance between Person 61 and Person 81: 147 bits
  Distance between Person 61 and Person 82: 129 bits
  Distance between Person 61 and Person 83: 126 bits
  Distance between Person 61 and Person 84: 118 bits
  Distance between Person 61 and Person 85: 114 bits
  Distance between Person 61 and Person 86: 122 bits
  Distance between Person 61 and Person 87: 138 bits
  Distance between Person 61 and Person 88: 123 bits
  Distance between Person 61 and Person 89: 120 bits
  Distance between Person 62 and Person 63: 120 bits
  Distance between Person 62 and Person 64: 103 bits
  Distance between Person 62 and Person 65: 129 bits
  Distance between Person 62 and Person 66: 146 bits
  Distance between Person 62 and Person 67: 141 bits
  Distance between Person 62 and Person 68: 121 bits
  Distance between Person 62 and Person 69: 117 bits
  Distance between Person 62 and Person 70: 115 bits
  Distance between Person 62 and Person 71: 117 bits
  Distance between Person 62 and Person 72: 133 bits
  Distance between Person 62 and Person 73: 131 bits
  Distance between Person 62 and Person 74: 136 bits
  Distance between Person 62 and Person 75: 117 bits
  Distance between Person 62 and Person 76: 124 bits
  Distance between Person 62 and Person 77: 131 bits
  Distance between Person 62 and Person 78: 90 bits
  Distance between Person 62 and Person 79: 135 bits
  Distance between Person 62 and Person 80: 131 bits
  Distance between Person 62 and Person 81: 118 bits
  Distance between Person 62 and Person 82: 140 bits
  Distance between Person 62 and Person 83: 129 bits
  Distance between Person 62 and Person 84: 115 bits
  Distance between Person 62 and Person 85: 131 bits
  Distance between Person 62 and Person 86: 141 bits
  Distance between Person 62 and Person 87: 117 bits
  Distance between Person 62 and Person 88: 124 bits
  Distance between Person 62 and Person 89: 127 bits
  Distance between Person 63 and Person 64: 139 bits
  Distance between Person 63 and Person 65: 121 bits
  Distance between Person 63 and Person 66: 124 bits
  Distance between Person 63 and Person 67: 107 bits
  Distance between Person 63 and Person 68: 141 bits
  Distance between Person 63 and Person 69: 131 bits
  Distance between Person 63 and Person 70: 151 bits
  Distance between Person 63 and Person 71: 137 bits
  Distance between Person 63 and Person 72: 147 bits
  Distance between Person 63 and Person 73: 131 bits
  Distance between Person 63 and Person 74: 126 bits
  Distance between Person 63 and Person 75: 117 bits
  Distance between Person 63 and Person 76: 114 bits
  Distance between Person 63 and Person 77: 131 bits
  Distance between Person 63 and Person 78: 124 bits
  Distance between Person 63 and Person 79: 143 bits
  Distance between Person 63 and Person 80: 149 bits
  Distance between Person 63 and Person 81: 128 bits
  Distance between Person 63 and Person 82: 122 bits
  Distance between Person 63 and Person 83: 121 bits
  Distance between Person 63 and Person 84: 137 bits
  Distance between Person 63 and Person 85: 129 bits
  Distance between Person 63 and Person 86: 127 bits
  Distance between Person 63 and Person 87: 131 bits
  Distance between Person 63 and Person 88: 136 bits
  Distance between Person 63 and Person 89: 115 bits
  Distance between Person 64 and Person 65: 116 bits
  Distance between Person 64 and Person 66: 123 bits
  Distance between Person 64 and Person 67: 136 bits
  Distance between Person 64 and Person 68: 136 bits
  Distance between Person 64 and Person 69: 118 bits
  Distance between Person 64 and Person 70: 120 bits
  Distance between Person 64 and Person 71: 122 bits
  Distance between Person 64 and Person 72: 130 bits
  Distance between Person 64 and Person 73: 128 bits
  Distance between Person 64 and Person 74: 123 bits
  Distance between Person 64 and Person 75: 124 bits
  Distance between Person 64 and Person 76: 127 bits
  Distance between Person 64 and Person 77: 128 bits
  Distance between Person 64 and Person 78: 135 bits
  Distance between Person 64 and Person 79: 118 bits
  Distance between Person 64 and Person 80: 130 bits
  Distance between Person 64 and Person 81: 139 bits
  Distance between Person 64 and Person 82: 123 bits
  Distance between Person 64 and Person 83: 130 bits
  Distance between Person 64 and Person 84: 130 bits
  Distance between Person 64 and Person 85: 134 bits
  Distance between Person 64 and Person 86: 138 bits
  Distance between Person 64 and Person 87: 124 bits
  Distance between Person 64 and Person 88: 127 bits
  Distance between Person 64 and Person 89: 136 bits
  Distance between Person 65 and Person 66: 113 bits
  Distance between Person 65 and Person 67: 134 bits
  Distance between Person 65 and Person 68: 122 bits
  Distance between Person 65 and Person 69: 120 bits
  Distance between Person 65 and Person 70: 124 bits
  Distance between Person 65 and Person 71: 130 bits
  Distance between Person 65 and Person 72: 126 bits
  Distance between Person 65 and Person 73: 118 bits
  Distance between Person 65 and Person 74: 133 bits
  Distance between Person 65 and Person 75: 116 bits
  Distance between Person 65 and Person 76: 125 bits
  Distance between Person 65 and Person 77: 128 bits
  Distance between Person 65 and Person 78: 125 bits
  Distance between Person 65 and Person 79: 114 bits
  Distance between Person 65 and Person 80: 142 bits
  Distance between Person 65 and Person 81: 149 bits
  Distance between Person 65 and Person 82: 123 bits
  Distance between Person 65 and Person 83: 126 bits
  Distance between Person 65 and Person 84: 122 bits
  Distance between Person 65 and Person 85: 114 bits
  Distance between Person 65 and Person 86: 144 bits
  Distance between Person 65 and Person 87: 116 bits
  Distance between Person 65 and Person 88: 117 bits
  Distance between Person 65 and Person 89: 128 bits
  Distance between Person 66 and Person 67: 123 bits
  Distance between Person 66 and Person 68: 137 bits
  Distance between Person 66 and Person 69: 127 bits
  Distance between Person 66 and Person 70: 129 bits
  Distance between Person 66 and Person 71: 133 bits
  Distance between Person 66 and Person 72: 119 bits
  Distance between Person 66 and Person 73: 139 bits
  Distance between Person 66 and Person 74: 130 bits
  Distance between Person 66 and Person 75: 131 bits
  Distance between Person 66 and Person 76: 122 bits
  Distance between Person 66 and Person 77: 125 bits
  Distance between Person 66 and Person 78: 116 bits
  Distance between Person 66 and Person 79: 123 bits
  Distance between Person 66 and Person 80: 117 bits
  Distance between Person 66 and Person 81: 120 bits
  Distance between Person 66 and Person 82: 122 bits
  Distance between Person 66 and Person 83: 127 bits
  Distance between Person 66 and Person 84: 143 bits
  Distance between Person 66 and Person 85: 123 bits
  Distance between Person 66 and Person 86: 115 bits
  Distance between Person 66 and Person 87: 131 bits
  Distance between Person 66 and Person 88: 140 bits
  Distance between Person 66 and Person 89: 145 bits
  Distance between Person 67 and Person 68: 146 bits
  Distance between Person 67 and Person 69: 122 bits
  Distance between Person 67 and Person 70: 134 bits
  Distance between Person 67 and Person 71: 132 bits
  Distance between Person 67 and Person 72: 136 bits
  Distance between Person 67 and Person 73: 118 bits
  Distance between Person 67 and Person 74: 133 bits
  Distance between Person 67 and Person 75: 126 bits
  Distance between Person 67 and Person 76: 121 bits
  Distance between Person 67 and Person 77: 128 bits
  Distance between Person 67 and Person 78: 129 bits
  Distance between Person 67 and Person 79: 134 bits
  Distance between Person 67 and Person 80: 118 bits
  Distance between Person 67 and Person 81: 117 bits
  Distance between Person 67 and Person 82: 131 bits
  Distance between Person 67 and Person 83: 146 bits
  Distance between Person 67 and Person 84: 130 bits
  Distance between Person 67 and Person 85: 116 bits
  Distance between Person 67 and Person 86: 134 bits
  Distance between Person 67 and Person 87: 142 bits
  Distance between Person 67 and Person 88: 127 bits
  Distance between Person 67 and Person 89: 138 bits
  Distance between Person 68 and Person 69: 130 bits
  Distance between Person 68 and Person 70: 126 bits
  Distance between Person 68 and Person 71: 132 bits
  Distance between Person 68 and Person 72: 130 bits
  Distance between Person 68 and Person 73: 126 bits
  Distance between Person 68 and Person 74: 131 bits
  Distance between Person 68 and Person 75: 130 bits
  Distance between Person 68 and Person 76: 115 bits
  Distance between Person 68 and Person 77: 116 bits
  Distance between Person 68 and Person 78: 129 bits
  Distance between Person 68 and Person 79: 126 bits
  Distance between Person 68 and Person 80: 136 bits
  Distance between Person 68 and Person 81: 121 bits
  Distance between Person 68 and Person 82: 137 bits
  Distance between Person 68 and Person 83: 128 bits
  Distance between Person 68 and Person 84: 118 bits
  Distance between Person 68 and Person 85: 138 bits
  Distance between Person 68 and Person 86: 134 bits
  Distance between Person 68 and Person 87: 118 bits
  Distance between Person 68 and Person 88: 137 bits
  Distance between Person 68 and Person 89: 136 bits
  Distance between Person 69 and Person 70: 116 bits
  Distance between Person 69 and Person 71: 120 bits
  Distance between Person 69 and Person 72: 118 bits
  Distance between Person 69 and Person 73: 124 bits
  Distance between Person 69 and Person 74: 119 bits
  Distance between Person 69 and Person 75: 106 bits
  Distance between Person 69 and Person 76: 119 bits
  Distance between Person 69 and Person 77: 134 bits
  Distance between Person 69 and Person 78: 107 bits
  Distance between Person 69 and Person 79: 120 bits
  Distance between Person 69 and Person 80: 124 bits
  Distance between Person 69 and Person 81: 135 bits
  Distance between Person 69 and Person 82: 119 bits
  Distance between Person 69 and Person 83: 126 bits
  Distance between Person 69 and Person 84: 146 bits
  Distance between Person 69 and Person 85: 134 bits
  Distance between Person 69 and Person 86: 124 bits
  Distance between Person 69 and Person 87: 118 bits
  Distance between Person 69 and Person 88: 121 bits
  Distance between Person 69 and Person 89: 130 bits
  Distance between Person 70 and Person 71: 138 bits
  Distance between Person 70 and Person 72: 128 bits
  Distance between Person 70 and Person 73: 118 bits
  Distance between Person 70 and Person 74: 121 bits
  Distance between Person 70 and Person 75: 112 bits
  Distance between Person 70 and Person 76: 133 bits
  Distance between Person 70 and Person 77: 118 bits
  Distance between Person 70 and Person 78: 105 bits
  Distance between Person 70 and Person 79: 124 bits
  Distance between Person 70 and Person 80: 132 bits
  Distance between Person 70 and Person 81: 117 bits
  Distance between Person 70 and Person 82: 125 bits
  Distance between Person 70 and Person 83: 128 bits
  Distance between Person 70 and Person 84: 134 bits
  Distance between Person 70 and Person 85: 116 bits
  Distance between Person 70 and Person 86: 120 bits
  Distance between Person 70 and Person 87: 118 bits
  Distance between Person 70 and Person 88: 131 bits
  Distance between Person 70 and Person 89: 130 bits
  Distance between Person 71 and Person 72: 132 bits
  Distance between Person 71 and Person 73: 142 bits
  Distance between Person 71 and Person 74: 137 bits
  Distance between Person 71 and Person 75: 134 bits
  Distance between Person 71 and Person 76: 123 bits
  Distance between Person 71 and Person 77: 122 bits
  Distance between Person 71 and Person 78: 129 bits
  Distance between Person 71 and Person 79: 128 bits
  Distance between Person 71 and Person 80: 122 bits
  Distance between Person 71 and Person 81: 137 bits
  Distance between Person 71 and Person 82: 135 bits
  Distance between Person 71 and Person 83: 120 bits
  Distance between Person 71 and Person 84: 130 bits
  Distance between Person 71 and Person 85: 120 bits
  Distance between Person 71 and Person 86: 132 bits
  Distance between Person 71 and Person 87: 134 bits
  Distance between Person 71 and Person 88: 123 bits
  Distance between Person 71 and Person 89: 116 bits
  Distance between Person 72 and Person 73: 118 bits
  Distance between Person 72 and Person 74: 127 bits
  Distance between Person 72 and Person 75: 126 bits
  Distance between Person 72 and Person 76: 137 bits
  Distance between Person 72 and Person 77: 136 bits
  Distance between Person 72 and Person 78: 131 bits
  Distance between Person 72 and Person 79: 126 bits
  Distance between Person 72 and Person 80: 114 bits
  Distance between Person 72 and Person 81: 129 bits
  Distance between Person 72 and Person 82: 137 bits
  Distance between Person 72 and Person 83: 136 bits
  Distance between Person 72 and Person 84: 130 bits
  Distance between Person 72 and Person 85: 128 bits
  Distance between Person 72 and Person 86: 118 bits
  Distance between Person 72 and Person 87: 108 bits
  Distance between Person 72 and Person 88: 135 bits
  Distance between Person 72 and Person 89: 130 bits
  Distance between Person 73 and Person 74: 135 bits
  Distance between Person 73 and Person 75: 126 bits
  Distance between Person 73 and Person 76: 131 bits
  Distance between Person 73 and Person 77: 152 bits
  Distance between Person 73 and Person 78: 111 bits
  Distance between Person 73 and Person 79: 140 bits
  Distance between Person 73 and Person 80: 72 bits
  Distance between Person 73 and Person 81: 121 bits
  Distance between Person 73 and Person 82: 125 bits
  Distance between Person 73 and Person 83: 130 bits
  Distance between Person 73 and Person 84: 116 bits
  Distance between Person 73 and Person 85: 140 bits
  Distance between Person 73 and Person 86: 120 bits
  Distance between Person 73 and Person 87: 124 bits
  Distance between Person 73 and Person 88: 131 bits
  Distance between Person 73 and Person 89: 114 bits
  Distance between Person 74 and Person 75: 129 bits
  Distance between Person 74 and Person 76: 120 bits
  Distance between Person 74 and Person 77: 139 bits
  Distance between Person 74 and Person 78: 122 bits
  Distance between Person 74 and Person 79: 119 bits
  Distance between Person 74 and Person 80: 123 bits
  Distance between Person 74 and Person 81: 136 bits
  Distance between Person 74 and Person 82: 140 bits
  Distance between Person 74 and Person 83: 125 bits
  Distance between Person 74 and Person 84: 119 bits
  Distance between Person 74 and Person 85: 121 bits
  Distance between Person 74 and Person 86: 127 bits
  Distance between Person 74 and Person 87: 133 bits
  Distance between Person 74 and Person 88: 130 bits
  Distance between Person 74 and Person 89: 133 bits
  Distance between Person 75 and Person 76: 113 bits
  Distance between Person 75 and Person 77: 116 bits
  Distance between Person 75 and Person 78: 129 bits
  Distance between Person 75 and Person 79: 132 bits
  Distance between Person 75 and Person 80: 126 bits
  Distance between Person 75 and Person 81: 125 bits
  Distance between Person 75 and Person 82: 133 bits
  Distance between Person 75 and Person 83: 142 bits
  Distance between Person 75 and Person 84: 142 bits
  Distance between Person 75 and Person 85: 124 bits
  Distance between Person 75 and Person 86: 132 bits
  Distance between Person 75 and Person 87: 114 bits
  Distance between Person 75 and Person 88: 119 bits
  Distance between Person 75 and Person 89: 122 bits
  Distance between Person 76 and Person 77: 123 bits
  Distance between Person 76 and Person 78: 126 bits
  Distance between Person 76 and Person 79: 131 bits
  Distance between Person 76 and Person 80: 131 bits
  Distance between Person 76 and Person 81: 128 bits
  Distance between Person 76 and Person 82: 124 bits
  Distance between Person 76 and Person 83: 147 bits
  Distance between Person 76 and Person 84: 127 bits
  Distance between Person 76 and Person 85: 133 bits
  Distance between Person 76 and Person 86: 147 bits
  Distance between Person 76 and Person 87: 117 bits
  Distance between Person 76 and Person 88: 130 bits
  Distance between Person 76 and Person 89: 121 bits
  Distance between Person 77 and Person 78: 113 bits
  Distance between Person 77 and Person 79: 122 bits
  Distance between Person 77 and Person 80: 136 bits
  Distance between Person 77 and Person 81: 135 bits
  Distance between Person 77 and Person 82: 143 bits
  Distance between Person 77 and Person 83: 130 bits
  Distance between Person 77 and Person 84: 128 bits
  Distance between Person 77 and Person 85: 136 bits
  Distance between Person 77 and Person 86: 120 bits
  Distance between Person 77 and Person 87: 130 bits
  Distance between Person 77 and Person 88: 135 bits
  Distance between Person 77 and Person 89: 140 bits
  Distance between Person 78 and Person 79: 121 bits
  Distance between Person 78 and Person 80: 119 bits
  Distance between Person 78 and Person 81: 106 bits
  Distance between Person 78 and Person 82: 112 bits
  Distance between Person 78 and Person 83: 129 bits
  Distance between Person 78 and Person 84: 139 bits
  Distance between Person 78 and Person 85: 125 bits
  Distance between Person 78 and Person 86: 103 bits
  Distance between Person 78 and Person 87: 119 bits
  Distance between Person 78 and Person 88: 130 bits
  Distance between Person 78 and Person 89: 143 bits
  Distance between Person 79 and Person 80: 146 bits
  Distance between Person 79 and Person 81: 139 bits
  Distance between Person 79 and Person 82: 125 bits
  Distance between Person 79 and Person 83: 120 bits
  Distance between Person 79 and Person 84: 122 bits
  Distance between Person 79 and Person 85: 128 bits
  Distance between Person 79 and Person 86: 136 bits
  Distance between Person 79 and Person 87: 118 bits
  Distance between Person 79 and Person 88: 131 bits
  Distance between Person 79 and Person 89: 122 bits
  Distance between Person 80 and Person 81: 125 bits
  Distance between Person 80 and Person 82: 127 bits
  Distance between Person 80 and Person 83: 142 bits
  Distance between Person 80 and Person 84: 112 bits
  Distance between Person 80 and Person 85: 130 bits
  Distance between Person 80 and Person 86: 112 bits
  Distance between Person 80 and Person 87: 148 bits
  Distance between Person 80 and Person 88: 131 bits
  Distance between Person 80 and Person 89: 144 bits
  Distance between Person 81 and Person 82: 138 bits
  Distance between Person 81 and Person 83: 123 bits
  Distance between Person 81 and Person 84: 135 bits
  Distance between Person 81 and Person 85: 129 bits
  Distance between Person 81 and Person 86: 117 bits
  Distance between Person 81 and Person 87: 141 bits
  Distance between Person 81 and Person 88: 140 bits
  Distance between Person 81 and Person 89: 139 bits
  Distance between Person 82 and Person 83: 117 bits
  Distance between Person 82 and Person 84: 133 bits
  Distance between Person 82 and Person 85: 149 bits
  Distance between Person 82 and Person 86: 121 bits
  Distance between Person 82 and Person 87: 125 bits
  Distance between Person 82 and Person 88: 130 bits
  Distance between Person 82 and Person 89: 115 bits
  Distance between Person 83 and Person 84: 120 bits
  Distance between Person 83 and Person 85: 128 bits
  Distance between Person 83 and Person 86: 130 bits
  Distance between Person 83 and Person 87: 122 bits
  Distance between Person 83 and Person 88: 127 bits
  Distance between Person 83 and Person 89: 128 bits
  Distance between Person 84 and Person 85: 122 bits
  Distance between Person 84 and Person 86: 138 bits
  Distance between Person 84 and Person 87: 126 bits
  Distance between Person 84 and Person 88: 115 bits
  Distance between Person 84 and Person 89: 128 bits
  Distance between Person 85 and Person 86: 132 bits
  Distance between Person 85 and Person 87: 142 bits
  Distance between Person 85 and Person 88: 125 bits
  Distance between Person 85 and Person 89: 142 bits
  Distance between Person 86 and Person 87: 126 bits
  Distance between Person 86 and Person 88: 139 bits
  Distance between Person 86 and Person 89: 126 bits
  Distance between Person 87 and Person 88: 113 bits
  Distance between Person 87 and Person 89: 114 bits
  Distance between Person 88 and Person 89: 123 bits
"""


# -----------------------------
# Extract Intra-person Hamming Distances
print("INTRA-HD with all DATA")
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