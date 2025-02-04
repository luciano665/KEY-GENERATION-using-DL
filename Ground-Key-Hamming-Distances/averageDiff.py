import pandas as pd

#Load CSV file
file_path = '/Ground-Key-Hamming-Distances/h_distance_mixed.csv'
data = pd.read_csv(file_path)


percentage_diff = data.columns[3]
average_value = data[percentage_diff].mean()

print(f"The average value is: {average_value}")
