# Load the CSV and calculate the sum of the last column

import pandas as pd

# Load the CSV file
file_path = "Generated_Instances/generated_instances.csv"
df = pd.read_csv(file_path, header=None)

# Sum the last column
last_column_sum = df.iloc[:, -1].sum()
print(last_column_sum/60)
