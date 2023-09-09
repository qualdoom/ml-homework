import pandas as pd

# Sample DataFrame
data = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}
df = pd.DataFrame(data)

# Create a 2D boolean array with the same number of rows as the DataFrame
bool_array = [[True, False], [False, True], [True, True], [False, False]]

# Flatten the boolean array and use it to filter rows in the DataFrame
bool_flattened = [any(sublist) for sublist in bool_array]
filtered_df = df.loc[bool_flattened]

# Display the filtered DataFrame
print(filtered_df)
