import pandas as pd

# Paths to your two CSV files
csv_file1 = 'Assignment1_2009_93_27.csv'
csv_file2 = 'Assignment1_2009_93_29.csv'

# Output path for combined file
output_file = 'Assignment1Combined.csv'

# Load both CSV files
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Combine (stack them vertically)
df_combined = pd.concat([df1, df2], ignore_index=True)

# Optional: Drop duplicates if needed
# df_combined = df_combined.drop_duplicates()

# Save to new CSV
df_combined.to_csv(output_file, index=False)

print(f"Files combined and saved to {output_file}")
