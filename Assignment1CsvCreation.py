import re
import pandas as pd

# Provided dataset as an example
data = open("Assignment1Files/2009_03_27.txt")

# Gender mapping based on your list
female_names = {"mara", "amy", "meg", "shelly", "ariel"}
male_names = {"george", "vincent"}

rows = []

for line in data:
    match = re.match(r"(\w+)\s+\([\d: ]+[APM]+\):\s+(.*)", line)
    if match:
        name = match.group(1).lower()
        text = match.group(2)
        if name in female_names:
            gender = 'female'
        elif name in male_names:
            gender = 'male'
        else:
            gender = 'unknown'  # fallback if needed
        rows.append({'name': name, 'text': text, 'gender': gender})

df_new = pd.DataFrame(rows)
df_new.to_csv("Assignment1_2009_93_29.csv", index=False)
print(df_new)
