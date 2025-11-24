'''
This script removes the trailing whitespaces found in Application_Data_Original.csv.

Usage: python util/format_csv.py
'''

import pandas as pd

df = pd.read_csv('data/Application_Data_Original.csv')

# Remove trailing whitespace in rows
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

df.to_csv('data/Application_Data_Cleaned.csv', index=False)