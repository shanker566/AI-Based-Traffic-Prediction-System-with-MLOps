import pandas as pd

def inspect():
    df = pd.read_csv('data/raw/traffic.csv')
    print("=== DATASET INSPECTION ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nLast 5 Rows:")
    print(df.tail())
    print("\nUnique Junctions:")
    print(df['Junction'].value_counts())
    print("\nDateTime range:")
    print(f"Min: {df['DateTime'].min()}, Max: {df['DateTime'].max()}")
    print("\nDescriptive Statistics:")
    print(df.describe())

if __name__ == '__main__':
    inspect()
