import pandas as pd

df = pd.read_csv(
    "yourfile.csv",
    sep=';',
    on_bad_lines='skip'
)

df = df.iloc[:, :18]
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['edu'] = pd.to_numeric(df['edu'], errors='coerce')

# Age and edu
print("=" * 50)
print("Descriptive Stats")
print("=" * 50)
print(f"\nage:")
print(f"  Mean: {df['age'].mean():.2f}")
print(f"  SD: {df['age'].std():.2f}")

print(f"\nedu:")
print(f"  Mean: {df['edu'].mean():.2f}")
print(f"  SD: {df['edu'].std():.2f}")

# Sex
print(f"\nsex:")
sex_counts = df['sex'].value_counts()
if 'F' in sex_counts.index:
    print(f"  F: {sex_counts['F']}")
if 'M' in sex_counts.index:
    print(f"  M: {sex_counts['M']}")
print("=" * 50)
