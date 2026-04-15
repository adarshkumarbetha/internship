
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("file.csv")

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())


for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\n--- DATA AFTER CLEANING ---")
print(df.describe())


for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df['year'] = df[col].dt.year
        df['month'] = df[col].dt.month

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

print("\n--- CORRELATION MATRIX ---")
corr = df.corr(numeric_only=True)
print(corr)

sns.set()

for col in numeric_cols:
    plt.figure()
    df[col].hist()
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

plt.figure()
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[numeric_cols])
plt.show()


print("\n--- KEY INSIGHTS ---")

for col in numeric_cols:
    print(f"{col}: Mean = {df[col].mean():.2f}, Median = {df[col].median():.2f}")

print("\nStrong correlations (>0.7):")
for i in corr.columns:
    for j in corr.columns:
        if i != j and abs(corr.loc[i, j]) > 0.7:
            print(f"{i} and {j} → Correlation: {corr.loc[i, j]:.2f}")


df.to_csv("cleaned_dataset.csv", index=False)

print("\n✅ Data Cleaning & Visualization Completed Successfully!")
