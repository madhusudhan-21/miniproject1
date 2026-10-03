import pandas as pd
import matplotlib.pyplot as plt 

# Load the dataset
df = pd.read_csv("data/spam.csv", encoding="latin1")

# Show the first 5 rows
print("----- FIRST 5 ROWS -----")
print(df.head())

# Show dataset size
print("\n----- DATASET SHAPE -----")
print(df.shape)

# Show column names
print("\n----- COLUMN NAMES -----")
print(df.columns)

# Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Check duplicate rows
print("\n----- DUPLICATE ROWS -----")
print(df.duplicated().sum())

# Check label distribution
print("\n----- LABEL DISTRIBUTION -----")
print(df["v1"].value_counts())

# Rename important columns
df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})

# Remove unnecessary columns
df = df[["label", "message"]]

print("\n----- CLEANED DATASET -----")
print(df.head())

print("\n----- CLEANED SHAPE -----")
print(df.shape)

print("\n----- CLEANED COLUMNS -----")
print(df.columns)

# Remove duplicate rows
df = df.drop_duplicates()

print("\n----- AFTER REMOVING DUPLICATES -----")
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_spam.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Message length analysis

df["message_length"] = df["message"].str.len()

print("\n----- MESSAGE LENGTH -----")
print(df["message_length"].describe())

print("\n----- AVERAGE MESSAGE LENGTH BY LABEL -----")
print(df.groupby("label")["message_length"].mean())

# Plot label distribution
df["label"].value_counts().plot(kind="bar")

plt.title("Spam vs Ham Messages")
plt.xlabel("Label")
plt.ylabel("Number of Messages")
plt.show()

# Plot message length distribution
df["message_length"].plot(kind="hist", bins=30)

plt.title("Message Length Distribution")
plt.xlabel("Message Length")
plt.ylabel("Number of Messages")
plt.show()