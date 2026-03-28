import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 1. LOAD DATASET
file_path = "project-root/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

df = pd.read_csv(file_path)

# Keep only needed columns
df = df[["instruction", "intent"]].copy()
df.rename(columns={
    "instruction": "text",
    "intent": "intent"
}, inplace=True)

print("Original Dataset Shape:", df.shape)

# 2. REMOVE NULLS
df.dropna(inplace=True)
print("After removing NULLs:", df.shape)

# 3. TEXT CLEANING FUNCTION
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    
    # keep useful punctuation for intent detection
    text = re.sub(r'[^a-z0-9\s\?\!]', '', text)
    
    return text

# Apply cleaning
df["text"] = df["text"].apply(clean_text)
df["intent"] = df["intent"].astype(str).str.lower().str.strip()

# 4. REMOVE DUPLICATES (AFTER CLEANING)
before = len(df)
df.drop_duplicates(subset=["text", "intent"], inplace=True)
after = len(df)

print(f"Removed {before - after} duplicate rows")
print("After duplicate removal:", df.shape)

# 5. TEXT LENGTH FILTER
df["length"] = df["text"].apply(lambda x: len(x.split()))

print("\nText Length Stats:")
print(df["length"].describe())

# Keep only reasonable inputs
df = df[df["length"] > 0]        # remove empty text
df = df[df["length"] <= 128]     # limit max length

print("After length filtering:", df.shape)

# 6. CHECK INTENT DISTRIBUTION
print("\nIntent Distribution:")
print(df["intent"].value_counts())

# 7. REMOVE RARE INTENTS (IMPORTANT FOR STRATIFY)
intent_counts = df["intent"].value_counts()
valid_intents = intent_counts[intent_counts > 1].index
df = df[df["intent"].isin(valid_intents)]

print("After removing rare intents:", df.shape)

# 8. FINAL DUPLICATE CHECK (SAFETY)
duplicates = df[df.duplicated(subset=["text", "intent"], keep=False)]
print("Remaining duplicate rows:", len(duplicates))

# 9. SAVE CLEANED DATASET
os.makedirs("output", exist_ok=True)
df.to_csv("output/cleaned_dataset.csv", index=False)

# 10. TRAIN / VALIDATION / TEST SPLIT
train, temp = train_test_split(
    df,
    test_size=0.2,
    stratify=df["intent"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp["intent"],
    random_state=42
)

print("\nDataset Split Sizes:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

# 11. SAVE SPLITS
train.to_csv("output/train.csv", index=False)
val.to_csv("output/validation.csv", index=False)
test.to_csv("output/test.csv", index=False)

# 12. FINAL CHECK
print("\nSample Data:")
print(df.head())

print("\nMissing Values Check:")
print(df.isnull().sum())
