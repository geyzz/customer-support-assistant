import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 1. LOAD DATASET
file_path = "dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
df = pd.read_csv(file_path)

# Keep only needed columns
df = df[["instruction", "intent"]]
df = df.rename(columns={
    "instruction": "text",
    "intent": "intent"
})

print("Original Dataset Shape:", df.shape)

# 2. REMOVE NULLS & DUPLICATES
df = df.dropna()
df = df.drop_duplicates()

print("After Cleaning (NA & duplicates removed):", df.shape)

# 3. TEXT CLEANING FUNCTION
def clean_text(text):
    text = text.lower()                      # lowercase
    text = text.strip()                      # remove spaces
    text = re.sub(r'\s+', ' ', text)         # remove extra spaces
    
    # keep useful punctuation for intent detection
    text = re.sub(r'[^a-z0-9\s\?\!]', '', text)
    
    return text

df["text"] = df["text"].apply(clean_text)

# Clean intent column
df["intent"] = df["intent"].str.lower().str.strip()

# 4. TEXT LENGTH FILTER
df["length"] = df["text"].apply(lambda x: len(x.split()))

print("\nText Length Stats:")
print(df["length"].describe())

# Keep only reasonable inputs
df = df[df["length"] <= 128]

# 5. CHECK INTENT DISTRIBUTION
print("\nIntent Distribution:")
print(df["intent"].value_counts())

# 6. SAVE CLEANED DATASET
df.to_csv("cleaned_dataset.csv", index=False)

# 7. TRAIN / VALIDATION / TEST SPLIT
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

# 8. SAVE SPLITS
train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

# 9. FINAL CHECK
print("\nSample Data:")
print(df.head())

print("\nMissing Values Check:")
print(df.isnull().sum())
