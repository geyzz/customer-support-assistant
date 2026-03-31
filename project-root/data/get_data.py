import os
from datasets import load_dataset

# Output path
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

def get_data():
    # Create data/ folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(OUTPUT_FILE):
        print(f"✓ Data already exists at {OUTPUT_FILE} — skipping download.")
        return

    print("Downloading Bitext dataset from HuggingFace...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    # Convert to pandas and save as CSV
    df = dataset["train"].to_pandas()
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✓ Data saved to {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

if __name__ == "__main__":
    get_data()

