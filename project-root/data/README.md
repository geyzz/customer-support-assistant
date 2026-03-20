# Customer Support Dataset

A data preprocessing pipeline for cleaning, normalizing, and splitting the Bitext Sample Customer Support Training Dataset in preparation for intent classification model training.

---

## Overview

This project provides two equivalent implementations of the same preprocessing pipeline:

| File | Description |
|------|-------------|
|`01_eda.ipynb` | Jupyter Notebook with step-by-step descriptions and inline output |
|`get_data.py` | Standalone Python script for running the full pipeline from the command line |

Both produce identical outputs and follow the same 12-step pipeline.

---

## Dataset

**Source:** [Bitext Sample Customer Support Training Dataset](https://github.com/bitext/customer-support-llm-chatbot-training-dataset)  
**File:** `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`  
**Size:** ~27,000 labeled customer support responses  
**Columns used:** `instruction` (renamed to `text`), `intent`

Place the dataset file at the following path before running:

```
dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

---

## Requirements

- Python 3.8+
- pandas
- scikit-learn

Install dependencies:

```bash
pip install pandas scikit-learn
```

For the notebook:

```bash
pip install jupyter
```

---

## Project Structure

```
├── dataset/
│   └── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
├── output/
│   ├── cleaned_dataset.csv   # Full cleaned dataset before splitting
│   ├── train.csv             # 80% training split
│   ├── validation.csv        # 10% validation split
│   └── test.csv              # 10% test split
├── get_data.py
├── 01_eda.ipynb
└── README.md
```

The `output/` directory is created automatically when the pipeline runs.

---

## Usage

**Run the script:**

```bash
python cleanData.py
```

**Run the notebook:**

```bash
jupyter notebook customer_support_cleaning.ipynb
```

Then run all cells from top to bottom (`Kernel > Restart & Run All`).

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load the raw CSV and select `text` and `intent` columns 
| 2 | Drop rows with missing values 
| 3 | Normalize text — lowercase, collapse whitespace, remove special characters (preserves `?` and `!`) 
| 4 | Remove duplicate `(text, intent)` pairs 
| 5 | Filter entries by word count (keep: 1–128 tokens) 
| 6 | Inspect intent class distribution 
| 7 | Remove intent classes with only 1 sample (required for stratified splitting) 
| 8 | Final duplicate safety check 
| 9 | Save master cleaned dataset to `output/cleaned_dataset.csv` 
| 10 | Stratified 80/10/10 train / validation / test split 
| 11 | Save splits to `output/` 
| 12 | Final sanity check — sample preview and null value audit 

---

## Output Files

| File | Split | Approx. Size |
|------|-------|--------------|
| `cleaned_dataset.csv` | Full cleaned data | ~27K rows |
| `train.csv` | 80% | ~21.6K rows |
| `validation.csv` | 10% | ~2.7K rows |
| `test.csv` | 10% | ~2.7K rows |

> Actual row counts may vary slightly due to duplicate removal and rare intent filtering.

All splits are **stratified** — each preserves the same intent class distribution as the original cleaned dataset.

---

## Text Cleaning Rules

The `clean_text()` function applies the following transformations in order:

1. Lowercase and strip surrounding whitespace
2. Collapse consecutive whitespace into a single space
3. Remove all characters except `a–z`, `0–9`, spaces, `?`, and `!`

**Rationale:** `?` and `!` are preserved because they carry signal for intent detection (e.g., questions vs. complaints). All other punctuation and special characters are removed to reduce noise.

---

## Split Strategy

Splits are generated using `sklearn.model_selection.train_test_split` with:
- `stratify=intent` — ensures proportional class representation in each split
- `random_state=42` — guarantees reproducibility across runs

The 20% held-out portion is split evenly (50/50) into validation and test sets, yielding a final 80/10/10 ratio.
