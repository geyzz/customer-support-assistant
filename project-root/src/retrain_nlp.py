import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
import evaluate
import os

# PATHS
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "../output")
MODEL_PATH = os.path.join(BASE, "../experiments/results/fine_tuned_model")
RESULTS_PATH = os.path.join(BASE, "../experiments/results/checkpoints")
FEEDBACK_PATH = os.path.join(BASE, "../feedback_data.csv")

# LOAD ORIGINAL DATA
train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(OUTPUT_DIR, "validation.csv"))
test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test.csv"))

# LOAD FEEDBACK DATA
if os.path.exists(FEEDBACK_PATH):
    feedback_df = pd.read_csv(FEEDBACK_PATH)
    print(f"Loaded {len(feedback_df)} feedback samples")

    # Ensure correct column names
    if "text" not in feedback_df.columns or "intent" not in feedback_df.columns:
        feedback_df.columns = ["text", "intent"]

    # Merge into training set only
    train_df = pd.concat([train_df, feedback_df]).drop_duplicates()
else:
    print("No feedback data found")

# CLEAN DATA (VERY IMPORTANT)
def clean_df(df):
    df = df.dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str)
    return df

train_df = clean_df(train_df)
val_df = clean_df(val_df)
test_df = clean_df(test_df)

print("Training samples:", len(train_df))

# CONVERT TO HF DATASET
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

dataset = dataset.class_encode_column("intent")

# TOKENIZER
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

def preprocess_function(examples):
    texts = [str(t) for t in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("intent", "labels")
raw_test_texts = dataset["test"]["text"]

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# MODEL
num_labels = len(dataset["train"].features["intent"].names)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=num_labels
)

# METRICS
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score_val = f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "macro_f1": f1_score_val["f1"]
    }

# TRAINING CONFIG
training_args = TrainingArguments(
    output_dir=RESULTS_PATH,
    eval_strategy="epoch",      # evaluate at the end of each epoch
    save_strategy="epoch",            # save checkpoint each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="tensorboard",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1"
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# TRAIN
trainer.train()

# EVALUATE
predictions = trainer.predict(tokenized_dataset["test"])
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = tokenized_dataset["test"]["labels"]

label_names = dataset["train"].features["intent"].names

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=label_names))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
print("\nMacro-F1:", macro_f1)

# SAVE MODEL
model.config.id2label = {i: label for i, label in enumerate(label_names)}
model.config.label2id = {label: i for i, label in enumerate(label_names)}

model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("\nRetraining complete. Model updated.")
