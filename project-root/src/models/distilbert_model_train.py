from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score   
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer,TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import evaluate
import os

# PATHS
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "../../output")
MODEL_PATH = os.path.join(BASE, "../../experiments/results/fine_tuned_model")
RESULTS_PATH = os.path.join(BASE, "../../experiments/results/checkpoints")

# Load dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": os.path.join(OUTPUT_DIR, "train.csv"),
        "validation": os.path.join(OUTPUT_DIR, "validation.csv"),
        "test": os.path.join(OUTPUT_DIR, "test.csv")
    }
)

dataset = dataset.class_encode_column("intent")
# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocessing
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("intent", "labels")
raw_test_texts = dataset["test"]["text"]
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

num_labels = len(dataset["train"].features["intent"].names)
# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels= num_labels
)
    
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "macro_f1": f1_score["f1"]
    }
  
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
# Start training
trainer.train()

# Get predictions
predictions = trainer.predict(tokenized_dataset['test'])
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = tokenized_dataset['test']['labels']

# Generate a classification report

label_names = dataset["train"].features["intent"].names

print(classification_report(true_labels, predicted_labels, target_names=label_names))
print(confusion_matrix(true_labels, predicted_labels))

macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
print("Macro-F1:", macro_f1)

# Inspect misclassified examples
for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
    if true != pred:
        print(f"Example {i}:")
        print(f"Text: {raw_test_texts[i]}")
        print(f"True Label: {label_names[true]}, Predicted Label: {label_names[pred]}")

model.config.id2label = {i: label for i, label in enumerate(label_names)}
model.config.label2id = {label: i for i, label in enumerate(label_names)}
       
# Save the model and tokenizer
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
