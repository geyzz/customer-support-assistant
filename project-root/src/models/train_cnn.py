import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = "output/train.csv"
VAL_PATH   = "output/validation.csv"
TEST_PATH  = "output/test.csv"
GLOVE_PATH = "data/embeddings/glove.6B.300d.txt"
SAVE_DIR   = "cnn_snapshot"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_train = le.fit_transform(train_df["intent"])
y_val   = le.transform(val_df["intent"])
y_test  = le.transform(test_df["intent"])

# -----------------------------
# Load GloVe
# -----------------------------
EMBEDDING_DIM = 300
MAX_LEN = 20

print("Loading GloVe embeddings...")
glove = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        values = line.split()
        glove[values[0]] = np.asarray(values[1:], dtype=np.float32)
print(f"GloVe loaded: {len(glove)} words")

# -----------------------------
# Encode text
# -----------------------------
def encode_text(df):
    X = []
    for text in df["text"]:
        tokens = text.lower().split()[:MAX_LEN]
        mat = np.zeros((MAX_LEN, EMBEDDING_DIM), dtype=np.float32)
        for i, token in enumerate(tokens):
            mat[i] = glove.get(token, np.random.normal(scale=0.6, size=(EMBEDDING_DIM,)))
        X.append(mat)
    return np.array(X)

X_train = encode_text(train_df)
X_val   = encode_text(val_df)
X_test  = encode_text(test_df)

# -----------------------------
# Dataset
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader   = DataLoader(TextDataset(X_val, y_val), batch_size=16)
test_loader  = DataLoader(TextDataset(X_test, y_test), batch_size=16)

# -----------------------------
# Text-CNN
# -----------------------------
class TextCNN(nn.Module):
    def __init__(self, embed_dim, num_classes, filter_sizes, num_filters=50, fine_tune=False):
        super().__init__()
        self.fine_tune = fine_tune
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        if not self.fine_tune:
            x = x.detach()
        x = x.unsqueeze(1)
        convs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [torch.max(c, dim=2)[0] for c in convs]
        out = torch.cat(pools, dim=1)
        out = self.dropout(out)
        return self.fc(out)

# -----------------------------
# Ablations
# -----------------------------
ablation_configs = [
    {"name": "static_2_3_4", "fine_tune": False, "filter_sizes": [2,3,4]},
    {"name": "fine_tuned_2_3_4", "fine_tune": True, "filter_sizes": [2,3,4]},
    {"name": "static_3_4_5", "fine_tune": False, "filter_sizes": [3,4,5]},
    {"name": "fine_tuned_3_4_5", "fine_tune": True, "filter_sizes": [3,4,5]},
]

EPOCHS = 50
PATIENCE = 2
all_metrics = []
best_model_path = None

# -----------------------------
# Training
# -----------------------------
for config in ablation_configs:
    print(f"\n=== Training {config['name']} ===")
    model = TextCNN(
        EMBEDDING_DIM,
        len(le.classes_),
        filter_sizes=config["filter_sizes"],
        fine_tune=config["fine_tune"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -----------------------------
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = torch.argmax(model(X_batch), dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = accuracy_score(val_labels, val_preds)

        # Log metrics
        all_metrics.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "config": config["name"]
        })

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_F1={val_f1:.4f}")

        # -----------------------------
        # Early stopping & save
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wait = 0
            best_model_path = os.path.join(SAVE_DIR, f"textcnn_{config['name']}.pt")
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping")
                break

# -----------------------------
# Save metrics CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(SAVE_DIR, "textcnn_ablation_metrics.csv"), index=False)

# -----------------------------
# Plot metrics per config
plt.figure(figsize=(10,5))
for cfg in ablation_configs:
    df_cfg = metrics_df[metrics_df["config"]==cfg["name"]]
    plt.plot(df_cfg["epoch"], df_cfg["val_macro_f1"], label=cfg["name"])
plt.xlabel("Epoch")
plt.ylabel("Validation Macro-F1")
plt.title("Text-CNN Ablation: Validation Macro-F1 per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "textcnn_ablation_macroF1.png"))
plt.show()

# -----------------------------
# Load best model & test
assert best_model_path is not None
print(f"\nLoading best model: {best_model_path}")
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = torch.argmax(model(X_batch), dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(y_batch.cpu().numpy())

test_f1 = f1_score(test_labels, test_preds, average="macro")
print("\nTest Macro-F1:", test_f1)
print(classification_report(test_labels, test_preds, target_names=le.classes_))

# -----------------------------
# Interactive inference
def encode_single(text):
    tokens = text.lower().split()[:MAX_LEN]
    mat = np.zeros((MAX_LEN, EMBEDDING_DIM), dtype=np.float32)
    for i, token in enumerate(tokens):
        mat[i] = glove.get(token, np.random.normal(scale=0.6, size=(EMBEDDING_DIM,)))
    return torch.tensor(mat).unsqueeze(0)

print("\nInteractive test (type 'quit')")
while True:
    msg = input("Enter message: ")
    if msg.lower() == "quit":
        break
    with torch.no_grad():
        probs = torch.softmax(model(encode_single(msg)), dim=1)
        pred = torch.argmax(probs).item()
    print("Predicted intent:", le.inverse_transform([pred])[0])
    print(f"Confidence: {probs[0, pred]:.2f}\n")
