"""
eval.py — Customer Support Assistant
Evaluation script covering:
  - DistilBERT (NLP): Accuracy, Macro-F1, Confusion Matrix, Per-class F1, Slice Analysis
  - TextCNN (CNN): Accuracy, Macro-F1, Confusion Matrix, Ablation Curves
  - RL Agent (LinUCB): Cumulative Reward, Regret, Success Rate vs Random Baseline
  - Baseline: TF-IDF + Logistic Regression (non-DL)
  - All results shown via CLI + matplotlib GUI (no files saved to disk)

Run from project-root/:
    python src/eval.py
"""

import os, sys, warnings, random, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.join(BASE, "..")
MODEL_PATH  = os.path.join(ROOT, "experiments/results/fine_tuned_model")
AGENT_PATH  = os.path.join(ROOT, "experiments/results/linucb_agent.pkl")
CNN_DIR     = os.path.join(ROOT, "cnn_snapshot")
TRAIN_PATH  = os.path.join(ROOT, "output/train.csv")
TEST_PATH   = os.path.join(ROOT, "output/test.csv")
GLOVE_PATH  = os.path.join(ROOT, "data/embeddings/glove.6B.300d.txt")

sys.path.insert(0, BASE)
from responses import responses
ACTIONS = list(responses.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP  = "=" * 60
SEP2 = "-" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── Load Data ─────────────────────────────────────────────────────────────────
section("Loading Data")
train_df        = pd.read_csv(TRAIN_PATH)
test_df         = pd.read_csv(TEST_PATH)
test_texts      = test_df["text"].tolist()
test_labels_str = test_df["intent"].tolist()

le = LabelEncoder()
le.fit(train_df["intent"])
label_names     = list(le.classes_)
NUM_CLASSES     = len(label_names)
test_labels_int = le.transform(test_labels_str)

print(f"  Test samples : {len(test_df)}")
print(f"  Classes      : {NUM_CLASSES}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE — TF-IDF + Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
section("Baseline — TF-IDF + Logistic Regression (non-DL)")

tfidf      = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tf = tfidf.fit_transform(train_df["text"])
X_test_tf  = tfidf.transform(test_texts)

lr = LogisticRegression(max_iter=1000, random_state=42, C=5.0)
lr.fit(X_train_tf, train_df["intent"])
lr_preds = lr.predict(X_test_tf)

lr_acc = accuracy_score(test_labels_str, lr_preds)
lr_f1  = f1_score(test_labels_str, lr_preds, average="macro")

print(f"  Accuracy  : {lr_acc:.4f}")
print(f"  Macro-F1  : {lr_f1:.4f}")
print(f"\n{classification_report(test_labels_str, lr_preds, target_names=label_names)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DistilBERT — NLP Evaluation
# ══════════════════════════════════════════════════════════════════════════════
section("DistilBERT — NLP + RL Model Evaluation")

db_available  = os.path.isdir(MODEL_PATH)
db_preds_str  = None
db_preds_int  = None
db_confs      = None
db_embeddings = None
db_acc = db_f1 = short_f1 = long_f1 = 0.0

if db_available:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        DistilBertModel
    )

    tokenizer   = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    clf_model   = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    embed_model = DistilBertModel.from_pretrained(MODEL_PATH)
    clf_model.eval()
    embed_model.eval()

    id2label     = clf_model.config.id2label
    db_preds_str = []
    db_confs_list = []
    db_embs      = []

    print("  Running inference on test set...")
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, padding=True, max_length=128)

            clf_out  = clf_model(**inputs)
            probs    = F.softmax(clf_out.logits, dim=1)
            conf, pred_id = torch.max(probs, dim=1)
            db_preds_str.append(id2label[pred_id.item()])
            db_confs_list.append(conf.item())

            emb_out = embed_model(**inputs)
            cls_vec = emb_out.last_hidden_state[:, 0, :].squeeze().numpy()
            cls_vec = cls_vec / (np.linalg.norm(cls_vec) + 1e-8)
            db_embs.append(np.append(cls_vec, conf.item()))

    db_confs      = np.array(db_confs_list)
    db_embeddings = np.array(db_embs)
    db_preds_int  = np.array([
        le.transform([p])[0] if p in le.classes_ else -1
        for p in db_preds_str
    ])

    db_acc = accuracy_score(test_labels_int, db_preds_int)
    db_f1  = f1_score(test_labels_int, db_preds_int, average="macro")

    print(f"  Accuracy              : {db_acc:.4f}")
    print(f"  Macro-F1              : {db_f1:.4f}")
    print(f"  Low-conf (<0.70)      : {int((db_confs < 0.70).sum())} flagged")
    print(f"\n{classification_report(test_labels_int, db_preds_int, target_names=label_names)}")

    # Slice analysis
    lengths    = np.array([len(t.split()) for t in test_texts])
    short_mask = lengths <= 8
    long_mask  = ~short_mask
    short_f1   = f1_score(test_labels_int[short_mask], db_preds_int[short_mask],
                          average="macro", zero_division=0)
    long_f1    = f1_score(test_labels_int[long_mask],  db_preds_int[long_mask],
                          average="macro", zero_division=0)

    print(SEP2)
    print("  Slice Analysis (by utterance length):")
    print(f"    Short (<=8 words) n={short_mask.sum():<5} Macro-F1: {short_f1:.4f}")
    print(f"    Long   (>8 words) n={long_mask.sum():<5} Macro-F1: {long_f1:.4f}")

    # Failure cases
    print(SEP2)
    print("  Failure Cases (first 5 misclassified):")
    count = 0
    for i, (true, pred) in enumerate(zip(test_labels_int, db_preds_int)):
        if true != pred and count < 5:
            print(f"    [{count+1}] \"{test_texts[i]}\"")
            print(f"         True: {label_names[true]}  |  Pred: {label_names[pred]}  |  Conf: {db_confs[i]:.2f}")
            count += 1
else:
    print("  [SKIP] fine_tuned_model/ not found.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TextCNN — CNN Evaluation + Ablations
# ══════════════════════════════════════════════════════════════════════════════
section("TextCNN — CNN Ablation Evaluation")

EMBEDDING_DIM = 300
MAX_LEN       = 20
CNN_ABLATIONS = [
    {"name": "static_2_3_4",     "filter_sizes": [2,3,4], "fine_tune": False},
    {"name": "fine_tuned_2_3_4", "filter_sizes": [2,3,4], "fine_tune": True},
    {"name": "static_3_4_5",     "filter_sizes": [3,4,5], "fine_tune": False},
    {"name": "fine_tuned_3_4_5", "filter_sizes": [3,4,5], "fine_tune": True},
]

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
        return self.fc(self.dropout(torch.cat(pools, dim=1)))

glove_available = os.path.isfile(GLOVE_PATH)
cnn_results     = {}
best_cnn_f1     = -1
best_cnn_preds  = None
best_cnn_name   = None

if glove_available:
    print("  Loading GloVe...")
    glove = {}
    with open(GLOVE_PATH, encoding="utf8") as f:
        for line in f:
            vals = line.split()
            glove[vals[0]] = np.asarray(vals[1:], dtype=np.float32)
    print(f"  GloVe loaded: {len(glove)} words")

    def encode_texts(texts):
        X = []
        for text in texts:
            tokens = text.lower().split()[:MAX_LEN]
            mat    = np.zeros((MAX_LEN, EMBEDDING_DIM), dtype=np.float32)
            for i, tok in enumerate(tokens):
                mat[i] = glove.get(tok, np.zeros(EMBEDDING_DIM))
            X.append(mat)
        return torch.tensor(np.array(X), dtype=torch.float32)

    X_test_cnn = encode_texts(test_texts)

    for cfg in CNN_ABLATIONS:
        CNN_DIR = os.path.join(BASE, "..", "experiments", "results", "cnn_snapshot")
        ckpt = os.path.join(CNN_DIR, f"textcnn_{cfg['name']}.pt")
        if not os.path.isfile(ckpt):
            print(f"  [SKIP] {cfg['name']} — checkpoint not found")
            continue

        model = TextCNN(EMBEDDING_DIM, NUM_CLASSES, cfg["filter_sizes"],
                        fine_tune=cfg["fine_tune"])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            preds = torch.argmax(model(X_test_cnn), dim=1).numpy()

        acc = accuracy_score(test_labels_int, preds)
        f1  = f1_score(test_labels_int, preds, average="macro")
        cnn_results[cfg["name"]] = {"accuracy": acc, "macro_f1": f1, "preds": preds}

        print(f"  {cfg['name']:25s} — Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")

        if f1 > best_cnn_f1:
            best_cnn_f1   = f1
            best_cnn_preds = preds
            best_cnn_name  = cfg["name"]

    if best_cnn_name:
        print(f"\n  Best CNN: {best_cnn_name} (Macro-F1: {best_cnn_f1:.4f})")
        print(f"\n{classification_report(test_labels_int, best_cnn_preds, target_names=label_names)}")
else:
    print("  [SKIP] GloVe not found.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RL Agent — Offline Simulation
# ══════════════════════════════════════════════════════════════════════════════
section("RL Agent — Offline Simulation (LinUCB)")

rl_available  = os.path.isfile(AGENT_PATH) and db_preds_str is not None
all_rewards   = []
all_regrets   = []
success_rate  = pct_gain = 0.0
rand_rw       = []

if rl_available:
    from rl_agent import LinUCB

    with open(AGENT_PATH, "rb") as f:
        agent = pickle.load(f)

    seeds = [42, 123, 999]

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        cumr = cumg = 0
        rewards, regrets = [], []

        for i in range(len(test_texts)):
            pred_str  = db_preds_str[i]
            conf      = db_confs[i]
            ctx       = db_embeddings[i]
            true_str  = test_labels_str[i]
            pred_idx  = ACTIONS.index(pred_str) if pred_str in ACTIONS else 0

            if conf >= 0.70:
                action_idx = pred_idx
            else:
                ctx = np.array(ctx).reshape(-1, 1)
                action_idx = agent.select_action(ctx, pred_idx, conf)

            action_str = ACTIONS[action_idx]
            reward     = 1.0 if action_str == true_str else 0.0
            cumr += reward
            cumg += (1.0 - reward)
            rewards.append(cumr)
            regrets.append(cumg)

        all_rewards.append(rewards)
        all_regrets.append(regrets)

    # Random baseline
    random.seed(42)
    total_rand = 0
    for t in test_labels_str:
        total_rand += 1.0 if random.choice(ACTIONS) == t else 0.0
        rand_rw.append(total_rand)

    rl_mean  = np.mean(all_rewards, axis=0)
    rl_std   = np.std(all_rewards,  axis=0)
    final_rl = float(rl_mean[-1])
    final_rd = float(rand_rw[-1])
    success_rate = (final_rl / len(test_texts)) * 100
    pct_gain     = ((final_rl - final_rd) / max(final_rd, 1)) * 100

    print(f"  Steps evaluated     : {len(test_texts)}")
    print(f"  RL final reward     : {final_rl:.1f}")
    print(f"  Random baseline     : {final_rd:.1f}")
    print(f"  Success rate        : {success_rate:.1f}%")
    print(f"  Gain vs random      : +{pct_gain:.1f}%")
else:
    print("  [SKIP] Agent or DistilBERT not available.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLE (CLI)
# ══════════════════════════════════════════════════════════════════════════════
section("Summary Table")
print(f"  {'Model':<38} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"  {'-'*60}")
print(f"  {'TF-IDF + Logistic Regression':<38} {lr_acc:>10.4f} {lr_f1:>10.4f}")
if db_available:
    print(f"  {'DistilBERT (NLP+RL)':<38} {db_acc:>10.4f} {db_f1:>10.4f}")
for name, res in cnn_results.items():
    print(f"  {f'TextCNN ({name})':<38} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f}")
if rl_available:
    print(f"\n  RL Success Rate : {success_rate:.1f}%  |  Improvement vs Random: +{pct_gain:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PLOTS (all shown, none saved)
# ══════════════════════════════════════════════════════════════════════════════
section("Generating Plots")
plt.rcParams.update({"figure.facecolor": "#1e1e2e",
                     "axes.facecolor":   "#2a2a3d",
                     "axes.edgecolor":   "#45475a",
                     "axes.labelcolor":  "#cdd6f4",
                     "xtick.color":      "#cdd6f4",
                     "ytick.color":      "#cdd6f4",
                     "text.color":       "#cdd6f4",
                     "grid.color":       "#45475a",
                     "legend.facecolor": "#313244"})

# ── Plot 1: Confusion Matrices ─────────────────────────────────────────────
n_cm = sum([db_available, best_cnn_preds is not None])
if n_cm > 0:
    fig, axes = plt.subplots(1, n_cm, figsize=(11 * n_cm, 10))
    if n_cm == 1:
        axes = [axes]
    idx = 0
    if db_available:
        cm = confusion_matrix(test_labels_int, db_preds_int)
        sns.heatmap(cm, ax=axes[idx], cmap="Blues", fmt="d", annot=False,
                    xticklabels=label_names, yticklabels=label_names, linewidths=0.2)
        axes[idx].set_title("DistilBERT — Confusion Matrix", fontsize=12, pad=8)
        axes[idx].set_xlabel("Predicted", fontsize=9)
        axes[idx].set_ylabel("True", fontsize=9)
        axes[idx].tick_params(axis="x", rotation=45, labelsize=6)
        axes[idx].tick_params(axis="y", rotation=0,  labelsize=6)
        idx += 1
    if best_cnn_preds is not None:
        cm2 = confusion_matrix(test_labels_int, best_cnn_preds)
        sns.heatmap(cm2, ax=axes[idx], cmap="Greens", fmt="d", annot=False,
                    xticklabels=label_names, yticklabels=label_names, linewidths=0.2)
        axes[idx].set_title(f"TextCNN ({best_cnn_name}) — Confusion Matrix", fontsize=12, pad=8)
        axes[idx].set_xlabel("Predicted", fontsize=9)
        axes[idx].set_ylabel("True", fontsize=9)
        axes[idx].tick_params(axis="x", rotation=45, labelsize=6)
        axes[idx].tick_params(axis="y", rotation=0,  labelsize=6)
    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ── Plot 2: Per-Class F1 Comparison ──────────────────────────────────────────
if db_available and best_cnn_preds is not None:
    db_rep  = classification_report(test_labels_int, db_preds_int,
                                    target_names=label_names, output_dict=True)
    cnn_rep = classification_report(test_labels_int, best_cnn_preds,
                                    target_names=label_names, output_dict=True)
    lr_rep  = classification_report(test_labels_str, lr_preds,
                                    target_names=label_names, output_dict=True)

    db_f1s  = [db_rep[l]["f1-score"]  for l in label_names]
    cnn_f1s = [cnn_rep[l]["f1-score"] for l in label_names]
    lr_f1s  = [lr_rep[l]["f1-score"]  for l in label_names]

    x, w = np.arange(len(label_names)), 0.27
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - w, lr_f1s,  w, label="TF-IDF + LR", color="#f38ba8", alpha=0.9)
    ax.bar(x,     cnn_f1s, w, label=f"TextCNN ({best_cnn_name})", color="#a6e3a1", alpha=0.9)
    ax.bar(x + w, db_f1s,  w, label="DistilBERT", color="#89b4fa", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Class F1 — All Models", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ── Plot 3: Confidence Distribution + Slice Analysis ─────────────────────────
if db_available:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(db_confs, bins=30, color="#89b4fa", edgecolor="#1e1e2e")
    axes[0].axvline(0.70, color="#f38ba8", linestyle="--", lw=1.8, label="Threshold = 0.70")
    axes[0].set_title("DistilBERT — Confidence Distribution")
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    bars = axes[1].bar(["Short (≤8 words)", "Long (>8 words)"],
                       [short_f1, long_f1],
                       color=["#f9e2af", "#89b4fa"], width=0.4, edgecolor="#1e1e2e")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("DistilBERT — Slice Analysis (Macro-F1)")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{bar.get_height():.4f}", ha="center", fontsize=12, fontweight="bold")

    plt.suptitle("DistilBERT — Confidence & Slice Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ── Plot 4: CNN Ablation Curves ───────────────────────────────────────────────
ablation_csv = os.path.join(CNN_DIR, "textcnn_ablation_metrics.csv")
if os.path.isfile(ablation_csv):
    metrics_df = pd.read_csv(ablation_csv)
    colors_abl = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"]
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    for i, cfg in enumerate(CNN_ABLATIONS):
        df_cfg = metrics_df[metrics_df["config"] == cfg["name"]]
        if df_cfg.empty:
            continue
        c = colors_abl[i]
        axes[0].plot(df_cfg["epoch"], df_cfg["val_macro_f1"],
                     label=cfg["name"], color=c, marker="o", markersize=3)
        axes[1].plot(df_cfg["epoch"], df_cfg["loss"],
                     label=cfg["name"], color=c, marker="o", markersize=3)

    axes[0].set_title("TextCNN — Validation Macro-F1 per Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Macro-F1")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_title("TextCNN — Training Loss per Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle("TextCNN — Ablation Study (4 Configurations)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ── Plot 5: RL Reward + Regret ────────────────────────────────────────────────
if rl_available:
    steps      = np.arange(1, len(rl_mean) + 1)
    rg_mean    = np.mean(all_regrets, axis=0)
    rg_std     = np.std(all_regrets,  axis=0)
    rand_regret = [i - r for i, r in enumerate(rand_rw, 1)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(steps, rl_mean, color="#89b4fa", lw=1.8, label="RL Agent (mean, 3 seeds)")
    axes[0].fill_between(steps, rl_mean-rl_std, rl_mean+rl_std,
                          alpha=0.2, color="#89b4fa", label="±1 std")
    axes[0].plot(steps, rand_rw, color="#f38ba8", linestyle="--", lw=1.4, label="Random Baseline")
    axes[0].set_title("RL — Cumulative Reward vs Steps")
    axes[0].set_xlabel("Steps"); axes[0].set_ylabel("Cumulative Reward")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, rg_mean, color="#89b4fa", lw=1.8, label="RL Agent Regret (mean)")
    axes[1].fill_between(steps, rg_mean-rg_std, rg_mean+rg_std, alpha=0.2, color="#89b4fa")
    axes[1].plot(steps, rand_regret, color="#f38ba8", linestyle="--", lw=1.4,
                 label="Random Baseline Regret")
    axes[1].set_title("RL — Cumulative Regret vs Steps")
    axes[1].set_xlabel("Steps"); axes[1].set_ylabel("Cumulative Regret")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(f"RL Agent — Success Rate: {success_rate:.1f}%  |  "
                 f"Gain vs Random: +{pct_gain:.1f}%",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

# ── Plot 6: Overall Model Comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

names  = ["TF-IDF + LR\n(Baseline)"]
f1s    = [lr_f1]
accs   = [lr_acc]
colors = ["#f38ba8"]

if db_available:
    names.append("DistilBERT\n(NLP+RL)")
    f1s.append(db_f1); accs.append(db_acc)
    colors.append("#89b4fa")

for name, res in cnn_results.items():
    names.append(f"TextCNN\n({name})")
    f1s.append(res["macro_f1"]); accs.append(res["accuracy"])
    colors.append("#a6e3a1")

x, w = np.arange(len(names)), 0.35
b1 = ax.bar(x - w/2, accs, w, color=colors, alpha=0.6, label="Accuracy")
b2 = ax.bar(x + w/2, f1s,  w, color=colors, alpha=1.0, label="Macro-F1",
            edgecolor="white", linewidth=0.5)

ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylim(0, 1.13); ax.set_ylabel("Score")
ax.set_title("Overall Model Comparison — Accuracy & Macro-F1",
             fontsize=14, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.show()

section("Evaluation Complete — No files saved to disk")

