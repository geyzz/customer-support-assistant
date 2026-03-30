#!/bin/bash
set -e  # stop immediately if any step fails

echo "============================================"
echo " Customer Support Assistant - Full Pipeline"
echo "============================================"
echo ""

# Step 1 - Data Pipeline
# src/data_pipeline.py — cleans Bitext, stratified 80/10/10 split, saves to output/
echo "[1/4] Running data pipeline..."
python src/data_pipeline.py
echo "✓ Data pipeline done"
echo ""

# Step 2 - Train DistilBERT (NLP core model)
# src/models/distilbert_model_train.py — AdamW, lr=2e-5, batch=16, 5 epochs,
# early stopping patience=2, saves to experiments/results/fine_tuned_model/
echo "[2/4] Training DistilBERT..."
python src/models/distilbert_model_train.py
echo "✓ DistilBERT training done"
echo ""

# Step 3 - Train TextCNN (CNN ablations A1 + A2)
# src/models/train_cnn.py — Adam, lr=1e-3, batch=16, early stopping patience=2
# A1: static vs fine-tuned GloVe | A2: filter sizes [2,3,4] vs [3,4,5]
# saves best model to cnn_snapshot/
echo "[3/4] Training TextCNN ablations..."
python src/models/train_cnn.py
echo "✓ TextCNN ablations done"
echo ""

# Step 4 - Evaluation (all models + baselines + ablation tables + plots)
# src/eval.py — TF-IDF baseline, DistilBERT metrics, TextCNN ablation curves,
# LinUCB reward/regret, confusion matrix, per-intent slice analysis
echo "[4/4] Running evaluation..."
python src/eval.py
echo "✓ Evaluation done"
echo ""

echo "============================================"
echo " Pipeline complete. Launching chatbot..."
echo "============================================"
echo ""

# Launch chatbot CLI
# src/chatbot.py — loads fine_tuned_model + linucb_agent.pkl
# modes: 'chat' (NLP+RL chatbot) or 'rl' (experiment mode with reward/regret)
python src/chatbot.py
