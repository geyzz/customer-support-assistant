<img width="765" height="748" alt="image" src="https://github.com/user-attachments/assets/952abd7b-1e08-467f-812d-1ac4a514f97a" /># Model Card Draft — Customer Support Assistant

## Model Details

- **Model Name:** Customer Support Assistant
- **Version:** v0.9 (Checkpoint)
- **Architecture:** DistilBERT-base-uncased (core NLP model) + Epsilon-Greedy 
  Contextual Bandit (RL component)
- **Supporting Experiment:** TextCNN (CNN component, in progress)
- **Task:** Intent classification + response selection for e-commerce 
  customer support
- **Language:** English
- **Status:** In development — Week 2 Checkpoint

## Intended Use

- **Primary use:** Classifying customer support messages into one of 27 
  e-commerce intent categories and returning an appropriate template response
- **Intended users:** Academic demonstration only
- **NOT intended for:** Real deployment, clinical or financial decisions, 
  or any safety-critical application

## Dataset

- **Source:** Bitext Customer Support LLM Chatbot Training Dataset
- **Link:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **License:** CC BY 4.0
- **Size:** 24,185 utterances across 27 intent categories
- **Splits:** 80/10/10 train/val/test, stratified by intent, seed = 42
- **Nature:** Synthetically generated — does not contain real user data or PII

## Supported Intent Categories (27)

cancel_order, change_order, change_shipping_address, check_cancellation_fee,
check_invoice, check_payment_methods, check_refund_policy, complaint,
contact_customer_service, contact_human_agent, create_account, delete_account,
delivery_options, delivery_period, edit_account, get_invoice, get_refund,
newsletter_subscription, payment_issue, place_order, recover_password,
registration_problems, review, set_up_shipping_address, switch_account,
track_order, track_refund

## Training Details (Preliminary)

- **Optimizer:** AdamW
- **Learning rate:** 2e-5
- **Batch size:** 16
- **Max epochs:** 5
- **Early stopping patience:** 2
- **Max token length:** 128
- **Seed:** 42
- **Hardware:** Google Colab / local CPU

## Preliminary Metrics (Week 2)

| Component | Metric | Status |
|---|---|---|
| DistilBERT | Macro-F1 | 1.00000 |
| DistilBERT | Accuracy | 1.00000 |
| TextCNN | Macro-F1 | 0.98395 |


## Known Limitations

- English-only — will not perform well on non-English messages
- Trained on synthetic data — may not reflect real-world customer language 
  variation, slang, or informal phrasing
- Class imbalance exists: cancel_order has significantly fewer samples (~428) 
  compared to other intents (~990+), which may affect per-class performance
- Confidence threshold (0.70) is manually set — not learned from data
- Not evaluated on out-of-scope queries (messages that don't match any intent)

## Ethical Considerations

See `docs/ethics_statement.md` for the full ethics statement.
Key risks: misclassification on sensitive queries, unequal performance across 
user groups, and RL agent bias toward frequent intents.


## Citation / Credits

- DistilBERT: Sanh et al. (2019)
- Dataset: Bitext (CC BY 4.0)
- TextCNN: Kim (2014)
- RL (LinUCB/Epsilon-Greedy): Li et al. (2010)
