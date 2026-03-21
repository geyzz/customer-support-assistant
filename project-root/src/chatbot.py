from rl_main import RLAgent
from responses import responses
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import random

# Load model + tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")

model.eval()

# Load label names from model config
label_names = model.config.id2label

def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return label_names[predicted_class_id]
def predict_intent_with_confidence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class_id = torch.max(probs, dim=1)

    return label_names[predicted_class_id.item()], confidence.item()

#not yet sure paano 
# def loop_input_check():
#     while True:
#         text = input("Enter message: ").lower()
#         print()
#         if text == "quit":
#             break
#         #NLP - Intent Classifier (DistilBERT-base-uncased)
#         intent, confidence = predict_intent_with_confidence(text)
#         print("NLP - Intent Classifier (DistilBERT-base-uncased)")
#         print(f"Predicted intent: {intent}")
#         print(f"Confidence: {confidence:.2f}")
        
#         print()
#         #RL – Contextual Bandit
#         agent = RLAgent(actions=list(responses.keys()))
#         intentR, confidenceR = predict_intent_with_confidence(text)
#         action = agent.select_action(intentR, confidenceR)
#         reply = random.choice(responses[intent])
#         print("RL – Contextual Bandit")
#         print("Intent:", intentR)
#         print("Confidence:", confidenceR)
#         print("Response:", reply)

# loop_input_check()
