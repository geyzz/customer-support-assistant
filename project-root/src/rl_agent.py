import random
import matplotlib.pyplot as plt
from responses import responses
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Load model + tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")

model.eval()

# Load label names from model config
label_names = model.config.id2label

actions = list(responses.keys())


class RLAgent:
    def __init__(self, actions, epsilon=0.1, threshold=0.7):
        self.actions = actions
        self.epsilon = epsilon
        self.threshold = threshold

    def select_action(self, predicted_intent, confidence):
        if confidence >= self.threshold:
            return predicted_intent

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return predicted_intent

    def get_reward(self, action, true_intent):
        return 1 if action == true_intent else 0

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

# STORAGE LISTS 
predicted_intents = []
confidences = []
true_intents = []

while True:
    text = input("Enter message: ").lower()

    if text == "quit":
        break

    # Get prediction + confidence
    predicted_label, confidence_score = predict_intent_with_confidence(text)

    print(f"Predicted: {predicted_label} (Confidence: {confidence_score:.2f})")

    # Ask user for TRUE label (for evaluation)
    actual_label = input("Enter TRUE intent: ").strip()

    # Store values
    predicted_intents.append(predicted_label)
    confidences.append(confidence_score)
    true_intents.append(actual_label)

print("\n--- Running RL Evaluation ---\n")

# RUN RL SIMULATION
agent = RLAgent(actions)

cumulative_rewards = []
cumulative_regret = []

total_reward = 0
total_regret = 0

for i in range(len(predicted_intents)):
    pred = predicted_intents[i]
    conf = confidences[i]
    true = true_intents[i]

    action = agent.select_action(pred, conf)

    reward = agent.get_reward(action, true)

    optimal_reward = 1
    regret = optimal_reward - reward

    total_reward += reward
    total_regret += regret

    cumulative_rewards.append(total_reward)
    cumulative_regret.append(total_regret)

    print(f"Step {i+1}")
    print(f"Predicted: {pred} | Confidence: {conf:.2f}")
    print(f"Chosen Action: {action}")
    print(f"True Intent: {true}")
    print(f"Reward: {reward}, Regret: {regret}")
    print("-" * 40)


# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure()
plt.plot(cumulative_rewards, label="Cumulative Reward")
plt.plot(cumulative_regret, label="Cumulative Regret")

plt.xlabel("Steps")
plt.ylabel("Value")
plt.title("RL Performance (Reward vs Regret)")
plt.legend()

plt.show()
