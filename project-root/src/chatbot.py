from rl_agent import LinUCB
from responses import responses
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import os

# LOAD MODELS
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "../experiments/results/fine_tuned_model")
AGENT_PATH = os.path.join(BASE, "../experiments/results/linucb_agent.pkl")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
embed_model = DistilBertModel.from_pretrained(MODEL_PATH)
clf_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

embed_model.eval()
clf_model.eval()

label_names = clf_model.config.id2label
actions = list(responses.keys())

context_dim = 768 + 1

# SAVE / LOAD RL AGENT

def save_agent(agent):
    with open(AGENT_PATH, "wb") as f:
        pickle.dump(agent, f)

def load_agent():
    if os.path.exists(AGENT_PATH):
        with open(AGENT_PATH, "rb") as f:
            print("Loaded existing RL agent")
            return pickle.load(f)
    else:
        print("Creating new RL agent")
        return LinUCB(n_actions=len(actions), context_dim=context_dim, alpha=1.5)

# SAVE FEEDBACK DATA
def save_feedback(text, true_intent):
    with open("feedback_data.csv", "a", encoding="utf-8") as f:
        f.write(f"\"{text}\",{true_intent}\n")

# GET CONTEXT
def get_intent_and_context(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = embed_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        cls_np = cls_embedding.squeeze().numpy()
        cls_np = cls_np / (np.linalg.norm(cls_np) + 1e-8)

        clf_outputs = clf_model(**inputs)
        probs = F.softmax(clf_outputs.logits, dim=1)
        confidence, pred_id = torch.max(probs, dim=1)

    intent = label_names[pred_id.item()]
    confidence = confidence.item()

    context_vector = np.append(cls_np, confidence)

    return intent, confidence, context_vector, pred_id.item()

# CHATBOT MODE
def run_chatbot():
    agent = load_agent()

    print("Chatbot mode (type 'quit' to stop)")

    while True:
        print("")
        text = input("User: ").lower()

        if text == "quit":
            print("Saving agent before exit...")
            save_agent(agent)
            break

        intent, confidence, context_vector, predicted_idx = get_intent_and_context(text)

        # Trust NLP if confident
        if confidence > 0.8:
            action_idx = predicted_idx
        else:
            action_idx = agent.select_action(context_vector, predicted_idx, confidence)

        action_intent = actions[action_idx]
        reply = random.choice(responses[action_intent])

        print(f"\n[NLP] Intent: {intent} | Confidence: {confidence:.2f}")
        print(f"[RL] Action: {action_intent}")
        print("Bot:", reply, "\n")

        if confidence < 0.75:
            true_intent = input("Low confidence. Enter TRUE intent (or press enter to skip): ").strip()
        else:
            true_intent = input("Enter TRUE intent (optional): ").strip()

        if true_intent:
            if action_intent == true_intent:
                reward = 1.0
            else:
                reward = 0.0

            agent.update(action_idx, reward, context_vector)

            if reward > 0:
                save_agent(agent)

            save_feedback(text, true_intent)

# EXPERIMENT MODE
def run_rl():
    seeds = [42, 123, 999]

    all_rewards = []
    all_regrets = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = load_agent()

        cumulative_reward = 0
        cumulative_regret = 0

        rewards = []
        regrets = []

        for step in range(50):
            text = input("Enter message (type 'quit' to stop & plot): ").lower()

            if text == "quit":
                print("\nStopping rl and showing graph...\n")
                break

            intent, confidence, context_vector, predicted_idx = get_intent_and_context(text)

            if confidence > 0.8:
                action_idx = predicted_idx
            else:
                action_idx = agent.select_action(context_vector, predicted_idx, confidence)

            action_intent = actions[action_idx]

            print(f"NLP: {intent} | RL: {action_intent}")

            true_intent = input("Enter TRUE intent: ").strip()

            if action_intent == true_intent:
                reward = 1.0
            else:
                reward = 0.0

            regret = 1 - reward

            agent.update(action_idx, reward, context_vector)

            if reward > 0:
                save_agent(agent)

            save_feedback(text, true_intent)

            cumulative_reward += reward
            cumulative_regret += regret

            rewards.append(cumulative_reward)
            regrets.append(cumulative_regret)

        if rewards:
            all_rewards.append(rewards)
            all_regrets.append(regrets)

        if text == "quit":
            break

    if all_rewards:
        plt.figure()
        for i in range(len(all_rewards)):
            plt.plot(all_rewards[i], label=f"Run {i+1}")
        plt.title("Cumulative Reward")
        plt.legend()
        plt.show()

    if all_regrets:
        plt.figure()
        for i in range(len(all_regrets)):
            plt.plot(all_regrets[i], label=f"Run {i+1}")
        plt.title("Cumulative Regret")
        plt.legend()
        plt.show()

# MAIN
if __name__ == "__main__":
    mode = input("Select mode (chat / rl): ").strip().lower()

    if mode == "chat":
        run_chatbot()
    elif mode == "rl":
        run_rl()
    else:
        print("Invalid mode.")
