from rl_agent import LinUCB
from responses import responses
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import numpy as np
import random
import matplotlib.pyplot as plt


# LOAD MODELS
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")
embed_model = DistilBertModel.from_pretrained("./fine_tuned_model")
clf_model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")

embed_model.eval()
clf_model.eval()

label_names = clf_model.config.id2label
actions = list(responses.keys())

context_dim = 768 + 1

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
    agent = LinUCB(n_actions=len(actions), context_dim=context_dim, alpha=1.5)

    print("Chatbot mode (type 'quit' to stop)")

    while True:
        print("")
        text = input("User: ").lower()

        if text == "quit":
            break

        intent, confidence, context_vector, predicted_idx = get_intent_and_context(text)

        action_idx = agent.select_action(context_vector, predicted_idx, confidence)
        action_intent = actions[action_idx]

        reply = random.choice(responses[action_intent])
        
        print(f"\n[NLP] Intent: {intent} | Confidence: {confidence:.2f}")
        print(f"[RL] Action: {action_intent}")
        print("Bot:", reply)

        true_intent = input("Enter TRUE intent (optional): ").strip()

        if true_intent:
            if action_intent == true_intent:
                reward = 1.0
            elif intent == true_intent:
                reward = 0.5
            else:
                reward = 0.0

            agent.update(action_idx, reward, context_vector)


# EXPERIMENT MODE
def run_experiment():
    seeds = [42, 123, 999]

    all_rewards = []
    all_regrets = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = LinUCB(n_actions=len(actions), context_dim=context_dim, alpha=1.5)

        cumulative_reward = 0
        cumulative_regret = 0

        rewards = []
        regrets = []

        for step in range(50):
            text = input("Enter message (type 'quit' to stop & plot): ").lower()

            if text == "quit":
                print("\nStopping experiment and showing graph...\n")
                break

            intent, confidence, context_vector, predicted_idx = get_intent_and_context(text)

            action_idx = agent.select_action(context_vector, predicted_idx, confidence)
            action_intent = actions[action_idx]

            print(f"NLP: {intent} | RL: {action_intent}")

            true_intent = input("Enter TRUE intent: ").strip()

            # Reward
            if action_intent == true_intent:
                reward = 1.0
            elif intent == true_intent:
                reward = 0.5
            else:
                reward = 0.0

            regret = 1 - reward

            agent.update(action_idx, reward, context_vector)

            cumulative_reward += reward
            cumulative_regret += regret

            rewards.append(cumulative_reward)
            regrets.append(cumulative_regret)

        if rewards:
            all_rewards.append(rewards)
            all_regrets.append(regrets)

        if text == "quit":
            break

    # PLOT
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
        
# MAIN SWITCH
if __name__ == "__main__":
    mode = input("Select mode (chat / experiment): ").strip().lower()

    if mode == "chat":
        run_chatbot()
    elif mode == "experiment":
        run_experiment()
    else:
        print("Invalid mode.")
