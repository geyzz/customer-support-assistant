import numpy as np

class LinUCB:
    def __init__(self, n_actions, context_dim, alpha=1.5):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha

        self.A = [np.identity(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_actions)]

    def select_action(self, context_vector, predicted_idx, confidence):
        # Trust NLP if confident
        if confidence >= 0.7:
            return predicted_idx

        context_vector = context_vector.reshape(-1, 1)

        p_values = []

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]

            context_vector = context_vector.reshape(-1, 1)

            p = float(
                (theta.T @ context_vector).item() +
                self.alpha * np.sqrt((context_vector.T @ A_inv @ context_vector).item())
            )

            p_values.append(p)

        return int(np.argmax(p_values))

    def update(self, action, reward, context_vector):
        context_vector = context_vector.reshape(-1, 1)

        self.A[action] += context_vector @ context_vector.T
        self.b[action] += reward * context_vector
