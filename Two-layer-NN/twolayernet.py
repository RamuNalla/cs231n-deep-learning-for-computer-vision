import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        # Initialize weights with small random values and biases with zeros
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # --- Forward Pass ---
        # 1. Hidden layer (ReLU activation)
        layer1 = np.maximum(0, X.dot(W1) + b1) 
        # 2. Output layer (Scores)
        scores = layer1.dot(W2) + b2

        if y is None: return scores

        # --- Compute Loss (Softmax) ---
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + reg_loss