import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        # Initialize weights with small random values and biases with zeros
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)  # W1: [D, H], b1: [H]
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size) # W2: [H, C], b2: [C]
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape          # N = batch size, D = features

        # --- Forward Pass ---
        # 1. Hidden layer (ReLU activation)
        layer1 = np.maximum(0, X.dot(W1) + b1)   # 1. Hidden Layer: [N, D] dot [D, H] -> [N, H]
        # 2. Output layer (Scores)
        scores = layer1.dot(W2) + b2        # 2. Output Layer: [N, H] dot [H, C] -> [N, C]

        if y is None: return scores

        # --- Compute Loss (Softmax) ---
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # doing this as exp(x) grows incredibly fast (overflow in the computer memory) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N            # average error of the batch
        reg_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        loss = data_loss + reg_loss

        # --- Backward Pass (Gradients) ---
        grads = {}
        
        # dLoss/dScores
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        # Backprop into W2 and b2
        grads['W2'] = layer1.T.dot(dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        # Backprop into hidden layer
        dhidden = dscores.dot(W2.T)
        dhidden[layer1 <= 0] = 0 # ReLU gradient

        # Backprop into W1 and b1
        grads['W1'] = X.T.dot(dhidden) + reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, iterations=100):
        for i in range(iterations):
            loss, grads = self.loss(X, y, reg=reg)
            
            # SGD Update
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            
            if i % 10 == 0:
                print(f"Iteration {i}: Loss {loss:.4f}")

# Create a dummy dataset: 5 samples, 4 features, 3 classes
X_toy = np.random.randn(5, 4)
y_toy = np.array([0, 1, 2, 2, 1])

# Initialize and train
net = TwoLayerNet(input_size=4, hidden_size=10, output_size=3)
net.train(X_toy, y_toy, learning_rate=1e-1, reg=1e-6, iterations=50)