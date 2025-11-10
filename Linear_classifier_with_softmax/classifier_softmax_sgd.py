import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SoftmaxClassifier:                # Linear classifier with Softmax loss and SGD optimization
   
    def __init__(self, input_dim, num_classes, learning_rate=0.01, reg_strength=0.0):     # Initialize the classifier
      
        """
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for SGD
            reg_strength (float): L2 regularization strength
        """

        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        
        # Initialize weights with small random values
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        
        # Track training history
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def softmax(self, scores):      # Compute softmax probabilities from scores
        
        """
        Args:
            scores (numpy.ndarray): Raw scores, shape (N, C)
            
        Returns:
            numpy.ndarray: Softmax probabilities, shape (N, C)
        """
        
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)     # Numerical stability: subtract max
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def forward(self, X):           # Forward pass: compute class scores
        """
        Args:
            X (numpy.ndarray): Input data, shape (N, D)
            
        Returns:
            numpy.ndarray: Class scores, shape (N, C)
        """
        scores = X.dot(self.W) + self.b
        return scores
    
    def compute_loss(self, X, y):       # Compute the softmax loss and gradients
        
        """
        Args:
            X (numpy.ndarray): Input data, shape (N, D)
            y (numpy.ndarray): Labels, shape (N,)
            
        Returns:
            tuple: (loss, dW, db) - loss value and gradients
        """
        N = X.shape[0]
        
        scores = self.forward(X)            # Forward pass
        probs = self.softmax(scores)
        
        correct_log_probs = -np.log(probs[range(N), y])     # Compute cross-entropy loss
        data_loss = np.sum(correct_log_probs) / N
        
        reg_loss = 0.5 * self.reg_strength * np.sum(self.W * self.W)        # Add L2 regularization
        loss = data_loss + reg_loss
        
        dscores = probs.copy()          # Backward pass: compute gradients (derive to get this equation wiht Loss = Data loss + regularization)
        dscores[range(N), y] -= 1       # Gradient of softmax + cross-entropy
        dscores /= N
        
        dW = X.T.dot(dscores)           # Gradient for weights and biases
        db = np.sum(dscores, axis=0)
        
        dW += self.reg_strength * self.W    # Add regularization gradient
        
        return loss, dW, db
    
    def predict(self, X):           # Predict class labels
        """
        Args:
            X (numpy.ndarray): Input data, shape (N, D)
            
        Returns:
            numpy.ndarray: Predicted labels, shape (N,)
        """
        scores = self.forward(X)
        return np.argmax(scores, axis=1)
      

