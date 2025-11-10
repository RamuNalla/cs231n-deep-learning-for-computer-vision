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
      

