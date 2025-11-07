
import numpy as np
from collections import Counter
import pickle
import os
from urllib import request
import tarfile

class KNNclassifier:            # k-Nearest neighbors for image classification

    def __init__(self, k=5):    # Initialize KNN classifier
        
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):    # store the training data
        """
        Args:
            X_train (numpy.ndarray): Training images (flattened)
            y_train (numpy.ndarray): Training labels
        """
        self.X_train = X_train
        self.y_train = y_train
        print(f"Trained with {len(X_train)} samples")

        



