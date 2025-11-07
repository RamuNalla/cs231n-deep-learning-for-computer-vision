
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

    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two vectors"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict_single(self, x):        # Predict label for a single image
        """

        Args:
            x (numpy.ndarray): Single flattened image
            
        Returns:
            int: Predicted label
        """
        # Calculate distances to all training samples
        distances = []
        for i, train_sample in enumerate(self.X_train):
            dist = self.euclidean_distance(x, train_sample)
            distances.append((dist, self.y_train[i]))
        
        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Extract labels of k nearest neighbors
        k_nearest_labels = [label for _, label in k_nearest]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

    def predict(self, X_test):      # Predict labels for multiple images
        """
        Args:
            X_test (numpy.ndarray): Test images (flattened)
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        predictions = []
        for i, x in enumerate(X_test):
            if (i + 1) % 10 == 0:
                print(f"Predicting sample {i+1}/{len(X_test)}")
            pred = self.predict_single(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def accuracy(self, y_true, y_pred):     # Calculate classification accuracy
        
        return np.mean(y_true == y_pred)
    






