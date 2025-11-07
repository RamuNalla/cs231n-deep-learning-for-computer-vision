
import numpy as np
from collections import Counter
import pickle
import os
from urllib import request
import ssl
import tarfile

class KNNClassifier:            # k-Nearest neighbors for image classification

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
    

def load_cifar10_batch(file_path):
    """Load a single CIFAR-10 batch file"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    
    # Extract images and labels
    images = batch[b'data']
    labels = batch[b'labels']
    
    return images, np.array(labels)

def download_and_extract_cifar10(data_dir='./cifar10_data'):    # Download and extract CIFAR-10 dataset
    
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tar_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check if already extracted
    extracted_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    if os.path.exists(extracted_dir):
        print("CIFAR-10 already downloaded and extracted")
        return extracted_dir
    
    # Download
    print("Downloading CIFAR-10 dataset...")

    # Configure SSL context using certifi (recommended on macOS where the
    # system Python sometimes lacks a proper cert bundle). We create an
    # HTTPSHandler with the context and install it so request.urlretrieve
    # uses the custom context.
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        opener = request.build_opener(request.HTTPSHandler(context=ctx))
        request.install_opener(opener)
    except Exception as e:
        # If certifi is not available, warn the user and fall back to the
        # default behavior (which may still raise the same SSL error).
        print("Warning: could not configure certifi-based SSL context:", e)
        print("If you see certificate verification errors, install certifi: `pip install certifi`")

    request.urlretrieve(url, tar_file)
    
    # Extract
    print("Extracting dataset...")
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    
    print("Download complete!")
    return extracted_dir


def load_cifar10(data_dir='./cifar10_data', num_train=1000, num_test=100):   # Load CIFAR-10 dataset with limited samples
    """

    Args:
        data_dir (str): Directory to store/load data
        num_train (int): Number of training samples
        num_test (int): Number of test samples
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names)
    """
    # Download if needed
    cifar_dir = download_and_extract_cifar10(data_dir)
    
    # Load training data (using only first batch for simplicity)
    train_file = os.path.join(cifar_dir, 'data_batch_1')
    X_train, y_train = load_cifar10_batch(train_file)
    
    # Load test data
    test_file = os.path.join(cifar_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file)
    
    # Limit dataset size
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]
    
    # Load class names
    meta_file = os.path.join(cifar_dir, 'batches.meta')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f"\nDataset loaded:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {class_names}")
    
    return X_train, y_train, X_test, y_test, class_names



def main():
    """Main function to demonstrate KNN classifier"""
    print("=" * 60)
    print("K-Nearest Neighbors Image Classifier")
    print("=" * 60)
    
    # Load CIFAR-10 dataset (limited size for demonstration)
    # Using 1000 training samples and 100 test samples
    X_train, y_train, X_test, y_test, class_names = load_cifar10(
        num_train=1000,
        num_test=100
    )

    # Initialize and train KNN classifier
    print("\n" + "=" * 60)
    print("Training KNN Classifier")
    print("=" * 60)
    k = 5
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)

    # Make predictions on test set
    print("\n" + "=" * 60)
    print(f"Predicting with k={k} nearest neighbors")
    print("=" * 60)
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = knn.accuracy(y_test, y_pred)
    print("\n" + "=" * 60)
    print(f"Results")
    print("=" * 60)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Show some example predictions
    print("\nExample predictions (first 10):")
    print("-" * 60)
    print(f"{'Index':<8} {'True Label':<20} {'Predicted':<20} {'Correct?'}")
    print("-" * 60)
    for i in range(min(10, len(y_test))):
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred[i]]
        correct = "✓" if y_test[i] == y_pred[i] else "✗"
        print(f"{i:<8} {true_label:<20} {pred_label:<20} {correct}")

    # Demonstrate finding k nearest neighbors for a specific image
    print("\n" + "=" * 60)
    print("Detailed Example: Finding Nearest Neighbors")
    print("=" * 60)
    test_idx = 0
    test_image = X_test[test_idx]

    # Calculate distances to all training samples
    distances = []
    for i, train_sample in enumerate(X_train):
        dist = knn.euclidean_distance(test_image, train_sample)
        distances.append((dist, y_train[i], i))

    # Sort and get k nearest
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    print(f"\nTest image index: {test_idx}")
    print(f"True label: {class_names[y_test[test_idx]]}")
    print(f"\nK={k} nearest neighbors:")
    print("-" * 60)
    for rank, (dist, label, train_idx) in enumerate(k_nearest, 1):
        print(f"{rank}. Distance: {dist:.2f} | Label: {class_names[label]} | Train Index: {train_idx}")
    
    # Show majority vote
    neighbor_labels = [label for _, label, _ in k_nearest]
    vote_counts = Counter(neighbor_labels)
    print(f"\nMajority vote:")
    for label, count in vote_counts.most_common():
        print(f"  {class_names[label]}: {count} votes")
    
    predicted = vote_counts.most_common(1)[0][0]
    print(f"\nFinal prediction: {class_names[predicted]}")

if __name__ == "__main__":
    main()