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

        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def accuracy(self, X, y):       # Compute classification accuracy

        y_pred = self.predict(X)
        return np.mean(y_pred == y) 
    
    def train_sgd(self, X_train, y_train, X_val=None, y_val=None, 
                  num_epochs=100, batch_size=32, verbose=True):         # Train the classifier using Stochastic Gradient Descent
        """
        Args:
            X_train (numpy.ndarray): Training data
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation data (optional)
            y_val (numpy.ndarray): Validation labels (optional)
            num_epochs (int): Number of training epochs
            batch_size (int): Size of mini-batches
            verbose (bool): Print training progress
        """
        N = X_train.shape[0]
        iterations_per_epoch = max(N // batch_size, 1)
        
        print("=" * 70)
        print("Starting SGD Training")
        print("=" * 70)
        print(f"Training samples: {N}")
        print(f"Batch size: {batch_size}")
        print(f"Iterations per epoch: {iterations_per_epoch}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Regularization: {self.reg_strength}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            # Shuffle training data
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            # Mini-batch SGD
            for i in range(0, N, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Compute loss and gradients
                loss, dW, db = self.compute_loss(X_batch, y_batch)
                epoch_loss += loss
                
                # Update weights using SGD
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db
            
            # Average loss for the epoch
            avg_loss = epoch_loss / iterations_per_epoch
            self.loss_history.append(avg_loss)
            
            # Compute accuracies
            train_acc = self.accuracy(X_train, y_train)
            self.train_acc_history.append(train_acc)
            
            if X_val is not None and y_val is not None:
                val_acc = self.accuracy(X_val, y_val)
                self.val_acc_history.append(val_acc)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f}", end="")
                if X_val is not None:
                    print(f" | Val Acc: {val_acc:.4f}")
                else:
                    print()
        
        print("=" * 70)
        print("Training completed!")
        print("=" * 70)

    def plot_training_history(self):                # Plot training loss and accuracy curves
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(self.loss_history, linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.train_acc_history, label='Train', linewidth=2)
        if self.val_acc_history:
            axes[1].plot(self.val_acc_history, label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("\nTraining history plot saved as 'training_history.png'")
        plt.show()           # test



def demonstrate_backward_pass():
    """Demonstrate the backward pass computation step by step"""
    print("\n" + "=" * 70)
    print("DETAILED BACKWARD PASS DEMONSTRATION")
    print("=" * 70)
    
    # Small example for clarity
    N, D, C = 3, 4, 3  # 3 samples, 4 features, 3 classes
    
    # Create simple data
    X = np.random.randn(N, D)
    y = np.array([0, 1, 2])
    W = np.random.randn(D, C) * 0.01
    b = np.zeros(C)
    
    print(f"\nInput dimensions:")
    print(f"  X shape: {X.shape} (N={N} samples, D={D} features)")
    print(f"  W shape: {W.shape} (D={D} features, C={C} classes)")
    print(f"  b shape: {b.shape}")
    print(f"  y shape: {y.shape} (true labels: {y})")
    
    # Forward pass
    print("\n" + "-" * 70)
    print("STEP 1: Forward Pass")
    print("-" * 70)
    scores = X.dot(W) + b
    print(f"Scores (logits):")
    print(scores)
    
    # Softmax
    print("\n" + "-" * 70)
    print("STEP 2: Softmax Probabilities")
    print("-" * 70)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print(f"Probabilities (rows sum to 1):")
    print(probs)
    print(f"\nRow sums: {np.sum(probs, axis=1)}")
    
    # Loss
    print("\n" + "-" * 70)
    print("STEP 3: Cross-Entropy Loss")
    print("-" * 70)
    correct_log_probs = -np.log(probs[range(N), y])
    print(f"Correct class probabilities: {probs[range(N), y]}")
    print(f"Negative log probabilities: {correct_log_probs}")
    loss = np.sum(correct_log_probs) / N
    print(f"Average loss: {loss:.6f}")
    
    # Backward pass
    print("\n" + "-" * 70)
    print("STEP 4: Backward Pass - Compute Gradients")
    print("-" * 70)
    
    # Gradient of loss w.r.t. scores
    dscores = probs.copy()
    dscores[range(N), y] -= 1
    dscores /= N
    
    print(f"Gradient w.r.t. scores (dL/dscores):")
    print(dscores)
    print(f"\nNote: The gradient is (probability - 1) for correct class,")
    print(f"      and just (probability) for incorrect classes")
    
    # Gradients for parameters
    dW = X.T.dot(dscores)
    db = np.sum(dscores, axis=0)
    
    print(f"\nGradient w.r.t. weights (dL/dW) shape: {dW.shape}")
    print(dW)
    print(f"\nGradient w.r.t. biases (dL/db) shape: {db.shape}")
    print(db)
    
    print("\n" + "-" * 70)
    print("STEP 5: SGD Update (example with lr=0.1)")
    print("-" * 70)
    lr = 0.1
    W_new = W - lr * dW
    b_new = b - lr * db
    print(f"Weight update: W_new = W - {lr} * dW")
    print(f"Bias update: b_new = b - {lr} * db")
    print(f"\nWeight change magnitude: {np.linalg.norm(W - W_new):.6f}")
    print(f"Bias change magnitude: {np.linalg.norm(b - b_new):.6f}")


def main():
    """Main function to demonstrate the Softmax classifier"""
    print("=" * 70)
    print("LINEAR CLASSIFIER WITH SOFTMAX LOSS")
    print("=" * 70)
    
    # Demonstrate backward pass with small example
    demonstrate_backward_pass()
    
    # Generate synthetic dataset
    print("\n" + "=" * 70)
    print("FULL TRAINING DEMONSTRATION")
    print("=" * 70)
    
    print("\nGenerating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Dataset split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Initialize classifier
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y))
    
    classifier = SoftmaxClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        learning_rate=0.5,
        reg_strength=0.001
    )
    
    # Train with SGD
    classifier.train_sgd(
        X_train, y_train,
        X_val, y_val,
        num_epochs=100,
        batch_size=32,
        verbose=True
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_acc = classifier.accuracy(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Show some predictions
    print("\n" + "-" * 70)
    print("Sample Predictions:")
    print("-" * 70)
    y_pred = classifier.predict(X_test[:10])
    print(f"{'Index':<8} {'True':<8} {'Predicted':<12} {'Correct?'}")
    print("-" * 70)
    for i in range(10):
        correct = "✓" if y_test[i] == y_pred[i] else "✗"
        print(f"{i:<8} {y_test[i]:<8} {y_pred[i]:<12} {correct}")
    
    # Detailed prediction for one sample
    print("\n" + "-" * 70)
    print("Detailed Prediction for Sample 0:")
    print("-" * 70)
    sample = X_test[0:1]
    scores = classifier.forward(sample)
    probs = classifier.softmax(scores)
    
    print(f"Raw scores: {scores[0]}")
    print(f"Softmax probabilities:")
    for cls in range(num_classes):
        print(f"  Class {cls}: {probs[0, cls]:.6f} ({probs[0, cls]*100:.2f}%)")
    print(f"Predicted class: {np.argmax(probs[0])}")
    print(f"True class: {y_test[0]}")
    
    # Plot training history
    classifier.plot_training_history()


if __name__ == "__main__":
    main()