import numpy as np

def batchnorm_forward(x, gamma, beta, eps):
    """
    Forward pass for batch normalization.
    
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - eps: Small constant to prevent division by zero (e.g., 1e-5)
    """
    N, D = x.shape
    
    # 1. Compute Step: Mean
    mu = np.mean(x, axis=0)
    
    # 2. Compute Step: Variance
    var = np.var(x, axis=0)
    
    # 3. Normalize Step
    # We add epsilon for numerical stability
    x_std = np.sqrt(var + eps)
    x_norm = (x - mu) / x_std
    
    # 4. Scale and Shift Step
    out = gamma * x_norm + beta
    
    # Store intermediate values for backward pass (if needed later)
    cache = (x, x_norm, mu, x_std, gamma, beta, eps)
    
    return out, cache

# --- Quick Test ---
# N=2 data points, D=3 features
x_test = np.array([[1.0, 2.0, 3.0], 
                   [4.0, 5.0, 6.0]])
gamma_test = np.ones(3)  # No scaling initially
beta_test = np.zeros(3)  # No shifting initially

out, _ = batchnorm_forward(x_test, gamma_test, beta_test, 1e-5)
print("Input:\n", x_test)
print("Normalized Output:\n", out)
# Expected: The values should be centered around 0 (roughly -1 and 1)