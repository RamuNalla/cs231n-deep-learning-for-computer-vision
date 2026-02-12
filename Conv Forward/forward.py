import numpy as np

def get_output_dims(input_dims, F, S, P):
    """
    input_dims: (N, C, H, W)
    F: Filter size (assuming square filter FxF)
    S: Stride
    P: Padding
    """
    N, C, H, W = input_dims
    
    # Calculate height and width using the standard formula
    H_out = int((H - F + 2 * P) / S) + 1
    W_out = int((W - F + 2 * P) / S) + 1
    
    return H_out, W_out