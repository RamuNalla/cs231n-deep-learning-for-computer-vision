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

def conv_forward_naive(x, w, b, conv_param):
    """
    x: Input data of shape (N, C, H, W)
    w: Weights of shape (F_count, C, HH, WW)
    b: Biases of shape (F_count,)
    conv_param: A dictionary with keys 'stride' and 'pad'
    """
    N, C, H, W = x.shape
    F_count, _, HH, WW = w.shape
    S, P = conv_param['stride'], conv_param['pad']
    
    # 1. Calculate output dimensions
    H_out, W_out = get_output_dims(x.shape, HH, S, P)
    
    # 2. Add zero padding to the spatial dimensions (H and W)
    # (0,0) for N and C; (P,P) for H and W
    x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    
    # 3. Initialize output volume
    out = np.zeros((N, F_count, H_out, W_out))
    
    # 4. The 4 Nested Loops (Batch, Filter, Height, Width)
    for n in range(N):               # Loop over examples in the batch
        for f in range(F_count):     # Loop over each filter/output channel
            for i in range(H_out):   # Loop over output height
                for j in range(W_out): # Loop over output width
                    
                    # Calculate the "window" boundaries in the padded input
                    start_i = i * S
                    end_i = start_i + HH
                    start_j = j * S
                    end_j = start_j + WW
                    
                    # Extract the local slice (all channels included: C x HH x WW)
                    x_slice = x_padded[n, :, start_i:end_i, start_j:end_j]
                    
                    # Compute the dot product + bias
                    # We sum across all channels and spatial positions
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
                    
    cache = (x, w, b, conv_param)
    return out, cache