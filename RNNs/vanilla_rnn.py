import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size):

        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))

    def rnn_step(self, x_t, h_prev):
        input_term = np.dot(self.W_xh, x_t)
        hidden_term = np.dot(self.W_hh, h_prev)
        z = hidden_term + input_term + self.b_h
        h_t = np.tanh(z)
        return h_t

    def forward_sequence(self, x_seq):
        """
        Processes an entire sequence of inputs (e.g., a sentence).
        """
        # Initialize the very first hidden state (h_0) as a vector of zeros
        h_t = np.zeros((self.hidden_size, 1))
        
        # List to store the history of hidden states
        h_states = []
        
        for x_t in x_seq:
            # The output of this step becomes the h_prev for the next step
            h_t = self.rnn_step(x_t, h_t)
            h_states.append(h_t)
            
        return h_states

# --- Testing the Implementation ---
np.random.seed(42) # For reproducibility

INPUT_SIZE = 3   # e.g., a 3-dimensional word embedding
HIDDEN_SIZE = 4  # The size of our "memory" vector
SEQ_LENGTH = 5   # A sequence of 5 words/timesteps

# Create a random sequence of inputs, List of 5 vectors, each of shape (3, 1)
sequence = [np.random.randn(INPUT_SIZE, 1) for _ in range(SEQ_LENGTH)]

# Initialize and run
rnn = VanillaRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
all_hidden_states = rnn.forward_sequence(sequence)

print(f"Generated {len(all_hidden_states)} hidden states.")
print(f"Shape of final hidden state: {all_hidden_states[-1].shape}")
print("\nFinal Hidden State (h_T):\n", all_hidden_states[-1])