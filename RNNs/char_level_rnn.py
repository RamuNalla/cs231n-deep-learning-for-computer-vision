import torch
import torch.nn as nn
import numpy as np

# 1. The Dataset (A tiny Shakespeare snippet for fast training)
text = """
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
"""
# Create vocabulary mapping
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])

vocab_size = len(chars)
print(f"Text length: {len(text)}, Vocabulary size: {vocab_size}")

# 2. The Model Architecture
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Step 1: Map character integers to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Step 2: The Vanilla RNN (batch_first=True makes inputs (Batch, Seq, Feature))
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        
        # Step 3: Linear layer to map hidden state back to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embeds = self.embedding(x) # shape: (batch_size, seq_length, embed_size)
        
        # RNN outputs: 
        # out: all hidden states across the sequence
        # hidden: the very last hidden state
        out, hidden = self.rnn(embeds, hidden) 
        
        # Pass all sequence outputs through the linear layer
        out = self.fc(out) # shape: (batch_size, seq_length, vocab_size)
        return out, hidden

# 3. Training Preparation
seq_length = 20  # How many characters to look at before predicting the next
embed_size = 16
hidden_size = 32
epochs = 200
lr = 0.01

model = CharRNN(vocab_size, embed_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 4. Training Loop (Many-to-Many Architecture)
print("\n--- Starting Training ---")
model.train()

for epoch in range(1, epochs + 1):
    # We will just do a single batch containing all possible sequences for simplicity
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_length):
        inputs.append(encoded[i : i + seq_length])
        targets.append(encoded[i + 1 : i + seq_length + 1]) # Target is shifted by 1
        
    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    
    # Initialize hidden state with zeros
    hidden = torch.zeros(1, x.size(0), hidden_size)
    
    optimizer.zero_grad()
    
    # Forward pass
    output, hidden = model(x, hidden)
    
    # Reshape for CrossEntropyLoss: (Batch * Seq_Len, Classes)
    loss = criterion(output.view(-1, vocab_size), y.view(-1))
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}/{epochs} | Loss: {loss.item():.4f}")

# 5. Generation Function (Autoregressive Decoding)
def generate_text(model, start_seq="O Romeo", predict_len=50):
    model.eval()
    
    # Convert start sequence to tensor
    chars = [ch for ch in start_seq]
    x = torch.tensor([[char2int[ch] for ch in chars]], dtype=torch.long)
    
    hidden = torch.zeros(1, 1, hidden_size)
    
    # Warm up the hidden state using the starting sequence
    with torch.no_grad():
        for i in range(x.size(1) - 1):
            _, hidden = model(x[:, i:i+1], hidden)
            
        # The last character of the seed sequence
        current_char = x[:, -1:]
        
        # Generation Loop
        for _ in range(predict_len):
            out, hidden = model(current_char, hidden)
            
            # Get the probabilities of the next character
            probs = torch.softmax(out.squeeze(), dim=0)
            
            # Pick the character with the highest probability (Greedy Search)
            char_idx = torch.argmax(probs).item()
            
            # Append to our generated text
            chars.append(int2char[char_idx])
            
            # The predicted character becomes the next input
            current_char = torch.tensor([[char_idx]], dtype=torch.long)
            
    return "".join(chars)

print("\n--- Generating Text ---")
print(generate_text(model, start_seq="O Romeo", predict_len=100))