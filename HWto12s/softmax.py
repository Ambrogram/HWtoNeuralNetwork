import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x)) # Shift values for numerical stability
    return e_x / e_x.sum(axis=0)

# Example usage
if __name__ == "__main__":
    # Assume these are the raw output scores (logits) from the last layer of a neural network
    logits = np.array([2.0, 1.0, 0.1])
    
    # Applying softmax to convert logits to probabilities
    probabilities = softmax(logits)
    
    print("Softmax probabilities:", probabilities)
    print("Sum of probabilities:", np.sum(probabilities))  # Should be 1
