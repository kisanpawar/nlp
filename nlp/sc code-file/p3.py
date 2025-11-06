# Implement Kohonen Self Organizing Map (SOM)

import numpy as np

# Input data (4 samples, 4 features each)
data = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1]
])

# Initial weights for 2 neurons (4 features each)
weights = np.array([
    [0.2, 0.9],
    [0.4, 0.7],
    [0.6, 0.5],
    [0.8, 0.3]
])

alpha = 0.5  # learning rate

print("Kohonen Self-Organizing Feature Map (KSOFM) Training:")

# One training epoch (each input presented once)
for idx, x in enumerate(data, start=1):
    # Calculate Euclidean distances between input and each neuron's weight vector
    dists = np.sum((weights - x[:, np.newaxis]) ** 2, axis=0)
    winner = np.argmin(dists)

    print(f"\nInput x{idx} = {x}, distances = {np.round(dists, 4)}, Winner Neuron = {winner + 1}")

    # Update weights of the winning neuron
    for i in range(len(x)):
        weights[i, winner] += alpha * (x[i] - weights[i, winner])

    # Show updated weights of winning neuron
    formatted_weights = [f"{weights[i, winner]:.4f}" for i in range(len(x))]
    print(f"Updated weights for Neuron {winner + 1}: {formatted_weights}")

# Display final weights after one pass
print("\nFinal Weights After One Pass:")
for i in range(weights.shape[0]):
    row = [f"{val:.4f}" for val in weights[i]]
    print(f"[{row[0]}, {row[1]}]")


"""

Kohonen Self-Organizing Feature Map (KSOFM) Training:

Input x1 = [0 0 1 1], distances = [0.4  2.04], Winner Neuron = 1
Updated weights for Neuron 1: ['0.1000', '0.2000', '0.8000', '0.9000']

Input x2 = [1 0 0 0], distances = [2.3  0.84], Winner Neuron = 2
Updated weights for Neuron 2: ['0.9500', '0.3500', '0.2500', '0.1500']

Input x3 = [0 1 1 0], distances = [1.5  1.91], Winner Neuron = 1
Updated weights for Neuron 1: ['0.0500', '0.6000', '0.9000', '0.4500']

Input x4 = [0 0 0 1], distances = [1.475 1.81 ], Winner Neuron = 1
Updated weights for Neuron 1: ['0.0250', '0.3000', '0.4500', '0.7250']

Final Weights After One Pass:
[0.0250, 0.9500]
[0.3000, 0.3500]
[0.4500, 0.2500]
[0.7250, 0.1500]

"""