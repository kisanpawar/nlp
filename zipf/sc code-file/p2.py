# Program to implement Hebb's Learning Rule

# Input patterns (each includes a bias term as the last value)
inputs = [
    [-1, -1, -1],
    [-1,  1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1]
]

# Target outputs
targets = [-1, -1, -1, 1]

# Initialize weights to zero
weights = [0, 0, 0]

print("Training using Hebbian Learning Rule:\n")

# Hebbian learning rule: w_new = w_old + x * t
for i in range(4):
    x = inputs[i]
    t = targets[i]
    print(f"Input {i+1}: {x}, Target: {t}")
    for j in range(3):
        weights[j] += x[j] * t
    print(f"Updated Weights: {weights}\n")

print("Final Weights:", weights)

# Testing the trained network
print("\nTesting the network with final weights:")
for x in inputs:
    y_in = sum(i * w for i, w in zip(x, weights))
    print(f"Input: {x[:-1]}, Bias: {x[2]}, Output (y_in): {y_in}")



"""

Training using Hebbian Learning Rule:

Input 1: [-1, -1, -1], Target: -1
Updated Weights: [1, 1, 1]

Input 2: [-1, 1, 1], Target: -1
Updated Weights: [2, 0, 0]

Input 3: [1, -1, 1], Target: -1
Updated Weights: [1, 1, -1]

Input 4: [1, 1, 1], Target: 1
Updated Weights: [2, 2, 0]

Final Weights: [2, 2, 0]

Testing the network with final weights:
Input: [-1, -1], Bias: -1, Output (y_in): -4
Input: [-1, 1], Bias: 1, Output (y_in): 0
Input: [1, -1], Bias: 1, Output (y_in): 0
Input: [1, 1], Bias: 1, Output (y_in): 4

"""