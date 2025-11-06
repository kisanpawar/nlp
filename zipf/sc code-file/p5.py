# Write a program for implementing BAM networks.

import numpy as np

# --- Helper Function ---
def bipolar_sign(x):
    """Convert values to bipolar (+1, -1) form."""
    return np.where(x >= 0, 1, -1)


# --- Training Data ---
X = np.array([
    [1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1],   # Pattern E
    [1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]  # Pattern F
])

Y = np.array([
    [-1, 1],  # Target for E
    [1, 1]    # Target for F
])


# --- Compute Weight Matrix ---
W = X.T @ Y
print("Weight matrix W:\n", W)


# --- Recall Function ---
def recall_bam(x_input, W, max_iters=10):
    """Recall pattern using BAM associative recall."""
    x = x_input.copy()
    for _ in range(max_iters):
        y = bipolar_sign(x @ W)
        x_new = bipolar_sign(y @ W.T)
        if np.array_equal(x_new, x):  # Convergence check
            break
        x = x_new
    return x, y


# --- Testing Recall ---
for idx, x in enumerate(X):
    x_out, y_out = recall_bam(x, W)
    print(f"\nInput Pattern {chr(69 + idx)}: {x}")
    print(f"Recalled Output Y: {y_out}")



"""

Weight matrix W:
 [[ 0  2]
 [ 0  2]
 [ 0  2]
 [ 0  2]
 [ 0 -2]
 [ 0 -2]
 [ 0  2]
 [-2  0]
 [ 0  2]
 [ 0 -2]
 [ 0  2]
 [ 0 -2]
 [ 0  2]
 [-2  0]
 [ 0  2]]

Input Pattern E: [ 1  1  1  1 -1 -1  1  1  1 -1  1 -1  1  1  1]
Recalled Output Y: [-1  1]

Input Pattern F: [ 1  1  1  1 -1 -1  1 -1  1 -1  1 -1  1 -1  1]
Recalled Output Y: [1 1]

"""