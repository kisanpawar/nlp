# Solve the Hamming Network given exemplar vectors

import numpy as np

# Exemplar vectors
e1 = np.array([1, -1, -1, -1])
e2 = np.array([-1, -1, -1, 1])

# Weight matrix
w = 0.5 * np.array([e1, e2]).T
print("Weight matrix (w):\n", w)

# Bias for each output neuron
bias = np.array([2, 2])

# Input patterns to test
x = np.array([
    [-1, -1, 1, -1],
    [-1, -1, 1, 1],
    [-1, -1, -1, 1],
    [1, 1, -1, -1]
])


# --- Hamming Net Computation ---
def hamming_net_output(x, w, bias):
    """Compute the initial response of the Hamming network."""
    y_in = bias + np.dot(x, w)
    return y_in


# --- Maxnet Competition Layer ---
def maxnet(y, epsilon=0.1, max_iter=10):
    """Perform lateral inhibition using the Maxnet algorithm."""
    y = np.copy(y)
    for _ in range(max_iter):
        new_y = np.copy(y)
        for i in range(len(y)):
            inhibition = epsilon * sum(y[j] for j in range(len(y)) if j != i)
            new_y[i] = max(0, y[i] - inhibition)
        if np.allclose(new_y, y):  # Converged
            break
        y = new_y
    return y


# --- Main Process ---
print("\n--- Hamming Net and Maxnet Output ---")

for i, input_x in enumerate(x):
    print(f"\nInput x{i + 1} = {input_x}")

    # Step 1: Compute Hamming Net output
    y_raw = hamming_net_output(input_x, w, bias)
    print(f"Initial Output (y_in) = {np.round(y_raw, 4)}")

    # Step 2: Apply Maxnet
    y_final = maxnet(y_raw)
    print(f"Maxnet Final Output = {np.round(y_final, 4)}")

    # Step 3: Identify winner neuron
    winner_indices = np.where(y_final == np.max(y_final))[0]
    if len(winner_indices) == 1:
        print(f"Assigned to cluster: e{winner_indices[0] + 1}")
    else:
        winner_clusters = [f"e{idx + 1}" for idx in winner_indices]
        print(f"Ambiguous Match: Possibly {' and '.join(winner_clusters)}")



"""

Weight matrix (w):
 [[ 0.5 -0.5]
 [-0.5 -0.5]
 [-0.5 -0.5]
 [-0.5  0.5]]

--- Hamming Net and Maxnet Output ---

Input x1 = [-1 -1  1 -1]
Initial Output (y_in) = [2. 2.]
Maxnet Final Output = [0.6974 0.6974]
Ambiguous Match: Possibly e1 and e2

Input x2 = [-1 -1  1  1]
Initial Output (y_in) = [1. 3.]
Maxnet Final Output = [0.     2.7763]
Assigned to cluster: e2

Input x3 = [-1 -1 -1  1]
Initial Output (y_in) = [2. 4.]
Maxnet Final Output = [0.     3.3659]
Assigned to cluster: e2

Input x4 = [ 1  1 -1 -1]
Initial Output (y_in) = [3. 1.]
Maxnet Final Output = [2.7763 0.    ]
Assigned to cluster: e1


"""