# Perform max-min composition on any two fuzzy relations

import numpy as np

def max_min_composition(R, S):
    """
    Perform max–min composition between two fuzzy relations R and S.
    T(x, z) = max_y [ min(R(x, y), S(y, z)) ]
    
    Args:
        R (np.array): Relation from X to Y (m x n)
        S (np.array): Relation from Y to Z (n x p)
        
    Returns:
        np.array: Composed relation T from X to Z (m x p)
    """
    if R.shape[1] != S.shape[0]:
        raise ValueError("Inner dimensions of R and S must match for composition.")
    
    m, n = R.shape
    n, p = S.shape
    T = np.zeros((m, p))
    
    for i in range(m):
        for k in range(p):
            # Compute element-wise min, then take max across Y
            min_vals = np.minimum(R[i, :], S[:, k])
            T[i, k] = np.max(min_vals)
    
    return T


# --- Main Program ---
# Fuzzy relation R: from X → Y (2×3)
R = np.array([
    [0.7, 0.5, 0.3],
    [0.8, 0.4, 0.9]
])

# Fuzzy relation S: from Y → Z (3×2)
S = np.array([
    [0.9, 0.2],
    [0.1, 0.7],
    [0.6, 0.5]
])

print("Fuzzy Relation R:")
print(R)

print("\nFuzzy Relation S:")
print(S)

# Compute the composition T = R ○ S (X → Z)
try:
    T = max_min_composition(R, S)
    print("\nMax–Min Composition (R ○ S):")
    print(T)

except ValueError as e:
    print(f"\nError: {e}")



"""

Fuzzy Relation R:
[[0.7 0.5 0.3]
 [0.8 0.4 0.9]]

Fuzzy Relation S:
[[0.9 0.2]
 [0.1 0.7]
 [0.6 0.5]]

Max–Min Composition (R ○ S):
[[0.7 0.5]
 [0.8 0.5]]

"""