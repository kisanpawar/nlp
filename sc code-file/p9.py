# Create fuzzy relation by Cartesian product of any two fuzzy sets.

import numpy as np

def fuzzy_cartesian_product(A, B):
    """
    Creates a fuzzy relation R from the Cartesian product of two fuzzy sets, A and B.
    The membership value R(i, j) is calculated as: R(i, j) = min(A(i), B(j))
    
    Args:
        A (np.array): A 1D array of membership values for fuzzy set A.
        B (np.array): A 1D array of membership values for fuzzy set B.
        
    Returns:
        np.array: A 2D array (matrix) representing the fuzzy relation R.
    """
    return np.minimum.outer(A, B)


# Define fuzzy sets A and B
A = np.array([1.0, 0.8, 0.4])
B = np.array([0.1, 0.5, 0.9])

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)
print("-" * 30)

# Compute Cartesian product
R = fuzzy_cartesian_product(A, B)

print("Fuzzy Relation R (A × B):")
print(R)


"""

Fuzzy Set A: [1.  0.8 0.4]
Fuzzy Set B: [0.1 0.5 0.9]
------------------------------
Fuzzy Relation R (A × B):
[[0.1 0.5 0.9]
 [0.1 0.5 0.8]
 [0.1 0.4 0.4]]

"""