# Implement De-Morgan's Law.

import numpy as np

# --- Fuzzy Operations ---
def fuzzy_union(A, B):
    """Standard fuzzy union: max(A, B)"""
    return np.maximum(A, B)

def fuzzy_intersection(A, B):
    """Standard fuzzy intersection: min(A, B)"""
    return np.minimum(A, B)

def fuzzy_complement(A):
    """Standard fuzzy complement: 1 - A"""
    return 1 - A


# --- Define fuzzy sets ---
A = np.array([0.1, 0.4, 0.8, 1.0, 0.5])
B = np.array([0.7, 0.9, 0.3, 0.6, 0.2])

print(f"Fuzzy Set A: {A}")
print(f"Fuzzy Set B: {B}")


# --- De-Morgan's First Law ---
print("\n--- Verifying De-Morgan's First Law ---")
print("NOT (A ∪ B) = (NOT A) ∩ (NOT B)")

lhs_1 = fuzzy_complement(fuzzy_union(A, B))
print(f"LHS (NOT (A ∪ B)): {lhs_1}")

rhs_1 = fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
print(f"RHS ((NOT A) ∩ (NOT B)): {rhs_1}")

is_law_1_valid = np.allclose(lhs_1, rhs_1)
print(f"Law 1 Valid: {is_law_1_valid}")


# --- De-Morgan's Second Law ---
print("\n--- Verifying De-Morgan's Second Law ---")
print("NOT (A ∩ B) = (NOT A) ∪ (NOT B)")

lhs_2 = fuzzy_complement(fuzzy_intersection(A, B))
print(f"LHS (NOT (A ∩ B)): {lhs_2}")

rhs_2 = fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))
print(f"RHS ((NOT A) ∪ (NOT B)): {rhs_2}")

is_law_2_valid = np.allclose(lhs_2, rhs_2)
print(f"Law 2 Valid: {is_law_2_valid}")


"""

Fuzzy Set A: [0.1 0.4 0.8 1.  0.5]
Fuzzy Set B: [0.7 0.9 0.3 0.6 0.2]

--- Verifying De-Morgan's First Law ---
NOT (A ∪ B) = (NOT A) ∩ (NOT B)
LHS (NOT (A ∪ B)): [0.3 0.1 0.2 0.  0.5]
RHS ((NOT A) ∩ (NOT B)): [0.3 0.1 0.2 0.  0.5]
Law 1 Valid: True

--- Verifying De-Morgan's Second Law ---
NOT (A ∩ B) = (NOT A) ∪ (NOT B)
LHS (NOT (A ∩ B)): [0.9 0.6 0.7 0.4 0.8]
RHS ((NOT A) ∪ (NOT B)): [0.9 0.6 0.7 0.4 0.8]
Law 2 Valid: True

"""