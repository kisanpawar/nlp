# Implement Union, Intersection, Complement and Difference operations on fuzzy sets.

def fuzzy_union(A, B):
    """Fuzzy Union (A ∪ B) = max(A(x), B(x))"""
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_intersection(A, B):
    """Fuzzy Intersection (A ∩ B) = min(A(x), B(x))"""
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def fuzzy_complement(A):
    """Fuzzy Complement (¬A) = 1 - A(x)"""
    return {x: 1 - A[x] for x in A}

def fuzzy_difference(A, B):
    """Fuzzy Difference (A - B) = A ∩ (¬B)"""
    B_comp = fuzzy_complement(B)
    return fuzzy_intersection(A, B_comp)


# --- Test the functions ---
if __name__ == "__main__":
    A = {"x1": 0.2, "x2": 0.7, "x3": 1.0}
    B = {"x1": 0.5, "x2": 0.4, "x3": 0.6}

    print("Fuzzy Set A:", A)
    print("Fuzzy Set B:", B)

    print("\nFuzzy Union (A ∪ B):", fuzzy_union(A, B))
    print("Fuzzy Intersection (A ∩ B):", fuzzy_intersection(A, B))
    print("Fuzzy Complement (¬A):", fuzzy_complement(A))
    print("Fuzzy Difference (A - B):", fuzzy_difference(A, B))


"""

Fuzzy Set A: {'x1': 0.2, 'x2': 0.7, 'x3': 1.0}
Fuzzy Set B: {'x1': 0.5, 'x2': 0.4, 'x3': 0.6}

Fuzzy Union (A ∪ B): {'x1': 0.5, 'x3': 1.0, 'x2': 0.7}
Fuzzy Intersection (A ∩ B): {'x1': 0.2, 'x3': 0.6, 'x2': 0.4}
Fuzzy Complement (¬A): {'x1': 0.8, 'x2': 0.30000000000000004, 'x3': 0.0}
Fuzzy Difference (A - B): {'x1': 0.2, 'x3': 0.4, 'x2': 0.6}

"""