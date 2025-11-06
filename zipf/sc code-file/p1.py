# Write a program to implement logical gates AND, OR and NOT with McCulloch-Pitts.


def mp_neuron(inputs, weights, threshold):
    y_in = sum(xi * wi for xi, wi in zip(inputs, weights))
    if y_in >= threshold:
        output = 1
    else:
        output = 0
    return output


def AND_gate(x1, x2):
    weights = [1, 1]
    threshold = 2
    inputs = [x1, x2]
    return mp_neuron(inputs, weights, threshold)


def OR_gate(x1, x2):
    weights = [1, 1]
    threshold = 1
    inputs = [x1, x2]
    return mp_neuron(inputs, weights, threshold)


def NOT_gate(x1):
    weights = [-1]
    threshold = 0
    inputs = [x1]
    return mp_neuron(inputs, weights, threshold)


def test_gates():
    print("AND GATE")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"AND({x1}, {x2}): {AND_gate(x1, x2)}")

    print("\nOR GATE")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            print(f"OR({x1}, {x2}): {OR_gate(x1, x2)}")

    print("\nNOT GATE")
    for x1 in [0, 1]:
        print(f"NOT({x1}): {NOT_gate(x1)}")


# Run the tests
test_gates()


"""

AND GATE
AND(0, 0): 0
AND(0, 1): 0
AND(1, 0): 0
AND(1, 1): 1

OR GATE
OR(0, 0): 0
OR(0, 1): 1
OR(1, 0): 1
OR(1, 1): 1

NOT GATE
NOT(0): 1
NOT(1): 0

"""