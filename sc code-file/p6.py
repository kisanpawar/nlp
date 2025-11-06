# Implement a program to find the winning neuron using MaxNet.







import numpy as np

def MaxNet():
    # Initial activations of neurons
    activations = np.array([0.3, 0.5, 0.7, 0.9])
    epsilon = 0.2  # inhibition factor
    m = len(activations)
    iteration = 0

    print("Initial Activations:", activations)

    # Continue until only one neuron remains active
    while np.count_nonzero(activations) > 1:
        new_activations = np.zeros_like(activations)

        for j in range(m):
            # Sum of all other activations except itself (lateral inhibition)
            inhibition = epsilon * sum(activations[i] for i in range(m) if i != j)
            new_activations[j] = activations[j] - inhibition

            # Clamp to zero if activation becomes negative
            new_activations[j] = max(0, new_activations[j])

        activations = new_activations
        iteration += 1
        print(f"After iteration {iteration}: {activations}")

    # Identify the winner neuron
    winner_index = np.argmax(activations)
    print(f"\nWinning Neuron: Neuron {winner_index + 1} with final activation {activations[winner_index]}")

# Run the MaxNet
MaxNet()


"""

Initial Activations: [0.3 0.5 0.7 0.9]
After iteration 1: [0.   0.12 0.36 0.6 ]
After iteration 2: [0.    0.    0.216 0.504]
After iteration 3: [0.     0.     0.1152 0.4608]
After iteration 4: [0.      0.      0.02304 0.43776]
After iteration 5: [0.       0.       0.       0.433152]

Winning Neuron: Neuron 4 with final activation 0.4331520000000001

"""