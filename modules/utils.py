import numpy as np


def get_probs(counts, shots):

    return [count/shots for count in counts]

def get_distances(vnew, t):
    res_vectors = np.subtract(vnew, t)

    distances = [np.linalg.norm(vector) for vector in res_vectors]

    return distances


def prettyprint(vnews, distances, probs, b_op, res_vector):
    distT_B = np.linalg.norm(res_vector)

    print(f"Vector más corto por algoritmo de Babai: \nb_op = {b_op}\n")
    print(f"El vector residual \nt - b_op = {res_vector}\n")
    print(f"La distancia |t - b_op| = {distT_B:.3f}\n")

    total_count = len(vnews)

    for i in range(total_count):
        print(f"{i}: Prob = {probs[i]:.5f}\n\tvnew = {vnews[i]} con distancia: {distances[i]:.3f}")