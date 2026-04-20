import numpy as np
from Crypto.Util import number
import random



def get_probs(counts, shots):

    return [count/shots for count in counts]

def get_distances(vnew, t):
    res_vectors = np.subtract(vnew, t)

    distances = [np.linalg.norm(vector) for vector in res_vectors]

    return distances


def get_distances2(vnew, t):
    res_vectors = np.subtract(vnew, t)

    distances2 = [np.dot(vector, vector) for vector in res_vectors]

    return distances2


def prettyprint(vnews, distances, probs, b_op, res_vector):
    distT_B = np.linalg.norm(res_vector)

    print(f"Vector más corto por algoritmo de Babai: \nb_op = {b_op}\n")
    print(f"El vector residual \nt - b_op = {res_vector}\n")
    print(f"La distancia |t - b_op| = {distT_B:.3f}\n")

    total_count = len(vnews)

    for i in range(total_count):
        print(f"{i}: Prob = {probs[i]:.5f}\n\tvnew = {vnews[i]} con distancia: {distances[i]:.3f}")


def generate_N(bitLength):
    """
    param bitLength: cantidad de bits de N

    return: N = p*q de bitLength bits
    """
    

    min_p_bit_length = max(2, bitLength // 4)
    max_p_bit_length = bitLength - min_p_bit_length


    while True:
        p_bit_length = random.randint(min_p_bit_length, max_p_bit_length)

        p = number.getPrime(p_bit_length)

        q_bit_length = bitLength - p_bit_length + 1

        q = number.getPrime(q_bit_length)

        N = p * q

        if N.bit_length() == bitLength:
            return N