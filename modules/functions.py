#Importaciones basicas
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from fpylll import IntegerMatrix, LLL, GSO
from docplex.mp.model import Model


from modules import schnorr_lattice as sl
from modules import qaoa
from modules import utils


def solve_cvp (
    cvp : sl.schnorrCVP, instance: sl.schnorrCVPInstance, 
    x0 = None, delta = 0.75, shots = 10_000, normalize: bool = False, 
    p = 1, min_method = 'Nelder-Mead'
):
    """
    TODO
    """

    babai_result = cvp.babai_algorithm(instance, delta)

    qubo = qaoa.define_qubo(babai_result.D, babai_result.res_vector, babai_result.step_sign, cvp.n)
    Hc, _ = qaoa.define_hamiltonian(qubo)

    if normalize:
        Hc, _ = qaoa.normalize_hamiltonian(Hc)

    circuit = qaoa.construct_circuit(Hc, p)

    #Optimizacion clasica
    _, opt_parameters = qaoa.qaoa_algorithm(circuit, Hc, x0, min_method = min_method)

    counts = qaoa.sample_from_parameters(circuit, opt_parameters, shots) # Obtengo un diccionario para obtener los bitstring

    nD = sl.integer_to_matrix(babai_result.D)

    #Obtengo los nuevos vectores y sus probabilidades
    vnews = sl.bitstring2latticeVectors(nD, counts.keys(), babai_result.step_sign, babai_result.b_op)
    probs = utils.get_probs(counts.values(), shots)

    return vnews, probs, babai_result.b_op, opt_parameters




def solve_cvp_with_opt_paramters(
    cvp : sl.schnorrCVP, instance: sl.schnorrCVPInstance,
    opt_parameters, delta = 0.75, shots = 10_000, p = 1
):
    """
    TODO
    """

    babai_result = cvp.babai_algorithm(instance, delta)

    qubo = qaoa.define_qubo(babai_result.D, babai_result.res_vector, babai_result.step_sign, cvp.n)
    Hc, _ = qaoa.define_hamiltonian(qubo)
    circuit = qaoa.construct_circuit(Hc, p)

    #Nos ahorramos la parte de la optimizacion clasica

    counts = qaoa.sample_from_parameters(circuit, opt_parameters, shots)


    nD = sl.integer_to_matrix(babai_result.D)

    vnews = sl.bitstring2latticeVectors(nD, counts.keys(), babai_result.step_sign, babai_result.b_op)
    probs = utils.get_probs(counts.values(), shots)

    return vnews, probs, babai_result.b_op, opt_parameters
    