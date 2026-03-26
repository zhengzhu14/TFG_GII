
from Schnorr_QAOA import SchnorrAlgQAOA

#Importaciones basicas
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from fpylll import IntegerMatrix, LLL, GSO
from docplex.mp.model import Model

#Librerias de Qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo


from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager

from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.visualization import plot_histogram


from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import QuantumCircuit, Parameter


def solve_cvp (cvp : SchnorrAlgQAOA, x0 : np.array, delta = 0.75, shots = 1_000, q = 10, p = 1):
    """
    param cvp: SchnorrAlgQAOA() clase con los datos necesarios y las funciones de cálculo

    return 
    """

    B, t = cvp.generate_cvp(q)

    D, b_op, res_vector, step_signs, w, dist  = cvp.babai_algorithm(B, t, delta)

    qubo = cvp.define_qubo(D, res_vector, step_signs)

    Hc, offset = cvp.define_hamiltonian(qubo)

    circuit = cvp.construct_circuit(Hc, reps = p)

    optParameters = cvp.qaoa_algorithm(circuit, Hc, x0)

    results = cvp.sample_from_parameters(circuit, optParameters, shots)

    nD = cvp.integer_to_matrix(D)

    vnew = cvp.bitstring2latticeVectors(nD, results.keys(), step_signs, b_op)

    distances = cvp.get_distances(vnew, t)

    probs = cvp.get_probs(results.values(), shots)

    


    pass