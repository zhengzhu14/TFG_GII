#Importaciones basicas
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from fpylll import IntegerMatrix, LLL, GSO
from docplex.mp.model import Model


from modules import schnorr_lattice as sl
from modules import qaoa as q
from modules import teoria_numeros as tn
from modules import utils


def solve_cvp (cvp : sl.schnorrCVP, q= 10, x0 = None, delta = 0.75, shots = 1_000, p = 1):
    """
    param cvp: SchnorrAlgQAOA() clase con los datos necesarios y las funciones de cálculo

    return 
    """

    pass