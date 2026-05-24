#Importaciones basicas
from modules import schnorr_lattice as sl
from modules import qaoa
from modules import utils


"""
Modulo con las funciones que resuelven el CVP siguiendo la formulacion de Schnorr-Yan

- Funcion: solve_cvp(cvp, instance, x0, delta, normalize, p, min_method): resuelve clasicamente el cvp y aplica el 
        refinamiento por QAOA. Devuelve los vectores cercanos calculados, sus probabilidades, la solucion aproximada
        y los angulos optimos para el QAOA.

- Funcion solve_cvp_with_opt_paramters(cvp, instance, opt_parameters, delta, normalize, p): funcion que resuelve el cvp
        sin pasar por el bucle variacional del QAOA sino mediante los ángulos fijos optimos calculados.

"""

def get_shots(n_qubits):
    if n_qubits <= 7:
        return 1_000
    elif n_qubits <= 15:
        return 10_000
    elif n_qubits <= 20:
        return 100_000
    else:
        return 1_000_000
    
    

def solve_cvp (
    cvp : sl.schnorrCVP, instance: sl.schnorrCVPInstance, 
    x0 = None, delta = 0.75, normalize: bool = False, 
    p = 1, min_method = 'Nelder-Mead'
):
    """
    Aplica todo el pipeline del algoritmo de Shnorr junto al refinamiento cuantico de Yan et al.

    param cvp: parametros del problema de factorizacion
    param instance: instancia CVP
    param x0: parametros iniciales si procede
    param delta: parametro de la reduccion LLL
    param normalize: booleano para indicar si normalizar o no el Hamiltoniano
    param p: profundidad del ansatze del QAOA
    param min_method: optimizador clasico a utilizar

    return vnews: vectores de la vecindad de b_op mas cercanos a t
           probs: las probabilidades de ser medidos de estos vectores
           b_op: solucion aproximada del CVP
           opt_parameters: angulos optimos calculados con el QAOA.
    """

    babai_result = cvp.babai_algorithm(instance, delta)

    qubo = qaoa.define_qubo(babai_result.D, babai_result.res_vector, babai_result.step_sign, cvp.n)
    Hc, _ = qaoa.define_hamiltonian(qubo)

    if normalize:
        Hc, _ = qaoa.normalize_hamiltonian(Hc)

    circuit = qaoa.construct_circuit(Hc, p)

    #Optimizacion clasica
    _, opt_parameters = qaoa.qaoa_algorithm(circuit, Hc, x0, min_method = min_method)

    
    #Obtengo una cantidad de shots adecuada en función de los qubits
    shots = get_shots(circuit.num_qubits)

    counts = qaoa.sample_from_parameters(circuit, opt_parameters, shots) # Obtengo un diccionario para obtener los bitstring

    nD = sl.integer_to_matrix(babai_result.D)

    #Obtengo los nuevos vectores y sus probabilidades
    vnews = sl.bitstring2latticeVectors(nD, counts.keys(), babai_result.step_sign, babai_result.b_op)
    probs = utils.get_probs(counts.values(), shots)

    return vnews, probs, babai_result.b_op, opt_parameters




def solve_cvp_with_opt_paramters(
    cvp : sl.schnorrCVP, instance: sl.schnorrCVPInstance,
    opt_parameters, delta = 0.75, normalize: bool = False,
    p = 1
):
    """
    Aplica todo el pipeline del algoritmo de Shnorr y elimina el bucle variacional del QAOA mediante 
    el uso de angulos fijos previamente calculados

    param cvp: parametros del problema de factorizacion
    param instance: instancia CVP
    param opt_parameters: angulos fijos optimos
    param delta: parametro de la reduccion LLL
    param normalize: booleano para indicar si normalizar o no el Hamiltoniano
    param p: profundidad del ansatze del QAOA

    return vnews: vectores de la vecindad de b_op mas cercanos a t
           probs: las probabilidades de ser medidos de estos vectores.
           b_op: solucion aproximada del CVP.
           opt_parameters: angulos optimos.
    """

    babai_result = cvp.babai_algorithm(instance, delta)

    qubo = qaoa.define_qubo(babai_result.D, babai_result.res_vector, babai_result.step_sign, cvp.n)
    Hc, _ = qaoa.define_hamiltonian(qubo)

    if normalize:
        Hc, _ = qaoa.normalize_hamiltonian(Hc)

    circuit = qaoa.construct_circuit(Hc, p)

    #Nos ahorramos la parte de la optimizacion clasica

    #Obtengo una cantidad de shots adecuada en función de los qubits
    shots = get_shots(circuit.num_qubits)
    
    counts = qaoa.sample_from_parameters(circuit, opt_parameters, shots)


    nD = sl.integer_to_matrix(babai_result.D)

    vnews = sl.bitstring2latticeVectors(nD, counts.keys(), babai_result.step_sign, babai_result.b_op)
    probs = utils.get_probs(counts.values(), shots)

    return vnews, probs, babai_result.b_op, opt_parameters
    