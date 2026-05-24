#Librerias de Qiskit
from qiskit_optimization.translators import from_docplex_mp

from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit.circuit.library import qaoa_ansatz

from docplex.mp.model import Model

from scipy.optimize import minimize, OptimizeResult

import numpy as np


"""
En este módulo se define todo las funciones para implementar el QAOA mediante Qiskit.

- Clase QaoaMonitor: clase monitor que contiene la informacion del bucle variacional del algoritmo QAOA.

- Funcion define_qubo(D, residual_vector, step_signs, n): devuelve la formulacion QUBO del problema.
- Funcion define_hamiltonian(qubo_p): calcula el Hamiltoniano de coste asociado a la funcion QUBO.
- Funcion construct_circuit(Hc, reps): construye el circuito parametrizado o Ansatze.
- Funcion evaluate_params(circuit, Hc, x, sim = None): dado unos parametros estimar la energia.

- Funcion qaoa_algorithm(circuit, Hc, x0, min_method): realiza toda la ejecucion del algoritmo de QAOA y devuelve 
    los parámetros optimos.

- Funcion sample_from_parameters(circuit, opt_parameters, shots): devuelve la distribucion aproximada de bitstrings
    del estado generado a partir de los parametros optimos.
    
- Funcion normalize_hamiltonian(Hc): normaliza el Hamiltoniano de coste.

"""


class QaoaMonitor:
    def __init__ (self):
        self.evaluation = []
        self.parameters = []
        self.iterations = 0
        
    
    
    def callback(self, intermediate_result: OptimizeResult):
        self.iterations += 1
        self.evaluation.append(intermediate_result.fun)
        self.parameters.append(intermediate_result.x)

    

#Funciones de la parte del QAOA
def define_qubo(D, residual_vector, step_signs, n): 
    """
    Genera una instancia de un problema Cuadratico de Qiskit.
    Aqui se va a generar directamente el problema asociada a la funcion QUBO del problema.

    D: Base del retículo LLL reducido.
    residual_vector: vector residual obtenido de restar  t - b_op
    step_signs: step_signs[i] = Sign(mu[i] - c[i]). Permite saber que si se ha aproximado hacia abajo el valor de mu
                                                    explorar la aproximacion hacia arriba de mu, y al reves tambien.
    
    return: QuadraticProgram()                                            

    """

    mdl = Model("quboProblem")

    z = mdl.binary_var_list(n, name = "z")

    objective = 0

    #Itero sobre los n + 1 elementos de los vectores
    for j in range (n + 1):

        #Realizo la operacion que hay dentro del valor absoluto
        #A cada posicion del vector residual t - bop le resto la suma del signo por la variable z y la posicion en concreto de
        #vector de la base reducida
        aux_ob = residual_vector[j] - mdl.sum([step_signs[i]*z[i]*D[i, j] for i in range(n)])

        #Lo elevo al cuadrado
        objective += aux_ob * aux_ob
    
    #Lo convierto en un problema de minimizacion
    mdl.minimize(objective)

    #Realizo la conversion de un modelo docplex a un modelo qiskit.
    mod = from_docplex_mp(mdl)
    
    return mod



def define_hamiltonian(qubo_p):
    op, offset = qubo_p.to_ising()
    return op, offset


def construct_circuit(Hc, reps = 1):
    circuit = qaoa_ansatz(cost_operator = Hc, reps = reps)
    return circuit



def circ_asign_params(circuit, parameters):
        ncircuit = circuit.assign_parameters(parameters)
        return ncircuit


def evaluate_params(circuit, Hc, x, sim = None):
    """
    Calcula la energia estimada de un circuito circuit con los parametros x 
    respecto del Hamiltoniano de coste  Hc.
    """

    if sim == None: 
        simulator = EstimatorV2(options = {'backend_options': 
                                            {'method': 'automatic',
                                             'device': 'CPU',
                                             'max_parallel_threads': 0,
                                             'max_parallel_experiments': 0,
                                            }})
    else: 
        simulator = sim
    
    pub = (circuit, Hc , x)

    results = simulator.run([pub]).result()
    result = results[0].data.evs

    return result

def qaoa_algorithm(circuit, Hc, x0 = None, min_method = 'Nelder-Mead'):
    """
    Realiza la ejecución de todo el algoritmo de QAOA.

    param circuit: circuito parametrizado o Ansatze
    param Hc: Hamiltoniano de coste
    param x0: parametros iniciales si procede
    param min_method: optimizador clasico a utilizar.

    return monitor: informacion del bucle variacional
           opt_parameters: angulos optimos
    """
    simulator = EstimatorV2(options = {'backend_options': 
                                            {'method': 'automatic',
                                             'device': 'CPU',
                                             'max_parallel_threads': 0,
                                             'max_parallel_experiments': 0,
                                            }}) #Instancio el simulador exacto sin ruido

    def func_to_minimize(x):
        job = simulator.run([(circuit, Hc, x)])
        result = job.result()[0]
        
        energy  = float(result.data.evs)
        return energy
    
    
    monitor = QaoaMonitor()

    parameters = circuit.parameters
    p2 = len(parameters)

    if x0 is None:
        x0 = np.asarray([0.0]*p2)
    
    result = minimize(func_to_minimize, x0, method = min_method, callback = monitor.callback)

    monitor.callback(result)

    return monitor, {param.name: val for param, val in zip(parameters, result.x)}




def sample_from_parameters(circuit, opt_parameters, shots):
    """
    Toma un circuito y realiza shots mediciones para obtener una distribucion de estados básicos
    que represente el estado cuántico.

    param circuit: circuito parametrizado
    param opt_parameters: parametros optimos del QAOA
    param shots: numero de shots a realizar


    return {bitstring: count}: diccionario de estados medidos y la cantidad de veces medidas 

    """

    sampler = SamplerV2() #Declaro un Sampler exacto

    ncircuit = circuit.copy()

    ncircuit.measure_all()

    #parameter_values = [opt_parameters[p.name] for p in circuit.parameters]

    pub = (ncircuit, opt_parameters)


    job = sampler.run([pub], shots = shots)
    
    result = job.result()[0] 

    counts = result.data.meas.get_counts() #Obtengo un diccionario [bitstring : frecuencia]

    ordered_counts = dict(sorted(counts.items(), key = lambda x: x[1], reverse = True))

    return ordered_counts


def normalize_hamiltonian(Hc):
    """
    Función que normaliza los coeficientes del Hamiltoniano por el mayor coeficiente en valor absoluto
    """

    norma = np.max(np.abs(Hc.coeffs))

    nHc = Hc.copy()

    nHc.coeffs = nHc.coeffs / norma

    return nHc, norma

