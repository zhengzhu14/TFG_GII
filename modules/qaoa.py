#Librerias de Qiskit
from qiskit_optimization.translators import from_docplex_mp


from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit.transpiler import generate_preset_pass_manager


from qiskit.circuit.library import qaoa_ansatz


from docplex.mp.model import Model

from scipy.optimize import minimize, OptimizeResult

import numpy as np


class QaoaMonitor:
    def __init__ (self):
        self.evaluation = []
        self.parameters = []
        self.total_evaluations = 0
        self.total_iterations = 0
        
    
    
    def callback(self, intermediate_result: OptimizeResult):
        self.evaluation.append(intermediate_result.fun)
        self.parameters.append(intermediate_result.x)

    def set_total_evaluations(self, evaluations):
        self.total_evaluations = evaluations

    def set_total_iterations(self, iterations):
        self.total_iterations = iterations
        

    

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


def evaluate_params(circuit, Hc, x):

    simulator = EstimatorV2(options = {'backend_options': 
                                            {'method': 'statevector',
                                             'device': 'CPU',
                                             'max_parallel_threads': 8
                                            }})
    
    pub = (circuit, Hc, x)

    result = simulator.run([pub]).result()[0].data.evs
    
    return result

def qaoa_algorithm(circuit, Hc, x0 = None, min_method = 'Nelder-Mead'):
    """
    TODO
    """

    simulator = EstimatorV2(options = {'backend_options': 
                                            {'method': 'statevector',
                                             'device': 'CPU',
                                             'max_parallel_threads': 8
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

    if result.success:
        print(result.x)
        print(result.message)
        monitor.set_total_evaluations(result.nfev)
        #monitor.set_total_iterations(result.nit)
    monitor.callback(result)

    return monitor, {param.name: val for param, val in zip(parameters, result.x)}




def sample_from_parameters(circuit, opt_parameters, shots):
    """
    TODO
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



