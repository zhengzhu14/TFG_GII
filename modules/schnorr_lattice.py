#Importaciones basicas
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt

from fpylll import IntegerMatrix, LLL, GSO

from math import log2, ceil


from dataclasses import dataclass

from .teoria_numeros import first_n_primes

"""
TODO
"""


@dataclass
class schnorrCVPInstance:
    B: IntegerMatrix
    t: tuple

@dataclass
class schnorrCVPResult:
    D: IntegerMatrix
    b_op: np.ndarray
    res_vector: np.ndarray
    step_sign: np.ndarray
    weight: tuple
    delta: float



class schnorrCVP:

    def __init__(self, N, c, l, seed, verbose = True):
        self.N = N

        self.c = c

        self.seed = seed
        np.random.seed(self.seed)

        self.l = l

        self.m = int(ceil(log2(self.N)))
        self.n = int(round((self.l*self.m) / ceil(log2(self.m))))

        self.smooth_bound = 2*self.n**2


        if verbose :
            print(f'El numero de bits de N = {self.N} es m = {self.m}')
            print(f'La dimension del reticulo que vamos a tratar es n = {self.n}')
            print(f'La cota smooth que vamos a tomar: {self.smooth_bound}')



       #Metodos publicos
    def generate_cvp(self, q, verbose = True, diagonal = None):
        """
        Generar una instancia aleatoria del cvp asociado al problema 
        de factorizar N.

        param q: valor con el que generar el ultimo valor de los vectores
        param diagonal: diagonal predefinido

        return: IntegerMatrix
        """

        if diagonal == None:
            f = np.random.permutation([(i + 1) // 2 for i in range(1, self.n + 1)])
        else:
            f = np.array(diagonal)

        # Crear una matriz de 0s y añadir en la diagonal la permutación escogida
        B = np.zeros(shape=(self.n, self.n))
        np.fill_diagonal(B, f)

        # Crear la ultima fila de la matriz

        basis = first_n_primes(self.n)
        final_row = np.round(q ** self.c * np.log(np.array(basis)))
       
        B = np.vstack((B, final_row))
        
        # fpylll solo acepta listas
        B = [[int(b) for b in bs] for bs in B]

        # Convertir B a una matriz del tipo IntegerMatrix de la libreria fpylll
        B = IntegerMatrix.from_matrix(B)

        # Definir vector objetivo
        t = np.zeros(self.n + 1)
        t[-1] = np.round(q ** self.c * np.log(self.N))
        t = tuple(t.astype(int).tolist())
        B.transpose()

        if verbose: 
            print(f'B = \n{B}')
            print(f't = \n{t}')

        return schnorrCVPInstance(B, t)
    


    def __lll_reduction_personal(self, B, delta = 0.75):
        #TODO
        pass


    
    def lll_reduced(self, B, delta = 0.75, fpylll = True):
        if fpylll:
            D = deepcopy(B)
            LLL.reduction(D, delta)
            return D
        else:
            Bmatrix = integer_to_matrix(B)
            return self.__lll_reduction_personal(Bmatrix, delta)


    def babai_algorithm(self, inst: schnorrCVPInstance,  delta = 0.75):
        """
        Calcula el vector cercano aproximado mediante el algoritmo de babai
        param B: 
        """
        D = deepcopy(inst.B) # D está transpuesta
        _ = LLL.reduction(D, delta) #Obtengo la matriz reducida

        G = GSO.Mat(D, update = True)
        w = G.babai(inst.t) # Obtengo los pesos del algoritmo de Babai

        b_op = np.array(D.multiply_left(w)) # w @ D

        res_vector = np.array(inst.t) - b_op


        #Necesito obtener hacia donde se aproxima cada peso del algoritmo de Babai
        #Con este módulo necesito crear una matrix del doble de tamanyo

        A = IntegerMatrix(2*self.n, self.n + 1) 

        for i in range(self.n):
            for j in range(self.n + 1):
                A[i, j] = D[i, j]

        b = np.array(inst.t)
        for i in reversed(range(self.n)):
            for j in range(self.n + 1):
                A[self.n + i, j] = int(b[j])
            b = b - w[i]*np.array(D[i])

        #De A[n], ..., A[2*n - 1] tenemos los valores intermedios del algoritmo de Babai

        M = GSO.Mat(A, update = True) #Al calcular la ortogonalizacion los vectores mas alla del indice n se quedan invariables
        round_direction = []
        for i in range(self.n):
            mu = M.get_mu(self.n + i, i) # <target_i, b^*_i>/<b^*_i, b^*_i>
            round_direction.append(w[i] > mu) # 1 si se aproxima hacia arriba y 0 en caso contrario

        step_sign =  (-2*np.array(round_direction).astype(int)) + 1

        #dist_t = np.linalg.norm(res_vector)

        return schnorrCVPResult(D, b_op, res_vector, step_sign, w, delta)





    #Getters
    def get_N (self):
        return self.N
    def get_c(self):
        return self.c
    def get_l(self):
        return self.l
    def get_random_seed(self):
        return self.seed
    def get_n (self):
        return self.n
    def get_smoothbound(self):
        return self.smooth_bound
    

    #Setters
    def set_random_seed(self, seed):
        self.seed = seed
        np.random(self.seed)

    def set_n(self, n):
        self.n = n
    
    def set_smoothbound(self, smoothbound):
        self.smooth_bound = smoothbound









def integer_to_matrix(B):
    """
    Convierte una matriz B de la clase IntegerMatrix
    
    param B: Base de la clase IntegerMatrix n x (n + 1) es decir ya transpuesta

    return np.array()
    """
    rows, cols = B.nrows, B.ncols
    A = np.zeros((rows, cols), dtype = int)
    B.to_matrix(A)

    return A




def bitstrings2vector(bitstrings):
    return np.array([[int(c) for c in reversed(bstring)] for bstring in bitstrings])


def bitstring2latticeVectors(D, state_bistrings, step_signs, b_op):
    """
    
    """
    bits = bitstrings2vector(state_bistrings) #Obtengo el vector de bits de la cadena

    aux = np.multiply(step_signs, bits)

    movement = aux @ D
    

    vnew = b_op + movement

    return vnew



    






# class SchnorrAlgQAOA:
#     #TODO
#     def __init__(self, N, c, l, seed, verbose = True):
#         self.N = N

#         self.c = c

#         self.seed = seed
#         np.random.seed(self.seed)

#         self.l = l

#         #self.m = int(round(log2(self.N)))
#         #self.n = int(round((self.l*self.m) // int(round(log2(self.m)))))
        
#         self.m = np.round(np.log2(N)).astype(int)
#         self.n = np.round(l * self.m // np.log2(self.m)).astype(int)
#         self.smooth_bound = self.n**2

#         self.prime_basis = primes[:self.smooth_bound]

#         self.pass_manager = generate_preset_pass_manager(3, AerSimulator())


#         print(f'El numero de bits de N = {N} es m = {self.m}')
#         print(f'La dimension del reticulo que vamos a tratar es n = {self.n}')
#         print(f'La cota smooth que vamos a tomar: {self.smooth_bound}')

    
#     #Metodos privados
#     def _lll_reduction_personal(self, B, delta = 0.75):
#         #TODO
#         pass


#     #Metodos publicos
#     def generate_cvp(self, q, verbose = True, diagonal = None):
#         """
#         Generar una instancia aleatoria del cvp asociado al problema 
#         de factorizar N.

#         param q: valor con el que generar el ultimo valor de los vectores
#         param diagonal: diagonal predefinido

#         return: IntegerMatrix
#         """

#         if diagonal == None:
#             f = np.random.permutation([(i + 1) // 2 for i in range(1, self.n + 1)])
#         else:
#             f = np.array(diagonal)

#         # Crear una matriz de 0s y añadir en la diagonal la permutación escogida
#         B = np.zeros(shape=(self.n, self.n))
#         np.fill_diagonal(B, f)

#         # Crear la ultima fila de la matriz
#         final_row = np.round(q ** self.c * np.log(np.array(primes[:self.n])))
       
#         B = np.vstack((B, final_row))
        
#         # fpylll solo acepta listas
#         B = [[int(b) for b in bs] for bs in B]

#         # Convertir B a una matriz del tipo IntegerMatrix de la libreria fpylll
#         B = IntegerMatrix.from_matrix(B)

#         # Definir vector objetivo
#         t = np.zeros(self.n + 1)
#         t[-1] = np.round(q ** self.c * np.log(self.N))
#         t = tuple(t.astype(int).tolist())
#         B.transpose()

#         if verbose: 
#             print(f'B = \n{B}')
#             print(f't = \n{t}')

#         return B, t
    
#     def integer_to_matrix(self, B):
#         """
#         Convierte una matriz B de la clase IntegerMatrix
        
#         param B: Base de la clase IntegerMatrix n x (n + 1) es decir ya transpuesta

#         return np.array()
#         """
#         rows, cols = B.nrows, B.ncols
#         A = np.zeros((rows, cols), dtype = int)
#         B.to_matrix(A)

#         return A

    
#     def lll_reduced(self, B, delta = 0.75, fpylll = True):
#         if fpylll:
#             D = deepcopy(B)
#             LLL.reduction(D, delta)
#             return D
#         else:
#             Bmatrix = self.integet_to_matrix(B)
#             return self._lll_reduction_personal(Bmatrix, delta)

#     def babai_algorithm(self, B, t, delta = 0.75):
#         """
#         Calcula el vector cercano aproximado mediante el algoritmo de babai
#         param B: 
#         """
#         D = deepcopy(B) # D está transpuesta
#         _ = LLL.reduction(D, delta) #Obtengo la matriz reducida

#         G = GSO.Mat(D, update = True)
#         w = G.babai(t) # Obtengo los pesos del algoritmo de Babai

#         b_op = np.array(D.multiply_left(w)) # w @ D

#         res_vector = np.array(t) - b_op


#         #Necesito obtener hacia donde se aproxima cada peso del algoritmo de Babai
#         #Con este módulo necesito crear una matrix del doble de tamanyo

#         A = IntegerMatrix(2*self.n, self.n + 1) 

#         for i in range(self.n):
#             for j in range(self.n + 1):
#                 A[i, j] = D[i, j]

#         b = np.array(t)
#         for i in reversed(range(self.n)):
#             for j in range(self.n + 1):
#                 A[self.n + i, j] = int(b[j])
#             b = b - w[i]*np.array(D[i])

#         #De A[n], ..., A[2*n - 1] tenemos los valores intermedios del algoritmo de Babai

#         M = GSO.Mat(A, update = True) #Al calcular la ortogonalizacion los vectores mas alla del indice n se quedan invariables
#         round_direction = []
#         for i in range(self.n):
#             mu = M.get_mu(self.n + i, i) # <target_i, b^*_i>/<b^*_i, b^*_i>
#             round_direction.append(w[i] > mu) # 1 si se aproxima hacia arriba y 0 en caso contrario

#         sign_step =  (-2*np.array(round_direction).astype(int)) + 1

#         dist_t = np.linalg.norm(res_vector)

#         return D, b_op, res_vector, sign_step, w, dist_t





#     #Funciones de la parte del QAOA
#     def define_qubo(self, D, residual_vector, step_signs): 
#         """
#         Genera una instancia de un problema Cuadratico de Qiskit.
#         Aqui se va a generar directamente el problema asociada a la funcion QUBO del problema.

#         D: Base del retículo LLL reducido.
#         residual_vector: vector residual obtenido de restar  t - b_op
#         step_signs: step_signs[i] = Sign(mu[i] - c[i]). Permite saber que si se ha aproximado hacia abajo el valor de mu
#                                                         explorar la aproximacion hacia arriba de mu, y al reves tambien.
        
#         return: QuadraticProgram()                                            

#         """

#         mdl = Model("quboProblem")

#         z = mdl.binary_var_list(self.n, name = "z")

#         objective = 0

#         #Itero sobre los n + 1 elementos de los vectores
#         for j in range (self.n + 1):

#             #Realizo la operacion que hay dentro del valor absoluto
#             #A cada posicion del vector residual t - bop le resto la suma del signo por la variable z y la posicion en concreto de
#             #vector de la base reducida
#             aux_ob = residual_vector[j] - mdl.sum([step_signs[i]*z[i]*D[i, j] for i in range(self.n)])

#             #Lo elevo al cuadrado
#             objective += aux_ob * aux_ob
        
#         #Lo convierto en un problema de minimizacion
#         mdl.minimize(objective)

#         #Realizo la conversion de un modelo docplex a un modelo qiskit.
#         mod = from_docplex_mp(mdl)
        
#         return mod


#     def define_hamiltonian(self, qubo_p):

#         op, offset = qubo_p.to_ising()

#         return op, offset
    
    
#     def construct_circuit(self, Hc, reps = 1):
#         circuit = qaoa_ansatz(cost_operator = Hc, reps = reps)
#         return circuit



#     def qaoa_algorithm(self, circuit, Hc, x0, min_method = 'Nelder-Mead'):
#         """
#         TODO
#         """

#         parameters  = circuit.parameters

#         simulator = EstimatorV2(options = {'backend_options': 
#                                            {'method': 'statevector'
#                                             }}) #Instancio el simulador exacto sin ruido

#         def func_to_minimize(x):

#             job = simulator.run([(circuit, Hc, x)])
#             result = job.result()[0]
            
#             energy  = float(result.data.evs)
#             #print(result.data.evs)
#             return energy
        

#         result = minimize(func_to_minimize, x0, method = min_method)

#         return {param.name: val for param, val in zip(parameters, result.x)}
    

#     def circ_asign_params(self,circuit, parameters):
#         ncircuit = circuit.assign_parameters(parameters)
#         return ncircuit
    
    
#     def sample_from_parameters(self, circuit, opt_parameters, shots):
#         """
#         TODO
#         """

#         sampler = SamplerV2() #Declaro un Sampler exacto


#         isa_circuit = self.pass_manager.run(circuit)
        
#         isa_circuit.measure_all()

#         parameter_values = [opt_parameters[p.name] for p in isa_circuit.parameters]

#         pub = (isa_circuit, parameter_values)


#         job = sampler.run([pub], shots = shots)
        
#         result = job.result()[0] 

#         counts = result.data.meas.get_counts() #Obtengo un diccionario [bitstring : frecuencia]

#         ordered_counts = dict(sorted(counts.items(), key = lambda x: x[1], reverse = True))

#         return ordered_counts
    
    
#     def bitstrings2vector(self, bitstrings):
#         return np.array([[int(c) for c in reversed(bstring)] for bstring in bitstrings])

#     def bitstring2latticeVectors(self, D, state_bistrings, step_signs, b_op):
#         """
        
#         """
#         bits = self.bitstrings2vector(state_bistrings) #Obtengo el vector de bits de la cadena

#         aux = np.multiply(step_signs, bits)

#         movement = aux @ D
        

#         vnew = b_op + movement

#         return vnew

#     def get_distances(self, vnew, t):
#         res_vectors = np.subtract(vnew, t)

#         distances = [np.linalg.norm(vector) for vector in res_vectors]

#         return distances
    

#     def get_distances2(self, vnew, t):
#         res_vector = np.subtract(vnew, t)

#         distances2 = [np.dot(vector, vector) for vector in res_vector]

#         return distances2


#     def is_smooth(self, u):

#         for p in self.prime_basis:
#             while u % p == 0:
#                 u //= p
#         return u == 1


#     def vectors2uv_pairs(self, B, vectors):
#         first_n_primes = primes[:self.n]

#         B_inv = np.linalg.pinv(B)

#         exponentes = np.rint(vectors @ B_inv).astype(int) #Obtengo el vector con los exponentes

#         u_exp = np.where(exponentes > 0, exponentes, 0) #Obtengo los exponentes positivos
#         v_exp = np.where(exponentes < 0, -exponentes, 0) #Obtengo los exponentes negativos y les cambio de signo

#         #Obtengo los valores u y v
#         u = np.prod(np.power(first_n_primes, u_exp, dtype = object), axis = 1) 
#         v = np.prod(np.power(first_n_primes, v_exp, dtype = object), axis = 1)
        

#         return np.stack((u, v), axis = 1)


#     def uv_pairs2sr_pairs(self, uv_pairs):

#         sr_pairs = [tuple(u_v) for u_v in uv_pairs if self.is_smooth(abs(int(u_v[0]) - self.N*int(u_v[1])))]

#         return sr_pairs

#     def get_probs(self, counts, shots):

#         return [count/shots for count in counts]
    
#     def prettyprint(self, vnews, distances, probs, b_op, t, res_vector, distT_B):
#         print(f"Vector más corto por algoritmo de Babai: \nb_op = {b_op}\n")
#         print(f"El vector residual \nt - b_op = {res_vector}\n")
#         print(f"La distancia |t - b_op| = {distT_B:.3f}\n")

#         total_count = len(vnews)

#         for i in range(total_count):
#             print(f"{i}: Prob = {probs[i]:.5f}\n\tvnew = {vnews[i]} con distancia: {distances[i]:.3f}")
        
        

#     #Getters
#     def get_N (self):
#         return self.N
#     def get_c(self):
#         return self.c
#     def get_l(self):
#         return self.l
#     def get_random_seed(self, seed):
#         return self.seed
#     def get_n (self):
#         return self.n
#     def get_smoothbound(self):
#         return self.smooth_bound
    

#     #Setters
#     def set_random_seed(self, seed):
#         self.seed = seed
#         np.random(self.seed)

#     def set_n(self, n):
#         self.n = n
    
#     def set_smoothbound(self, smoothbound):
#         self.smooth_bound = smoothbound
#         self.prime_basis = primes[:self.smooth_bound]


    

