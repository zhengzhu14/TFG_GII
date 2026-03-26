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


"""En este modulo se creara una clase SchnorrAlgQAOA que centraliza todas las operaciones para implementar 
el algorimo de Schnorr potenciado mediante QAOA que en nuestro caso sera simulado"""

#Una lista de los 1000 primeros numeros primos
primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919]

class SchnorrAlgQAOA:
    #TODO
    def __init__(self, N, c, l, seed):
        self.N = N

        self.c = c

        self.seed = seed
        np.random.seed(self.seed)

        self.l = l
        
        self.m = np.round(np.log2(self.N)).astype(int)
        self.n = np.round(self.l * self.m // np.log2(self.m)).astype(int)
        self.smooth_bound = self.n**2

        self.prime_basis = primes[:self.smooth_bound]

        self.pass_manager = generate_preset_pass_manager(3, AerSimulator())


        print(f'El numero de bits de N = {N} es m = {self.m}')
        print(f'La dimension del reticulo que vamos a tratar es n = {self.n}')
        print(f'La cota smooth que vamos a tomar: {self.smooth_bound}')

    
    #Metodos privados
    def _lll_reduction_personal(self, B, delta = 0.75):
        #TODO
        pass


    #Metodos publicos
    def generate_cvp(self, q):
        """
        Generar una instancia aleatoria del cvp asociado al problema 
        de factorizar N.

        param q: valor con el que generar el ultimo valor de los vectores

        return: IntegerMatrix
        """

        f = np.random.permutation([(i + 1) // 2 for i in range(1, self.n + 1)])

        # Crear una matriz de 0s y añadir en la diagonal la permutación escogida
        B = np.zeros(shape=(self.n, self.n))
        np.fill_diagonal(B, f)

        # Crear la ultima fila de la matriz
        final_row = np.round(q ** self.c * np.log(np.array(primes[:self.n])))
        B = np.vstack((B, final_row))
        
        # fpylll solo acepta listas
        B = [[int(b) for b in bs] for bs in B]

        # Convertir B a una matriz del tipo IntegerMatrix de la libreria fpylll
        B = IntegerMatrix.from_matrix(B)

        # Definir vector objetivo
        t = np.zeros(self.n + 1)
        t[-1] = np.round(q ** self.c * np.log(self.N))
        t = tuple(t.astype(int).tolist())

        return B.transpose(), t
    
    def integer_to_matrix(self, B):
        """
        Convierte una matriz B de la clase IntegerMatrix
        
        param B: Base de la clase IntegerMatrix n x (n + 1) es decir ya transpuesta

        return np.array()
        """
        rows, cols = B.nrows, B.ncols
        A = np.zeros((rows, cols), dtype = int)
        B.to_matrix(A)

        return A

    
    def lll_reduced(self, B, delta = 0.75, fpylll = True):
        if fpylll:
            D = deepcopy(B)
            LLL.reduction(D, delta)
            return D
        else:
            Bmatrix = self.integet_to_matrix(B)
            return self._lll_reduction_personal(Bmatrix, delta)

    def babai_algorithm(self, B, t, delta = 0.75):
        """
        Calcula el vector cercano aproximado mediante el algoritmo de babai
        param B: 
        """
        D = deepcopy(B) # D está transpuesta
        _ = LLL.reduction(D, delta) #Obtengo la matriz reducida

        G = GSO.Mat(D, update = True)
        w = G.babai(t) # Obtengo los pesos del algoritmo de Babai

        b_op = np.array(D.multiply_left(w)) # w @ D

        res_vector = np.array(t) - b_op


        #Necesito obtener hacia donde se aproxima cada peso del algoritmo de Babai
        #Con este módulo necesito crear una matrix del doble de tamanyo

        A = IntegerMatrix(2*self.n, self.n + 1) 

        for i in range(self.n):
            for j in range(self.n + 1):
                A[i, j] = D[i, j]

        b = np.array(t)
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

        sign_step =  (-2*np.array(round_direction).astype(int)) + 1

        dist_t = np.linalg.norm(res_vector)

        return D, b_op, res_vector, sign_step, w, dist_t


    #Funciones de la parte del QAOA
    def define_qubo(self, D, residual_vector, step_signs): 
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

        z = mdl.binary_var_list(self.n, name = "z")

        objective = 0

        #Itero sobre los n + 1 elementos de los vectores
        for j in range (self.n + 1):

            #Realizo la operacion que hay dentro del valor absoluto
            #A cada posicion del vector residual t - bop le resto la suma del signo por la variable z y la posicion en concreto de
            #vector de la base reducida
            aux_ob = residual_vector[j] - mdl.sum([step_signs[i]*z[i]*D[i, j] for i in range(self.n)])

            #Lo elevo al cuadrado
            objective += aux_ob * aux_ob
        
        #Lo convierto en un problema de minimizacion
        mdl.minimize(objective)

        #Realizo la conversion de un modelo docplex a un modelo qiskit.
        mod = from_docplex_mp(mdl)
        
        return mod


    def define_hamiltonian(self, qubo_p):

        op, offset = qubo_p.to_ising()

        return op, offset
    
    
    def construct_circuit(self, Hc, reps = 1):
        circuit = QAOAAnsatz(cost_operator = Hc, reps = reps)
        return circuit



    def qaoa_algorithm(self, circuit, Hc, x0, min_method = 'Nelder-Mead'):
        """
        TODO
        """

        isa_circuit = self.pass_manager.run(circuit)

        parameters = isa_circuit.parameters
        #Obtengo parametros = ParameterView([Parameter(beta[]0),.., Parameter(alfa[0]),.., Parameter[alfa[n - 1]]])

        #num_parameters = circuit.num_parameters

        simulator = EstimatorV2(options = {'backend_options': 
                                           {'method': 'statevector'
                                            }}) #Instancio el simulador exacto sin ruido

        def func_to_minimize(x):

            job = simulator.run([(isa_circuit, Hc, x)])
            result = job.result()[0]
            
            #print(result.data.evs)
            return result.data.evs
        
        #x0 = np.asarray([0.0]*num_parameters) #Parametros iniciales

        result = minimize(func_to_minimize, x0, method = min_method)

        return {param: val for param, val in zip(parameters, result.x)}
    

    def circ_asign_params(self,circuit, parameters):
        ncircuit = circuit.assign_parameters(parameters)
        return ncircuit
    
    
    def sample_from_parameters(self, circuit, opt_parameters, shots):
        """
        TODO
        """

        sampler = StatevectorSampler() #Declaro un Sampler exacto

        circWithParam = circuit.assign_parameters(opt_parameters)

        circWithParam.measure_all()

        job = sampler.run([(circWithParam, None)], shots = shots)
        
        result = job.result()[0] 

        counts = result.data.meas.get_counts() #Obtengo un diccionario [bitstring : frecuencia]

        ordered_counts = dict(sorted(counts.items(), key = lambda x: x[1], reverse = True))

        return ordered_counts
    
    
    def bitstrings2vector(self, bitstrings):
        return np.array([[int(c) for c in bstring] for bstring in bitstrings])

    def bitstring2latticeVectors(self, D, state_bistrings, step_signs, b_op):
        """
        
        """
        bits = self.bitstrings2vector(state_bistrings) #Obtengo el vector de bits de la cadena

        aux = np.multiply(step_signs, bits)

        movement = aux @ D
        

        vnew = b_op + movement

        return vnew

    def get_distances(self, vnew, t):
        res_vectors = np.subtract(vnew, t)

        distances = [np.linalg.norm(vector) for vector in res_vectors]

        return distances


    def is_smooth(self, u):

        for p in self.prime_basis:
            while u % p == 0:
                u //= p
        return u == 1


    def vectors2uv_pairs(self, B, vectors):
        first_n_primes = primes[:self.n]

        B_inv = np.linalg.pinv(B)

        exponentes = np.rint(vectors @ B_inv).astype(int) #Obtengo el vector con los exponentes

        u_exp = np.where(exponentes > 0, exponentes, 0) #Obtengo los exponentes positivos
        v_exp = np.where(exponentes < 0, -exponentes, 0) #Obtengo los exponentes negativos y les cambio de signo

        #Obtengo los valores u y v
        u = np.prod(np.power(first_n_primes, u_exp, dtype = object), axis = 1) 
        v = np.prod(np.power(first_n_primes, v_exp, dtype = object), axis = 1)
        

        return np.stack((u, v), axis = 1)


    def uv_pairs2sr_pairs(self, uv_pairs):

        sr_pairs = [tuple(u_v) for u_v in uv_pairs if self.is_smooth(abs(int(u_v[0]) - self.N*int(u_v[1])))]

        return sr_pairs

    def get_probs(self, counts, shots):

        return [count/shots for count in counts]

    #Getters
    def get_N (self):
        return self.N
    def get_c(self):
        return self.c
    def get_l(self):
        return self.l
    def get_random_seed(self, seed):
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


    

