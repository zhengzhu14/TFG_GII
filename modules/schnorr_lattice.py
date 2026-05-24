#Importaciones basicas
import numpy as np
from copy import deepcopy
from fpylll import IntegerMatrix, LLL, GSO
from math import log2, ceil
from dataclasses import dataclass


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

    def __init__(self, N, c, l, seed, set_seed = True, verbose = True):
        self.N = N

        self.c = c

        self.seed = seed
        if set_seed:
            np.random.seed(seed)
        
        

        self.l = l

        self.m = int(ceil(log2(self.N)))
        self.n = int(round((self.l*log2(self.N)) / log2(log2(self.N))))

        self.smooth_bound = 2*self.n**2

        self.basis = get_primes(self.smooth_bound)


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
            f = np.asarray(diagonal)

        # Crear una matriz de 0s y añadir en la diagonal la permutación escogida
        B = np.zeros(shape=(self.n, self.n))
        np.fill_diagonal(B, f)

        # Crear la ultima fila de la matriz

        basis = get_primes(self.n)
        final_row = np.round(q ** self.c * np.log(np.array(basis)))
       
        B = np.vstack((B, final_row))
        
        # fpylll solo acepta listas
        B = [[int(b) for b in bs] for bs in B]

        # Convertir B a una matriz del tipo IntegerMatrix de la libreria fpylll
        B = IntegerMatrix.from_matrix(B)

        # Definir vector objetivo
        t = np.zeros(self.n + 1)
        t[-1] = np.round(q ** self.c * np.log(float(self.N)))
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


    def is_smooth(self, u):
        for p in self.basis:
            while u % p == 0:
                u //= p
        return u == 1
    
    def get_factors(self, u):
        factor_list = [0]
        if u < 0:
            factor_list[0] = 1
        for p in self.basis:
            count = 0
            while u % p == 0:
                count += 1
                u //= p
            
            factor_list.append(count)
        return factor_list
    
    def get_valor_by_factors(self, factors):
        Y = 1
        for i, p in enumerate(self.basis):
            Y = Y * pow(p, int(factors[i + 1]), self.N)
        
        Y = Y % self.N
        
        return Y



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
        np.random.seed(self.seed)

    def set_n(self, n):
        self.n = n
    
    def set_smoothbound(self, smoothbound):
        self.smooth_bound = smoothbound
        self.basis = get_primes(self.smooth_bound)



def get_primes(n):
    if n < 1:
        return []
    primes = [2]
    current = 3
    while len(primes) < n:
        es_primo = True
        for p in primes:
            if p * p > current:   
                break
            if current % p == 0:  
                es_primo = False
                break
        if es_primo:
            primes.append(current)
        current += 2
    return primes

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




#Funciones de teoria de numeros

def vectors2uv_pairs(B, vectors, n):
    base = get_primes(n)

    B_inv = np.linalg.pinv(B) #Calculo la matriz inversa

    exponentes = np.rint(vectors @ B_inv).astype(int) #Obtengo el vector con los exponentes

    u_exp = np.where(exponentes > 0, exponentes, 0) #Obtengo los exponentes positivos
    v_exp = np.where(exponentes < 0, -exponentes, 0) #Obtengo los exponentes negativos y les cambio de signo

    #Obtengo los valores u y v
    u = np.prod(np.power(base, u_exp, dtype = object), axis = 1) 
    v = np.prod(np.power(base, v_exp, dtype = object), axis = 1)
    

    return np.stack((u, v), axis = 1)



def uv_pairs2sr_pairs(uv_pairs, cvp: schnorrCVP):

    aux = [int(u_v[0]) - cvp.N*int(u_v[1]) for u_v in uv_pairs]

    sr_pairs = [(tuple(u_v), sr) for u_v, sr in zip(uv_pairs, aux) if cvp.is_smooth(abs(sr))]

    return sr_pairs
    


