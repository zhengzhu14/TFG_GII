# Optimización Cuántica para el Problema del Vector más Cercano en Criptografía de Rejillas

Trabajo de fin de grado de Ingeniería Informática 2025/2026

## Instalación

### Requistos
- Python 3.12 o superior
- Sistema operativo **Linux**. El proyecto usa dependencias que no están disponibles en Windows a menos que se haga uso de un **WSL**. Se desconoce si está disponible para macOS.

### Pasos
1. **Clona el repositorio** y entra en la carpeta del proyecto:

    ```bash
    #git clone <URL_DEL_REPOSITORIO>
    cd TFG_GII
    ```

2. **Crea y activa el entorno virtual** (`.venv`):
    
    En el caso de este trabajo se utiliza el entorno `.venv`, pero puede ser cualquier otro.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ``` 
3. **Instala las dependencias y los módulos del proyecto**:
   ```bash
   #Instalamos las librerías del proyecto
   pip install -r requirements.txt

   #Instalamos los módulos propios de la carpeta ./modules/
   pip install -e .
   ```
    > Con estos comandos se instalan todas las **librerías externas** como los **módulos propios**.
    
### Desactivar el entorno
 
Cuando termines de trabajar, puedes salir del entorno virtual con:
 
```bash
deactivate
```

## Partes del proyecto

### 1. P1_introduccion
Implementación del pipeline del algoritmo de Schnorr y el refinamiento cuántico por QAOA de Yan et al.
- IntroQiskit_SchnorrAlg_1: implementación propia de todas las partes del algoritmo de factorización se Schnorr. La formulación QAOA asociada al problema CVP mediante el framework Qiskit.
- IntroQiskit_SchnorrAlg_2_fpylllModule: uso de la libreria fpylll para la reducción LLL y el algoritmo de Babai.
- IntroQiskit_SchnorrAlg_3_NoisySimulation: implementacion del algoritmo mediante simulación por Qiskit con ruido.

### 2. P2_casos_ejemplo
Evaluación de la implementación con Qiskit sobre 3 casos de ejemplo y la factorización de uno de ellos
- 3QubitCase: resolucion de una instancia de CVP para $N = 1961$ con 3 cúbits.
- 5QubitCase: resolucion de una instancia de CVP para $N = 48567227$ con 5 cúbits.
- 10QubitCase: resolución de una instancia de CVP para $N = 261980999226229$ con 10 cúbits.
- resolucion_caso_simple: implementación de todo el pipeline de factorización de $N = 48567227$ mediante el algoritmo de Schnorr-Yan.

### 3. P3_optimizar_QAOA
Análisis de optimizaciones sobre el algoritmo de QAOA
- 5Qubit_PruebasGrid: análisis del paisaje de optimización para el caso de 5 cúbits para profundidades de circuito $p = 1, 2$
- Estudio_norm_hamiltoniano: análisis de los paisajes de optmización tras normalizar los Hamiltoninos de coste para los casos de 3, 5 y 10 cúbits con profundidades de circuito Ansatze $p = 1,2$.
- Evaluacion_metodos_optimizacion: evaluación del funcionamiento de COBYLA frente a Nelder-Mead como optimizador clásico, tanto en la formulación original del problema como tras aplicar la normalización del Hamiltoniano de coste

### 4. P4_entreno_simple
Aplicación de extensiones sobre el modelo de preentreno de Priestley y Wallden.
- entreno_BenPriestley: aplicación de las extensiones al modelo de preentreno y la ejecución de 3 preentrenos para diferentes tamaños de conjuntos de entrenamiento y validación.
- obtencion_resultados: evaluación de los ángulos óptimos obtenidos durante el entreno para numerosas nuevas instancias. También se evalúa la tranferibilidad de los ángulos para instancias de mayor tamaño que no se han considerado durante los entrenos.
- analisis_resultados: extracción de gráficas de los resultados del notebook anterior para analizar la calidad de las soluciones y cómo se degrada la probabilidad de obtener mejores soluciones que las obtenidas clásicamente respecto de la función
$\frac{1}{2^{\alpha \cdot n}}$,
donde $n$ es la dimensión del retículo y $\alpha$ es el factor de escalado, que cuanto menor sea, mejor escala los ángulos obtenidos.

### Carpeta *./modules/*  

Carpeta con módulos donde se implementa las funciones utilizadas durante todo el proyecto:
- schnorr_lattice.py: funciones para realizar el pipeline clásico del algoritmo de criba mediante CVP.
- qaoa.py: funciones para implementar el algoritmo de QAOA mediante Qiskit.
- utils.py: funciones auxiliares para obtener más información de los datos devueltos en los algoritmos.
- functions.py: funciones donde se ejecuta el pipeline del algoritmo de criba por CVP refinado por QAOA y también con ángulos fijos.

