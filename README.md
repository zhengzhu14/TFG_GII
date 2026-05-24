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


