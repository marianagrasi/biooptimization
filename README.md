# Biooptimization: ACO + SSA + RBF + MongoDB

Actividad de la clase de Inteligencia Computacional que implementa
desde cero tres algoritmos bioinspirados y los aplica a un problema
de optimización y regresión, registrando cada corrida en MongoDB Atlas.

## Estructura del proyecto
biooptimization/
├── src/
│ ├── __init__.py
│ ├── aco.py # Ant Colony Optimization
│ ├── ssa.py # Sparrow Search Algorithm
│ ├── rbf.py # Radial Basis Function Network
│ ├── db.py # Conexión y operaciones MongoDB
│ └── experiment.py # Orquestador de experimentos
├── tests/
│ ├── __init__.py
│ └── test_basico.py # Pruebas automáticas
├── data/
│ └── Cities.csv # Dataset con ciudades europeas
├── results/ # Gráficas y CSV generados
├── requirements.txt
└── README.md


## Requisitos

- Python 3.11
- MongoDB Atlas (o local)
- Dependencias en `requirements.txt`

## Instalación

```bash
# 1. Crear entorno virtual
py -3.11 -m venv venv

# 2. Activar (Windows)
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar URI de MongoDB en src/db.py

# Ejecutar pruebas
pytest tests/ -v

# Ejecutar experimentos completos
python src/experiment.py
