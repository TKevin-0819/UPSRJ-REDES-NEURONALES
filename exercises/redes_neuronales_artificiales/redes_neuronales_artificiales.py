from exercises.redes_neuronales_artificiales.single_perceptron import InputData, Perceptron
# Adjust the import path to include the parent directory for py_utils
import sys
import os
from logging import DEBUG, INFO, WARNING, ERROR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from py_utils.logger import set_logging, plog

# Llamada a la funcion para configurar el logging
set_logging(log_file='redes_neuronales_artificiales.log')

# ======================================================================
# Ejercicio 1: Red de una neurona.
#   x ──► [ Neurona ] ──► a
#              ↑
#            (w, b)
# ======================================================================

# Crear entrada
entrada = InputData(1.0)  # <-- remplaza "valor" con un número (ej. 1.0)

# Crear perceptrón
neurona = Perceptron(weight=1, bias=1)  # <-- pon aquí w y b

# Calcular salida
single_layer_perceptron = neurona(entrada)

plog(f"Salida de la red de una sola capa: {single_layer_perceptron}",
     level=ERROR if single_layer_perceptron is None else DEBUG, eol=True)

# ======================================================================
# Ejercicio 2: Red de dos neuronas en serie y una entrada.
#   x1 ──► [ Neurona 1 ] ──► a1 ──► [ Neurona 2 ] ──► a2
#              ↑                          ↑
#            (w1, b1)                   (w2, b2)
# ======================================================================

entrada = InputData(valor)  # <-- remplaza con un valor de entrada

# Primera neurona
neurona1 = Perceptron(weight=w1, bias=b1)
a1 = neurona1(entrada)

# Segunda neurona, recibe la salida de la primera
neurona2 = Perceptron(weight=w2, bias=b2)
two_layer_network = neurona2(InputData(value=a1))

plog(f"Salida de la red de dos capas: {two_layer_network}",
     level=ERROR if two_layer_network is None else DEBUG, eol=True)

# ======================================================================
# Ejercicio 3: Red de dos neuronas en paralelo con dos entradas.
#
# Entrada         Capa Oculta          Salida
#   x1 ─────────► [ Neurona 1 ]
#   x2 ─────────► [ Neurona 2 ] ───► [ Neurona salida ] ──► a_final
# ======================================================================

x1 = InputData(1)   # <-- pon valor de entrada 1
x2 = InputData(0.1)   # <-- pon valor de entrada 2

# Neuronas de la capa oculta
neurona1 = Perceptron(weight=w1, bias=b1)
neurona2 = Perceptron(weight=w2, bias=b2)

# Salidas de la capa oculta
a1 = neurona1(x1)
a2 = neurona2(x2)

# Neurona de salida que combina ambas
neurona_salida = Perceptron(weight=w3, bias=b3)
small_network = neurona_salida(InputData(value=a1 + a2))

plog(f"Salida de la red pequeña: {small_network}",
     level=ERROR if small_network is None else DEBUG, eol=True)
