# Curso de Fundamentos de Redes Neuronales con Python y Keras

Las redes neuronales se utilizan en deep learning para generar predicciones, análisis de sentimiento y otros análisis de texto, voz e imagen cuando tenemos muchos datos. Aprende cómo funcionan y cómo empezar a utilizarlas en tus proyectos en ciencia de datos.

- Crea una red neuronal con Python.
- Utiliza Keras para el uso profesional de redes neuronales.
- Conoce los modelos principales de redes neuronales.
- Comprende la estructura y matemática que hay detrás de una red neuronal.

# Índice:

- [1. Fundamentos en la arquitectura de redes neuronales](#1-fundamentos-en-la-arquitectura-de-redes-neuronales)
  - [1.1 La importancia de las redes neuronales en la actualidad](#11-la-importancia-de-las-redes-neuronales-en-la-actualidad)
  - [1.2 ¿Qué herramientas usaremos para redes neuronales?](#12-qué-herramientas-usaremos-para-redes-neuronales)
  - [1.3 ¿Qué es Deep Learning?](#13-qué-es-deep-learning)
  - [1.4 Tu primer red neuronal con Keras](#14-tu-primer-red-neuronal-con-keras)
  - [1.5 Entrenando el modelo de tu primer red neuronal](#15-entrenando-el-modelo-de-tu-primer-red-neuronal)
  - [1.6 La neurona: una pequeña y poderosa herramienta](#16-la-neurona--una-pequeña-y-poderosa-herramienta)
  - [1.7 Arquitectura de una red neuronal](#17-arquitectura-de-una-red-neuronal)
  - [1.8 Funciones de activación](#18-funciones-de-activación)
  - [1.9 Función de pérdida (Loss Function)](#19-función-de-pérdida--loss-function-)
  - [1.10 Descenso del gradiente](#110-descenso-del-gradiente)
  - [1.11 Backpropagation](#111-backpropagation)
  - [1.12 Playground - Tensorflow](#112-playground---tensorflow)
  - [1.13 Quiz: Fundamentos en la arquitectura de redes neuronales](#113-quiz--fundamentos-en-la-arquitectura-de-redes-neuronales)
- [2. Redes Neuronales con Python](#2-redes-neuronales-con-python)
  - [2.1 Dimensiones, tensores y reshape](#21-dimensiones-tensores-y-reshape)
  - [2.2 Creando nuestra red neuronal usando numpy y matemáticas](#22-creando-nuestra-red-neuronal-usando-numpy-y-matemáticas)
  - [2.3 Entrenamiento forward de la red neuronal](#23-entrenamiento-forward-de-la-red-neuronal)
  - [2.4 Aplicando backpropagation y descenso del gradiente](#24-aplicando-backpropagation-y-descenso-del-gradiente)
  - [2.5 Entrenamiento y análisis de resultados de tu red neuronal](#25-entrenamiento-y-análisis-de-resultados-de-tu-red-neuronal)
  - [2.6 Quiz: redes neuronales con python](#26-quiz--redes-neuronales-con-python)
- [3. Manejo de Redes Neuronales con Keras](#3-manejo-de-redes-neuronales-con-keras)
  - [3.1 Data: Train, Validation, Test](#31-data--train-validation-test)
  - [3.2 Resolviendo un problema de clasificación binaria](#32-resolviendo-un-problema-de-clasificación-binaria)
  - [3.3 Entrenamiento del modelo de clasificación binaria](#33-entrenamiento-del-modelo-de-clasificación-binaria)
  - [3.4 Regularización - Dropout](#34-regularización---dropout)
  - [3.5 Reduciendo el overfitting](#35-reduciendo-el-overfitting)
  - [3.6 Resolviendo un problema de clasificación multiple](#36-resolviendo-un-problema-de-clasificación-multiple)
  - [3.7 Entrenamiento del modelo de clasificación multiple](#37-entrenamiento-del-modelo-de-clasificación-multiple)
  - [3.8 Resolviendo un problema de regresión](#38-resolviendo-un-problema-de-regresión)
  - [3.9 Entrenamiento del modelo de regresión](#39-entrenamiento-del-modelo-de-regresión)
  - [3.10 Análisis de resultados del modelo de regresión](#310-análisis-de-resultados-del-modelo-de-regresión)

# 1. Fundamentos en la arquitectura de redes neuronales

## 1.1 La importancia de las redes neuronales en la actualidad
### ¿Qué es una red neuronal?
Una red neuronal es un método de la inteligencia artificial que enseña a las computadoras a procesar datos de una manera 
que está inspirada en la forma en que lo hace el cerebro humano. Se trata de un tipo de proceso de machine learning llamado 
aprendizaje profundo, que utiliza los nodos o las neuronas interconectados en una estructura de capas que se parece al cerebro humano. 
Crea un sistema adaptable que las computadoras usan para aprender de sus errores y mejorar continuamente. De esta manera, 
las redes neuronales artificiales intentan resolver problemas complicados, como la realización de resúmenes de documentos o 
el reconocimiento de rostros, con mayor precisión.

### ¿Por qué son importantes las redes neuronales?
Las redes neuronales pueden ayudar a las computadoras a tomar decisiones inteligentes con asistencia humana limitada. 
Esto se debe a que pueden aprender y modelar las relaciones entre los datos de entrada y salida que no son lineales y que 
son complejos. Por ejemplo, pueden realizar las siguientes tareas.

**Hacer generalizaciones y sacar conclusiones**

Las redes neuronales pueden comprender datos no estructurados y hacer observaciones generales sin un entrenamiento explícito. 
Por ejemplo, pueden reconocer que dos oraciones de entrada diferentes tienen un significado similar:

- ¿Puede explicarme cómo hacer el pago?
- ¿Cómo puedo transferir dinero?

Una red neuronal sabría que ambas oraciones significan lo mismo. O sería capaz de reconocer, en términos generales, que 
Baxter Road es un lugar, pero que Baxter Smith es el nombre de una persona.

## 1.2 ¿Qué herramientas usaremos para redes neuronales?

En este curso nos estaremos enfocando principalmente en el uso de Keras para dearrollar nuestros proyectos de
redes neuronales. Así que debemos empezar por definir:

**¿Qué es Keras?**

Keras es una biblioteca de código abierto (con licencia MIT) escrita en Python, que se basa principalmente en el trabajo 
de François Chollet, un desarrollador de Google, en el marco del proyecto ONEIROS (Open-ended Neuro-Electronic Intelligent 
Robot Operating System). La primera versión de este software multiplataforma se lanzó el 28 de marzo de 2015. 
El objetivo de la biblioteca es acelerar la creación de redes neuronales: para ello, Keras no funciona como un 
framework independiente, sino como una interfaz de uso intuitivo (API) que permite acceder a varios frameworks de 
aprendizaje automático y desarrollarlos. Entre los frameworks compatibles con Keras, se incluyen Theano, Microsoft 
Cognitive Toolkit (anteriormente CNTK) y TensorFlow.



![keras_api](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/keras_api.png)

Entonces Keras no es más que una forma más amigable de acceder a Frameworks independientes, a su vez estos Frameworks
usaran libererías de más bajo nivel para comunicarse directamente con el Hardware de nuestro dispositivo para de esta
forma acceder y utilizar a la `GPU` o `CPU` de nuestro computador. 

El curso original de [Platzi](https://platzi.com/cursos/redes-neuronales/) propone utilizar Google Colab para desarrollar
todos los elemenos vistos en este curso, sin embargo, me tome la libertad de adaptar todo lo visto para correr el código 
de forma nativa en una computadora utilizando [venv](https://docs.python.org/es/3/library/venv.html) para gestionar la creación
de entornos virtuales en python.

Si quieres conocer más acerca de `VENV` y `entornos virtuales en python` te recomiendo visites otro de mis repositorios de
`Github` dónde encontraras más información sobre `¿Qué es un ambiente virtual?`. [Más información](https://github.com/ichcanziho/cursos_platzi/tree/master/pip_entornos_curso)


## 1.3 ¿Qué es Deep Learning?

`Inteligencia artificial` son los intentos de replicar la inteligencia humana en sistemas artificiales.

`Machine learning` son las técnicas de aprendizaje automático, en donde mismo sistema aprende como encontrar una respuesta sin que alguien lo esté programando.

`Deep learning` es todo lo relacionado con las redes neuronales. Se llama aprendizaje profundo porque a mayor capas conectadas ente sí se obtiene un aprendizaje más fino.

![deep learning](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/deep_learning_circle.png)

### Ciclo de Machine Learning vs Deep Learning

Los métodos tradicionales de `Machine Learning` tienen un parámetro de entrada, en este caso un `Tweet`, el ciclo convencional
nos indica el uso de `Feature Engineering` qué consiste en utilizar nuestro conocimiento sobre el negocio para transformar los datos
de entrada en formas más entendibles por el modelo de clasificación, limpieza de datos, selección de características 
y muchas otras herramientas que permiten al modelo de clasificación estar listo para trabajar con los datos de entrada.


![comparación](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/comparacion.png)

Por otro lado, los modelos basados en `Deep Learning` si bien aún necesitan una forma de convertir ciertos tipos de datos
no estructurados en representaciones estructuradas véase como convertir un texto en un vector de tamaño fijo por ejemplo. 

Sin embargo, la principal ventaja que nos permite `Deep Learning` es NO preocuparnos por limpiar u optimizar las variables de 
entrada una vez definidas, puestos que serán las propias neuronas del modelo las que automáticamente permitirán decidir la
importancia de cada una de nuestras variables de entrada.  Sin embargo, este tipo de pensamiento lleva a: **Principales problemas
de deep learning**


### Problemas de deep learning

- **Overfitting**. De forma reduccionista: Es cuando el algoritmo “memoriza” los datos y la red neuronal no sabe generalizar. 
- **Black box**. De forma reduccionista: Es cuando nosotros conocemos las entradas a las redes neuronales. Sin embargo, no conocemos que es lo que pasa dentro de las capas intermedias de la red.


## 1.4 Tu primer red neuronal con Keras
## 1.5 Entrenando el modelo de tu primer red neuronal
## 1.6 La neurona: una pequeña y poderosa herramienta
## 1.7 Arquitectura de una red neuronal
## 1.8 Funciones de activación
## 1.9 Función de pérdida (Loss Function)
## 1.10 Descenso del gradiente
## 1.11 Backpropagation
## 1.12 Playground - Tensorflow
## 1.13 Quiz: Fundamentos en la arquitectura de redes neuronales

# 2. Redes Neuronales con Python

## 2.1 Dimensiones, tensores y reshape
## 2.2 Creando nuestra red neuronal usando numpy y matemáticas
## 2.3 Entrenamiento forward de la red neuronal
## 2.4 Aplicando backpropagation y descenso del gradiente
## 2.5 Entrenamiento y análisis de resultados de tu red neuronal
## 2.6 Quiz: redes neuronales con python

# 3. Manejo de Redes Neuronales con Keras

## 3.1 Data: Train, Validation, Test
## 3.2 Resolviendo un problema de clasificación binaria
## 3.3 Entrenamiento del modelo de clasificación binaria
## 3.4 Regularización - Dropout
## 3.5 Reduciendo el overfitting
## 3.6 Resolviendo un problema de clasificación multiple
## 3.7 Entrenamiento del modelo de clasificación multiple
## 3.8 Resolviendo un problema de regresión
## 3.9 Entrenamiento del modelo de regresión
## 3.10 Análisis de resultados del modelo de regresión
