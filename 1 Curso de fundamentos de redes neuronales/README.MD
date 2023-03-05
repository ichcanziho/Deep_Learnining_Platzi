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

El objetivo de esta y la próxima clase serán utilizar el dataset de MNIST el cual contiene 60,000 imágenes de números escritos
a mano, cada imagen es de 64x64 píxeles y a su vez cada imagen contiene una etiqueta con el valor real del número.

Empecemos por definir los objetivos de esta práctica:

1. Utilizar datasets pre-cargados de Keras
2. Familiarizarnos con los shapes de los datos de entrenamiento y validación de un dataset de DL
3. Crear un simple Modelo Secuencial de 1 capa y Multiples Salidas
4. Compilar el modelo con ciertos parámetros
5. Ver el resumen de la configuración de nuestro Modelo
6. Modificar los datos de entrenamiento y validación para hacerlos más manejables por el modelo de DL
7. Entrenar a nuestra red neuronal
8. Evaluar el rendimiento de nuestra red neuronal con datos de validación
9. Guardar el modelo para usarlo después

> El código completo de esta sección se encuentra [aquí](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/primer%20red%20neuronal/main.py)

Empecemos por cargar las librerías necesarias:

```python
# Estas librerias solo son necesario importarlas porque estoy corriendo de forma local el código
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Empiezan las librerias que vamos a utilizar para crear nuestro modelo de DEEP LEARNING
from keras import layers, models
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
```

### 1: Utilizar dataset pre-cargados de keras

```python
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
```

Keras cuenta con una sección de datasets para poder aprender DL, y sus datasets ya cuentan con una función muy útil llamada
`load_data()` la cuál permite cargar 4 particiones de la misma:

- `Train data`: Contiene datos de entrada para entrenar a tu modelo.
- `Train labels`: Contiene la clasificación de cada uno de los datos de `Train data`.
- `Test data`: Es una partición de datos con características similares a  `Train data` pero que el modelo de DL NO conoció durante el proceso de entrenamiento.
- `Test labels`: Corresponde a la clasificación de los datos de `Test data`.

### 2: Familiarizarnos con los shapes de los datos de entrenamiento y validación de un dataset de DL

Para cualquier ejercicio de DL es indispensable conocer la distribución de nuestros datos de entrenamiento y testing asi como
conocer la forma de cada uno de los elementos que lo componen, en este caso veremos 

```python
print("Train data shape:", train_data.shape)
print("Train data example shape:", train_data[0].shape)
print("Train label shape:", train_labels.shape)
print("Train label example shape:", train_labels[0].shape)
print("Test data shape:", test_data.shape)
print("Test data example shape:", test_data[0].shape)
print("Test label shape:", test_labels.shape)
print("Test label example shape:", test_labels[0].shape)
```
Respuesta esperada:
```
Train data shape: (60000, 28, 28)
Train data example shape: (28, 28)
Train label shape: (60000,)
Train label example shape: ()
Test data shape: (10000, 28, 28)
Test data example shape: (28, 28)
Test label shape: (10000,)
Test label example shape: ()
```

Podemos observar que nuestros datos de entrenamiento constan de un conjunto de 60,000 muestras de imágenes de 28x28, mientras que
el conjunto de testing es de 10,000 muestras. Podemos observar que tanto en train como test los datos en este caso imágenes 
tienen las mismas dimensiones. Adicionalmente, vemos como las `labels` de ambos conjuntos son arreglos unidimencionales, pues
solamente guardan la etiqueta con el nombre de la clase, en este caso un número.

Podemos observar cómo luce un valor del conjunto de entrenamiento tanto su dato como su label.

```python
plt.imshow(train_data[0])
plt.savefig("outputs/numero.png")
print(train_labels[0])
```

Respuesta esperada:
![numero](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/primer%20red%20neuronal/outputs/numero.png)
```commandline
5
```

### 3: Crear un simple Modelo Secuencial de 1 capa y Multiples Salidas

En este preciso momento del curso NO nos enfocaremos en dar mucha información de qué significa cada uno de los parámetros 
del objeto `model` de la clase `models.Sequential()` pero es un ejemplo ilustrativo de lo simple que puede ser crear modelos 
de DL con keras.

```python
model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(28*28, )))
model.add(layers.Dense(10, activation="softmax"))
```

En 3 simples líneas de código hemos definido la arquitectura de un modelo de DL para la clasificación de los números decimales.
En este momento del curso, lo único que debemos entender es lo siguiente: 
`input_shape(28*28, )`Indica que la entrada de la red neuronal tendré 728 neuronas, una para cada pixel de la imagen.
`layerse.Dense(10, )` El 10 es debido a que queremos que la última capa, la de clasificación clasifique entre 10 posibles
clases, los números del (0,9).

### 4: Compilar el modelo con ciertos parámetros

A pesar de que ya hemos creado la arquitectura del modelo de Deep Learning, aún es necesario indicar ciertos parámetros que 
modificaran la forma en que la red neuronal es entrenada, estros parámetros son:  `optimizer`, `loss`, `metrics`. Estos 
conceptos serán definidos más adelante. 

De forma muy simple: 
- `optimizer`: Es el algoritmo matemático que será utilizado para cambiar la distribución de pesos y bias de las neuronas. Este es el punto donde la red "va aprendiendo iteración tras iteración"
- `loss`: Es la forma que tenemos para definir que tan lejos o cerca estamos de nuestro objetivo a optimizar.
- `metrics`: Nos permite evaluar el rendimiento de nuestra red tanto en el training set como en el testing set.

Cada uno de estos parámetros tiene diferentes configuraciones, y más adelante serán explicados y definidos.

```python
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics="accuracy")
```

### 5: Ver el resumen de la configuración de nuestro Modelo

Algo realmente útil en Deep Learning es poder ver un resumen de la arquitectura de nuestra red neuronal.

```python
print(model.summary())
```

Respuesta esperada:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 512)               401920    
                                                                 
 dense_1 (Dense)             (None, 10)                5130      
                                                                 
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
```
### 6: Modificar los datos de entrenamiento y validación para hacerlos más manejables por el modelo de DL
Para este momento nosotros ya tenemos el modelo listo para empezar a ser entrenado. Sin embargo, antes de ponerles los datos
de data train y data train label podemos hacer un par de ajustes matemáticos que permitan a la red trabajar de mejor manera
con los datos de entrada.

```python

x_train = train_data.reshape((60000, 28*28))
x_train = x_train.astype("float32")/255

x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype("float32")/255

print(x_train[0].shape)
```
Respuesta esperada:
```
(784,)
```

En esta clase no se explica realmente el porqué de estas transformaciones, sin embargo la forma trivial de explicar
por qué hacemos esto es la siguiente:

Al momento de crear la arquitectura de nuestra red, la entrada de información tiene que ser un vector unidimensional. Sin embargo,
directamente nuestras imágenes son de 2 dimensiones, por eso el primer paso es transformar la forma del `train_data` a una forma
que tenga `60,000` muestras, pero cada una de esas muestras sea un vector de 1 dimensión de `784` valores.

Para las imágenes del dataset que estamos usando, están codificadas a 8 bits, eso significa que la cantidad de niveles de gris posibles son 2^8 = 256, por tanto, el mínimo nivel es 0 (negro puro) y 255 (blanco puro). Es importante resaltar que existen imágenes a 16 bits (por ejm las imágenes médicas de mamografías), en este caso se tendría 2^16 = 65536 niveles de gris, que irían desde 0 (negro puro) hasta 65535 (blanco puro).

Dado entonces que sabemos que las imágenes tienen 8 bits, podemos normalizar los datos de entrada dividiendo entre el número
más grande de esta forma pasamos de una escala de [0, 255] a una de [0, 1] y las redes neuronales trabajan más cómodamente 
con números en decimal que con enteros. 

## 1.5 Entrenando el modelo de tu primer red neuronal

### 7: Entrenar a nuestra red neuronal
Ahora que nuestros datos de entrenamiento y testing tienen un mejor formato, podemos entrenar a nuestra red neuronal.
Lo único que cabe destacar aquí por el momento es que: `epochs`: es la cantidad de iteraciones que queremos que el modelo 
pueda realizar para irse optimizando. `batch_size`: de forma muy sencilla, es que como el dataset es muy grande `60,000`
es más conveniente irse entrenando de forma paralela con conjuntos más pequeños de 128 datos que los 60,000 al mismo tiempo.

```python
model.fit(x_train, y_train, epochs=5, batch_size=128)
```
Respuesta esperada:
```
Epoch 1/5
469/469 [==============================] - 2s 1ms/step - loss: 0.2679 - accuracy: 0.9237
Epoch 2/5
469/469 [==============================] - 1s 1ms/step - loss: 0.1073 - accuracy: 0.9686
Epoch 3/5
469/469 [==============================] - 1s 2ms/step - loss: 0.0709 - accuracy: 0.9793
Epoch 4/5
469/469 [==============================] - 1s 1ms/step - loss: 0.0499 - accuracy: 0.9851
Epoch 5/5
469/469 [==============================] - 1s 1ms/step - loss: 0.0385 - accuracy: 0.9883
```

### 8: Evaluar el rendimiento de nuestra red neuronal con datos de validación

Ahora que hemos visto que el accuracy de nuestro modelo sobre los datos de entrenamiento es de 0.9883 vamos a ponerlo a prueba
con los datos de testing que NO ha visto mientras era entrenado.

```python
model.evaluate(x_test, y_test)
```
Valor esperado:
```
313/313 [==============================] - 0s 853us/step - loss: 0.0710 - accuracy: 0.9773
```
Excelente nuestro modelo ha obtenido `0.9773` puntos de accuracy sobre un dataset desconocido.

### 9: Guardar el modelo para usarlo después

Finalmente como paso extra, podemos exportar el modelo con todos los pesos, arquitectura y su compilación para cargar después
y ponerlo a clasificar lo que nosotros queramos.

```python
model.save("Model/numeros.h5")
```

Ahora en un archivo diferente podemos cargar el modelo y ponerlo a clasificar:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

model = load_model("Model/numeros.h5")
(_, _), (test_data, test_labels) = mnist.load_data()
x_test = test_data.reshape((10000, 28*28))
x_test = x_test.astype("float32")/255
y_test = to_categorical(test_labels)
model.evaluate(x_test, y_test)
```

Respuesta esperada:
```
313/313 [==============================] - 1s 854us/step - loss: 0.0710 - accuracy: 0.9773
```

## 1.6 La neurona: una pequeña y poderosa herramienta

La neurona, también llamado perceptrón (nacido en los años 50’s) está inspirado en las redes neuronales biológicas.

El funcionamiento del perceptrón se describe de la siguiente manera:

- Se realiza una suma ponderada de las entradas con los pesos (weights w). Esto da como resultado una salida lineal.

- Esta salida se pasa por una función de activación que introduce no linealidades al perceptrón.

- Si el modelo no satisface de forma adecuada el problema entonces se itera. Se itera actualizando los pesos hasta resolver el problema.

![neurona1](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/neurona1.png)

Vemos como la Neurona simplemente toma variables de entrada `X1 - X2` a cada una de estas variables la acompaña su respectivo peso
`WX1 - WX2` La neurona es la suma ponderada entre la multiplicación de las entradas `Xs` y los pesos `Ws` adicionalmente a esta 
secuencia de sumas se le suma un elemento de Bias `b` que puede estar o no acompañado de su peso `Wb`.

![neurona2](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/neurona2.png)

El bias de la neurona le permite a la misma tener más elasticidad. Por ejemplo si tenemos una función que en un punto específico
da 0 (véase) la función 2x^2 cuando X es 0 entonces f(x) es 0; sin embargo, si le añadimos un Bias la función va a estar recorrida
B unidades.

![neurona3](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/neurona3.png)

Las neuronas por sí mismas nos permiten solucionar problemas, por ejemplo con una correcta distribución de Pesos y Bias
podemos crear una compuerta lógica AND asignado los valores de (2, 1, -3) a sus (WX1, WX2, BX) respectivamente. Sin embargo,
cabe destacar que es Necesaria una `Función de activación` que me permite `deformar` la salida lineal. De esta manera nuestra función
puede ser algo como `0 if z <=0 else 1`de modo que si z es negativa o 0 entonces la salida será 0 de otro modo será 1.

![escalon](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/escalon.png)

Sin embargo, éxisten muchos problemas que NO se pueden solucionar de forma lineal con una sola neurona, tal es el caso de la función
XOR:

![neurona4](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/neurona4.png)

La cual vemos que para poder clasificar de forma correcta, necesita al menos 2 funciones lineales, es decir 2 neuronas:

![neurona5](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/neurona5.png)

Esto quiere decir que a mayor cantidad de neuronas tengo mayor posibilidad de responder problemas más complejos. Esto NO es 100% cierto
para el 100% de los casos, claro que existen problemas que con una menor cantidad de neuronas pueden resolver de forma más
eficiente ciertos problemas, pero en general tener mayor cantidad de neuronas facilita la tarea, y en algunos problemas como
el XOR es indispensable tener varias. 

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
