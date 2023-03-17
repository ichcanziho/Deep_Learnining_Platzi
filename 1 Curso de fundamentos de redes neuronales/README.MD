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
  - [3.8 Validación de nuestro modelo usando cross validation](#38-validación-de-nuestro-modelo-usando-cross-validation)
  - [3.9 Resolviendo un problema de regresión](#39-resolviendo-un-problema-de-regresión)
  - [3.10 Entrenamiento del modelo de regresión](#310-entrenamiento-del-modelo-de-regresión)
  - [3.11 Análisis de resultados del modelo de regresión](#311-análisis-de-resultados-del-modelo-de-regresión)
- [4. Cierre](#4-cierre)

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

La arquitectura de la red puede ser dividida en tres partes:

- La capa de entrada en donde los datos son introducidos.
- Las capas ocultas, que se encuentran entre la capa de salida y la capa de entrada. Las capas ocultas son quienes hacen las operaciones matemáticas.
- La cada de salida que hace una predicción

Dentro de la arquitectura de la red neuronal ocurren muchas operaciones de producto punto entre las entradas de cada perceptron con sus respectivos pesos. Estas operaciones son lineales.

Las funciones de activación son la solución al colapso de las linealidades de las capas de la red neuronal.

![red1](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/red0.png)

> ### Notas adicionales:

- Todas las capas que estan entre la capa de entrada y la de salida son conocidas como `hidden layers` sin importar cuantas sean.
- Las capas más cercanas a las de la entrada obtienen características más generales del problema, entre más profunda la capa
más específica es la característica aprendida.
- Las últimas capas tienden más al overfitting.

Nuestras redes neuronales van a tener varias `m` variables de entrada, a su vez una capa tendrá `n` cantidad de neuronas.
 
![red2](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/red2.png)

Podemos observar como entonces la cantidad de multiplicaciones está dada por `m x n` dicho de otra manera, todas las neuronas
se multiplican por todas las variables de la capa anterior.

![red3](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/red3.png)

Esta multiplicación es conocida en matemáticas como: `producto punto entre una matriz y un vector`

Finalmente, para tener nuestra salida de la multiplicación es necesario añadir el bias:

![red4](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/red4.png)

Sin embargo, hasta este preciso momento aún hay un área de oportunidad enorme:

![red5.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/imgs/red5.png)

Si yo comienzo a apilar varías capas de redes neuronales todas y cada una de ellas a su salida tienen una función lineal
y la suma de varias funciones lineales es otra función lineal, esto NO tiene mucho sentido porque entonces toda la información de
en medio pierde funcionalidad. La solución a esto es que las salidas de estas capas intermedias NO sea lineal, y para ello
usaremos `funciones de activación`.

## 1.8 Funciones de activación

Las `funciones de activación` permiten quitar la linealidad a nuestras salidas de las neuronas. Estas funciones pueden
ser **discretas** (tener un conjunto finito de valores) o **continuas** (estar dentro de un intervalo de valores)

Algunas de las funciones de activación más utilizadas son las siguientes:

![function1.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Ffunction1.png)

Pero ¿cómo sabemos qué función de activación utilizar en cada momento?

![function2.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Ffunction2.png)

Esto va a depender del tipo de problema, y siempre se puede jugar con las activaciones de cada capa, sin embargo, es normalmente
aceptado que las capas ocultas utilicen ReLU y la última dependa de si el problema es de clasificación binaria o multiple, 
por lo general si el problema tiene más de dos clases a ser clasificadas se utilizará `Softmax` o `Sigmoid` por otro lado
si es binaria puede ser `Sigmoid` o `step`, finalmente, si el problema es una regresión entonces basta con usar una `linear`

> ### Código de esta sección [aquí](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Ffunciones%2Fmain.py)

### Función Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
![Sigmoid.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales/funciones/imgs/Sigmoid.png)

### Función Step

```python
def step(x):
    return np.piecewise(x, [x < 0.0, x > 0.0], [0, 1])
```
![Step.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Ffunciones%2Fimgs%2FStep.png)

### Función ReLU

```python
def relu(x):
    return np.maximum(0, x)
```
![ReLu.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Ffunciones%2Fimgs%2FReLu.png)

### Función Tanh
```python
def tanh(x):
    return np.tanh(x)
```
![Tanh.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Ffunciones%2Fimgs%2FTanh.png)


## 1.9 Función de pérdida (Loss Function)

Es importante tener muy en cuenta que: nuestra red neuronal tiene como finalidad generar una `predicción` ya sea de una 
`regresion` (un valor continuo) o una `clasificación` (clases pre-definidas). 

Con esto en cuenta, entonces una pregunta completamente lógica es: ¿cómo sabemos si la predicción fue buena o fue mala, 
o qué tan cerca o lejos está del valor real?. Para responder esta pregunta, es que ahora cobra sentido que las redes neuronales
necesiten tener valores conocidos para tener un marco de referencia de a dónde llegar.

En el ejemplo pasado veíamos que nuestro dataset de MNIST tenía dos grandes grupos: 

- Training
  - data training
  - label training
- Testing
  - data testing
  - label testing

Justamente la `data` eran los valores de entrada y las `label` eran los valores de salida o valores a predecir. Es en este 
conjunto de valores donde encontramos los ejemplos de las respuestas correctas, que dado los valores de entrada deben llegar
a la salida esperada de `label`. 

Tomando todo esto en cuenta podemos decir que el objetivo de la red neuronal es utilizar los pesos y los bias de las capas
del modelo para generar una salida que sea lo más similar posible a las etiquetas esperadas en `label` y no solo a una etiqueta
sino a todas las etiquetas del conjunto de `training y testing.

Es aquí cuando entra el concepto de `loss function` una función que me permite comparar que tan cerca o lejos está mi red neuronal
respecto a mis valores reales. De acuerdo al tipo de problema existen diferentes `loss functions` pero todas tienen como objetivo
el permitirme observar que tan buena es mi red neuronal para predecir mi variable de salida. Con base en este `coste` puedo
ir actualizando los pesos de mis neuronas para que vayan reduciendo el coste. Esto lo veremos en la siguiente clase cuando
hablemos de la técnica de `gradient descend`.

En esta clase vamos a hablar sobre 2 de las funciones de perdida más utilizadas:

**MSE - MEAN SQUARED ERROR:**

Esta función de perdida está diseñada para problemas de regresiones, dónde queremos obtener un valor continuo como lo es
el valor de una casa por ejemplo. 
![f1.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Ff1.png)

La función `MSE` toma el cuadrado de la distancia entre el valor real y la predicción para castigar con más fuerza a los valores más 
alejados a mi predicción.

**CROSS ENTROPY**

Esta función de perdida está diseñada para problemas de clasificación, de forma muy simple mide la distancia entre la predicción de nuestro 
algoritmo contra el valor real, para cada una de las clases del problema.

![f2.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Ff2.png)

En este ejemplo podemos observar como el valor real a predecir era el círculo marcado como su representación `one hot encoding` como
`(1, 0, 0)` y la predicción del algoritmo fue `(0.5, 0.3, 0.2)` entonces la fórmula de `Cross Entropy` toma en cuenta el valor real
`p(x)` y el logaritmo de la predicción `log q(x)` de esta forma podemos saber que tan buena fue nuestra predicción.

En términos de programación de deep learning vamos a nombrar a nuestra predicción como `y_hat` y al valor real como `y`

Vamos a implementar la función de `MSE` en python:

> Código completo [perdida.py](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Ffunciones%2Fperdida.py)

```python
import numpy as np


def mse(y: np.array, y_hat: np.array, derivative: bool = False):
    if derivative:
        return y_hat - y
    else:
        return np.mean((y_hat - y)**2)


if __name__ == '__main__':
    real = np.array([0, 0, 1, 1])
    prediction = np.array([0.9, 0.5, 0.2, 0])
    print(mse(real, prediction))
``` 
Valor esperado: 
```
0.675
```
Más adelante explicaremos por qué tenemos el parámetro de `derivative` y porque lo que regresa es diferente cuando la función
está derivada respecto a cuando no lo está.

## 1.10 Descenso del gradiente

Matemáticamente hablando las funciones que son continuas también son derivables. Hablando en términos sumamente SIMPLES de entender
y NADA FORMALES. Intuitivamente, se puede decir que una función es continua cuando en su gráfica no aparecen saltos o cuando el trazo de la gráfica no tiene "huecos".

![descenso1.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso1.png)

Podemos decir que (a) y (b) NO son continuas, pero (c) sí lo es. Entonces, que una función sea continua me permite que sea 
diferenciable, y a su vez, de forma simple de explicar esto quiere decir que dado un punto que pertenezca a la función f(x)
yo soy capaz de encontrar una recta tangente a dicho punto (una recta que únicamente va a tocar en un solo punto a dicha función)

![descenso2.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso2.png)

**¿Y para qué me interesa obtener está recta tangente a un punto?** 

Imaginemos que yo tengo la siguiente función:

![descenso3.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso3.png)

Y me interesa encontrar en qué puntos la función tiene un valor máximo y un valor mínimo, en otras palabras estoy optimizando.
Puedo usar la derivada de la función para ir obteniendo la recta pendiente en todos y cada uno de los puntos de la función en
un rango establecido. La pendiente a su vez tendrá un cierto grado de inclinación, este puede ser una inclinación positiva o negativa
de acuerdo a si el punto de la función se encuentra en la parte creciente o decreciente de la misma. 

Sin embargo, únicamente en los puntos máximos y mínimos la pendiente de dicho punto tendrá una inclinación de 0

![descenso4.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso4.png)

En otras palabras, la derivada me permite encontrar los valores locales máximos y mínimos de una función cuando la derivada se hace 0.

Esto es increíblemente útil en deep learning porque recordemos que lo que nosotros buscamos con las funciones de perdida es
optimizarlas acercándolas lo más posible a 0, dicho de otra forma nuestro deseo con las funciones de perdida es que tengan
la menor perdida posible y es justo por eso que utilizamos las derivadas de las funciones para irnos acercando poco a poco a 
este objetivo de optimización. 

Entonces justamente el algoritmo del gradiente descendiente (gradient descend) tiene como objetivo dada una función encontrar
dónde se encuentra su mínimo local:

![descenso5.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso5.gif)

En este escenario hay un término muy importante conocido como `learning rate` que significa la velocidad de aprendizaje,
en términos sencillos es el tamaño del salto que mi algoritmo de `gradient descend` tiene permiso de dar en cada iteración. 

![descenso6.gif](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso6.gif)

Si él `learning rate` es MUY pequeño, me puede tomar muchísimos saltos el llegar a mi punto de optimización deseado. Pero si
él `learning rate` es demasiado grande entonces este tamaño de salto NO me va a permitir convergir a ninguno punto. 

Idealmente, debo tener un `learning rate` lo suficientemente pequeño para que me asegure de que puedo convergir, pero lo suficientemente
grande para que me tome la menor cantidad de iteraciones. En la actualidad existen métodos de optimización que NO necesitan
especificar este parámetro por defecto. Un learning rate utilizado con bastante frecuencia es: `0.001` 

Existe una gran variedad de optimizadores en la actualidad y lo más nuevos tienden a ser más eficientes y veloces al momento de encontrar
el mínimo de una función:

![descenso7.gif](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso7.gif)

Finalmente, para encontrar la dirección de la derivada es necesario entender el concepto de derivadas parciales:

![descenso8.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fdescenso8.png)

La derivada parcial de una función de varias variables es la derivada con respecto a cada una de esas variables manteniendo 
las otras como constantes. A final de cuentas en una red neuronal tengo funciones de muchísimas variables, y cada una de ellas
expresa una dimensión, entonces para encontrar el descenso del gradiente debo obtener la derivada para cada una de las dimensiones
del problema. En realidad la derivada va a darme la dirección en la que crece la función, pero si yo quiero obtener la dirección
en la que decrece basta con que yo la multiplique por -1. 

## 1.11 Backpropagation

Backpropagation es un proceso es especialmente importante, ya que, a medida que se entrena una red neuronal, los nodos de 
las capas intermedias son capaces de organizarse por sí mismas. De esta manera, cada uno de estos nodos son capaces de 
aprender a reconocer distintas características de los datos de entrada.

Gracias al método de backpropagation las redes neuronales son capaces de identificar patrones de datos incompletos o 
arbitrarios y encontrar la solución más adecuada para el problema que se les haya planteado, ya que serán capaces de hallar 
un patrón similar a las características que hayan aprendido a reconocer durante su entrenamiento. Es decir, este algoritmo 
sirve para detectar errores en procesos que implican el uso de redes neuronales.

El entrenamiento de las redes neuronales es un proceso complejo que implica distintas etapas. El método de backpropagation 
es la cuarta etapa del proceso y, al mismo tiempo se compone de distintas fases:

- `Elección de entrada y salida:` Este es el primer paso en el funcionamiento del algoritmo y es el momento en el que se determina una entrada para todo el proceso de retropropagación, desde el punto de entrada hasta la salida deseada.
- `Configuración:` Una vez configurados los valores de entrada y de salida, el algoritmo procede a asignar una serie de valores secundarios que le permiten modificar parámetros dentro de cada capa y nodo que conforman la red neuronal.
- `Cálculo de error:` En este paso se determina el erro total a partir del análisis de los nodos y capas de red neuronal.
- `Minimización de errores:` Una vez detectados los errores, el algoritmo procede a minimizar su efecto en el conjunto de la red neuronal.
- `Actualización de parámetros:` Si la tasa de error es muy alta, el método de bakcpropagation ajusta y actualiza los parámetros para reducirla lo máximo posible.
- `Modelado para la predicción:` Tras la optimización de los errores, el método de cálculo de backpropagation evalúa las entradas de prueba adecuadas para garantizar que se obtienen el resultado deseado.

![back1.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fback1.png)


Nosotros hasta este momento entendemos que para llegar a computar la predicción las salidas de las capas anteriores en una red funcionan
como las entradas de la siguiente capa y así hasta llegar a la última capa que será la encargada de dar la predicción final.

![back2.gif](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fback2.gif)

Sin embargo, el error calculado por la `cost function` viene dado únicamente por la respuesta de la última capa, y una vez que se tiene este
error la última capa puede argumentar que ella NO tuvo la culpa sino que el error viene de la capa anterior, y así sucesivamente. 

Entonces haciendo uso de derivadas parciales es como podemos ir hacia atrás capa a capa distribuyendo los cambios necesarios para ir disminuyendo
el error, esto es en sí la forma más fácil de entender a `backpropagation`

Veamos un ejemplo de derivadas parciales utilizando `Computation Graph`

Dada la siguiente función:

`J(a, b, c) = 3(a +bc)`

Obtener las derivadas parciales `da, db, dc`. Primero debemos generar el grafo computacional de la función J. Para ello 
utilizaremos 2 variables auxiliares siendo `v = a +bc` y `u = bc` de tal manera que podemos rescribir la fórmula de las siguientes maneras:

`u = bc`

`v = a + u`

`J(a, b, c) = 3v = 3(a + u) = 3(a + bc)`

![back3.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fback3.png)

Dado que `J` es una función de 3 variables `(a, b, c)` sus derivadas parciales serían:

`dJ/da`

`dJ/db`

`dJ/dc`

Sin embargo, vemos como pudimos expresar nuestra función yendo de derecha a izquierda utilizando las variables auxiliares `u` y `v`

Para calcular las derivadas parciales tenemos que hacer backpropagation. Partamos del primer bloque de la derecha

`J = 3v`

Entonces:

`dJ/dv = 3`

Tenemos nuestra primera derivada que apunta al bloque de en medio al bloque `v`. Ahora obtengamos las derivadas parciales de la función `v`:

`v = a + u`

Entonces:

`dv/da = 1`

`dv/du = 1`

Ahora podemos observar que ya llegamos a la variable `a` que se puede expresar de la siguiente manera:

`dJ/da = dJ/dv * dv/da`

Pero nosotros YA calculamos `dJ/dv` y `dv/da` entonces ->

`dJ/dv * dv/da = (3)*(1) = 3`

Por lo tanto:

`dJ/da = 3`

Continuemos con el bloque de la función u:

`u = bc`

`du/db = c`

`du/dc = b`

Ahora ya podemos obtener las derivadas parciales de `c` y `b` con respecto a `J`:

`dJ/db = dJ/dv * dv/du * du/db = (3)(1)(c) = 3c`

`dJ/dc = dJ/dv * dv/du * du/dc = (3)(1)(b) = 3b`

Finalmente, entonces la respuesta a la pregunta sería:

`dJ/da = 1`

`dJ/db = 3c`

`dJ/dc = 3b`

## 1.12 Playground - Tensorflow

Tensorflow nos ofrece una herramienta sumamente útil que nos provee de un entorno gráfico para practicar nuestros conceptos 
de redes neuronales en problemas de `Classification` y `Regression`: [Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.87931&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


![hub1.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fhub1.png)

**Playground - Tensorflow**

Nos permite tener acceso de forma simple e inmediata a los siguientes parámetros configurables:

- Learning rate
- Activation Function
- Regularization
- Regularization Rate
- Problem Type

Hasta la izquierda podemos seleccionar la distribución de datos, así como la cantidad de datos utilizada para entrenar y evaluar al modelo.

Al centro vemos la arquitectura de la red, podemos configurar que entradas de datos queremos, la cantidad de capas ocultas y la cantidad
de neuronas para cada capa.

Hasta la derecha podemos ver el Test loss y un gráfico que representa la clasificación de la red.

## 1.13 Quiz: Fundamentos en la arquitectura de redes neuronales

![q1p1.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fq1p1.png)

![q1p2.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fq1p2.png)

![q1p3.png](1%20Fundamentos%20en%20la%20arquitectura%20de%20redes%20neuronales%2Fimgs%2Fq1p3.png)

# 2. Redes Neuronales con Python

## 2.1 Dimensiones, tensores y reshape

Python nos ofrece una amplia gama de tipos de datos y dimensiones con ayuda de numpy, entre ellos podemos observar:

- `Escalares`: son números aislados y únicos su dimensión es 0. Ejemplo: 5, 1, 3
- `Vectores`: en python son listas de números, su dimensión es 1. Ejemplo: [1, 2, 3]
- `Matrices`: en python por ejemplo son los dataframes de Pandas, su dimension es 2. Ejemplo [[1, 2, 3], [4, 5 ,6]]
- `Tensores`: Son conjuntos de datos de más de 2 dimensiones. 

![dim1.png](2%20Redes%20neuronales%20con%20Python%2Fimgs%2Fdim1.png)

> [código completo](2%20Redes%20neuronales%20con%20Python/dimensiones/dimensiones_tensores_reshape.py)

Veamos las dimensiones de los datos con python:

**1. Scalar:**
```python
x = np.array(42)
print("scalar:", x, "- shape:", x.shape, "dims:", x.ndim)
```
Respueta esperada:
```commandline
scalar: 42 - shape: () dims: 0
```
**2. Vector:**
```python
x = np.array([1, 2, 3, 4, 5])
print("vector:", x, "- shape:", x.shape, "dims:", x.ndim)
```
Respuesta esperada:
```commandline
vector: [1 2 3 4 5] - shape: (5,) dims: 1
```
**3. Matrix:**
```python
x = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])
print("matrix:", x, "- shape:", x.shape, "dims:", x.ndim)
```
Respuesta esperada:
```commandline
matrix: [[ 1  2  3  4  5]
         [ 6  7  8  9 10]] - shape: (2, 5) dims: 2
```
**4. Tensor:**
```python
x = np.array([[[1, 2, 3, 4, 5],
               [6, 7, 8, 9, 10]],
              [[1, 2, 3, 4, 5],
               [6, 7, 8, 9, 10]],
              [[1, 2, 3, 4, 5],
               [6, 7, 8, 9, 10]]
              ])
print("Tensor:", x, "- shape:", x.shape, "dims:", x.ndim)
```
Respuesta esperada:
```commandline
Tensor: [[[ 1  2  3  4  5]
          [ 6  7  8  9 10]]

         [[ 1  2  3  4  5]
          [ 6  7  8  9 10]]

         [[ 1  2  3  4  5]
          [ 6  7  8  9 10]]] - shape: (3, 2, 5) dims: 3
```
**5. Función Reshape:**

```python
x = np.array([[0, 1],
              [2, 3],
              [4, 5],
              [6, 7]])

print(x, x.shape)
x = x.reshape(8, 1)
print()
print(x, x.shape)
x = x.reshape(2, 4)
print()
print(x, x.shape)
print()
x = np.array([[0, 1],
              [2, 3],
              [4, 5],
              [6, 7]])
xt = x.T
print("original:")
print(x)
print("transpose:")
print(xt)
```
Respuesta esperada:
```commandline
[[0 1]
 [2 3]
 [4 5]
 [6 7]] (4, 2)

[[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]] (8, 1)

[[0 1 2 3]
 [4 5 6 7]] (2, 4)

original:
[[0 1]
 [2 3]
 [4 5]
 [6 7]]
transpose:
[[0 2 4 6]
 [1 3 5 7]]

Process finished with exit code 0

```

## 2.2 Creando nuestra red neuronal usando numpy y matemáticas

En estas próximas clases vamos a poner en práctica todos los conocimientos adquiridos hasta este punto del curso, implementando nuestra
propia red neuronal desde 0 sin utilizar ningún framework, solamente numpy. Temas a ver:

1. Creación de un dataset Artificial
2. Definimos nuestras funciones de activación
3. Función de perdida
4. Función inicializadora de pesos
5. Forward propagation
6. Backpropagation
7. Gradient descent
8. Train Model Function
9. Definir arquitectura de la red
10. Entrenamos el modelo
11. Probando el modelo sobre datos nuevos

> ## Nota:
> El código completo de estas clases lo puedes encontrar [aquí](2%20Redes%20neuronales%20con%20Python%2Fprimer%20red%2Fred_neuronal.py)

### 1. Creación de un dataset Artificial

Para este primer paso vamos a crear un dataset artificial de 2 dimensiones, esto nos permitirá gráficarlo y ver fácilmente
la distribución de clases del mismo.

Nuestro dataset será creado utilizando la función `make_gaussian_quantiles` de [sklear.datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html)

Las características de nuestro dataset artificial serán los siguientes:

- `Cantidad de muestras:` n_samples = 1000
- `Cantidad de características de entrada:` n_features = 2
- `Cantidad de clases a predecir:` n_classes = 2


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

N = 1000
gq = make_gaussian_quantiles(mean=None, cov=0.1, n_samples=N, n_features=2, n_classes=2, shuffle=True,
                             random_state=21)

X, Y = gq
# Esto es necesario para hacer el plot más cómodo
Y = Y[:, np.newaxis]
# X son las entradas de mi red, tienen 2 dimensiones, Y son las predicciones que corresponde a 2 clases.
print(X.shape, Y.shape)
# Muestro un scatter plot de la distribución de mis datos
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=40)
plt.title("Problema de clasificación")
plt.savefig("imgs/clasificacion.png")
plt.close()
```
Respuesta esperada:
```commandline
(1000, 2) (1000, 1)
```

![clasificacion.png](2%20Redes%20neuronales%20con%20Python%2Fprimer%20red%2Fimgs%2Fclasificacion.png)

Podemos observar que la salida:
```commandline
(1000, 2) (1000, 1)
```
Tenemos `(1000)` muestras de puntos coordinados `(x,y)` por eso nuestras variables de entrada: `X.shape = (1000, 2)`, mientras
que tenemos un vector de labels `Y` que solo tienen dos clases `(0, 1)` por eso nuestra variable de salida: `Y.shape = (1000, 1)`

### 2. Definimos nuestras funciones de activación

Puedes repasar este tema en [Funciones de activación](#18-funciones-de-activación)

```python
def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
    else:
        return 1 / (1 + np.exp(-x))


def relu(x, derivate=False):
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(0, x)
```

Utilizaremos la función `relu` como función de activación para las capas ocultas y finalmente
`sigmoid` como la última función de activación para la capa de clasificación.

### 3. Función de perdida

Puedes repasar este tema en [Función de pérdida](#19-función-de-pérdida--loss-function-)

Usaremos mean squared error como función de perdida para esta red neuronal:

```python
def mse(y, y_hat, derivate=False):
    if derivate:
        return 2*(y_hat - y)
    else:
        return np.mean((y_hat - y) ** 2)
```
### 4. Función inicializadora de pesos

Cómo hemos visto en anteriores clases, cada capa de una red neuronal está definida por una serie de pesos `W` y de sesgos `b`.
Cuando creamos una red neuronal debemos empezar definiendo estos valores con un valor por defecto aleatorio. Eventualmente
los procesos de optimización iran mejorando estos pesos y sesgos aleatorios.

> ## Nota:
> TODO el código de aquí en adelante está optimizado para trabajar con n cantidad de layers.
> De forma automática los pasos de forward y backward propagation se adaptan a la cantidad n de layers.

```python
def initialize_parameters_deep(layer_dims: list) -> dict:
    """
    Genera un diccionario de pesos y sesgos de una red neuronal de acuerdo a su arquitectura de capas
    :param layer_dims: lista que representa la cantidad de neuronas presente en cada capa de la red
    :return: dict: parameters. 
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(0, L - 1):
        parameters[f'W{l + 1}'] = (np.random.rand(layer_dims[l],
                                                  layer_dims[l + 1]) * 2) - 1  # Multiplicar por 2 y restar
        # 1 es una forma de normalizar los datos para que vayan de -1 a 1, de esta forma encajan mejor con la
        # distribución de datos de entrada de nuestro problema, pero tampoco es indispensable
        parameters[f'b{l + 1}'] = (np.random.rand(1, layer_dims[l + 1]) * 2) - 1
        print(f"Inicializando PESO W{l + 1} con dimensiones:", parameters[f'W{l + 1}'].sahpe)
        print(f"Inicializando BIAS b{l + 1} con dimensiones:", parameters[f'b{l + 1}'].sahpe)

    return parameters

```
## 2.3 Entrenamiento forward de la red neuronal

### 5. Forward Propagation

Vamos primeramente a definir en términos de variables de programación algunos de los conceptos que hemos estado manejando hasta el momento.

- `X` son las variables de entrada del modelo
- `Y` son las etiquetas reales de las clases a predecir
- `Wi` serán los pesos de la i-th capa.
- `bi` serán los sesgos de la i-th capa.
- `Zi` será la parte lineal del proceso de la red neuronal `np.dot(X, W) + b` de la i-th capa.
- `Ai` será la función de activación aplicada a `Z` de la i-th capa.
- `y_hat` es la predicción final de la red neuronal, correspondiente a `A` de la última capa.
- `d{variable}i` representa la derivada de cierta variable por ejemplo `dW3`corresponde a la derivada de los pesos de la capa 3.

Programamos un paso de nuestra función de forward propagation:

```python
def linear_forward(A, W, b):
    Z = np.dot(A, W) + b
    return Z


def linear_activation_forward(A_prev, W, b, activation_function):
    Z = linear_forward(A_prev, W, b)
    A = activation_function(Z)
    return A


def forward_step(A0, params, activations_functions, n_layers):
    L = n_layers
    params["A0"] = A0
    for i in range(1, L + 1):
        params[f"A{i}"] = linear_activation_forward(params[f"A{i - 1}"], params[f"W{i}"], params[f"b{i}"],
                                                    activations_functions[i])
    y_hat = params[f"A{L}"]
    return y_hat
```

> ### Nota:
> Por nomenclatura, la primera capa de una red corresponde con la entrada de las varibles a clasificar, a su vez podemos llamar a estos datos como:
> `A0` en realidad, la función `linear_forward` necesita los pesos y bias de la capa actual y la respuesta de la función de activación de la capa anterior.
> Sin embargo, en la capa oculta 1, la respuesta A anterior corresponde con la entrada de datos. Solamente es en este caso.
> 

## 2.4 Aplicando backpropagation y descenso del gradiente

### 6. Backpropagation

En el proceso de `backpropagation` el primer paso es obtener el error entre el valor real `Y` y el valor predicho por la red
`y_hat`. Una vez que se calculan las derivadas de `Z`y `W` de la última capa, entonces podemos ir para atrás calculando
las otras `dZ` y `dW`para las capas anteriores.

```python
def backpropagation(Y, y_hat, params, activations_functions, error_function, n_layers):
    L = n_layers
    params[f'dZ{L}'] = error_function(Y, y_hat, True) * activations_functions[L](params[f'A{L}'], True)
    params[f'dW{L}'] = np.dot(params[f'A{L - 1}'].T, params[f'dZ{L}'])

    for l in reversed(range(2, L + 1)):
        params[f'dZ{l - 1}'] = np.matmul(params[f'dZ{l}'], params[f'W{l}'].T) * activations_functions[l - 1](
            params[f'A{l - 1}'], True)

    for l in reversed(range(1, L)):
        params[f'dW{l}'] = np.matmul(params[f'A{l - 1}'].T, params[f'dZ{l}'])

    return params
```

### 7. Gradient descent

El último paso es ya con las derivadas calculadas podemos actualizar los pesos `Wi` y los bias `bi` de cada capa utilizando
las derivadas calculadas en el punto anterior y un learning rate `lr`.

```python
def gradient_descent(params, lr, n_layers):
    L = n_layers

    for l in reversed(range(1, L + 1)):
        params[f'W{l}'] = params[f'W{l}'] - params[f'dW{l}'] * lr
        params[f'b{l}'] = params[f'b{l}'] - (np.mean(params[f'dZ{l}'], axis=0, keepdims=True)) * lr

    return params
```

## 2.5 Entrenamiento y análisis de resultados de tu red neuronal

### 8. Train model function

Ahora con todas las funciones que hemos definido anteriormente podemos crear una nueva función que sirva como orquestador de funciones
y permita poner de forma secuencial todos los pasos de entrenamiento de la red para una cantidad de iteraciones `epochs` definida:

```python
def train_model(X, Y, layer_dims, params, activations_functions, error_function, lr, epochs):
    errors = []
    n_layers = len(layer_dims) - 1
    j = 1
    for _ in range(epochs):
        y_hat = forward_step(X, params, activations_functions, n_layers)
        params = backpropagation(Y, y_hat, params, activations_functions, error_function, n_layers)
        params = gradient_descent(params, lr, n_layers)

        if _ % 100 == 0:
            e = error_function(Y, y_hat)
            if _ % 1000 == 0:
                print(j, "error:", e)
                j += 1
            errors.append(e)
            
    return errors, params
```

### 9. Definir arquitectura de la red

En este punto podemos definir todas las variables que orquestan a la arquitectura de la red como: el número de capas y número
de neuronas para capa. El learning rate. Las funciones de activaciones para cada capa oculta. La cantidad de epochs.

```python
layer_dims = [2, 4, 8, 1]
lr = 0.002
activations_functions = [0, relu, relu, sigmoid]
params = initialize_parameters_deep(layer_dims)
epochs = 10000
```
Repuesta esperada:
```commandline
Inicializando PESO W1 con dimensiones: (2, 4)
Inicializando BIAS b1 con dimensiones: (1, 4)
Inicializando PESO W2 con dimensiones: (4, 8)
Inicializando BIAS b2 con dimensiones: (1, 8)
Inicializando PESO W3 con dimensiones: (8, 1)
Inicializando BIAS b3 con dimensiones: (1, 1)
```
### 10. Entrenamos el modelo

Ya tenemos todo listo para ejecutar nuestra función `train_model` durante n `epochs` utilizando `mse` como nuestra 
`función de perdida`.

```python
errors, params = train_model(X, Y, layer_dims, params, activations_functions, mse, lr, epochs)
plt.plot(errors)
plt.title("MSE over epochs")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.savefig("imgs/model.png")
plt.close()
```
Respuesta esperada:
```commandline
1 error: 0.24757298982437223
2 error: 0.05393229428625825
3 error: 0.059482476810252996
4 error: 0.05376995780762481
5 error: 0.045345930433615116
6 error: 0.03746680657241106
7 error: 0.03602928839608933
8 error: 0.03626453751967989
9 error: 0.031568468850135964
10 error: 0.022516339724916422
```
![model.png](2%20Redes%20neuronales%20con%20Python%2Fprimer%20red%2Fimgs%2Fmodel.png)

### 11. Probando el modelo sobre datos nuevos
Finalmente, podemos crear un nuevo dataset de prueba aleatorio que tenga una distribución similar al dataset de entrenamiento y observar
como clasifico nuestro modelo al dataset de prueba.
```python
data_test = (np.random.rand(1000, 2) * 2) - 1
prediction = forward_step(data_test, params, activations_functions, 3)
y = np.where(prediction >= 0.5, 1, 0)
plt.scatter(data_test[:, 0], data_test[:, 1], c=y[:, 0], s=40)
plt.title("NN prediction")
plt.savefig("imgs/prediction.png")
plt.close()
```
Respuesta esperada:
![prediction.png](2%20Redes%20neuronales%20con%20Python%2Fprimer%20red%2Fimgs%2Fprediction.png)

## 2.6 Quiz: redes neuronales con python

![q2p1.png](2%20Redes%20neuronales%20con%20Python%2Fimgs%2Fq2p1.png)

![q2p2.png](2%20Redes%20neuronales%20con%20Python%2Fimgs%2Fq2p2.png)

![q2p3.png](2%20Redes%20neuronales%20con%20Python%2Fimgs%2Fq2p3.png)

![q2p4.png](2%20Redes%20neuronales%20con%20Python%2Fimgs%2Fq2p4.png)

![q2p5.png](2%20Redes%20neuronales%20con%20Python%2Fimgs%2Fq2p5.png)

# 3. Manejo de Redes Neuronales con Keras

## 3.1 Data: Train, Validation, Test

**Training Dataset**

> **Training Dataset:** Es la muestra de datos utilizados para entrenar al modelo.

El dataset actual que utilizados para entrenar el modelo (pesos y sesgos) en el caso de una red neuronal. El modelo observa
y aprende de este conjunto de datos.

**Validation Dataset**

> **Validation Dataset:** Es la muestra de datos utilizada para proveer una evaluación no sesgada de un modelo entrenado en el 
> dataset de entrenamiento mientras ajustados los hiperparámetros. La evaluación se vuelve más sesgada a medida que la habilidad 
> en el conjunto de datos de validación se incorpora a la configuración del modelo.
> 

El conjunto de validación se utiliza para evaluar un modelo dado, pero esto es para una evaluación frecuente. Nosotros, 
como ingenieros de aprendizaje automático, usamos estos datos para ajustar los hiperparámetros del modelo. Por lo tanto, 
el modelo ocasionalmente ve estos datos, pero nunca lo "aprende" de esto. Utilizamos los resultados del conjunto de validación 
y actualizamos hiperparámetros de nivel superior. Entonces, el conjunto de validación afecta a un modelo, pero solo 
indirectamente. El conjunto de validación también se conoce como conjunto de desarrollo o el conjunto de desarrollo. Esto
tiene sentido, ya que este conjunto de datos ayuda durante la etapa de "desarrollo" del modelo.
 
**Test Dataset**

>**Test Dataset**: la muestra de datos utilizados para proporcionar una evaluación imparcial de un modelo de modelo final 
> en el conjunto de datos de capacitación.

El conjunto de datos de prueba proporciona el estándar de oro utilizado para evaluar el modelo. Solo se usa una vez que 
un modelo está completamente entrenado (usando el tren y los conjuntos de validación). El conjunto de pruebas es generalmente 
lo que se utiliza para evaluar modelos competidores (por ejemplo, en muchas competiciones de Kaggle, el conjunto de validación 
se lanza inicialmente junto con el conjunto de capacitación y el conjunto de pruebas real únicamente se lanza cuando la competencia 
está a punto de cerrar, y es El resultado del modelo en el conjunto de pruebas que decide el ganador). 
Muchas veces el conjunto de validación se usa como conjunto de pruebas, pero no es una buena práctica. El conjunto de pruebas 
generalmente está bien curado. Contiene datos cuidadosamente muestreados que abarcaron las diversas clases que enfrentaría el modelo, 
cuando se utiliza en el mundo real.

![datatest.gif](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fdatatest.gif)


## 3.2 Resolviendo un problema de clasificación binaria

En las siguientes clases estaremos resolviendo un problema de clasificación binaria. Para ello vamos a utilizar el dataset
[IMDB movie review sentiment classification](https://keras.io/api/datasets/imdb/). El cual es un dataset que contiene una muestra de 
25,000 reseñas de películas de IMDB etiquetadas por sentimiento (positivo/negativo). Adicionalmente, este dataset contiene 
10,000 de las palabras más usadas en cada una de las reseñas, estas palabras están guardadas en un catálogo que asigna 
un índice a cada una de las palabras más empleadas.

El objetivo de esta práctica es utilizar nuestros conocimientos de normalizado de datos para preparar a este dataset para ser 
clasificado por una red neuronal que pueda predecir el sentimiento de una nueva reseña. Los puntos a seguir serán los siguientes:

1. Obtención de datos imdb - Keras
2. Normalizando datos
3. Arquitectura del Modelo
4. Entrenando el modelo
5. Analizando resultados
6. Arquitectura del modelo normalizado

> ## Nota:
> El código completo de este proyecto lo puedes encontrar [aquí](3%20Manejo%20de%20redes%20neuronales%20con%20Keras/clasificacion%20binaria/main.py)

### 1. Obtención de datos IMDB - Keras

Utilizando `keras.datasets` vamos a importar `imdb` y a cargar el dataset con su método `load_data`
```python
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# ¿Cómo luce uno de los datos de entrenamiento?
print("Train data example")
print(train_data[0])
print("Train label example")
print(train_labels[0])
```
Respuesta esperada:
```commandline
Train data example
[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
Train label example
1
```
Sin embargo, nos gustaría ver, por fines didácticos, cuál es el verdadero texto que tiene una muestra del dataset de entrenamiento.

Para ello nos apoyaremos de la función `get_word_index()` el cual nos regresara el catalogo de las palabras utilizadas con su 
respectivo índice, después convertiremos este `word_index` en un diccionario, para de esta forma acceder de forma sencilla
a las palabras del `word_index` de acuerdo al índice de la palabra.
```python
def convert_number_to_word(example):
    word_index = imdb.get_word_index()
    word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(" ".join([str(word_index.get(_ - 3)) for _ in example]))

# Observemos entonces cómo luce realmente un texto de entrada
convert_number_to_word(train_data[0])
```
Respuesta esperada:
```commandline
None this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert None is an amazing actor and now the same being director None father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for None and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also None to the two little boy's that played the None of norman and paul they were just brilliant children are often left out of the None list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all
```

### 2. Normalizado de datos

Antes de continuar con este punto, veamos un poco de contexto del porqué las palabras del dataset han sido guardadas como 
índices en lugar de utilizar las palabras directamente.

En general en cualquier problema de machine learning/deep learning, nos topamos con que los datos que vamos a utilizar para
entrenar a nuestro modelo de inteligencia artificial pueden ser de dos clases:

- `Estructurados:` Datos que tengan forma de fila - columna (similar a una tabla de excel)
- `No estructurados:` Datos que explícitamente NO tengan una forma de fila - columna, cómo lo son: El audio, las imágenes o el texto.

A su vez, los datasets `estructurados` cuentan con dos tipos principales de variables: **Cualitativas y Cuantitativas**.

![tipos_de_variable.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Ftipos_de_variable.png)

- **Variables cualitativas**
  - Explicación: Este tipo de variables representan características o cualidades de un objeto y no pueden ser medidas de forma numerica.
  - Tipos de variables cualitativas:
    - Ordinal: A pesar de NO tener un valor asignado, sí posee un orden conocido. Por ejemplo: Muy frio, frio, templado, caliente, muy caliente.
    - Nominal: No poseen un valor asignado y tampoco cuentan con un orden conocido. Por ejemplo tipos de colores: Rojo, Azul, Verde.
- **Variables cuantitativas:**
  - Explicación: Este tipo de variables representan cantidades numéricas, lo cual les permite utilizar operaciones aritméticas.
  - Tipos de variables cuantitativas:
    - Discreta: Se puede contar con números enteros. Por ejemplo: Años de vida, número de hijos en una familia etc.
    - Continúa: Se debe expresar con números decimales. Por ejemplo: Peso de una persona puede ser 64.512 Kg o 72.018 Kg etc.

Tomando en cuenta lo dicho anteriormente, podemos observar que la propia naturaleza de las redes neuronales, incentiva el uso de
variables cuantitativas continuas como tipo de variable predilecta para ser utilizada por el modelo. Sin embargo, ya sabemos
que el texto es un tipo de dato `No estructurado` y esto NO le gusta a los modelos de deep learning, tendríamos entonces que
primero usar algún mecanismo que nos permita estructurar a este tipo de dato y de forma preferible llevarlo a una variable
cuantitativa continua. 

El texto puede ser pensado de cierta manera como una variable `cualitativa nominal` en donde cada una de las palabras que existen 
puede ser en sí misma una variable nominal. Tomando este primer pensamiento como base, entonces es fácil de observar que una
forma de convertir variables `nominales` a `discretas` es creando un catálogo de índices, que logre mapear cada valor nominal
a un número discreto. `[gato, azul, coche]` Se puede convertir en `[1, 2, 3]` dado un diccionario de índices - palabras como el siguiente:

`{1: "gato", 2: "azul", 3: "coche"}`

A este tipo de conversión entre variables nominales y discretas se le conoce como: `label encoding` puedes leer más de esta
técnica en -> [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

Sin embargo, pese a que esta técnica es sumamente fácil de implementar y es bastante eficiente en terminos de tiempo y memoria, 
tiene una gran desventaja muy clara. Este tipo de transformaciones podría hacernos pensar que un `coche` es equivalente a tres `gatos`
numéricamente es cierto, pero sabemos que esa analogía no tiene ningún sentido. 

Una alternativa al `Label Encoding` es el método llamado `One Hot Encoding`. La estrategia que implementa es crear una 
columna para cada valor distinto que exista en la característica que estamos codificando y, para cada registro, marcar 
con un 1 la columna a la que pertenezca dicho registro y dejar las demás con 0. En el ejemplo visto en la sección anterior 
en el que codificábamos el sexo de los pasajeros, `One Hot Encoding generaría dos columnas (una para el valor "male" y otra 
para el valor "female") y, para cada pasajero, asignaría un 1 a la columna de "male" y un 0 a la de "female" en el caso 
de ser un hombre, y viceversa en el caso de ser mujer (un 0 en la columna de "male" y un 1 en la de "female").

En nuestro ejemplo anterior entonces las variables nominales de: `[gato, azul, coche]` en lugar de ser representadas
utilizando `Label enconding` como `{1: "gato", 2: "azul", 3: "coche"}` podrían tener la siguiente reprsentación de `One Hot
Encoding`

```
gato =  [1, 0, 0]
azul =  [0, 1, 0]
coche = [0, 0, 1]
```

Esta es una mejor aproximación, puesto que entonces un coche tiene el mismo peso que un gato. Y todas las palabras tienen la misma
distancia entre ellas. Sin embargo, esta NO es la representación definitiva para la conversión de Texto a un tipo de dato
estructurado. Un ejemplo sencillo es que en el lenguaje existe un concepto llamado semántica que nos permite darnos cuenta que:

Entre las palabras `cuchillo` y `tenedor` hay menos distancia que entre `cuchillo` y `ornitorrico`. Esta explicación será utilizada
en un próximo curso de deep learning, pero es la introducción al tema de word embeddings. Pero por ahora con el conocimiento que he explicado
podemos continuar con la explicación de este tema.

Ahora que ya tenemos un conocimiento sobre qué es `Label Encoding` y `One Hot Encoding` podemos proceder a transformar la representación
de `Label Encoding` que utilizo el dataset de IMDB por una versión en `One Hot Encoding` la cuál es una mejor representación puesto que 
tiene la forma de un `tensor` que es ideal para deep learning. Revisar apuntes de tipos de datos de deep learning [Dimensiones y tensores](#21-dimensiones-tensores-y-reshape).

```python

def one_hot_encoding(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results

# Covertirmos la representacion de label encoding por one hot encoding para nuestros parámetros de entrada
x_train = one_hot_encoding(train_data)
x_test = one_hot_encoding(test_data)

# Siempre que podamos transformemos los datos enteros en float32 puesto que tensorflow esta optimizado para trabajar con este tipo de datos
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print(train_data.shape)
print(x_train.shape)
print(x_train[0], x_train[0].shape)
```
Valor esperado:
```commandline
(25000,)
(25000, 10000)
[0. 1. 1. ... 0. 0. 0.] (10000,)
```

### 3. Arquitectura del Modelo.

Para esta arquitectura veremos un modelo de 2 capa ocultas de 16 neuronas con función de activación `relu`. Nuestra capa de clasificación
solo tendrá 1 neurona, puesto que es suficiente para un problema de clasificación binaria y tendrá una función de activación `sigmoid`

Finalmente, el modelo será compilado con el optimizador `rmsprop` y como función de perdida usaremos `binary_crossentropy` la cual funciona
muy bien para problemas de clasificación binaria, y como medida de desempeño usaremos `accuracy`

```python
def architecture(model: models.Sequential, input_shape) -> models.Sequential:
    model.add(layers.Dense(16, activation="relu", input_shape=input_shape))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


model = models.Sequential()
model = architecture(model, (10000, ))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
```

## 3.3 Entrenamiento del modelo de clasificación binaria

### 4. Entrenando el Modelo

Entrenaremos el modelo por 50 épocas, con un batch size de 512 y finalmente, usaremos un 30% de los datos de entrenamiento
como datos de validación. Esto nos permitirá observar si el modelo tiene o no overfitting.

```python
history = model.fit(x_train, y_train, epochs=50, batch_size=512, validation_split=0.3)
```

### 5. Analizando resultados

Veamos cuales son los resultados de nuestro modelo base:

```python
results = model.evaluate(x_test, y_test)
print(results)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
fig = plt.figure(figsize=(10, 10))
epoch = range(1, len(loss_values) + 1)
plt.plot(epoch, loss_values, 'o-r', label='training')
plt.plot(epoch, val_loss_values, '--', label='validation')
plt.title("Error in training and validation datasets")
plt.xlabel("epochs")
plt.ylabel("Binary Cross Entropy")
plt.legend()
plt.savefig("imgs/errores.png")
plt.close()
```
Respuesta esperada:
```commandline
782/782 [==============================] - 1s 1ms/step - loss: 1.2614 - acc: 0.8542
```

![errores.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fclasificacion%20binaria%2Fimgs%2Ferrores.png)


## 3.4 Regularización - Dropout

Antes de continuar con la parte final de este mini proyecto de ejemplo, es necesario explicar el concepto de `overfitting` y con ello
conocer diferentes técnicas de `regularización`:

El `overfitting` o `sobre ajuste` es cuando nuestro modelo no es capaz de generalizar el conocimiento adquirido en la etapa
de `trainig`. Es fácil de observar este problema, cuando comparamos las funciones de coste a lo largo de las épocas entre 
el dataset de `trainig` y `validation`. En nuestro problema de clasificación binaria de reseñas de `IMDB`. La curva roja representa
el error en el dataset de entrenamiento, pero a pesar de que a lo largo de 10 épocas podemos observar como de poco en poco este
error fue disminuyendo, esto no fue el mismo caso para el error en el conjunto de `validation` aproximadamente, después de
la época 4 el error en el dataset de `validation` empezó a subir, marcando una pauta para detectar el `overfitting`.

Ahora listemos algunas de las principales técnicas con las que contamos en deep learning para reducir el problema del overfitting.

### Regularización L2 (Ridge)

La idea detrás de este tipo de regularización es reducir el valor de los parámetros para que sean pequeños.

Esta técnica introduce un término adicional de penalización en la función de coste original (L), añadiendo a su valor la suma de los cuadrados de los parámetros (ω).

La mala noticia es que este nuevo término puede ser alto; tanto que la red minimizaría la función de coste haciendo los parámetros muy cercanos a 0, lo que no sería nada conveniente. Es por ello que multiplicaremos ese sumando por una constante (λ) pequeña, cuyo valor escogeremos de forma arbitraria (0.1, 0.01, …).

La función de coste queda, por tanto, así:

![L2](https://latex2png.com/pngs/3457ebd84be9eb2d9e89ed7df3349a77.png)

**¿Cuándo es efectiva Ridge (L2)?**

Ridge nos va a servir de ayuda cuando sospechemos que varios de los atributos de entrada (features) estén correlacionados 
entre ellos. Ridge hace que los coeficientes acaben siendo más pequeños. Esta disminución de los coeficientes minimiza 
el efecto de la correlación entre los atributos de entrada y hace que el modelo generalice mejor. Ridge funciona mejor cuando la mayoría de los atributos son relevantes.

### Regularización L1 (Lasso)

Existe otra técnica muy parecida a la anterior denominada regularización L1 donde los parámetros en el sumatorio del término de penalización no se elevan al cuadrado, sino que se usa su valor absoluto.

![L1](https://latex2png.com/pngs/fb0f2fb18c9fc8d7db7288c0fb99fad3.png)

Esta variante empuja el valor de los parámetros hacia valores más pequeños, haciendo incluso que la influencia de algunas
variables de entrada sea nula en la salida de la red, lo que supone una selección de variables automática. El resultado es 
una mejor generalización, pero solo hasta cierto punto (la elección del valor de λ cobra más importancia en este caso).

**¿Cuándo es efectiva Lasso (L1)?**

Lasso nos va a servir de ayuda cuando sospechemos que varios de los atributos de entrada (features) sean irrelevantes. 
Al usar Lasso, estamos fomentando que la solución sea poco densa. Es decir, favorecemos que algunos de los coeficientes 
acaben valiendo 0. Esto puede ser útil para descubrir cuáles de los atributos de entrada son relevantes y, en general, 
para obtener un modelo que generalice mejor. Lasso nos puede ayudar, en este sentido, a hacer la selección de atributos 
de entrada. Lasso funciona mejor cuando los atributos no están muy correlacionados entre ellos.

### Regularización ElasticNet (L1 y L2)

ElasticNet combina las regularizaciones L1 y L2. Con el parámetro r podemos indicar que importancia relativa tienen 
Lasso y Ridge respectivamente. Matemáticamente:

![elastic](https://latex2png.com/pngs/ec1b4f7f8e2e71c50ae9957e4c41afe4.png)

**¿Cuándo es efectiva ElasticNet?**

Usaremos ElasticNet cuando tengamos un gran número de atributos. Algunos de ellos serán irrelevantes y otros estarán 
correlacionados entre ellos.

### Dropout

Esta técnica difiere de las vistas hasta el momento. El procedimiento es sencillo: por cada nueva entrada a la red en fase 
de entrenamiento, se desactivará aleatoriamente un porcentaje de las neuronas en cada capa oculta, acorde a una probabilidad 
de descarte previamente definida. Dicha probabilidad puede ser igual para toda la red, o distinta en cada capa.

![dropout.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fdropout.png)

Lo que se consigue con esto es que ninguna neurona memorice parte de la entrada; que es precisamente lo que sucede cuando 
tenemos sobreajuste.

Una vez tengamos el modelo listo para realizar predicciones sobre muestras nuevas, debemos compensar de alguna manera el 
hecho de que no todas las neuronas permanecieran activas en entrenamiento, ya que en inferencia sí que estarán todas 
funcionando y, por tanto, habrá más activaciones contribuyendo a la salida de la red. Un ejemplo de dicha compensación 
podría ser multiplicar todos los parámetros por la probabilidad de no descarte.

### Normalización por lotes (Batch normalization)
La historia de esta técnica es curiosa. Se presentó como una solución para reducir algo llamado Internal Covariate Shift, 
pero parece que no es eso lo que hace. Aún así es una técnica esencial para redes neuronales por todo lo que aporta, como 
explicamos a continuación.

La normalización en lotes consiste básicamente en añadir un paso extra, habitualmente entre las neuronas y la función de 
activación, con la idea de normalizar las activaciones de salida. Lo ideal es que la normalización se hiciera usando la 
media y la varianza de todo el conjunto de entrenamiento, pero si estamos aplicando el descenso del gradiente estocástico 
para entrenar la red, se usará la media y la varianza de cada mini-lote de entrada.

Nota: cada salida de cada neurona se normalizará de forma independiente, lo que quiere decir que en cada iteración se 
calculará la media y la varianza de cada salida para el mini-lote en curso.

A continuación de la normalización se añaden 2 parámetros: un bias como sumando, y otra constante similar a un bias, pero 
que aparece multiplicando cada activación. Esto se hace para que el rango de la entrada escale fácilmente hasta el rango 
de salida, lo que ayudará mucho a nuestra red a la hora de ajustar a los datos de entrada, y reducirá las oscilaciones 
de la función de coste. Como consecuencia de esto podremos aumentar la tasa de aprendizaje (no hay tanto riesgo de acabar 
en un mínimo local) y la convergencia hacia el mínimo global se producirá más rápidamente.

![batch_norm.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fbatch_norm.png)

La normalización por lotes es más una técnica de ayuda al entrenamiento que una estrategia de regularización en sí misma. 
Esto último se logra realmente aplicando algo adicional conocido como momentum. La idea de este momentum es que cuando 
introduzcamos un nuevo mini-batch de entrada (N muestras procesadas en paralelo) no se usen una media y una desviación 
muy distintas a las de la iteración anterior, para lo que se tendrá en cuenta el histórico, y se elegirá una constante 
que pondere la importancia de los valores del mini-batch actual frente a los valores del anterior. Gracias a todo esto 
se conseguirá reducir el sobreajuste.

### Data augmentation

La idea es aplicar diversas transformaciones sobre las entradas originales, obteniendo muestras ligeramente diferentes, 
pero iguales en esencia, lo que permite a la red desenvolverse mejor en la fase de inferencia.

Esta técnica se utiliza mucho en el campo de la visión artificial porque funciona de maravilla (en otros campos está por 
explorar). Dentro de dicho contexto, una misma imagen de entrada será procesada por la red neuronal tantas veces como epochs 
ejecutemos en entrenamiento; provocando que la red acabe memorizando la imagen si estamos entrenamos demasiado. Lo que 
haremos es aplicar transformaciones de forma aleatoria cada vez que volvamos a introducir la imagen a la red.

Ejemplos de transformaciones son:

- Voltear la imagen en horizontal / vertical.
- Rotar la imagen X grados.
- Recortar, añadir relleno, redimensionar.
- Aplicar deformaciones de perspectiva.
- Ajustar brillo, contraste, saturación.
- Introducir ruido, defectos.
- Combinaciones de las anteriores.

![data_augmentation.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fdata_augmentation.png)

De esta forma contaremos con más información para entrenamiento sin necesidad de obtener muestras adicionales, y también 
sin alargar los tiempos. Lo mejor es que si por ejemplo nuestra red se dedica a clasificar imágenes o detectar objetos, 
esta técnica conseguirá que el modelo sea capaz de obtener buenos resultados para imágenes tomadas desde distintos ángulos 
o bajo distintas condiciones de luz. Por tanto, conseguiremos que la red no sobreajuste y que generalice mejor.


### Early Stopping 

Es una técnica que trata de aplicar ciertas reglas para saber cuándo es momento de parar el entrenamiento, de forma que 
no se produzca sobreajuste a los datos de entrada, ni tampoco subajuste.

La regla más extendida sería la de entrenar el modelo monitorizando su rendimiento y guardando sus parámetros al finalizar 
cada epoch, hasta que apreciemos que el error de validación aumenta de forma sostenida (hay empeoramientos que son debidos 
a la componente estocástica del algoritmo). Nos quedaremos con el modelo que teníamos justo en el momento anterior.

## 3.5 Reduciendo el overfitting

De forma completamente didactica y para poder observar como se utilizan algunas de las técnicas de normalización anteriormente
mencionadas, vamos a crear un nuevo modelo con una nueva arquitectura que tenga las siguientes características:

- Un modelo más simple: Menos cantidad de neuronas por capa.
- Uso de regularizadores l1 y l2
- Uso de BatchNormalization
- Uso de dropout
- Early Stopping - Entrenar con menos épocas. 

Primero definimos la nueva arquitectura del modelo:
```python
def architecture(model: models.Sequential) -> models.Sequential:
    model.add(layers.Dense(8, activation="relu", input_shape=(10000,), kernel_regularizer=regularizers.l1_l2()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l1_l2()))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
```
Y ahora procedemos a compilar
```python
model_norm = models.Sequential()
model_norm = architecture(model_norm)
model_norm.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history_norm = model_norm.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.3)
```
Finalmente evaluamos los resultados del nuevo modelo:
```python
results = model_norm.evaluate(x_test, y_test)
print(results)
history_dict = history_norm.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

fig2 = plt.figure(figsize=(10, 10))
epoch = range(1, len(loss_values) + 1)
plt.plot(epoch, loss_values, 'o-r', label='training')
plt.plot(epoch, val_loss_values, '--', label='validation')
plt.title("Error in training and validation datasets using normalization")
plt.xlabel("epochs")
plt.ylabel("Binary Cross Entropy")
plt.legend()
plt.savefig("imgs/errores_norm.png")
plt.close()
```
Respuseta esperada:
```commandline
782/782 [==============================] - 1s 1ms/step - loss: 0.4057 - acc: 0.8629
```
![errores_norm.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fclasificacion%20binaria%2Fimgs%2Ferrores_norm.png)

El modelo ahora ha obtenido una performance ligeramente superior en el dataset de testing pasando de 0.85 a 0.86 con un modelo más
simple, y en la gráfica de comparación del error de training y validation vemos una disminución de overfitting.


## 3.6 Resolviendo un problema de clasificación multiple

Este problema en esencia es MUY similar al problema anterior, simplemente vamos a explicar en qué consiste este nuevo problema de clasificación
y solamente voy a poner los puntos diferenciadores respecto al caso de estudio anterior.

En esta ocasión el problema consiste en entrenar a un modelo de deep learning que sea capaz de diferenciar entre 46 tópicos
diferentes relacionados con temas de noticias. Para ello vamos a utilizar el dataset [Reuters](https://keras.io/api/datasets/reuters/).

Este dataset cuenta con propiedades similares al dataset anterior, también está representado como `label encoding` y por ende
utilizaremos las mismas técnicas de preprocesamiento de datos para transformarlo en `one hot encoding`. Sin embargo, el hecho
de que la clasificación sea multiple, genera un par de diferencias importantes respecto a la clasificación binaria. Veamos
los pasos a continuación.

1. Importando bibliotecas:
    ```python
    import os
    import matplotlib.pyplot as plt
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras.datasets import reuters
    from keras import layers, models
    from keras.utils.np_utils import to_categorical
    import numpy as np
    from keras import regularizers
    ```
2. Cargando Dataset:
   ```python
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    ```
3. Normalizando datos:
    ```python
    def one_hot_encoding(sequences, dim=10000):
        results = np.zeros((len(sequences), dim))
        for i, sequences in enumerate(sequences):
            results[i, sequences] = 1
        return results
    
    
    x_train = one_hot_encoding(train_data)
    x_test = one_hot_encoding(test_data)
    
    # Aquí hay un cambio interesante respecto al ejemplo anterior:
    # Observemos como luce una etiqueta de y_train
    
    print(train_labels[0], train_labels[0].shape)
    
    # Debemos transformar esta salida en una salida de clasificación multiple, esto es lo mismo
    # que hicimos en el problema de clasificación de números escritos a mano del dataset MNIST
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    
    print(y_train[0], y_train[0].shape)
    ```
   Resultados esperados:
   ```commandline
    3 ()
    [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] (46,)
    ```
4. Definiendo la Arquitectura de nuestra red
    ```python
    def architecture(model: models.Sequential, input_shape: tuple, n_classes: int) -> models.Sequential:
        model.add(layers.Dense(128, activation="relu", input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        # IMPORTANTE: Ahora que nuestro problema es de clasificación MULTIPLE nuestra activación de la capa de predicción
        # Es diferente, en este caso usamos softmax, porque nos interesa tener la probabilidad de cada clase a la salida.
        model.add(layers.Dense(n_classes, activation="softmax"))
        return model
    
    
    model_norm = models.Sequential()
    model_norm = architecture(model=model_norm, input_shape=(10000, ), n_classes=46)
    ```

## 3.7 Entrenamiento del modelo de clasificación multiple
5. Compilando la red:
    ```python
    # Dado que nuestro problema tiene varias clases, entonces usaremos "categorical_crossentropy"
    # en lugar de "binary_crossentropy"
    model_norm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    ```
6. Entrenando la red:
    ```python
    history_norm = model_norm.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.3)
    ```
7. Análisis de resultados:

    ```python
    results = model_norm.evaluate(x_test, y_test)
    print(results)
    history_dict = history_norm.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict["acc"]
    val_acc_values = history_dict["val_acc"]
    epoch = range(1, len(loss_values) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Neural Network's Result")
    ax1.set_title("Loss function over epoch")
    ax2.set_title("Acc over epoch")
    ax1.set(ylabel="loss", xlabel="epochs")
    ax2.set(ylabel="acc", xlabel="epochs")
    ax1.plot(epoch, loss_values, 'o-r', label='training')
    ax1.plot(epoch, val_loss_values, '--', label='validation')
    ax2.plot(epoch, acc_values, 'o-r', label='training')
    ax2.plot(epoch, val_acc_values, '--', label='validation')
    ax1.legend()
    ax2.legend()
    plt.savefig("imgs/results.png")
    plt.close()
    ```
   Resultados esperados:
   ```commandline
   71/71 [==============================] - 0s 1ms/step - loss: 1.2337 - acc: 0.8045
   ```
   ![results.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fclasificaci%C3%B3n%20multiple%2Fimgs%2Fresults.png)


## 3.8 Validación de nuestro modelo usando Cross Validation

Antes de continuar con este tema, es un buen momento para repasar conceptos de machine learning. En este momento hablaremos un
poco sobre la estrategia de `Cross Validation` la cual implementaremos en este mini proyecto. 


- **La última palabra siempre la van a tener los datos.**
  - Todas nuestras intuiciones no tiene nada que hacer frente a lo que digan los datos y las matemáticas que aplicamos sobre estos datos. Por eso es importante siempre tener rigurosidad a la hora de evaluar los resultados que estamos recibiendo.

- **Necesitamos mentalidad de testeo.**
  - No se trata solamente de probar un poco al principio y un poco al final, sino que tendremos que probar constantemente durante todo el proceso, para poder encontrar cuál es la solución óptima que realmente nos soluciona el problema que tenemos pendiente, todo esto:
    - con varias formas
    - con varios conjuntos de datos
    - con varias configuraciones de parámetros
    - con varias distribuciones de nuestros datos

- **Todos los modelos son malos, solamente algunos son útiles.**
  - Todos los modelos que nosotros hacemos en últimas son una sobre simplificación de lo que pasa realmente. Entonces nunca nuestros modelos van a corresponder con la realidad al cien por ciento. Si jugamos lo suficiente y si somos lo suficientemente hábiles para configurar, vamos a llegar a un punto donde el modelo que estamos trabajando va a ser útil para ciertos casos específicos dentro del mundo real.

### Tipos de validación

#### Hold-Out

Se trata de dividir nuestros datos entrenamiento/pruebas, básicamente consiste en usar porcentajes fijos, por lo regular 70% de entrenamiento y 30% de pruebas.

![ho1.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fho1.png)

![ho2.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fho2.png)

**¿Cuándo utilizar Hold-out?**

- Se requiere un prototipado rápido.
- No se tiene mucho conocimiento en ML.
- No se cuenta con abundante poder de cómputo.

#### K-Folds

Usar validación cursada K-Fold, aquí vamos a plegar nuestros datos k veces, el k es un parámetro que nosotros definimos y en esos pliegues vamos a utilizar diferentes partes de nuestro dataset como entrenamiento y como test, de tal manera que intentemos cubrir todos los datos de entrenamiento y de test, al finalizar el proceso.

![kf1.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fkf1.png)

![kf2.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Fkf2.png)

**¿Cuándo utilizar K-Folds?**

- Recomendable en la mayoría de los casos.
- Se cuenta con un equipo suficiente para desarrollar ML.
- Se require la integración con técnicas de optimización paramétrica.
- Se tiene más tiempo para las pruebas.

#### LOOCV

Validación cruzada LOOCV, Leave One Out Cross Validation. Este es el método más intensivo, ya que haremos una partición entre entrenamiento y pruebas, porque vamos a hacer entrenamiento con todos los datos, salvo 1 y vamos a repetir este proceso tantas veces hasta que todos los datos hayan sido probados.

![lo.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fimgs%2Flo.png)

**¿Cuándo utilizar LOOCV?**

- Se tiene gran poder de cómputo
- Se cuenta con pocos datos para poder dividir por Train/Test
- Cuando se quiere probar todos los casos posibles (para personas con TOC)


## 3.9 Resolviendo un problema de regresión

Tomando lo aprendido en la sección anterior, en esta ocasión vamos a resolver un problema de `house pricing`, para ello 
vamos a utilizar el dataset de [Boston Housing price regression dataset](https://keras.io/api/datasets/boston_housing/)
el cual contiene `404` muestras de casas, cada casa contiene `13` variables continuas que expresan atributos de la casa.

Nuestra tarea entonces será predecir el valor de la casa con base en los `13` atributos mencionados anteriormente. La 
metodología como es costumbre será similar a los ejemplos anteriores, sin embargo; para este específico mini proyecto utilizaremos
una estrategia de validación del modelo diferente. 

En este ejemplo usaremos `k-Fold Cross Validation` descrito en la clase anterior. Para ello crearemos una función que contenga
el modelo, su arquitectura y compilado y lo correremos para cada uno de los `folds` de validación.

> ## Nota:
> El código completo lo puedes encontrar [aquí](3%20Manejo%20de%20redes%20neuronales%20con%20Keras/regresión/main.py)


1. **Importando bibliotecas necesarias:**
    ```python
    import pandas as pd
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import numpy as np
    from keras.datasets import boston_housing
    from keras import models, layers
    # Biblioteca para implementar K-Fold CrossValidation
    from sklearn.model_selection import KFold
    # Biblioteca para la correcta normalización de datos numéricos
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    ```
    Lo más relevante a resaltar aquí es el uso de KFold y StandardScaler provenientes de `sklearn`


2. **Definiendo la arquitectura del modelo:**

    La siguiente función se encarga de crear la arquitectura y compilar al nuestro modelo de regresión:
    ```python
    def build_model_regression(dim):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_dim=dim))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        # Como la última capa es una predicción de regresión, NO necesita una capa de activación
        model.add(layers.Dense(1))
        # El error sí será el mean squared error, pero la métrica debe ser diferente, en este caso max absolute error
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        return model
    ```
    Los puntos claves a destacar son: 
    - La función de perdida es `mean squared error (mse)`
    - Como nuestro problema es una regresión la métrica de éxito es `mean absolut error (mae)`
    - Dado que la última capa del modelo NO es para clasificar, NO es necesario que tenga ninguna función de activación.
    - `input_dim` es una variante de `input_shape` la cual solo necesita especificar la canatidad de elementos en la entrada
    - de la red.

## 3.10 Entrenamiento del modelo de regresión

3. **Cargando el dataset:**
    
    Este paso es básicamente el mismo que el de los ejemplos anteriores pero con el nuevo dataset de este ejemplo:
    ```python
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    print(train_data[0])
    print(train_targets[0])
    ```
    Respuesta esperada:
    ```commandline
    [  1.23247   0.        8.14      0.        0.538     6.142    91.7
       3.9769    4.      307.       21.      396.9      18.72   ]
    15.2
    ```
    Podemos conocer el valor de los 13 atributos de la casa y conocer que el precio de la misma es de 15.2 Mil dólares.
4. **Generando particiones de K-fold**

    El primer paso es definir el objeto `kf` de la clase `KFold`y pedirle que genere 5 `folds`
    ```python
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ```
    Adicionalmente, para que estas particiones sean replicables pondremos su `random_state` en un valor fijo.
    Ahora definimos una función auxiliar que nos permita dado unos vectores de valores de datos y de targets regrese las particiones
    de `x_train, x_test, y_train, y_test`:
    ```python
    def train_test_split_kf(xs: np.array, ys: np.array, train_size: np.array, test_size: np.array) -> np.array:
        x_train_ = xs[train_size]
        x_test_ = xs[test_size]
        y_train_ = ys[train_size]
        y_test_ = ys[test_size]
        return x_train_, x_test_, y_train_, y_test_
    ```
    Con la función definida podemos proceder a utilizar los rangos de `train y test` creados por el objeto `kf` y de ahí conseguir
    las particiones esperadas:
    
    ```python
    all_history = []  # Aquí guardaremos los resultados de cada fold
    
    for n_fold, (train, test) in enumerate(kf.split(train_data)):
        print(f"\t-I'm running fold {n_fold + 1}")
        x_train, x_test, y_train, y_test = train_test_split_kf(xs=train_data, ys=train_targets,
                                                               train_size=train, test_size=test)
    ```
    > Nota:
    > La función kf.split(vector) toma un vector de entrada y regresa los indices de las particiones para `train y test`


5. **Normalizando datos de entrenamiento**

    Dado que estamos en un problema de regresión es MUY buena idea normalizar los datos, puesto que si tenemos una variable llamada:
    número de cuartos, quizá el rango de valores pueda ir de (1, 5) sin embargo, si tenemos otra que sea año de construcción cuyos valores
    esten entre (1930, 2022) podría aparentar que numéricamente hablando es más importante el año de construcción que el número de cuartos.
    
    Para evitar estos inconvenientes lo que se hace es normalizar los datos, para ello a cada muestra del vector se le resta su 
    promedio y se le divide entre su desviación standard. A este tipo de normalizado se le conoce como: `StandardScaler`
    Primero: Creamos un objeto de la clase `StandarScaler`
    ```python
    standard_scaler = StandardScaler()
    ```
    A este objeto lo vamos a `entrenar` con la distribución de datos de nuestra partición de entrenamiento `x_train`:
    ```python
    standard_scaler.fit(x_train)
    ```
    Esto le permite al modelo aprender los valores de `mean` y `std` Ahora podemos proceder a normalizar los datos de entrenamiento y validación:
    ```python
    x_train_s = standard_scaler.transform(x_train)
    # Es MUY importante tener en cuenta que los datos de prueba NO LOS CONOZCO entonces NO tiene sentido obtener
    # el promedio y desviación standard de una muestra que NO conozco por eso estoy normalizando con el promedio
    # y std de la muestra a la que mi modelo sí tenía acceso mientras fue entrenada.
    x_test_s = standard_scaler.transform(x_test)
    ```

6. **Entrenando el modelo**
    
    Ahora podemos construir un modelo que sirva para las particiones de entrenamiento y validación de nuestros `fold` y ponerlo a
    entrenar por un número de épocas predefinido:
    ```python
    model = build_model_regression(dim=13)
    history = model.fit(x_train_s, y_train, epochs=n_epochs, batch_size=16,
                        validation_data=(x_test_s, y_test), verbose=0)
    all_history.append(history.history)
    ```

## 3.11 Análisis de resultados del modelo de regresión

7. Re estructurando datos de salida

    Observemos los resultados de nuestros modelos que fueron guardados en `all_history`:
    ```python
    out = all_history
    print(out)
    ```
    Respuesta esperada:
    ```commandline
    [{'loss': [546.1715087890625, 396.9112548828125, 247.38885498046875, 133.14358520507812, 76.148681640625, ...], 
       'mae': [21.413293838500977, 17.873849868774414, 13.284331321716309, 8.88094425201416, 6.3826823234558105, ...], 
       'val_loss': [354.6539611816406, 238.64332580566406, 141.55055236816406, 78.4019546508789, 51.63487243652344, ...], 
       'val_mae': [17.48577117919922, 14.001570701599121, 10.338260650634766, 7.059571266174316, 5.430708885192871, ...]}, 
       
       {'loss': [508.8610534667969, 375.5047607421875, 231.6179656982422, 115.08528137207031, 63.184120178222656, ...], 
       'mae': [20.488609313964844, 16.794652938842773, 12.525876998901367, 8.449064254760742, 6.081153392791748, ...], 
       'val_loss': [424.28839111328125, 291.0716552734375, 168.71376037597656, 106.91384887695312, 76.11128234863281, ...], 
       'val_mae': [17.950767517089844, 14.14747142791748, 9.914441108703613, 7.694438934326172, 6.680331230163574]}, 
       
       ... 
       
       ] 
    ```
    Esta estructura de diccionarios guardados en una lista NO es tan conveniente, lo que desearemos es un dataframe que contenga
    el promedio de los 5 `folds` sobre cada una de las n `epochs`eso lo podemos hacer de la siguiente manera:
    ```python
    df = {}
    for key in out[0].keys():
        row = []
        for fold in out:
            row.append(fold[key])
        row = np.array(row).mean(axis=0)
        df[key] = row
    frame = pd.DataFrame(df)
    frame = frame[offset:]
    print(frame)
    ```
    Respuesta esperada:
    ```commandline
             loss       mae   val_loss   val_mae
    5   39.485257  4.583744  36.213194  4.253143
    6   30.369080  3.897518  30.883676  3.902345
    7   25.733508  3.562452  26.227117  3.475329
    8   22.751280  3.313163  25.029307  3.454233
    9   20.649399  3.139169  23.158773  3.239379
    10  18.837821  2.997525  21.098302  3.168812
    ```
8. Análisis de resultados final

    Finalmente, podemos graficar los resultados de nuestro `frame` de resultados y observar los resultados de nuestro modelo con 
    nuestra partición de `test` que hasta este momento NO habíamos utilizado.
    
    ```python
    metric = "mae"
    offset = 5
    loss_values = frame['loss']
    val_loss_values = frame['val_loss']
    metric_values = frame[metric]
    val_metric_values = frame[f"val_{metric}"]
    epoch = range(1, len(loss_values) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.suptitle("Neural Network's Result")
    ax1.set_title("Loss function over epoch")
    ax2.set_title(f"{metric} over epoch")
    ax1.set(ylabel="loss", xlabel="epochs")
    ax2.set(ylabel=metric, xlabel="epochs")
    ax1.plot(epoch, loss_values, 'o-r', label='training')
    ax1.plot(epoch, val_loss_values, '--', label='validation')
    ax2.plot(epoch, metric_values, 'o-r', label='training')
    ax2.plot(epoch, val_metric_values, '--', label='validation')
    ax1.legend()
    ax2.legend()
    plt.savefig("imgs/results.png")
    plt.close()
    ```
    Respuesta esperada:
    ![results.png](3%20Manejo%20de%20redes%20neuronales%20con%20Keras%2Fregresi%C3%B3n%2Fimgs%2Fresults.png)
    
    Finalmente, evaluando la partición de pruebas:
    ```python
    results = model.evaluate(test_data, test_targets)
    ```
    Respuesta esperada:
    ```commandline
    4/4 [==============================] - 0s 1ms/step - loss: 960760.6875 - mae: 906.1437
    ```
    ÉXITO TOTAL ÉXITO ROTUNDO 
    Hemos conseguido un error de menos de 1000 dólares en la predicción del precio de una nueva casa.

# 4 Cierre

Muchas Felicidades si has llegado hasta aquí has terminado exitosamente el curso de [Fundamentos de Redes Neuronales con Python y Keras](https://platzi.com/cursos/redes-neuronales/)
El curso es impartido por el profesor [Carlos Alarcón](https://platzi.com/profes/alarcon7a/) Y este repositorio me pertenece a mí:
[Gabriel Ichcanziho](https://www.linkedin.com/in/ichcanziho/).

Antes de que te vayas, te dejo un breve resumen de cuando utilizar diferentes tipos de funciones de activación y funciones de perdida
de acuerdo al tipo de problema que estás enfrentando.

![c1.png](4%20cierre%2Fc1.png)

## ¿Qué sigue a continuación?

Ahora que hemos terminado el curso de **Fundamentos de Redes Neuronales con Python y Keras** ¿Qué podemos esperar de Deep Learning?

Pues próximos tópicos que podemos aprender son:

**CNN (Convolutional Neural Networks)**

![cnn.gif](4%20cierre%2Fcnn.gif)

**NLP (Natural Language Processing)**

![nlp.gif](4%20cierre%2Fnlp.gif)

**Hyper Parameters tuning**

![hpt.gif](4%20cierre%2Fhpt.gif)

**Tensorboards**

![tensorboard.gif](4%20cierre%2Ftensorboard.gif)

**Generative Adversarial Networks (Gans)**

![gans.gif](4%20cierre%2Fgans.gif)

**Long short-term memory (LSTM)**

![lstm.gif](4%20cierre%2Flstm.gif)

Ahora acompañame al siguiente curso: [Curso de Redes neuronales convolucionales con Python y Keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)


