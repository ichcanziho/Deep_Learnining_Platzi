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
> El código completo de estas clases lo puedes encontrar [aquí]([red_neuronal.py](2%20Redes%20neuronales%20con%20Python%2Fprimer%20red%2Fred_neuronal.py))

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
## 3.2 Resolviendo un problema de clasificación binaria
## 3.3 Entrenamiento del modelo de clasificación binaria
## 3.4 Regularización - Dropout
## 3.5 Reduciendo el overfitting
## 3.6 Resolviendo un problema de clasificación multiple
## 3.7 Entrenamiento del modelo de clasificación multiple
## 3.8 Resolviendo un problema de regresión
## 3.9 Entrenamiento del modelo de regresión
## 3.10 Análisis de resultados del modelo de regresión
