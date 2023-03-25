# Curso de Redes Neuronales Convolucionales con Python y keras

En este curso avanzado eleva tus conocimientos de deep learning. Comprende el funcionamiento de las redes neuronales convolucionales. Sigue el camino de la visión artificial donde este tipo de red neuronal es utilizada.

- Optimizarás tus redes aplicando data augmentation, callbacks y batch normalization.
- Comprenderás el kernel, padding, strides y la capa de pooling.
- Manejarás imágenes con Python para su uso en redes neuronales.

> ## NOTA:
> Antes de continuar te invito a que revises el curso anterior:
> 
> [1: Curso de fundamentos de redes neuronales con Python y Keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
>
> Sin más por agregar disfruta de este curso

# Índice:

- [1. Redes convolucionales y su importancia](#1-redes-convolucionales-y-su-importancia)
  - [1.1 La importancia del computer vision](#11-la-importancia-del-computer-vision)
  - [1.2 ¿Qué herramientas usaremos para redes neuronales convolucionales?](#12-qué-herramientas-usaremos-para-redes-neuronales-convolucionales)
  - [1.3 ¿Qué son las redes convolucionales?](#13-qué-son-las-redes-convolucionales)
- [2. Mi primera red neuronal convolucional](#2-mi-primera-red-neuronal-convolucional)
  - [2.1 Creando nuestra primer red convolucional](#21-creando-nuestra-primer-red-convolucional)
  - [2.2 Entrenando nuestra primera red convolucional](#22-entrenando-nuestra-primera-red-convolucional)
- [3. Manejo de imágenes](#3-manejo-de-imágenes)
  - [3.1 Consejos para el manejo de imágenes](#31-consejos-para-el-manejo-de-imágenes)
  - [3.2 Manejo de imágenes con Python](#32-manejo-de-imágenes-con-python)
- [4. Fundamentos de redes neuronales convolucionales](#4-fundamentos-de-redes-neuronales-convolucionales)
  - [4.1 Kernel en redes neuronales](#41-kernel-en-redes-neuronales)
  - [4.2 El Kernel en acción](#42-el-kernel-en-acción)
  - [4.3 Padding y Strides](#43-padding-y-strides)
  - [4.4 Capa de pooling](#44-capa-de-pooling)
  - [4.5 Arquitectura de redes convolucionales](#45-arquitectura-de-redes-convolucionales)
  - [4.6 Quizz Fundamentos de redes neuronales convolucionales](#46-quizz-fundamentos-de-redes-neuronales-convolucionales)
- [5. Resolviendo un problema de clasificación](#5-resolviendo-un-problema-de-clasificación)
  - [5.1 Clasificación con redes neuronales convolucionales](#51-clasificación-con-redes-neuronales-convolucionales)
  - [5.2 Creación de red convolucional para clasificación](#52-creación-de-red-convolucional-para-clasificación)
  - [5.3 Entrenamiento de un modelo de clasificación con redes convolucionales](#53-entrenamiento-de-un-modelo-de-clasificación-con-redes-convolucionales)
- [6. Optimización de una red neuronal convolucional](#6optimización-de-una-red-neuronal-convolucional)
  - [6.1 Data augmentation](#61-data-augmentation)
  - [6.2 Aplicando data augmentation](#62-aplicando-data-augmentation)
  - [6.3 Callbacks: early stopping y checkpoints](#63-callbacks--early-stopping-y-checkpoints)
  - [6.4 Batch normalization](#64-batch-normalization)
  - [6.5 Optimización de modelo de clasificación](#65-optimización-de-modelo-de-clasificación)
  - [6.6 Entrenamiento de nuestro modelo de clasificación optimizado](#66-entrenamiento-de-nuestro-modelo-de-clasificación-optimizado)
  - [6.7 Quizz: Optimización de redes neuronales convolucionales](#67-quizz--optimización-de-redes-neuronales-convolucionales)
- [7. Resolviendo una competencia de Kaggle](#7-resolviendo-una-competencia-de-kaggle)
  - [7.1 Clasificando entre perros y gatos](#71-clasificando-entre-perros-y-gatos)
  - [7.2 Entrenamiento del modelo de clasificación de perros y gatos](#72-entrenamiento-del-modelo-de-clasificación-de-perros-y-gatos)
- [8. Cierre](#8-cierre)
  - [8.1 Siguientes pasos](#81-siguientes-pasos)

# 1. Redes convolucionales y su importancia

## 1.1 La importancia del computer vision

La visión artificial es un campo de la IA que permite que las computadoras y los sistemas obtengan información significativa 
de imágenes digitales, videos y otras entradas visuales, y tomen acciones o hagan recomendaciones basadas en esa información. 
Si la IA permite que las computadoras piensen, la visión artificial les permite ver, observar y comprender.

La visión artificial funciona de manera muy similar a la visión humana, excepto que los humanos tienen una ventaja. La 
vista humana tiene la ventaja de las experiencias y los contextos aprendidos para diferenciar entre los objetos, qué tan 
lejos están, si se están moviendo o si hay algo mal en una imagen.

La visión artificial entrena a las máquinas para realizar estas funciones, pero tiene que hacerlo en mucho menos tiempo 
con cámaras, datos y algoritmos en lugar de retinas, nervios ópticos y una corteza visual. Debido a que un sistema capacitado 
para inspeccionar productos o la manufactura de estos puede analizar miles de productos o procesos por minuto puede superar 
rápidamente las capacidades humanas, notando defectos o problemas imperceptibles.

La visión artificial se utiliza en industrias que van desde la energía y los servicios públicos hasta la manufactura y la 
industria automotriz, y el mercado sigue creciendo. Se espera que alcance los USD 48.6 miles de millones en 2022.1

### ¿Cómo funciona la visión artificial?

Machine learning utiliza modelos algorítmicos que permiten que una computadora se enseñe a sí misma sobre el contexto de 
los datos visuales. Si se alimentan suficientes datos a través del modelo, la computadora "observará" los datos y se enseñará 
a diferenciar una imagen de otra. Los algoritmos permiten que la máquina aprenda por sí misma, en lugar de que alguien la 
programe para reconocer una imagen.

Una CNN ayuda a un modelo de machine learning o deep learning a "ver" al dividir las imágenes en píxeles a los que se les 
asignan etiquetas o rótulos. Utiliza las etiquetas para realizar convoluciones (una operación matemática en dos funciones 
para producir una tercera función) y hace predicciones sobre lo que está "viendo". La red neuronal ejecuta convoluciones 
y verifica la precisión de sus predicciones en una serie de iteraciones hasta que las predicciones comienzan a hacerse realidad. 
Luego reconocerá o verá imágenes de una manera similar a los humanos.

## Principales usos de la visión artificial

1. **Clasificación de imágenes.**

    ![reco1.png](imgs%2Fintroducci%C3%B3n%2Freco1.png)
    
    Ve una imagen y puede clasificarla (un perro, una manzana, la cara de una persona). Más precisamente, puede predecir 
    con precisión que una imagen determinada pertenece a un cierto tipo. Por ejemplo, una empresa de redes sociales podría 
    querer usarlo para identificar y segregar automáticamente las imágenes objetables cargadas por los usuarios.
2. **Detección de objetos.**

    ![reco2.png](imgs%2Fintroducci%C3%B3n%2Freco2.png)
    Puede usar la clasificación de imágenes para identificar una determinada clase de imagen y luego detectar y tabular su 
    apariencia en una imagen o video. Los ejemplos incluyen la detección de daños en una línea de montaje o la identificación 
    de maquinaria que requiera mantenimiento.
3. **Transferencia de estilos.**

    ![reco3.png](imgs%2Fintroducci%C3%B3n%2Freco3.png)
    La transferencia de estilo neuronal (Neural Style Transfer) es una técnica de aprendizaje automático para combinar el 
    contenido semántico de una imagen con el estilo artístico de otra. Este proceso considera dos imágenes, la imagen de 
    contenido y la imagen de estilo. Podemos calcular una imagen de salida con el contenido original, pero con un nuevo estilo, 
    utilizando Redes Neuronales Convoluciones (CNN). 

    La transferencia de estilo neuronal fue descrita por primera vez por Gatys et al. en A Neural Algorithm of Artistic Style (2015), 
    donde se muestra que la tarea de transferir el estilo de una imagen al contenido de otra puede plantearse como un problema de 
    optimización que puede resolverse mediante el entrenamiento de una red neuronal.

4. **GANs**

    ![reco4.png](imgs%2Fintroducci%C3%B3n%2Freco4.png)
    Las Redes Neuronales Generativas Adversarias también se denominan GANs por sus siglas en inglés (Generative Adversarial Networks). 
    También lo he visto traducido al español como Redes Antagónicas. Las GANs son una forma nueva de usar deep learning para 
    generar imágenes que parecen reales. También pueden generar otro tipo de datos tales como música.

5. **Reconocimiento facial**

    ![reco5.png](imgs%2Fintroducci%C3%B3n%2Freco5.png)
    Es un software que identifica o confirma la identidad de una persona a partir del rostro. Funciona mediante la identificación 
    y medición de los rasgos faciales en una imagen. El reconocimiento facial puede identificar rostros humanos en imágenes o videos, 
    determinar si el rostro que aparece en dos imágenes pertenece a la misma persona o buscar un rostro entre una gran 
    colección de imágenes existentes. Los sistemas de seguridad biométricos utilizan el reconocimiento facial para identificar 
    de forma exclusiva a las personas durante la incorporación o el inicio de sesión de los usuarios, así como para reforzar 
    la actividad de autenticación de estos. Los dispositivos móviles y personales también utilizan con frecuencia la tecnología 
    de los analizadores faciales para proteger los dispositivos.

## 1.2 ¿Qué herramientas usaremos para redes neuronales convolucionales?

Al igual que en el curso anterior de esta ruta: [Curso de fundamentos de redes neuronales con Python y Keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales). Este curso
se enfocará en el uso de Keras y Python para programar redes neuronales.

**¿Qué es Keras?**

Keras es una biblioteca de código abierto (con licencia MIT) escrita en Python, que se basa principalmente en el trabajo 
de François Chollet, un desarrollador de Google, en el marco del proyecto ONEIROS (Open-ended Neuro-Electronic Intelligent 
Robot Operating System). La primera versión de este software multiplataforma se lanzó el 28 de marzo de 2015. 
El objetivo de la biblioteca es acelerar la creación de redes neuronales: para ello, Keras no funciona como un 
framework independiente, sino como una interfaz de uso intuitivo (API) que permite acceder a varios frameworks de 
aprendizaje automático y desarrollarlos. Entre los frameworks compatibles con Keras, se incluyen Theano, Microsoft 
Cognitive Toolkit (anteriormente CNTK) y TensorFlow.

![herra1.png](imgs%2Fintroducci%C3%B3n%2Fherra1.png)

Entonces Keras no es más que una forma más amigable de acceder a Frameworks independientes, a su vez estos Frameworks
usaran libererías de más bajo nivel para comunicarse directamente con el Hardware de nuestro dispositivo para de esta
forma acceder y utilizar a la `GPU` o `CPU` de nuestro computador. 

> ## NOTA del curso:
> 
> Al igual que en el curso anterior, el curso de PLATZI está creado para utilizar NoteBooks como herramienta principal.
> Sin embargo, este repositorio está pensando para implementar el código en local. 
> 
> El curso propone utilizar [Kaggle](https://www.kaggle.com/)
> ![herra2.png](imgs%2Fintroducci%C3%B3n%2Fherra2.png)
> Como página para acceder a NoteBooks de python. 
> 
> ¿Por qué kaggle sobre Google Colab?
> 
> Debemos recordar que nosotros estamos usando recursos computacionales, los cuales, al proveedor no le son gratis; así 
> que todos los servicios nos pondrán un límite de uso, por ejemplo:
> 
> - **Google Colab (version free)** - no publica estos límites. Uno de los motivos es que pueden (y suelen) variar rápidamente. 
> Pero, los usuarios que usan Colab para ejecutar operaciones informáticas de larga duración o que han usado más recursos 
> recientemente tienen más posibilidades de que se les establezcan límites de uso y de que se les restrinja temporalmente 
> el acceso a las GPUs y TPUs
>
> - **Kaggle** - en 2020 implementaron un nuevo sistema el cual por ~30 horas semanales tendrás un poder computacional que no varía.
> 
> **Conclusión**: Kaggle nos da un poder específico que no varía, sin importar la cantidad de cómputo que tengamos (pero tenemos 30 horas semanales), mientras Google nos limita dependiendo de la actividad.

## 1.3 ¿Qué son las redes convolucionales?

Las Redes Neuronales Convolucionales (Convolutional Neural Networks o CNNs, por sus siglas en inglés) son un tipo de algoritmo 
de aprendizaje profundo que se utiliza comúnmente en tareas de visión por computadora, como la clasificación de imágenes 
y la detección de objetos.

![que son.gif](imgs%2Fintroducci%C3%B3n%2Fque%20son.gif)

Las CNNs se llaman así porque utilizan una operación matemática llamada convolución para procesar los datos de entrada. 
La convolución implica la aplicación de un filtro o kernel a la imagen de entrada para detectar características específicas, 
como bordes o texturas.

A diferencia de las redes neuronales tradicionales, que procesan los datos de entrada como una matriz unidimensional, 
las CNNs pueden trabajar con datos de entrada en dos o más dimensiones, lo que las hace ideales para tareas que involucran 
imágenes, videos y otros datos similares.

![que son 2.gif](imgs%2Fintroducci%C3%B3n%2Fque%20son%202.gif)

En una CNN, los datos de entrada se procesan a través de capas de neuronas que realizan la convolución y otros cálculos 
matemáticos, seguidos de capas de agrupamiento o pooling para reducir el tamaño de la imagen. Luego, la red pasa por 
varias capas de neuronas totalmente conectadas que actúan la clasificación final.

![que son 3.gif](imgs%2Fintroducci%C3%B3n%2Fque%20son%203.gif)

Las CNNs han demostrado ser muy efectivas en una variedad de tareas de visión por computadora y han sido utilizadas en 
aplicaciones como la identificación de objetos en imágenes, la detección de rostros, la segmentación de imágenes y la 
clasificación de imágenes médicas.

# 2. Mi primera red neuronal convolucional

A lo largo de esta sección estaremos creando nuestra primera CCN con Python y Keras. En este momento del curso hay muchos
términos que NO serán explicados explícitamente, el objetivo de estas siguientes dos clases es tener familiaridad con el
api de Keras y como podemos usarla para implementar nuestra primera CNN. Más adelante en el curso los términos utilizados se
irán explicando con más detalle. 

Nuestro trabajo de esta sección consiste en resolver un problema de clasificación multiple. Poder clasificar entre 10 tipos
de prenda de vestir. Para ello vamos a utilizar el dataset [Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/)

El cual cuenta con 60,000 imágenes de 28x28 píxeles en escala de grises. Adicionalmente, cuenta con un set de prueba de 
10,000 imágenes con las mismas características que el set de entrenamiento. 

Las clases disponibles son:

| **Label** | **Description** |
|:---------:|-----------------|
|     0     |   T-shirt/top   |
|     1     |     Trouser     |
|     2     |     Pullover    |
|     3     |      Dress      |
|     4     |       Coat      |
|     5     |      Sandal     |
|     6     |      Shirt      |
|     7     |     Sneaker     |
|     8     |       Bag       |
|     9     |    Ankle boot   |


> ## Nota:
> El código completo de esta sección lo puedes encontrar [aqui](Mi%20primera%20CNN/main.py) 

## 2.1 Creando nuestra primer red convolucional

**1: Importando bibliotecas necesarias**
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.utils import to_categorical
from keras import Sequential
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
```

**2: Descargando los datos necesarios**
```python
# Descargando dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Análisis exploratorio
print(train_images.shape)
plt.imshow(train_images[0])
plt.savefig("imgs/train0.jpg")
plt.close()
```
Respuesta esperada:
```commandline
(60000, 28, 28)
```
![train0.jpg](Mi%20primera%20CNN%2Fimgs%2Ftrain0.jpg)

**3: Normalizando datos**
> Nota: Este es un paso común de normalizado que YA hemos explorado en el curso anterior.
```python
# Normalizado de imágenes
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
# a diferencia de las redes neuronales normales, dónde la entrada debía ser un vector de 1 dim
# en las CNN la entrada es una matriz, es por eso que en el reshape debemos tomar en cuenta
# [[]].reshape(n, x, y, c)
# n, x, y, c -> n = número de imágenes, x = ancho de la imagen, y = largo de la imagen, c = número de canales
# Dado que nuestras imágenes están en escala de grises, entonces el número de canales que maneja es 1.
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
# CONOCIMIENTO del curso anterior -> Transformando números del 0 al 9 (10 clases) en su One Hot Encoding
train_labels_categorical = to_categorical(train_labels, 10)
test_labels_categorical = to_categorical(test_labels, 10)
```
Cabe destacar que por ser una CNN NO fue necesario usar un vector 1d para esta arquitectura.

**4: Definimos la arquitectura**

```python
def architecture(model_: Sequential):
    model_.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    # Esta capa sirve para aplanar y pasar de redes convolucionales a normales
    model_.add(Flatten())
    model_.add(Dense(256, activation="relu"))
    model_.add(Dropout(0.5))
    # Como es un problema de clasificación multilabel usamos softmax como activación de la última capa
    model_.add(Dense(10, activation="softmax"))
    print(model_.summary())
    # Compilamos el modelo con la información que YA conocemos (la última capa de la red CNN es igual a las que ya hemos
    # trabajo anteriormente)
    model_.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model_
```


## 2.2 Entrenando nuestra primera red convolucional

**Auxiliar: Definiendo función para graficar resultados**
```python
def plot_results(history_, metric, fname):
    history_dict = history_.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    metric_values = history_dict[metric]
    val_metric_values = history_dict[f"val_{metric}"]
    epoch = range(1, len(loss_values) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
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
    plt.savefig(f"imgs/{fname}")
    plt.close()
```

**5: Entrenando el modelo**
```python
model = Sequential()
model = architecture(model)
history = model.fit(train_images, train_labels_categorical, batch_size=64, epochs=10, validation_split=0.3)
```
Respuesta esperada:
- architecture(model)
```commandline
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 64)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 14, 14, 64)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 14, 32)        8224      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 7, 7, 32)         0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 7, 7, 32)          0         
                                                                 
 flatten (Flatten)           (None, 1568)              0         
                                                                 
 dense (Dense)               (None, 256)               401664    
                                                                 
 dropout_2 (Dropout)         (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                2570      
                                                                 
=================================================================
Total params: 412,778
Trainable params: 412,778
Non-trainable params: 0
```
- model.fit()
```commandline
Epoch 1/10
657/657 [==============================] - 5s 6ms/step - loss: 0.6350 - accuracy: 0.7688 - val_loss: 0.4131 - val_accuracy: 0.8517
Epoch 2/10
657/657 [==============================] - 4s 6ms/step - loss: 0.4317 - accuracy: 0.8450 - val_loss: 0.3430 - val_accuracy: 0.8776
Epoch 3/10
657/657 [==============================] - 4s 6ms/step - loss: 0.3867 - accuracy: 0.8617 - val_loss: 0.3216 - val_accuracy: 0.8836
Epoch 4/10
657/657 [==============================] - 4s 6ms/step - loss: 0.3534 - accuracy: 0.8725 - val_loss: 0.3114 - val_accuracy: 0.8878
Epoch 5/10
657/657 [==============================] - 4s 6ms/step - loss: 0.3344 - accuracy: 0.8782 - val_loss: 0.2914 - val_accuracy: 0.8934
Epoch 6/10
657/657 [==============================] - 4s 6ms/step - loss: 0.3200 - accuracy: 0.8856 - val_loss: 0.2692 - val_accuracy: 0.9022
Epoch 7/10
657/657 [==============================] - 4s 6ms/step - loss: 0.3092 - accuracy: 0.8871 - val_loss: 0.3074 - val_accuracy: 0.8886
Epoch 8/10
657/657 [==============================] - 4s 6ms/step - loss: 0.3005 - accuracy: 0.8912 - val_loss: 0.2636 - val_accuracy: 0.9043
Epoch 9/10
657/657 [==============================] - 4s 6ms/step - loss: 0.2999 - accuracy: 0.8929 - val_loss: 0.2756 - val_accuracy: 0.8996
Epoch 10/10
657/657 [==============================] - 4s 6ms/step - loss: 0.2927 - accuracy: 0.8944 - val_loss: 0.3042 - val_accuracy: 0.8992
```

**6: Análisis de resultados**
```python
score = model.evaluate(test_images, test_labels_categorical)
print(score)
plot_results(history, "accuracy", "results_base.png")
```
Respuesta esperada:
```commandline
313/313 [==============================] - 1s 2ms/step - loss: 0.3146 - accuracy: 0.8929
```
![results_base.png](Mi%20primera%20CNN%2Fimgs%2Fresults_base.png)

### BONUS: Una forma alternativa de resolución del problema.

Para esta forma alternativa, vamos a modificar la arquitectura del modelo levemente, en esta ocasión
vamos a utilizar como función de perdida `SparseCategoricalCrossentropy` está nos va a permitir trabajar directamente con
los valores originales de los `labels, train_labels, test_labels` sin necesidad de pasarlos por la función `to_categorical`
adicionalmente, nos permitirá NO usar la función `softmax` que hemos utilizado siempre en problemas de clasificación multiple.

**Nueva arquitectura:**
```python
def architecture_sparse(model_: Sequential):
    model_.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Flatten())
    model_.add(Dense(256, activation="relu"))
    model_.add(Dropout(0.5))
    # Utilizando como perdida la SparceCategoricalCrossentropy NO es necesario usar "Softmax" como activación
    model_.add(Dense(10))
    print(model_.summary())
    model_.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer="rmsprop", metrics=["accuracy"])
    return model_
```
**Entrenamos nuestro nuevo modelo:**
```python
model = Sequential()
model = architecture_sparse(model)
# Este tipo de modelo NO me exige usar las etiquetas como categóricas, por eso puedo usar train_labels normal.
history = model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_split=0.3)
```
Respuesta esperada:
```commandline
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_2 (Conv2D)           (None, 28, 28, 64)        320       
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 14, 14, 64)       0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 14, 14, 64)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 14, 14, 32)        8224      
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 32)         0         
 2D)                                                             
                                                                 
 dropout_4 (Dropout)         (None, 7, 7, 32)          0         
                                                                 
 flatten_1 (Flatten)         (None, 1568)              0         
                                                                 
 dense_2 (Dense)             (None, 256)               401664    
                                                                 
 dropout_5 (Dropout)         (None, 256)               0         
                                                                 
 dense_3 (Dense)             (None, 10)                2570      
                                                                 
=================================================================
Total params: 412,778
Trainable params: 412,778
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/10
657/657 [==============================] - 4s 5ms/step - loss: 0.6205 - accuracy: 0.7737 - val_loss: 0.3955 - val_accuracy: 0.8567
Epoch 2/10
657/657 [==============================] - 4s 5ms/step - loss: 0.4216 - accuracy: 0.8498 - val_loss: 0.3708 - val_accuracy: 0.8619
Epoch 3/10
657/657 [==============================] - 4s 5ms/step - loss: 0.3773 - accuracy: 0.8650 - val_loss: 0.3188 - val_accuracy: 0.8849
Epoch 4/10
657/657 [==============================] - 3s 5ms/step - loss: 0.3505 - accuracy: 0.8736 - val_loss: 0.2995 - val_accuracy: 0.8918
Epoch 5/10
657/657 [==============================] - 3s 5ms/step - loss: 0.3326 - accuracy: 0.8799 - val_loss: 0.2840 - val_accuracy: 0.8966
Epoch 6/10
657/657 [==============================] - 3s 5ms/step - loss: 0.3195 - accuracy: 0.8846 - val_loss: 0.2849 - val_accuracy: 0.8952
Epoch 7/10
657/657 [==============================] - 4s 5ms/step - loss: 0.3121 - accuracy: 0.8872 - val_loss: 0.2726 - val_accuracy: 0.9024
Epoch 8/10
657/657 [==============================] - 3s 5ms/step - loss: 0.3054 - accuracy: 0.8915 - val_loss: 0.2749 - val_accuracy: 0.8991
Epoch 9/10
657/657 [==============================] - 3s 5ms/step - loss: 0.2993 - accuracy: 0.8938 - val_loss: 0.2634 - val_accuracy: 0.9070
Epoch 10/10
657/657 [==============================] - 4s 6ms/step - loss: 0.2942 - accuracy: 0.8944 - val_loss: 0.2539 - val_accuracy: 0.9071
```
**Análisis de resultados**
```python
score = model.evaluate(test_images, test_labels)
print(score)
plot_results(history, "accuracy", "results_sparse.png")
```
Respuesta esperada:
```commandline
313/313 [==============================] - 0s 1ms/step - loss: 0.2676 - accuracy: 0.9012
```
![results_sparse.png](Mi%20primera%20CNN%2Fimgs%2Fresults_sparse.png)

# 3. Manejo de imágenes

## 3.1 Consejos para el manejo de imágenes

### Consejo 1: Es más fácil trabajar en escala de grises que a color

Una imagen es una composición de pixeles, a su vez dichos pixeles tienen un rango numérico de `0 a 255` siendo 0 negro y 255 blanco.
Las imágenes tienen un ancho y largo correspondiente al tamaño de la imagen. 

![m1.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm1.png)

La computadora lo que interpreta de una imagen es una matriz numerica. Esto es una gran noticia, puesto que los modelos de DL
trabajan directamente con número, lo cual significa que las imágenes por si mismas son un tipo de estructura de datos con
la que es fácilmente trabajable por Deep Learning. 

**¿Qué pasa con las imágenes a color?**

![m2.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm2.png)

En nuestro ejemplo de clasificación de ropa usando `fashion MNIST` las imágenes de ropa estaban en una escala de grises. Sin embargo,
la mayoría de imágenes están en 3 canales Red Green Blue (RGB) 

> Incluso algunas imágenes tienen 4 canales, las imágenes PNG pueden contar con un canal adicional para marcar la transparencia de una imagen.
> Este canal es conocido como `alpha` (RGBA).
> 

Las imágenes a color tienen las mismas dimensiones que las imágenes en escala de grises, sin embargo, estás al poseer 3 canales, cuentan
son su respectiva escala de color de 0 - 255 para cada color. Es la combinación de estas 3 escalas la que genera un color 
para cada pixel. Al sumar el color de la escala de Red con el Green con el Blue tenemos un color definido. Por ende el color
(0, 0, 0) sería negro, puesto que se están sumando 0 unidades de cada color, por otro el color (255, 0, 0) sería un Rojo puro,
puesto que se está sumando el máximo de la escala de Rojo con 0 unidades de las otras escalas.

![m3.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm3.png)

RGB NO es la única forma numerica que tenemos para representar el color, existen varios modelos de color que tienen diferentes
propiedades, por ejemplo una variable bastante utilizada al modelo RGB es el modelo HSV el cual tiene como parámetros:
`H: Hue, V: Value, S: Saturation` de forma simple de entender, la saturación va del blanco al color real, el value va del negro
al color real y finalmente el hue es el color en sí mismo.

![m4.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm4.png)

Para efector prácticos en DL seguiremos hablando de RGB como modelo de color base.

> Nota: En opencv la forma de leer las imágenes NO es por defecto RGB sino BGR.

Hablando computacionalmente, NO suele convenir trabajar las imágenes en color, puesto que aumenta la complejidad del modelo en 3,
puesto que el proceso de clasificación se debe hacer tomando en cuanta cada uno de los canales. En general si el problema
lo permite, lo mejor es convertir nuestra imagen de color a una escala de grises.

**¿Dónde si importa el color?**

Va a depender del tipo de problema que estemos resolviendo, pero tomemos en cuenta el siguiente ejemplo.

![m5.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm5.png)

Si estoy trabajando en un proceso de condición automática, necesito poder diferenciar el color de las marcas de la carretera
para saber si estoy dentro o fura del área permitida de conducción. En resumen si el problema intrínsecamente necesita del color
para poder clasificar, entonces vamos a tener que usar los 3 canales. 

### Consejo 2: Es la mejor idea manejar dimensiones definidas para las imágenes.

Si hemos entrenado una red neuronal para trabajar con imágenes de 255 x 255 en escala de grises, entonces todas las nuevas
imágenes que vaya a utilizar para clasificar deben seguir el mismo formato de imagen.

![m6.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm6.png)

Estás dimensiones y características de las imágenes deben ser compartidas entre las diferentes particiones del dataset y
las nuevas imágenes a clasificar.

### Consejo 3: Aumenta tus imágenes

Son los humanos los responsables de etiquetar manualmente cada una de las imágenes a clasificar, sin embargo, este proceso no
es escalable, es lento y puede ser costoso. No siempre se dispone de más imágenes completamente nuevas para entrenar al modelo.
Una técnica bastante utilizada es hacer `data augmentation` está es una técnica que implica la transformación de una imagen
base en una nueva imagen basada en rotaciones, traslaciones, zoom, cambios de brillo y contraste etc.

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

![m7.png](imgs%2Fmanejo%20de%20im%C3%A1genes%2Fm7.png)

De esta forma contaremos con más información para entrenamiento sin necesidad de obtener muestras adicionales, y también 
sin alargar los tiempos. Lo mejor es que si por ejemplo nuestra red se dedica a clasificar imágenes o detectar objetos, 
esta técnica conseguirá que el modelo sea capaz de obtener buenos resultados para imágenes tomadas desde distintos ángulos 
o bajo distintas condiciones de luz. Por tanto, conseguiremos que la red no sobreajuste y que generalice mejor.


## 3.2 Manejo de imágenes con Python

En este pequeño ejercicio vamos a ver un par de transformaciones a una imagen, utilizando `numpy` para modificar dicha imagen.

> ## Nota:
> El código de esta sección lo puedes encontrar [aquí](Manejo%20de%20imágenes/main.py)

**1: Importando bibliotecas necesarias**

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io  # Una alternativa podría ser opencv o pill
```

**2: Abrimos nuestra imagen de prueba**
```python
im = io.imread("imgs/perro.png")
print(im.shape)
print(im)
```
Respuesta esperada:
```commandline
(800, 600, 3)
[[[ 98 112  99]
  [103 117 104]
  [ 98 112  97]
  ...]]]
```
![perro.png](Manejo%20de%20im%C3%A1genes%2Fimgs%2Fperro.png)

**3: Transformaciones**

- Separemos cada canal:
  ```python
    red_channel = im[:, :, 0]
    green_channel = im[:, :, 1]
    blue_channel = im[:, :, 2]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(red_channel, cmap="gray")
    ax2.imshow(blue_channel, cmap="gray")
    ax3.imshow(green_channel, cmap="gray")
    ax1.set_title("Red channel")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title("Green channel")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_title("Blue channel")
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.savefig("imgs/channels.png")
    plt.close()
    ```
  ![channels.png](Manejo%20de%20im%C3%A1genes%2Fimgs%2Fchannels.png)

- Observemos el color de cada canal:
  ```python
    # este es una capa AUXILIAR llena de 0s nos permite completar los 3 canles
    aux_dim = np.zeros(im.shape[:2])
    # Este es rojo porque es (R Aux Aux)
    red = np.dstack((red_channel, aux_dim, aux_dim)).astype(np.uint8)
    green = np.dstack((aux_dim, green_channel, aux_dim)).astype(np.uint8)
    blue = np.dstack((aux_dim, aux_dim, blue_channel)).astype(np.uint8)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(red)
    ax2.imshow(green)
    ax3.imshow(blue)
    ax1.set_title("Red")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_title("Green")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_title("Blue")
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.savefig("imgs/channels2.png")
    plt.close()
  ```
    ![channels2.png](Manejo%20de%20im%C3%A1genes%2Fimgs%2Fchannels2.png)
- Zoom a la imagen en una zona específica:
  ```python
    plt.imshow(im[270:530, 150:400])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig("imgs/zoom.png")
    plt.close()
  ```
  ![zoom.png](Manejo%20de%20im%C3%A1genes%2Fimgs%2Fzoom.png)

# 4. Fundamentos de redes neuronales convolucionales

## 4.1 Kernel en redes neuronales

El Kernel en Computer Vision es una matriz o filtro utilizado para realizar operaciones de convolución en una imagen. 
La convolución es un proceso matemático que se usa para analizar una imagen y extraer características relevantes de ella. 
El kernel se aplica a cada píxel de la imagen y se multiplica por los valores de los píxeles en su vecindario. El resultado 
se suma y se coloca en la posición correspondiente en la imagen de salida.

![2D_Convolution_Animation.gif](imgs%2FFundamentos%20de%20CNNs%2F2D_Convolution_Animation.gif)

El tamaño del kernel y los valores de sus elementos son importantes para determinar el resultado de la convolución. Los 
kernels pueden ser diseñados para detectar características específicas en una imagen, como bordes, esquinas, texturas, etc. 
También se pueden utilizar para suavizar la imagen o para resaltar ciertas áreas de interés.

![giphy.gif](imgs%2FFundamentos%20de%20CNNs%2Fgiphy.gif)

En resumen, el Kernel en Computer Vision es una herramienta importante para procesar imágenes y extraer información útil 
de ellas.

Una página muy útil para observar el funcionamiento de los kernels es la siguiente:

https://setosa.io/ev/image-kernels/

La cual nos permitirá comparar una imagen de entrada, seleccionar un kernel y observar la imagen respuesta ya convolucionada.

![conv1.png](imgs%2FFundamentos%20de%20CNNs%2Fconv1.png)

## 4.2 El Kernel en acción

Creemos nuestro ejemplo de kernel horizontal y vertical, utilizando una imagen nuestra como base.

> ## Nota:
> El código completo lo puedes encontrar [aquí](Fundamentos%20de%20redes/kernel%20en%20acción/main.py)

**1: importando bibliotecas**
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage import io, color
```

**2: Abrimos nuestra imagen y la convertimos a gris**
```python
img = io.imread('input/gabriel.png')
print("Original's shape:", img.shape)
img_gray = color.rgb2gray(img)
print("Gray's shape:", img_gray.shape)
```
Respuesta esperada:
```commandline
Original's shape: (1365, 2048, 3)
Gray's shape: (1365, 2048)
```
Vemos como la imagen original tiene 3 canales RGB mientras que la imagen en escala de Grises no tiene una tercera dimensión.

**3: Creamos nuestros kernels vertical y horizontal**
```python
# Kernel para detectar bordes verticales

kernel_v = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])

# Kernel para detectar bordes horizontales

kernel_h = np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])
```
**4: Aplicamos la convolución a la imagen en escala de grises**
```python
img_kernel_v = nd.convolve(img_gray, kernel_v)
img_kernel_h = nd.convolve(img_gray, kernel_h)
```
**5: Graficamos los resultados**
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.imshow(img)
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(img_gray, cmap="gray")
ax2.set_title('Gray scale')
ax2.axis('off')
ax3.imshow(img_kernel_v, cmap="gray")
ax3.set_title('Vertical Kernel Conv')
ax3.axis('off')
ax4.imshow(img_kernel_h, cmap="gray")
ax4.set_title('Horizontal Kernel Conv')
ax4.axis('off')
plt.savefig("imgs/resultados.png")
```
Respuesta esperada:
![resultados.png](Fundamentos%20de%20redes%2Fkernel%20en%20acci%C3%B3n%2Fimgs%2Fresultados.png)

## 4.3 Padding y Strides

En una red neuronal convolucional, el padding y los strides son dos parámetros importantes que se utilizan para controlar 
la forma en que se realizan las convoluciones.

El padding se refiere a la adición de ceros alrededor de los bordes de la entrada antes de realizar la convolución. La 
razón por la que se hace esto es para evitar que el tamaño de la salida de la convolución se reduzca demasiado en 
comparación con el tamaño de la entrada original. El padding puede ser "valid" (sin padding) o "same" (con padding para 
que el tamaño de la salida sea el mismo que el de la entrada).

![padding1.png](imgs%2FFundamentos%20de%20CNNs%2Fpadding1.png)

Los strides, por otro lado, se refieren a la cantidad de píxeles que se desplazan entre cada convolución. Un stride de 1 
significa que la ventana de convolución se mueve un píxel a la vez, mientras que un stride de 2 significa que la ventana 
de convolución se mueve dos píxeles a la vez. Los strides más grandes resultan en una reducción en el tamaño de la salida, 
mientras que los strides más pequeños resultan en una salida más grande.

![strides.gif](imgs%2FFundamentos%20de%20CNNs%2Fstrides.gif)

**¿Cómo se implementan estas ideas en código?**

En keras es muy sencillo, simplemente cuando añadimos una `layer` de `Conv2D` debemos tener en cuenta los siguientes parámetros:

```python
model.add(Conv2D(filters=32, 
                 kernel_size=2, 
                 padding="same",  # Puede ser ["valid", "same"] "same" hace que el tamaño de la imagen de entrada sea igual al de salida
                                  # valid NO aplica padding lo cual produce una pequeña reducción en el tamaño de la imagen dependiendo el tamaño del kernel.
                 strids = (1, 1),  # El primer número es cuanto se desplaza a la derecha y el segundo es cuanto se desplaza verticalmente.
                 activation="relu"))
```

## 4.4 Capa de pooling

Una capa de pooling, también conocida como capa de submuestreo, es una capa común en una red neuronal convolucional (CNN). 
Esta capa se utiliza para reducir la dimensión espacial (alto x ancho) de la representación de características obtenida a 
través de las capas de convolución, lo que reduce la cantidad de parámetros y computación necesarios en la red.

La operación de pooling se realiza en cada canal de la salida de la capa de convolución (normalmente, el canal de salida 
es la imagen filtrada por diferentes kernels o filtros), aplicando una función de reducción que suele ser una operación de 
máxima o promedio. La operación de pooling se realiza con ventanas deslizantes de tamaño predefinido y con un stride 
determinado.

Por ejemplo, en una operación de pooling máxima (max pooling), la ventana de pooling de tamaño (2x2) se desliza por la 
salida de la capa de convolución con un stride de 2, y en cada ventana, se toma el valor máximo del mapa de características. 
Esto reduce la dimensión espacial a la mitad. El promedio de pooling (average pooling) funciona de manera similar, pero 
en lugar de tomar el valor máximo, se toma el promedio de los valores en cada ventana.

![maxpool.gif](imgs%2FFundamentos%20de%20CNNs%2Fmaxpool.gif)


En general, las capas de pooling tienen dos objetivos principales:

1. `Reducción de la dimensión espacial:` al reducir la dimensión espacial de la representación de características, se reduce 
la cantidad de parámetros y computación necesarios en la red, lo que hace que la red sea más eficiente.

2. `Introducción de invarianza de desplazamiento: el pooling puede hacer que la red sea invariante a pequeñas variaciones 
en la posición de las características, ya que la operación de pooling tomará el valor máximo o promedio de las características 
en una ventana de pooling, independientemente de su ubicación exacta en la imagen.

En resumen, las capas de pooling son una herramienta importante en las redes neuronales convolucionales, ya que ayudan a reducir la dimensión espacial de la representación de características y a introducir invarianza de desplazamiento, lo que puede mejorar el rendimiento de la red en tareas de clasificación de imágenes y otros problemas de visión por computadora.

**Ejemplos de Pooling**

Aquí podemos observar la diferencia entre una imagen original (al centro), a la izquierda la imagen después de max pooling 
y a la derecha después de Average pooling.

![pooling2.png](imgs%2FFundamentos%20de%20CNNs%2Fpooling2.png)

En el ejemplo del coche podemos ver como se ha reducido las dimensiones de la imagen, pero en general las características del coche
se mantienen presentes. 

**¿Qué tan eficiente es utilizar pooling?**

A continuación podemos observar dos arquitecturas muy similares, solo que a la izquierda se usa max_pooling y a la derecha no.

![pooling3.png](imgs%2FFundamentos%20de%20CNNs%2Fpooling3.png)

El resultado final es que en la arquitectura con max_pooling se entrenan `528,054` parámetros mientras que en la arquitectura
que no usa max_pooling la cantidad de parámetros se dispara a `32,784,054`. 

## 4.5 Arquitectura de redes convolucionales

La arquitectura general de una red neuronal convolucional CNN buscar apilar capas convolucionales, después de capas de 
reducción de dimensionalidad, max_pooling. Para finalmente utilizar una capa flatten que me permita volver la imagen a un 
vector 1d y poder ocupar capas densamente conectadas.

![arq1.png](imgs%2FFundamentos%20de%20CNNs%2Farq1.png)

Una vez llegados a este punto se pueden ocupar las mismas técnicas que hemos visto en cursos anteriores y ocupar los mismos principios.

El uso de kernels y número de filtros `filters` lo que hace es modificar la profundidad de mis capas covolucionales.

![arq2.png](imgs%2FFundamentos%20de%20CNNs%2Farq2.png)

Cada uno de los kernels estará buscando detectar algo específicamente, bordes, líneas, rayas texturas etc. A mayo cantidad de 
filtros mayor será la profundidad de la capa.

Por el contrario, las capas de max_pooling reducen el ancho y largo de las capas de entrada:
![arq3.png](imgs%2FFundamentos%20de%20CNNs%2Farq3.png)

Si funcionamos estás dos ideas, entonces agrupar capas convolucionales seguidas de capas de max_pooling genera la siguiente distribución:

![arq4.png](imgs%2FFundamentos%20de%20CNNs%2Farq4.png)

Cada salto me permite ir observando características más finas de la imagen, puesto que en cada paso la imagen se reduce de dimensiones, 
las primeras capas tienen filtros con características más generales, puesto que la imagen es muy amplia solo puedo observar 
algunos patrones como bordes horizontales o verticales, pero conforme se hace más profunda la red entonces puedo observar
detalles más específicos como lo serían por ejemplo las llantas de un coche o los ojos de un rostro.

Cuando ya he conseguido reducir considerablemente la dimensión de las matrices, entonces es el mejor momento para aplanar
la matriz y volverla un vector 1D y utilizar capas fully connected para proceder con la clasificación final de la imagen.

![arq5.png](imgs%2FFundamentos%20de%20CNNs%2Farq5.png)

En resumen, la arquitectura básica de una CNN es la siguiente:

![arq6.png](imgs%2FFundamentos%20de%20CNNs%2Farq6.png)

Sin embargo, esto no quiere decir que sea la única arquitectura que existe. En este momento te estarás preguntando ¿Y cómo elijo
cuantos filtros usar en cada capa? ¿Cuántas capas convolucionales debo elegir? ¿Y los strides y todos los demás parámetros?

La respuesta es que NO hay respuesta, esto es un fino arte de prueba y error, lo más común es empezar un nuevo proyecto con muy
pocas capas, observar los resultados y apoyarse de las gráficas para identificar el underfitting o el overfitting y con base en esto
decidir si se necesita un modelo más complejo (más capas, más filtros etc.) o si se necesita uno más simple que pueda generalizar
mejor (menos capas, o técnicas de normalizado de datos).

Finalmente, la siguiente página web nos permite construir imágenes que sirvan para explicar de forma simple la arquitectura
de nuestras CNNs: https://alexlenail.me/NN-SVG/LeNet.html

![arq7.png](imgs%2FFundamentos%20de%20CNNs%2Farq7.png)

## 4.6 Quizz Fundamentos de redes neuronales convolucionales

Para resolver este quizz es importante tener en cuenta las siguientes fórmulas y tomar el siguiente ejemplo de base:

![form1.png](imgs%2FFundamentos%20de%20CNNs%2Fform1.png)

Por ejemplo para una imagen con medidas iniciales de 28x28x1 que tiene 1 filtro de 3x3 , un padding de “same” , stride 1 y 16
canales de salida se tiene lo siguiente:

n eje x= (28+2(1) -3)/1 +1 =28

n eje y= (28+2(1) -3)/1 +1 =28

Siendo las dimensiones de salida 28x28x16

![q1.png](imgs%2FFundamentos%20de%20CNNs%2Fq1.png)

![q2.png](imgs%2FFundamentos%20de%20CNNs%2Fq2.png)

![q3.png](imgs%2FFundamentos%20de%20CNNs%2Fq3.png)

# 5. Resolviendo un problema de clasificación

A lo largo de esta sección repasaremos los conceptos de CNNs aprendidos hasta el momento, creando una red neuronal capas de
clasificar entre 10 diferentes tipos de clases. Para ello vamos a utilizar el dataset [CIFAR10](https://keras.io/api/datasets/cifar10/)
el cual consta de 50,000 imágenes de 32x32 píxeles a color, para el training set y 
10,000 imágenes con las mismas características para el conjunto de pruebas.

Las clases disponibles son:

| **Label** | **Description** |
|:---------:|-----------------|
|     0     | airplane        |
|     1     | automobile      |
|     2     | bird            |
|     3     | cat             |
|     4     | deer            |
|     5     | dog             |
|     6     | frog            |
|     7     | horse           |
|     8     | ship            |
|     9     | truck           |

> ## Nota:
> El código completo de esta sección lo puedes encontrar [aquí](Resolviendo%20un%20problema%20de%20clasificación/main.py)

## 5.1 Clasificación con redes neuronales convolucionales

**1: importamos bibliotecas necesarias:**
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
```
**Definimos nuestra función auxiliar para gráficar resultados:**
```python
def plot_results(history_, metric, fname):
    history_dict = history_.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    metric_values = history_dict[metric]
    val_metric_values = history_dict[f"val_{metric}"]
    epoch = range(1, len(loss_values) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
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
    plt.savefig(f"imgs/{fname}")
    plt.close()
```

**2: Descargamos el dataset CIFAR10**
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Análisis exploratorio
print("x_train shape:", x_train.shape)
plt.imshow(x_train[0])
plt.savefig("imgs/train0.jpg")
plt.close()
```
Respuesta esperada:
```commandline
x_train shape: (50000, 32, 32, 3)
```
![train0.jpg](Resolviendo%20un%20problema%20de%20clasificaci%C3%B3n%2Fimgs%2Ftrain0.jpg)

**3: Limpiamos nuestros datos**
Como ya sabemos, lo primero que debemos hacer es normalizar la escala de nuestros pixeles de 0-255 a 0-1
```python
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
```
También sabemos que aunque NO es la única forma de trabajar para clasificación de multiples clases, es una estrategia convertir
el label encoding en one hot encoding:
```python
num_clases = len(np.unique(y_train))
y_train = to_categorical(y_train, num_clases)
y_test = to_categorical(y_test, num_clases)
```

**4: Creando particiones de validación** 
```python
# Creando nuevas particiones de los datos
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]
print('x_train shape', x_train.shape)
print('x_train shape [0]', x_train[0].shape)

print('train:', x_train.shape[0])
print('val:', x_valid.shape[0])
print('test:', x_test.shape[0])
```
Respuesta esperada:
```commandline
x_train shape (45000, 32, 32, 3)
x_train shape [0] (32, 32, 3)
train: 45000
val: 5000
test: 10000
```
Podemos observar como ahora contamos con 3 conjuntos de datos, para el set de entrenamiento tenemos `45,000` muestras, 
de las cuales cada una de ellas tiene una forma de `(32, 32, 3)`. Para el conjunto de validación tenemos `5,000`muestras y
para el conjunto de pruebas `10,000`.

## 5.2 Creación de red convolucional para clasificación

> ## Nota:
> Algunos de los conceptos explicados en este punto son conceptos que YA FUERON explicados en el curso anterior.
> Si NO estás familiarizado con las técnicas de regularización puedes leer [Regularizadores de deep learning](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales#34-regularizaci%C3%B3n---dropout)
> 

**5: Definiendo la arquitectura del modelo**
Para este paso es importante recordar:
- Debemos tener presente el shape de las imágenes del dataset
- La cantidad de filtros aplicados en cada capa es arbitrario, pero generalmente conforme más profunda es la capa mayor cantidad de filtros
- La capa de `MaxPooling2D` sirve para reducir la complejidad del modelo, al ir reduciendo las dimensiones de la capa.
- Utilizar una capa de `DropOut` es una buena idea para reducir el posible overfitting que puede presentar el modelo.
- La arquitectura propuesta a continuación NO es una receta de cocina y NO tiene por qué ser la mejor para todos los escenarios,
es simplemente una propuesta que sigue la arquitectura básica de cualquier CNN vista en [Arquitectura de una CNN](#45-arquitectura-de-redes-convolucionales)

```python
def architecture(base_filtros: int, w_regularized: float, shape: tuple, num_classes: int):
    """
    Definiendo la arquitectura de nuestra CNN
    :param base_filtros: Número de filtros que tomara como base la CNN (capas posteriores usaran multiplos de este número)
    :param w_regularized: Peso para utilizar por el regularizador L2
    :param shape: forma del tensor de entrada (dimensiones de las imágenes de entrenamiento)
    :param num_classes: número de clases a clasificar por la CNN
    :return: 
    """
    model = Sequential()

    # Conv 1
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized), input_shape=shape))
    model.add(Activation("relu"))
    # Conv 2
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 3
    model.add(Conv2D(filters=2*base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    # Conv 4
    model.add(Conv2D(filters=2 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 5
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    # Conv 6
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # Capa de clasificación
    model.add(Dense(units=num_classes, activation="softmax"))
    print(model.summary())
    return model
```

## 5.3 Entrenamiento de un modelo de clasificación con redes convolucionales
**6: Creando el modelo**
```python
md = architecture(base_filtros=32, w_regularized=1e-4, shape=x_train[0].shape, num_classes=num_clases)
```
Respuesta esperada:
```commandline
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 32, 32, 32)        896       
                                                                 
 activation (Activation)     (None, 32, 32, 32)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 32, 32, 32)        9248      
                                                                 
 activation_1 (Activation)   (None, 32, 32, 32)        0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 16, 16, 32)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 16, 16, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 16, 16, 64)        18496     
                                                                 
 activation_2 (Activation)   (None, 16, 16, 64)        0         
                                                                 
 dropout_1 (Dropout)         (None, 16, 16, 64)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 16, 16, 64)        36928     
                                                                 
 activation_3 (Activation)   (None, 16, 16, 64)        0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 8, 8, 64)         0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 8, 8, 64)          0         
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 128)         73856     
                                                                 
 activation_4 (Activation)   (None, 8, 8, 128)         0         
                                                                 
 conv2d_5 (Conv2D)           (None, 8, 8, 128)         147584    
                                                                 
 activation_5 (Activation)   (None, 8, 8, 128)         0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 128)        0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 4, 4, 128)         0         
                                                                 
 flatten (Flatten)           (None, 2048)              0         
                                                                 
 dense (Dense)               (None, 10)                20490     
                                                                 
=================================================================
Total params: 307,498
Trainable params: 307,498
Non-trainable params: 0
_________________________________________________________________
```
**7: Compilamos el modelo**
```python
md.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
```
**8: Entrenamos el modelo**
```python
history = md.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_valid, y_valid), shuffle=True,)
```
Respuesta esperada:
```commandline
Epoch 1/50
352/352 [==============================] - 8s 14ms/step - loss: 1.9319 - acc: 0.3002 - val_loss: 1.7272 - val_acc: 0.3886
Epoch 2/50
352/352 [==============================] - 4s 12ms/step - loss: 1.5370 - acc: 0.4534 - val_loss: 1.3019 - val_acc: 0.5526
Epoch 3/50
352/352 [==============================] - 4s 12ms/step - loss: 1.3065 - acc: 0.5464 - val_loss: 1.1814 - val_acc: 0.5988
Epoch 4/50
352/352 [==============================] - 4s 12ms/step - loss: 1.1477 - acc: 0.6091 - val_loss: 1.0774 - val_acc: 0.6410
Epoch 5/50
352/352 [==============================] - 4s 12ms/step - loss: 1.0509 - acc: 0.6481 - val_loss: 1.0610 - val_acc: 0.6396
...
Epoch 45/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4845 - acc: 0.8835 - val_loss: 0.7058 - val_acc: 0.8308
Epoch 46/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4874 - acc: 0.8804 - val_loss: 0.6386 - val_acc: 0.8404
Epoch 47/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4870 - acc: 0.8824 - val_loss: 0.7363 - val_acc: 0.8226
Epoch 48/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4824 - acc: 0.8851 - val_loss: 0.7366 - val_acc: 0.8234
Epoch 49/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4830 - acc: 0.8838 - val_loss: 0.6489 - val_acc: 0.8466
Epoch 50/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4794 - acc: 0.8876 - val_loss: 0.6581 - val_acc: 0.8378
```
**9: Análisis de resultados**
```python
plot_results(history, "acc", "primer_resultado.png")
md.evaluate(x_test, y_test)
```
Resultados esperados:
![primer_resultado.png](Resolviendo%20un%20problema%20de%20clasificaci%C3%B3n%2Fimgs%2Fprimer_resultado.png)
```commandline
313/313 [==============================] - 1s 3ms/step - loss: 0.7079 - acc: 0.8265
```

Hemos tenido muy buenos resultados, sin embargo, más adelante veremos más técnicas que podemos utilizar con el objetivo de 
optimizar este resultado.

# 6.Optimización de una red neuronal convolucional

## 6.1 Data augmentation

La clasificación de imágenes es una tarea bastante complicada y laboriosa que require de mucho trabajo humano. Tomar las 
fotografías de las clases de interés y etiquetar cada una de ellas. Esto puede sonar sencillo, pero repetir este proceso
cientos o miles de veces puede ser bastante tardado y costoso. Una de las técnicas que podemos utilizar para simplificar
este proceso es conocida como `data augmentation`, está técnica ya la hemos mencionado anteriormente. Puedes volver a leer
[Consejo 3: aumenta tus imágenes](#consejo-3--aumenta-tus-imágenes), sin embargo; aquí te dejo un breve resumen:

Data Augmentation, o aumentación de datos, es una técnica comúnmente utilizada en el aprendizaje automático para aumentar 
la cantidad de datos de entrenamiento a partir de un conjunto de datos existente. La idea detrás de esta técnica es generar 
nuevas instancias de datos a partir de las que ya se tienen, de manera que la red neuronal tenga más datos para entrenarse 
y pueda mejorar su capacidad de generalización.

La aumentación de datos se puede realizar de varias formas, por ejemplo:

1. Rotación: se gira la imagen en un cierto ángulo.
2. Desplazamiento: se desplaza la imagen horizontal o verticalmente.
3. Aumento de tamaño: se aumenta el tamaño de la imagen.
4. Reducción de tamaño: se reduce el tamaño de la imagen.
5. Espejo: se crea una imagen espejo reflejando horizontalmente la imagen original.
6. Cambio de brillo, contraste, saturación, etc.
7. Recorte: se toma una parte de la imagen original.
8. Ruido: se agrega ruido a la imagen.

![dataA.png](imgs%2FFundamentos%20de%20CNNs%2FdataA.png)

La idea es que al aplicar estas transformaciones a las imágenes de entrada, se creen nuevas versiones de las mismas que 
permitan mejorar el rendimiento de la red neuronal en situaciones en las que se encuentre con imágenes similares, pero no 
idénticas a las que se encuentran en el conjunto de datos original.

La aumentación de datos es especialmente útil cuando se tiene un conjunto de datos pequeño, lo que puede llevar a problemas de 
sobreajuste (overfitting), es decir, que la red neuronal se adapte demasiado a los datos de entrenamiento y no sea capaz 
de generalizar bien a nuevos datos. Al aumentar el conjunto de datos, se puede reducir el riesgo de sobreajuste y mejorar 
el rendimiento de la red.

Para ver más ejemplos en acción: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


## 6.2 Aplicando data augmentation

En este pequeño ejercicio veremos una forma muy simple de aplicar data augmentation utilizando `keras`

> ## Nota:
> El código completo lo puedes encontrar [aquí](Optimización%20de%20una%20red%20neuronal/aplicando%20data%20augmentation/main.py)

**1: Importando bibliotecas necesarias**
```python
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
```

**2: Definiendo nuestro generador de imágenes**
```python
datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest',
                                 brightness_range=[0.4, 1.5]
                                 )
```
Toda la información sobre `ImageDataGenerator` la puedes encontrar [aquí](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

**3: Cargando una imagen de ejemplo**
```python
img = load_img('imgs/perro.png')
x = img_to_array(img)
print(x.shape)
x = x.reshape((1,) + x.shape)
print(x.shape)
```
Respuesta esperada:
```commandline
(800, 600, 3)
(1, 800, 600, 3)
```
![perro.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Faplicando%20data%20augmentation%2Fimgs%2Fperro.png)

**4: Empezamos a generar alteraciones de la imagen original**
> Nota:
> Como en este ejemplo solo tenemos una imagen, entonces el batch_size debe ser igual a 1.
> A pesar de que sea una sola imagen, `batch` es un arreglo de imágenes, por eso accedemos a cada
> imagen unitaria del batch de tamaño 1 como `batch[0]`
```python
for i, batch in enumerate(datagen.flow(x, batch_size=1)):
    plt.imshow(array_to_img(batch[0]))
    plt.savefig(f"imgs/transformation_{i}.png")
    plt.close()
    if i == 3:
        break
```

Respuesta esperada:


![transformation_0.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Faplicando%20data%20augmentation%2Fimgs%2Ftransformation_0.png)

![transformation_1.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Faplicando%20data%20augmentation%2Fimgs%2Ftransformation_1.png)

![transformation_2.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Faplicando%20data%20augmentation%2Fimgs%2Ftransformation_2.png)

![transformation_3.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Faplicando%20data%20augmentation%2Fimgs%2Ftransformation_3.png)

Adicionalmente, existe una forma de generar estás imágenes tomando como referencia un directorio:
```python
train_generator = datagen.flow_from_directory(
    '/train',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
    )
```
El diretorio debe tener la siguiente estructura:
```commandline
train/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

## 6.3 Callbacks: early stopping y checkpoints

> ## Nota:
> El código completo lo puedes encontrar [aquí](Optimización%20de%20una%20red%20neuronal/Callbacks/main.py)


Los `Callbacks` son una herramienta sumamente poderosa que nos permite monitorear el entrenamiento de nuestras redes neuronales
tienen varias funcionalidades, sin embargo, en este ejemplo hablaremos de 2 de las más utilizadas:

1. Early stopping: Detener el proceso de entrenamiento cuando este deja de optimizar
2. Checkpoints: Permite guardar el estado de un modelo mientras está siendo entrenado.

Para estos ejemplos usaremos la misma red que ya vimos en [Mi primera CNN](#2-mi-primera-red-neuronal-convolucional) que
utiliza `Fashion MNIST`.

**1: Empecemos por importar las bibliotecas que ya conocemos**
```python
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.utils import to_categorical
from keras import Sequential
```
Sin embargo, aquí vamos a añadir un par de nuevas bibliotecas que nos permitirán usar callbacks y cargar modelos guardados.
```python
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
```
**2: Creando arquitectura del modelo**
```python
def architecture(shape, name):
    model_ = Sequential(name=name)
    model_.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu", input_shape=shape))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model_.add(MaxPool2D(pool_size=2))
    model_.add(Dropout(0.3))
    model_.add(Flatten())
    model_.add(Dense(256, activation="relu"))
    model_.add(Dropout(0.5))
    model_.add(Dense(10, activation="softmax"))
    print(model_.summary())
    return model_
```
**3: Descargando el dataset y normalizando datos**
```python
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
train_labels_categorical = to_categorical(train_labels, 10)
test_labels_categorical = to_categorical(test_labels, 10)
```

**4: Creando Callbacks**

Esta es la parte interesante de esta nueva sección, vamos a crear `Callbacks` para monitorear el entrenamiento de nuestra red.


**Early stopping**

Empecemos por definir un callback de [Early Stopping](https://keras.io/api/callbacks/early_stopping/)
Te aconsejo leer toda la documentación para entender mejor su uso.
```
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)
```
Ejemplo:
```python
# Callback de Early Stopping
early_stopping_cb = EarlyStopping(monitor="val_accuracy", patience=1, verbose=1)
# Creando arquitectura del modelo
model = architecture(shape=train_images[0].shape, name="Early_Stopping")
# Compilando el modelo
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model.fit(train_images, train_labels_categorical, batch_size=256, epochs=10, validation_split=0.3,
          callbacks=[early_stopping_cb])
score = model.evaluate(test_images, test_labels_categorical)
print(score)
```
En este momento hemos definido un callback de early stopping, para efectos demostrativos estamos monitoreando el valor
de accuracy NO del dataset de entrenamiento sino del de validación. En otras palabras, le estamos pidiendo al modelo que se
detenga cuando NO haya logrado optimizar el accuracy del dataset de validación, tomando como restricción 1 única epoch. 
Esto es sumamente restrictivo, aquí solo está hecho para poder ejemplificar el caso, sin embargo, la `patience` debería ser mayor a 1
para evitar el ruido.

Respuesta esperada:
```commandline
Model: "Early_Stopping"
Epoch 1/10
165/165 [==============================] - 3s 8ms/step - loss: 0.6910 - accuracy: 0.7489 - val_loss: 0.7549 - val_accuracy: 0.7353
Epoch 2/10
165/165 [==============================] - 1s 7ms/step - loss: 0.4211 - accuracy: 0.8467 - val_loss: 0.5268 - val_accuracy: 0.8148
Epoch 3/10
165/165 [==============================] - 1s 7ms/step - loss: 0.3562 - accuracy: 0.8713 - val_loss: 0.4886 - val_accuracy: 0.8238
Epoch 4/10
165/165 [==============================] - 1s 7ms/step - loss: 0.3148 - accuracy: 0.8843 - val_loss: 0.3971 - val_accuracy: 0.8611
Epoch 5/10
165/165 [==============================] - 1s 7ms/step - loss: 0.2874 - accuracy: 0.8951 - val_loss: 0.5576 - val_accuracy: 0.8001
Epoch 5: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 0.5766 - accuracy: 0.7930
[0.5765808820724487, 0.7929999828338623]
```
Podemos ver que inmediatamente que él `val_accuracy` cayó el modelo dejo de ser entrenado. Esto en medida se debe al valor de
`batch_size` que elegimos, un valor más pequeño es más lento de entrenar, aumenta en menor velocidad el accuracy, pero suele ser más
estable. De cualquier forma, aquí SOLO buscabamos demostrar el comportamiento de `early stopping` no optimizar la red.

**Checkpoint**

Este es un callback de extremada utilidad, pues permite ir guardando en cada epoch o n epoch el modelo que mejor optimice una feature.
Te aconsejo ampliamente leer [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/).
```
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor: str = "val_loss",
    verbose: int = 0,
    save_best_only: bool = False,
    save_weights_only: bool = False,
    mode: str = "auto",
    save_freq="epoch",
    options=None,
    initial_value_threshold=None,
    **kwargs
)
```
Ejemplo:

Creamos nuestro callback de checkpoint, le decimos que vamos a guardar el modelo entero (incluyendo, arquitecura, pesos etc)
vamos a monitorear el `accuracy` del modelo, vamos a guardar el mejor modelo que encontremos y el modelo será guardado en
`models/best_model.h5`

```python
# Callback de Checkpoint
checkpoint_cb = ModelCheckpoint(filepath="models/best_model.h5", save_weights_only=False, monitor="accuracy",
                                mode="max",
                                save_best_only=True, verbose=1)
# Creando arquitectura del modelo
model = architecture(shape=train_images[0].shape, name="Checkpoint")
# Compilando el modelo
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(train_images, train_labels_categorical, batch_size=64, epochs=10, validation_split=0.3,
          callbacks=[checkpoint_cb])
```
Respuesta esperada:
```commandline
Model: "Checkpoint"
Epoch 1/10
655/657 [============================>.] - ETA: 0s - loss: 0.4926 - accuracy: 0.8198
Epoch 1: accuracy improved from -inf to 0.81990, saving model to models/best_model.h5
657/657 [==============================] - 3s 4ms/step - loss: 0.4924 - accuracy: 0.8199 - val_loss: 0.4184 - val_accuracy: 0.8423
Epoch 2/10
653/657 [============================>.] - ETA: 0s - loss: 0.3170 - accuracy: 0.8853
Epoch 2: accuracy improved from 0.81990 to 0.88510, saving model to models/best_model.h5
657/657 [==============================] - 3s 4ms/step - loss: 0.3172 - accuracy: 0.8851 - val_loss: 0.3127 - val_accuracy: 0.8860
Epoch 3/10
652/657 [============================>.] - ETA: 0s - loss: 0.2682 - accuracy: 0.9015
Epoch 3: accuracy improved from 0.88510 to 0.90143, saving model to models/best_model.h5
657/657 [==============================] - 3s 4ms/step - loss: 0.2681 - accuracy: 0.9014 - val_loss: 0.2737 - val_accuracy: 0.9013
Epoch 4/10
655/657 [============================>.] - ETA: 0s - loss: 0.2375 - accuracy: 0.9126
Epoch 4: accuracy improved from 0.90143 to 0.91271, saving model to models/best_model.h5
657/657 [==============================] - 3s 4ms/step - loss: 0.2374 - accuracy: 0.9127 - val_loss: 0.2665 - val_accuracy: 0.9021
Epoch 5/10
647/657 [============================>.] - ETA: 0s - loss: 0.2119 - accuracy: 0.9217
Epoch 5: accuracy improved from 0.91271 to 0.92181, saving model to models/best_model.h5
657/657 [==============================] - 3s 4ms/step - loss: 0.2117 - accuracy: 0.9218 - val_loss: 0.2613 - val_accuracy: 0.9092
Epoch 6/10
653/657 [============================>.] - ETA: 0s - loss: 0.1920 - accuracy: 0.9274
Epoch 6: accuracy improved from 0.92181 to 0.92757, saving model to models/best_model.h5
657/657 [==============================] - 3s 5ms/step - loss: 0.1918 - accuracy: 0.9276 - val_loss: 0.2802 - val_accuracy: 0.8978
Epoch 7/10
653/657 [============================>.] - ETA: 0s - loss: 0.1750 - accuracy: 0.9362
Epoch 7: accuracy improved from 0.92757 to 0.93617, saving model to models/best_model.h5
657/657 [==============================] - 2s 4ms/step - loss: 0.1749 - accuracy: 0.9362 - val_loss: 0.2596 - val_accuracy: 0.9116
Epoch 8/10
657/657 [==============================] - ETA: 0s - loss: 0.1582 - accuracy: 0.9416
Epoch 8: accuracy improved from 0.93617 to 0.94160, saving model to models/best_model.h5
657/657 [==============================] - 2s 3ms/step - loss: 0.1582 - accuracy: 0.9416 - val_loss: 0.2814 - val_accuracy: 0.9027
Epoch 9/10
650/657 [============================>.] - ETA: 0s - loss: 0.1426 - accuracy: 0.9478
Epoch 9: accuracy improved from 0.94160 to 0.94788, saving model to models/best_model.h5
657/657 [==============================] - 2s 4ms/step - loss: 0.1424 - accuracy: 0.9479 - val_loss: 0.2621 - val_accuracy: 0.9139
Epoch 10/10
645/657 [============================>.] - ETA: 0s - loss: 0.1294 - accuracy: 0.9522
Epoch 10: accuracy improved from 0.94788 to 0.95198, saving model to models/best_model.h5
657/657 [==============================] - 2s 4ms/step - loss: 0.1298 - accuracy: 0.9520 - val_loss: 0.2493 - val_accuracy: 0.9179
```
Ahora veamos los resultados con el dataset de prueba:
```python
# Podemos asegurarnos de que el modelo se guardo correctamente, si lo cargamos y comparamos con el modelo que ya tenemos listo
loaded_model = load_model("models/best_model.h5")
print("model:", model.evaluate(test_images, test_labels_categorical))
print("loaded model:", loaded_model.evaluate(test_images, test_labels_categorical))
```
Respuesta esperada:
```commandline
313/313 [==============================] - 0s 1ms/step - loss: 0.2766 - accuracy: 0.9152
model: [0.2766445279121399, 0.9151999950408936]
313/313 [==============================] - 0s 1ms/step - loss: 0.2766 - accuracy: 0.9152
loaded model: [0.2766445279121399, 0.9151999950408936]
```
Excelente, vemos como ambos tienen el mismo resultado, puesto que se trata de exactamente el mismo modelo. Con esto podemos
concluir que el proceso de guardado del modelo por el callback ha sido completamente exitoso.

## 6.4 Batch normalization

La normalización es un procedimiento llevado a cabo sobre un conjunto de datos que busca estandarizar sus valores, reducir 
la cantidad de números que lo compone y en una escala homogénea de datos ayudando al descenso del gradiente `para converger mucho más rápido`
cuando se ejecuta el `Backpropagation`.

- Valores pequeños: típicamente entre 0 a 1
- Data homogénea: todos los píxeles tienen datos en el mismo rango.


![norma1.png](imgs%2FOptimizaci%C3%B3n%20de%20una%20red%2Fnorma1.png)

También, se puede dentro de las capas ocultas de la red neuronal llevar a cabo una normalización para así estandarizarla, 
a este proceso se le conoce como: `Batch Normalization.

Batch Normalization (normalización por lotes) es una técnica utilizada en el aprendizaje profundo para mejorar la estabilidad 
y eficacia del entrenamiento de redes neuronales. La técnica consiste en normalizar las activaciones de cada capa en una red 
neuronal para que tengan una media cercana a cero y una varianza cercana a uno.

![norma2.png](imgs%2FOptimizaci%C3%B3n%20de%20una%20red%2Fnorma2.png)

La normalización por lotes se realiza sobre un lote (batch) de muestras de entrenamiento en cada iteración del proceso de 
entrenamiento. Para normalizar las activaciones, se calcula la media y la desviación estándar del batch y se utiliza esta 
información para escalar y centrar las activaciones.

El efecto de la normalización por lotes es que la red neuronal se vuelve menos sensible a los valores iniciales de los 
pesos y sesgos de la red, lo que a su vez reduce la necesidad de ajustar constantemente la tasa de aprendizaje durante el 
entrenamiento. Además, la normalización por lotes reduce la propagación de los gradientes a través de la red, lo que permite 
el uso de tasas de aprendizaje más altas sin que se produzca inestabilidad en el entrenamiento.

![norma3.png](imgs%2FOptimizaci%C3%B3n%20de%20una%20red%2Fnorma3.png)

La normalización por lotes también puede ayudar a prevenir el sobreajuste (overfitting) y mejorar la capacidad de generalización 
de la red. Al estabilizar las activaciones de cada capa, se reduce la necesidad de regularización y se mejora el rendimiento 
en conjuntos de datos nuevos.

**Implementación en Keras**

En Keras es tan sencillo como agregar una capa al modelo:
```python
model.add(BatchNormalization())
```
Esta capa suele ser agregada justamente después de la salida de la capa de activación, pues lo que nos intereza es normalizar estos datos.
```python
model.add(Activation("relu"))
model.add(BatchNormalization())
```

## 6.5 Optimización de modelo de clasificación

En estas siguientes clases vamos a estar optimizando los resultados obtenidos en [Resolviendo un problema de clasificación](#5-resolviendo-un-problema-de-clasificación)

Recordemos que los últimos resultados obtenidos fueron:

```commandline
Entrenamiento:
Epoch 50/50
352/352 [==============================] - 4s 12ms/step - loss: 0.4794 - acc: 0.8876 - val_loss: 0.6581 - val_acc: 0.8378
Test set:
313/313 [==============================] - 1s 3ms/step - loss: 0.7079 - acc: 0.8265
```
Veamos si logramos aumentar un poco estos resultados.

> ## Nota:
> El código completo de esta sección lo puedes encontrar [aquí](Optimización%20de%20una%20red%20neuronal/optmización%20del%20modelo/main.py)

**1: Importando Bibliotecas**
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
```
Nuevas bibliotecas de este ejemplo
```python
# Nuevas bibliotecas
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from os.path import exists
```

**2: Cargando datos y creando particiones**
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Análisis exploratorio
print("x_train shape:", x_train.shape)
# Limpieza de datos

num_clases = len(np.unique(y_train))
y_train = to_categorical(y_train, num_clases)
y_test = to_categorical(y_test, num_clases)

# Creando nuevas particiones de los datos
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]
print('x_train shape', x_train.shape)
print('x_train shape [0]', x_train[0].shape)

print('train:', x_train.shape[0])
print('val:', x_valid.shape[0])
print('test:', x_test.shape[0])
```

**3: Normalizando datos**

Aquí es MUY IMPORTANTE tener en cuenta que estamos normalizando TODAS las particiones con los datos del dataset de entrenamiento.
Esto porque NOSOTROS NO debemos tener ningún tipo de información de los datos de validación o prueba. Esto puede afectar los
resultados finales, pero reflejan mucho mejor la realidad de predicción del modelo.

```python
# Normalizado de datos
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')

mean_train = np.mean(x_train)
std_train = np.std(x_train)

x_train = (x_train - mean_train) / (std_train + 1e-7)
x_test = (x_test - mean_train) / (std_train + 1e-7)
x_valid = (x_valid - mean_train) / (std_train + 1e-7)
```
Respuesta esperada:
```commandline
x_train shape: (50000, 32, 32, 3)
x_train shape (45000, 32, 32, 3)
x_train shape [0] (32, 32, 3)
train: 45000
val: 5000
test: 10000
```

**4: Creamos nuestro aumentador de datos**
```python
# Data Augmentation
datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)
```

**5: Creando nuestro callback de monitoreo de val accuracy**
```python
# Checkpoint Callback
checkpoint_cb = ModelCheckpoint("models/best_model.h5", verbose=1, save_best_only=True, monitor="val_acc")    
```

**DEFINIMOS la arquitectura de nuestra CNN**

La principal diferencia respecto a la versión original es que aquí estamos añadiendo `BatchNormalization()` a la salida de cada
capa de `Activation()`

```python
def architecture(base_filtros: int, w_regularized: float, shape: tuple, num_classes: int):
    """
    Definiendo la arquitectura de nuestra CNN
    :param base_filtros: Número de filtros que tomara como base la CNN (capas posteriores usaran multiplos de este número)
    :param w_regularized: Peso para utilizar por el regularizador L2
    :param shape: forma del tensor de entrada (dimensiones de las imágenes de entrenamiento)
    :param num_classes: número de clases a clasificar por la CNN
    :return:
    """
    model = Sequential()

    # Conv 1
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized), input_shape=shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    # Conv 2
    model.add(Conv2D(filters=base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 3
    model.add(Conv2D(filters=2 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Conv 4
    model.add(Conv2D(filters=2 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # Conv 5
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    # Conv 6
    model.add(Conv2D(filters=4 * base_filtros, kernel_size=(3, 3), padding="same",
                     kernel_regularizer=regularizers.l2(w_regularized)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Flatten
    model.add(Flatten())
    # Capa de clasificación
    model.add(Dense(units=num_classes, activation="softmax"))
    print(model.summary())
    return model
```

## 6.6 Entrenamiento de nuestro modelo de clasificación optimizado

Aquí vamos a agregar una pequeña línea que nos permitirá entrenar el modelo por primera vez, o en caso de que ya éxista 
el modelo, lo puede cargar para continuar con su proceso de entrenamiento posterior.

```python
if not exists("models/best_model.h5"):
```

**6: Creando el modelo por primera vez**
```python
print("Entrenando por primera vez")
print("*" * 64)
print("Creando arquitectura")
md = architecture(base_filtros=32, w_regularized=1e-4, shape=x_train[0].shape, num_classes=num_clases)
```
Respuesta esperada:
```commandline
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 32, 32, 32)        896       
                                                                 
 activation (Activation)     (None, 32, 32, 32)        0         
                                                                 
 batch_normalization (BatchN  (None, 32, 32, 32)       128       
 ormalization)                                                   
                                                                 
 conv2d_1 (Conv2D)           (None, 32, 32, 32)        9248      
                                                                 
 activation_1 (Activation)   (None, 32, 32, 32)        0         
                                                                 
 batch_normalization_1 (Batc  (None, 32, 32, 32)       128       
 hNormalization)                                                 
                                                                 
 max_pooling2d (MaxPooling2D  (None, 16, 16, 32)       0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 16, 16, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 16, 16, 64)        18496     
                                                                 
 activation_2 (Activation)   (None, 16, 16, 64)        0         
                                                                 
 batch_normalization_2 (Batc  (None, 16, 16, 64)       256       
 hNormalization)                                                 
                                                                 
 dropout_1 (Dropout)         (None, 16, 16, 64)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 16, 16, 64)        36928     
                                                                 
 activation_3 (Activation)   (None, 16, 16, 64)        0         
                                                                 
 batch_normalization_3 (Batc  (None, 16, 16, 64)       256       
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 8, 8, 64)         0         
 2D)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 8, 8, 64)          0         
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 128)         73856     
                                                                 
 activation_4 (Activation)   (None, 8, 8, 128)         0         
                                                                 
 batch_normalization_4 (Batc  (None, 8, 8, 128)        512       
 hNormalization)                                                 
                                                                 
 conv2d_5 (Conv2D)           (None, 8, 8, 128)         147584    
                                                                 
 activation_5 (Activation)   (None, 8, 8, 128)         0         
                                                                 
 batch_normalization_5 (Batc  (None, 8, 8, 128)        512       
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 128)        0         
 2D)                                                             
                                                                 
 dropout_3 (Dropout)         (None, 4, 4, 128)         0         
                                                                 
 flatten (Flatten)           (None, 2048)              0         
                                                                 
 dense (Dense)               (None, 10)                20490     
                                                                 
=================================================================
Total params: 309,290
Trainable params: 308,394
Non-trainable params: 896
_________________________________________________________________
```
**7: Compilando y entrenando el modelo**

Como pequeña variante al código original aquí vamos a utilizar `Adam()` como optimizador en lugar del que siempre hemos usado
`rmsprop`

```python
print("Empezando a entrenar")
md.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["acc"])
history = md.fit(datagen.flow(x_train, y_train, batch_size=128),
                 callbacks=[checkpoint_cb],
                 steps_per_epoch=x_train.shape[0] // 128,
                 epochs=50,
                 verbose=2,
                 validation_data=(x_valid, y_valid))
plot_results(history, "acc", "resultado_tuneado.png")
```
Resultado esperado:
![resultado_tuneado.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Foptmizaci%C3%B3n%20del%20modelo%2Fimgs%2Fresultado_tuneado.png)

Podemos comparar estos resultados con el primer modelo que hicimos:
![primer_resultado.png](Resolviendo%20un%20problema%20de%20clasificaci%C3%B3n%2Fimgs%2Fprimer_resultado.png)

Podemos observar que en el mismo número de iteraciones, en general el primer modelo obtiene mejores resultados, pero observamos
una mayor diferencia entre los resultados de training y validation, indicando que el primer modelo es más cercano a presentar
overfitting en contraste al segundo modelo que a pesar de no presentar una mejora circunstancial, sí ha mostrado un comportamiento
más cercano al set de entrenamiento.

Ahora como ya tenemos un modelo guardado, podemos cargarlo y seguir entrenando un par de iteraciones más:

**8: Retomando entrenamiento**
```python
print("Abriendo modelo y continuando entrenamiento")
md = load_model("models/best_model.h5")
history = md.fit(datagen.flow(x_train, y_train, batch_size=128),
                 callbacks=[checkpoint_cb],
                 steps_per_epoch=x_train.shape[0] // 128,
                 epochs=20,
                 verbose=2,
                 validation_data=(x_valid, y_valid))

plot_results(history, "acc", "resultado_tuneado_2.png")
```
Resultado esperado:
```commandline
Epoch 1: val_acc improved from -inf to 0.80880, saving model to models/best_model.h5
351/351 - 10s - loss: 0.7392 - acc: 0.7879 - val_loss: 0.7140 - val_acc: 0.8088 - 10s/epoch - 29ms/step
Epoch 2/20

Epoch 2: val_acc did not improve from 0.80880
351/351 - 8s - loss: 0.7438 - acc: 0.7859 - val_loss: 0.7734 - val_acc: 0.7806 - 8s/epoch - 23ms/step
Epoch 3/20

...

Epoch 15: val_acc improved from 0.81900 to 0.82280, saving model to models/best_model.h5
351/351 - 9s - loss: 0.7210 - acc: 0.7967 - val_loss: 0.6565 - val_acc: 0.8228 - 9s/epoch - 24ms/step
Epoch 16/20

Epoch 16: val_acc did not improve from 0.82280
351/351 - 9s - loss: 0.7226 - acc: 0.7970 - val_loss: 0.7217 - val_acc: 0.8076 - 9s/epoch - 24ms/step
Epoch 17/20
```
![resultado_tuneado_2.png](Optimizaci%C3%B3n%20de%20una%20red%20neuronal%2Foptmizaci%C3%B3n%20del%20modelo%2Fimgs%2Fresultado_tuneado_2.png)

**9: Resultados finales:**
```python
acc = md.evaluate(x_test, y_test)
print("normal acc: ", acc)
best_model = load_model("models/best_model.h5")
acc = best_model.evaluate(x_test, y_test)
print("loaded model acc:", acc)
```
Respuesta esperada:
```commandline
313/313 [==============================] - 1s 3ms/step - loss: 0.7037 - acc: 0.8105
normal acc:  [0.7037441730499268, 0.8105000257492065]
313/313 [==============================] - 1s 3ms/step - loss: 0.7040 - acc: 0.8095
loaded model acc: [0.7039660215377808, 0.809499979019165]
```

### Conclusiones

Este mini proyecto nos ha servido para condensar MUCHÍSIMOS temas vistos hasta el momento:

- Data generation
- Data normalization
- Callbacks
- Arquitectura de una CNN
- Guardado de CNN models
- Restableciendo una CNN para seguir entrenando.

El primer modelo obtenido parece haber tenido un mejor resultado, esto nos demuestra que no hay una fórmula secreta para 
mejorar los resultados obtenidos y que a veces menos es más. Sin embargo, lo más importante es tener conocer una metodología
correcta y profunda de diferentes herramientas de deep learning. Es nuestra responsabilidad como científicos de datos encontrar
diferentes combinaciones más complejas o sencillas que nos permitan optimizar nuestros problemas de clasificación.

## 6.7 Quizz: Optimización de redes neuronales convolucionales

![q1.png](imgs%2FOptimizaci%C3%B3n%20de%20una%20red%2Fq1.png)

![q2.png](imgs%2FOptimizaci%C3%B3n%20de%20una%20red%2Fq2.png)

![q3.png](imgs%2FOptimizaci%C3%B3n%20de%20una%20red%2Fq3.png)


# 7. Resolviendo una competencia de Kaggle

En este mini proyecto haremos una CNN muy sencilla para clasificar entre perros y gatos, sin embargo, la principal diferencia
respecto a todos los otros ejemplos es que en esta ocasión NO partiremos de un dataset de Keras, sino que tendremos a nuestra
disposición un dataset de imágenes acomodadas en carpetas con la siguiente distribución:

> ## Nota:
> Puedes encontrar el código completo [aquí](Competencia%20Kaggle/main.py)

```
data
|----test
|----|----cats
|----|----|----cat.1500.jpg
|----|----dogs
|----|----|----dog.1500.jpg
|----train
|----|----cats
|----|----|----cat.0.jpg
|----|----dogs
|----|----|----dog.0.jpg
|----validation
|----|----cats
|----|----|----cat.1000.jpg
|----|----dogs
|----|----|----dog.1000.jpg
```

El dataset lo puedes encontrar en el siguiente repositorio: https://www.kaggle.com/datasets/alarcon7a/cnn-data-sources?select=cats_and_dogs

El dataset contiene la siguiente distribución de datos:

- Train: 2000 imágenes, 1000 gatos - 1000 perros
- Test: 1000 imágenes, 500 gatos - 500 perros
- Validation: 1000 imágenes, 500 gatos - 500 perros

El nombre de las carpetas indica explícitamente en nombre de la etiqueta a la que pertenece la imagen. Sin embargo, cada
imagen puede tener dimensiones diferentes, y estas no están normalizadas de ninguna manera. Nuestro trabajo será entonces
convertir esta carpeta en un dataset de entrenamiento, validación y prueba para nuestra CNN. Para ello usaremos el método
`datagen.flow_from_directory` del cual hablamos brevemente en [Aplicando data augmentation](#62-aplicando-data-augmentation), 
pero en este ejemplo lo usaremos concretamente.


## 7.1 Clasificando entre perros y gatos

**1: Importando bibliotecas**

Como es costumbre, empecemos importando las bibliotecas necesarias.
```python
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
```

**2: Definamos una arquitectura sencilla**

Para este ejemplo, no nos centraremos en crear un modelo óptimo, sino en mostrar el flujo de información y como generar el dataset
desde las imágenes de entrada distribuidas en diferentes carpetas.
```python
def architecture(n_filters, input_shape):
    model = Sequential()

    model.add(Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=2*n_filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=4 * n_filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=4 * n_filters, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    
    return model
```

**3: Creación de Image Data Generators**

Para este particular problema usaremos `Image Data Generators` porque tenemos muy pocos datos de entrenamiento para nuestro
problema, apenas contamos con `2,000` ejemplos `1,000` de cada clase.

Para el generador de entrenamiento tendremos la siguiente configuración:
```python
train_dategen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
```
Como podemos observar, podemos permitir varias alteraciones a las imágenes originales. Pero el primer punto de rescale es que
va a dividir toda la imagen entre 255, una normalización básica pero bastante útil.

Para el generador de test lo único que haremos será utilizar él `rescale` para normalizar los datos, pero este objeto no será
útil más adelante para cargar las imágenes de validation y test.

```python
test_datagen = ImageDataGenerator(rescale=1./255)
```

**4: Definimos el batch_size con el cual van a trabajar los generadores de imágenes**

Este valor también será utilizado más adelante para definir `steps_per_epoch` y `validation_steps`
```python
bs = 32
```

**5: Creando los datasets de entrenamiento, validación y pruebas**

Como podemos apreciar, nuestra data de entrenamiento éxiste en el `train_generator` que a su vez se crea por el objeto 
`train_datage` que definimos anteriormente, adicionalmente, usamos el método `flow_from_directory` para decirle que las imáquenes
que va a aumentar provienen del directorio `data/train`, adicionalmente, le pedimos que re-escale todas las imágenes a una 
resolución de (150x150) píxeles, y que ocupe el `batch_size` que definimos anteriormente como `bs`, finalmente, este directorio
solo cuenta con 2 clases, es por ello que elegimos `class_mode` como `binary`.

Puedes leer más del uso de `ImageDataGenerator` [aquí](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

```python
train_generator = train_dategen.flow_from_directory(directory="data/train", target_size=(150, 150), batch_size=bs,
                                                    class_mode="binary")

validation_generator = test_datagen.flow_from_directory(directory="data/validation", target_size=(150, 150),
                                                        batch_size=bs, class_mode="binary")

test_generator = test_datagen.flow_from_directory(directory="data/test", target_size=(150, 150),
                                                  batch_size=bs, class_mode="binary")
```
Respuesta esperada:
```commandline
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
```
Finalmente, podemos observar como los datasets de `validation` y `test` no tienen ninguna transformación a los datos más que
la normalización de dividir entre 255.

## 7.2 Entrenamiento del modelo de clasificación de perros y gatos

Antes de entrenar el modelo, es buena idea crear un callback de tipo `checkpoint` para siempre tener guardado el `best_model`

```python
checkpoint_cb = ModelCheckpoint("models/cats_vs_dogs.h5", verbose=1, save_best_only=True, monitor="val_acc")
```

**6: Creando y compilando el modelo**

```python
md = architecture(n_filters=32, input_shape=(150, 150, 3))
md.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["acc"])
```
Respuesta esperada:
```commandline
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 74, 74, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 6272)              0         
                                                                 
 dropout (Dropout)           (None, 6272)              0         
                                                                 
 dense (Dense)               (None, 512)               3211776   
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
```
**7: Entrenamos el modelo**
```python
history = md.fit(train_generator, steps_per_epoch=2000//bs, epochs=100, validation_data=validation_generator,
                 validation_steps=1000//bs, callbacks=[checkpoint_cb], batch_size=64)
```
Respuesta esperada:
```commandline
Epoch 1/100
62/62 [==============================] - ETA: 0s - loss: 0.6966 - acc: 0.4939
Epoch 1: val_acc improved from -inf to 0.50000, saving model to models/cats_vs_dogs.h5
62/62 [==============================] - 7s 93ms/step - loss: 0.6966 - acc: 0.4939 - val_loss: 0.6931 - val_acc: 0.5000
Epoch 2/100
62/62 [==============================] - ETA: 0s - loss: 0.6921 - acc: 0.5229
Epoch 2: val_acc improved from 0.50000 to 0.53226, saving model to models/cats_vs_dogs.h5
...
Epoch 99: val_acc did not improve from 0.81653
62/62 [==============================] - 5s 88ms/step - loss: 0.3666 - acc: 0.8359 - val_loss: 0.4386 - val_acc: 0.8075
Epoch 100/100
62/62 [==============================] - ETA: 0s - loss: 0.3402 - acc: 0.8471
Epoch 100: val_acc did not improve from 0.81653
62/62 [==============================] - 5s 88ms/step - loss: 0.3402 - acc: 0.8471 - val_loss: 0.4600 - val_acc: 0.8075
```

**8: Análisis de resultados**

```python
plot_results(history, "acc", "primeros_resultados.png")

best_model = load_model("models/cats_vs_dogs.h5")

acc = best_model.evaluate(test_generator)
print(acc)
```
![primeros_resultados.png](Competencia%20Kaggle%2Fimgs%2Fprimeros_resultados.png)

```commandline
32/32 [==============================] - 1s 22ms/step - loss: 0.4797 - acc: 0.7920
[0.4797414541244507, 0.7919999957084656]
```

### Conclusiones:

Podemos observar como con muy pocas líneas de código y poca optimización y teniendo un modelo simple e intuitivo hemos logrado
obtener aproximadamente un 80% de precisión al momento de clasificar entre perros y gatos. Sin embargo, nuestras gráficas de
resultados muestran un ligero overfitting, el cual podríamos mejorar con algunas técnicas de normalización que ya hemos visto 
a lo largo del curso.

Este pequeño proyecto nos ha permitido tener el conocimiento de como entrenar a nuestros modelos de CNNs directamente desde
datasets de imágenes obtenidas y clasificadas directamente por nosotros, esto nos brida mucha libertad, puesto que en problemas
anteriores siempre habíamos utilizado datasets ya existentes de keras, los cuales ya estaban relativamente limpios y listos para su uso.

> Nota: sería interesante hacer un problema con más de 2 clases, para ver como cambia el uso de `flow_from_directory`
> 
> Nota 2: La documentación de keras contiene el siguiente ejemplo:
>
> class_mode: One of "categorical", "binary", "sparse",
>                "input", or None. Default: "categorical".
>
>                Determines the type of label arrays that are returned:
>                - "categorical" will be 2D one-hot encoded labels,
>                - "binary" will be 1D binary labels,
>                    "sparse" will be 1D integer labels,
>                - "input" will be images identical
>                    to input images (mainly used to work with autoencoders).
>                - If None, no labels are returned
>                  (the generator will only yield batches of image data,
>                  which is useful to use with `model.predict_generator()`).
>                  Please note that in case of class_mode None,
>                  the data still needs to reside in a subdirectory
>                  of `directory` for it to work correctly.
>
# 8. Cierre

Muchas Felicidades si has llegado hasta aquí has terminado exitosamente el curso de [Redes Neuronales Convolucionales con Python y Keras](https://platzi.com/cursos/redes-neuronales-convolucionales/)
El curso es impartido por el profesor [Carlos Alarcón](https://platzi.com/profes/alarcon7a/) Y este repositorio me pertenece a mí:
[Gabriel Ichcanziho](https://www.linkedin.com/in/ichcanziho/).

Este curso nos ha permitido conocer los temas de:

- Clasificación de imágenes: Binary Output - Multiple Output
- Arquitectura de redes CNNs
- Kernel y convoluciones
- Padding y Strides
- Pooling
- Regularization: Data Augmentation, Batch normalization, Dropout
- Callbacks: Checkpoint, Early Stopping

Aún queda mucho por aprender:

## 8.1 Siguientes pasos

### Arquitecturas más complejas de CNN:

**Alexnet**

![A1.png](imgs%2FCierre%2FA1.png)

**VGGNet**

![A2.png](imgs%2FCierre%2FA2.png)

**LeNet**

![A3.png](imgs%2FCierre%2FA3.png)

**ResNet**

![resnet.png](imgs%2FCierre%2Fresnet.png)


### Uso de redes pre-entrenadas

![A5.png](imgs%2FCierre%2FA5.png)

### Tensorboard

![A6.png](imgs%2FCierre%2FA6.png)

### Inception Net

![A7.png](imgs%2FCierre%2FA7.png)


Ahora acompáñame al siguiente curso de [Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
