# Curso de Redes Neuronales Convolucionales con Python y keras

En este curso avanzado eleva tus conocimientos de deep learning. Comprende el funcionamiento de las redes neuronales convolucionales. Sigue el camino de la visión artificial donde este tipo de red neuronal es utilizada.

- Optimizarás tus redes aplicando data augmentation, callbacks y batch normalization.
- Comprenderás el kernel, padding, strides y la capa de pooling.
- Manejarás imágenes con Python para su uso en redes neuronales.

> ## NOTA:
> Antes de continuar te invito a que revises el curso anterior:
> 
> [Curso de fundamentos de redes neuronales con Python y Keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
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
  - [5.1 Clasificación con redes neuronlaes convolucionales](#51-clasificación-con-redes-neuronlaes-convolucionales)
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

## 3.2 Manejo de imágenes con Python

# 4. Fundamentos de redes neuronales convolucionales

## 4.1 Kernel en redes neuronales

## 4.2 El Kernel en acción

## 4.3 Padding y Strides

## 4.4 Capa de pooling

## 4.5 Arquitectura de redes convolucionales

## 4.6 Quizz Fundamentos de redes neuronales convolucionales

# 5. Resolviendo un problema de clasificación

## 5.1 Clasificación con redes neuronlaes convolucionales

## 5.2 Creación de red convolucional para clasificación

## 5.3 Entrenamiento de un modelo de clasificación con redes convolucionales

# 6.Optimización de una red neuronal convolucional

## 6.1 Data augmentation

## 6.2 Aplicando data augmentation

## 6.3 Callbacks: early stopping y checkpoints

## 6.4 Batch normalization

## 6.5 Optimización de modelo de clasificación

## 6.6 Entrenamiento de nuestro modelo de clasificación optimizado

## 6.7 Quizz: Optimización de redes neuronales convolucionales

# 7. Resolviendo una competencia de Kaggle

## 7.1 Clasificando entre perros y gatos

## 7.2 Entrenamiento del modelo de clasificación de perros y gatos

# 8. Cierre

## 8.1 Siguientes pasos
