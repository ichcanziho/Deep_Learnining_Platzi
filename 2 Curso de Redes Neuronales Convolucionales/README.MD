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

# 2. Mi primera red neuronal convolucional

## 2.1 Creando nuestra primer red convolucional

## 2.2 Entrenando nuestra primera red convolucional

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
