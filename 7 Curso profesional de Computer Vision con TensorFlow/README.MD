# Curso Profesional de Computer Vision con TensorFlow

Utiliza el poder de todo lo que has aprendido de redes neuronales y TensorFlow para crear productos de visión artificial. 
Aplica detección y clasificación de imágenes, seguimiento de objetos y más.

- Entrena y optimiza modelos de visión computarizada con TensorFlow.
- Pon en producción tu modelo.
- Dimensiona un proyecto de visión computarizada.
- Obtén y procesa datos de imágenes a TFRecord.


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> - [5: Curso de Experimentación en Machine Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/5%20Curso%20de%20introducci%C3%B3n%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
> - [6: Curso de detección y segmentación de objetos con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow)
> 
> Este Curso es el Número 7 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice

- [1 Comprender la visión computarizada](#1-comprender-la-visión-computarizada)
  - [1.1 ¿Por qué aprender computer vision?](#11-por-qué-aprender-computer-vision)
  - [1.2 ¿Qué es la visión computarizada?](#12-qué-es-la-visión-computarizada)
  - [1.3 Tipos de visión computarizada](#13-tipos-de-visión-computarizada)
  - [1.4 Introducción a object detection: clasificación + localización](#14-introducción-a-object-detection-clasificación--localización)
  - [1.5 Aprende a identificar problemas](#15-aprende-a-identificar-problemas)
- [2 Dimensionamiento de proyecto de visión computarizada](#2-dimensionamiento-de-proyecto-de-visión-computarizada)
  - [2.1 Cómo definir los tiempos de tu proyecto](#21-cómo-definir-los-tiempos-de-tu-proyecto)
  - [2.2 Cómo costear tu proyecto](#22-cómo-costear-tu-proyecto)
  - [2.3 Cómo identificar los roles necesarios en el proyecto](#23-cómo-identificar-los-roles-necesarios-en-el-proyecto)
  - [2.4 Producto mínimo viable en computer vision](#24-producto-mínimo-viable-en-computer-vision)
- [3 Obtención y procesamiento de los datos](#3-obtención-y-procesamiento-de-los-datos)
  - [3.1 Obtención de datos para tu proyecto](#31-obtención-de-datos-para-tu-proyecto)
  - [3.2 Limpieza de la base de datos](#32-limpieza-de-la-base-de-datos)
  - [3.3 Distribución de datos en entrenamiento y testeo](#33-distribución-de-datos-en-entrenamiento-y-testeo)
  - [3.4 Etiquetado de los datos de test](#34-etiquetado-de-los-datos-de-test)
  - [3.5 Etiquetado de los datos de train](#35-etiquetado-de-los-datos-de-train)
  - [3.6 Transforma tu base de datos a TFRecord](#36-transforma-tu-base-de-datos-a-tfrecord)
  - [3.7 Transformar CSV a TFRecord](#37-transformar-csv-a-tfrecord)
- [4 Entrena, testea y optimiza tus modelos](#4-entrena-testea-y-optimiza-tus-modelos)
  - [4.1 Librerías a importar durante fase de entrenamiento](#41-librerías-a-importar-durante-fase-de-entrenamiento)
  - [4.2 Fase de entrenamiento del modelo](#42-fase-de-entrenamiento-del-modelo)
  - [4.3 Balanceo de imágenes y data augmentation](#43-balanceo-de-imágenes-y-data-augmentation)
  - [4.4 Entrena, evalúa y optimiza con TensorBoard](#44-entrena-evalúa-y-optimiza-con-tensorboard)
  - [4.5 Validación de modelo en un entorno de ejecución](#45-validación-de-modelo-en-un-entorno-de-ejecución)
  - [4.6 Re-entrenamiento del modelo para obtener mejores resultados](#46-re-entrenamiento-del-modelo-para-obtener-mejores-resultados)
  - [4.7 Seguimiento de centroides con OpenCV](#47-seguimiento-de-centroides-con-opencv)
  - [4.8 Configuración de los centroides con OpenCV](#48-configuración-de-los-centroides-con-opencv)
  - [4.9 Algoritmo de dirección y conteo con OpenCV](#49-algoritmo-de-dirección-y-conteo-con-opencv)
  - [4.10 Crea un ciclo de entrenamiento de tu modelo: MLOps](#410-crea-un-ciclo-de-entrenamiento-de-tu-modelo-mlops)
- [5 Producto con visión computarizada en producción](#5-producto-con-visión-computarizada-en-producción)
  - [5.1 Prepara tu entorno en Google Cloud Platform](#51-prepara-tu-entorno-en-google-cloud-platform)
  - [5.2 Carga y preprocesamiento de modelos](#52-carga-y-preprocesamiento-de-modelos)
  - [5.3 Postprocesamiento de modelos](#53-postprocesamiento-de-modelos)
  - [5.4 Despliega y consume tu modelo en producción](#54-despliega-y-consume-tu-modelo-en-producción)
  - [5.5 Bonus: aprende a apagar las máquinas de GCP para evitar sobrecostos](#55-bonus-aprende-a-apagar-las-máquinas-de-gcp-para-evitar-sobrecostos)
- [6 Siguientes pasos en inteligencia artificial](#6-siguientes-pasos-en-inteligencia-artificial)
  - [6.1 Siguientes pasos en inteligencia artificial](#61-siguientes-pasos-en-inteligencia-artificial)


# 1 Comprender la visión computarizada

## 1.1 ¿Por qué aprender computer vision?

A lo largo de este curso vamos a estudiar los siguientes tópicos:

- Convertir múltiples formatos a `TFRecord`
- Pasar de clasificar un objeto a localizarlo y clasificarlo
- Poner en producción tu modelo

El camino de aprendizaje está delimitado por:

1. Problema
2. Alcance
3. Etiquetado
4. Preprocesamiento
5. Transformación
6. Entrenamiento
7. Evaluar resultado
8. Optimizar
9. Desplegar en Google
10. Monitorizar
11. Cierre del bucle

Para llevar a cabo nuestros objetivos desarrollaremos el `Proyecto del curso`.

![1.png](imgs%2F1%2F1.png)

Un sistema de clasificación, localización y seguimiento de carros y motos. Adicionalmente, vamos a poder contar la cantidad 
de objetos. El objetivo es pasar de nuestra base de `clasificación de objetos` a:

- Detección de objetos
- Segmentación de objetos
- Segmentación de instancias de objetos
- Segmentación panóptica


## 1.2 ¿Qué es la visión computarizada?

Debemos empezar por definir un par de conceptos, para entender de mejor manera ¿Qué es la visión computarizada? Empecemos 
hablando sobre ¿Qué es la inteligencia artificial?

> La inteligencia artificial (IA) se refiere a la capacidad de las máquinas y sistemas informáticos para realizar tareas 
> que normalmente requieren inteligencia humana, como el razonamiento, el aprendizaje, la percepción, la comprensión del 
> lenguaje natural y la toma de decisiones. La IA se basa en algoritmos y modelos matemáticos que permiten a los sistemas 
> informáticos procesar grandes cantidades de datos y reconocer patrones para tomar decisiones informadas.

![2.png](imgs%2F1%2F2.png)

De acuerdo con `Alan Turing` él define a una computadora como `inteligente` si era capaz de engañar a una persona haciéndole
creer que es un humano.

La IA se divide en dos categorías principales: la `inteligencia artificial débil` y la `inteligencia artificial fuerte.` 
La `inteligencia artificial débil` se refiere a sistemas que están diseñados para realizar tareas específicas y limitadas, 
como la clasificación de imágenes o la traducción automática. La `inteligencia artificial fuerte`, por otro lado, se 
refiere a sistemas que tienen una capacidad de aprendizaje y razonamiento similares a los humanos y que pueden realizar 
una amplia gama de tareas cognitivas.

Partiendo de la suposición de que la inteligencia en las máquinas está dada por la imitación y replica del comportamiento humano,
podemos hablar de la `visión computarizada` como una máquina que busca replicar el funcionamiento de los ojos humanos. Pero,
entonces ¿cómo le ponemos `ojos` a las máquinas? La respuesta es bastante sencilla e intuitiva, necesitamos hacer uso de `cámaras`
estás actuarán como los sensores que le permitirán a la máquina percibir el mundo de una forma bastante similar a la de un
humano. Pero la `cámara` por sí sola no es `visión computarizada`, una vez que tenemos el sensor listo es necesario programar
el `cerebro` que le permitirá procesar estos datos y obtener información valiosa.

> La visión computarizada (también conocida como procesamiento de imágenes o visión por computadora) se refiere al campo 
> de la inteligencia artificial que se ocupa del procesamiento de imágenes y videos para obtener información útil de ellos. 
> La visión computarizada se basa en el uso de algoritmos y técnicas de procesamiento de imágenes para analizar y extraer 
> características de imágenes y videos.

![3.png](imgs%2F1%2F3.png)

La visión computarizada tiene una amplia gama de aplicaciones en la vida cotidiana, desde el reconocimiento de rostros y
la detección de objetos en imágenes hasta la monitorización de la salud de las plantas y la inspección de la calidad de los 
productos en las fábricas. 

Sin embargo, ahora la pregunta sería: ¿cómo procesan las imágenes los sistemas computacionales? Esta información ya la hemos
discutido ampliamente en otros cursos dentro de este `learning path`, te recomiendo ampliamente leer: el [curso de redes neuronales convolucionales
con Python y Keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales). Sin embargo,
y a forma de pequeño resumen, una imagen no es más que una matriz de píxeles que guarda la información de color de cada pixel.

![4.png](imgs%2F1%2F4.png)

En una imagen en escala de grises, por ejemplo, cada pixel tiene un valor que va de 0 a 255 indicando que tan negro o blanco
es dicho pixel. A menor el valor, más oscuro el pixel, a mayor el valor más blanco. Esto es fácilmente extrapolable a imágenes
a color, puesto que estás imágenes podemos descomponer su color en términos de los tres colores principales `Red Green Blue (RGB)`
y almacenar el valor de cada uno de estos componentes que al ser combinados generan el color en cuestión. 

Cuando trabajamos con imágenes es necesario tomar en cuenta los siguientes aspectos:

- `Resolución en píxeles:` A mayor cantidad de resolución en píxeles mayor información en la imagen pero mayor coste computacional.
- `Escala de colores:` Existen varias escalas de colores que definen cómo se conforma el color, pudiendo ser desde escala de grises, hasta otros modelos de color como: RGB. BGR, HSV, CMYK.
- `Tamaño de la imagen:` Esto referente a la cantidad de memoria en disco que ocupa la imagen, no es lo mismo almacenar mil imágenes de 1 Kb a mil imágenes de 10 Mb cada una. 
- `Frames per Second (FPS):` Cuando analizamos videos estamos hablando de una colección de imágenes consecutivas una después de otra, a mayor 
cantidad de imágenes consecutivas por segundo, mayor fluidez de video pero mayor coste de procesamiento.

Algunos ejemplos específicos de aplicaciones de la visión computarizada incluyen:

- La detección de rostros en imágenes y videos para la seguridad y la vigilancia
- La detección y seguimiento de objetos en videos para la automatización de procesos industriales
- El reconocimiento de caracteres en documentos escaneados para la digitalización y la indexación de documentos
- La medición de la calidad de las cosechas agrícolas a través de la detección de enfermedades y plagas en las plantas
- La detección de anomalías en imágenes médicas para el diagnóstico de enfermedades

## 1.3 Tipos de visión computarizada

Ya hemos hablado anteriormente de que la clasificación de imágenes es un problema de visión computarizada, pero también
vale la pena hablar de cuáles son los demás tipos de visión computarizada que éxisten. Empecemos por decir que la clasificación
de imágenes puede ser `multiclase` o de `múltiples etiquetas`. En la clasificación `multiclase` el modelo es capaz de diferenciar
entre varios objetos con atributos diferentes, pero uno a la vez, mientras que el `clasificador de múltiples etiquetas` me permite
encontrar la clasificación de varios objetos dentro de la misma imagen, pero sin llegar a decirme dónde se encuentran estos entes.

![5.png](imgs%2F1%2F5.png)

Para este problema existe él `object detection`. El modelo primero identifica dónde se encuentran los objetos de interés y 
después los clasifica. Del `object detection` se generan `bounding boxes` que esencialmente son recuadros que encapsulan la 
posición del objeto de interés. Un salto más allá de la detección de objetos sería `image segmentation` que sería clasificar pixel
a pixel a que clase corresponde cada uno. Básicamente, existen dos tipos de `image segmentation` la `semantic` y la de `instance`.

![6.png](imgs%2F1%2F6.png)

La `semantic segmnetation` busca identificar todos los elementos de una clase de la misma manera, mientras que la `instance segementation`
busca separar cada elemento a pesar de que ambos pertenezcan a la misma clase.

- La segmentación semántica se refiere al proceso de dividir una imagen en diferentes regiones basadas en su contenido semántico. En otras palabras, la segmentación semántica clasifica cada píxel de una imagen en diferentes categorías de objetos, como personas, coches, árboles, edificios, etc. La segmentación semántica se utiliza en aplicaciones como la detección de objetos, la identificación de áreas peligrosas en imágenes de vigilancia, y la automatización de tareas en la industria.

- La segmentación de instancias, por otro lado, se refiere al proceso de identificar y separar cada objeto individual en una imagen. En otras palabras, la segmentación de instancias no solo identifica los objetos en una imagen, sino que también los separa en diferentes instancias. Por ejemplo, si hay dos personas en una imagen, la segmentación de instancias las separará en dos objetos individuales, en lugar de clasificarlos como un solo objeto del tipo "persona". La segmentación de instancias se utiliza en aplicaciones como la conducción autónoma, la robótica y la detección de objetos en videos.

 Adicionalmente, a estas dos técnicas tenemos una tercera conocida como: `panoptic segmentation` que de forma simple, no es más que
la combinación de las dos técnicas anteriores.

![7.png](imgs%2F1%2F7.png)

La segmentación panóptica es una técnica de visión computarizada que combina la segmentación semántica y de instancias para proporcionar una comprensión completa de una imagen o video. La segmentación panóptica identifica tanto los objetos individuales en una imagen como las categorías semánticas a las que pertenecen. En otras palabras, la segmentación panóptica permite identificar no solo qué objetos están en la imagen, sino también a qué clase pertenecen.

La segmentación panóptica se basa en la idea de que una imagen o video se puede descomponer en dos componentes: los objetos individuales (instancias) y los objetos colectivos (categorías semánticas). Por lo tanto, la segmentación panóptica utiliza un enfoque híbrido para identificar y segmentar tanto las instancias individuales como las categorías semánticas en una imagen o video.

## 1.4 Introducción a object detection: clasificación + localización

Como comentamos en la sección anterior, la detección de objetos está compuesta por la suma de: localización del objeto + clasificación del mismo.
La localización del objeto se hace a través de un `boungind box` un sistema de 4 cordenadas que engloba la posición del objeto. 
Existen varios formatos de `bounding boxes` entre los que podemos mencionar:

![8.png](imgs%2F1%2F8.png)

- Formato `XYWH`: Este es el formato más común y simple. Se define por cuatro números que indican las coordenadas X e Y del borde superior izquierdo de la caja delimitadora, seguido por su ancho (width) y altura (height).

- Formato `XYXY`: También conocido como el formato "punto inicial-punto final", este formato utiliza cuatro números para definir las coordenadas de los puntos inicial y final de la caja delimitadora en lugar de su ancho y altura. Los dos primeros números representan las coordenadas del punto superior izquierdo, mientras que los dos últimos números representan las coordenadas del punto inferior derecho.

Te recomiendo leer: [Albumentation: Bounding boxes](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/)

## Arquitecturas de detección de objetos

Recomiendo ampliamente leer: [Tipos de arquitecturas en detección de objetos](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow#25-tipos-de-arquitecturas-en-detecci%C3%B3n-de-objetos) Para entender
los enfoques `single-stage` y `multi-stage` en las arquitecturas de `object detection`.

![arquitecturas](https://github.com/ichcanziho/Deep_Learnining_Platzi/raw/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow/ims%2F2a%2F7.png)

## Conceptos básicos de Object detection:

Te recomiendo leer [Introducción a object detection: backbone, non-max suppression y métricas](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow#23-introducci%C3%B3n-a-object-detection-backbone-non-max-suppression-y-m%C3%A9tricas)

![iou](https://github.com/ichcanziho/Deep_Learnining_Platzi/raw/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow/ims%2F2a%2F4.png)


## Algoritmos más utilizados:

Te recomiendo leer: [Arquitecturas relevantes en object detection](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow#26-arquitecturas-relevantes-en-object-detection)

![fast](https://github.com/ichcanziho/Deep_Learnining_Platzi/raw/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow/ims%2F2a%2F10.png)

## Principios de detección de objetos

En las arquitecturas de una sola etapa, se necesita un `backbone` o una `columna vertebral` un algoritmo pre-entrenado de
clasificación que nos ayude a extraer las características generales de la imagen, después agregamos un par de capas extras
convolucionales con el objetivo de encontrar los `bounding boxes` de los posibles objetos encontrados y finalmente usamos
la técnica de `non-max suppresion` para limpiar los resultados.

![9.png](imgs%2F1%2F9.png)

En esta clase el profesor hablo sobre `MobileNet V2` como `backbone` para nuestros problemas de `object detection` a continuación
una breve introducción a `MobileNet V2`:

MobileNetV2 es una arquitectura de red neuronal convolucional (CNN) diseñada para la clasificación de imágenes en dispositivos móviles y embebidos con recursos limitados de cómputo. Fue desarrollada por Google y se presentó en 2018.

MobileNetV2 utiliza una serie de bloques de construcción llamados bloques Inverted Residual (Residual Invertido en español) que permiten una mayor eficiencia computacional en comparación con otras arquitecturas de CNN. Los bloques Inverted Residual constan de dos capas de convolución separable, que separan la convolución espacial y la convolución de canal, y una conexión de salto (skip connection) para mejorar la gradiente y la capacidad de generalización de la red.

![10.png](imgs%2F1%2F10.png)

La arquitectura MobileNetV2 tiene aproximadamente 3,4 millones de parámetros entrenables, lo que la hace mucho más pequeña que otras redes neuronales convolucionales utilizadas en la clasificación de imágenes, como ResNet y VGG. Además, MobileNetV2 es capaz de lograr un alto rendimiento en la clasificación de imágenes con una tasa de error comparable a la de otras arquitecturas de redes neuronales más grandes.

Una de las ventajas de MobileNetV2 es su capacidad de procesamiento en dispositivos móviles con recursos limitados. La red es lo suficientemente pequeña para ser ejecutada en dispositivos móviles en tiempo real, lo que la hace adecuada para aplicaciones de detección y clasificación de objetos en tiempo real. Además, MobileNetV2 es fácilmente transferible y se puede utilizar para la detección y segmentación de objetos en aplicaciones de visión por computadora.


## 1.5 Aprende a identificar problemas

Como desarrolladores `NO TODO ES CÓDIGO`, mucho de la resolución de problemas nace en el propio planteamiento del problema
más allá de la integración en código. Es indispensable conocer completamente el problema a resolver, y el contexto en el que
se desenvuelve el problema, es necesario conocer del negocio y giro de la empresa. 

La identificación de problemas consta de tre etapas:

- Encontrar el problema: por ejemplo - me toma 2 horas ir al trabajo.
- Entender cuál es el resultado que buscas: por ejemplo - idealmente me gustaría que fueran 15 min
- Conocer a los usuarios: por ejemplo - todas las personas que viven lejos de su trabajo

La respuesta última NO siempre es IA.

En este momento quizá no tengo ni idea de como voy a hacer que moverme a mi empresa me tome 15 minutos, pero ya e identificado
el problema, ya tengo un resultado ideal y ya conozco cuál es el `objetivo` de usuarios al que mi respuesta va a ayudar. 
Entre mayor sea el número de usuarios que tengan el mismo problema, mayor será el beneficio de solucionarlo y será más fácil
encontrar recursos e incluso inversiones para llevar a cabo la solución.

![11.png](imgs%2F1%2F11.png)

Definamos más formalmente nuestro `Caso de estudio`:

- `Problema:` Todos los días me toma alrededor de 2 horas para llegar a mi casa: congestión vehicular
- `Causa:` no existen métricas en tiempo real que nos permitan monitorizar y analizar la distribución de vehículos en mi ciudad, no hay analítica en tiempo real
- `Solución:` generar métricas en tiempo real a la ciudadanía para que se pueda modificar vías (en el largo plazo), gestionar el flujo por las rutas (en el corto plazo)
- `Viabilidad de la solución:` las ciudades mexicanas pierden 94 mil millones de pesos (4.6 mil millones de USD) por el tráfico al año.

# 2 Dimensionamiento de proyecto de visión computarizada

## 2.1 Cómo definir los tiempos de tu proyecto

En este módulo vamos a aprender acerca de los siguientes temas:

- Cómo definir el alcance de tu proyecto: tiempos y costos
- Identificar roles para tu proyecto
- Definir el producto mínimo viable

Empecemos hablando con el alcance del proyecto. Para llevar a cabo satisfactoriamente un proyecto debemos pensar en
`must have`, `should have`, `nice to have` y `won't have` los cuales son tres categorías que se utilizan comúnmente en la gestión de 
proyectos para clasificar los requisitos o características de un proyecto. Estas categorías ayudan a priorizar las 
características del proyecto en función de su importancia y necesidad.

![1.png](imgs%2F2%2F1.png)

1. `Must have:` son los requisitos o características críticas y esenciales del proyecto que deben ser entregados para que el proyecto sea considerado exitoso. En otras palabras, son las funcionalidades o características imprescindibles que el proyecto debe tener. Si alguno de estos requisitos no se cumple, el proyecto no se considera completado o exitoso. Por lo tanto, estos elementos deben ser entregados a cualquier costo.

2. `Should have:` son los requisitos o características importantes del proyecto que deben ser entregados si es posible, pero que no son esenciales para considerar el proyecto exitoso. Si es posible, estos requisitos deben ser entregados, pero si se presentan problemas o limitaciones en el proceso, se pueden posponer o incluso eliminar sin afectar significativamente el éxito del proyecto.

3. `Nice to have:` son requisitos o características deseables, pero no esenciales para el éxito del proyecto. Estos elementos son considerados "extra" y no son críticos para el funcionamiento o cumplimiento del objetivo principal del proyecto. Si el tiempo y los recursos lo permiten, se pueden incluir estas características para mejorar la calidad del proyecto, pero si hay limitaciones de tiempo o recursos, se pueden eliminar sin afectar el éxito del proyecto.

4. `Won't have:` son los requisitos o características que se han identificado como no esenciales o no necesarios para el éxito del proyecto, pero que se han decidido explícitamente no incluir en el proyecto. En otras palabras, son los elementos que se han descartado o rechazado debido a limitaciones de tiempo, recursos, presupuesto, o por otras razones estratégicas o técnicas.

Cuado hablamos del `tiempo` del proyecto nos referimos principalmente a dos tipos de tiempo el `externo` y el `interno`:

- los `tiempos externos` se refieren a los plazos y restricciones que se derivan de factores externos al proyecto en sí mismo, como pueden ser las limitaciones presupuestarias, los plazos de entrega, las regulaciones y leyes aplicables, entre otros.

- los `tiempos internos` se refieren a los plazos y restricciones que se derivan del propio proyecto, como son los tiempos necesarios para llevar a cabo cada tarea, los recursos disponibles, las interdependencias entre las diferentes tareas y los hitos importantes del proyecto.

Te compartimos un [plan de trabajo](recursos%2Fplan_de_trabajo.xlsx) para que veas como se puede organizar
el proyecto propuesto en este curso.

![2.png](imgs%2F2%2F2.png)

Otro punto indispensable es conocer exactamente cuáles son las tareas y actividades que se deben realizar y conocer los
tiempos aproximados de cada actividad así como sus KPIS:

![3.png](imgs%2F2%2F3.png)

> KPI son las siglas en inglés de "Key Performance Indicators" o "Indicadores Clave de Desempeño" en español. Son métricas que se utilizan para medir y evaluar el rendimiento de una empresa, un departamento, un proyecto o una persona en relación con los objetivos establecidos.
>
> Los KPI se utilizan para medir el éxito en el cumplimiento de los objetivos y metas de una empresa o proyecto. Se basan en datos numéricos y pueden medirse en diferentes áreas de la empresa, como ventas, marketing, finanzas, operaciones, atención al cliente, entre otros.
>
> Los KPI deben ser cuidadosamente seleccionados y definidos para que sean relevantes, específicos, medibles, alcanzables y relevantes para los objetivos y metas del negocio. Algunos ejemplos comunes de KPI incluyen la tasa de conversión de ventas, el índice de satisfacción del cliente, el retorno sobre la inversión (ROI), el tiempo promedio de respuesta al cliente, entre otros.



## 2.2 Cómo costear tu proyecto

Realmente NO hay una fórmula mágica que permita obtener el costo perfecto de un proyecto, sin embargo, a continuación hablaremos
de algunos consejos que podemos tomar en cuenta y hablaremos un poco sobre la teoría del tema. Las recomendaciones que podemos tomar
en cuenta son:

![4.png](imgs%2F2%2F4.png)

- `No existe una única forma de costear el proyecto:` existen varias formas de obtener capital para desarrollar tu proyecto, desde préstamos a bancos, a familiares y conocidos hasta hablar con inversores o patrocinadores. Se debe tener una mente abierta y un amplio criterio para buscar las mejores opciones que se adapten a tus necesidades.

- `Crea tres escenarios:` Para costear tu proyecto, es importante crear tres escenarios diferentes: el escenario optimista, el escenario realista y el escenario pesimista. El escenario optimista debe incluir los mejores resultados posibles, el escenario realista debe incluir una estimación precisa de los costos (puedes pensar en un costo 20% superior al escenario ideal) y los ingresos, y el escenario pesimista debe incluir los peores resultados posibles (puedes pensar en un 20% adicional al escenario realista).

- `Agrega a los tres escenarios un margen de falla:` Para cada escenario, es importante agregar un margen de falla que tenga en cuenta posibles imprevistos o costos adicionales que puedan surgir durante el desarrollo del proyecto. Este margen de falla debe ser lo suficientemente amplio para cubrir los gastos imprevistos y evitar que el proyecto se detenga o tenga dificultades financieras.

- `Agrega costos de servidores o herramientas externas:` Al costear un proyecto, es importante tener en cuenta los costos de los servidores y herramientas externas necesarias para su desarrollo. Estos costos pueden incluir la compra o alquiler de servidores, software y herramientas de análisis, entre otros. Es importante incluir estos costos en los escenarios de costeo para asegurarse de que el presupuesto del proyecto sea preciso y completo.

- `Divide tu proyecto por entregas mensuales:` Una buena forma de costear un proyecto es dividirlo en entregas mensuales y establecer un precio por cada una de ellas. De esta manera, podrás tener un mejor control de los costos y asegurarte de que estás cobrando de manera justa y equitativa por cada etapa del proyecto.

- `Válida y firme el alcance propuesto (cambios representan costos adicionales):` Es importante validar y firmar el alcance propuesto antes de comenzar el proyecto para evitar que los cambios o adiciones no planificadas representen costos adicionales. Al validar y firmar el alcance, se establecen los límites y los requisitos específicos del proyecto, lo que ayuda a asegurar que los costos sean precisos y que no haya sorpresas en el futuro.

- `Analiza la metodología de cobro:` Antes de comenzar un proyecto, es importante analizar la metodología de cobro para asegurarte de que se ajuste a tus necesidades y a las de tu cliente. Algunas opciones incluyen el pago por hora, por proyecto, por hito o por resultados. Es importante elegir una metodología de cobro que sea justa y transparente y que tenga en cuenta los costos y la complejidad del proyecto.

- `Que no te dé miedo cobrar:` Por último, es importante recordar que no debes tener miedo de cobrar por tu trabajo. Cobrar por tu tiempo y esfuerzo es una parte esencial de cualquier negocio y es importante asegurarte de que estás siendo compensado de manera justa por tu trabajo. Asegúrate de establecer precios justos y equitativos y de que estás cobrando de acuerdo con el alcance acordado y las metodologías de cobro establecidas.

## 2.3 Cómo identificar los roles necesarios en el proyecto

Definir los roles en un proyecto es una de las partes más importantes de la planificación inicial, ya que ayuda a establecer expectativas claras para los miembros del equipo y a garantizar que cada persona entienda sus responsabilidades y tareas dentro del proyecto. Algunas de las razones por las que es importante tener bien definidos los roles al momento de empezar un proyecto son:

- `Claridad en las responsabilidades:` Al definir los roles, cada miembro del equipo sabe exactamente qué se espera de él y cuáles son sus responsabilidades. Esto ayuda a evitar confusiones y a asegurar que cada persona se concentre en su área de especialización.

- `Evita duplicación de tareas:` Cuando los roles están bien definidos, es menos probable que dos o más miembros del equipo estén trabajando en la misma tarea sin saberlo. Esto puede ahorrar tiempo y minimizar los errores.

- `Promueve la colaboración:` Al saber quién es responsable de qué tarea, se fomenta la colaboración y la comunicación entre los miembros del equipo. Cada persona puede apoyar a los demás en su área de especialización y trabajar juntos hacia un objetivo común.

- `Ayuda a establecer plazos y objetivos:` Al saber quién es responsable de cada tarea, es más fácil establecer plazos y objetivos realistas. Cada persona puede trabajar en su tarea con un plazo específico en mente y asegurarse de que se cumpla dentro del tiempo establecido.

Los proyectos de Machine Learning siempre necesitan: Experto en el área + Equipo de ML


![5.png](imgs%2F2%2F5.png)


- `Experto en el área:` Es la persona con conocimiento específico y avanzado en el tema de interés, no necesariamente tiene conocimiento de programación o conoce el mercado, pero su mayor contribución es tener amplio criterio y conocimiento sobre el tema y el problema a resolver.

- `Líder del proyecto:` Es la persona encargada de dirigir y coordinar el equipo de trabajo en la ejecución del proyecto. Es responsable de establecer objetivos, plazos y recursos, y de supervisar el progreso del proyecto en general. Debe tener habilidades de liderazgo y gestión de proyectos, así como una comprensión general del proceso de desarrollo de software.

- `Legal:` Es la persona encargada de revisar y gestionar todos los aspectos legales relacionados con el proyecto. Esto incluye la revisión de contratos, acuerdos de confidencialidad, licencias de software, propiedad intelectual, entre otros. Debe tener un conocimiento especializado en derecho y experiencia en la gestión de asuntos legales.

- `MKT:` Es la persona encargada de planificar y ejecutar la estrategia de marketing del producto o servicio desarrollado por el equipo de tecnología. Debe tener habilidades de marketing y publicidad, así como una comprensión general del mercado objetivo y la competencia.

- `Arquitecto de software:` Es la persona encargada de diseñar la estructura y arquitectura del software que se está desarrollando. Debe tener un conocimiento profundo de las tecnologías y herramientas que se utilizan en el proyecto, y ser capaz de diseñar soluciones escalables y sostenibles.

- `Científico de datos:` Es la persona encargada de analizar y procesar los datos del proyecto, con el objetivo de extraer información valiosa y tomar decisiones informadas. Debe tener habilidades en estadística, modelado de datos y programación, y ser capaz de trabajar con grandes volúmenes de datos.

- `Diseñador de experiencia de usuario:` Es la persona encargada de diseñar la experiencia del usuario en el producto o servicio que se está desarrollando. Debe tener habilidades en diseño gráfico, diseño de interacción y usabilidad, y ser capaz de diseñar interfaces atractivas y fáciles de usar.

- `Machine learning engineer:` Es la persona encargada de diseñar e implementar soluciones basadas en algoritmos de machine learning. Debe tener un conocimiento profundo en estadística, programación y aprendizaje automático, y ser capaz de diseñar soluciones escalables y sostenibles.

- `Frontend:` Es la persona encargada de desarrollar la interfaz de usuario y la parte visual del producto o servicio que se está desarrollando. Debe tener habilidades en diseño gráfico, programación web y tecnologías de front-end.

- `Backend:` Es la persona encargada de desarrollar la lógica de negocios y la parte de programación que no está visible para el usuario final. Debe tener habilidades en programación web, bases de datos y tecnologías de backend.

- `Bases de datos:` Es la persona encargada de diseñar y gestionar la base de datos del proyecto. Debe tener habilidades en diseño de bases de datos, programación y gestión de datos.

- `Ingeniero de datos:` Es la persona encargada de diseñar e implementar soluciones de gestión y procesamiento de datos a gran escala. Debe tener habilidades en programación, estadística y gestión de datos.

- `Tester:` Es la persona encargada de probar el software que se está desarrollando y asegurarse de que funciona correctamente. Debe

## 2.4 Producto mínimo viable en computer vision

El Producto Mínimo Viable (PMV o MVP, por sus siglas en inglés) es una estrategia de desarrollo de productos que implica crear y lanzar al mercado un producto con un conjunto mínimo de características que satisfagan las necesidades básicas del cliente.

La idea detrás del PMV es validar la viabilidad de una idea de producto o negocio con la menor cantidad de inversión posible. Esto se logra mediante la creación de un prototipo funcional que pueda ser utilizado por los primeros clientes, y que permita recopilar comentarios y retroalimentación para mejorar el producto de manera iterativa.

> Entre más rápido entregué un PMV más rápido puedo mejorarlo para adaptarme a las necesidades del cliente.

> Tener un PMV ayuda a agilizar el proceso de desarrollo del producto. 

En el contexto del PMV, los siguientes términos se refieren a principios o estrategias que se utilizan para desarrollar y lanzar un producto mínimo viable:

![6.png](imgs%2F2%2F6.png)


- `Try small` (probar en pequeña escala): se refiere a la idea de comenzar con una versión reducida de un producto, para evaluar su aceptación por parte del mercado, antes de invertir tiempo y recursos significativos en el desarrollo de una versión más completa.

- `Fail fast` (fracasar rápido): se refiere a la idea de identificar y solucionar problemas rápidamente para evitar un mayor desperdicio de tiempo y recursos. Se trata de realizar iteraciones frecuentes y recibir retroalimentación temprana del mercado para hacer ajustes y mejoras rápidas.

- `Fail cheap` (fracasar a bajo costo): se refiere a la idea de minimizar los costos de desarrollo y lanzamiento del producto, para que, en caso de fracasar, el costo financiero sea mínimo. Esto permite a los emprendedores tomar riesgos y probar nuevas ideas sin el temor de perder grandes sumas de dinero.

- `Fail forward` (fracasar hacia adelante): se refiere a la idea de aprender de los errores y fracasos, para mejorar y avanzar en el desarrollo del producto. Se trata de utilizar el fracaso como una oportunidad para identificar lo que no funciona y encontrar formas de mejorar.

- `Don't aim for perfection (no apuntar a la perfección): se refiere a la idea de no buscar la perfección en la primera versión del producto, sino centrarse en las características básicas que satisfagan las necesidades del cliente. Se trata de aceptar que el producto será imperfecto y buscar mejorarlo iterativamente en futuras versiones a medida que se recibe retroalimentación del mercado.

---------------

En el contexto de nuestro problema a resolver podemos identificar los siguientes objetivos:


![7.png](imgs%2F2%2F7.png)


# 3 Obtención y procesamiento de los datos

## 3.1 Obtención de datos para tu proyecto

## 3.2 Limpieza de la base de datos

## 3.3 Distribución de datos en entrenamiento y testeo

## 3.4 Etiquetado de los datos de test

## 3.5 Etiquetado de los datos de train

## 3.6 Transforma tu base de datos a TFRecord

## 3.7 Transformar CSV a TFRecord

# 4 Entrena, testea y optimiza tus modelos

## 4.1 Librerías a importar durante fase de entrenamiento

## 4.2 Fase de entrenamiento del modelo

## 4.3 Balanceo de imágenes y data augmentation

## 4.4 Entrena, evalúa y optimiza con TensorBoard

## 4.5 Validación de modelo en un entorno de ejecución

## 4.6 Re-entrenamiento del modelo para obtener mejores resultados

## 4.7 Seguimiento de centroides con OpenCV

## 4.8 Configuración de los centroides con OpenCV

## 4.9 Algoritmo de dirección y conteo con OpenCV

## 4.10 Crea un ciclo de entrenamiento de tu modelo: MLOps

# 5 Producto con visión computarizada en producción

## 5.1 Prepara tu entorno en Google Cloud Platform

## 5.2 Carga y preprocesamiento de modelos

## 5.3 Postprocesamiento de modelos

## 5.4 Despliega y consume tu modelo en producción

## 5.5 Bonus: aprende a apagar las máquinas de GCP para evitar sobrecostos

# 6 Siguientes pasos en inteligencia artificial

## 6.1 Siguientes pasos en inteligencia artificial


