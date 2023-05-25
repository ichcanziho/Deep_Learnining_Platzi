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

Para crear un dataset de object detection es buena idea pensar en la mayor cantidad de escenarios posibles
en los cuales pueda presentarse nuestro problema a resolver. Tomando como base la problemática de nuestro proyecto podemos
pensar en las siguientes preguntas:

- ¿Qué pasa si el vehículo va a 100 km/h?
- ¿Qué pasa si un camión tapa a un vehículo?
- ¿El algoritmo funcionaría de noche?
- ¿Qué sucede si llueve?
- ¿En dónde se integrará nuestra tecnología?
- ¿Quién tiene acceso a la base de datos?

El objetivo es tener un dataset muy amplio y con diferentes contextos, nuestro problema no siempre será la detección de
coches un martes a las 5 pm en un día soleado. Se necesita tener una gran diversidad de escenarios para que el modelo 
de DL que entrenemos pueda reaccionar ante todo tipo de contextos.

![1.png](imgs%2F3%2F1.png)

Como hemos hablado en otros cursos existen varios lugares donde puedes obtener datasets disponibles. Idealmente, es la empresa
la que debería proporcionarte el dataset con las características que estás buscando para poder trabajar. También podrías 
hacerla manualmente. Puedes leer sobre otros recursos online en: 

- [Aprende a buscar bases de datos para DL](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow#27-aprende-a-buscar-bases-de-datos-para-deep-learning)
- [Dónde podemos conseguir datasets de OD?](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow#d%C3%B3nde-podemos-conseguir-datasets)


## 3.2 Limpieza de la base de datos

El primer paso fue descargar un conjunto de imágenes con coches y motocicletas. El primer conjunto contenía 127 imágenes sin embargo
varias de estas eran `basura`, con una limpieza manual, cada imagen que no nos aportaba información relevante fue eliminada.

![3.png](imgs%2F3%2F3.png)

Quedando al final una base limpia con un total de 57 imágenes validadas. 

![2.png](imgs%2F3%2F2.png)


## 3.3 Distribución de datos en entrenamiento y testeo

> ## Nota:
> Puedes encontrar el código de está sección en: [Distribución de datos.py](scripts%2F3%2F3%20Distribuci%C3%B3n%20de%20datos%2FDistribuci%C3%B3n%20de%20datos.py)

En esta sección vamos a crear un código que nos permita dividir nuestro `dataset` en dos conjuntos `train` y `test` con una
proporción `70-30` respectivamente.

Empecemos por importar las bibliotecas necesarias y definir las rutas que vamos a utilizar:

```python
import os
import random
import shutil

dirs = {"root": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/",
        "dataset": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/dataset",
        "clean": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean",
        "train": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean/train",
        "test": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean/test"}
```

Creamos una función que nos permita crear directamente nuestra ditribución de carpetas:

```python
def make_dirs(paths):
    for _, path in paths.items():
        if not os.path.exists(path):
            os.mkdir(path)
```

Finalmente, creamos una función para dividir el conjunto original de imágenes en `train` y `test` y poner las imágenes
en sus respectivas carpetas objetivo:

```python
def split_data(paths, train_ratio):
    train = train_ratio
    # Obtenemos una lista con todas las imágenes de la ruta `dataset`
    content = os.listdir(paths["dataset"])
    for nCount in range(int(len(content) * train)):
        # Seleccionamos una imagen aleatoria
        random_choice_img = random.choice(content)
        random_choice_img_abs = "{}/{}".format(paths["dataset"], random_choice_img)
        target_img = "{}/{}".format(paths["train"], random_choice_img)
        # Copiamos la imagen de la ruta actual a la ruta objetivo
        shutil.copyfile(random_choice_img_abs, target_img)
        # Eliminamos de la lista de posibles opciones, la imagen actual que hemos seleccionado. Esto con el fin de 
        # evitar imágenes repetidas entre los conjuntos de training y testing. 
        content.remove(random_choice_img)

    # La lista de imágenes que queda en `content` representan el testing set
    for img in content:
        random_choice_img_abs = "{}/{}".format(paths["dataset"], img)
        target_img = "{}/{}".format(paths["test"], img)
        # Aquí solo es necesario copiar las imágenes a la nueva ruta
        shutil.copyfile(random_choice_img_abs, target_img)
```

Finalmente, el código completo queda de la siguiente manera:

```python
import os
import random
import shutil

def split_data(paths, train_ratio):
    train = train_ratio # este es el porcentaje de imágenes que quiero utilizar para la carpeta `train`
    # Obtenemos una lista con todas las imágenes de la ruta `dataset`
    content = os.listdir(paths["dataset"])
    for nCount in range(int(len(content) * train)):
        # Seleccionamos una imagen aleatoria
        random_choice_img = random.choice(content)
        random_choice_img_abs = "{}/{}".format(paths["dataset"], random_choice_img)
        target_img = "{}/{}".format(paths["train"], random_choice_img)
        # Copiamos la imagen de la ruta actual a la ruta objetivo
        shutil.copyfile(random_choice_img_abs, target_img)
        # Eliminamos de la lista de posibles opciones, la imagen actual que hemos seleccionado. Esto con el fin de 
        # evitar imágenes repetidas entre los conjuntos de training y testing. 
        content.remove(random_choice_img)

    # La lista de imágenes que queda en `content` representan el testing set
    for img in content:
        random_choice_img_abs = "{}/{}".format(paths["dataset"], img)
        target_img = "{}/{}".format(paths["test"], img)
        # Aquí solo es necesario copiar las imágenes a la nueva ruta
        shutil.copyfile(random_choice_img_abs, target_img)

def make_dirs(paths):
    for _, path in paths.items():
        if not os.path.exists(path):
            os.mkdir(path)

if __name__ == '__main__':
    dirs = {"root": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/",
            "dataset": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/dataset",
            "clean": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean",
            "train": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean/train",
            "test": "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean/test"}

    make_dirs(dirs)

    split_data(dirs, train_ratio=0.7)
```

## 3.4 Etiquetado de los datos de test

En la clase se explora la página: https://www.linkedai.co/ que es un servicio web gratuito que permite el etiquetado de imágenes
para proyectos de deep learning.

![4.png](imgs%2F3%2F4.png)

Entre las cualidades más interesantes que tiene la página podemos encontrar:

- Te permite trabajar con diferentes tipos de etiquetado:
  
![5.png](imgs%2F3%2F5.png)

- A cada clase que se crea se le puede asignar un color y nombre:

![6.png](imgs%2F3%2F6.png)

- Te permite alterar una previsualización de la imagen para ayudar al proceso de etiquetado:

![7.png](imgs%2F3%2F7.png)

- Te da el tiempo invertido en etiquetar cada imagen:

![8.png](imgs%2F3%2F8.png)

- Te comparte el porcentaje de etiquetado, así como la distribución de clases:

![9.png](imgs%2F3%2F9.png)

- Una imagen puede contener muestras de varios objetos de iguales o diferentes clases:

![10.png](imgs%2F3%2F10.png)

- La página regresa las etiquetas en formato `json`

## 3.5 Etiquetado de los datos de train

En la página de https://www.linkedai.co/ se necesito crear un nuevo proyecto para continuar con el etiquetado de la partición de entrenamiento.

![11.png](imgs%2F3%2F11.png)

Se repitió el proceso de etiquetado para el conjunto de entrenamiento.

## 3.5b Creando Proyecto Final

Ahora para no repetir el proceso idéntico de la clase vamos a seguir en paralelo el proyecto propuesto junto con nuestro 
propio problema de DL. El problema es la verificación de vida para un sistema de validación de identidad.

La propuesta de solución es: crear un sistema que permita identificar una contraseña numerica mostrada con la mano derecha.

Vamos a empezar creando el dataset que vamos a utilizar para la identificación de `¿qué número representa la mano?`

### Primero configuramos nuestro entorno:

```commandline
python3.10 -m venv odenv
source odenv/bin/activate
pip install --upgrade pip
pip install opencv-python==4.5.5.62
pip install pandas
```

### Obtención del dataset

El objetivo del siguiente pequeño código es tener una forma de utilizar la cámara de nuestra computadora/laptop para tomar
`n_examples` muestras de imágenes pertenecientes a `n_classes` diferentes. Para este proyecto queremos detectar los números 
del `1 al 5` con base en la posición de los dedos de la mano. Entonces nuestro `n_classes` será de 5, por otro lado, el 
número de instancias por clase `n_examples` será de 14 imágenes. 

> ## Nota:
> El código de esta sección está: [get_images.py](scripts%2Fproject%2Fcore%2Fget_images.py)


```python
import cv2


def get_data(n_classes, n_examples):

    video = cv2.VideoCapture(0)
    current_class = 1
    current_example = 1
    print("Collecting data for class", current_class)
    while video.isOpened():

        ret, frame = video.read()

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print(f"collecting example: {current_example}/{n_examples}")
            cv2.imwrite(f"../dataset/c{current_class}_e{current_example}.png", frame)
            current_example += 1

        if current_example > n_examples:
            current_example = 1
            current_class += 1
            print("Collecting data for class", current_class)

        cv2.putText(frame, f"C{current_class}/{n_classes} E{current_example}/{n_examples}", (20, 40),
                    color=(255, 0, 0), fontScale=1, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

        cv2.imshow("Recording", frame)

        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')) or current_class > n_classes:
            video.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    get_data(n_classes=5, n_examples=14)

```

![0.png](imgs%2Fproject%2F0.png)

Ahora procedemos a crear 2 particiones: `training` y `testing` en `training` vamos a ocupar `10 imágenes` por cada clase para un
total de `50 imágenes` de entrenamiento y en `testing` las restantes `4 imágenes` para un total de `20 imágenes`.

Carpeta con las imágenes de `training`:
![1.png](imgs%2Fproject%2F1.png)

Carpeta con las imágenes de `testing`:
![2.png](imgs%2Fproject%2F2.png)



### Etiquetado de la base de datos:

Como variante al curso vamos a usar [labelImg](https://github.com/heartexlabs/labelImg) para etiquetar nuestras imágenes:
Clonamos el repositorio desde github e instalamos lo necesario:
```commandline
git clone https://github.com/heartexlabs/labelImg.git
cd labelImg/
make qt5py3
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
```

![3.png](imgs%2Fproject%2F3.png)

Después de un proceso manual de etiquetado, tenemos ambos conjuntos de datos (training, testing) completamente etiquetados:

![4.png](imgs%2Fproject%2F4.png)

Cada etiqueta de `object detection` dada por `labelImg` luce de la siguiente manera:

```python
<annotation>
	<folder>train</folder>
	<filename>c1e5.png</filename>
	<path>/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/project/dataset/train/c1e5.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>640</width>
		<height>480</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>d1</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>145</xmin>
			<ymin>153</ymin>
			<xmax>243</xmax>
			<ymax>278</ymax>
		</bndbox>
	</object>
</annotation>
```

Ahora continuaremos la clase en paralelo explicando tanto las transformaciones pertinentes al proyecto del curso y al proyecto
personal.

## 3.6 Unificando tu Base de datos



### Proyecto de la clase:
Recordemos que tenemos una base de datos limpia que contiene: 39 elementos para train. Después de haber etiquetado todos 
los datos manualmente en: https://www.linkedai.co/ hemos descargado las etiquetas en formato `json` quedando de la siguiente manera:

![12.png](imgs%2F3%2F12.png)

Cada etiqueta luce de la siguiente manera:

```json
{
  "id":"4e59f0ea-c47d-42c1-a201-b3f158c0a93a7ab94317-b86c-45bd-924b-b2016321f41d",
  "image":"fc15988c9a.jpg",
  "width":"1300",
  "height":"953",
  "classification":[],
  "tags":
       [
         {
         "parent":null
         },
         "color":"#f10c27",
         "pos":
              {
                "x":63.03030303030303,
                "y":436.7099567099567,
                "w":746.2337662337662,
                "h":427.7056277056277
              },
         "classes":[],
         "name":"motos",
         "id":"d3de93b5-309a-45d2-b072-542118fe66c5",
         "type":"bounding_box"
       ]
}
```
El archivo completo contiene todas las etiquetas de la base de datos. Pero sería mucho más cómodo para nosotros tener un
único archivo en formato CSV que contenga todas las etiquetas de las imágenes, de esta forma y junto con la base de datos de
imágenes sería muy fácil convertirlas a formato `TF-RECORD` un formato optimizado para entrenar modelos de `Deep Learning`
Desarrollado por `Google`. Así que vamos a convertir este archivo en formato `json` a `csv` y prepararlo para convertirlo a
`tf-record` más adelante.

Ahora tenemos el código de conversión: [json2csv.py](scripts%2F3%2F6%20Unificando%20tu%20Base%20de%20datos%2Fjson2csv.py)

```python
import pandas as pd
import json


def json2csv(labelmap, json_dir, csv_dir=None):
    if csv_dir is None:
        csv_dir = json_dir.replace(".json", ".csv")
    data = json.load(open(json_dir))
    csv_list = []
    for classification in data:
        width, height = classification['width'], classification['height']
        image = classification['image']
        for item in classification['tags']:
            name = item['name']
            id_name = labelmap[name]
            xmin = int(item['pos']['x'])
            ymin = int(item['pos']['y'])
            # Hacemos esto porque el formato PASCAL VOC NO utiliza x, w,
            # por el contrario usa xmin y xmax siendo xmax = xmin+w
            xmax = int(item['pos']['x'] + item['pos']['w'])
            # Caso similar con ymax, es y + h
            ymax = int(item['pos']['y'] + item['pos']['h'])
            value = (image, width, height, xmin, ymin, xmax, ymax, name, id_name)
            csv_list.append(value)

    column_name = ['filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'class_id']
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    print(csv_df)
    csv_df.to_csv(csv_dir, index=False)


if __name__ == '__main__':
    root = "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/3/3 Distribución de datos/clean"
    paths = {"train": f"{root}/train.json", "test": f"{root}/test.json"}
    lb_map = {"carro": 1, "motos": 2}
    # Converting train.json 2 train.csv
    json2csv(lb_map, paths["train"])
    # Converting test.json 2 test.csv
    json2csv(lb_map, paths["test"])
```

Puntos relevantes de este código es que para correrlo es necesario tener un mini `labelmap` como diccionario indicando
cuál id debe tener el nombre de cada clase.

```
lb_map = {"carro": 1, "motos": 2}
```

Adicionalmente, el formato de etiquetado de la página web regresaba `x, y, w, h` y nosotros trabajamos con el formato 
`xmin, ymin, xmax, ymax` por eso tuvimos que hacer una ligera conversión en estos datos.

Resultado esperado:

```commandline
          filename width height  xmin  ymin  xmax  ymax  class  class_id
0   fc15988c9a.jpg  1300    953    63   436   809   864  motos         2
1   0c535fcdfa.jpg  1138    635   578   237   724   440  motos         2
2   0c535fcdfa.jpg  1138    635   519   257   555   316  motos         2
3   0c535fcdfa.jpg  1138    635   498   255   515   292  motos         2
4   2b37a718bc.jpg  1024    768   450   346   933   653  carro         1
..             ...   ...    ...   ...   ...   ...   ...    ...       ...
68  ea2cb5917b.jpg   800    599   443   119   798   595  carro         1
69  ea8ce9da21.jpg   328    418    54   163   266   380  motos         2
70  f8a2cc7848.jpg   450    363    29   198   395   362  motos         2
71  f74248c57f.jpg   960    640   361   371   397   444  motos         2
72  f74248c57f.jpg   960    640   424   355   456   417  motos         2
```

### Proyecto personal:

El objetivo es el mismo que el de la clase, unir y transformar las etiquetas XML a CSV para que sea más sencillo operar con ellas.
Puedes acceder al código en [xml_to_csv.py](scripts%2Fproject%2Fcore%2Fxml_to_csv.py). Como vimos en la sección anterior, `labelImg.py` regresa
las etiquetas en formato `XML` una por una, sin embargo, sería muchísimo más cómodo tener todas las etiquetas juntas, en un sólo
lugar en formato `CSV` esto nos podría facilitar la tarea de convertir este archivo `CSV` a un formato `TF-RECORD` que está
completamente optimizado por `Google` para entrenar modelos de `Deep learning`. ASí que en este paso vamos a empezar por convertir nuestras
etiquetas `XML2CSV`

```python
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os


def generate_dataframe(dataset_dir, name2id):

    dataset = {"filename": [], "width": [], "height": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "class": [],
               "class_id": []}

    for item in glob.glob(os.path.join(dataset_dir, "*.xml")):
        tree = ET.parse(item)
        filename, width, height, xmin, ymin, xmax, ymax, name = None, None, None, None, None, None, None, None
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
            elif 'width' in elem.tag:
                width = int(elem.text)
            elif 'height' in elem.tag:
                height = int(elem.text)
            elif 'name' in elem.tag:
                name = elem.text
            elif 'xmin' in elem.tag:
                xmin = int(elem.text)
            elif 'ymin' in elem.tag:
                ymin = int(elem.text)
            elif 'xmax' in elem.tag:
                xmax = int(elem.text)
            elif 'ymax' in elem.tag:
                ymax = int(elem.text)

                dataset['filename'].append(filename)
                dataset['width'].append(width)
                dataset['height'].append(height)
                dataset['xmin'].append(xmin)
                dataset['ymin'].append(ymin)
                dataset['xmax'].append(xmax)
                dataset['ymax'].append(ymax)
                dataset["class"].append(name)
                dataset["class_id"].append(name2id[name])
    df = pd.DataFrame(dataset)
    return df


if __name__ == '__main__':
    labelmap = {"d1": 1, "d2": 2, "d3": 3, "d4": 4, "d5": 5}
    root = "/media/ichcanziho/Data/datos/Deep Learning/7 Object Detection/project/dataset"
    paths = {"train_labels": f"{root}/train_labels",
             "test_labels": f"{root}/test_labels",
             "train": f"{root}/train",
             "test": f"{root}/test"}

    train = generate_dataframe(paths["train_labels"], labelmap)
    print(train)
    train.to_csv(f"{paths['train']}/train_labels.csv", index=False)
    test = generate_dataframe(paths["test_labels"], labelmap)
    print(test)
    test.to_csv(f"{paths['test']}/train_labels.csv", index=False)

```


## 3.7 Requisitos Locales para transformar CSV a TF Record

Con nuestro `odenv` activado, debemos instalar las dependencias de `Object Detection` y `TensorFlow` Necesarias:

```commandline
git clone https://github.com/tensorflow/models.git
cd models/
git checkout 58d19c67e1d30d905dd5c6e5092348658fed80af
pip install -q Cython contextlib2 pillow lxml matplotlib
pip install pycocotools
cd research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
pip install tensorflow==2.8.0
pip uninstall protobuf -y
pip install protobuf==3.20.3
```

Este código te va a permitir ejecutar de forma `LOCAL` la creación de `csv2tfrecord` sin embargo, por problemas de compatibilidad
no he logrado poder entrenar el modelo 100% de forma `LOCAL`.

ESTO ME LLEVA A LA NECESIDAD de que la siguiente sección será una combinación hibrida entre utilizar Archivos Python pensados
para ejecutarse de forma local y un `NoteBook` [Entrenamiento_Base.ipynb](scripts%2Fproject%2Fcore%2FEntrenamiento_Base.ipynb) 
que está pensado para ejecutarse desde `Google Colab` esto con la finalidad de que el entrenamiento del modelo se haga en la nube
y NO de forma `LOCAL`. 

La explicación de esta sección se ha movido a: [4.3 Creación de TF Records](#43-creación-de-tf-records)

# 4 Entrena, testea y optimiza tus modelos

Para el entrenamiento de esta sección vamos a utilizar Google Colab, puedes encontrar el notebook 
[Entrenamiento_Base.ipynb](scripts%2Fproject%2Fcore%2FEntrenamiento_Base.ipynb). El mismo funciona como
un orquestador de funciones de python. Cada una de las funciones descritas será explicada a detalle 
en las siguientes clases. Sin embargo, el notebook gestiona el uso de los siguientes archivos de python:

- [config.py](scripts%2Fproject%2Fcore%2Fconfig.py)
- [csv_to_tf_record.py](scripts%2Fproject%2Fcore%2Fcsv_to_tf_record.py)
- [json2pbtxt.py](scripts%2Fproject%2Fcore%2Fjson2pbtxt.py)
- [update_config_file.py](scripts%2Fproject%2Fcore%2Fupdate_config_file.py)

Cada archivo tiene su propia e individual función y será explicada a continuación.

## 4.1 Configuration File

El primer paso que nos va a ayudar a entrenar nuestro modelo de `Object Detection` es crear un archivo
de configuración en donde tengamos todas en variables todas las rutas y nombres de directorios que se van a utilizar
durante el proceso de entrenamiento y exportación del modelo.

Este es el documento: [config.py](scripts%2Fproject%2Fcore%2Fconfig.py) que contiene los parámetros ajustables para el modelo

Las siguientes variables son `editables` a comodidad del usuario y proyecto:

```python
# cuál es el nombre de la carpeta que se utilizará para crear mi modelo custom
CUSTOM_MODEL_NAME = 'vehicle_detection'
# de qué dirección voy a descargar el modelo pre-entrenado
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
# ¿Cuántas clases tiene mi problema de detección de objetos?
N_CLASSES = 2
# Cuando empiece a entrenar, de cuánto será el batch-size?
BATCHSIZE = 4
# De qué clase es el tipo de problema que estoy resolviendo?
PTYPE = "detection"
# Una vez que el modelo haya sido entrenado en qué carpeta guardare los resultados para hacer inferencia?
OUTPUT_DIRECTORY = "vehicle_fine_tuned"
# Número de pasos para el entrenamiento
NUM_STEPS = 5000
# --------------------------------------------------------------
```

Los nombres de las variables son bastante autodescriptivas. Lo más importante es definir cuál será el modelo pre-entrenado que usaremos
como base para aplicar nuestro `fine-tuning process`. Para eso podemos dirigirnos a: [TF2 detection zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) en
donde encontraremos una amplia gama de modelos pre-entrenados, cada uno con diferente velocidad de inferencia y precisión. En cada uno cabe
la responsabilidad de ponderar que es más importante la precisión o la velocidad. Así como la resolución con la que desea trabajar las imágenes.

![1.png](imgs%2F4%2F1.png)

Para este proyecto vamos a utilizar: `ssd_mobilenet_v2_fpnlite_320x320` básicamente porque para un ejemplo demostrativo nos es
muy importante que el modelo sea perfecto clasificando o detectando el objeto, basta con conocer la metodología y específicamente
este modelo es bastante ligero y rápido de entrenar.

Adicionalmente, en nuestro archivo de configuración debemos definir un par de rutas más. Pese a que cada ruta está comentada
y dice el funcionamiento de la misma es `MUY IMPORTANTE` que te detengas a leer cada una con detenimiento para que más
adelante entiendas profundamente cuando y para qué se utilizan cada una de las siguientes rutas.

```python
# Automáticamente, obtiene el nombre del modelo pre-entrenado con base en su URL de descarga
PRETRAINED_MODEL_NAME = PRETRAINED_MODEL_URL.split("/")[-1].split(".")[0]
# Ruta por defecto en dónde se va a encontrar el archivo de configuración de entrenamiento del modelo
PIPELINE_CONFIG = CUSTOM_MODEL_NAME+'/pipeline.config'
# Los modelos pre-entrenados vienen con diferentes puntos de control, nosotros podemos decidir desde que punto de
# control podemos reentrenar el modelo
CHECKPOINT = PRETRAINED_MODEL_NAME+"/checkpoint/ckpt-0"
# Cuál es la dirección del archivo que contiene las clases del modelo (está en formato json porque es bastante fácil
# de crear a mano)
JSON_LABEL_MAP_NAME = 'label_map.json'
# Ruta del `label_map` pero en su formato final listo para ser utilizado por `tensorflow`
LABEL_MAP_NAME = CUSTOM_MODEL_NAME + "/label_map.pbtxt"
# Ruta donde se encuentra la carpeta que contiene las imágenes para crear el tf-record de entrenamiento
TRAIN_IMAGES_FOLDER = "dataset"
# Archivo que contiene las etiquetas de las imágenes de entrenamiento en formato csv
TRAIN_IMAGES_DATASET_FILE = "dataset/train.csv"
# Ruta donde se encuentra la carpeta que contiene las imágenes para crear el tf-record de test
TEST_IMAGES_FOLDER = "dataset"
# Archivo que contiene las etiquetas de las imágenes de test en formato csv
TEST_IMAGES_DATASET_FILE = "dataset/test.csv"
# Ruta de dónde se guardará el archivo de train en formato tf-record
TRAIN_RECORD_FILE = f"{CUSTOM_MODEL_NAME}/train.record"
# Ruta de dónde se guardará el archivo de test en formato tf-record
TEST_RECORD_FILE = f"{CUSTOM_MODEL_NAME}/test.record"
```

Con todo lo anterior ya definido en nuestro archivo de configuración entonces podemos seguir adelante con nuestro proyecto.

## 4.2 Creación de LabelMap

El `LabelMap` es un archivo indispensable que necesita nuestro modelo de `Object Detection` para entender ¿Cuántas? Y ¿Cuáles? 
Son las clases que debe reconocer el mismo. Contiene el nombre propio de la clase y un ID numérico que la distingue. Es
`INDISPENSABLE` que el `LabelMap` contenga exactamente el mismo nombre que tienen las etiquetas de la base de datos. 

Para este proceso vamos a hacer un simple código que convierta un `LabelMap` en formato `json` a formato `pbtxt` que es 
el necesario para entrenar al modelo utilizando `tensorflow`. Partamos del siguiente archivo: [label_map.json](scripts%2Fproject%2Fcore%2Flabel_map.json)

```json
[
  {"name": "carro", "id": 1},
  {"name": "motos", "id": 2}
]
```
Podemos observar que es un archivo `json` muy simple de entender, una lista con 2 elementos, cada uno definiendo el nombre
de una clase y su correspondiente id.

Ahora vamos a crear el archivo que nos permita convertir de formato `json` a `pbtxt` como vamos a utilizar `Google Colab` como
orquestador de nuestros códigos python, vamos a hacer cada unos de ellos como un `CLI (Command Line Interface)` eso significa
que estos códigos deben ser ejecutados desde `consola` con ciertos parámetros establecidos.

```python
import argparse
import json

parser = argparse.ArgumentParser(description="Json to pbtxt file converter")

parser.add_argument("-j",
                    "--json_file",
                    help="json file directory",
                    type=str)
parser.add_argument("-p",
                    "--pbtxt_file",
                    help="pbtxt file directory. Defaults to the same name and directory as json_file.",
                    type=str)

args = parser.parse_args()

if args.pbtxt_file is None:
    args.pbtxt_file = args.json_file.replace(".json", ".pbtxt")


def convert_json_to_pbtxt(json_file, pbtxt_file):

    with open(json_file, "r") as data:
        json_data = json.load(data)

    with open(pbtxt_file, "w") as f:
        for label in json_data:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


if __name__ == '__main__':
    print("Input parameters are:")
    print(args.json_file, args.pbtxt_file)
    convert_json_to_pbtxt(args.json_file, args.pbtxt_file)
    print("Done")

```

Como podemos observar el código necesita solamente 2 parámetros para funcionar: 

- **--json_file**: Contiene la ruta al archivo de origen en formato `json`
- **--pbtxt_file**: Contiene la ruta al archivo de salida en formato `pbtxt` por defecto utilizará la misma ruta y nombre que el archivo de entrada.

Ejecutamos el código desde terminal: 
```bash
python json2pbtxt.py --json_file label_map.json
```
Respuesta esperada:
```commandline
Input parameters are:
label_map.json label_map.pbtxt
Done
```
Obtenemos el siguiente archivo a la salida: [label_map.pbtxt](scripts%2Fproject%2Fcore%2Flabel_map.pbtxt)
```commandline
item { 
	name:'carro'
	id:1
}
item { 
	name:'motos'
	id:2
}
```
Es un archivo bastante similar al `json` pero con este flujo, no tenemos que preocuparnos por crear el archivo basta con
tener las clases en el formato cómodo y estandarizado que ya conocemos. 

## 4.3 Creación de TF Records

Ahora que ya tenemos el `LabelMap` podemos continuar con la creación de nuestros `TF-Records` de entrenamiento y test. Pero a
todo esto ¿Qué es un `TF-Record`?

> Un archivo TFRecord es un formato de almacenamiento utilizado en TensorFlow, un popular framework de aprendizaje automático desarrollado por Google. Los archivos TFRecord están diseñados para almacenar grandes cantidades de datos en un formato eficiente y optimizado para el rendimiento.

> Un archivo TFRecord almacena datos en forma de registros secuenciales. Cada registro consiste en una secuencia de pares clave-valor, donde la clave es una cadena que identifica el tipo de datos almacenados y el valor es la representación codificada de los datos en sí. La codificación de los datos se realiza utilizando el protocolo de serialización de Google (Protocol Buffers), que es un formato binario eficiente y compatible con múltiples lenguajes de programación.

> Los archivos TFRecord son especialmente útiles para procesar grandes conjuntos de datos que se utilizan en el entrenamiento de modelos de aprendizaje automático. Al almacenar los datos en archivos TFRecord, se puede lograr una lectura eficiente y secuencial de los datos durante el entrenamiento, lo que ayuda a optimizar el rendimiento y la velocidad de procesamiento.

Básicamente, son archivos muy eficientes que nos permiten almacenar tanto las imágenes como las etiquetas de las mismas. Utilizan `ProtoBuf` para esta tarea
por lo mismo es bastante necesario tener instalado una versión compatible. (Esto lo veremos en la sección de entrenamiento)

Recordemos que el código [csv_to_tf_record.py](scripts%2Fproject%2Fcore%2Fcsv_to_tf_record.py) está pensado como un `CLI` esto facilitará
mucho el proceso de utilización en `Google Colab`, adicionalmente en nuestro archivo [config.py](scripts%2Fproject%2Fcore%2Fconfig.py) ya tenemos todas las variables
que contienen las rutas que necesitamos para correr el proyecto.

```python
import os
import pandas as pd
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
import argparse

parser = argparse.ArgumentParser(description="CSV To TFRECORD file converter")

parser.add_argument("-c",
                    "--csv_file",
                    help="csv file directory. Format '.csv'",
                    type=str)
parser.add_argument("-i",
                    "--images_dir",
                    help="path where the images are stored",
                    type=str)

parser.add_argument("-o",
                    "--output_file",
                    help="tf record file directory. Format '.record'",
                    type=str)

args = parser.parse_args()


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(row["class_id"])  # Esta es la única alteración al código original de TensorFlow, en mi caso el
                                         # "row" ya tenía el "class_id" que contiene el número de la clase "class"

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(csv_file, images_dir, output_file):

    writer = tf.io.TFRecordWriter(output_file)
    path = os.path.join(images_dir)
    examples = pd.read_csv(csv_file)
    print(examples.head())
    print("Starting :)")
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file at: {}'.format(output_file))


if __name__ == '__main__':
    c_file = args.csv_file
    i_dir = args.images_dir
    o_file = args.output_file
    main(csv_file=c_file, images_dir=i_dir, output_file=o_file)
```

## 4.4 Actualización de Pipeline Config

Como estamos trabajando con modelos `pre-entrenados` estos tienen un archivo configurable que permite al modelo ser reentrenado
con ciertas alteraciones. A este documento de configuración se le conoce como `pipeline.config`.

> El archivo `pipeline.config` contiene la configuración necesaria para entrenar, evaluar o utilizar un modelo de detección de objetos. Proporciona información sobre la arquitectura del modelo, los hiperparámetros de entrenamiento, los directorios de datos, las rutas de archivos, las etiquetas de clase y otros detalles relacionados.

Algunas de las principales secciones y configuraciones presentes en un archivo "pipeline.config" son las siguientes:

- `Configuración del modelo:` Se especifica el tipo de arquitectura de red utilizada, como Faster R-CNN, SSD, EfficientDet, etc. Además, se definen aspectos como el número de clases, el tamaño de las imágenes de entrada y las características específicas del modelo.

- `Configuración del entrenamiento:` Aquí se establecen los hiperparámetros de entrenamiento, como la tasa de aprendizaje, el número de pasos de entrenamiento, el optimizador utilizado, los pesos de regularización, entre otros.

- `Configuración del conjunto de datos:` Se especifican los directorios y archivos que contienen los datos de entrenamiento, validación y prueba. También se definen las etiquetas de clase y los archivos de mapeo entre identificadores y nombres de clase.

- `Configuración de salida:` Se establecen los directorios de salida donde se guardarán los modelos entrenados, los registros de eventos para el seguimiento del entrenamiento y los archivos de evaluación.

Pese a que te recomiendo AMPLIAMENTE leer a profundad el `pipeline.config` de tu modelo pre-entrenado para que tengas una mejor
noción de que contiene, a continuación comparto un código que permite su edición fácil con los requisitos mínimos (y un poco más)
para poder empezar el proceso de entrenamiento.

> ## Nota:
> Al descargar un modelo este trae un pipeline.config con algunos parámetros `to_be_configured` puesto que estan preparados 
> para ser editables por el usuario final. Ejemplo:
> 
>     train_input_reader {
>        label_map_path: "PATH_TO_BE_CONFIGURED"
>        tf_record_input_reader {
>        input_path: "PATH_TO_BE_CONFIGURED"
>        }
>     }

Este es el último paso que necesitamos programar en código antes de ponernos manos a la obra a entrenar el modelo.
Si bien el formato de `pipeline.config` es básicamente un `json` la realidad es que es un poco más complicado que eso, y al ser 
un archivo tan importante, pues es en este donde recae todo lo necesario para configurar el proceso de entrenamiento del 
modelo. Es por ello que la biblioteca `object_detection` cuenta con un par de métodos que nos ayudan a editar este documento:

- config_util
- pipeline_pb2
- text_format (este es de google)



```python
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import argparse


parser = argparse.ArgumentParser(description="Pipeline config updater")

parser.add_argument("--input_config",
                    help="pipeline configuration file",
                    type=str)
parser.add_argument("--n_classes",
                    help="number of classes",
                    type=int)
parser.add_argument("--bs",
                    help="batch size for training",
                    type=int)
parser.add_argument("--checkpoint",
                    help="checkpoint file to start training",
                    type=str)
parser.add_argument("--ptype",
                    help="problem type {object detection, classification, segmentation}",
                    type=str)
parser.add_argument("--label_map",
                    help="label map file",
                    type=str)

parser.add_argument("--train_record",
                    help="train record file in tf_record format",
                    type=str)

parser.add_argument("--test_record",
                    help="test record file in tf_record format",
                    type=str)

args = parser.parse_args()


def update_config(input_config, n_classes, batch_size, checkpoint, checkpoint_type, label_map, train_record, test_record):
    # Obtenemos la configuración del archivo pipeline
    print("start")
    config = config_util.get_configs_from_pipeline_file(input_config)
    # Creamos una variable proto_str para poder modificar las variables del archivo pbtxt
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(input_config, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    # Cantidad de clases del modelo
    pipeline_config.model.ssd.num_classes = n_classes
    # El tamaño del batch size, entre más grande más costo computacional va a necesitar en el proceso de entrenamiento,
    # pero a su vez entrenará más rapido.
    pipeline_config.train_config.batch_size = batch_size
    # Donde almacenaremos los resultados del entrenamiento
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint
    # Qué tipo de detección aplicaremos (Object detection)
    pipeline_config.train_config.fine_tune_checkpoint_type = checkpoint_type
    # Dirección del label map
    pipeline_config.train_input_reader.label_map_path = label_map
    # Dirección del train TFRecord
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = train_record
    # Dirección del label map
    pipeline_config.eval_input_reader[0].label_map_path = label_map
    # Dirección del test TFRecord
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = test_record
    # Almacenamos nuestro archivo final
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(input_config, "wb") as f:
        f.write(config_text)
    print("done")


if __name__ == '__main__':

    c_file = args.input_config
    n_c = args.n_classes
    bs = args.bs
    ckp = args.checkpoint
    ckp_t = args.ptype
    lb_map = args.label_map
    tra_re = args.train_record
    tst_re = args.test_record
    update_config(c_file, n_c, bs, ckp, ckp_t, lb_map, tra_re, tst_re)
```

Los parámetros que recibe este `CLI` son los mismos que va a configurar en el `pipeline.config` los cuáles son los siguientes:

- **input_config:** es el archivo base `pipeline.config` que va a ser editado
- **n_classes:** corresponde con el número de clases que contiene nuestro problema de `object detection`
- **bs:** es el batch-size
- **checkpoint:** se debe indicar desde que checkpoint se debe continuar el aprendizaje
- **ptype:** es el tipo de problema, en este caso es `detection`
- **labelmap:** contiene el archivo `pbtxt` con el nombre de las clases y su `id`
- **train_record:** archivo `record` de entrenamiento
- **test_record:** archivo `record` de test.

Una vez que se ejecuta el siguiente comando:
```python
command = "python update_config_file.py --input_config={} --n_classes={} --bs={} --checkpoint={} --ptype={} --label_map={} --train_record={} --test_record={}".format(
    PIPELINE_CONFIG, N_CLASSES, BATCHSIZE, CHECKPOINT, PTYPE, LABEL_MAP_NAME, TRAIN_RECORD_FILE, TEST_RECORD_FILE)
```
Valor esperado:
```commandline
python update_config_file.py --input_config=vehicle_detection/pipeline.config --n_classes=2 --bs=4 --checkpoint=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0 --ptype=detection --label_map=vehicle_detection/label_map.pbtxt --train_record=vehicle_detection/train.record --test_record=vehicle_detection/test.record
```
Al ejecutar dicho commando se actualizará el archivo `pipeline.config` y por fin estaremos listos para entrenar el modelo.

## 4.5 Entrenamiento

Si has llegado hasta aquí sin problemas entonces ahora debes tener los siguientes archivos listos para entrenar tu modelo:

- labelmap.pbtxt
- test.record
- train.record
- pipeline.config

Si tienes todos los archivos generados, entonces el proceso de entrenamiento es sumamente sencillo, porque cuando descargamos
el repositorio de `github` de `object detection` en la siguiente ruta se encuentra una archivo de python de entrenamiento:

```commandline
models/research/object_detection/model_main_tf2.py
```

La cual recibe `3` parámetros de entrada para poder ejecutar el entrenamiento del modelo:

- **pipeline_config:** el archivo `pipeline.config` contiene TODA la información necesaria para entrenar el modelo
- **model_dir:** el directorio que se creará para almacenar los checkpoint que se vayan generando.
- **num_train_steps:** el número de pasos para entrenar el modelo.

Sin embargo, como hemos estado hablando a lo largo de esta sección vamos a utilizar todos los códigos que hemos desarrollado
hasta este momento y vamos a subir toda esta información a `Google Colab` para entrenar el modelo directamente desde la nube.

[Entrenamiento_Base.ipynb](scripts%2Fproject%2Fcore%2FEntrenamiento_Base.ipynb)

Lo que necesitas es lo siguiente:

- Debes crear una carpeta comprimida en zip llamada: `dataset.zip` que contenga las imágenes de entrenamiento y test de tu base de datos, 
dentro de esa carpeta también deben estar los dataset etiquetados en formato csv, en este caso: `train_labels.csv` y `test_labels.csv`. Puedes encontrar un ejemplo
del dataset etiquetado en [dataset.zip](scripts%2Fproject%2Fcore%2Fdataset.zip).
- Debes subir el archivo de configuración: [config.py](scripts%2Fproject%2Fcore%2Fconfig.py)
- Debes subir el archivo de conversión: [csv_to_tf_record.py](scripts%2Fproject%2Fcore%2Fcsv_to_tf_record.py)
- Debes subir el archivo de: [json2pbtxt.py](scripts%2Fproject%2Fcore%2Fjson2pbtxt.py)
- Debes subir el archivo: [label_map.json](scripts%2Fproject%2Fcore%2Flabel_map.json)
- Y finalmente, debes subir el archivo: [update_config_file.py](scripts%2Fproject%2Fcore%2Fupdate_config_file.py)

Una vez que hayas subido TODOS los documentos necesarios a colab, puedes empezar a ejecutar cada una de las celdas.
![3.png](imgs%2F4%2F3.png)

Es en este momento que se entiende por qué `Google Colab` funciona como un `Orquestador` para TODO el código que hemos estado
desarrollando manualmente en formato de `CLI`.

1: Creando LabelMap

![4.png](imgs%2F4%2F4.png)

2: Creando TF-Records

![5.png](imgs%2F4%2F5.png)

3: Descargamos el modelo pre-entrenado:

![6.png](imgs%2F4%2F6.png)

4: Actualizamos el pipeline:

![7.png](imgs%2F4%2F7.png)

Y ahora podemos observar como ya tenemos en nuestra carpeta `vehicle_detection` TODO lo necesario para entrenar el modelo.

![8.png](imgs%2F4%2F8.png)

5: Finalmente, entrenamos el modelo:

```commandline
!python /content/models/research/object_detection/model_main_tf2.py \
--pipeline_config_path={PIPELINE_CONFIG} \
--model_dir={CUSTOM_MODEL_NAME} \
--num_train_steps={NUM_STEPS}
```



## 4.6 TensorBoard

Tras correr el entrenamiento en `Google Colab` ejecutamos una simple línea de código y podemos visualizar toda la información
de `TensorBoard` directamente en `Colab`:

![2.png](imgs%2F4%2F2.png)

## 4.7 Exportando Modelo

Es tan sencillo como utilizar otro archivo de python de `object detection` que nos permita guardar el modelo para después
comprimirlo en formato `zip` descargarlo y posteriormente cargarlo para hacer inferencia con él:

```commandline
%%capture
!python /content/models/research/object_detection/exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path {PIPELINE_CONFIG} \
--trained_checkpoint_dir {CUSTOM_MODEL_NAME} \
--output_directory {OUTPUT_DIRECTORY}

!cp {LABEL_MAP_NAME} {OUTPUT_DIRECTORY+'/label_map.pbtxt'}

%%capture
!zip -r fine_tuned_model.zip {OUTPUT_DIRECTORY}
```

```python
from google.colab import files
files.download('fine_tuned_model.zip')
```

![9.png](imgs%2F4%2F9.png)

## 4.8 Función de inferencia

El código de `COLAB` para hacer inferencia es el siguiente:

```python
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from cv2 import cvtColor, COLOR_BGR2RGB

def classify(image_path):
  # La convertimos a array
  image_np = np.array(Image.open(image_path))

  # La convertimos a tensor y la agregamos una dimensión para que pueda leerla nuestro modelo
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  ## Revisar que la ruta sea la misma de la carpeta.
  PATH_TO_MODEL_DIR = f"/content/{OUTPUT_DIRECTORY}"
  PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

  detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)
  # Cargamos el label map para utilizarlo.
  label_map_pbtxt_fname = f"/content/{OUTPUT_DIRECTORY}/label_map.pbtxt"
  category_index = label_map_util.create_category_index_from_labelmap(label_map_pbtxt_fname)


  # Realizamos la detección del objeto
  detections = detect_fn(input_tensor)

  # Analizamos cuántas detecciones se obtuvieron
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0,:num_detections].numpy() for key, value in detections.items()}

  detections['num_detections'] = num_detections

  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  # Tomamos una imagen y la copiamos para dibujar los bounding box
  image_np_with_detections = image_np.copy()

  # Utilizamos la libreria de obejct detection para visualizar le bounding box y la clasificación
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      max_boxes_to_draw=200,
      min_score_thresh=0.30,
      use_normalized_coordinates = True
  )

  return image_np_with_detections

# Importamos la imagen
image_path = "coche.jpg"

image_np_with_detections = classify(image_path)

cv2_imshow(cvtColor(image_np_with_detections, COLOR_BGR2RGB))
```

Y produce el siguiente resultado:

![10.png](imgs%2F4%2F10.png)

Sin embargo, el código `LOCAL` que programe para poder hacer `inferencia` es diferente y lo puedes encontrar en: [predict.py](scripts%2Fproject%2Fpredict.py)

el resultado de forma `LOCAL` que se produce al hacer inferencia sobre la misma imagen base es:

![annotated.png](scripts%2Fproject%2Fannotated.png)

Bastante similar pero con menos score en la clase `carro`.

## 4.9 Re-entrenamiento del modelo para obtener mejores resultados

Básicamente, esta clase solamente se comentó que dado un nuevo conjunto de datos, debíamos rehacer todo 
el procedimiento y comparar los nuevos resultados.

- Descargar base de datos
- Limpiar manualmente
- Distribuir en train-test
- Etiquetar datos
- Descargar JSON
- Convertir JSON - CSV - TF RECORD
- Entrenar el modelo



## 4.10 Seguimiento de centroides con OpenCV

Una vez que se ha detectado un objeto, a dicho objeto se le calcula su centroide. A cada objeto se le asigna un número,
cuando los objetos se mueven se busca volver a calcular el centroide de los objetos movidos y se utiliza la distancia 
euclideana para determinar el id del objeto. (se entiende que el centroide con menor distancia al punto anterior es el mismo objeto)

![11.png](imgs%2F4%2F11.png)

Si un nuevo elemento aparece en escena a este elemento se le asigna un nuevo ID. Si después de un determinado tiempo ya no soy
capaz de detectar el objeto al que perteneció un centroide eso quiere decir que el elemento salió de escena y puedo proceder a
eliminar su centroide. 


![12.png](imgs%2F4%2F12.png)

Para esta parte de la clase vamos a utilizar 2 archivos de python auxiliares:

- [trackableobject.py](scripts%2F4%2Ftrackableobject.py)
- [centroidtracker.py](scripts%2F4%2Fcentroidtracker.py)


Empecemos con: `trackableobject`:

```python
class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
```

El código en sí mismo es solo una estructura de dato que me permite almacenar el ID del objeto su centroide y adicionalmente
una bandera llamada `counted` para monitorear si ya he contado el objeto cuando paso de un punto A a uno B esto con el objetivo
de que un mismo objeto NO me haga contar 2 veces que paso por una zona.

El siguiente código que explicaré es: `CentroidTracker` el cuál necesita bastante atención para realmente entender su funcionamiento.
Afortunadamente, el código viene bastante comentado y eso ayuda a entenderlo mejor:

```python
# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects
```

Dividamos la explicación del código en sus componentes principales:

1. `Importación de paquetes:` En las primeras líneas, se importan los paquetes necesarios para el funcionamiento del código. "scipy.spatial.distance" se utiliza para calcular la distancia entre los centroides, "collections.OrderedDict" se utiliza para mantener el orden de los objetos y los centroides, y "numpy" se utiliza para realizar operaciones numéricas.

2. `Clase CentroidTracker:` Esta clase contiene métodos para registrar, desregistrar y actualizar los objetos rastreados.

3. `Método init:` Este método se llama cuando se crea una instancia de la clase CentroidTracker. Establece los valores iniciales, como el número máximo de cuadros desaparecidos permitidos para un objeto y la distancia máxima para asociar un objeto.

4. `Método register:` Registra un objeto asignándole un ID único y almacenando su centroide en un diccionario.

5. `Método deregister:` Elimina un objeto del seguimiento a través de su ID.

6. `Método update:` Actualiza el seguimiento de objetos en cada cuadro o imagen nueva. Toma una lista de rectángulos delimitadores de los objetos detectados en el cuadro actual.

   - Si no hay rectángulos de entrada, se incrementa el contador de cuadros desaparecidos para cada objeto existente y, si excede un límite, se elimina el objeto del seguimiento.

   - Si hay rectángulos de entrada, se calcula el centroide para cada uno y se almacenan en una matriz.

   - Si no hay objetos en seguimiento, se registran todos los centroides de entrada como nuevos objetos.

   - Si ya hay objetos en seguimiento, se calcula la distancia entre cada par de centroides de seguimiento y los centroides de entrada. Luego, se busca la mejor coincidencia entre ellos basándose en la distancia.

   - Se itera sobre las filas y columnas de la matriz de distancias, y se selecciona la menor distancia disponible que aún no ha sido utilizada.

   - Si la distancia es mayor que la distancia máxima establecida, los centroides no se asocian al mismo objeto.

   - Si la distancia es menor o igual a la distancia máxima, el centroide se asocia al objeto correspondiente, se restablece el contador de cuadros desaparecidos y se marca la fila y la columna como utilizadas.

   - Se revisan las filas y columnas no utilizadas para ver si algunos objetos han desaparecido.

   - Si hay más centroides de seguimiento que centroides de entrada, se incrementa el contador de cuadros desaparecidos para los centroides no utilizados. Si el contador excede el límite, se eliminan los objetos del seguimiento.

   - Si hay más centroides de entrada que centroides de seguimiento, se registran los nuevos centroides como objetos.

   - Finalmente, se devuelve el diccionario de objetos rastreados.

El código general de esta sección lo puedes encontrar en: [Entrenamiento_Base.ipynb](scripts%2Fproject%2Fcore%2FEntrenamiento_Base.ipynb)

Sin embargo, aquí vamos a explicar el código utilizado en la clase conforme se fue explicando.

Empezamos como siempre importando las bibliotecas necesarias:

```python
import numpy as np
import imutils
import time
import dlib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import tensorflow as tf
```

Ahora vamos a explicar el código de la clase:

```python
PATH_TO_SAVE_MODEL = "/content/fine_tuned_model/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Ruta del video (Se debe cargar de manera manual)
PATH_VIDEO = "/content/test_video.mp4"

# Ruta del video en donde almacenaremos los resultados
PATH_OUTPUT = "/content/video_out.mp4"

# Cuántos frames vamos a saltarnos (Durante estos frames nuestro algoritmo de seguimiento funciona)
SKIP_FPS = 30

# Cuál será el umbral mínimo par que se considere una detección
TRESHOLD = 0.5

# Cargamos el video
vs = cv2.VideoCapture(PATH_VIDEO)

# Inicializamos el writer para poder guardar el video
writer = None

# Definimos ancho y alto
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializamos la clase centroid tracker con dos variable fundamentales
# maxDissapared (Si pasa ese tiempo y no se detecta más el centroide lo elimina)
# Si la distancia es mayor a maxDistance no lo podra asociar como si fuera el mismo objecto.
ct = CentroidTracker(maxDisappeared= 40, maxDistance = 50)

# Inicializamos variables principales
trackers = []
trackableObjects = {}

totalFrame = 0
# variables para contar la cantidad de coches que suben y bajan
totalDown = 0
totalUp = 0

# Esta variable sirve para definir la orientación de los coches, dependiendo el video que los coches bajen NO necesariamente implica
# que el cotador de totalDown tiene que subir, piensalo si el "norte" esta en el sur del video, entonces en realidad cuando los coches
# "bajan" se debe aumentar el "totalUp"
DIRECTION_PEOPLE = True

# Creamos un umbral para sabre si el carro paso de izquierda a derecha o viceversa
# En este caso lo deje fijo pero se pudiese configurar según la ubicación de la cámara.
POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]

# Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
fps = FPS().start()

# Definimos el formato del archivo resultante y las rutas.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(PATH_OUTPUT, fourcc, 20.0, (W, H), True)
```

## 4.11 Configuración de los centroides con OpenCV

En la clase anterior ya preparamos todo el código para iniciar la lectura de nuestro video real y empezar a hacer
`object detection` sobre el mismo y utilizar `trackable objects` para ponerles un identificador a cada uno.

Ahora el objetivo de esta clase es relativamente simple: lograr hacer el `object detection` y el `object tracking` para NO tener
que obligatoriamente hacer una detección de objetos por cada fotograma que pasa es que usamos el `object tracking` este consume menos recursos.

```python
# Bucle que recorre todo el video
while True:
  # Leemos el primer frame
  ret, frame = vs.read()

  # Si ya no hay más frame, significa que el video termino y por tanto se sale del bucle
  if frame is None:
    break
  
  status = "Waiting"
  rects = []

  # Nos saltamos los frames especificados. Como al inicio totalFrame es 0 entonces en la primera vuelta el modelo de 0 con 30 es 0
  # por eso en su primera interación empieza con la detección de objetos
  if totalFrame % SKIP_FPS == 0:
    status = "Detecting"
    trackers = []
    # -------- aqui empieza un código que FUNCIONA EN COLAB PERO NO ME FUNCIONO EN LOCAL
    # Tomamos la imagen la convertimos a array luego a tensor
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    # Predecimos los objectos y clases de la imagen
    detections = detect_fn(input_tensor)
    # -------- AQUI TERMINA EL CÓDIGO DE COLAB
    # A final de cuentas el objetivo es que en este punto tengamos las detecciones del modelo de object detection
    
    detection_scores = np.array(detections["detection_scores"][0])
    # Realizamos una limpieza para solo obtener las clasificaciones mayores al umbral.
    # es posible que el modelo detecte cosas con un pequeño score entonces esos los eliminamos
    detection_clean = [x for x in detection_scores if x >= TRESHOLD]

    # Recorremos las detecciones
    for x in range(len(detection_clean)):
      idx = int(detections['detection_classes'][0][x])
      # Tomamos los bounding box 
      # es un poco raro como te regresa el orden del bounding box, pero esta es la orientación correcta
      ymin, xmin, ymax, xmax = np.array(detections['detection_boxes'][0][x])
      # los bouding boxes necesitan ser convertidos al tamaño del video porque el modelo regresaba las coordenadas 
      # normalizadas en porcentaje de alto y ancho
      box = [xmin, ymin, xmax, ymax] * np.array([W,H, W, H])
      # aqui solo nos aseguramos de que cada valor sea un número entero
      (startX, startY, endX, endY) = box.astype("int")

      # Con la función de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
      tracker = dlib.correlation_tracker()
      rect = dlib.rectangle(startX, startY, endX, endY)
      tracker.start_track(frame, rect)

      trackers.append(tracker)
  else:
    # En caso de que no hagamos detección haremos seguimiento
    # Recorremos los objetos que se les está realizando seguimiento
    for tracker in trackers:
      status = "Tracking"
      # Actualizamos y buscamos los nuevos bounding box
      tracker.update(frame)
      pos = tracker.get_position()
      # los datos obtenidos por la posición del tracker se basa en el centroide del objeto
      startX = int(pos.left())
      startY = int(pos.top())
      endX = int(pos.right())
      endY = int(pos.bottom())
      # de forma similar cada cordenada debe ser convertida a un entero
      rects.append((startX, startY, endX, endY))

  # Dibujamos el umbral de conteo
  cv2.rectangle(frame, (POINT[0], POINT[1]), (POINT[0]+ POINT[2], POINT[1] + POINT[3]), (255, 0, 255), 2)

  objects = ct.update(rects)
```

Excelente, hasta este momento ya hemos explicado con comentarios, el código que nos permite hacer detección y seguimiento de objetos.
Ahora lo que veremos en la próxima clase es como contar la cantidad de coches que van en una u otra dirección.

## 4.12 Algoritmo de dirección y conteo con OpenCV

 Vamos a terminar el código de esta sección guardando cada fotograma del video con sus nuevas anotaciones e incluyendo el contador
de coches que suben y bajan;

```python
# Recorremos cada una de las detecciones
  for (objectID, centroid) in objects.items():
    # Revisamos si el objeto ya se ha contado
    to = trackableObjects.get(objectID, None)
    if to is None:
      to = TrackableObject(objectID, centroid)

    else:
      # Si no se ha contado, analizamos la dirección del objeto
      y = [c[1] for c in to.centroids]
      direction = centroid[1] - np.mean(y)
      to.centroids.append(centroid)
      if not to.counted:
        # básicamente esto determina que el centroide este dentro del cuadrado definido anteriormente
        if centroid[0] > POINT[0] and centroid[0] < (POINT[0]+ POINT[2]) and centroid[1] > POINT[1] and centroid[1] < (POINT[1]+POINT[3]):
          if DIRECTION_PEOPLE:
            if direction >0:
              totalUp += 1
              to.counted = True
            else:
              totalDown +=1
              to.counted = True
          else:
            if direction <0:
              totalUp += 1
              to.counted = True
            else:
              totalDown +=1
              to.counted = True

    trackableObjects[objectID] = to

    # Dibujamos el centroide y el ID de la detección encontrada
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
  
  # Totalizamos los resultados finales
  info = [
          ("Subiendo", totalUp),
          ("Bajando", totalDown),
          ("Estado", status),
  ]

  for (i, (k,v)) in enumerate(info):
    text = "{}: {}".format(k,v)
    cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

  # Almacenamos el framme en nuestro video resultante.
  writer.write(frame)
  totalFrame += 1
  fps.update()

# Terminamos de analizar FPS y mostramos resultados finales
fps.stop()

print("Tiempo completo {}".format(fps.elapsed()))
print("Tiempo aproximado por frame {}".format(fps.fps()))

# Cerramos el stream the almacenar video y de consumir el video.
writer.release()
vs.release()
```

Resultados esperados:

![13.gif](imgs%2F4%2F13.gif)

## 4.13 Crea un ciclo de entrenamiento de tu modelo: MLOps

En esta clase vamos a introducir el concepto de `DevOps`

![14.png](imgs%2F4%2F14.png)

Es necesario pensar en el flujo de datos, cada cuanto debo pensar en el re-entrenamineto de mi modelo, a su vez, puedo
automarizar el proceso de entrada y etiquetado de datos y el entrenamiento del modelo?

![15.png](imgs%2F4%2F15.png)

Idealmente, el proceso a seguir sería algo similar al siguiente:

- Automatizar el proceso de recolección de datos, en este caso recolección de imágenes de la cámara de seguridad.
- Utilizar herramientas de IA para realizar un etiquetado automático de las mismas.
- Transformación de datos a TF-Records
- Entrenamiento del modelo de forma trimestral

La arquitectura de nuestro problema sería similar al siguiente:

![16.png](imgs%2F4%2F16.png)

La imagen anterior resume la arquitectura del proceso de automatización que estamos buscando. La imagen se lee de izquierda
a derecha de arriba a abajo. Empezamos con una "alarma de tiempo aleatoria" que dispara una `cloud function` que le permite
acceder a la información de la cámara de vigilancia. Posteriormente, estos datos se almacenan en `cloud storage` y el proceso 
de etiquetado de las imágenes puede ser `manual` o `automático` de ahí con una alarma semanal por ejemplo, pasamos a activar
otra `cloud function` para entrenar al modelo, creamos un `docker container`y si cumple con ciertos requisitos de calidad 
hacemos `deploy` al mismo en Vertex AI.


# 5 Producto con visión computarizada en producción

## 5.1 Prepara tu entorno en Google Cloud Platform

## 5.2 Carga y preprocesamiento de modelos

## 5.3 Postprocesamiento de modelos

## 5.4 Despliega y consume tu modelo en producción

## 5.5 Bonus: aprende a apagar las máquinas de GCP para evitar sobrecostos

# 6 Siguientes pasos en inteligencia artificial

## 6.1 Siguientes pasos en inteligencia artificial


![2.png](imgs%2F3%2F2.png)[Entrenamiento_Base.ipynb](scripts%2Fproject%2Fcore%2FEntrenamiento_Base.ipynb)