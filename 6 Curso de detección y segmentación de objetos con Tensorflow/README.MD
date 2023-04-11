# Curso de Detección y Segmentación de Objetos con TensorFlow

Eleva tu nivel en inteligencia artificial aprendiendo tareas avanzadas de deep learning para detectar y segmentar objetos en imágenes y videos. 
Entrena modelos pre-entrenados de computer vision de acuerdo a las necesidades de tus proyectos.

- Evalúa desempeño de modelos para visión computarizada.
- Utiliza modelos pre-entrenados para visión computarizada.
- Segmenta objetos dentro de imágenes.
- Aplica detección de objetos con Python y TensorFlow.


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> - [5: Curso de Experimentación en Machine Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/5%20Curso%20de%20introducci%C3%B3n%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
> 
> Este Curso es el Número 6 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice
- [1 Introducción a Computer Vision](#1-introducción-a-computer-vision)
  - [1.1 ¿Qué es la visión computarizada y cuáles son sus tipos?](#11-qué-es-la-visión-computarizada-y-cuáles-son-sus-tipos)
  - [2 Detección de objetos](#2-detección-de-objetos)
  - [2.1 Introducción a object detection: sliding window y bounding box](#21-introducción-a-object-detection-sliding-window-y-bounding-box)
  - [2.2 Generando video de sliding window](#22-generando-video-de-sliding-window)
  - [2.3 Introducción a object detection: backbone, non-max suppression y métricas](#23-introducción-a-object-detection-backbone-non-max-suppression-y-métricas)
  - [2.4 Visualización de IoU en object detection](#24-visualización-de-iou-en-object-detection)
  - [2.5 Tipos de arquitecturas en detección de objetos](#25-tipos-de-arquitecturas-en-detección-de-objetos)
  - [2.6 Arquitecturas relevantes en object detection](#26-arquitecturas-relevantes-en-object-detection)
  - [2.7 Utilizando un dataset de object detection](#27-utilizando-un-dataset-de-object-detection)
  - [2.8 Carga de dataset de object detection](#28-carga-de-dataset-de-object-detection)
  - [2.9 Exploración del dataset de object detection](#29-exploración-del-dataset-de-object-detection)
  - [2.10 Visualización de bounding boxes en el dataset de object detection](#210-visualización-de-bounding-boxes-en-el-dataset-de-object-detection)
  - [2.11 Aumentando de datos con Albumentation](#211-aumentando-de-datos-con-albumentation)
  - [2.12 Implementando Albumentation en object detection](#212-implementando-albumentation-en-object-detection)
  - [2.13 Visualizando imágenes con aumentado de datos](#213-visualizando-imágenes-con-aumentado-de-datos)
  - [2.14 Utilizando un modelo de object detection pre-entrenado](#214-utilizando-un-modelo-de-object-detection-pre-entrenado)
  - [2.15 Fine-tuning en detección de objetos](#215-fine-tuning-en-detección-de-objetos)
  - [2.16 Fine-tuning en detección de objetos: carga de datos](#216-fine-tuning-en-detección-de-objetos-carga-de-datos)
  - [2.17 Fine-tuning en detección de objetos: data augmentation](#217-fine-tuning-en-detección-de-objetos-data-augmentation)
  - [2.18 Fine-tuning en detección de objetos: entrenamiento](#218-fine-tuning-en-detección-de-objetos-entrenamiento)
  - [2.19 Fine tuning en detección de objetos: visualización de objetos](#219-fine-tuning-en-detección-de-objetos-visualización-de-objetos)
  - [2.20 Quiz módulo object detection](#220-quiz-módulo-object-detection)
- [3 Segmentación de objetos](#3-segmentación-de-objetos)
  - [3.1 Introduciendo la segmentación de objetos](#31-introduciendo-la-segmentación-de-objetos)
  - [3.2 Tipos de segmentación de objetos](#32-tipos-de-segmentación-de-objetos)
  - [3.3 Tipos de segmentación y sus arquitecturas relevantes](#33-tipos-de-segmentación-y-sus-arquitecturas-relevantes)
  - [3.4 ¿Cómo es un dataset de segmentación?](#34-cómo-es-un-dataset-de-segmentación)
  - [3.5 Utilizando un dataset de segmentación de objetos](#35-utilizando-un-dataset-de-segmentación-de-objetos)
  - [3.6 Visualización de nuestro dataset de segmentación](#36-visualización-de-nuestro-dataset-de-segmentación)
  - [3.7 Creando red neuronal U-Net para segmentación](#37-creando-red-neuronal-u-net-para-segmentación)
  - [3.8 Entrenando y estudiando una red de segmentación](#38-entrenando-y-estudiando-una-red-de-segmentación)
  - [3.9 Generando predicciones con modelo de object segmentation](#39-generando-predicciones-con-modelo-de-object-segmentation)
  - [3.10 Quiz módulo segmentación](#310-quiz-módulo-segmentación)
- [4 Un paso más allá](#4-un-paso-más-allá)
  - [4.1 El estado de la cuestión en computer vision](#41-el-estado-de-la-cuestión-en-computer-vision)


# 1 Introducción a Computer Vision

## 1.1 ¿Qué es la visión computarizada y cuáles son sus tipos?

Te recomiendo ampliamente dirigirte a la siguiente lectura de este mismo curso: [Principales usos de la visión por computadora](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales#11-la-importancia-del-computer-vision)

A modo de repaso puedes leer el siguiente resumen:

La `visión computarizada` es una rama de la inteligencia artificial y la informática que se enfoca en el desarrollo de sistemas 
y técnicas para la interpretación de imágenes y videos digitales. El objetivo de la `visión computarizada` es automatizar 
la tarea de análisis y reconocimiento visual que normalmente realizaría un ser humano.

![1.png](ims%2F1%2F1.png)

Existen varios tipos de visión computarizada, entre ellos:

- `Reconocimiento de objetos:` se enfoca en la identificación y localización de objetos específicos en una imagen o video.

- `Seguimiento de objetos:` se enfoca en el seguimiento del movimiento de objetos en un video.

- `Detección de patrones:` se enfoca en la identificación de patrones en una imagen, como por ejemplo la identificación de formas geométricas.

- `Reconocimiento facial:` se enfoca en la identificación y seguimiento de rasgos faciales en imágenes o videos.

- `Análisis de escenas:` se enfoca en la interpretación de una escena completa en una imagen o video, incluyendo la identificación de objetos y su relación espacial.

- `Reconocimiento de texto:` se enfoca en la identificación y extracción de texto en imágenes.

- `Detección de movimiento:` se enfoca en la identificación de cambios en el movimiento de objetos en un video.

# 2 Detección de objetos

## 2.1 Introducción a object detection: sliding window y bounding box

A lo largo de cursos anteriores hemos visto la tarea de clasificación de imágenes, la cual consistía en entrenar una red
neuronal con ejemplos conocidos de diferentes imágenes. Cada imagen perteneciente a una clase en específico, con la finalidad de
que la red pudiera diferenciar entre estas clases de acuerdo a los patrones de cada imagen. El siguiente paso lógico en nuestra 
aventura por computer vision es que la red NO solo sea capaz de decirte a que clase pertenece una imagen, sino también decirte
en qué parte de la imagen se encuentra esta clase. A esta tarea se le conoce como `localization`.

![1.png](ims%2F2a%2F1.png)

Pero, ¿qué pasaría si en una misma imagen tuviéramos varios objetos pertenecientes a clases diferentes? Entonces nos estaríamos
enfrentando a un problema de `Object Detection` un problema de `classification + localization`de multiples objetos. Para ello
podría utilizar una `bounding box` un rectángulo auxiliar que englobe la posición del objeto de interés. Y finalmente
si quisiera poder saber exactamente cuáles son los límites del objeto entonces el problema ya no sería de `Object Detection`
sino de `Instance Segmentation` y él `bounding box` ya no sería suficiente preciso.

### Conceptos: `Sliding Window`

Hay diferentes enfoques para la detección de objetos, pero uno de los más comunes es el uso de una técnica conocida como 
ventana deslizante o `sliding window` en inglés.

![sliding_window_example.gif](ims%2F2a%2Fsliding_window_example.gif)

La ventana deslizante es una técnica en la que una pequeña ventana rectangular se mueve a través de la imagen en pasos 
fijos y se aplica un clasificador para determinar si la ventana contiene o no un objeto de interés. La ventana se desliza 
por toda la imagen y cada vez que el clasificador detecta un objeto, se marca la ubicación de la ventana. Una de las 
desventajas de esta técnica es que puede ser computacionalmente costosa, ya que se debe aplicar el clasificador a múltiples 
ventanas de diferentes tamaños y posiciones en la imagen.

### Conceptos: `Bounding Box`

Una bounding box o cuadro delimitador en español es un rectángulo que se dibuja alrededor de un objeto específico en una imagen o video, con el fin de identificar y localizar la posición del objeto en la imagen. La bounding box se define mediante cuatro valores numéricos que representan las coordenadas x e y del borde superior izquierdo del rectángulo, y el ancho y la altura del rectángulo.

![2.png](ims%2F2a%2F2.png)

En la detección de objetos, la tarea principal es identificar los objetos de interés en una imagen o video y localizar su posición exacta en la imagen. La utilización de bounding boxes permite a los algoritmos de visión computarizada identificar y ubicar objetos de forma más eficiente y precisa. Una vez que se detecta el objeto de interés, se puede dibujar un bounding box alrededor del objeto para resaltar su posición en la imagen.

### Ejemplo en código: Creando nuestro `Sliding Window`

> ## Nota:
> El código lo puedes encontrar en: [sliding_window.py](Detecci%C3%B3n%20de%20Objetos%2F1%20Sliding%20Window%20Example%2Fsliding_window.py)

Vamos a partir de la siguiente imagen de una mujer corriendo:

![mujer.jpg](Detecci%C3%B3n%20de%20Objetos%2F1%20Sliding%20Window%20Example%2Fmujer.jpg)

Nuestro trabajo en esta clase será crear un código que nos permita explicar mediante una animación el funcionamiento de `sliding window`

Empecemos por importar las bibliotecas necesarias:

```python
import imageio  # biblioteca para creación de gifs
import cv2
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  # Lo usaremos para crear un recuadro de color
import numpy as np
```

Vamos a empezar por leer la imagen de referencia:

```python
img = cv2.imread("mujer.jpg")
# Nota, por defecto cv lee las imágenes en formato BGR por eso es necesario pasarlas a RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Es importante tener la imagen en dimenciones conocidas, y para este ejemplo es perfecto que sea cuadrada.
img = cv2.resize(img, dsize=(1000, 1000))
```

Definimos nuestra función de `sliding window` como un generador de python:

> Nota si no conoces `yield` y qué son los generadores de python puedes leer esta entrada que pertenece a este mismo repositorio:
> [Generadores](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow#26-datasets-generators)

```python
def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1] + 1, step):
        for x in range(0, image.shape[1] - ws[0] + 1, step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]
```

El código es bastante explicativo, devuelve dos coordenadas, `x, y` y la imagen recortada en esas coordenadas tomando como
alto `ws[1]` y como ancho `ws[0]` mientras que `step` es una vez que recorto la imagen, cuanto debo moverme para recortar la próxima imagen.

Ahora vamos a generar una función que nos permita mostrar una imagen con la imagen original a la izquierda y la imagen recortada a la derecha:

```python
def get_window(window):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(img)
    rect = patches.Rectangle((window[0], window[1]), 200, 200, linewidth=2, edgecolor='g', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(window[2])
    ax2.set_xticks([])
    ax2.set_yticks([])
```
Hasta este punto ya hemos creado una figura de `matplotlib` y bien podríamos guardar cada imagen en disco con un nombre diferente,
pero para esta implementación y por variar lo visto en la clase vamos a utilizar un método que NO requiere guardar la imagen
para posteriormente leerla.

Lo único que necesitamos hacer es obtener el `numpy array` de la figura de `matplotlib` para ello usamos su `canvas` y creamos
el `numpy array` usando `np.frombuffer` y llevamos el contenido del `canvas` a `tostring_rgb`:

```python
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    img_array = img_array[380:840, 140:1090]
    plt.close()
    return img_array
```

Excelente, ahora solamente necesitamos llamar a esta función por cada sliding window generada. Pero eso lo veremos en la siguiente clase.

## 2.2 Generando video de sliding window

De forma sencilla ya tenemos todo lo necesario para crear nuestra `animación` de `sliding window`

Hasta la clase pasada teníamos el siguiente código:

```python
import imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1] + 1, step):
        for x in range(0, image.shape[1] - ws[0] + 1, step):
            yield x, y, image[y:y + ws[1], x:x + ws[0]]

def get_window(window):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(img)
    rect = patches.Rectangle((window[0], window[1]), 200, 200, linewidth=2, edgecolor='g', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(window[2])
    ax2.set_xticks([])
    ax2.set_yticks([])
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    img_array = img_array[380:840, 140:1090]
    plt.close()
    return img_array

if __name__ == '__main__':
    img = cv2.imread("mujer.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(1000, 1000))

```

Ahora lo único que necesitamos es utilizar la función `get_window` para cada `w` generada por `sliding_window` y almacenar
estos valores en una lista para finalmente guardarlos como un `.gif`

```python
    img_gif = []
    for w in sliding_window(img, 200, (200, 200)):
        img_gif.append(get_window(w))
        
    imageio.mimwrite('animación.gif', img_gif, 'GIF', duration=0.2)
```
Resultado Final:

![animación.gif](Detecci%C3%B3n%20de%20Objetos%2F1%20Sliding%20Window%20Example%2Fanimaci%C3%B3n.gif)

## 2.3 Introducción a object detection: backbone, non-max suppression y métricas

### BackBone

Empecemos definiendo en concepto de `BackBone` o columna vertebral:

![3.png](ims%2F2a%2F3.png)

El concepto es muy sencillo y hace referencia a que la "columna vertebral" de nuestros sistemas de `object detection`
no es más que tomar algún modelo pre-entrenado para la clasificación de imágenes por ejemplo `AlexNet` entonces mediante
`transfer learning` podemos aprovechar todo este conocimiento. Removemos las últimas capas de clasificador pre-entrenado y añadimos
nuestra funcionalidad con nuevas capas, en este caso una `head` de clasificación de `gatos` y otra `head` de regresión para
obtenerlos `bounding boxes`.

### Intersection Over Union IoU

En clasificación de objetos era bastante sencillo saber si un modelo había predicho bien si la imagen de un gato era clasificada
como un gato. Nuestras etiquetas mostraban el nombre de la clase a la que perteneció la imagen original. Sin embargo, en el mundo
de `object detection` no solo tenemos etiquetas con el nombre de una clase, sino que también tenemos coordenadas expresando 
la locación de donde se encuentra el objeto de interés. Entonces la pregunta es ¿Cómo saber que tan bien ha identificado 
las coordenadas el modelo respecto a la etiqueta real? Para eso ocupamos la Intersección entre Uniones:

Este termino es bastante simple de entender con la siguiente imagen.

![4.png](ims%2F2a%2F4.png)

Supongamos que nuestro objeto de interés está delimitado por el cuadrado de color `Verde` que supongamos tiene un área de 100 unidades.
Ahora la predicción de nuestro modelo es el cuadrado de color `Rojo`. La pregunta es ¿qué tan buena fue la predicción? Pues muy fácil
vamos a usar IoU para definir este nivel de precisión. Supongamos que el cuadrado rojo también tiene un área de 100 unidades.

En el mejor de los casos el cuadrado rojo y el verde estarán justo uno encima de otro entonces su área de intersección será de 100 unidades,
puesto que comparten el 100% del espacio y el área de Union también será de 100 porque al sumar las 2 áreas que están en las mismas coordenadas
su suma sigue siendo 100. Entonces su IoU sería de `100/100 = 1` un resultado perfecto.

Por otro lado, si las cajas están completamente separadas entre sí, lo que tendríamos sería un área de intersección de 0 y un 
área de Union de 200 dando un IoU de `0/200 = 0` el peor resultado posible.

Sin embargo, en el ejemplo de la imagen supongamos los siguientes valores: área de intersección 90, área de unión 110
entonces IoU `90/110 = 0.81` en general un IoU superior a `0.6` se considera buena. Sin embargo, este límite se puede cambiar
el cualquier momento, por ejemplo `0.8` si es superior lo damos por bueno y si es inferior por malo. En el ejemplo anterior
sería un buen ejemplo.


### Non-max suppression

Es bastante habitual que usando la técnica de `sliding window` varias zonas de la imagen nos aparezca con una zona de clasificación
de objeto y tenga su propia `bounding box` con su correspondiente `IoU` un método de limpieza para este escenario es: `Non-max suppression`.

La siguiente imagen explica perfectamente el proceso de Non-max suppression `NMS`. 
![5.png](ims%2F2a%2F5.png)
Como tal solamente nos estamos quedando con el `bounding box` que tenga el mayor grado de `confidence` para su predicción.

A continuación te explico el proceso general de cómo funciona el NMS:

- `Detección de objetos:` Primero, se obtienen las detecciones de objetos en la imagen utilizando un algoritmo de detección, que puede proporcionar las coordenadas de caja delimitadora (bounding box) alrededor de los objetos detectados, así como sus puntuaciones de confianza que indican la probabilidad de que la detección sea correcta.

- `Ordenamiento:` Las detecciones se ordenan en función de sus puntuaciones de confianza en orden descendente. Esto permite seleccionar primero las detecciones con las puntuaciones más altas, que se consideran más confiables.

- `Supresión de no máxima:` A continuación, se toma la detección con la puntuación más alta (la detección con la mayor confianza) 
y se la considera como una detección válida. Luego, se comparan las áreas de solapamiento de las detecciones restantes con la detección 
válida utilizando una medida de solapamiento, como la Intersección sobre Unión `(IoU)`, que es la proporción del área de solapamiento entre dos bounding boxes dividida por el área de su unión.

- `Eliminación de detecciones redundantes:` Si el `IoU` entre una detección restante y la detección válida es mayor que un umbral 
predefinido, se considera que las detecciones se solapan y se descartan las detecciones restantes. Esto se hace para eliminar 
detecciones redundantes y seleccionar la detección más confiable. Si el `IoU` es menor que el umbral, la detección se considera como una detección válida adicional.

- `Repetición del proceso:` El proceso se repite iterativamente hasta que todas las detecciones hayan sido procesadas y se hayan seleccionado las detecciones válidas finales.

Al final del proceso de `NMS`, se obtiene un conjunto de detecciones no superpuestas y confiables, lo que ayuda a reducir 
falsos positivos y obtener una salida más limpia y precisa en aplicaciones de detección de objetos. El umbral de `IoU` es 
un parámetro ajustable que puede ser configurado según las necesidades específicas de la aplicación.



### Otras métricas

Como hemos definido anteriormente, el problema de `object detection` puede ser usado para clasificar entre multiples instancias
estas pueden pertenecer a la misma o a diferentes clases. Pero cada objeto tendrá su correspondiente `ground truth` y su propia
predicción.

![6.png](ims%2F2a%2F6.png)

Una forma muy sencilla de ver que tan bueno fue el modelo sobre todas estas predicciones es utilizando en `mean average precision`
básicamente el promedio de todas las `IoU` individuales. Finalmente en entornos reales es sumamente importante monitorear
la cantidad de `Frames per Second FPS` que ofrece el modelo para la detección de video. Normalmente, buscamos modelos los 
más rápidos y eficientes posibles en terminus de FPS puede haber modelos más precisos, pero si son lentos quizá no sean la 
solución que buscamos para nuestro problema. 


### Ejemplo en código: Trabajando con IoU

> ## Nota:
> El código completo está [IOU.py](Detecci%C3%B3n%20de%20Objetos%2F2%20IOU%2FIOU.py)

Vamos a suponer que tenemos un detector de rostros, entonces volvamos a usar la imagen de la mujer corriendo anteriormente.

Empecemos importando bibliotecas necesarias:

```python
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
```
Leemos la imagen de la mujer
```python
image = cv2.imread('mujer.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
Definimos nuestros `bounding boxes:`

```python
bb_truth = [680, 380, 830, 580]
bb_predicted = [700, 400, 840, 600]
```

Vamos a crear la función de obtención de `IoU`:

```python
def bb_intersection_over_union(ground_truth_bbox, predicted_bbox):
    xA = max(ground_truth_bbox[0], predicted_bbox[0])
    yA = max(ground_truth_bbox[1], predicted_bbox[1])
    xB = min(ground_truth_bbox[2], predicted_bbox[2])
    yB = min(ground_truth_bbox[3], predicted_bbox[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    ground_truth_bbox_area = (ground_truth_bbox[2] - ground_truth_bbox[0] + 1) * (
                ground_truth_bbox[3] - ground_truth_bbox[1] + 1)
    predicted_bbox_area = (predicted_bbox[2] - predicted_bbox[0] + 1) * (predicted_bbox[3] - predicted_bbox[1] + 1)

    iou_ = intersection_area / float(ground_truth_bbox_area + predicted_bbox_area - intersection_area)

    return iou_
```

El código utiliza las siguientes variables y operaciones:

- `ground_truth_bbox:` Una lista que contiene las coordenadas del cuadro delimitador de la verdad fundamental (ground truth) en el formato `[xA, yA, xB, yB]`, donde xA e yA son las coordenadas del punto superior izquierdo del cuadro delimitador, y xB e yB son las coordenadas del punto inferior derecho del cuadro delimitador.

- `predicted_bbox:` Una lista que contiene las coordenadas del cuadro delimitador predicho por el algoritmo en el formato `[xA, yA, xB, yB]`, con la misma estructura que ground_truth_bbox.

- `xA, yA, xB, yB:` Variables que representan las coordenadas del cuadro delimitador de la intersección entre ground_truth_bbox y predicted_bbox. xA y yA se calculan como el máximo valor entre las coordenadas xA y yA de ambos cuadros delimitadores, mientras que xB y yB se calculan como el mínimo valor entre las coordenadas xB y yB de ambos cuadros delimitadores. Estos valores se utilizan para determinar el área de intersección entre los cuadros delimitadores.

- `intersection_area:` Variable que almacena el área de intersección entre los cuadros delimitadores, calculada como el producto de las longitudes de los lados del cuadro delimitador de la intersección. Si no hay intersección entre los cuadros delimitadores, se establece en 0.

- `ground_truth_bbox_area:` Variable que almacena el área del cuadro delimitador de la verdad fundamental, calculada como el producto de las longitudes de los lados del cuadro delimitador de la verdad fundamental.

- `predicted_bbox_area:` Variable que almacena el área del cuadro delimitador predicho, calculada como el producto de las longitudes de los lados del cuadro delimitador predicho.

- `iou:` Variable que almacena el valor del Índice de Intersección sobre Unión, calculado como la división del área de intersección entre el área de unión de los cuadros delimitadores.

La función retorna el valor del Índice de Intersección sobre Unión (IoU) como resultado. Un valor alto de IoU (cerca de 1) indica una mayor superposición y, por lo tanto, una mayor similitud entre los cuadros delimitadores, mientras que un valor bajo de IoU (cerca de 0) indica poca superposición y, por lo tanto, poca similitud entre los cuadros delimitadores.

Llamamos a la función para conocer cuál es el `IoU` de nuestras boxes.
```python
iou = bb_intersection_over_union(bb_truth, bb_predicted)
print(iou)
```
Respuesta esperada:
```commandline
0.677825105057031
```
En la siguiente clase veremos el código para crear la imagen con los boxes adecuados.

## 2.4 Visualización de IoU en object detection

Hasta este momento nuestro código de la clase anterior era el siguiente:

```python
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def bb_intersection_over_union(ground_truth_bbox, predicted_bbox):
    xA = max(ground_truth_bbox[0], predicted_bbox[0])
    yA = max(ground_truth_bbox[1], predicted_bbox[1])
    xB = min(ground_truth_bbox[2], predicted_bbox[2])
    yB = min(ground_truth_bbox[3], predicted_bbox[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    ground_truth_bbox_area = (ground_truth_bbox[2] - ground_truth_bbox[0] + 1) * (
                ground_truth_bbox[3] - ground_truth_bbox[1] + 1)
    predicted_bbox_area = (predicted_bbox[2] - predicted_bbox[0] + 1) * (predicted_bbox[3] - predicted_bbox[1] + 1)

    iou_ = intersection_area / float(ground_truth_bbox_area + predicted_bbox_area - intersection_area)

    return iou_

if __name__ == '__main__':
    image = cv2.imread('mujer.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bb_truth = [680, 380, 830, 580]
    bb_predicted = [700, 400, 840, 600]

    iou = bb_intersection_over_union(bb_truth, bb_predicted)
    print(iou)
```

Vamos a añadirle la porción de código para graficar los `bounding boxes` tanto el `ground truth` como el `predicted`

```python
def show_bounding_boxes(truth, predicted):
    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(image)
    rect = patches.Rectangle(tuple(truth[:2]), truth[2] - truth[0], truth[3] - truth[1],
                             linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle(tuple(predicted[:2]), predicted[2] - predicted[0], predicted[3] - predicted[1],
                             linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig("iou.png")
    plt.close()
```
Como podemos observar la función se apoya de `patches.Rectangle` el cual ya habíamos usado en la clase anterior para el ejemplo
del `sliding window`.

Ahora solo resta llamar a la función:

```python
show_bounding_boxes(bb_truth, bb_predicted)
```
Respuesta esperada:

![iou.png](Detecci%C3%B3n%20de%20Objetos%2F2%20IOU%2Fiou.png)

Excelente podemos ver como aparece un recuadro verde con la `bounding box` real y en rojo la predicha.

## 2.5 Tipos de arquitecturas en detección de objetos

Cuando hablamos de detección de objetos debemos mencionar que existen básicamente 2 tipos de arquitecturas que permiten resolver
este problema. La arquitectura de una sola etapa `(one-stage)` y la multietapa `(two-stage)`. Estos enfoques difieren en cómo se 
lleva a cabo la detección de objetos y cómo se manejan las detecciones múltiples.

![7.png](ims%2F2a%2F7.png)

**Enfoque de una etapa (Single-stage):**

El enfoque de una etapa, como su nombre lo indica, realiza la detección de objetos en una sola etapa, directamente en la 
imagen de entrada. No hay una etapa previa de generación de regiones de interés (region proposal) como en el enfoque 
multietapa. Estas arquitecturas suelen ser más rápidas, pero pueden tener una precisión ligeramente menor en comparación 
con las arquitecturas multietapa.

![8.png](ims%2F2a%2F8.png)

Ejemplos de arquitecturas de detección de una etapa son YOLO (You Only Look Once) y SSD (Single Shot MultiBox Detector). 
Estas arquitecturas utilizan redes neuronales convolucionales (CNN) para generar predicciones de clases y coordenadas de 
cajas delimitadoras directamente desde la imagen de entrada en una sola pasada. Luego, aplican supresión de no máxima (NMS) 
para eliminar detecciones redundantes y seleccionar las detecciones finales.

**Enfoque multietapa (Two-stage):**

El enfoque multietapa, por otro lado, involucra dos etapas distintas en la detección de objetos. En la primera etapa, se 
generan regiones de interés (region proposals) utilizando métodos como Selective Search o RPN (Region Proposal Network), 
que son áreas de la imagen que podrían contener objetos. Luego, en la segunda etapa, se realiza la clasificación y la 
regresión de las coordenadas de las cajas delimitadoras dentro de estas regiones propuestas.

![9.png](ims%2F2a%2F9.png)

Ejemplos de arquitecturas de detección de dos etapas son R-CNN (Region-based Convolutional Neural Networks), Fast R-CNN, 
y Faster R-CNN. Estas arquitecturas utilizan CNN para generar regiones propuestas en la primera etapa y luego utilizan 
una segunda red neuronal para clasificar y refinar estas regiones propuestas en la segunda etapa.

> # Resumen:
> El enfoque de una etapa realiza la detección de objetos directamente en una sola pasada, mientras que el enfoque multietapa 
> involucra una etapa previa de generación de regiones propuestas y una etapa posterior de clasificación y regresión de las 
> coordenadas de las cajas delimitadoras. Ambos enfoques tienen ventajas y desventajas en términos de velocidad, precisión 
> y complejidad, y la elección de la arquitectura adecuada depende de las necesidades específicas de la aplicación y los 
> recursos computacionales disponibles.

## 2.6 Arquitecturas relevantes en object detection

### Arquitecturas multietapa:

**R-CNN (Region-based Convolutional Neural Networks):**

R-CNN fue la primera arquitectura que introdujo el enfoque de dos etapas en la detección de objetos. En la primera etapa, R-CNN utiliza el algoritmo Selective Search para generar regiones propuestas en la imagen de entrada, que son áreas de la imagen que podrían contener objetos. Luego, en la segunda etapa, R-CNN utiliza una red neuronal convolucional (CNN) para extraer características de cada región propuesta y clasificarlas en diferentes clases de objetos, así como para refinar las coordenadas de las cajas delimitadoras que rodean a los objetos detectados.

![9.png](ims%2F2a%2F9.png)

Sin embargo, R-CNN tiene una desventaja en términos de velocidad, ya que procesa cada región propuesta de manera independiente, lo que resulta en un alto costo computacional. Además, el proceso de entrenamiento de R-CNN es lento debido a que se entrena cada región propuesta de forma individual.

**Fast R-CNN:**

Fast R-CNN es una mejora de R-CNN que aborda algunas de las limitaciones de la arquitectura original. En lugar de aplicar la CNN a cada región propuesta por separado, Fast R-CNN realiza la extracción de características de la imagen de entrada una sola vez y luego utiliza una capa de RoI (Region of Interest) para extraer las características correspondientes a cada región propuesta. Esto permite compartir la extracción de características entre todas las regiones propuestas, lo que resulta en una mayor eficiencia en términos de velocidad durante la inferencia.

![10.png](ims%2F2a%2F10.png)

Además, Fast R-CNN introduce una capa de regresión que se encarga de predecir las coordenadas de las cajas delimitadoras de los objetos directamente en la CNN, en lugar de hacerlo en una etapa separada como en R-CNN. Esto hace que el proceso de refinamiento de las coordenadas sea más eficiente y rápido.

**Faster R-CNN:**

Faster R-CNN es una evolución adicional de la arquitectura de R-CNN y Fast R-CNN que propone el uso de una Red de Propuestas de Regiones (RPN, por sus siglas en inglés) para generar regiones propuestas en lugar de utilizar el algoritmo Selective Search. El RPN es una red neuronal convolucional que se entrena para generar regiones propuestas en función de la probabilidad de que contengan un objeto. El RPN comparte las características de la CNN utilizada para la detección, lo que hace que la generación de regiones propuestas sea más rápida y eficiente.

![11.png](ims%2F2a%2F11.png)

Una vez que se generan las regiones propuestas con el RPN, el proceso de clasificación y regresión de las coordenadas de las cajas delimitadoras se realiza de manera similar a Fast R-CNN, utilizando la capa de RoI y la capa de regresión.

### Arquitecturas de una etapa:

**SSD (Single Shot MultiBox Detector):**

![1.png](ims%2F2b%2F1.png)

SSD es una arquitectura de detección de objetos en imágenes que propone una estrategia basada en "anchors" o anclas para detectar objetos en diferentes escalas y aspectos en una sola pasada. SSD utiliza múltiples capas de detección a diferentes escalas, que están conectadas directamente a la salida de una red neuronal convolucional (CNN). Cada capa de detección está diseñada para detectar objetos de diferentes tamaños y aspectos, lo que permite capturar objetos de diferentes escalas en una sola inferencia. SSD también incorpora la predicción de clases y regresión de coordenadas de cajas delimitadoras en cada capa de detección, lo que le permite ser eficiente en términos de tiempo de procesamiento.

**YOLO (You Only Look Once):**

![2.png](ims%2F2b%2F2.png)

YOLO es una arquitectura de detección de objetos en imágenes que propone una aproximación basada en rejillas (grids) para dividir la imagen de entrada en celdas y realizar la detección y clasificación de objetos en cada celda en una sola pasada. YOLO utiliza una sola red neuronal convolucional (CNN) para predecir las clases y las coordenadas de las cajas delimitadoras de los objetos en cada celda de la rejilla. Esto hace que YOLO sea rápido y eficiente en términos de tiempo de procesamiento, aunque puede tener ciertas limitaciones en la detección de objetos pequeños o en la captura de objetos con aspectos diferentes.

**RetinaNet:**

![3.png](ims%2F2b%2F3.png)

RetinaNet es una arquitectura de detección de objetos en imágenes que aborda el desafío de la detección de objetos en diferentes escalas y la resolución de objetos difíciles (objetos pequeños o con oclusiones) mediante la utilización de una red neuronal convolucional (CNN) con una estructura de dos ramas. Una rama se encarga de la predicción de las clases de objetos y la otra de la regresión de las coordenadas de las cajas delimitadoras. Lo innovador de RetinaNet es el uso de una función de pérdida focal que prioriza el entrenamiento en ejemplos difíciles, lo que mejora la detección de objetos difíciles en comparación con otras arquitecturas.

**EfficientNet:**

![4.png](ims%2F2b%2F4.png)

EfficientNet es una arquitectura de red neuronal convolucional (CNN) que se destaca por su eficiencia en términos de capacidad de representación y rendimiento computacional. EfficientNet utiliza un enfoque de escalado compuesto por la escala de ancho (width), profundidad (depth) y resolución (resolution) de la red para obtener un equilibrio óptimo entre el rendimiento y la eficiencia computacional. EfficientNet ha demostrado ser muy efectivo en tareas de visión por computadora, incluyendo la detección de objetos en imágenes, gracias a su capacidad para capturar características relevantes y su eficiencia en términos de tiempo de procesamiento

**DETR:**

DETR es una arquitectura de detección de objetos en imágenes que utiliza la arquitectura de transformers, originalmente propuesta para tareas de procesamiento del lenguaje natural (NLP), en el contexto de detección de objetos en imágenes. A diferencia de las arquitecturas de detección de objetos tradicionales que utilizan anclas o rejillas, DETR utiliza una aproximación basada en "set prediction" o predicción de conjuntos, lo que significa que no requiere de anclas o propuestas previas.

![5.png](ims%2F2b%2F5.png)

La arquitectura DETR consta de dos componentes principales: el encoder y el decoder. El encoder es una red neuronal convolucional que se encarga de extraer características de la imagen de entrada. El decoder es un transformer que toma las características de la imagen del encoder y genera predicciones de clases y coordenadas de las cajas delimitadoras para los objetos en la imagen.

Una característica distintiva de DETR es que realiza la predicción de objetos como un problema de asignación bipartita. En lugar de predecir las coordenadas de las cajas delimitadoras y clasificar los objetos simultáneamente, DETR asigna un conjunto fijo de cajas delimitadoras a los objetos detectados y predice las coordenadas de las cajas asignadas y las clases correspondientes. Esto se realiza mediante la utilización de un mecanismo de atención en el decoder del transformer, que permite asignar las cajas delimitadoras a los objetos de manera eficiente.




## 2.7 Utilizando un dataset de object detection

## 2.8 Carga de dataset de object detection

## 2.9 Exploración del dataset de object detection

## 2.10 Visualización de bounding boxes en el dataset de object detection

## 2.11 Aumentando de datos con Albumentation

## 2.12 Implementando Albumentation en object detection

## 2.13 Visualizando imágenes con aumentado de datos

## 2.14 Utilizando un modelo de object detection pre-entrenado

## 2.15 Fine-tuning en detección de objetos

## 2.16 Fine-tuning en detección de objetos: carga de datos

## 2.17 Fine-tuning en detección de objetos: data augmentation

## 2.18 Fine-tuning en detección de objetos: entrenamiento

## 2.19 Fine tuning en detección de objetos: visualización de objetos

## 2.20 Quiz módulo object detection

# 3 Segmentación de objetos

## 3.1 Introduciendo la segmentación de objetos

## 3.2 Tipos de segmentación de objetos

## 3.3 Tipos de segmentación y sus arquitecturas relevantes

## 3.4 ¿Cómo es un dataset de segmentación?

## 3.5 Utilizando un dataset de segmentación de objetos

## 3.6 Visualización de nuestro dataset de segmentación

## 3.7 Creando red neuronal U-Net para segmentación

## 3.8 Entrenando y estudiando una red de segmentación

## 3.9 Generando predicciones con modelo de object segmentation

## 3.10 Quiz módulo segmentación

# 4 Un paso más allá

## 4.1 El estado de la cuestión en computer vision
