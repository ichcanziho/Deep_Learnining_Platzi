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

## 2.4 Visualización de IoU en object detection

## 2.5 Tipos de arquitecturas en detección de objetos

## 2.6 Arquitecturas relevantes en object detection

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
