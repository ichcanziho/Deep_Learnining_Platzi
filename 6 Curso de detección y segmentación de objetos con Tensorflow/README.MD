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

# Cómo correr estos códigos:

Hola, vengo del futuro. Este específico curso requiere de descargar varias cosas, entre las que se encuentran: Modelos, Datasets, repositorios de GitHub
entre varias otras cosas. Es por ello que para utilizar de forma más eficiente el entorno de python vamos a crear un entorno virtual extra para este curso.

A lo largo del curso voy a estar ocupando 2 paths generales uno será el siguiente:

```commandline
/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow
```

A este path lo llamaré `root` y será el path dónde almacene TODO lo que NO es código directamente, puesto que se encuentra en un disco duro externo.

Por otro lado, todo el código de este repositorio se hará referencia desde la raíz de este curso:

```commandline
6 Curso de detección y segmentación de objetos con Tensorflow
```

Con la información anteriormente dicha vamos a empezar por crear nuestro entorno virtual:

```bash
cd /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow
python3.10 -m venv C6venv
soruce C6venv/bin/activate
```

Verificamos que todo este bien:

```bash
which pip
```
Respuesta esperada:
```commandline
/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/C6venv/bin/pip
```
Vemos nuestras bibliotecas instaladas:
```bash
pip freeze
```
Respuesta esperada:
```commandline

```

Como buena práctica debemos actualizar `pip` antes que anda

```bash
pip install --upgrade pip
```
Respuesta esperada:

```commandline
Collecting pip
  Using cached https://files.pythonhosted.org/packages/ae/db/a8821cdac455a1740580c92de3ed7b7f257cfdbad8b1ba8864e6abe58a08/pip-23.1-py3-none-any.whl
Installing collected packages: pip
  Found existing installation: pip 10.0.1
    Uninstalling pip-10.0.1:
      Successfully uninstalled pip-10.0.1
Successfully installed pip-23.1
```

Excelente ya podemos continuar con la instalación de los `requirements.txt`

A lo largo del curso estaré poniendo individualmente los requisitos de bibliotecas que hay que ir instalando para que no
parezca que mágicamente ya tienes todo el ambiente listo para usarse, sino que a medida que fuiste avanzando con el curso 
fuiste instalando lo que ibas necesitando a medida que lo ibas utilizando.

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

Este es el primer ejemplo en código que veremos en este curso, es necesario que tengas activado tu entorno virtual que en este ejemplo
llamamos `C6venv` y procedas con instalar las siguientes dependencias:

```bash
pip install imageio==2.27.0
pip install opencv-python==4.5.5.62
pip install matplotlib==3.5.3
```

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

Para este ejercicio NO es necesario que instales más dependencias de las que ya instalamos la clase pasada.

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

En general los dataset pueden ser clasificados de dos maneras: De propósito general y particular o especializado.

Entre los dataset de propósito general podemos mencionar a los siguientes:

![6.png](ims%2F2b%2F6.png)

- `COCO dataset:` COCO (Common Objects in Context) es un conjunto de datos ampliamente utilizado para la detección y segmentación de objetos en imágenes. Contiene más de 200,000 imágenes anotadas en 80 categorías de objetos, lo que lo convierte en uno de los conjuntos de datos más grandes y desafiantes en términos de diversidad de objetos y contextos. COCO dataset es utilizado para evaluar y entrenar modelos en tareas como detección de objetos, segmentación semántica, y clasificación de imágenes.

- `Pascal VOC:` Pascal VOC (Visual Object Classes) es un conjunto de datos popularmente utilizado para la detección y clasificación de objetos en imágenes. Pascal VOC contiene alrededor de 20,000 imágenes anotadas en 20 categorías de objetos, y ha sido utilizado en desafíos de detección de objetos como el Desafío de Clasificación de Objetos PASCAL VOC, que ha impulsado el desarrollo de muchas técnicas de detección de objetos populares.

- `ImageNet:` ImageNet es un conjunto de datos masivo que contiene millones de imágenes anotadas en más de 1,000 categorías de objetos. Ha sido ampliamente utilizado para el entrenamiento de modelos de aprendizaje profundo, incluyendo modelos de clasificación de imágenes, detección de objetos, y segmentación semántica. ImageNet es conocido por su desafío anual, el Desafío de Clasificación de Imágenes ImageNet, que ha impulsado el avance en la precisión de clasificación de imágenes.

Por otro lado, por mencionar algunos de los dataset especializados tenemos:

![7.png](ims%2F2b%2F7.png)

- `KITTI:` El conjunto de datos KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute) es ampliamente utilizado para la detección, seguimiento y predicción de objetos en escenarios de conducción autónoma. Contiene imágenes estéreo, datos LIDAR y datos de sensores de otros vehículos, anotados con información detallada de objetos como automóviles, peatones y ciclistas en situaciones de tráfico real. KITTI se utiliza para evaluar y entrenar modelos en tareas de detección de objetos en el contexto de la conducción autónoma.

- `nuScenes:` El conjunto de datos nuScenes es otro conjunto de datos utilizado para la detección y seguimiento de objetos en el contexto de la conducción autónoma. Contiene más de 1,000 horas de datos de sensores, incluyendo imágenes, datos LIDAR y datos de sensores de otros vehículos, anotados con información detallada de objetos en entornos urbanos y suburbanos. nuScenes es conocido por su desafío anual de detección de objetos, que impulsa el desarrollo de algoritmos y modelos para la conducción autónoma.

- `VisDrone:` El conjunto de datos VisDrone es utilizado para la detección y seguimiento de objetos en el contexto de aplicaciones de vigilancia y monitoreo aéreo. Contiene imágenes y videos de drones anotados con información de objetos como peatones, vehículos, y ciclistas en diversos escenarios urbanos y rurales. VisDrone es utilizado para evaluar y entrenar modelos en tareas de detección de objetos en entornos aéreos, lo que lo hace relevante para aplicaciones de vigilancia y monitoreo.

### ¿Dónde podemos conseguir datasets?

Ya tenemos una entrada en este repositorio sobre este tema puedes consultar la información en: [Aprende a buscar bases de datos para deep learning](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow#27-aprende-a-buscar-bases-de-datos-para-deep-learning).
Sin embargo, aquí volvemos a mencionar los más populares:

Para cada problema de DL es necesario tener un dataset que se acople a la necesidad del mismo, entre mayor sea la calidad del dataset
mejores serán los resultados del modelo creado a partir del mismo. Algunas de las plataformas libres que podemos utilizar 
para obtener dataset son los siguientes:

- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/overview?hl=es-419)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Papers with Code](https://paperswithcode.com/datasets)
- [Hugging Face Datasets](https://huggingface.co/datasets)

### Dataset de prueba

Para continuar con las siguientes clases, estaremos utilizando el siguiente dataset: [Self-Driving Cars](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

![8.png](ims%2F2b%2F8.png)

El cual consta de las siguientes labels: `classic_id labels: 'car', 'truck', 'pedestrian', 'bicyclist', 'light'`


## 2.8 Carga de dataset de object detection

Para correr el siguiente código debes instalar las siguientes bibliotecas:

```bash
pip install pandas==1.1.5
pip install object-detection==0.0.3
```

> ## Nota:
> El código completo se encuentra: [carga.py](Detecci%C3%B3n%20de%20Objetos%2F3%20Carga%20de%20dataset%2Fcarga.py)
> El código incluye las secciones 2.8, 2.9 y 2.10 sin embargo, para fines didacticos cada sección tendrá su explicación
> de código individual

Una vez que hemos descargado el dataset anterior y descomprimido en una ruta que conozcamos podemos empezar este tutorial, 
yo usare un `config.ini` para guardar la ruta de dónde descargue el dataset, pero esto `NO` es necesario.

Vamos a empezar observando una imagen de referencia, importamos bibliotecas necesarias:

```python
import cv2
import matplotlib.pyplot as plt
```

Vamos a observar una imagen del dataset:

```python
root = read_ini()["dataset"]  # esto NO es necesario, aquí podrías poner una ruta por ejemplo "proyecto/"

img = cv2.imread(f'{root}/images/1479506176491553178.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.savefig("ejemplo.png")
plt.close()
print("img shape:", img.shape)
```
Respuesta esperada:
```commandline
img shape: (300, 480, 3)
```
![ejemplo.png](Detecci%C3%B3n%20de%20Objetos%2F3%20Carga%20de%20dataset%2Fejemplo.png)

Genial, ya sabemos que cada imagen tiene dimensiones de `480 ancho x 300 largo`


## 2.9 Exploración del dataset de object detection

Ya conocemos las imágenes del dataset, pero necesitamos conocer también las labels correspondientes a la clase de cada objeto
y las cordenadas asignadas a dicho objeto:

```python
data = pd.read_csv(f"{root}/labels_train.csv")
print(data)
print("train images", len(data["frame"].unique()))
```
Respuesta esperada:
```commandline
                          frame  xmin  xmax  ymin  ymax  class_id
0       1478019952686311006.jpg   237   251   143   155         1
1       1478019952686311006.jpg   437   454   120   186         3
2       1478019953180167674.jpg   218   231   146   158         1
3       1478019953689774621.jpg   171   182   141   154         2
4       1478019953689774621.jpg   179   191   144   155         1
...                         ...   ...   ...   ...   ...       ...
132401  1479506176491553178.jpg   166   186   139   156         1
132402  1479506176491553178.jpg   182   204   142   153         1
132403  1479506176491553178.jpg   239   261   139   155         1
132404  1479506176491553178.jpg   259   280   139   157         1
132405  1479506176491553178.jpg   284   324   137   168         1

[132406 rows x 6 columns]
train images 18000
```
Podemos observar que tenemos `18000` imágenes pero `132406` muestras etiquetadas, esto se debe a que una misma imagen puede contener
varios objetos etiquetados. Esto lo podemos ver en el ejemplo: `1478019952686311006.jpg ` el cual contiene 2 objetos pertenecientes a las clases `1, 3`

Como humanos que somos es mucho más comodo leer el nombre de la clase en lugar del id de la misma. Es por ello que vamos a crear un `labelmap` que mapee
el id_label al nombre del label. Esto nos será de utilidad para mostrar el nombre de cada objeto detectado más adelante.
```python
labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
            4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}
```

Ahora vamos a crear una función que nos permita obtener el `bounding box` de cada imagen de referencia. Antes de continuar es
`INDISPENSABLE` entender que cada imagen tiene un nombre por ejemplo: `1478019952686311006.jpg` dicho nombre corresponde con
una única imagen, sin embargo, esta imagen puede contener varios bounding boxes: `[[237   251   143   155], [437   454   120   186]]`
y a su vez cada bounding box puede estar asociado a un `class_id` diferente: `[[1], [3]]` entendiendo esto será mucho más sencillo
programar el código que nos permita crear un diccionario cuya clave sea el nombre de la imagen y valor sea otro diccionario que contenga
los bounding boxes y class_id.

Empecemos definiendo nuestra función:

```python
def get_bounding_boxes(mask: tuple, width: int, height: int) -> dict:
    """
    Returns a dictionary with the correspondent bounding boxes
    :param mask: tuple specifying the `class_id` you want to keep. Example: (1, 2, 3, 4, 5)
    :param width: Image width
    :param height: Image Height
    :return: gt_boxes
    """
    for index, row in data.iterrows():
        id_label = row["class_id"]
        if id_label in mask:
```

`data` es el dataset que hemos leído con pandas, usamos iterrows para ir de una fila en una. 

Ahora vamos a obtener el bounding box normalizado (dividiendo entre las dimensiones de la imagen)

```python
            bbox = np.array([[row['ymin'] / height, row['xmin'] / width, row['ymax'] / height, row['xmax'] / width]],
                            dtype=np.float32)

            im_name = row['frame']
```
Ahora viene la parte más interesante de la lógica de este código:
Sí es la primera vez que vemos el nombre de la imagen, entonces vamos a añadirla como una `key` de nuestro diccionario `get_boxes`
y como valor va a tener un diccionario que contenga dos nuevas `keys`: `boxes & ids` cuyo valor inicial será un numpy array que contenga
el valor de las coordenadas normalizadas del bounding box y class_id:

```python
            if im_name not in gt_boxes:
                gt_boxes[im_name] = {"boxes": np.array(bbox),
                                     "ids": np.array([id_label])}
```
Ahora, pregunta, qué pasa si `im_name` ya existe en `gt_boxes` esto significa que la imagen ya ha sido vista anteriormente y por lo mismo ya
existe un diccionario con los keys `boxes & ids` entonces para añadir nuevos valores debemos hacer un `numpy.append`
```python
            else:
                gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                     "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

```

Excelente, al final el código completo queda de la siguiente manera:

```python
def get_bounding_boxes(mask: tuple, width: int, height: int) -> dict:
    """
    Returns a dictionary with the correspondent bounding boxes
    :param mask: tuple specifying the `class_id` you want to keep. Example: (1, 2, 3, 4, 5)
    :param width: Image width
    :param height: Image Height
    :return: gt_boxes
    """
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["class_id"]
        if id_label in mask:
            bbox = np.array([[row['ymin'] / height, row['xmin'] / width, row['ymax'] / height, row['xmax'] / width]],
                            dtype=np.float32)

            im_name = row['frame']
            if im_name not in gt_boxes:
                gt_boxes[im_name] = {"boxes": np.array(bbox),
                                     "ids": np.array([id_label])}
            else:
                gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                     "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes
```

Ahora en nuestro código principal solo hace falta llamar a la función `get_bounding_boxes`:

```python
im_width, im_height = img.shape[1], img.shape[0]
get_boxes = get_bounding_boxes(mask=(1, 3), width=im_width, height=im_height)
```
Para este ejemplo nos enfocamos únicamente en las clases `1, 3` que corresponden a `coches y caminantes`.

En la siguiente clase vamos a crear una función para plotear esta información.

## 2.10 Visualización de bounding boxes en el dataset de object detection

Hasta este momento este es nuestro código:

```python
import cv2
import matplotlib.pyplot as plt
import pandas as pd

def get_bounding_boxes(mask: tuple, width: int, height: int) -> dict:
    """
    Returns a dictionary with the correspondent bounding boxes
    :param mask: tuple specifying the `class_id` you want to keep. Example: (1, 2, 3, 4, 5)
    :param width: Image width
    :param height: Image Height
    :return: gt_boxes
    """
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["class_id"]
        if id_label in mask:
            bbox = np.array([[row['ymin'] / height, row['xmin'] / width, row['ymax'] / height, row['xmax'] / width]],
                            dtype=np.float32)

            im_name = row['frame']
            if im_name not in gt_boxes:
                gt_boxes[im_name] = {"boxes": np.array(bbox),
                                     "ids": np.array([id_label])}
            else:
                gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                     "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes

if __name__ == '__main__':
    root = read_ini()["dataset"]

    img = cv2.imread(f'{root}/images/1479506176491553178.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.savefig("ejemplo.png")
    plt.close()
    print("img shape:", img.shape)

    data = pd.read_csv(f"{root}/labels_train.csv")
    print(data)
    print("train images", len(data["frame"].unique()))
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    im_width, im_height = img.shape[1], img.shape[0]
    get_boxes = get_bounding_boxes(mask=(1, 3), width=im_width, height=im_height)
```

Vamos a terminar este código incluyendo una funcionalidad de mostrar los bounding boxes de cada imagen así como su `confidence score` y `label`:

Debemos tener instalado la siguiente biblioteca:

```bash
pip install object-detection
```

Importemos otras bibliotecas necesarias para este paso:

```python
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
```
Creemos nuestra función auxiliar para plotear imágenes de object detection:

```python
def plot_example(boxes: dict, limit: int, layout: tuple) -> None:
    """
    Makes an image with multiple detection object examples
    :param boxes: Dictionary with the image name as key and bounding boxes and label ids as a value
    :param limit: Number of examples to include
    :param layout: Distribution of the examples. Example given a limit of 4 a valid layout could be (1, 4), (2, 2) (4, 1)
    :return: None
    """
    limit_images = limit
    i_image = 0
    plt.figure(figsize=(30, 30))
    for key, value in boxes.items():
        bboxes = value["boxes"]
        classes = value["ids"]
```
Recordemos que `boxes`es un diccionario `key, value` donde `key` es el nombre de la imagen y `value` es otro diccionario.
Aquí estamos accediendo a los `bounding boxes` y `labels ids` de cada `imagen` disponible. Vamos ocupar `key`como el nombre de la imagen
y usamos `cv2` para leer la imagen y llevarla a `RGB`.

```python
    im = cv2.imread(root + "/images/" + key)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dummy_scores = np.ones(shape=[bboxes.shape[0]], dtype=np.float32)
```
Creamos una mini función auxiliar para dibujar el recuadro de la bounding box:
```python
def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux
```
Y terminamos nuestra función principal `plot_example` usando nuestra nueva función auxiliar `plot_detections`:
```python
        a = plot_detections(im, bboxes, classes, dummy_scores, labelmap)  # Aqui se obtiene una imagen con su respectivo bounding box
        plt.subplot(layout[0], layout[1], i_image + 1)   # Esto crea el layout de sub imagenes
        plt.imshow(a)
        if i_image >= limit_images-1:  # Detenemos la creación de imagenes
            break
        i_image += 1
```
Nuestro código completo queda de la siguiente manera:
```python
def plot_example(boxes: dict, limit: int, layout: tuple) -> None:
    """
    Makes an image with multiple detection object examples
    :param boxes: Dictionary with the image name as key and bounding boxes and label ids as a value
    :param limit: Number of examples to include
    :param layout: Distribution of the examples. Example given a limit of 4 a valid layout could be (1, 4), (2, 2) (4, 1)
    :return: None
    """
    limit_images = limit
    i_image = 0
    plt.figure(figsize=(30, 30))
    for key, value in boxes.items():
        bboxes = value["boxes"]
        classes = value["ids"]
        im = cv2.imread(root + "/images/" + key)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dummy_scores = np.ones(shape=[bboxes.shape[0]], dtype=np.float32)
        a = plot_detections(im, bboxes, classes, dummy_scores, labelmap)
        plt.subplot(layout[0], layout[1], i_image + 1)
        plt.imshow(a)
        if i_image >= limit_images-1:
            break
        i_image += 1

    plt.savefig("object_detection.png")
    plt.close()

```

Y ya solamente falta en nuestro código principal llamar a la función `plot_example`

```python
if __name__ == '__main__':
    root = read_ini()["dataset"]

    img = cv2.imread(f'{root}/images/1479506176491553178.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.savefig("ejemplo.png")
    plt.close()
    print("img shape:", img.shape)

    data = pd.read_csv(f"{root}/labels_train.csv")
    print(data)
    print("train images", len(data["frame"].unique()))
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    im_width, im_height = img.shape[1], img.shape[0]
    get_boxes = get_bounding_boxes(mask=(1, 3), width=im_width, height=im_height)
    plot_example(boxes=get_boxes, limit=20, layout=(5, 4))
```
Respuesta esperada:

![object_detection.png](Detecci%C3%B3n%20de%20Objetos%2F3%20Carga%20de%20dataset%2Fobject_detection.png)

Excelente hemos terminado esta sección exitosamente.

## 2.11 Aumentando de datos con Albumentation

A lo largo de este repositorio hemos tocado el tema de la importancia de `data augmentation` como una técnica para reducir
el `overfitting` de un modelo. Hemos hablado de las ventajas que tiene y de porque es una técnica muy importante cuando contamos
con un dataset que contiene pocas muestras etiquetadas. Entradas anteriores a este tema:

- [Basic Data Augmentation](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales#data-augmentation)
- [Data Agumentation](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales#6optimizaci%C3%B3n-de-una-red-neuronal-convolucional)
- [Métodos de Regularizacion](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow#31-m%C3%A9todos-de-regularizaci%C3%B3n-overfitting-y-underfitting)


![9.png](ims%2F2b%2F9.png)

En el enfoque de `object detection` podemos hablar sobre `data augmentation` de la siguiente manera:

> Data augmentation, es una técnica comúnmente utilizada en la detección de objetos para aumentar la cantidad y diversidad de datos de entrenamiento. Esto ayuda a mejorar la capacidad del modelo para generalizar a datos de prueba no vistos previamente. 

Aquí hay un resumen general de cómo se usa la ampliación de datos en la detección de objetos:

![10.png](ims%2F2b%2F10.png)

- `Carga de datos:` Primero, carga tus datos de entrenamiento, que generalmente incluyen imágenes y anotaciones de caja delimitadora que indican la ubicación de los objetos de interés en las imágenes.

- `Definición de transformaciones:` A continuación, define las transformaciones que deseas aplicar a tus imágenes y anotaciones. Estas transformaciones pueden incluir rotación, cambio de escala, recorte, volteo horizontal, cambio de brillo/contraste, entre otros.

- `Aplicación de transformaciones:` Aplica las transformaciones definidas a tus imágenes y anotaciones para generar imágenes y anotaciones aumentadas. Esto se puede hacer utilizando bibliotecas de procesamiento de imágenes en Python, como OpenCV, o bibliotecas específicas de ampliación de datos, como Albumentations.

- `Actualización de anotaciones:` Asegúrate de actualizar las anotaciones de caja delimitadora después de aplicar las transformaciones, ya que las ubicaciones de los objetos pueden haber cambiado debido a las transformaciones aplicadas. Esto implica ajustar las coordenadas de las cajas delimitadoras para que coincidan con las nuevas ubicaciones de los objetos en las imágenes aumentadas.

![11.png](ims%2F2b%2F11.png)

Para este curso vamos a utilizar la biblioteca de `Albumentation` la cual es compatible con los Frameworks de `TensorFlow`y `PyTorch`

```python
pip install albumentations==1.3.0
```

## 2.12 Implementando Albumentation en object detection

> ## Nota:
> El código completo lo puedes encontrar [DataAugmentation.py](Detecci%C3%B3n%20de%20Objetos%2F4%20Albumentation%2FDataAugmentation.py)
> Para efectos prácticos se explicará el código por bloques.

Vamos a empezar con una base de código similar a la del código anterior:

```python
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from object_detection.utils import visualization_utils as viz_utils

def get_bounding_boxes(mask: tuple, width: int, height: int) -> dict:
    """
    Returns a dictionary with the correspondent bounding boxes
    :param mask: tuple specifying the `class_id` you want to keep. Example: (1, 2, 3, 4, 5)
    :param width: Image width
    :param height: Image Height
    :return: gt_boxes
    """
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["class_id"]
        if id_label in mask:
            bbox = np.array([[row['ymin'] / height, row['xmin'] / width, row['ymax'] / height, row['xmax'] / width]],
                            dtype=np.float32)

            im_name = row['frame']
            if im_name not in gt_boxes:
                gt_boxes[im_name] = {"boxes": np.array(bbox),
                                     "ids": np.array([id_label])}
            else:
                gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                     "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes

def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux

if __name__ == '__main__':

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/dataset"

    data = pd.read_csv(f"{root}/labels_train.csv", nrows=2)
    get_boxes = get_bounding_boxes(mask=(1, 3), width=480, height=300)
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    print(data)
    print(get_boxes)
```
Partiendo de este código base que es de la clase anterior vamos a agregar el código de `Albumentation`. Primero para efectos prácticos
vamos a notar que en `data` solo incluimos los primeros 2 `rows` que corresponden a la siguiente información:

```commandline
                     frame  xmin  xmax  ymin  ymax  class_id
0  1478019952686311006.jpg   237   251   143   155         1
1  1478019952686311006.jpg   437   454   120   186         3
```

Observamos que es una única imagen que contiene 2 objetos pertenecientes a 2 clases diferentes. Al mismo tiempo observamos que
su correspondiente `get_boxes` es el siguiente:

```commandline
{'1478019952686311006.jpg': {'boxes': array([[0.47666666, 0.49375   , 0.51666665, 0.5229167 ],
       [0.4       , 0.91041666, 0.62      , 0.9458333 ]], dtype=float32), 'ids': array([1, 3])}}
```
De nuevo un único `key` con el nombre de la imagen, 2 boxes con las coordenadas normalizadas y 2 ids.

Antes de continuar vamos a leer la documentación de `Albumentation`: [Albumentations](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#albumentations)
Que tiene la siguiente información:

```commandline
albumentations is similar to pascal_voc, because it also uses four values [x_min, y_min, x_max, y_max] to represent a bounding box. But unlike pascal_voc, albumentations uses normalized values. To normalize values, we divide coordinates in pixels for the x- and y-axis by the width and the height of the image.

Coordinates of the example bounding box in this format are [98 / 640, 345 / 480, 420 / 640, 462 / 480] which are [0.153125, 0.71875, 0.65625, 0.9625].

Albumentations uses this format internally to work with bounding boxes and augment them.
```

Podemos observar que claramente el formato que necesita `albumentation` para los `bounding box`es: `[x_min, y_min, x_max, y_max]` pero esto es un inconveniente porque nuestro dataset
y función de `plot_detections` usan el formato: `[xmin,  xmax,  ymin,  ymax]`. Esto es importante tenerlo en cuenta para una
limpia implementación de `albumentation`. 

Finalmente, debes leer la documentación de `albumentation` para entender mejor su composición y no solo copies y pegues el siguiente
código sin entenderlo. Lo sé, es lento, pero te permitirá crear tus propias implementaciones sin solo copiar y pegar los siguientes ejemplos. [Bounding boxes augmentation for object detection](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/).

Lo más interesante es lo siguiente: `Albumentation` soporta 4 tipos de `bboxes`

- pascal_voc
- albumentations
- coco
- yolo

Cada uno con sus respectivas características, con esa información podemos deducir que el formato `albumentations` es el formato
más similar al que estamos usando nosotros y nos servirá de base. La forma más simple de usar `albumentations` es la siguiente:

```python
transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```

`Compose` nos permite hacer un `pipeline` de transformaciones, la pregunta es ¿Qué transformaciones puedo hacer? Te toca leer 
el siguiente documento para saber más información [Transforms (augmentations.transforms)](https://albumentations.ai/docs/api_reference/augmentations/transforms/)

Ahora tenemos todo el conocimiento necesario para implementar nuestro código:

Empezamos importando nuestrta biblioteca extra:

```python
import albumentations as A
```

Creamos nuestro `transform` como en el ejemplo anterior:

```python
transform = A.Compose([A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.8),
                           A.ShiftScaleRotate(scale_limit=0.9, rotate_limit=10, p=0.7)],
                          bbox_params=A.BboxParams(format='albumentations'))
```

Ahora vamos a crear una función que nos permita crear datos aumentados a partir de una imagen fuente, pero antes debemos convertir
las coordenadas de nuestro `bbox` - `[xmin, xmax, ymin, ymax]` al formato `[x_min, y_min, x_max, y_max]`:

```python
def transform_boxes(bboxes):
    return np.array([[b[1], b[0], b[3], b[2]] for b in bboxes])
```

Excelente ahora podemos pasar nuestro formato de `bbox` al de `albumentations` y viceversa. Ya podemos empezar a aplicar
la `data augmentation`.

```python
def apply_augmentation(img: np.array, bboxes: dict, q: int) -> dict:
    """
    Make q new images using a transform object
    :param img: original image to be transformed
    :param bboxes: original bounding boxes including classes id
    :param q: number of image to create from the original one
    :return: dict
    """

    ab_box = transform_boxes(bboxes["boxes"])
    label = bboxes["ids"]
    ab_box = [list(row) + [label[i]] for i, row in enumerate(ab_box)]
    print(ab_box)
```

El objetivo de esta función es transformar nuestro formato:
```commandline
{'boxes': array([[0.47666666, 0.49375   , 0.51666665, 0.5229167 ],
       [0.4       , 0.91041666, 0.62      , 0.9458333 ]], dtype=float32), 'ids': array([1, 3])}
```
Al formato que acepta `albumentation` el cual es el siguiente:
```commandline
[[0.49375   , 0.47666666, 0.5229167, 0.51666665, 1], 
 [0.91041666,        0.4, 0.9458333,       0.62, 3]]
```
Podemos observar que los primeros 4 valores de cada row corresponden a las coordenadas del `bounding box` y el 5 valor es
el `class id`. Vamos a generar las imágenes alteradas con nuestro objeto `transform`:

```python
    images_, boxes_, labels_ = [], [], []
    for _ in range(q):
        transformed = transform(image=img, bboxes=ab_box)
        transformed_image = transformed['image']
        transformed_bboxes = transform_boxes(transformed['bboxes'])
        images_.append(transformed_image)
        boxes_.append(transformed_bboxes)
        labels_.append(label)
```
Excelente con esta porción de código podemos crear `q` imágenes nuevas con sus respectivos `bboxes` solo que debemos regresar
estos valores a nuestro formato normal, por eso guardamos los `transformed_bboxes` con la función `transform_boxes`.

Nuestra función Global es la siguiente:

```python
def apply_augmentation(img: np.array, bboxes: dict, q: int) -> dict:
    """
    Make q new images using a transform object
    :param img: original image to be transformed
    :param bboxes: original bounding boxes including classes id
    :param q: number of image to create from the original one
    :return: dict
    """

    ab_box = transform_boxes(bboxes["boxes"])
    label = bboxes["ids"]
    ab_box = [list(row) + [label[i]] for i, row in enumerate(ab_box)]
    print(ab_box)
    images_, boxes_, labels_ = [], [], []
    for _ in range(q):
        transformed = transform(image=img, bboxes=ab_box)
        transformed_image = transformed['image']
        transformed_bboxes = transform_boxes(transformed['bboxes'])
        images_.append(transformed_image)
        boxes_.append(transformed_bboxes)
        labels_.append(label)
    return {"images": images_, "bboxes": boxes_, "labels": labels_}
```

Volvemos a nuestro `main` y vamos a usar nuestra nueva radiante función `apply_augmentation`:

```python
if __name__ == '__main__':

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/dataset"

    data = pd.read_csv(f"{root}/labels_train.csv", nrows=2)
    get_boxes = get_bounding_boxes(mask=(1, 3), width=480, height=300)
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    print(data)
    print(get_boxes)
    transform = A.Compose([A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.8),
                           A.ShiftScaleRotate(scale_limit=0.9, rotate_limit=10, p=0.7)],
                          bbox_params=A.BboxParams(format='albumentations'))

    for key, value in get_boxes.items():
        image = cv2.imread(root + "/images/" + key)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ans = apply_augmentation(image, value, 3)
```
Ahora solamente necesitamos crear una función que nos permita visualizar nuestros datos. Esto lo veremos en la siguiente clase.


## 2.13 Visualizando imágenes con aumentado de datos

Para este momento tenemos el siguiente código completo:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
import albumentations as A


def get_bounding_boxes(mask: tuple, width: int, height: int) -> dict:
    """
    Returns a dictionary with the correspondent bounding boxes
    :param mask: tuple specifying the `class_id` you want to keep. Example: (1, 2, 3, 4, 5)
    :param width: Image width
    :param height: Image Height
    :return: gt_boxes
    """
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["class_id"]
        if id_label in mask:
            bbox = np.array([[row['ymin'] / height, row['xmin'] / width, row['ymax'] / height, row['xmax'] / width]],
                            dtype=np.float32)

            im_name = row['frame']
            if im_name not in gt_boxes:
                gt_boxes[im_name] = {"boxes": np.array(bbox),
                                     "ids": np.array([id_label])}
            else:
                gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                     "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes


def transform_boxes(bboxes):
    return np.array([[b[1], b[0], b[3], b[2]] for b in bboxes])


def apply_augmentation(img: np.array, bboxes: dict, q: int) -> dict:
    """
    Make q new images using a transform object
    :param img: original image to be transformed
    :param bboxes: original bounding boxes including classes id
    :param q: number of image to create from the original one
    :return: dict
    """

    ab_box = transform_boxes(bboxes["boxes"])
    label = bboxes["ids"]
    ab_box = [list(row) + [label[i]] for i, row in enumerate(ab_box)]
    print(ab_box)
    images_, boxes_, labels_ = [], [], []
    for _ in range(q):
        transformed = transform(image=img, bboxes=ab_box)
        transformed_image = transformed['image']
        transformed_bboxes = transform_boxes(transformed['bboxes'])
        images_.append(transformed_image)
        boxes_.append(transformed_bboxes)
        labels_.append(label)
    return {"images": images_, "bboxes": boxes_, "labels": labels_}


def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux


if __name__ == '__main__':

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/dataset"

    data = pd.read_csv(f"{root}/labels_train.csv", nrows=2)
    get_boxes = get_bounding_boxes(mask=(1, 3), width=480, height=300)
    labelmap = {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'truck'}, 3: {'id': 3, 'name': 'pedestrian'},
                4: {'id': 4, 'name': 'bicyclist'}, 5: {'id': 5, 'name': 'light'}}

    print(data)
    print(get_boxes)
    transform = A.Compose([A.HorizontalFlip(p=0.5),
                           A.RandomBrightnessContrast(p=0.8),
                           A.ShiftScaleRotate(scale_limit=0.9, rotate_limit=10, p=0.7)],
                          bbox_params=A.BboxParams(format='albumentations', min_visibility=0.3))

    for key, value in get_boxes.items():
        image = cv2.imread(root + "/images/" + key)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ans = apply_augmentation(image, value, 3)

```

Vamos a crear nuestra última función auxiliar para mostrar nuestro ejemplo de `data augmentation`:

Este código debería ser bastante sencillo de leer para ti si has puesto suficiente atención al proceso de este código. Es una variación
del código de `plotting` de la clase anterior. 

```python
def plot_example(im_to_plot: dict, layout: tuple) -> None:
    """
    Makes an image with multiple detection object examples
    :param im_to_plot: dictionary containing images, bboxes and labels.
    :param layout: Distribution of the examples. Example given a limit of 4 a valid layout could be (1, 4), (2, 2) (4, 1)
    :return: None
    """
    plt.figure(figsize=(30, 30))
    images = im_to_plot["images"]
    bboxes = im_to_plot["bboxes"]
    labels = im_to_plot["labels"]

    for index in range(len(images)):
        img = images[index]
        bbox = bboxes[index]
        label = labels[index]
        dummy_scores = np.ones(shape=[bbox.shape[0]], dtype=np.float32)
        a = plot_detections(img, bbox, label, dummy_scores, labelmap)
        plt.subplot(layout[0], layout[1], index + 1)
        plt.imshow(a)

    plt.tight_layout()
    plt.savefig("object_detection.png")
    plt.close()
```

Ahora para terminar podemos llamar a nuestra función `plot_example`:

```python
plot_example(ans, layout=(3, 1))
```

![object_detection.png](Detecci%C3%B3n%20de%20Objetos%2F4%20Albumentation%2Fobject_detection.png)

Felicidades ya hemos creado 3 nuevas imágenes con diferentes alteraciones de la imagen original y con sus propias y nuevas
`bounding boxes`.



## 2.14 Utilizando un modelo de object detection pre-entrenado

Antes de poder correr con este código será necesario hacer una configuración un poco más grande de lo habitual.
Vamos a empezar por instalar una versión de `tensorflow` compatible:

```bash
pip install tensorflow==2.8.0
```

Vamos a dirigirnos a la raiz de mi carpeta `axiliar` en dónde cree mi entorno virtual y descargue el `dataset`

```bash
cd /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow 
```
Creamos una nueva carpeta donde tendremos los datos necesarios para esta clase:

```bash
mkdir modelos
cd modelos
```
Ahora vamos a clonar el repositorio de `models` de `TensorFlow`:
```bash
git clone https://github.com/tensorflow/models
```
Una vez se ha descargado el repositorio éxitosamente debemos compilar un paquete de `protoc`

Primero será necesario instalar `Protobuf` en linux: (el sistema operativo que he usado para todo el repo)

```bash
sudo apt update
sudo apt install protobuf-compiler
```

Ahora ejecutamos el siguiente código:

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

Excelente, vamos muy bien, pero debemos continuar, ahora nos toca descargar un modelo te tensorflow pre-entrenado para hacer
object detection. Para este ejemplo vamos a usar un modelo `pre-entrenado` de tensorflow: [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)

Vamos a atrás un par de carpetas:
```bash
cd ../..
```
Y ahora descargamos el modelo:
```commandline
wget  http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
rm ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

Esto nos dará la siguiente distribución de carpetas:

```commandline
>models            |
>ssd_resnet50v1_fpn|
-------------------|>checkpoint  |
-------------------|-------------|>checkpoint
-------------------|>saved_model |
-------------------|-------------|>assets         |
-------------------|-------------|>variables      |
-------------------|-------------|saved_model.pb
-------------------|pipeline.config
```
 Por comodidad para más adelante, vamos a copiar un archivo del repositorio de github de tensorflow `models` a nuestro modelo
`ssd_resnet50v1_fpn`. Vamos a copiar `mscoco_label_map.pbtxt`:

```bash
cd /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/modelo/models/research/object_detection/data
cp ./mscoco_label_map.pbtxt ../../../../ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
```

> ## Nota:
> Todo el código de esta sección lo puedes encontrar en: [resnet.py](Detecci%C3%B3n%20de%20Objetos%2FResNet%2Fresnet.py)

Vamos a describir el código mediante su flujo lógico de funcionamiento:

### Ejemplo en código

**Primer paso: Crear la función `load_model()`**

El objetivo de esta función será cargar el modelo y devolver una función que sea capaz de predecir una nueva
clasificación de imagen a través de una función `detect_fn` y también devuelva las etiquetas de las clases para después
poder poner el bounding box de la predicción del modelo con su respectivo nombre de clase.

Partamos de lo más sencillo, dónde está la raíz de todos los demás datos que estaremos utilizando y cuál es el nombre del modelo que
vamos a usar:

```python
def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
```

Ahora debemos tener 3 `paths` muy importantes: 
- `pipeline_config:` donde se encuentra la arquitectura del modelo
- `model_dir:` donde están los pesos pre-entrenados del modelo
- `label_map_path:` donde está el archivo que contiene información de las clases que puede detectar el modelo

```python
import os


def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    pipeline_config = os.path.join(root + '/models/research/object_detection/configs/tf2/' + model_name + '.config')
    model_dir = f"{root}/{model_name}/checkpoint/"
    label_map_path = f'{root}/{model_name}/mscoco_label_map.pbtxt'
```
Ahora podemos cargar la arquitectura del modelo desde su archivo de configuración:

```python
import os
from object_detection.utils import config_util
from object_detection.builders import model_builder


def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    pipeline_config = os.path.join(root + '/models/research/object_detection/configs/tf2/' + model_name + '.config')
    model_dir = f"{root}/{model_name}/checkpoint/"
    label_map_path = f'{root}/{model_name}/mscoco_label_map.pbtxt'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
```
Notamos como para cargar el modelo fue necesaria utilizar `config_util` y `model_builder` métodos provenientes de `object_detection`.

Ahora vamos a crear una función que nos permita a través del un `model` regresar una función que permita al modelo `detectar`:

```python
import tensorflow as tf


def get_model_detection_function(model):
    @tf.function
    def detect_fn_(image):
        image, shape = model.preprocess(image)
        prediction_dict = model.predict(image, shape)
        detections_ = model.postprocess(prediction_dict, shape)

        return detections_

    return detect_fn_
```

Nosotros definimos una función llamada `get_model_detection_function` que toma un `modelo` de TensorFlow como entrada y devuelve otra función llamada `detect_fn_`.

La función `detect_fn_` es una función decoradora `@tf.function`, lo que significa que se compila para el grafo de TensorFlow. En otras palabras, se optimiza para ejecutarse más rápido cuando se llama con datos.

Dentro de la función `detect_fn_`, se llama al método `preprocess` del modelo con la imagen de entrada para prepararla para su procesamiento. Luego, se llama al método `predict` del modelo para realizar la detección de objetos en la imagen. El resultado de la detección de objetos se procesa aún más mediante el método `postprocess` del modelo para obtener los resultados finales.

Finalmente, se devuelve el resultado de la detección de objetos desde la función `detect_fn_`.

Muy bien por el momento nuestro código sigue la siguiente estructura:

```python
import os
from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf

def get_model_detection_function(model):
    @tf.function
    def detect_fn_(image):
        image, shape = model.preprocess(image)
        prediction_dict = model.predict(image, shape)
        detections_ = model.postprocess(prediction_dict, shape)

        return detections_

    return detect_fn_


def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    pipeline_config = os.path.join(root + '/models/research/object_detection/configs/tf2/' + model_name + '.config')
    model_dir = f"{root}/{model_name}/checkpoint/"
    label_map_path = f'{root}/{model_name}/mscoco_label_map.pbtxt'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    detect_fn = get_model_detection_function(detection_model)
```

Pero no es suficiente con tener la arquitectura del modelo o la capacidad de hacer nuevas inferencias si no hemos cargado
los pesos `pre-entrenados` del modelo, para eso tenemos la variable `model_dir` que apunta a su `checkpoint`:

```python
import os
from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf

def get_model_detection_function(model):
    @tf.function
    def detect_fn_(image):
        image, shape = model.preprocess(image)
        prediction_dict = model.predict(image, shape)
        detections_ = model.postprocess(prediction_dict, shape)

        return detections_

    return detect_fn_


def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    pipeline_config = os.path.join(root + '/models/research/object_detection/configs/tf2/' + model_name + '.config')
    model_dir = f"{root}/{model_name}/checkpoint/"
    label_map_path = f'{root}/{model_name}/mscoco_label_map.pbtxt'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    detect_fn = get_model_detection_function(detection_model)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
```

Excelente, ya hemos cargado al modelo y ya podría predecir nuevas imágenes, pero a nosotros como humanos nos gusta más leer
el nombre de las clases que el id de las mismas, entonces vamos a convertir el archivo de `label_map_path` en básicamente un diccionario
que contenga el id de la clase y su nombre.

```python
import os
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
import tensorflow as tf

def get_model_detection_function(model):
    @tf.function
    def detect_fn_(image):
        image, shape = model.preprocess(image)
        prediction_dict = model.predict(image, shape)
        detections_ = model.postprocess(prediction_dict, shape)

        return detections_

    return detect_fn_


def load_model():

    root = "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/" \
           "modelo"
    model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    pipeline_config = os.path.join(root + '/models/research/object_detection/configs/tf2/' + model_name + '.config')
    model_dir = f"{root}/{model_name}/checkpoint/"
    label_map_path = f'{root}/{model_name}/mscoco_label_map.pbtxt'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    detect_fn = get_model_detection_function(detection_model)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index_ = label_map_util.create_category_index(categories)

    return detect_fn, category_index_
```

Genial, ya hemos terminado de definir nuestra función que carga el modelo y regresa una función que es capaz de predecir nuevos
datos y también su category_index.

Ahora creamos una pequeña función auxiliar que nos permita convertir una imagen a tensor y nos devuelva ambos datos:
```python
import tensorflow as tf
import numpy as np
import cv2

def image_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
```

Finalmente, podemos hacer una función que nos permita gráficar los bounding boxes de las imágenes clasificadas de la siguiente manera:

```python
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt


def show_image_classified(image, bboxes, c_index):
    detections = bboxes
    label_id_offset = 1
    image_np_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        c_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.7
    )

    plt.figure(figsize=(12, 16))
    plt.imshow(image_np_with_detections)
    plt.tight_layout()
    plt.savefig("anotacion.png")
    plt.close()
```

Y ahora ya tenemos todos los ingredientes necesarios para clasificar una nueva imagen y visualizar los resultados de la clasificación:

```python
img = cv2.imread("1479506176491553178.jpg")
image_np, input_tensor = image_to_tensor(img)
print("Loaded image")
predict, category_index = load_model()
print("Loaded Model")
boxes = predict(input_tensor)
print("BBoxes generated")
show_image_classified(image_np, boxes, category_index)
print("Image saved")
```
Imagen de entrada:

![1479506176491553178.jpg](Detecci%C3%B3n%20de%20Objetos%2FResNet%2F1479506176491553178.jpg)

Resultado de modelo `pre-entrenado`:

![anotacion.png](Detecci%C3%B3n%20de%20Objetos%2FResNet%2Fanotacion.png)

Felicidades, has llegado hasta este punto :D

## Proyecto: Detección de placas de coche

Para este proyecto vamos a hacer un uso de los conocimientos adquiridos a lo largo de este curso e intentar crear un modelo
de object detection pre-entrenado y hacerle `fine tuning` para que sea capaz de detectar la ubicación de la placa de un coche.

Para eso vamos a empezar con el siguiente dataset: [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).

![1.png](ims%2F3%2F1.png)


Vamos a empezar por crear un archivo [config.ini](Detecci%C3%B3n%20de%20Objetos%2F5%20Proyecto%2Fconfig.ini)
```commandline
[APP]
MODEL = /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/modelo/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint
CONFIG = /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/modelo/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config
LABELMAP = /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/modelo/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/mscoco_label_map.pbtxt
DATASET = /media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/modelo/raw
```
El cual apunte a todos los archivos y rutas que vamos a necesitar para la realización de este proyecto.

Adicionalmente, implementamos [utils.py](Detecci%C3%B3n%20de%20Objetos%2F5%20Proyecto%2Futils.py) para poder hacer una lectura de este documento:
```python
import configparser


def read_ini():
    config = configparser.ConfigParser()
    config.read("config.ini")
    out = {}
    for section in config.sections():
        for key in config[section]:
            out[key] = config[section][key]

    return out["model"], out["config"], out["labelmap"], out["dataset"]
```

Y finalmente en nuestro archivo [main.py](Detecci%C3%B3n%20de%20Objetos%2F5%20Proyecto%2Fmain.py)

```python
from utils import read_ini

model_dir, config_dir, labelmap_dir, dataset_dir = read_ini()

print(dataset_dir)
```
Podemos comprobar que todo funcione correctamente.
Respuesta esperada:
```commandline
/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/modelo/raw
```
Nuestro primer trabajo será convertir la información del dataset en formato `xml` en algo más conocido y fácil de trabajar para nostros
en este caso un `DataFrame` de pandas. Cada etiqueta del dataset tiene la siguiente estructura:

```commandline

<annotation>
    <folder>images</folder>
    <filename>Cars0.png</filename>
    <size>
        <width>500</width>
        <height>268</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>licence</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <difficult>0</difficult>
        <bndbox>
            <xmin>226</xmin>
            <ymin>125</ymin>
            <xmax>419</xmax>
            <ymax>173</ymax>
        </bndbox>
    </object>
</annotation>
```

Creamos nuestra función que nos permita crear nuestro pandas dataframe:

```python
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os

def generate_dataframe():
    _, _, _, dataset_dir = read_ini()

    dataset = {"file": [], "width": [], "height": [], "xmin": [], "xmax": [], "ymin": [], "ymax": []}

    for item in glob.glob(os.path.join(dataset_dir+"/annotations", "*.xml")):
        tree = ET.parse(item)
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
            elif 'width' in elem.tag:
                width = int(elem.text)
            elif 'height' in elem.tag:
                height = int(elem.text)
            elif 'xmin' in elem.tag:
                xmin = int(elem.text)
            elif 'ymin' in elem.tag:
                ymin = int(elem.text)
            elif 'xmax' in elem.tag:
                xmax = int(elem.text)
            elif 'ymax' in elem.tag:
                ymax = int(elem.text)

                dataset['file'].append(filename)
                dataset['width'].append(width)
                dataset['height'].append(height)
                dataset['xmin'].append(xmin/width)
                dataset['ymin'].append(ymin/height)
                dataset['xmax'].append(xmax/width)
                dataset['ymax'].append(ymax/height)

    df = pd.DataFrame(dataset)
    df["label_id"] = 1
    df.to_csv("data/dataset.csv", index=False)
    print(df)

```
Respuesta esperada:
```commandline
            file  width  height  ...      ymin      ymax  label_id
0      Cars0.png    500     268  ...  0.466418  0.645522         1
1      Cars1.png    400     248  ...  0.516129  0.645161         1
2     Cars10.png    400     225  ...  0.022222  0.657778         1
3    Cars100.png    400     267  ...  0.426966  0.490637         1
4    Cars101.png    400     300  ...  0.673333  0.733333         1
..           ...    ...     ...  ...       ...       ...       ...
466   Cars95.png    600     400  ...  0.455000  0.657500         1
467   Cars96.png    400     248  ...  0.508065  0.645161         1
468   Cars97.png    400     300  ...  0.340000  0.636667         1
469   Cars98.png    370     400  ...  0.490000  0.647500         1
470   Cars99.png    375     500  ...  0.396000  0.542000         1

[471 rows x 8 columns]
```

Aprovechando que estamos en [utils.py](Detecci%C3%B3n%20de%20Objetos%2F5%20Proyecto%2Futils.py) vamos a crear nuestras funciones auxiliares: `get_bounding_boxes` y `plot_example`:

```python
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import cv2


def plot_detections(im_array, boxes, classes, scores, category_index):
    aux = im_array.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(aux, boxes, classes, scores, category_index,
                                                        use_normalized_coordinates=True)
    return aux


def plot_example(boxes: dict, limit: int, layout: tuple, dataset_dir: str) -> None:
    labelmap = {1: {'id': 1, 'name': 'plate'}}
    limit_images = limit
    i_image = 0
    plt.figure(figsize=(30, 30))
    for key, value in boxes.items():
        bboxes = value["boxes"]
        classes = value["ids"]
        im = cv2.imread(dataset_dir + "/images/" + key)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dummy_scores = np.ones(shape=[bboxes.shape[0]], dtype=np.float32)
        a = plot_detections(im, bboxes, classes, dummy_scores, labelmap)
        plt.subplot(layout[0], layout[1], i_image + 1)
        plt.imshow(a)
        if i_image >= limit_images-1:
            break
        i_image += 1

    plt.savefig("object_detection.png")
    plt.close()


def get_bounding_boxes(data):
    gt_boxes = {}
    for index, row in data.iterrows():
        id_label = row["label_id"]
        bbox = np.array([[row['ymin'], row['xmin'], row['ymax'], row['xmax']]], dtype=np.float32)
        im_name = row['file']
        if im_name not in gt_boxes:
            gt_boxes[im_name] = {"boxes": np.array(bbox),
                                 "ids": np.array([id_label])}
        else:
            gt_boxes[im_name] = {"boxes": np.append(gt_boxes[im_name]["boxes"], np.array(bbox), axis=0),
                                 "ids": np.append(gt_boxes[im_name]["ids"], np.array([id_label]), axis=0)}

    return gt_boxes

```

Finalmente, en nuestro [main.py](Detecci%C3%B3n%20de%20Objetos%2F5%20Proyecto%2Fmain.py) ya podemos observar a detalle el `bounding box` de nuestros ejemplos:
```python
from utils import read_ini, get_bounding_boxes, plot_example
import pandas as pd


if __name__ == '__main__':
    model_dir, config_dir, labelmap_dir, dataset_dir = read_ini()
    data = pd.read_csv("data/dataset.csv")
    bboxes = get_bounding_boxes(data)
    plot_example(bboxes, 6, (2, 3), dataset_dir)

```
Respuesta esperada:
![object_detection.png](Detecci%C3%B3n%20de%20Objetos%2F5%20Proyecto%2Fobject_detection.png)


## 2.15 Fine-tuning en detección de objetos

## 2.16 Fine-tuning en detección de objetos: carga de datos

## 2.17 Fine-tuning en detección de objetos: data augmentation

## 2.18 Fine-tuning en detección de objetos: entrenamiento

## 2.19 Fine tuning en detección de objetos: visualización de objetos

## 2.20 Quiz módulo object detection

# 3 Segmentación de objetos

## 3.1 Introduciendo la segmentación de objetos

En términos coloquiales, la segmentación de objetos es utilizar diferentes técnicas de `computer vision` para poder clasificar cada uno de los pixeles
de una imagen en una categoría diferente. La segmentación de objetos implica dividir una imagen en regiones que corresponden 
a los diferentes objetos presentes en ella. Existen varios enfoques para la segmentación de objetos, que van desde los 
métodos basados en umbrales simples hasta los algoritmos más complejos basados en `deep learning`. Algunos métodos comunes 
incluyen la segmentación basada en bordes, la segmentación basada en regiones y la segmentación semántica.

![osegmentation.gif](ims%2F3%2Fosegmentation.gif)

### Aplicaciones de Object Detection

Entre los principales casos de usos de `object segmentation` podemos encontrar:

- `Detección y seguimiento de objetos:` La segmentación de objetos se utiliza para identificar y separar los objetos de interés en una imagen o video para su posterior seguimiento. Esto se utiliza comúnmente en aplicaciones como la vigilancia y la seguridad, el seguimiento de vehículos y la navegación autónoma.

- `Reconocimiento facial:` La segmentación de objetos se utiliza para separar las caras de las imágenes de fondo en aplicaciones de reconocimiento facial. Esto puede ser útil en aplicaciones de seguridad y vigilancia, así como en sistemas de autenticación biométrica.

- `Realidad aumentada:` La segmentación de objetos se utiliza para separar objetos específicos de una escena para que puedan ser manipulados y mejorados en una experiencia de realidad aumentada. Esto se utiliza comúnmente en aplicaciones de entretenimiento, publicidad y juegos.

- `Edición de imágenes:` La segmentación de objetos se utiliza en herramientas de edición de imágenes para separar objetos específicos de una imagen para su posterior manipulación. Esto puede incluir la eliminación de objetos no deseados, la selección de objetos específicos para aplicar efectos y la combinación de objetos de diferentes imágenes.

- `Robótica:` La segmentación de objetos se utiliza en aplicaciones de robótica para permitir que los robots detecten y manipulen objetos específicos en su entorno. Esto es útil en aplicaciones como la clasificación y el ensamblaje de piezas en la línea de producción.


### Métricas de Object Detection

Hay varias métricas utilizadas comúnmente en la evaluación de los resultados de la segmentación de objetos. Estas métricas se utilizan para medir la calidad de la segmentación y comparar diferentes modelos y algoritmos. 
A continuación, te presento algunas de las métricas más comunes utilizadas en la segmentación de objetos:

- `Precision:` se refiere a la proporción de píxeles correctamente identificados como objeto en la segmentación. Es una medida de la capacidad del modelo para identificar correctamente los objetos de interés en la imagen.

- `Recall:` se refiere a la proporción de píxeles pertenecientes al objeto que se han identificado correctamente en la segmentación. Es una medida de la capacidad del modelo para identificar todos los píxeles que pertenecen al objeto de interés.

- `F1-score:` es una métrica que combina la precisión y la recuperación en una sola medida. Se calcula como la media armónica de la precisión y la recuperación y se utiliza para evaluar el equilibrio entre la precisión y la recuperación en la segmentación.

- `Intersection Over Union (IoU):` se refiere a la proporción de píxeles de la segmentación que coinciden con los píxeles del objeto real. Es una medida de la superposición entre la segmentación y el objeto real.

- `Mean Surface Difference (MSD):` La MSD es una métrica que mide la diferencia promedio de superficie entre la segmentación y el objeto real. Es una medida de la precisión de la segmentación en términos de la forma y tamaño del objeto.

### Arquitectura Encoder - Decoder

La arquitectura Encoder-Decoder es una de las arquitecturas más utilizadas en la segmentación de objetos mediante el uso de redes neuronales convolucionales profundas. Esta arquitectura se compone de dos partes principales: el Encoder y el Decoder.

![2.png](ims%2F3%2F2.png)

El Encoder es una red neuronal convolucional que se utiliza para extraer características de la imagen de entrada. Generalmente, esta red convolucional es una variante de la popular arquitectura ResNet que tiene una profundidad variable. El Encoder toma como entrada la imagen de entrada y la procesa a través de varias capas convolucionales y de pooling. Cada capa de la red convolucional reduce gradualmente el tamaño espacial de la imagen de entrada mientras aumenta la profundidad de las características extraídas.

Una vez que las características han sido extraídas por el Encoder, se utiliza el Decoder para realizar la segmentación de objetos. El Decoder es una red neuronal convolucional que se utiliza para generar una máscara de segmentación para cada objeto en la imagen de entrada. El Decoder consta de varias capas convolucionales y de upsampling que aumentan gradualmente el tamaño espacial de las características extraídas por el Encoder. La última capa del Decoder produce una máscara de segmentación que se utiliza para identificar los píxeles correspondientes a cada objeto.

## 3.2 Tipos de segmentación y sus arquitecturas relevantes

Si bien existen varias formas de segmentar imágenes que no necesariamente hacen uso de técnicas de deep learning para el enfoque 
de este curso vamos a hablar únicamente de las técnicas principales que sí hacen uso de deep learning. De forma sencilla podemos
decir que existen 2 clases de segmentación la `semantic segmentation` y la `instance segmentation` adicionalmente podemos hablar sobre
`panoptic segmentation` que es una combinación de ambas.

![4.png](ims%2F3%2F4.png)

**Segmentación Semántica:**

- Se utiliza para clasificar y separar diferentes objetos en una imagen en función de su significado semántico.
- Cada píxel de la imagen se clasifica en una clase específica.
- Se basa en el uso de modelos de aprendizaje profundo, como redes neuronales convolucionales.
- Se utiliza en aplicaciones como detección de objetos en imágenes de tráfico o identificación de objetos en imágenes de satélite.

**Segmentación de Instancias:**
- Se utiliza para detectar y separar cada objeto individual en una imagen.
- Cada objeto se separa mediante una máscara binaria única.
- También se basa en el uso de modelos de aprendizaje profundo, como Mask R-CNN.
- Se utiliza en aplicaciones como seguimiento de objetos y detección de objetos en tiempo real.

**Segmentación Panóptica:**
- Combina la segmentación semántica y la segmentación de instancias para detectar y separar objetos en una imagen en función de su significado semántico y su instancia individual.
- Cada objeto se separa mediante una máscara binaria única y se clasifica en una clase semántica específica.
- Se utiliza en aplicaciones como la detección y seguimiento de objetos en videos y la interpretación de escenas en robótica y sistemas autónomos.

### Arquitecturas relevantes

**Fully Convolutional Network (FCN):**

![6.png](ims%2F3%2F6.png)

La primera arquitectura de red neuronal completamente convolucional para la segmentación de objetos.
Convierte una red neuronal convolucional estándar (como VGG16) en una red neuronal convolucional completamente convolucional mediante la eliminación de las capas totalmente conectadas y la adición de capas de convolución transpuestas.
Utiliza la técnica de interpolación bilineal para aumentar el tamaño de la máscara segmentada a la resolución de la imagen original.
Ha sido superado en precisión por otras arquitecturas más avanzadas, como U-Net y DeepLab.

**U-Net:**

![5.png](ims%2F3%2F5.png)

Una arquitectura de red neuronal convolucional que se enfoca en la segmentación de instancias y la detección de bordes.
Se compone de una estructura en forma de U con capas de convolución, deconvolución y unión.
Utiliza una técnica llamada "skip connection" para preservar la información de la imagen original y mejorar la precisión de la segmentación.
Es especialmente efectiva en la segmentación de imágenes biomédicas.

**DeepLab**

![3.png](ims%2F3%2F3.png)

Una arquitectura de red neuronal convolucional profunda que se enfoca en la segmentación semántica.
Utiliza una técnica llamada "atrous convolution" para aumentar el campo receptivo de la red neuronal y mejorar la precisión de la segmentación.
Utiliza una técnica de reducción de dimensionalidad llamada "global pooling" para reducir el número de parámetros y mejorar la eficiencia de la red neuronal.
Existen varias variantes de DeepLab, incluyendo DeepLabv3 y DeepLabv3+.

## 3.3 ¿Cómo es un dataset de segmentación?

A continuación compartimos una serie de ejemplos de datasets de segmentación:

![7.png](ims%2F3%2F7.png)

- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [City Scapes](https://www.cityscapes-dataset.com/)
- [Human Segmentation Dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)
- [CamSeq 2007](https://www.kaggle.com/datasets/carlolepelaars/camseq-semantic-segmentation)
- [COCO](https://cocodataset.org/#home)
- [Pascal](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Kitti](https://www.cvlibs.net/datasets/kitti/)


Para efectos prácticos, en esta sección del curso vamos a utilizar - [CamSeq 2007](https://www.kaggle.com/datasets/carlolepelaars/camseq-semantic-segmentation)

> 101 image pairs for semantic segmentation from the University of Cambridge

![8.png](ims%2F3%2F8.png)


## 3.4 Utilizando un dataset de segmentación de objetos

Ya tenemos conocimiento que el dataset que estaremos utilizando consta de 101 imágenes con sus respectivas etiquetas, que por
el tipo de etiqueta llamaremos `ground truth` puesto que representan una etiqueta pixel a pixel de la imagen original. 

Preparemos nuestro código en python para poder leer de forma correcta estás etiquetas.

Empecemos con el siguiente archivo muy simple: [config.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Fconfig.py)

```python
paths = {"dataset": "/media/ichcanziho/Data/datos/Deep Learning/6 Curso de detección y segmentación de objetos con TensorFlow/Segmentation/dataset"}
```

Lo único que contiene es la ruta a la carpeta donde estaremos guardando todas las imágenes del Dataset `CamSeq 2007`

Ahora vamos a crear un archivo [utils.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Futils.py) donde pondremos
todas las funciones `auxiliares` que vamos a usar para cargar el dataset, generar las particiones y visualizar los datos.

Importamos las bibliotecas necesarias:

```python
from sklearn.model_selection import train_test_split
from config import paths  # aquí guardamos la ruta del dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
```

La primera función que vamos a crear servirá para dividir las imágenes que son de entrada respecto a las que son las `labels`:

```python
def read_dataset():
    dataset_path = paths["dataset"]
    filenames = glob.glob(os.path.join(dataset_path, "*.png"))
    filenames.sort()
    image_list = []
    masks_list = []
    for filename in filenames:
        if filename.endswith("L.png"):
            masks_list.append(filename)
        else:
            image_list.append(filename)
    print(len(image_list), len(masks_list))
    return image_list, masks_list
```

Esta función lee la ruta `dataset` del diccionario `paths` e itera sobre cada elemento y devuelve todos los que terminen en `.png`
el formato de las imágenes de este dataset. Después tenemos dos listas `image_list` y `mask_list` para cada `filename` de 
`filenames` vamos a preguntar si dicho elemento termina en `L.png` indicando que la imagen es una `label` y será agregada a la lista
de `mask_list` de otra forma irá a `image_list`.

Cómo tal el paso anterior solo devuelve listas que contienen las direcciones de las imágenes, pero no son las imágenes en sí, 
aún necesitamos abrirlas y guardar su contenido como un numpy array, para ello vamos a crear la función `load_data`:

```python
def load_data(images_path, masks_path):
    samples = {'images': [], 'masks': []}

    for i in range(len(images_path)):
        img = plt.imread(images_path[i])
        mask = plt.imread(masks_path[i])
        img = cv2.resize(img, (256, 256))
        masks = cv2.resize(mask, (256, 256))

        samples['images'].append(img)
        samples['masks'].append(masks)

    samples = {
        'images': np.array(samples['images']),
        'masks': np.array(samples['masks']),
    }
    return samples
```

El código es muy simple de leer, vemos como el mismo regresa un diccionario que contiene tanto las imágenes reales como las
máscaras de las mismas ya en formato numpy array.

Adicionalmente, vamos a crear 2 funciones que vamos a utilizar más adelante:

```python
def get_partitions(image_list, masks_list, ts=0.2):
    return train_test_split(image_list, masks_list, test_size=ts, random_state=42)
```

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
    ax1.plot(epoch, loss_values, 'go-', label='training')
    ax1.plot(epoch, val_loss_values, 'ro-', label='validation')
    ax2.plot(epoch, metric_values, 'go-', label='training')
    ax2.plot(epoch, val_metric_values, 'ro-', label='validation')
    ax1.legend()
    ax2.legend()
    plt.savefig(f"{fname}")
    plt.close()
```

## 3.5 Visualización de nuestro dataset de segmentación

Antes de continuar con la creación de la arquitectura podemos observar un ejemplo de nuestro dataset ya cargado.
Esto es bastante sencillo gracias a las funciones que ya hemos creado anteriormente. Volvemos al código [utils.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Futils.py) y creamos
un punto de entrada para observar un ejemplo de imagen y su etiqueta correspondiente.

```python
if __name__ == '__main__':
    images, masks = read_dataset()
    train_samples = load_data(images, masks)
    plt.rcParams["figure.figsize"] = [8, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.subplot(1, 2, 1)
    plt.imshow(train_samples['images'][10])
    plt.subplot(1, 2, 2)
    plt.imshow(train_samples['masks'][10])
    plt.savefig("example.png")
```

Respuesta esperada:

![example.png](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Fexample.png)


## 3.6 Creando red neuronal U-Net para segmentación

> U-Net es un tipo de red neuronal convolucional (CNN) utilizada para la segmentación de objetos en imágenes médicas, aunque también puede ser aplicada a otros tipos de imágenes.

![9.png](ims%2F3%2F9.png)

La arquitectura de U-Net consiste en una serie de capas convolucionales, seguidas de capas de pooling, que se encargan de reducir la dimensión de las características aprendidas. Luego, la red se "expande" hacia arriba utilizando capas de convolución transpuestas y concatenación con características de la misma escala aprendidas durante la "contracción". En este proceso de "expansión", la red utiliza la información de las capas de la "contracción" para generar una segmentación detallada del objeto en la imagen.

> Más adelante en la implementación de código vamos a ver que será necesario utiliza `skip connrections` lo cuál implica que 
> no vamos a poder utilizar directamente una estructura `sequential` para definir la arquitectura de la red. Vamos a utilizar
> algo un poco más complejo para poder trabajar con este modelo.
> 

Las `skip-connections` permiten que la información de las capas de "contracción" se transmita directamente a las capas de "expansión". Esto significa que la red tiene acceso a información de diferentes escalas, lo que puede ayudar a la red a segmentar objetos de diferentes tamaños y formas.

Otra característica importante de U-Net es que utiliza una función de pérdida llamada `dice coefficient`, que es una métrica 
comúnmente utilizada en la segmentación de imágenes médicas. La función de pérdida mide la similitud entre la segmentación predicha 
por la red y la segmentación verdadera. Esta función de pérdida ayuda a la red a ajustar sus parámetros de manera que se maximice la similitud entre la segmentación predicha y la verdadera.

Vamos al código: [architecture.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Farchitecture.py) donde plasmaremos todo el código
de la arquitectura de U-NET.

Algo interesante de U-Net es que está compuesta de varios bloques, donde cada bloque tiene una arquitectura similar, entonces podemos
aprovechar este detalle para crear una función que agrupe a los bloques `convolucionales`:

```python
def create_conv_block(input_tensor, num_filters):
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(
        input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(
        x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x
```

Sus parámetros de entrada serán input_tensor, refiriéndose al tamaño de la capa anterior y num_filters, cuantos filtros aplicar en dicho bloque.

Ahora programemos la arquitectura U_net:

Cómo hemos visto, la misma tiene 3 etapas, la de `encoder` y `decoder` y es en el `decoder` donde usamos las `skip connections` es por esto
que necesitamos utilizar el API `FUNCIONAL` de tensorflow en lugar de nuestro amado y conocido API `SECUENCIAL`.


> El API secuencial es la forma más simple y común de definir un modelo en TensorFlow. En este enfoque, se define el modelo como una secuencia de capas en orden lineal. Las capas se agregan una tras otra utilizando el método add() y luego se compila y entrena el modelo utilizando los métodos compile() y fit(), respectivamente.

> Por otro lado, el API funcional permite una mayor flexibilidad en la definición del modelo. En este enfoque, se pueden crear modelos más complejos y con múltiples entradas y salidas, lo que lo hace especialmente útil para modelos que tienen ramificaciones o múltiples salidas. Además, en el API funcional, las capas se pueden conectar arbitrariamente, lo que permite una mayor personalización del modelo.

> En el API funcional, se define el modelo creando una instancia de la clase tf.keras.Model y especificando las entradas y las capas intermedias utilizando funciones como Input() y layers. Luego, se conectan las capas utilizando operaciones de tensor y se especifican las salidas del modelo utilizando la instancia del modelo y las entradas.

```python
def create_unet(input_shape, num_filters=16, dropout=0.1):
    # Encoder
    c1 = create_conv_block(input_shape, num_filters * 1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dropout)(p1)

    c2 = create_conv_block(p1, num_filters * 2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)

    c3 = create_conv_block(p2, num_filters * 4)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)

    c4 = create_conv_block(p3, num_filters * 8)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)

    c5 = create_conv_block(p4, num_filters * 16)
```

Vemos como estamos almacenando cada capa en el proceso `c1, c2, c3, c4, c5`. Observamos como este `Functional API` necesita
como parámetro de entrada de qué capa viene precedida la capa actual. Esto hace la magia, porque no infiere directamente que la
capa anterior forzosamente debe ser la capa de entrada de la capa actual.

Ahora terminemos nuestra arquitectura programando el `decoder`

```python
    # Decoder
    u6 = tf.keras.layers.Convolution2DTranspose(num_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = create_conv_block(u6, num_filters * 8)

    u7 = tf.keras.layers.Convolution2DTranspose(num_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = create_conv_block(u7, num_filters * 4)

    u8 = tf.keras.layers.Convolution2DTranspose(num_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    c8 = create_conv_block(u8, num_filters * 2)

    u9 = tf.keras.layers.Convolution2DTranspose(num_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    c9 = create_conv_block(u9, num_filters * 1)
```

Algo sumamente intersante de esta arquitectura es que necesita de `Convolutional2DTranspose` para regresar al tamaño original de la
imagen y adicionalmente, el uso de las `skip-connections` queda plasmado cuando hacemos el siguiente código:

`tf.keras.layers.concatenate([u6, c4])` aquí literalmente estamos juntando las capas `c4 y u6`.

Finalmente, agreguemos la última capa que será la de clasificación:

```python
    output = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_shape], outputs=[output])

    return model
```

Excelente ya hemos programado manualmente U-NET :D


## 3.7 Entrenando y estudiando una red de segmentación

Creemos el siguiente archivo para entrenar el modelo de U-Net desde 0 [train_model.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Ftrain_model.py)

Empecemos por importar las bibliotecas necesarias:

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from architecture import create_unet
import tensorflow as tf
from datetime import date
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils import read_dataset, get_partitions, load_data, plot_results
import matplotlib.pyplot as plt
import numpy as np
```

Ahora creamos la función que va a crear, compilar y entrenar el modelo:

```python
def create_and_compile_model(train_samples, test_samples):
    inputs = tf.keras.layers.Input((256, 256, 3))
    print("Generating Architecture")
    model = create_unet(inputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)
    print("Model's plot architecture done")
    callback = EarlyStopping(monitor="accuracy", patience=20, mode="auto")
    checkpoint = ModelCheckpoint(filepath="models/best_model.h5", save_best_only=True, save_weights_only=False,
                                 mode="auto", verbose=1, monitor="val_accuracy")
    current_day = date.today().strftime("%dd_%mm_%yyyy")
    tensorboard_cb = TensorBoard(log_dir=f"logs/cnn_model_{current_day}")

    model_history = model.fit(train_samples['images'], train_samples['masks'], epochs=200, verbose=1,
                              validation_split=0.3, callbacks=[callback, checkpoint, tensorboard_cb])

    plot_results(model_history, "accuracy", "conv_results.png")

    score = model.evaluate(test_samples['images'], test_samples['masks'])
    print("="*64)
    print("score", score)
    print("="*64)
    return model
```

Cabe resaltar que estoy usando 3 callbacks para tener un mejor manejo del modelo:

- EarlyStopping
- ModelCheckpoint
- TensorBoard

Ahora definimos un `entry-point` y entrenamos al modelo:

```python
if __name__ == '__main__':
    print("Reading Data")
    images, masks = read_dataset()
    print("Making Partitions")
    train_input_img, val_input_img, train_target_mask, val_target_mask = get_partitions(images, masks)
    train_samples_data = load_data(train_input_img, train_target_mask)
    test_samples_data = load_data(val_input_img, val_target_mask)
    model = create_and_compile_model(train_samples_data, test_samples_data)
```

Es necesario entender el flujo de información, primero `leimos los datos` después creamos particiones de entrenamiento y validación
con la función `get_partitions` y convertimos estas particiones a `samples` que son los diccionarios que ya contienen las 
imágenes en formato numpy array. Finalmente, llamamos a la función `create_and_compile_model` y las callbacks harán el resto por
nosotros.



## 3.8 Generando predicciones con modelo de object segmentation

Vamos a crear 2 funciones auxiliares para predecir nuevas imágenes y graficar un caso de predicción. Estás funciones van a estar
disponibles en: [utils.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Futils.py)

Así que debemos importarlas en nuestro [train_model.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Ftrain_model.py)

```python
from utils import read_dataset, get_partitions, load_data, plot_results, predict_test_samples, plot_images
```

```python
def predict_test_samples(val_map, model_):
    img = val_map['images']
    mask = val_map['masks']

    test_images_ = np.array(img)

    predictions = model_.predict(test_images_)

    return predictions, test_images_, mask


def plot_images(test_image, predicted_maks, ground_truth):
    plt.rcParams["figure.figsize"] = [10, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(predicted_maks)
    plt.title('Predicted mask')

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth)
    plt.title('Ground truth mask')
    plt.savefig("predictions.png")
```

Excelente, ahora podemos entrenar el modelo y observar sus predicciones:

```python
if __name__ == '__main__':
    print("Reading Data")
    images, masks = read_dataset()
    print("Making Partitions")
    train_input_img, val_input_img, train_target_mask, val_target_mask = get_partitions(images, masks)
    train_samples_data = load_data(train_input_img, train_target_mask)
    test_samples_data = load_data(val_input_img, val_target_mask)
    model = create_and_compile_model(train_samples_data, test_samples_data)
    print("Predicting")
    predicted_masks, test_images, ground_truth_masks = predict_test_samples(test_samples_data, model)
    plot_images(test_images[20], predicted_masks[20], ground_truth_masks[20])
```

Respuesta esperada:

![predictions.png](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Fpredictions.png)


Adicionalmente NO olvidemos que usamos `TensorBoard` entonces podemos ver los resultados de entrenamineto a detalle 
[Aquí](https://tensorboard.dev/experiment/kp2z9lHXQuKKIXD5D8MNLg/#scalars)

![tensorboard.png](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Ftensorboard.png)

Aunque un resumen en matplotlib es el siguiente:

![conv_results.png](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fcore%2Fconv_results.png)

Finalmente, como hemos gurdado el mejor modelo de acuerdo al `val_accuracy` podemos cargarlo desde memoria y hacer inferencia
para nuevos archivos. [main.py](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fmain.py)

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from keras.models import load_model
from core.utils import predict_test_samples, plot_images, load_data


if __name__ == '__main__':

    test_sample = load_data(["test/0016E5_08159.png"], ["test/0016E5_08159_L.png"])
    model = load_model("core/models/best_model.h5")
    predicted_masks, test_images, ground_truth_masks = predict_test_samples(test_sample, model)
    plot_images(test_images[0], predicted_masks[0], ground_truth_masks[0])

```

Respuesta esperada:

![predictions.png](Detecci%C3%B3n%20de%20Objetos%2F6%20Segmentation%2Fpredictions.png)



## 3.9 Quiz módulo segmentación

![10.png](ims%2F3%2F10.png)


# 4 Un paso más allá

## 4.1 El estado de la cuestión en computer vision

No podemos dejar de recordar las siguientes herramientas que nos pueden proveer de grandes recursos relacionados a deep learning
incluyendo object segmentation.

![1.png](ims%2F4%2F1.png)

Otras áreas interesantes dentro de Computer Vision con Deep Learning son Modelos de `High Resolution` 

![2.png](ims%2F4%2F2.png)

Y claro no puede faltar `Image generation` con Dall-E-2 Por ejemplo.

![3.png](ims%2F4%2F3.png)