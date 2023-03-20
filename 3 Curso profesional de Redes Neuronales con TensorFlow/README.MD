# Curso profesional de Redes Neuronales con TensorFlow

Ya conoces cómo funcionan las redes neuronales. Incrementa tus habilidades usando TensorFlow y todo su ecosistema de herramientas. Crea modelos de deep learning que podrás poner a funcionar en ambientes profesionales.

- Almacena modelos y reutilízalos.
- Utiliza modelos pre-entrenados con transfer learning.
- Optimiza la precisión de tus modelos de deep learning.
- Pre-procesa datos con Keras datasets y datasets generators.

> ## NOTA:
> Antes de continuar te invito a que revises el curso anterior:
> 
> [Curso de redes neuronales convolucionales con Python y Keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
>
> Este Curso es el Número 3 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso


# Índice:

- [1 Cómo utilizar TensorFlow 2.0 con Python](#1-cómo-utilizar-tensorflow-20-con-python)
	- [1.1 Redes Neuronales con TensorFlow](#11-redes-neuronales-con-tensorflow)
	- [1.2 Introducción a TensorFlow 2.0](#12-introducción-a-tensorflow-20)
- [2 Manejo y preprocesamiento de datos para redes neuronales](#2-manejo-y-preprocesamiento-de-datos-para-redes-neuronales)
	- [2.1 Uso de data pipelines](#21-uso-de-data-pipelines)
	- [2.2 Cómo cargar bases de datos JSON](#22-cómo-cargar-bases-de-datos-json)
	- [2.3 Cargar bases de datos CSV y BASE 64](#23-cargar-bases-de-datos-csv-y-base-64)
	- [2.4 Preprocesamiento y limpieza de datos](#24-preprocesamiento-y-limpieza-de-datos)
	- [2.5 Keras datasets](#25-keras-datasets)
	- [2.6 Datasets generators](#26-datasets-generators)
	- [2.7 Aprende a buscar bases de datos para deep learning](#27-aprende-a-buscar-bases-de-datos-para-deep-learning)
	- [2.8 Cómo distribuir los datos](#28-cómo-distribuir-los-datos)
	- [2.9 Crear la red neuronal, definir capas, compilar, entrenar, evaluar y predicciones](#29-crear-la-red-neuronal-definir-capas-compilar-entrenar-evaluar-y-predicciones)
- [3 Optimización de precisión de modelos](#3-optimización-de-precisión-de-modelos)
	- [3.1 Métodos de regularización: overfitting y underfitting](#31-métodos-de-regularización--overfitting-y-underfitting)
	- [3.2 Recomendaciones prácticas para ajustar un modelo](#32-recomendaciones-prácticas-para-ajustar-un-modelo)
	- [3.3 Métricas para medir la eficiencia de un modelo: Callback](#33-métricas-para-medir-la-eficiencia-de-un-modelo--callback)
	- [3.4 Monitoreo del entrenamiento en tiempo real: early stopping y patience](#34-monitoreo-del-entrenamiento-en-tiempo-real--early-stopping-y-patience)
	- [3.5 kerasTuner: Construyendo el modelo](#35-kerastuner--construyendo-el-modelo)
	- [3.6 KerasTuner: Buscando la mejor configuración para tu modelo](#36-kerastuner--buscando-la-mejor-configuración-para-tu-modelo)
- [4 Almacenamiento y carga de modelos](#4-almacenamiento-y-carga-de-modelos)
	- [4.1 Almacenamiento y carga de modelos: pesos y arquitectura](#41-almacenamiento-y-carga-de-modelos--pesos-y-arquitectura)
	- [4.2 Criterios para almacenar los modelos](#42-criterios-para-almacenar-los-modelos)
- [5 Fundamentos de aprendizaje por transferencia](#5-fundamentos-de-aprendizaje-por-transferencia)
	- [5.1 Introducción al aprendizaje por transferencia](#51-introducción-al-aprendizaje-por-transferencia)
	- [5.2 Cuándo utilizar aprendizaje por transferencia](#52-cuándo-utilizar-aprendizaje-por-transferencia)
	- [5.3 Carga de sistemas pre-entrenados en Keras](#53-carga-de-sistemas-pre-entrenados-en-keras)
	- [5.4 API funcional de Keras](#54-api-funcional-de-keras)
	- [5.5 Uso de sistemas pre-entrenados de TensorFlow Hub](#55-uso-de-sistemas-pre-entrenados-de-tensorflow-hub)
- [6 Resultados de entrenamiento](#6-resultados-de-entrenamiento)
	- [6.1 Introducción a variables relevantes del TensorBoard](#61-introducción-a-variables-relevantes-del-tensorboard)
	- [6.2 Análisis y publicación de resultados del entrenamiento](#62-análisis-y-publicación-de-resultados-del-entrenamiento)
	- [6.3 Introducción al despliegue de modelos en producción](#63-introducción-al-despliegue-de-modelos-en-producción)
	- [6.4 Siguientes pasos con deep learning](#64-siguientes-pasos-con-deep-learning)

# 1 Cómo utilizar TensorFlow 2.0 con Python

## 1.1 Redes Neuronales con TensorFlow

Los requisitos para poder tomar este curso (repositorio de github):

- Fundamentos de redes neuronales
- Uso de Entornos Virtuales en Python
- Creación de proyectos de ciencia de datos e inteligencia Artificial

### Ciclo de vida de un proyecto de Inteligencia Artificial

El ciclo de vida de una IA (Inteligencia Artificial) es un proceso iterativo que consta de varias etapas y que tiene como 
objetivo crear y mejorar un sistema de IA a lo largo del tiempo. A continuación se describen las principales etapas del ciclo 
de vida de una IA:

1. Identificación del problema: se identifica el problema a resolver y se define el objetivo de la IA. Esta etapa implica entender el problema, definir las metas y los requisitos para el sistema de IA y determinar si la IA es la mejor solución para el problema.

2. Adquisición de datos: se recopila y prepara el conjunto de datos necesario para entrenar y validar el modelo de IA. Esta etapa implica identificar las fuentes de datos, recopilar los datos necesarios, limpiar y preprocesar los datos.

3. Preprocesamiento de datos: se realiza una exploración de los datos y se aplican técnicas de preprocesamiento para preparar los datos para su uso en el modelo de IA. Esto incluye la normalización, la reducción de dimensiones y la selección de características.

4. Desarrollo del modelo: se desarrolla el modelo de IA utilizando una arquitectura específica (por ejemplo, redes neuronales convolucionales para clasificación de imágenes). Esta etapa implica el entrenamiento, la validación y la optimización del modelo de IA.

5. Evaluación del modelo: se evalúa la precisión y el rendimiento del modelo de IA utilizando conjuntos de datos de prueba. Esta etapa implica la comparación de los resultados del modelo con los resultados esperados y la identificación de los posibles errores.

6. Implementación del modelo: se implementa el modelo de IA en un entorno de producción y se realiza un seguimiento continuo del rendimiento del modelo en tiempo real. Esto implica la integración con otros sistemas, la monitorización y la actualización del modelo.

7. Mantenimiento del modelo: se realiza un mantenimiento continuo del modelo de IA para garantizar su precisión y rendimiento en el tiempo. Esto incluye la actualización del modelo con nuevos datos, la optimización de la arquitectura y la resolución de problemas de calidad de datos.

![1.png](imgs%2F1%2F1.png)

En resumen, el ciclo de vida de una IA es un proceso iterativo que involucra la identificación del problema, la adquisición y preparación de datos, el desarrollo y entrenamiento del modelo, la evaluación del modelo, la implementación del modelo en un entorno de producción y el mantenimiento continuo del modelo. Este ciclo permite crear y mejorar sistemas de IA de manera efectiva y eficiente.

### ¿Qué vamos a hacer en este curso?

1. Redes neuronales: Llevar redes neuronales a la práctica
2. Cargar bases de datos: conocer diferentes formatos de bases de datos
3. Optimizar modelos: Aumentar el accuracy y reducir la perdida
4. Evitar overfitting y underfittig
5. Transferencia de aprendizaje (transfer learning)
6. Almacenar y cargar modelos

### Proyecto principal del curso:

El objetivo del curso será entrenar una CNN capas de detectar entre 24 letras del vocabulario de lenguaje de señas.
(se evitará el reconocimiento de la J y la Z, pues son letras que necesitan del movimiento de la mano)

![2.png](imgs%2F1%2F2.png)

La base de datos que se usará tiene las siguientes características:

- 27455 imágenes
- Escala de grises
- 28 x 28 píxeles
- 24 clases
- JPEG
- TecPerson - kaggle

### Objetivos del curso

- Cómo cargar tus propias bases de datos
- Cargar bases de datos en formatos como CSV, JSON, BASE64, imágenes
- Aplicar técnicas para optimizar tus modelos
- Agregar métricas en el entrenamiento de tus modelos
- Cargar y guardar modelos
- Auto-tuner de Keras para encontrar mejores variables
- Bases de aprendizaje por transferencia
- Uso de TensorBoard y cómo mostrar tu proyecto al mundo entero
- Tener tu modelo listo para utilizarlo como inferencia

### Consiguiendo datos del proyecto de lenguaje de señas


```bash
mkdir data
cd data
wget --no-check-certificate https://storage.googleapis.com/platzi-tf2/sign-language-img.zip -O sign-language-img.zip
unzip sign-language-img.zip
rm sign-language-img.zip
```

> ## Nota:
> Por temas de almacenamiento con GitHub la carpeta `data` ha sido añadida en el gitignore


## 1.2 Introducción a TensorFlow 2.0

Hay varias librerías de Deep Learning en Python. A continuación se mencionan algunas de las más populares:

1. TensorFlow: es una librería de Deep Learning desarrollada por Google. Es muy utilizada en aplicaciones de visión por computadora, procesamiento de lenguaje natural y otros campos del aprendizaje automático. TensorFlow es conocida por su escalabilidad y su capacidad para trabajar con grandes conjuntos de datos.

2. Keras: es una librería de alto nivel para el desarrollo de modelos de Deep Learning. Keras se enfoca en la simplicidad y facilidad de uso, permitiendo a los desarrolladores crear modelos de IA con pocas líneas de código. Keras también es compatible con TensorFlow y otras librerías de Deep Learning.

3. PyTorch: es una librería de aprendizaje profundo desarrollada por Facebook. PyTorch es conocida por su facilidad de uso y su capacidad para crear modelos de IA en tiempo real. Es muy popular en el ámbito de la investigación y se utiliza en aplicaciones de visión por computadora, procesamiento de lenguaje natural y otros campos.

4. Theano: es una librería de aprendizaje profundo que permite la definición, optimización y evaluación de expresiones matemáticas que involucran matrices multidimensionales. Theano es conocida por su rapidez y eficiencia en la realización de cálculos matemáticos.

5. Caffe: es una librería de aprendizaje profundo que se utiliza principalmente en aplicaciones de visión por computadora. Es conocida por su velocidad y eficiencia, lo que la hace muy útil en aplicaciones en tiempo real.

6. MXNet: es una librería de aprendizaje profundo desarrollada por Amazon. MXNet es conocida por su capacidad de escalabilidad y su eficiencia en la utilización de múltiples procesadores. Es muy popular en aplicaciones de visión por computadora, procesamiento de lenguaje natural y otros campos.

Para este curso nos enfocaremos principalmente en Tensorflow 2.0 + Keras

Sin embargo, Tensorflow No es solo una librería de python es un ecosistema general:

![3.png](imgs%2F1%2F3.png)

Tensorflow nos apoya a llevar el deploy de nuestros modelos no solo en computadoras, sino también en otras plataformas como
celulares, páginas web, crear API, utilizar Arduinos, o cloud computing. 

Para más información puedes visitar: https://www.tensorflow.org/api_docs


# 2 Manejo y preprocesamiento de datos para redes neuronales

En este módulo aprenderemos:

- Cómo cargar bases de datos en formatos CSV, JSON, BASE64, etc
- Pre-procesar los datos
- Cómo cargar datasets de keras
- Dataset Generators
- Cómo cargar tus propios datasets con TF.data
- Cómo distribuir los datos

## 2.1 Uso de data pipelines

Un `Data Pipeline` en Python es un proceso automatizado que permite la ingestión, procesamiento, transformación y almacenamiento de datos en una secuencia ordenada de pasos. El Data Pipeline en Python se utiliza para automatizar el flujo de datos de una fuente a otra, como por ejemplo, desde una base de datos hasta un modelo de aprendizaje automático o un sistema de visualización de datos.

El Data Pipeline en Python generalmente se compone de varios pasos. Estos pasos incluyen la lectura de los datos de una fuente de datos, la limpieza y preprocesamiento de los datos, la transformación de los datos en un formato adecuado para su uso en el modelo de aprendizaje automático, el entrenamiento del modelo, la evaluación del modelo y la visualización de los resultados.

Para implementar un Data Pipeline en Python, se pueden utilizar varias herramientas y librerías, tales como:

- `Pandas:` es una librería de Python que se utiliza para el análisis de datos y el procesamiento de datos en memoria. Pandas ofrece funciones para la limpieza, transformación y filtrado de datos.

- `NumPy:` es una librería de Python que se utiliza para el procesamiento numérico. NumPy ofrece una gran cantidad de funciones matemáticas para el procesamiento de datos.

- `Scikit-Learn:` es una librería de Python que se utiliza para el aprendizaje automático. Scikit-Learn ofrece una gran cantidad de algoritmos de aprendizaje automático para la clasificación, regresión y clustering.

- `TensorFlow:` es una librería de Python que se utiliza para el aprendizaje profundo. TensorFlow ofrece una gran cantidad de herramientas para el desarrollo de modelos de aprendizaje profundo.

![1.png](imgs%2F2%2F1.png)

### Basura que entra basura que sale

El rendimiento de que tan bueno puede ser cualquier modelo de machine learning o deep learning para clasificar un problema
se ve determinado mayoritariamente por la calidad de la base de datos con la que fue entrenado. Si la base, no es buena, entonces
es mejor buscar una base diferente o intentar limpiar lo mejor posible dicha base de datos.

Es muy importante contar con etiquetas de calidad que realmente reflejen el comportamiento deseado por el modelo de clasificación.
Es relevante lidiar con los problemas que conlleva crear una buena base de datos como: lidiar con los datos perdidos, con
los valores atípicos o con datos corruptos. Construir un conjunto de datos adecuado y de alta calidad es fundamental para resolver
cualquier problema de ML o DL a continuación enlisto algunos consejos a tener en cuenta:

1. Definir claramente el objetivo del modelo: Es importante tener claro cuál es el objetivo del modelo de machine learning, ya que esto permitirá determinar los datos necesarios para el entrenamiento. Por ejemplo, si el objetivo es predecir el riesgo de fraude de una transacción, los datos necesarios podrían incluir información de transacciones previas, información financiera y datos de perfil de los usuarios.

2. Recolectar una cantidad suficiente de datos: Es importante tener suficientes datos para el entrenamiento del modelo. La cantidad de datos requeridos depende de la complejidad del problema y la cantidad de características necesarias para el entrenamiento. En general, se recomienda tener al menos unas miles de muestras para entrenar un modelo de machine learning.

3. Seleccionar características relevantes: Es importante elegir características relevantes para el modelo. Las características irrelevantes pueden agregar ruido y afectar el rendimiento del modelo. Se debe tener cuidado en seleccionar las características que sean más relevantes para el modelo y descartar aquellas que no sean útiles.

4. Limpiar y preprocesar los datos: Es importante limpiar y preprocesar los datos para eliminar datos incompletos, inconsistentes y ruidosos. Además, el preprocesamiento de datos puede incluir normalización, escalamiento y codificación de variables categóricas.

5. Verificar la calidad de los datos: Es importante verificar la calidad de los datos antes de entrenar un modelo de machine learning. Esto incluye comprobar la consistencia de los datos, la distribución de las características y la presencia de valores atípicos o datos faltantes.

6. Considerar el desequilibrio de clases: Si el conjunto de datos está desequilibrado (es decir, hay muchas más instancias de una clase que de otra), es importante considerar técnicas para manejar el desequilibrio. Esto puede incluir técnicas de submuestreo, sobremuestreo o ajuste de pesos.

7. Realizar una validación cruzada: Es importante realizar una validación cruzada para evaluar el rendimiento del modelo en un conjunto de datos que no se utilizó en el entrenamiento. Esto ayuda a evitar el sobreajuste del modelo.

Si tienes algunas dudas con los puntos anteriormente mencionados, también te invito a que le eches un vistazo a mi repositorio
de [introducción a machine learning con scikit-learn](https://github.com/ichcanziho/cursos_platzi/tree/master/machine_learning_scikit_learn)

## 2.2 Cómo cargar bases de datos JSON

En este escenario vamos a utilizar el formato `json` que contiene `urls` almacenadas en la nube de `gcp`. Es importante conocer
la mayor cantidad de herramientas que nos permitan cargar y manipular datos de todas las fuentes posibles. Muchas empresas
cargan sus imágenes en nubes como AMAZON, GCP, AZURE, entre otras y luego estas son consumidas.

> ## Nota:
> El código de esta sección lo puedes encontrar [aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos/2%20Cómo%20cargar%20datasets%20json/read_json.py)

Vamos a descargar las bases de datos que ocuparemos para esta clase y la siguiente:

```bash
cd 2\ Manejo\ y\ preprocesamiento\ de\ datos/
mkdir datasets
cd datasets
wget --no-check-certificate https://storage.googleapis.com/platzi-tf2/sign-language-img.zip -O sign-language-img.zip --no-check-certificate https://storage.googleapis.com/platzi-tf2/databasesLoadData.zip     -O databasesLoadData.zip
unzip databasesLoadData.zip
rm databasesLoadData.zip
```
La respuesta esperada es la siguiente estructura de carpetas:
```commandline
datasets
--------|sign_mnist_base64
--------|----------------|data.json
--------|sign_mnist_json
--------|----------------|data.json
--------|sign_mnist_test
--------|----------------|sign_mnist_test.csv
--------|sign_mnist_train
--------|----------------|sing_mnist_train.csv
--------|----------------|sing_mnist_train_clean.csv
--------|----------------|sing_mnist_train_no_clean.csv
pixeles.png
```

Analicemos brevemente el archivo [data.json](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_json%2Fdata.json)

```commandline
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/29_B.jpg","label":"b"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/30_B.jpg","label":"b"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/95_B.jpg","label":"b"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/58_A.jpg","label":"a"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/50_A.jpg","label":"a"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/46_A.jpg","label":"a"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/3_C.jpg","label":"c"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/32_C.jpg","label":"c"}
{"content": "https://storage.googleapis.com/platzi-tf2/img_mnist/2_C.jpg","label":"c"}
```
Básicamente, es una secuencia de diccionarios, que tiene como llaves: `content` para mostrar el URL de la imagen y `label` el
cual contiene la clasificación de dicha imagen. Observamos que como tal NO es el formato más `correcto` de json, puesto que
para que puediera ser cargado por `json.load` el archivo debería indicar que es una lista con `[]` y cada elemento debería
ir separado por `,`, entonces debemos leer este archivo de una forma ligeramente diferente.

### Ejemplo en código:

El flujo más simple de trabajo para este ejemplo es:
1. Leer el archivo [data.json](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_json%2Fdata.json)
2. Para cada url de content:
   1. hacer un request para recibir la imagen
   2. transformar el response en un numpy array
3. Mostrar un ejemplo de la imagen y el label recibido

**1: Importamos bibliotecas necesarias**
```python
import requests
from json import loads
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
```

**2: Leemos el archivo data.json**

Lo podemos leer como un texto plano, y por cada línea dentro del archivo decodificar dicha linea como json ocupando su método `loads`
```python
with open("../datasets/sign_mnist_json/data.json", "r", encoding='utf-8') as d:
    data = [loads(line) for line in d.readlines()]
```

**3: Acceder a cada elemento dentro de `data`**

Como ahora cada línea es un archivo json, puedo acceder al valor de sus llaves utilizando `get`
```python
X, y = [], []
for example in data:
    print(example)
    url_image = example.get("content", 0)
    label = example.get("label", 0)
```
Respuesta esperada:
```commandline
{'content': 'https://storage.googleapis.com/platzi-tf2/img_mnist/29_B.jpg', 'label': 'b'}
```

**4: Descargando cada imagen y convirtiéndola a un numpy array**

```python
# Petición al servidor
    response = requests.get(url_image).content
    print(type(response), response)
    # transformado `bytes` en PIL image
    pil_image = Image.open(BytesIO(response))
    print(pil_image)
    # transformando pil_image en un numpy array
    img = np.asarray(pil_image).reshape(28, 28)
    X.append(img)
    y.append(label)
```
Aquí adicionalmente, añadimos a las listas `X`, `y` los valores decodificados de las imágenes y labels pertinentes.

**5: Mostrando un ejemplo de imagen clasificada**

```python
    plt.imshow(img, cmap="gray")
    plt.title(f"label = {label}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("test.png")
```
Respuesta esperada:

![test.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F2%20C%C3%B3mo%20cargar%20datasets%20json%2Ftest.png)

> ## Nota:
> El código de esta sección lo puedes encontrar [aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos/2%20Cómo%20cargar%20datasets%20json/read_json.py)



## 2.3 Cargar bases de datos CSV y BASE 64

### Ejemplo de Base 64

Primero entendamos qué es Base 64:

Base64 es una forma de codificar datos binarios en caracteres ASCII para que puedan ser transmitidos a través de canales que no admiten datos binarios, como el correo electrónico o la web. En Base64, cada conjunto de tres bytes (24 bits) se convierte en una cadena de cuatro caracteres ASCII.

Base64 se utiliza para enviar archivos adjuntos de correo electrónico, imágenes y otros tipos de datos a través de Internet. Por ejemplo, si tienes una imagen en formato binario, puedes convertirla en Base64 y luego enviarla en un correo electrónico como una cadena de texto. Cuando el destinatario recibe el correo electrónico, puede decodificar la cadena Base64 y obtener la imagen original.

Base64 es útil en situaciones donde los datos binarios no pueden ser transmitidos directamente. Sin embargo, es importante tener en cuenta que la codificación Base64 aumenta el tamaño de los datos en aproximadamente un tercio. Además, Base64 no proporciona ningún tipo de encriptación o seguridad, por lo que no se debe utilizar como una forma de proteger datos sensibles.

> ## Nota:
> Por experiencia laboral Base64 también es un formato muy útil para enviar imágenes a ser procesadas por un API, tiene sentido
> que esta codificación b64 sea utilizada en un endpoint para analizar dicha imagen.

Ahora que ya entendemos un poco más acerca de Base 64, veamos un ejemplo de nuestra base de datos [data.json](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_base64%2Fdata.json):

```commandline
{
  "b": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AOhS246VBdyJbqyDmbaGVSOuWxTUk3XckDKcGVkjI/2VBOf1qcwc9K00i4rnNbCLrcSPMkQKJ97PUNnHHtmrllEf7US3dSJIEklcEdd7DBHtjP5VrGLmrSpxWHewQnxNHNOBsAJOfYZpvh8zXup394xPkKBAg9SDk/lx+db5j5pCcICO+K47X7+cL5uRv2kZxXW6TaRWek20UIIBQOSepZhkk/iasMxDV//Z"
}
```

Nuestro dataset, vuelve a ser un archivo en formato `json`, sin embargo, este es diferente al anterior, ahora en lugar de
tener como valores de las llaves la dirección de la imagen y su label, ahora tenemos algo más `simplificado`, tenemos
únicamente como llave la `label` de la imagen y como valor la propia imagen pero en formato `b64`.

#### Ejemplo en código

>## Nota:
> El código de esta sección lo puedes encontrar [Aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F2%20C%C3%B3mo%20cargar%20datasets%20json%2Fread_b64.py)

Cómo ejemplo escalable ocuparemos el siguiente dataset [data2.json](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_base64%2Fdata2.json)

Es básicamente el mismo que el anterior, pero tiene varios datos en lugar de uno solo:

```commandline
[
{"b": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AOhS246VBdyJbqyDmbaGVSOuWxTUk3XckDKcGVkjI/2VBOf1qcwc9K00i4rnNbCLrcSPMkQKJ97PUNnHHtmrllEf7US3dSJIEklcEdd7DBHtjP5VrGLmrSpxWHewQnxNHNOBsAJOfYZpvh8zXup394xPkKBAg9SDk/lx+db5j5pCcICO+K47X7+cL5uRv2kZxXW6TaRWek20UIIBQOSepZhkk/iasMxDV//Z"},
{"b": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AOhS246VBdyJbqyDmbaGVSOuWxTUk3XckDKcGVkjI/2VBOf1qcwc9K00i4rnNbCLrcSPMkQKJ97PUNnHHtmrllEf7US3dSJIEklcEdd7DBHtjP5VrGLmrSpxWHewQnxNHNOBsAJOfYZpvh8zXup394xPkKBAg9SDk/lx+db5j5pCcICO+K47X7+cL5uRv2kZxXW6TaRWek20UIIBQOSepZhkk/iasMxDV//Z"}
]
```

**1: Importando bibliotecas necesarias**

Vamos a utilizar OpenCv para acceder a un método muy útil `imdecode` que me permitirá junto con numpy convertir un texto
en formato b64 a un numpy array
```python
from json import load
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
```

**2: Creamos función auxiliar de conversión b64 a numpy.array**

```python
def b64_to_np(b_string: str):
    jpg_original = base64.b64decode(b_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv.imdecode(jpg_as_np, flags=1)
    return image_buffer
```

**3: Leemos el archivo** [data2.json](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_base64%2Fdata2.json)

```python
    with open("../datasets/sign_mnist_base64/data2.json", "r", encoding="utf-8") as d:
        data = load(d)
```

**4: Convertimos cada b64 en un numpy array**

```python
	X, y = [], []
    for example in data:
        for label, b_image in example.items():
            print(label, "-", b_image)
            img = b64_to_np(b_image)
			X.append(img)
            y.append(label)
```
Respuesta esperada
```commandline
b - /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AOhS246VBdyJbqyDmbaGVSOuWxTUk3XckDKcGVkjI/2VBOf1qcwc9K00i4rnNbCLrcSPMkQKJ97PUNnHHtmrllEf7US3dSJIEklcEdd7DBHtjP5VrGLmrSpxWHewQnxNHNOBsAJOfYZpvh8zXup394xPkKBAg9SDk/lx+db5j5pCcICO+K47X7+cL5uRv2kZxXW6TaRWek20UIIBQOSepZhkk/iasMxDV//Z
```

**5: graficamos para observar el resultado**

```python
 	    plt.imshow(img, cmap="gray")
            plt.title(f"label = {label}")
            plt.xticks([])
            plt.yticks([])
            plt.savefig("test_b64.png")
```
Respuesta esperada:

![test_b64.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F2%20C%C3%B3mo%20cargar%20datasets%20json%2Ftest_b64.png)

### Ejemplo de CSV

> ## Nota 
> El código de esta sección lo puedes encontrar [aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F2%20C%C3%B3mo%20cargar%20datasets%20json%2Fread_csv.py)

Antes de continuar con este tema te recomiendo repasar la clase de: [Consejos para el manejo de imágenes](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales#31-consejos-para-el-manejo-de-im%C3%A1genes)
Del curso anterior. Por si note queda muy claro el código presente en este ejemplo.

Primero familiaricémonos con nuestros datos [sign_mnist_test.csv](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_test%2Fsign_mnist_test.csv)

**1: Conozcamos los datos**

```python
data = pd.read_csv("../datasets/sign_mnist_train/sign_mnist_train.csv")
samples = len(data)
print("samples:", samples)
print(data)
```
Respuesta esperada:
```commandline
samples: 27455
       label  pixel1  pixel2  pixel3  ...  pixel781  pixel782  pixel783  pixel784
0          3     107     118     127  ...       206       204       203       202
1          6     155     157     156  ...       175       103       135       149
2          2     187     188     188  ...       198       195       194       195
3          2     211     211     212  ...       225       222       229       163
4         13     164     167     170  ...       157       163       164       179
...      ...     ...     ...     ...  ...       ...       ...       ...       ...
27450     13     189     189     190  ...       234       200       222       225
27451     23     151     154     157  ...       195       195       195       194
27452     18     174     174     174  ...       203       202       200       200
27453     17     177     181     184  ...        47        64        87        93
27454     23     179     180     180  ...       197       205       209       215

[27455 rows x 785 columns]
```
Podemos observar que la primera columna contiene un `label encoding` de las clases. Mientras que las siguientes columnas
son el `flatten` de la imagen (28x28) píxeles.

**2. Separamos el dataset en `X` & `y`**

```python
	y = data["label"].values
    X = data.drop('label', axis=1).values.reshape((samples, 28, 28))
```
En realidad es todo, pandas almacena los datos como numpy arrays, solo necesitamos acceder a ellos y hacerles un
`reshape` indicándole que tenemos `n` imágenes correspondientes al valor de `samples` y que cada imagen es de (28x28)

**3. Graficamos**
```python
 	    plt.imshow(img, cmap="gray")
            plt.title(f"label = {label}")
            plt.xticks([])
            plt.yticks([])
            plt.savefig("test_csv.png")
```
Respuesta esperada:

![test_csv.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F2%20C%C3%B3mo%20cargar%20datasets%20json%2Ftest_csv.png)


## 2.4 Preprocesamiento y limpieza de datos

El preprocesamiento y la limpieza de datos son procesos fundamentales en el análisis de datos, en particular, cuando se trabaja con conjuntos de datos para machine learning. El preprocesamiento se refiere a la transformación de datos brutos en una forma que sea adecuada para el análisis y la modelización. La limpieza de datos es un paso específico del preprocesamiento que se enfoca en detectar y corregir errores en los datos.

A continuación, se describen algunos de los procesos comunes de preprocesamiento y limpieza de datos:

- `Eliminación de valores atípicos:` Los valores atípicos `outliers` son valores que se encuentran fuera del rango esperado de una distribución. Pueden ser causados por errores de medición o entrada de datos incorrecta. Los valores atípicos pueden distorsionar los resultados del análisis y por lo tanto, deben ser identificados y eliminados o corregidos si es posible.

![2.png](imgs%2F2%2F2.png)

- `Imputación de valores faltantes:` Los valores faltantes pueden ser causados por diversas razones, como errores de medición o problemas de entrada de datos. La imputación se refiere a la estimación de valores faltantes en el conjunto de datos utilizando técnicas como la media, la mediana, la moda, la regresión u otros modelos estadísticos.

![3.png](imgs%2F2%2F3.png)

- `Normalización y estandarización:` Normalización se refiere al proceso de escalar los valores de una variable para que tengan una escala común. La estandarización, por otro lado, es el proceso de transformar los datos para que tengan una media cero y una desviación estándar de uno. La normalización y la estandarización se utilizan para que las variables tengan escalas similares y para mejorar el rendimiento de los modelos de machine learning.

![4.png](imgs%2F2%2F4.png)

- `Eliminación de variables irrelevantes:` Las variables irrelevantes son aquellas que no contribuyen significativamente al análisis y por lo tanto, se pueden eliminar. Eliminar variables irrelevantes puede mejorar la precisión del análisis y reducir el tiempo de procesamiento.

![5.png](imgs%2F2%2F5.png)

- `Detección y eliminación de duplicados:` Los datos duplicados pueden ser causados por errores de entrada de datos o por la duplicación de registros. La detección y eliminación de duplicados se realiza para asegurar que los datos sean precisos y no estén sesgados.

![6.png](imgs%2F2%2F6.png)

## Otro problema común: El desequilibrio de clases.

El desequilibrio de clases es un problema común en machine learning, donde una clase minoritaria tiene muy pocos ejemplos en comparación con una clase mayoritaria. Si no se aborda este problema adecuadamente, los modelos pueden tener dificultades para aprender patrones en la clase minoritaria y pueden estar sesgados hacia la clase mayoritaria. A continuación, se describen algunas formas comunes de abordar el problema de desequilibrio de clases en machine learning:

- `Oversampling:` El oversampling implica aumentar el número de ejemplos de la clase minoritaria mediante la generación de nuevos ejemplos sintéticos. Las técnicas comunes de oversampling incluyen SMOTE (Synthetic Minority Over-sampling Technique) y ADASYN (Adaptive Synthetic Sampling).
	![7.png](imgs%2F2%2F7.png)

- `Undersampling:` El undersampling implica reducir el número de ejemplos de la clase mayoritaria. Las técnicas comunes de undersampling incluyen la eliminación aleatoria de ejemplos de la clase mayoritaria y la selección de un subconjunto de ejemplos de la clase mayoritaria basado en algún criterio específico.
	![8.png](imgs%2F2%2F8.png)

- `Métodos híbridos:` Los métodos híbridos combinan técnicas de oversampling y undersampling para lograr un equilibrio en el conjunto de datos. Un ejemplo común de un método híbrido es la combinación de oversampling de la clase minoritaria y undersampling de la clase mayoritaria.
	![9.png](imgs%2F2%2F9.png)
	
- `Cambio de umbral:` El cambio de umbral implica ajustar el umbral de decisión de un clasificador para que favorezca la clasificación de la clase minoritaria. Por ejemplo, en lugar de clasificar una instancia como positiva solo si la probabilidad es mayor que 0,5, se puede cambiar el umbral para clasificar una instancia como positiva si la probabilidad es mayor que 0,3.

- `Cost-sensitive learning:` El cost-sensitive learning implica asignar un costo diferente a los errores de clasificación para cada clase. Por ejemplo, se puede asignar un mayor costo a los errores de clasificación de la clase minoritaria para que el modelo se enfoque en la clasificación correcta de la clase minoritaria.



### Ejemplo en código 

> ## Nota
> El código de este ejemplo lo puedes encontrar [Aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F4%20preprocesamiento%20y%20limpieza%2Fmain.py)

Veamos de nuevo nuestro dataset csv de imágenes de lenguaje de señas: [sign_mnist_train_clean.csv](2%20Manejo%20y%20preprocesamiento%20de%20datos%2Fdatasets%2Fsign_mnist_train%2Fsign_mnist_train_clean.csv)

Vamos a hacer un análisis de datos y limpieza del mismo:

**1: Importando bibliotecas**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```

**2: Leyendo nuestro dataset**

```python
train = pd.read_csv("../datasets/sign_mnist_train/sign_mnist_train_clean.csv")
print(train)
```
Respuesta esperada:
```commandline
       label pixel1 pixel2 pixel3  ... pixel781 pixel782 pixel783 pixel784
0          3    107    118    127  ...      206      204      203      202
1          6    155    157    156  ...      175      103      135      149
2          2    187    188    188  ...      198      195      194      195
3          2    211    211    212  ...      225      222      229      163
4         13    164    167    170  ...      157      163      164      179
...      ...    ...    ...    ...  ...      ...      ...      ...      ...
27450     13    189    189    190  ...      234      200      222      225
27451     23    151    154    157  ...      195      195      195      194
27452     18    174    174    174  ...      203      202      200      200
27453     17    177    181    184  ...       47       64       87       93
27454     23    179    180    180  ...      197      205      209      215

[27455 rows x 785 columns]
```
¿Cuántas clases tiene nuestro dataset?
```python
y = train[["label"]]
n_clases = sorted(y["label"].unique())
print(len(n_clases), n_clases)
```
Respuesta esperada:
```commandline
24 [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
```
¿Cuál es la distribución de datos de nuestro dataset por clase?
```python
fig, ax = plt.subplots(figsize=(10, 10))
sns.countplot(y, x="label")
ax.bar_label(ax.containers[0], rotation=45, label_type="edge", fmt=lambda x: '{:.1f}%'.format(x/len(train) * 100))
plt.xlabel("Classes", size=15)
plt.ylabel("Frequency", size=15)
plt.title("Data distribution")
plt.savefig("freq.png")
```
Respuesta esperada:

![freq.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F4%20preprocesamiento%20y%20limpieza%2Ffreq.png)

En general podemos observar que nuestro dataset NO cuenta con un desequilibrio de claes notorio.

¿Qué tipo de dato tienen nuestras features?

```python
print(X.dtypes)
```
Respuesta esperada:
```commandline
pixel1      object
pixel2      object
pixel3      object
pixel4      object
pixel5      object
             ...  
pixel780    object
pixel781    object
pixel782    object
pixel783    object
pixel784    object
Length: 784, dtype: object
```
Esto indica la presencia de datos de texto en nuestras columnas, lo cuál NO es bueno porque se supone que solamente
estamos guardando variables numéricas.

¿Tenemos datos faltantes?

```python
print(X.isnull().values.any())
```
Respuesta esperada
```commandline
False
```
No, no tenemos missing values.

¿Existen datos duplicados?
```python
print(X[X.duplicated()])
```
Respuesta esperada:
```commandline
     pixel1  pixel2  pixel3  pixel4  ... pixel781 pixel782 pixel783 pixel784
317       0       0       0       0  ...        0        0        0        0
487       0       0       0       0  ...        0        0        0        0
595       0       0       0       0  ...        0        0        0        0
689       0       0       0       0  ...        0        0        0        0
802  fwefew  fwefew  fwefew  fwefew  ...   fwefew   fwefew   fwefew   fwefew
861  fwefew  fwefew  fwefew  fwefew  ...   fwefew   fwefew   fwefew   fwefew
```
Sí, sí tenemos.

**3: Limpiando el dataset**

```python
X = X.drop([595, 689, 727, 802, 861], axis=0)
X = X.astype(str).astype(int)
X /= 255
print(X.head())
print(X.dtypes)
```
Respuesta esperada:
```commandline
     pixel1    pixel2    pixel3  ...  pixel782  pixel783  pixel784
0  0.419608  0.462745  0.498039  ...  0.800000  0.796078  0.792157
1  0.607843  0.615686  0.611765  ...  0.403922  0.529412  0.584314
2  0.733333  0.737255  0.737255  ...  0.764706  0.760784  0.764706
3  0.827451  0.827451  0.831373  ...  0.870588  0.898039  0.639216
4  0.643137  0.654902  0.666667  ...  0.639216  0.643137  0.701961

[5 rows x 784 columns]
pixel1      float64
pixel2      float64
pixel3      float64
pixel4      float64
pixel5      float64
             ...   
pixel780    float64
pixel781    float64
pixel782    float64
pixel783    float64
pixel784    float64
Length: 784, dtype: object

Process finished with exit code 0
```
Perfecto, nuestros datos ya están normalizados y ahora todos son datos numéricos.

## 2.5 Keras datasets

## 2.6 Datasets generators

## 2.7 Aprende a buscar bases de datos para deep learning

## 2.8 Cómo distribuir los datos

## 2.9 Crear la red neuronal, definir capas, compilar, entrenar, evaluar y predicciones

# 3 Optimización de precisión de modelos

## 3.1 Métodos de regularización: overfitting y underfitting

## 3.2 Recomendaciones prácticas para ajustar un modelo

## 3.3 Métricas para medir la eficiencia de un modelo: Callback

## 3.4 Monitoreo del entrenamiento en tiempo real: early stopping y patience

## 3.5 kerasTuner: Construyendo el modelo

## 3.6 KerasTuner: Buscando la mejor configuración para tu modelo

# 4 Almacenamiento y carga de modelos

## 4.1 Almacenamiento y carga de modelos: pesos y arquitectura

## 4.2 Criterios para almacenar los modelos

# 5 Fundamentos de aprendizaje por transferencia

## 5.1 Introducción al aprendizaje por transferencia

## 5.2 Cuándo utilizar aprendizaje por transferencia

## 5.3 Carga de sistemas pre-entrenados en Keras

## 5.4 API funcional de Keras

## 5.5 Uso de sistemas pre-entrenados de TensorFlow Hub

# 6 Resultados de entrenamiento

## 6.1 Introducción a variables relevantes del TensorBoard

## 6.2 Análisis y publicación de resultados del entrenamiento

## 6.3 Introducción al despliegue de modelos en producción

## 6.4 Siguientes pasos con deep learning

![test.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F2%20C%C3%B3mo%20cargar%20datasets%20json%2Ftest.png)