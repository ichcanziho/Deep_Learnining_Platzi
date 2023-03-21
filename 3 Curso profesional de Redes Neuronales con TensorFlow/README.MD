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

A lo largo de los cursos anteriores, hemos utilizado varios Datasets de Keras, entre los que pudimos utilizar se encuentra:

- [MNIST digits classification dataset](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales#14-tu-primer-red-neuronal-con-keras)
- [CIFAR10 small image classsification dataset](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales#5-resolviendo-un-problema-de-clasificaci%C3%B3n)
- [IMDB movie review sentiment classification dataset](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales#32-resolviendo-un-problema-de-clasificaci%C3%B3n-binaria)
- [Fashion MNIST dataset, an alternative to MNIST](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales#2-mi-primera-red-neuronal-convolucional)
- [Boston Housing price regression dataset](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales#39-resolviendo-un-problema-de-regresi%C3%B3n)

Para esta clase, veremos el último dataset disponible en Keras y repasaremos rápidamente como cargarlos y utilizarlos.

El dataset de repaso que veremos será: [CIFAR100 small images classification dataset](https://keras.io/api/datasets/cifar100/)
una curiosidad de este dataset es que cuenta con 2 clasificaciones `fine` y `coarse` indicando si quieres clasificar por super clase o por clase. A continuación las siguientes clases disponibles:

![10.png](imgs%2F2%2F10.png)

|         **Superclass**         |                      **Classes**                      |
|:------------------------------:|:-----------------------------------------------------:|
|         aquatic mammals        |          beaver, dolphin, otter, seal, whale          |
|              fish              |       aquarium fish, flatfish, ray, shark, trout      |
|             flowers            |      orchids, poppies, roses, sunflowers, tulips      |
|         food containers        |           bottles, bowls, cans, cups, plates          |
|      fruit and vegetables      |    apples, mushrooms, oranges, pears, sweet peppers   |
|  household electrical devices  | clock, computer keyboard, lamp, telephone, television |
|       household furniture      |           bed, chair, couch, table, wardrobe          |
|             insects            |     bee, beetle, butterfly, caterpillar, cockroach    |
|        large carnivores        |            bear, leopard, lion, tiger, wolf           |
|  large man-made outdoor things |        bridge, castle, house, road, skyscraper        |
|  large natural outdoor scenes  |          cloud, forest, mountain, plain, sea          |
| large omnivores and herbivores |     camel, cattle, chimpanzee, elephant, kangaroo     |
|      medium-sized mammals      |         fox, porcupine, possum, raccoon, skunk        |
|    non-insect invertebrates    |           crab, lobster, snail, spider, worm          |
|             people             |              baby, boy, girl, man, woman              |
|            reptiles            |       crocodile, dinosaur, lizard, snake, turtle      |
|          small mammals         |        hamster, mouse, rabbit, shrew, squirrel        |
|              trees             |             maple, oak, palm, pine, willow            |
|           vehicles 1           |     bicycle, bus, motorcycle, pickup truck, train     |
|           vehicles 2           |      lawn-mower, rocket, streetcar, tank, tractor     |


#### Ejemplo en código:

>## Nota:
> El código de esta sección lo puedes encontrar [Aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F5%20Keras%20datasets%2Fmain.py)

**1: Importando bibliotecas**

```python
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import json
```

**2: Descargando el dataset**

```python
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
print(x_train.shape)
print(y_train.shape)
```
Respuesta esperada:
```commandline
(50000, 32, 32, 3)
(50000, 1)
```

**3: Obteniendo el nombre de las clases**

Primero debemos descargar el nombre de las clases
```commandline
mkdir labels
cd labels
wget --no-check-certificate https://storage.googleapis.com/platzi-tf2/cifar100_labels.json \
    -O cifar100_labels.json
cd ..
```
Cargamos los nombres:
```python
with open("labels/cifar100_labels.json", "r") as fine_labels:
    cifar100_labels = json.load(fine_labels)
```

**4: Inspeccionando el dataset**

```python
num_image = 40
y = y_train[num_image][0]
x = x_train[num_image]
y_name = cifar100_labels[y]
plt.imshow(x)
title = f"Class: {y_name} - Id Class: {y}"
plt.title(title)
plt.savefig("fig.png")
```
Respuesta esperada:

![fig.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F5%20Keras%20datasets%2Ffig.png)

## 2.6 Datasets generators

Este tema será un repaso del tema visto en: [3: Creación de Image Data Generators](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales#71-clasificando-entre-perros-y-gatos)

Recordemos que el dataset de Lenguaje de señas MNIST ya lo habíamos descargado
[Consiguiendo datos del proyecto](#consiguiendo-datos-del-proyecto-de-lenguaje-de-señas)

Hay un total de 27,455 imágenes en escala de grises de tamaño 28 * 28 píxeles cuyo valor oscila entre 0-255. Cada caso representa una etiqueta (0-25) como un mapa uno a uno para cada letra alfabética A-Z (y ningún caso para 9 = J o 25 = Z debido a movimientos gestuales).

Los datos se almacenan de forma ordenada y son compatibles para su uso con generadores de flujo de datos en la API de TensorFlow. Cada carpeta recibe un nombre de acuerdo con la clase de imágenes almacenadas en su interior, lo que facilita su carga y visualización.

Las imágenes se almacenan en formato de archivo 'JPEG'.

Antes de continuar con el código de ejemplo, es necesario definir qué es un `generador` en python:

### Generadores

En Python, un generador es una función que produce una secuencia de valores, pero en lugar de crear y retornar una lista completa de valores de una sola vez, produce cada valor uno por uno, en respuesta a las solicitudes del código que lo llama.

Los generadores son muy útiles cuando se trabaja con grandes conjuntos de datos o cuando se desea generar valores de manera eficiente en función de alguna lógica o algoritmo. En lugar de calcular todos los valores de la secuencia al mismo tiempo y almacenarlos en memoria, un generador calcula cada valor a medida que se solicita, lo que puede ahorrar tiempo y recursos.

![11.png](imgs%2F2%2F11.png)

Para crear un generador en Python, se utiliza la sentencia "yield" en lugar de "return". Cuando el generador se llama, devuelve un objeto de generador, que se puede iterar para obtener cada valor de la secuencia. Cada vez que se solicita el siguiente valor, la función continúa desde donde se detuvo en la última llamada, en lugar de comenzar desde el principio.

Por ejemplo, el siguiente código define una función generadora que produce los números impares menores que un número dado:

```python
def impares(n):
    i = 1
    while i < n:
        yield i
        i += 2
```
Para usar esta función generadora y obtener los números impares menores que 10, se puede hacer lo siguiente:

```python
for num in impares(10):
    print(num)
```

#### Ejemplo en código

> ## Nota:
> El código de esta sección lo puedes encontrar [Aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F6%20Dataset%20generators%2Fmain.py)

**1: Importando bibliotecas**

```python
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import string
```

**2: Creando nuestros ImageDataGenerators**

Para este punto de la clase NO hemos hablado de `Data Augmentation` cosa que si has tomado este repositorio completo entonces 
ya debes conocer y entender. Ahora lo único que vamos a hacer es utilizar el `ImageDataGenerator` para normalizar las imágenes
y esta clase nos permita generar un objeto con el método `frow_from_directory`

> Nota: 
> Algo interesante es que del test_dataget también vamos a construir el validation test, por haber definido el parámetro `validation_split`
```python
    train_dir = "../../data/Train"
    test_dir = "../../data/Test"

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)
```

**3: Creamos nuestros generadores de imágenes, para los conjuntos de train, validation y test**

```python
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training"
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation"
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=128,
        class_mode="categorical",
        color_mode="grayscale"
    )
```
Respuesta esperada:
```commandline
Found 27455 images belonging to 24 classes.
Found 1425 images belonging to 24 classes.
Found 7172 images belonging to 24 classes.
```

**4: Generando el nombre de las clases**

Sabemos que las clases están en `label_encoding` entonces hace falta definir el nombre de las labels de la siguiente manera
```python
    classes = [char for char in string.ascii_uppercase if char not in ("J", "Z")]
    print("Classes:", classes)
```
Respuesta esperada:
```commandline
Classes: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
```

**4: Muestra de entrenamiento**

Definimos una función auxiliar para mostrar imágenes:
```python
def plot_images(images_arr, title):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img[:, :, 0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
```
Generamos una muestra de 5 imágenes de nuestro `train_generator`

```python
    sample_training_images, _ = next(train_generator)
    plot_images(sample_training_images[:5], "Train Example")
```
Respuesta esperada:

![Train Example.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F6%20Dataset%20generators%2FTrain%20Example.png)


## 2.7 Aprende a buscar bases de datos para deep learning

Deep Learning NO es solamente código. Como ya hemos mencionado anteriormente, es sumamente importante contar con una base de
datos de calidad para asegurar que un proyecto tenga éxito. Sin embargo, no existen el 100% de bases de datos para el 100% de
nuestros problemas. Si estamos lidiando con un problema muy específico o de un nicho muy concreto como puede ser clasificar
un tipo de planta que solamente crece en nuestro jardin es 100% probable que nadie haya compartido un dataset público de los tipos
de planta que crecen en nuestra casa.

A pesar de ello, existen diferentes plataformas donde podemos encontrar una gran variedad de datasets que podemos usar para
nuestros proyectos de machine learning y deep learning. Entre algunas páginas donde puedes encontrar bases de datos podemos nombrar:

- Páginas de datos públicos de tu pais. En México es: [datos.gob.mx](https://datos.gob.mx/)
- Kaggle 
- Dataset Search
- Data.world (tiene costo)
- GitHub [Awesome public datasets](https://github.com/awesomedata/awesome-public-datasets)

## 2.8 Cómo distribuir los datos

Antes de continuar, te recomiendo darte una vuelta por mi repositorio de [Machine Learning con Python y Scikit-learn](https://github.com/ichcanziho/cursos_platzi/tree/master/machine_learning_scikit_learn)

Especialmente al siguiente tema: [Optimización Paramétrica](https://github.com/ichcanziho/cursos_platzi/tree/master/machine_learning_scikit_learn#7-optimizaci%C3%B3n-param%C3%A9trica)

Sin embargo, a continuación te brindo un poco de información por si no tienes muy claro como dividir nuestros datasets.

La distribución de las particiones de los datasets en deep learning puede variar según el enfoque específico utilizado, pero aquí te presento algunas formas comunes en que se dividen los datos:

- `Validación cruzada:` este método implica dividir el conjunto de datos en k pliegues, y realizar k experimentos diferentes en los que se utiliza un pliegue diferente como conjunto de validación cada vez. Este enfoque puede ayudar a obtener una estimación más precisa del rendimiento del modelo.

- `División de entrenamiento y validación:` en este enfoque, el conjunto de datos se divide en dos conjuntos, uno para el entrenamiento y otro para la validación. El conjunto de entrenamiento se utiliza para ajustar los parámetros del modelo, mientras que el conjunto de validación se utiliza para evaluar el rendimiento del modelo y ajustar sus hiperparámetros.

- `División de entrenamiento, validación y prueba:` este enfoque es similar al anterior, pero se divide el conjunto de datos en tres conjuntos: entrenamiento, validación y prueba. El conjunto de prueba se utiliza para evaluar el rendimiento final del modelo después de haber ajustado los parámetros y los hiperparámetros utilizando el conjunto de entrenamiento y el conjunto de validación.

- `Bootstrap:` este método implica muestrear el conjunto de datos con reemplazo varias veces para generar múltiples conjuntos de entrenamiento y validación. Esto puede ayudar a reducir la varianza del modelo y mejorar su capacidad para generalizar a nuevos datos.

- `División basada en el tiempo:` en algunos casos, puede ser útil dividir el conjunto de datos en función de la fecha o el tiempo. Por ejemplo, se podría utilizar el conjunto de datos más antiguo para el entrenamiento, el conjunto de datos más reciente para la validación y un conjunto de datos futuro para la prueba.

> Nota:
> Entre más pequeña es la base de datos más se recomienda usar `validación cruzada`, sin embargo, si la base de datos es muy larga
> esto puede ser contraproducente y se recomienda mejor utilizar el enfoque: train, validation, test con una distribución 90, 5, 5.
>

![12.png](imgs%2F2%2F12.png)

### Errores comunes en datasets

- Agregar datos de entrenamiento a testeo
- Bases de datos no balanceadas en clases
- Muy pocos datos


## 2.9 Crear la red neuronal, definir capas, compilar, entrenar, evaluar y predicciones

> ## Nota:
> El código completo lo puedes encontrar [Aquí](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F7%20Crear%20la%20red%2Fmain.py)

Este será nuestra primera aproximación a resolver el problema de clasificación y de lenguaje de señas. Pero antes de continuar
vamos a utilizar un par de funciones que ya hemos manejado en cursos anteriores.

Primero: utilizaremos nuestra muy querida función `plot_results` que hemos usado en los cursos anteriores de `deep learning`

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

Para efectos de continuidad, vamos a definir una función `get_data()` con todo lo aprendido en la clase [Dataset generators](#26-datasets-generators)

```python
def get_data():
    train_dir = "../../data/Train"
    test_dir = "../../data/Test"

    _bs = 128

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

    _train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        batch_size=_bs,
        class_mode="categorical",
        color_mode="grayscale",
        subset="training"
    )

    _validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=_bs,
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation"
    )

    _test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        batch_size=_bs,
        class_mode="categorical",
        color_mode="grayscale"
    )

    _classes = [char for char in string.ascii_uppercase if char not in ("J", "Z")]

    return _classes, _bs, _train_generator, _validation_generator, _test_generator
```

Ahora continuemos con la clase actual.

Los pasos a seguir para esta clase serán los siguientes:

1. Cargar las particiones de datos
2. Crear la arquitectura base del modelo
3. Compilar el modelo
4. Entrenar el modelo
5. Mostrar resultados

**1: Cargar las particiones de datos**
```python
    classes, batch_size, train_generator, validation_generator, test_generator = get_data()
```
Respuesta esperada:
```commandline
Found 27455 images belonging to 24 classes.
Found 1425 images belonging to 24 classes.
Found 7172 images belonging to 24 classes.
```

**2: Crear la arquitectura base del modelo**

Creamos una función auxiliar:
```python
def base_architecture(input_shape, n_clases):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(n_clases, activation="softmax"))
    print(model.summary())
    return model
```

Llamamos a nuestra función auxiliar en nuestro ciclo principal.

```python
    base_model = base_architecture(input_shape=(28, 28, 1), n_clases=len(classes))
```
Respuesta esperada:
```commandline
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 256)               200960    
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 24)                3096      
                                                                 
=================================================================
Total params: 236,952
Trainable params: 236,952
Non-trainable params: 0
```
**3: Compilamos el modelo**

```python
    base_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
```
Recordemos que como es un problema de clasificación multiple nuestra perdida debe ser:
`categorical_crossentropy` y la última capa de clasificación del modelo tendrá 24 neuronas una por cada clase disponible 
y a su vez esto nos hace necesitar usar `softmax` como función de activación.


**4: Entrenamos el modelo**

```python
    history = base_model.fit(train_generator, epochs=20, validation_data=validation_generator, batch_size=128)
```
Respuesta esperada:
```commandline
Epoch 1/20
215/215 [==============================] - 3s 10ms/step - loss: 2.1329 - accuracy: 0.3710 - val_loss: 1.6110 - val_accuracy: 0.4821
Epoch 2/20
215/215 [==============================] - 2s 10ms/step - loss: 1.2105 - accuracy: 0.6255 - val_loss: 1.3437 - val_accuracy: 0.5923
Epoch 3/20
215/215 [==============================] - 2s 10ms/step - loss: 0.8712 - accuracy: 0.7350 - val_loss: 1.1443 - val_accuracy: 0.6421
Epoch 4/20
...
Epoch 18/20
215/215 [==============================] - 2s 10ms/step - loss: 0.0271 - accuracy: 0.9955 - val_loss: 1.2362 - val_accuracy: 0.7474
Epoch 19/20
215/215 [==============================] - 2s 10ms/step - loss: 0.1214 - accuracy: 0.9633 - val_loss: 1.3080 - val_accuracy: 0.7011
Epoch 20/20
215/215 [==============================] - 2s 10ms/step - loss: 0.0210 - accuracy: 0.9970 - val_loss: 1.2360 - val_accuracy: 0.7607
```

**5: Análisis de resultados**

```python
  plot_results(history, "accuracy", "base_results.png")

  results = base_model.evaluate(test_generator)
```
Respuesta esperada:
```commandline
57/57 [==============================] - 1s 9ms/step - loss: 1.1455 - accuracy: 0.7655
```
![base_results.png](2%20Manejo%20y%20preprocesamiento%20de%20datos%2F7%20Crear%20la%20red%2Fbase_results.png)

Para este momento hemos conseguido un `accuracy` del `76.5%` del modelo, sin embargo, nuestras gráficas de perdida y accuracy
muestran como el desempeño en el conjunto de validación es bastante inferior al obtenido en entrenamiento, esto refleja un 
claro caso de overfitting. Nuestro modelo debe ser optimizado para mejorar los resultados de validación y con ello mejorar
los resultados de testing. En próximas clases iremos mejorando estos resultados. Las propuestas de mejora son:

1. Usar regularizadores
2. Cambiar arquitectura de la red por una CNN
3. Usar DataAugmentation
4. Usar transfer learning

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

