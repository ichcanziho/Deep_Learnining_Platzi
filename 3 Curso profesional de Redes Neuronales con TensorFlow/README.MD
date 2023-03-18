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

## 2.3 Cargar bases de datos CSV y BASE 64

## 2.4 Preprocesamiento y limpieza de datos

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

