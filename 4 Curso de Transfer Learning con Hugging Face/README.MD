# Curso de Transfer Learning con Hugging Face

Eleva tus habilidades en deep learning aplicando transfer learning a modelos pre-entrenados. Aprende a utilizar todo el poder de la biblioteca de modelos de machine learning y datasets open-source de Hugging Face de la mano de sus expertos.

- Comparte tus modelos en el Hub de Hugging Face.
- Afina modelos de visión computarizada y procesamiento de lenguaje natural.
- Aplica transfer learning de forma sencilla y rápida.
- Utiliza el Hub de Hugging Face y su biblioteca de modelos.

> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
>
> Este Curso es el Número 4 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso


# Índice:
- [1 Introducción al Hub de Hugging Face](#1-introducción-al-hub-de-hugging-face)
- [1 Introducción al transfer learning](#11-introducción-al-transfer-learning)
- [1.1 Machine Learning abierto para todo el mundo](#12-machine-learning-abierto-para-todo-el-mundo)
- [1.2 Las tasks en machine learning](#13-las-tasks-en-machine-learning)
- [1.3 Quiz: introducción al hub de hugging face](#14-quiz--introducción-al-hub-de-hugging-face)
- [2 Primeros pasos con transfer learning y transformers](#2-primeros-pasos-con-transfer-learning-y-transformers)
- [2.1 Tus primeros modelos pre-entrenados usando pipeline](#21-tus-primeros-modelos-pre-entrenados-usando-pipeline)
- [2.2 Explorando los datasets](#22-explorando-los-datasets)
- [2.3 Introducción a los Transformers](#23-introducción-a-los-transformers)
- [2.4 Aplicando transfer learning](#24-aplicando-transfer-learning)
- [2.5 Quiz: Primeros pasos con transfer learning](#25-quiz--primeros-pasos-con-transfer-learning)
- [3 Computer Vision](#3-computer-vision)
- [3.1 Carga de dataset para computer vision](#31-carga-de-dataset-para-computer-vision)
- [3.2 Procesamiento de dataset para computer vision](#32-procesamiento-de-dataset-para-computer-vision)
- [3.3 Configurando un Trainer para computer vision](#33-configurando-un-trainer-para-computer-vision)
- [3.4 Entrenamiento y evaluación de modelo de computer vision](#34-entrenamiento-y-evaluación-de-modelo-de-computer-vision)
- [3.5 Quiz: Computer vision con Hugging Face](#35-quiz--computer-vision-con-hugging-face)
- [4 Natural Language Processing](#4-natural-language-processing)
- [4.1 Carga de datasets para NLP](#41-carga-de-datasets-para-nlp)
- [4.2 Procesamiento de dataset para NLP](#42-procesamiento-de-dataset-para-nlp)
- [4.3 Configurando un Trainer para NLP](#43-configurando-un-trainer-para-nlp)
- [4.4 Entrenamiento y evaluación de modelo de NLP](#44-entrenamiento-y-evaluación-de-modelo-de-nlp)
- [4.5 Quiz: NLP con hugging face](#45-quiz--nlp-con-hugging-face)
- [5 Comparte en el HUB](#5-comparte-en-el-hub)
- [5.1 El Hub como tu curriculum para machine learning](#51-el-hub-como-tu-curriculum-para-machine-learning)
- [5.2 Compartir tu modelo en el HUB de Hugging Face](#52-compartir-tu-modelo-en-el-hub-de-hugging-face)

# 1 Introducción al Hub de Hugging Face

## 1.1 Introducción al transfer learning

**Una breve introducción a Hugging Face**

El Hub de Hugging Face es una plataforma en línea que permite a los desarrolladores de inteligencia artificial compartir, descubrir y usar modelos pre-entrenados de aprendizaje profundo y otros recursos de procesamiento de lenguaje natural (NLP). Hugging Face es una empresa de tecnología de inteligencia artificial que se especializa en herramientas y modelos de NLP, y su Hub es una parte importante de su plataforma.

![1.png](imgs%2F1%20Introducci%C3%B3n%2F1.png)

En el Hub de Hugging Face, los usuarios pueden encontrar una amplia variedad de modelos de NLP pre-entrenados en varios idiomas, incluyendo modelos para tareas como la traducción automática, el reconocimiento de entidades nombradas, la generación de lenguaje natural y mucho más. Los modelos están disponibles en diferentes tamaños y configuraciones, lo que permite a los usuarios encontrar el modelo adecuado para su caso de uso específico.

Además de los modelos pre-entrenados, el Hub también contiene recursos de datos, códigos y herramientas para ayudar a los desarrolladores a construir y mejorar sus propios modelos de NLP. La plataforma es de código abierto, lo que significa que los usuarios pueden contribuir y mejorar los recursos existentes para el beneficio de la comunidad en general.

**¿Qué aprenderemos en este curso?**

- Qué es y cómo hacer transfer learning?
- Qué es el Hub de Hugging Face y cómo usarlo
- Utilizar modelos modernos para nuestras propias aplicaciones.
- Afinar modelos de lenguaje y visión
- Utilizar datasets open source compartidos por la comunidad en el Hub

**Bases de Transfer Learning**

Transfer learning (aprendizaje transferido, en español) es una técnica de aprendizaje automático en la que se aprovecha 
el conocimiento adquirido de un modelo entrenado en una tarea para mejorar el desempeño de otro modelo en una tarea relacionada.

![2.png](imgs%2F1%20Introducci%C3%B3n%2F2.png)

En lugar de entrenar un modelo desde cero para una tarea específica, el aprendizaje transferido utiliza un modelo previamente 
entrenado en una tarea similar como punto de partida y ajusta sus parámetros para la tarea en cuestión. Esto se logra mediante 
la reutilización de algunas o todas las capas de la red neuronal del modelo previo, que contienen conocimientos generales 
sobre el lenguaje o las imágenes.

**¿Por qué deberíamos utilizar Transfer Learning?**

El beneficio de esta técnica es que puede reducir significativamente la cantidad de datos y el tiempo necesarios para entrenar 
un modelo para una nueva tarea, especialmente si la tarea en cuestión tiene una cantidad limitada de datos de entrenamiento 
disponibles. Además, el modelo pre-entrenado puede contener características generales útiles para tareas relacionadas, lo 
que puede mejorar la calidad de la predicción.

El aprendizaje transferido se ha demostrado que es efectivo en una amplia variedad de tareas de aprendizaje automático, 
incluyendo la clasificación de imágenes, el procesamiento del lenguaje natural, el reconocimiento de voz, entre otros.

## 1.2 Machine Learning abierto para todo el mundo

En la actualidad es MUY complejo e incluso inocente pensar que alguna empresa será capaz de "resolver la IA" por sí misma.
Esto debido al alto nivel de complejidad de la tarea. Este tipo de problemas requieren el esfuerzo de una gran cantidad de
personas trabajando por un bien común. Se requiere de una `comunidad`. Tener a más personas contribuyendo en los proyectos de
IA es la forma más efectiva y eficiente de generar mejores y más poderosos modelos.

Actualmente, `Hugging Face` es una comunidad de IA en donde podremos encontrar:

- más 50 mil modelos
- más de 5 mil datasets
- más de 5 mil Spaces (demos)

Los modelos son libres, y podemos usarlos como base para crear nuestras nuevas propuestas de modelos y solucionar problemas
cada vez más específicos.

Empecemos por conocer la página de Hugging Face: https://huggingface.co/

![3.png](imgs%2F1%20Introducci%C3%B3n%2F3.png)


De forma muy sencilla, podemos decir que dentro de Hugging Face, cada modelo es en sí mismo un repositorio de Git. Hugging Face
nos permite filtrar por una amplia variedad de parámetros, desde el tipo de problema: Multimodal, NLP, Audio, Computer Vision.
Hasta por el tipo de biblioteca que manejamos: Pytorch, TensorFlow, Keras, Scikit-learn etc. Por idioma, por licencia entre otros.


![4.png](imgs%2F1%20Introducci%C3%B3n%2F4.png)

Nuestros filtros no arrojaran que modelos cumplen con nuestros requisitos:

![5.png](imgs%2F1%20Introducci%C3%B3n%2F5.png)

Por ejemplo conozcamos a: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

![6.png](imgs%2F1%20Introducci%C3%B3n%2F6.png)

Podemos observar como es una página bastante similar a GitHub. Incluso en su pestaña de `Files and versions` podemos encontrar
una estructura muy similar a Github.

![7.png](imgs%2F1%20Introducci%C3%B3n%2F7.png)

Varios modelos de Hugging Face me van a permitir usar su `Hosted inference API`
Una herramienta sumamente práctica que me va a dejar probar el funcionamiento del modelo de una forma muy simple.

Por ejemplo: busquemos [xlm-roberta-base](https://huggingface.co/xlm-roberta-base?text=Hola+soy+una+persona+muy+%3Cmask%3E+y+me+encanta%21)

![8.png](imgs%2F1%20Introducci%C3%B3n%2F8.png)

De forma sumamente rápida podemos ver el funcionamiento de este modelo. Inlcuso en la ventaja de `Use in Transformers`
nos dan un ejemplo de implementación simple en código:

![9.png](imgs%2F1%20Introducci%C3%B3n%2F9.png)


## 1.3 Las tasks en machine learning

En esta clase aprenderemos ¿Qué son los tasks? Y ¿Cómo elegir el mejor modelo para nuestra aplicación?

Una task es una aplicación específica para la que creas un modelo de Machine Learning. Por ejemplo, pensemos en las siguientes
áreas de aplicación de Deep Learning y mencionemos algunas de sus `tasks`:

- Computer Vision:
  - Image Classification
  - Image Segmentation
  - Image-to-image
  - Unconditional Image Generation
  - Object Detection
- Natural Language Processing
  - Translation
  - Token Classification
  - Sentence Similarity
  - Question Answering
  - Summarization
  - Zero-Shot Classification
  - Text Classification
  - Text2Text Generation
- Audio
  - Audio-Classification
  - Text-to-Speech
  - Audio-to-Audio
- Multimodal
  - Feature Extraction
  - Image-to-Text
  - Text-to-Image
- Tabular
  - Tabular Classification
  - Tabular Regression
- Reinforcement Learning
  - Reinforcement Learning

Todos estos y más `tasks` están disponibles en `Hugging Face`: https://huggingface.co/tasks

![10.png](imgs%2F1%20Introducci%C3%B3n%2F10.png)

Cada `task` nos brinda más información al respecto del mismo, modelos disponibles, bibliotecas compatibles y en algunos
casos incluso códigos de ejemplo en formato NoteBook e incluso Datasets. Por ejemplo veamos el task de `Text Classification`

![11.png](imgs%2F1%20Introducci%C3%B3n%2F11.png)



## 1.4 Quiz: introducción al hub de hugging face

![12.png](imgs%2F1%20Introducci%C3%B3n%2F12.png)

# 2 Primeros pasos con transfer learning y transformers

## 2.1 Tus primeros modelos pre-entrenados usando pipeline

## 2.2 Explorando los datasets

## 2.3 Introducción a los Transformers

## 2.4 Aplicando transfer learning

## 2.5 Quiz: Primeros pasos con transfer learning

# 3 Computer Vision

## 3.1 Carga de dataset para computer vision

## 3.2 Procesamiento de dataset para computer vision

## 3.3 Configurando un Trainer para computer vision

## 3.4 Entrenamiento y evaluación de modelo de computer vision

## 3.5 Quiz: Computer vision con Hugging Face

# 4 Natural Language Processing

## 4.1 Carga de datasets para NLP

## 4.2 Procesamiento de dataset para NLP

## 4.3 Configurando un Trainer para NLP

## 4.4 Entrenamiento y evaluación de modelo de NLP

## 4.5 Quiz: NLP con hugging face

# 5 Comparte en el HUB

## 5.1 El Hub como tu curriculum para machine learning

## 5.2 Compartir tu modelo en el HUB de Hugging Face