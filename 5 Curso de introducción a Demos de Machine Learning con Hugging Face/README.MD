# Curso de Experimentación en Machine Learning con Hugging Face

Descubre cómo crear aplicaciones donde tus modelos de machine learning demuestran su funcionamiento. Programa estos demos de manera sencilla con Python y compártelos con todo el mundo con el Hub de Hugging Face.

- Configura tus demos con Gradio y Streamlit.
- Configura demos de otras personas de manera libre para tus necesidades.
- Crea demos de NLP y computer vision.
- Comparte tus demos en los Spaces de Hugging Face.
- Explorando los Spaces


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> 
> Este Curso es el Número 5 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice

- [1 Introducción al Hub de Hugging Face](#1-introducción-al-hub-de-hugging-face)
  - [1.1 Explorando los Spaces](#11-explorando-los-spaces)
- [2 Gradio](#2-gradio)
  - [2.1 Introducción a Gradio](#21-introducción-a-gradio)
  - [2.2 Tus primeros demos con Gradio](#22-tus-primeros-demos-con-gradio)
  - [2.3 Demos para clasificación de imágenes con Gradio](#23-demos-para-clasificación-de-imágenes-con-gradio)
  - [2.4 Compartir demos en Spaces de Hugging Face](#24-compartir-demos-en-spaces-de-hugging-face)
  - [2.5 Demo de transcripción de audio a texto con Gradio](#25-demo-de-transcripción-de-audio-a-texto-con-gradio)
  - [2.6 El futuro de los demos: Blocks](#26-el-futuro-de-los-demos-blocks)
  - [2.7 Blocks con Tabs y TabItem](#27-blocks-con-tabs-y-tabitem)
  - [2.8 Quizz Gradio](#28-quizz-gradio)
- [3 Streamlit](#3-streamlit)
  - [3.1 Introducción a Streamlit](#31-introducción-a-streamlit)
  - [3.2 Primeros pasos con Streamlit](#32-primeros-pasos-con-streamlit)
  - [3.3 Demo de generación de imágenes con Streamlit](#33-demo-de-generación-de-imágenes-con-streamlit)
  - [3.4 Creando interfaz de demo de GAN](#34-creando-interfaz-de-demo-de-gan)
  - [3.5 Probando demo de generación de imágenes](#35-probando-demo-de-generación-de-imágenes)
  - [3.6 Configurar un demo clonado desde Hugging Face](#36-configurar-un-demo-clonado-desde-hugging-face)
  - [3.7 Quizz Streamlit](#37-quizz-streamlit)
- [4 Tu historia en machine learning](#4-tu-historia-en-machine-learning)
  - [4.1 Sube tus propios demos](#41-sube-tus-propios-demos)

# 1 Introducción al Hub de Hugging Face
## 1.1 Explorando los Spaces

Los Spaces de Hugging Face son una plataforma en línea que permite a los usuarios crear y compartir experiencias interactivas basadas en modelos de lenguaje de Hugging Face. Los Spaces ofrecen una interfaz visual y amigable que permite a los usuarios crear fácilmente experiencias interactivas basadas en texto, como chatbots, juegos de texto, sistemas de recomendación de productos y más.

Los usuarios pueden comenzar a crear un Space de Hugging Face simplemente proporcionando un nombre y una descripción, y luego seleccionando un modelo de lenguaje de Hugging Face para alimentar la experiencia interactiva. Luego, pueden diseñar la experiencia interactiva utilizando una variedad de herramientas de construcción de flujo de trabajo basadas en texto, como la creación de preguntas y respuestas, la definición de reglas de conversación, la selección de respuestas de una lista predefinida, la creación de ramas de conversación y más.

Una vez que se ha creado un Space de Hugging Face, los usuarios pueden compartirlo fácilmente con otros para que lo experimenten y proporcionar retroalimentación. Además, los Spaces de Hugging Face se integran con la plataforma Hugging Face Hub, lo que significa que los modelos de lenguaje entrenados por la comunidad pueden ser utilizados en la creación de Spaces.


![1.png](imgs%2F1%20Intro%2F1.png)


Un demo de Hugging Face se refiere a una herramienta que permite interactuar con los modelos de lenguaje de Hugging Face de manera interactiva y visual. Hugging Face es una biblioteca de código abierto para el aprendizaje automático basado en PyTorch y TensorFlow que se centra en el procesamiento del lenguaje natural (NLP, por sus siglas en inglés).

Los demos de Hugging Face son interfaces web donde los usuarios pueden probar modelos de lenguaje de Hugging Face, como modelos de generación de lenguaje natural o modelos de clasificación de texto, sin necesidad de escribir código o entrenar los modelos ellos mismos. En lugar de eso, los usuarios pueden proporcionar una entrada en forma de texto y el modelo generará una respuesta.

Los demos de Hugging Face son una excelente herramienta para los desarrolladores y los no expertos en programación que desean probar y entender cómo funcionan los modelos de lenguaje de Hugging Face antes de integrarlos en sus propias aplicaciones o proyectos.



# 2 Gradio
## 2.1 Introducción a Gradio
## 2.2 Tus primeros demos con Gradio
## 2.3 Demos para clasificación de imágenes con Gradio
## 2.4 Compartir demos en Spaces de Hugging Face
## 2.5 Demo de transcripción de audio a texto con Gradio
## 2.6 El futuro de los demos: Blocks
## 2.7 Blocks con Tabs y TabItem
## 2.8 Quizz Gradio

# 3 Streamlit
## 3.1 Introducción a Streamlit
## 3.2 Primeros pasos con Streamlit
## 3.3 Demo de generación de imágenes con Streamlit
## 3.4 Creando interfaz de demo de GAN
## 3.5 Probando demo de generación de imágenes
## 3.6 Configurar un demo clonado desde Hugging Face
## 3.7 Quizz Streamlit

# 4 Tu historia en machine learning
## 4.1 Sube tus propios demos
