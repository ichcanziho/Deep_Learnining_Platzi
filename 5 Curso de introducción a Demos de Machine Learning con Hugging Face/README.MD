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

![3.png](imgs%2F2%20Gradio%2F3.png)

> Gradio es una biblioteca de Python que permite crear interfaces de usuario personalizadas para modelos de aprendizaje automático de forma rápida y sencilla, sin necesidad de conocimientos avanzados en programación o diseño. Gradio facilita la creación de aplicaciones de aprendizaje automático interactivas, lo que permite a los usuarios interactuar con los modelos de aprendizaje automático de una manera más intuitiva. Gradio es compatible con una variedad de marcos de aprendizaje automático, como TensorFlow, PyTorch, Keras, Scikit-learn, entre otros. Con Gradio, se pueden crear aplicaciones web de aprendizaje automático en cuestión de minutos.

![4.png](imgs%2F2%20Gradio%2F4.png)

https://gradio.app/

**¿Por qué es importante crear demos?**

- Presentar fácilmente a una audiencia muy amplia.
- Aumentar la reproducibilidad.
- Diferentes usuarios pueden identificar puntos de falla.

**Antes se tenía que:**

![1.png](imgs%2F2%20Gradio%2F1.png)

- Entrenar el modelo con frameworks como TensorFlow o Pytorch usando Python.
- Crear contenedores que lanzaran el modelo en frameworks como flash o Docker con lenguaje Bash
- Almacenar las muestras entrantes con MySQL o PostgreSQL
- Crear una interfaz de usuario activa con lenguajes en CSS, HTLM, Java.

**Actualmente se enfoca en:**

![2.png](imgs%2F2%20Gradio%2F2.png)

- Entrenar el modelo con TensorFlow y Pytorch con lenguaje Python
- Crear contenedores, almacenar las muestras entrantes y crea una interfaz con Gradio, Streamlit en Python 


> Gradio te permite crear demos y compartirlos. Todo en Python y con pocas líneas de código.
Se puede crear modelos multimodales. Simple de usar

## 2.2 Tus primeros demos con Gradio

Gradio ofrece dos API diferentes según el nivel de detalle que se busque:

- `gradio.Interface`: API de alto nivel que permite crear demos de ML simplemente proporcionando una lista de entradas y salidas.

- `gradio.Blocks`: API de bajo nivel que permite tener un control total sobre los flujos de datos y el diseño de la aplicación. Se pueden crear aplicaciones muy complejas de varios pasos utilizando Blocks (como si fueran bloques de construcción).

Comenzaremos utilizando `Interface` y al final mostraremos un ejemplo de `Blocks`.

```bash
pip install gradio
```
Respuesta esperada:
```commandline
Installing collected packages: rfc3986, pydub, ffmpy, websockets, uc-micro-py, toolz, sniffio, semantic-version, python-multipart, pyrsistent, pydantic, orjson, mdurl, h11, entrypoints, click, aiofiles, uvicorn, markdown-it-py, linkify-it-py, jsonschema, anyio, starlette, mdit-py-plugins, httpcore, gradio-client, altair, httpx, fastapi, gradio
Successfully installed aiofiles-23.1.0 altair-4.2.2 anyio-3.6.2 click-8.1.3 entrypoints-0.4 fastapi-0.95.0 ffmpy-0.3.0 gradio-3.24.1 gradio-client-0.0.5 h11-0.14.0 httpcore-0.16.3 httpx-0.23.3 jsonschema-4.17.3 linkify-it-py-2.0.0 markdown-it-py-2.2.0 mdit-py-plugins-0.3.3 mdurl-0.1.2 orjson-3.8.9 pydantic-1.10.7 pydub-0.25.1 pyrsistent-0.19.3 python-multipart-0.0.6 rfc3986-1.5.0 semantic-version-2.10.0 sniffio-1.3.0 starlette-0.26.1 toolz-0.12.0 uc-micro-py-1.0.1 uvicorn-0.21.1 websockets-11.0
```

Ejemplo usando una función para saludar que tiene `text` como input y `text` como output.

[1 introducción E1.py](2%20Gradio%2F1%20introducci%C3%B3n%20E1.py)
```python
import gradio as gr


def saluda(nombre):
    return "hola " + nombre


demo = gr.Interface(fn=saluda, inputs="text", outputs="text")

demo.launch()

```
Respuesta esperada:
```commandline

Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```
![5.png](imgs%2F2%20Gradio%2F5.png)

La clase de `gr.Interface` es una forma fácil de crear demos que pueden ser desde una calculadora hasta una aplicación para reconocimiento de voz. 

Se inicializa con tres parámetros necesarios:


*   `fn`: la función.

*   `inputs`: qué componente(s) usar para los inputs de la función, por ejemplo, "texto", "imagen" o "audio"
* `outputs`: qué componente(s) usar para los outputs de la función, por ejemplo, "texto", "imagen" o "etiqueta"


Gradio incluye más de 20 componentes diferentes, la mayoría de los cuales se pueden utilizar como inputs y outputs. En la documentación está la [lista completa](https://gradio.app/docs/#components).

![6.png](imgs%2F2%20Gradio%2F6.png)

[1 Introducción E2.py](2%20Gradio%2F1%20Introducci%C3%B3n%20E2.py)
```python
import gradio as gr


def saluda(nombre):
    return "hola " + nombre


demo = gr.Interface(fn=saluda, inputs=gr.Textbox(lines=10, placeholder="Dime tu nombre porfa"), outputs="text")

demo.launch()

```
Respuesta esperada:
```commandline
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.

```

![7.png](imgs%2F2%20Gradio%2F7.png)



## 2.3 Demos para clasificación de imágenes con Gradio

**Ejemplo 1:** demo con modelo cargado de las aplicaciones de TensorFlow.

[2 Clasificación Imágenes E1.py](2%20Gradio%2F2%20Clasificaci%C3%B3n%20Im%C3%A1genes%20E1.py)

Importamos las bibliotecas necesarias:
```python
import tensorflow as tf
import requests
import gradio as gr
```

Descargamos nuestro modelo pre-entrenado y el nombre de sus labels:
```python
inception_net = tf.keras.applications.MobileNetV2()
ans = requests.get("https://git.io/JJkYN").text
etiquetas = ans.split("\n")
print(etiquetas[:10])
```
Respuesta esperada:
```commandline
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
14536120/14536120 [==============================] - 3s 0us/step
['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich']
```

Creamos nuestra función de clasificación:
```python
def clasifica_imagen(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    print(prediction[:5])
    confidences = {etiquetas[i]: float(prediction[i]) for i in range(1000)}
    return confidences
```
Creamos nuestra demo con Gradio:
```commandline
demo = gr.Interface(fn=clasifica_imagen,
                    inputs=gr.Image(shape=(224, 224)),
                    outputs=gr.Label(num_top_classes=3)
                    )
demo.launch()
```
Respuesta esperada:
```commandline
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```
![8.png](imgs%2F2%20Gradio%2F8.png)
```commandline
1/1 [==============================] - 0s 14ms/step
[2.3379953e-05 6.3485393e-05 1.9028716e-05 7.3523290e-05 9.2093105e-05]
```

**Ejemplo 2:** demo cargando un modelo del Hub de Hugging Face.

> Nota: este método es sumamente superficial, sin embargo, es bueno conocer la forma en la que podemos crear demos directamente
> desde los modelos pre-entrenados de `Hugging Face`, este me permite simplificar el proceso que de por sí es simple.

[2 Clasificación Imágenes E2.py](2%20Gradio%2F2%20Clasificaci%C3%B3n%20Im%C3%A1genes%20E2.py)

```python
import gradio as gr

titulo = "Mi primer demo con Hugging Face"
desc = "Este es un demo ejecutado durante la clase con Platzi."

gr.Interface.load(
    "huggingface/microsoft/swin-tiny-patch4-window7-224",
    inputs=gr.Image(label="Carga una imagen aquí"),
    title=titulo,
    description=desc
).launch()

```
Respuesta esperada:
```commandline
Fetching model from: https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

![9.png](imgs%2F2%20Gradio%2F9.png)


## 2.4 Compartir demos en Spaces de Hugging Face

Primero debemos crear un `space` en Hugging Face

![10.png](imgs%2F2%20Gradio%2F10.png)

https://huggingface.co/new-space

![11.png](imgs%2F2%20Gradio%2F11.png)

Existen otros `Hardware` disponibles para nuestros spaces, pero son de cobro y por eso vamos a obviarlos por este momento.

Hugging Face funciona básicamente igual que un repositorio de GitHub

![12.png](imgs%2F2%20Gradio%2F12.png)

Podemos ver que tenemos nuestro url para hacer un `git clone` y de igual forma nos dan el código para empezar a hacer nuestros
commits y push:

```commandline
git clone https://huggingface.co/spaces/Ichcanziho/Demo_mi_primer_space
```

```commandline
git add app.py
git commit -m "Add application file"
git push
```

Sin embargo, para esta primera clase vamos a usar la Interfaz de Hugging Face para crear nuestra `App.py` donde estará nuestra primera demo.

![13.png](imgs%2F2%20Gradio%2F13.png)

Ahora vamos a `files/add file/create new file`

Vamos a utilizar el código de la clase pasada:

```python
import gradio as gr

titulo = "Mi primer demo con Hugging Face | Clasificación de imágenes"
desc = "Este es un demo creado en la clase de Platzi."

gr.Interface.load(
    "huggingface/microsoft/swin-tiny-patch4-window7-224",
    inputs=gr.Image(label="Carga una imagen aquí"),
    title=titulo,
    description=desc
).launch()

```

![14.png](imgs%2F2%20Gradio%2F14.png)

Ahora vamos a ver como quedo nuestro primer space:

![15.png](imgs%2F2%20Gradio%2F15.png)

Puedes acceder a él desde la siguiente liga:

https://huggingface.co/spaces/Ichcanziho/Demo_mi_primer_space

Ahora mi perfil tiene mi primer demo:

![16.png](imgs%2F2%20Gradio%2F16.png)


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
