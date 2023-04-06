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

En esta clase veremos como de forma sencilla podemos crear una demo de Audio2Text

Para ubuntu necesite instalar la siguiente dependencia:
```bash
sudo apt-get install ffmpeg
```

[3 Audio2Text.py](2%20Gradio%2F3%20Audio2Text.py)

```python
from transformers import pipeline
import gradio as gr


def transcribe(audio):
    text = modelo(audio)["text"]
    return text


if __name__ == '__main__':
    modelo = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish")

    gr.Interface(
        fn=transcribe,
        inputs=[gr.Audio(source="microphone", type="filepath")],
        outputs=["textbox"]
    ).launch()

```

> Para más información sobre el input `Audio` podemos ir a su documentación: https://gradio.app/docs/#audio
>

Respuesta esperada:
```commandline
Downloading (…)lve/main/config.json: 100%|██████████| 1.29k/1.29k [00:00<00:00, 1.05MB/s]
Downloading pytorch_model.bin: 100%|██████████| 1.26G/1.26G [04:05<00:00, 5.14MB/s]
Downloading (…)okenizer_config.json: 100%|██████████| 380/380 [00:00<00:00, 365kB/s]
Downloading (…)olve/main/vocab.json: 100%|██████████| 370/370 [00:00<00:00, 339kB/s]
Downloading (…)cial_tokens_map.json: 100%|██████████| 85.0/85.0 [00:00<00:00, 79.9kB/s]
Downloading (…)rocessor_config.json: 100%|██████████| 158/158 [00:00<00:00, 208kB/s]
Running on local URL:  http://127.0.0.1:7860
```

![17.png](imgs%2F2%20Gradio%2F17.png)

Puedes probar el demo en el siguiente space: https://huggingface.co/spaces/Ichcanziho/Voice2Text

![18.png](imgs%2F2%20Gradio%2F18.png)

> ## Nota importante:
> Para este space fue necesario crear un archivo requirements.txt para añadir las siguientes dependencias de código:
>   > transformers
>   > 
>   > torch

## 2.6 El futuro de los demos: Blocks

En este ejercicio vamos a ver como podemos enlazar de forma secuencial la salida de un modelo como entrada para otro modelo.
Para este trabajo vamos a utilizar `blocks` de `gradio` para cumplir con la función:

[4 Blocks E1.py](2%20Gradio%2F4%20Blocks%20E1.py)

Importamos las bibliotecas habituales:

```python
import gradio as gr
from transformers import pipeline
```

Utilizamos `pipeline` para descargar los modelos que vamos a utilizar:

```python
trans = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish")
clasificador = pipeline("text-classification", model="pysentimiento/robertuito-sentiment-analysis")
```

Creamos las funciones que utilizan a los modelos de `pipeline`:

```python
def audio_a_text(audio):
    text = trans(audio)["text"]
    return text


def texto_a_sentimiento(text):
    sentiment = clasificador(text)[0]["label"]
    return sentiment
```

Y Ahora viene la parte más interesante, creamos un bloque secuencial en donde estaremos poniendo diferentes `widgets` para
interactuar con el flujo de información:

```python
def make_block(dem):
    with dem:
        # Este bloque nos permite incertar código en MarkDown
        gr.Markdown("# Demo para la clase de Platzi")
        # Creamos un widget de Audio para grabar la voz con el micrófono
        audio = gr.Audio(source="microphone", type="filepath")
        # Creamos un widget de Textbox para poner el texto transcrito por el modelo o para escribir directamente en él
        texto = gr.Textbox()
        # Creamos un widget de Botón para que al presionarlo empiece la clasificación
        b1 = gr.Button("Transcribe porfa")
        # AL momento de hacer click en el botón va a llamar a la función Audio a Text, usando como entrada el "audio" grabado por el widget Audio
        # y la salida será un texto que se mostrará en el widget texto.
        b1.click(audio_a_text, inputs=audio, outputs=texto)
        
        # Podemos usar un label vacío para usarlo como placeholder y poner aquí la clasificación del sentimiento del texto
        label = gr.Label()
        # Asignamos un segundo botón para clasificar el texto y obtener su sentimiento
        b2 = gr.Button("Clasifica porfa el sentimiento")
        # Al momento de hacer click vamos a llamar a la función texto_a_sentimiento, que tiene como entrada el texto presente en el TextBox texto
        # y Como salida devuelve el sentimiento en el Widget Label.
        b2.click(texto_a_sentimiento, inputs=texto, outputs=label)
```

Finalmente, creamos un demo, llamamos a nuestra función `make_block()` y lanzamos el demo:
```python
demo = gr.Blocks()
make_block(demo)
demo.launch()
```

Respuesta esperada:

![19.png](imgs%2F2%20Gradio%2F19.png)

¡Excelente! Hemos creado una aplicación completa compuesta de 2 modelos de DeepLearning y usando un flujo de información
compuesta de widgets contenidos en un block.


## 2.7 Blocks con Tabs y TabItem

En este proyecto vamos a utilizar 3 clasificadores y a cada uno de ellos lo vamos a asignar a una `pestaña` diferente:

[5 Blocks con Tabs.py](2%20Gradio%2F5%20Blocks%20con%20Tabs.py)

Importamos bibliotecas necesarias:
```python
from transformers import pipeline
import gradio as gr
from PIL import Image
```
Descargamos los modelos que vamos a utilizar:
```python
image_cla = pipeline("image-classification", model="microsoft/swin-tiny-patch4-window7-224")
voice_cla = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53-spanish")
text_cla = pipeline("text-classification", model="pysentimiento/robertuito-sentiment-analysis")
```
Creamos la función de clasificación de cada modelo de acuerdo a sus necesidades:
```python
def classify_img(im):
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    ans = image_cla(im)
    labels = {v["label"]: v["score"] for v in ans}
    return labels


def voice2text(audio):
    text = voice_cla(audio)["text"]
    return text


def text2sentiment(text):
    sentiment = text_cla(text)[0]["label"]
    return sentiment
```

Creamos nuestra función que va a construir nuestro `bloque` a partir del `demo`:

```python
def make_block(dem):
    with dem:
        # Empezamos por decir que vamos a utilizar una estructura de pestañas
        with gr.Tabs():
            # Definimos que esta pestaña va a tener el siguiente título
            with gr.TabItem("Transcribe audio en español"):
                # La pestaña va a estar compuesta de un único ROW
                with gr.Row():
                    # Dentro del cuál tendra un widget de Audio como input y un widget de TextBox como output
                    audio = gr.Audio(source="microphone", type="filepath")
                    transcripcion = gr.Textbox()
                b1 = gr.Button("Voz a Texto")
            # El proceso se repite con las demás pestañas
            with gr.TabItem("Análisis de sentimiento en español"):
                with gr.Row():
                    texto = gr.Textbox()
                    label = gr.Label()
                b2 = gr.Button("Texto a Sentimiento")
            # Cada pestaña tiene sus propias necesidades en cuanto a Widgets de input y Output
            with gr.TabItem("Clasificación de Imágenes"):
                with gr.Row():
                    image = gr.Image(label="Carga una imagen aquí")
                    label_image = gr.Label(num_top_classes=5)
                b3 = gr.Button("Clasifica")
            
            # Al finalizar cada pestaña agregamos un boton al fondo para accionar el proceso de clasificación de cada modelo.
            b1.click(voice2text, inputs=audio, outputs=transcripcion)
            b2.click(text2sentiment, inputs=texto, outputs=label)
            b3.click(classify_img, inputs=image, outputs=label_image)

```
Corremos nuestro código:
```python
demo = gr.Blocks()
make_block(demo)
demo.launch()
```
Y subimos los resultados a `Hugging Face`

![20.png](imgs%2F2%20Gradio%2F20.png)

## 2.8 Quizz Gradio

![21.png](imgs%2F2%20Gradio%2F21.png)

# 3 Streamlit
## 3.1 Introducción a Streamlit

Streamlit es una biblioteca de Python de código abierto diseñada para crear aplicaciones web interactivas y personalizadas con rapidez y facilidad. Permite a los desarrolladores crear aplicaciones de datos dinámicas y visualizaciones interactivas de una manera simple y eficiente, sin tener que preocuparse por la implementación de la lógica de back-end o el diseño de la interfaz de usuario.

Con Streamlit, puedes escribir código Python y generar automáticamente una aplicación web interactiva. Incluye un conjunto de herramientas y widgets preconstruidos que pueden utilizarse para crear aplicaciones de manera más rápida.

![1.png](imgs%2F3%20Streamlit%2F1.png)

Además, Streamlit también ofrece una serie de características, como la actualización en vivo de los datos y las visualizaciones, lo que significa que los usuarios pueden ver cambios en tiempo real en sus aplicaciones a medida que interactúan con ellas. También es fácil de instalar y configurar, lo que lo convierte en una herramienta popular para aquellos que desean crear aplicaciones web de datos personalizadas y visualizaciones interactivas.

https://streamlit.io/

```bash
pip install streamlit
```

Ejemplo básico de uso:

```python
import streamlit as st

x = st.slider("Selecciona un valor")
st.write(x, "al cuadrado es:", x*x)

```

## 3.2 Primeros pasos con Streamlit

Primero vamos a crear un nuevo `space` de hugging face

![2.png](imgs%2F3%20Streamlit%2F2.png)

Nos dirigimos a la carpeta de `streamlit` de este curso:

[3 streamlit](3%20streamlit)

```commandline
(venv) ichcanziho@ichcanziho:~/Documentos/programacion/Deep Learnining/5 Curso de introducción a Demos de Machine Learning con Hugging Face/3 streamlit$ 
```

Clonamos el repositorio de hugging face
```bash
git clone https://huggingface.co/spaces/Ichcanziho/stremlit_space_test
cd stremlit_space_test
```
Ahora vamos a crear un nuevo código python:

[app.py](3%20streamlit%2Fstremlit_space_test%2Fapp.py)

```python
import streamlit as st

st.title("Este es mi demo")
st.markdown("También puedo usar `markdown`")

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)
```
Vamos a terminal y ejecutamos el siguiente comando para correr el código:
```bash
streamlit run app.py
```
Respuesta esperada:
```commandline
Summary:
  - This open source library collects usage statistics.
  - We cannot see and do not store information contained inside Streamlit apps,
    such as text, charts, images, etc.
  - Telemetry data is stored in servers in the United States.
  - If you'd like to opt out, add the following to ~/.streamlit/config.toml,
    creating that file if necessary:

    [browser]
    gatherUsageStats = false


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.100.14:8501

```

![3.png](imgs%2F3%20Streamlit%2F3.png)

Ahora como estamos usando `git` debemos hacer los siguientes pasos convencionales de git:

```bash
git add.py
git commit -m "cambio en app"
git push
```
Podemos ir a nuestro `space` de hugging face y observar los resultados:

![5.png](imgs%2F3%20Streamlit%2F5.png)

https://huggingface.co/spaces/Ichcanziho/stremlit_space_test

Para tener más información sobre todas las posibilidadades que nos ofrece `streamlit` es siempre buena idea tener a la mano
su `api reference`: https://docs.streamlit.io/library/api-reference

![4.png](imgs%2F3%20Streamlit%2F4.png)


## 3.3 Demo de generación de imágenes con Streamlit

Para este proyecto crearemos un demo capaz de generar imágenes de mariposas de forma artificial utilizando Gans.

El modelo base que estaremos utilizando es el siguiente: https://huggingface.co/ceyda/butterfly_cropped_uniq1K_512

Es indispensable entender el `how to use` del modelo:

```python

import torch
from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN # install the community-events repo above

gan = LightweightGAN.from_pretrained("ceyda/butterfly_cropped_uniq1K_512")
gan.eval()
batch_size = 1
with torch.no_grad():
        ims = gan.G(torch.randn(batch_size, gan.latent_dim)).clamp_(0., 1.)*255
        ims = ims.permute(0,2,3,1).detach().cpu().numpy().astype(np.uint8)
        # ims is [BxWxHxC] call Image.fromarray(ims[0])
```

Este ejemplo lo usaremos más adelante, pero este código NO lo vamos a correr de forma local, por ello no será necesario instalar
`huggan` por el contrario vamos a hacerlo un requisito de instalación cuando hagamos el deploy a hugging face. Esto lo conseguimos
de forma sencilla utilizando un archivo de `requirements.txt` y añadiendo el siguiente requisito:

```commandline
git+https://github.com/huggingface/community-events.git@3fea10c5d5a50c69f509e34cd580fe9139905d04#egg=huggan
```

Con todo esto definido, empecemos por el `backend` del proyecto, la parte de código que será encargada de generar las mariposas:

[utils.py](3%20streamlit%2Fmariposas_test%2Futils.py)

Empezamos por importar las bibliotecas necesarias:
```python
import numpy as np
import torch
from huggan.pytorch.lightweight_gan.lightweight_gan import LightweightGAN
```

Podemos crear una función aislada encargada de cargar el modelo de `LightweightGAN` desde un `model_name`y `version`
```python
# Cargamos el modelo desde el Hub de Hugging Face
def carga_modelo(model_name="ceyda/butterfly_cropped_uniq1K_512", model_version=None):
    gan = LightweightGAN.from_pretrained(model_name, version=model_version)
    gan.eval()
    return gan
```

Definimos la función objetivo `genera` que será la que utilice el modelo para generar las imágenes de mariposas:
```python
# Usamos el modelo GAN para generar imágenes
def genera(gan, batch_size=1):
    with torch.no_grad():
        ims = gan.G(torch.randn(batch_size, gan.latent_dim)).clamp_(0.0, 1.0) * 255
        ims = ims.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
    return ims
```
Perfecto, a este punto ya hemos creado todos los servicios necesarios que utilizará nuestra aplicación, en la siguiente clase
vamos a ver como usar `streamlit` para crear el `front` de nuestro `space`


## 3.4 Creando interfaz de demo de GAN

Nuestro proyecto busca tener la siguiente interfaz gráfica:

![6.png](imgs%2F3%20Streamlit%2F6.png)

La interfaz se compone principalmente de dos secciones, la barra lateral con información `adicional` y la zona principal
a la derecha. Cada uno de los widgets estarán en forma de `stack` esto quiere decir que se van a acomodar uno debajo del otro
de forma vertical.

Con esto en mente podemos empezar a crear el siguiente código:

[app.py](3%20streamlit%2Fmariposas_test%2Fapp.py)

Primero, importamos las bibliotecas necesarias:
```python
import streamlit as st
from utils import carga_modelo, genera
```

Lo primero que podemos observar con streamlit importado como `st` es que NO será necesario instanciar un objeto de esta clase, entonces podemos
usar sus métodos de clase directamente. Empezamos construyendo la página principal:

```python
# Página principal
st.title("Butterfly GAN (GAN de mariposas)")
st.write(
    "Modelo Light-GAN entrenado con 1000 imágenes de mariposas tomadas de la colección del Museo Smithsonian."
)
```

Antes de continuar debemos crear en nuestro proyecto una carpeta llamada `assets` (por convención) y poner ahí la imagen que vamos a usar
como logotipo del proyecto (esto es solo con fines estéticos) ![logo.png](3%20streamlit%2Fmariposas_test%2Fassets%2Flogo.png) con los requisitos anteriores
cumplidos vamos a crear la barra lateral de información adicional.

```python
# Barra lateral
st.sidebar.subheader("¡Estas mariposas no existen! 🤯.")
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.caption(
    f"[Modelo](https://huggingface.co/ceyda/butterfly_cropped_uniq1K_512) y [Dataset]("
    f"https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) usados."
)
st.sidebar.caption(f"*Disclaimers:*")
st.sidebar.caption(
    "* Este demo creada a partir del curso de Platzi: Curso de Experimentación en Machine Learning con Hugging Face."
)
```

Podemos observar que en general todo tiene un formato de `markdown` es importante tener conocimientos de este tipo de escritura.

Ahora podemos seguir definiendo un par de parámetros del modelo que vamos a utilizar:

```python
# Cargamos modelo
repo_id = "ceyda/butterfly_cropped_uniq1K_512"
version_modelo = "57d36a15546909557d9f967f47713236c8288838"
modelo_gan = carga_modelo(repo_id, version_modelo)

# Generamos 4 mariposas
n_mariposas = 4
```

Para generar información `imagenes`, `texto` etc, es necesario que estos datos estén almacenados en algún lugar, para ello vamos a utilizar
`streamlit.session_state` algo simbólicamente similar a las `cookies` de un navegador. Cuando se utiliza `session_state`, 
los datos se almacenan en una instancia de SessionState que se crea la primera vez que se accede a ella. 
Esta instancia se mantiene en memoria mientras la aplicación se esté ejecutando, lo que permite que los datos almacenados 
en ella persistan a través de distintas ejecuciones de las funciones.

La forma de utilizarlas es igual al uso de `diccionarios` en python, podemos crear `keys` usando `session_state["mi_variable"]` 
o `session_state.mi_variable` y asignarles un valor para usarlo más adelante. Es importante tener en cuenta que session_state 
no se puede utilizar fuera de una función de Streamlit, ya que requiere acceso al contexto de la aplicación.

Con todo esto en mente, vamos a definir el siguiente código para la generación de mariposas y lo explicaremos con más detalle.

```python
# Función que genera mariposas y lo guarda como un estado de la sesión
def corre():
    with st.spinner("Generando, espera un poco..."):
        ims = genera(modelo_gan, n_mariposas)
        st.session_state["ims"] = ims


# Si no hay una imagen generada entonces generala
if "ims" not in st.session_state:
    st.session_state["ims"] = None
    corre()

# ims contiene las imágenes generadas
ims = st.session_state["ims"]
```
Empezamos con una función `corre` encargada de generar las imágenes, pero en lugar de simplemente regresar las imágenes con un `return` vamos a guardar
esta información en una `cookie` de `streamlit` con `st.session_state`. La primera vez que carguemos él `space` `streamlit`
se va a preguntar: 
> Acaso dentro de mis cookies actuales tengo alguna llamada `ims`

Dado que es la primera vez que se carga el `space` la respuesta va a ser `NO` entonces debo crear esa `cookie` asignándole un
valor `None` este valor no importa porque justo después vamos a mandar a la función `corre` la cual ya sabemos que asigna
el valor de las mariposas a la `cookie` de `ims.

Entonces a este punto ya sabemos que ya sea que es la primera vez que cargamos él `space` o que se empezó una petición para generar
nuevas mariposas sí o sí existe una `cookie` llamada `ims` que contiene nuestras 4 mariposas, así que podemos acceder al valor
de esta variable: `ims = st.session_state["ims"]`

Ahora tenemos que crear un espacio para poder poner las imágenes de nuestras mariposas:

```python
if ims is not None:
    cols = st.columns(n_mariposas)
    for j, im in enumerate(ims):
        i = j % n_mariposas
        cols[i].image(im, use_column_width=True)
```
De forma muy sencilla el código de arriba es encargado de generar una columna para cada imagen de mariposa generada y a cada una de esas 
imágenes la pone en su lugar con: `cols[i].image(im, use_column_width=True)` el parámetro `use_column_width=True` es por cuestiónes de estética
para que las imágenes tengan el mismo ancho que la columna calculada.

Para terminar con nuestra interfaz gráfica podemos agregar un boton `triger` que corra de nuevo la función `corre` y de esta manera
se creen nuevas imágenes de mariposas:

```python
# Si la usuaria da click en el botón entonces corremos la función genera()
corre_boton = st.button(
    "Crea nuevas mariposas :D",
    on_click=corre,
    help="Estamos en pleno vuelo, puede tardar.",
)

```

## 3.5 Probando demo de generación de imágenes

Genial, ahora solo nos falta crear nuestro archivo `requirements.txt`

```commandline
git+https://github.com/huggingface/community-events.git@3fea10c5d5a50c69f509e34cd580fe9139905d04#egg=huggan
transformers
```

y hacemos los pasos que ya conocemos de git:

```bash
git add .
git commit -m "commit de mariposas"
git push
```

Excelente, ahora nuestro proyecto se encuentra en el hub de hugging face, puedes acceder a él en el siguiente link:

https://huggingface.co/spaces/Ichcanziho/mariposas_test


## 3.6 Configurar un demo clonado desde Hugging Face

Esta clase se puede resumir en que puedes tomar código de otros repositorios y utilizarlo libremente para partir de ahì
y crear tus nuevos proyectos. No agrega mucha información en realidad. 

## 3.7 Quizz Streamlit

![7.png](imgs%2F3%20Streamlit%2F7.png)
