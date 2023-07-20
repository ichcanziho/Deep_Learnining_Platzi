# Curso de Desarrollo de Chatbots con OpenAI

Aprende a usar la API de OpenAI para desarrollar un chatbot con identidad propia. Da este paso hacia una experiencia conversacional impulsada por la inteligencia artificial con GPT-4, GPT-3.5 y Davinci.

- Integra un LLM de OpenAI a una aplicación de chat.
- Estima el costo de consumo por tokens de la API.
- Aplica fine-tuning a un modelo para personalizarlo.


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> - [5: Curso de Experimentación en Machine Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/5%20Curso%20de%20introducci%C3%B3n%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
> - [6: Curso de detección y segmentación de objetos con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow)
> - [7: Curso profesional de Computer Vision con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/7%20Curso%20profesional%20de%20Computer%20Vision%20con%20TensorFlow)
> - [8: Curso de generación de imágenes](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/8%20Curso%20de%20generaci%C3%B3n%20de%20im%C3%A1genes)
> - [9: Cursos de Fundamentos de NLP](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/9%20Curso%20de%20Fundamentos%20de%20NLP)
> - [10: Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/10%20Curso%20de%20Algoritmos%20de%20Clasificaci%C3%B3n%20de%20Texto)
> - [11: Curso de redes Neuronales con PyTorch](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/11%20Curso%20de%20Redes%20Neuronales%20con%20PyTorch%20)
> 
> Este Curso es el Número 12 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso


# ÍNDICE:

- [1 OpenAI API](#1-openai-api)
  - [1.1 ¿Cómo usar la API de OpenAI en tu producto?](#11-cómo-usar-la-api-de-openai-en-tu-producto)
  - [1.2 Conociendo la documentación de la API de OpenAI](#12-conociendo-la-documentación-de-la-api-de-openai)
  - [1.3 Cargar modelo de la API de OpenAI con Python](#13-cargar-modelo-de-la-api-de-openai-con-python)
  - [1.4 Creación de ejemplo utilizando la API de OpenAI](#14-creación-de-ejemplo-utilizando-la-api-de-openai)
  - [1.5 Parámetros de Text Completion: temperature, top_p y n](#15-parámetros-de-text-completion-temperature-topp-y-n)
  - [1.6 Buenas prácticas al usar modelos de OpenAI](#16-buenas-prácticas-al-usar-modelos-de-openai)
  - [1.7 Chat Completions](#17-chat-completions)
  - [1.8 Actualizaciones de la API de OpenAI: GPT-4 disponible y modelos deprecados](#18-actualizaciones-de-la-api-de-openai-gpt-4-disponible-y-modelos-deprecados)
  - [Quiz de OpenAI API](#quiz-de-openai-api)
- [2 Fine-tuning de modelos de OpenAI](#2-fine-tuning-de-modelos-de-openai)
  - [2.1 ¿Por qué hacer fine-tuning a modelos de OpenAI?](#21-por-qué-hacer-fine-tuning-a-modelos-de-openai)
  - [2.2 Costos de uso de OpenAI: tokenización de texto](#22-costos-de-uso-de-openai-tokenización-de-texto)
  - [2.3 Configuración de entorno local de OpenAI con Anaconda](#23-configuración-de-entorno-local-de-openai-con-anaconda)
  - [2.4 Formato de datos para fine-tuning](#24-formato-de-datos-para-fine-tuning)
  - [2.5 Preparar datos para fine-tuning](#25-preparar-datos-para-fine-tuning)
  - [2.6 Fine-tuning de modelo de OpenAI](#26-fine-tuning-de-modelo-de-openai)
  - [2.7 ¿Cómo usar PlayGround de OpenAI para probar modelos?](#27-cómo-usar-playground-de-openai-para-probar-modelos)
  - [2.8 Pruebas al modelo con fine-tuning](#28-pruebas-al-modelo-con-fine-tuning)
  - [2.9 Optimizar el modelo: ajuste de parámetros en Playground](#29-optimizar-el-modelo-ajuste-de-parámetros-en-playground)
  - [2.10 Validación de modelos fine-tuned de OpenAI](#210-validación-de-modelos-fine-tuned-de-openai)
  - [Quiz de fine-tuning de modelos de OpenAI](#quiz-de-fine-tuning-de-modelos-de-openai)
- [3 Integración de modelo a aplicación de chat](#3-integración-de-modelo-a-aplicación-de-chat)
  - [3.1 ¿Cómo crear un chatbot con Telegram?](#31-cómo-crear-un-chatbot-con-telegram)
  - [3.2 Procesando la entrada del usuario para el chatbot](#32-procesando-la-entrada-del-usuario-para-el-chatbot)
  - [3.3 Prueba de envío de mensajes del chatbot](#33-prueba-de-envío-de-mensajes-del-chatbot)
  - [3.4 Función main() del chatbot](#34-función-main-del-chatbot)
  - [3.5 Integración del modelo de OpenAI a Telegram](#35-integración-del-modelo-de-openai-a-telegram)
  - [3.6 Manejo de errores y excepciones de la API de OpenAI](#36-manejo-de-errores-y-excepciones-de-la-api-de-openai)
  -[Quiz de integración de LLM a chat](#quiz-de-integración-de-llm-a-chat)
- [4 Conclusión](#4-conclusión)
  - [4.1 Recomendaciones finales y proyectos alternativos con el API de OpenAI](#41-recomendaciones-finales-y-proyectos-alternativos-con-el-api-de-openai)

# 1 OpenAI API

## 1.1 ¿Cómo usar la API de OpenAI en tu producto?

Entre las tareas más comunes que podemos hacer utilizando LLM's tenemos:

![1.png](ims%2F1%2F1.png)

- **Clasificación**: Los LLM's se pueden utilizar para clasificar textos en categorías específicas. Por ejemplo, pueden ser entrenados para clasificar correos electrónicos como "spam" o "no spam", noticias como "política" o "deportes", comentarios como "positivos" o "negativos", entre otros.

- **Generación**: Los LLM's son muy útiles para generar texto coherente y cohesivo a partir de un prompt o una pregunta inicial. Pueden ser utilizados para generar respuestas automáticas, redacciones, resúmenes de texto, descripciones de imágenes, entre otros.

- **Traducción**: Los LLM's también pueden ser utilizados para tareas de traducción automática. Al entrenarlos con datos de pares de idiomas, se puede lograr que el modelo genere traducciones coherentes y precisas de textos en diferentes idiomas.

- **Chatbots**: Los LLM's son una herramienta fundamental para la creación de chatbots. Pueden ser entrenados con datos de diálogos para que el modelo pueda responder preguntas, entablar conversaciones y simular interacciones humanas de manera natural.

- **Programación**: Los LLM's también pueden utilizarse para generar código o ayudar en tareas de programación. Pueden ser entrenados con datos de código fuente y utilizados para autocompletar código, proporcionar sugerencias de código, corregir errores o incluso generar código completo a partir de una descripción.

En este curso vamos a utilizar modelos tipo GPT para tareas relacionadas con Texto.

Veamos nuestro primer ejemplo de Código. ¿Cómo acceder a API de OpenAI desde código en Python?

> ## Nota: 
> El código de esta sección lo puedes encontrar en: [1_tweet_sentiment.py](scripts%2F1_tweet_sentiment.py)

> Antes de continuar debemos instalar un par de librerías en python

En este ejercicio vamos a utilizar Chat GPT para hacer un clasificador de Sentimientos **positivos, neutrales, negativos** de **tweets**,

```bash
pip install python-dotenv
pip install openai
```

Antes de continuar asegurate de generar tu primer **API KEY** de OpenAI: https://platform.openai.com/account/api-keys

![2.png](ims%2F1%2F2.png)

Esta API KEY ES PERSONAL cuídala mucho y NO la compartas con nadie. Una vez generada tienes que guardarla porque no podrás volver
a acceder a ella desde la página de OpenAI.

Personalmente, voy a crear un archivo llamado `ap.env` en donde almacenaré mi `API KEY` de OpenAI

```commandline
OPENAI_API_KEY=sk-clavellenadeamor
```
De esta manera en Python cuando quiera acceder a ella no tendré que tenerla visible al público. Vamos a empezar importando
las bibliotecas y accediendo a nuestra variable de entorno.
```python
import os
from dotenv import load_dotenv
import openai

# Carga las variables de entorno desde el archivo .env
load_dotenv("../envs/ap.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
```

Ahora vamos a conocer un poco sobre nuestro primer método: `openai.Completion.create()`

- **model**: Especifica el modelo de lenguaje que se utilizará para generar el texto. Puedes elegir entre diferentes modelos, como "gpt-3.5-turbo" o "text-davinci-003". Cada modelo tiene diferentes características y capacidades, por lo que es importante seleccionar el adecuado para tu caso de uso.

- **prompt**: Es el texto inicial o la consulta que proporcionas al modelo como punto de partida para generar el texto continuado. Puedes utilizar un prompt específico para guiar al modelo en la dirección deseada o para establecer el contexto para la generación de texto.

- **temperature**: Controla la aleatoriedad de las respuestas generadas por el modelo. Un valor más bajo, como 0.2, produce respuestas más determinísticas y coherentes, mientras que un valor más alto, como 0.8, genera respuestas más creativas pero potencialmente menos coherentes.

- **max_tokens**: Define el número máximo de tokens que se generarán en la respuesta. Un token puede ser una palabra o un carácter, según el modelo que estés utilizando. Si deseas limitar la longitud del texto de salida, puedes establecer este parámetro en un valor adecuado.

- **top_p**: También conocido como "nucleus sampling" o "restricción de token de parche". Limita la generación de texto a una distribución de probabilidad acumulativa superior a un cierto umbral. Un valor más bajo, como 0.2, hará que las respuestas sean más restrictivas y enfocadas, mientras que un valor más alto, como 0.8, permitirá una mayor diversidad en las respuestas generadas.

- **frequency_penalty**: Este parámetro controla la preferencia del modelo por evitar la repetición de frases. Un valor más alto, como 0.6, penalizará más la repetición y hará que el modelo evite generar frases similares. Un valor de 0 no penaliza la repetición.

- **presence_penalty**: Este parámetro controla la preferencia del modelo por evitar la mención de ciertas palabras o temas en el texto de salida. Un valor más alto, como 0.6, penalizará más la presencia de ciertas palabras o temas y hará que el modelo evite mencionarlos. Un valor de 0 no penaliza la presencia de palabras o temas específicos.


Ahora vamos a presentar nuestra primera petición al API de OpenAI:

```python
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Decide si el sentimiento de un Tweet es positivo, neutral, o negativo. \
  \n\nTweet: \"#LoNuevoEnPlatzi es el Platzibot 🤖. Un asistente creado con Inteligencia Artificial para acompañarte en tu proceso de aprendizaje.\
  \"\nSentiment:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)
print(response.choices[0].text)
```
Respuesta esperada:
```commandline
 Positivo
```

Excelente, en esta clase hemos aprendido a conectarnos al API de OpenAI a través de su biblioteca en Python, hemos creado
un archivo .env para almacenar nuestra API KEY y hemos conocido algunos de los parámetros básicos que tiene el método
`Completion.create` y hemos logrado hacer un clasificador de sentimientos de tweets.

## 1.2 Conociendo la documentación de la API de OpenAI

Es indispensable que tengamos muy presente en todo momento la Documentación de OpenAI: https://platform.openai.com/docs/introduction/overview

![3.png](ims%2F1%2F3.png)

Este sitio será nuestro mejor aliado para conocer a profundidad las características de los modelos y servicios que nos puede
ofrecer OpenAI. 

Algunos de los enlaces más interesantes a los que debemos prestar atención son:

- [Models](https://platform.openai.com/docs/models)
  ![4.png](ims%2F1%2F4.png)
- [Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
  ![5.png](ims%2F1%2F5.png)
- [Chat Completions](https://platform.openai.com/docs/guides/gpt/chat-completions-api)
  ![6.png](ims%2F1%2F6.png)

Cada uno de estos elementos los estaremos visitando a lo largo del curso, sin embargo, es importante tener en cuenta que hay
mucho más que explorar en la documentación y que la misma tiene tutoriales que nos pueden enseñar a como utilizarla.

Finalmente, también podemos leer su [API REFERENCE](https://platform.openai.com/docs/api-reference)

![7.png](ims%2F1%2F7.png)

Es similar a la documentación, pero está más orientada en como consumir sus servicios desde el API y sus tutoriales son a menor profundidad.

## 1.3 Cargar modelo de la API de OpenAI con Python

Esta clase será un repaso de la primer clase en donde veremos como fácilmente cargar nuestra API KEY, seleccionar un modelo y finlamente
enviar una prompt a chat GPT desde código:

> ## Nota:
> Puedes encontrar el código en: [2_cargar_modelo.py](scripts%2F2_cargar_modelo.py)

```python
import os
from dotenv import load_dotenv
import openai

load_dotenv("../envs/ap.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="¿Quién descubrió América?",
    max_tokens=100
)

print(response.choices[0].text)
```

Respuesta esperada:
```commandline
Ninguna persona descubrió América, ya que el continente ya estaba habitado por pueblos indígenas cuando los europeos llegaron. El navegante generalmente atribuido como el descubridor de América es Cristóbal Colón, quien realizó su primera expedición en 1492.
```
Aclaración, es probable que tu respuesta sea diferente a la que yo he conseguido. ChatGPT no necesariamente va a responder exactamente igual a la misma pregunta.

## 1.4 Creación de ejemplo utilizando la API de OpenAI

En este ejercicio vamos a crear un pequeño juego de `adivina el animal`. Primero dado una lista de animales seleccionaremos
uno de ellos al azar y vamos a dar una pista genérica al usuario. Su trabajo es intentar adivinar el animal. En cada itento
cuando el usuario falle, vamos a hacer que ChatGPT generé una nueva pista que mencione atributos del animal en cuestión, evitando
mencionar el nombre del animal, esto será indefinido hasta que el usuario adivine el animal.

> ## Nota:
> El código lo puedes encontrar en: [3_adivina_animal.py](scripts%2F3_adivina_animal.py)

Empezamos importando las bibliotecas necesarias:

```python
from dotenv import load_dotenv
import random
import openai
import os
```

Creamos nuestra función que elige un animal al azar y brinda la primera pista:

```python
def get_base_clue():
    words = ['elefante', 'león', 'jirafa', 'hipopótamo', 'mono']
    random_word = random.choice(words)
    prompt = 'Adivina la palabra que estoy pensando. Es un animal que vive en la selva.'
    return prompt, random_word
```

Ahora crearemos nuestro código de generación de nuevas pistas, que utiliza a ChatGPT para que dado un prompt me pueda regresar
características de cierto animal.

```python
def get_new_clue(animal):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt='Dame una caracteristica del tipo animal' + animal + ', pero jamás digas el nombre del animal',
        max_tokens=100)
    return response.choices[0].text
```

A continuación vamos a programar la mecánica principal del juego:

```python
def play_game():
    # Empezamos con nuestro animal aleatorio y primer pista genérica
    prompt, real_animal = get_base_clue()
    print(prompt)
    # Mientras la respuesta del usuario sea diferente al verdadero animal
    while (user_input := input("Ingresa tu respuesta: ")) != real_animal:
        # Le decimos que se equivocó
        print('Respuesta incorrecta. Intentalo de nuevo')
        # Y le damos una nueva pista
        print(get_new_clue(real_animal))
    # Si salimos del ciclo while es porque el usuario ha acertado
    print('Correcto! La respuesta era:', real_animal)
```
Y finalmente como ya tenemos todos los ingredientes, vamos a crear nuestro punto de acceso:
```python
if __name__ == '__main__':
    load_dotenv("../envs/ap.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    play_game()
```
Respuesta esperada:
```commandline
Adivina la palabra que estoy pensando. Es un animal que vive en la selva.
Ingresa tu respuesta: gato
Respuesta incorrecta. Intentalo de nuevo


Cuerpo voluminoso con piel áspera.
Ingresa tu respuesta: elefante
Respuesta incorrecta. Intentalo de nuevo


Tiene una piel gruesa y desagradable al tacto.
Ingresa tu respuesta: hipopótamo
Correcto! La respuesta era: hipopótamo
```
Excelente, ya hemos desarrollado un minijuego con ayuda del API de OpenAI.

## 1.5 Parámetros de Text Completion: temperature, top_p y n

Pese a que esta clase sí tiene código, es básicamente el mismo que ya hemos visto en: [2_cargar_modelo.py](scripts%2F2_cargar_modelo.py)
Así que en realidad solo vamos a repasar una breve explicación de los parámetros que dispone `Completion`

Para obtener la información directamente desde su Api Reference https://platform.openai.com/docs/api-reference/completions/create

**Parámetros de Text Completion**

- **model**: ID del modelo a utilizar.
- **prompt**: Las solicitudes para generar finalizaciones, codificadas como una cadena, una matriz de cadenas, una matriz de tokens o una matriz de matrices de tokens.
- **suffix**:Predeterminado a nulo
El sufijo que viene después de completar el texto insertado.
- **max_tokens**: Predeterminado a 16
El número máximo de tokens a generar en la finalización.
- **temperature**: Predeterminado a 1
Qué temperatura de muestreo usar, entre 0 y 2. Los valores más altos, como 0,8, harán que la salida sea más aleatoria, mientras que los valores más bajos, como 0,2, la harán más enfocada y determinista.
Usa esto o top_p pero no ambos.
- **top_p**: Predeterminado a 1
Una alternativa al muestreo con temperatura, llamado muestreo de núcleo, donde el modelo considera los resultados de los tokens con masa de probabilidad top_p. Por lo tanto, 0.1 significa que solo se consideran las fichas que comprenden el 10 % de la masa de probabilidad superior.
- **n**: Predeterminado a 1
Cuántas completions generar para cada prompt.
>Nota: debido a que este parámetro genera muchas finalizaciones, puede consumir rápidamente su cuota de token. Úselo con cuidado y asegúrese de tener configuraciones razonables para max_tokens y stop.
- **stream**: Predeterminado a falso
Ya sea para transmitir el progreso parcial. Si se establece, los tokens se enviarán como eventos enviados por el servidor solo de datos a medida que estén disponibles, y la secuencia terminará con un mensaje de data: [DONE]. [Ejemplo en python](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)
- **logprobs**: Si logprobs es 5, la API devolverá una lista de los 5 tokens más probables. La API siempre devolverá el logprob del token muestreado, por lo que puede haber hasta logprobs+1 elementos en la respuesta.
- **echo**: Repita el prompt además de la finalización(Predeterm: Falso)
stop: Hasta 4 secuencias donde la API dejará de generar más tokens. El texto devuelto no contendrá la secuencia de parada.
- **best_of**: Predeterminado a 1
Genera el mejor de completions del lado del servidor y devuelve el “mejor” (el que tiene la mayor probabilidad de registro por token). Los resultados no se pueden transmitir(stream).
- **user**: Un identificador único que representa a su usuario final, que puede ayudar a OpenAI a monitorear y detectar abusos. Aprende más.

## 1.6 Buenas prácticas al usar modelos de OpenAI

Es necesario tener en cuenta el costo del servicio del API de OpenAI

https://platform.openai.com/account/usage

![8.png](ims%2F1%2F8.png)
Aquí podemos ver el consumo de dinero que hemos tenido por cada día. Pero también estamos limitados no solamente en términos 
de efectivo, sino también en términos de: `Tokens Per Minute TPM` y `Request Per Minute` de acuerdo al modelo y acción que estemos haciendo:

https://platform.openai.com/account/rate-limits
![9.png](ims%2F1%2F9.png)

Diferentes miembros tienen diferentes roles, y solamente los miembros `owner` tienen acceso al `billing` o facturación.
Lo más importante a destacar es que existen dos conceptos:

> Nota, la siguiente imagen es tomada directo de la clase:

![10.png](ims%2F1%2F10.png)

Podemos definir un `soft limit` que es una Advertencia, OpenAI nos avisará por correo que ya hemos alcanzado este límite pero
no nos va a poner restricciones para seguir utilizándolo.

Mientras que si llegamos al `hard limit` entonces ahí se detendrán operaciones. 
También podemos pedir un `Request Increase` para aumentar nuestro límite mensual.

**Buenas prácticas al usar modelos de OpenAI**

- Uso:
  - Especifica claramente tu solicitud. Mayor contexto.
  - Utiliza la instrucción inicial
  - Controla la longitud de la respuesta.
  - Experimenta con la temperatura
- Facturación (Billing):
  - Consumo de usuarios y facturación.
  - Soft Limit y hard Limit:
  - Pedir si queremos aumentar límites.
  - Precios de Fine tuning models: https://openai.com/pricing
- Seguridad:
  - Gestión y solución de problemas.
  - Ética y consideraciones legales.
  - Privacidad de los datos.
  - Control de users(Owner y Reader).

## 1.7 Chat Completions

En esta clase vamos a ver ejemplos simples de como utilizar [Chat Completions API](https://platform.openai.com/docs/guides/gpt/chat-completions-api)
Es similar al método pasado, pero permite brindar mayor contexto al modelo.

También puedes ver más información en [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)

> ## Nota:
> Puedes ver el código de en: [4_chat_completion.py](scripts%2F4_chat_completion.py)

Partimos de la misma base que hemos estado utilizando en los otros ejemplos:

```python
import os
from dotenv import load_dotenv
import openai

load_dotenv("../envs/ap.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
```
Sin embargo, aquí viene el primer cambio. Ahora vamos a utilizar `openai.ChatCompletion.create`
y veremos que la estructura es ligeramente diferente, en lugar de enviar simplemente un `prompt` sencillo vamos a enviar
una lista de diccionarios en `messages`:

```python
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": "Eres un asistente que da informacion sobre deportes"},
        {"role": "user", "content": "¿Quién ganó el mundial de fútbol?"},
        {"role": "assistant", "content": "El mundial de 2022 lo ganó Argentina"},
        {"role": "user", "content": "¿Dónde se jugó?"}
    ],
    temperature=1,  # Este es el valor default
    max_tokens=60

)
print(response['choices'][0]['message']['content'])
print("*"*64)
```
Respuesta esperada:
```commandline
El Mundial de fútbol de 2022 se jugó en Qatar.
```

Podemos observar como para utilizar `ChatCompletion` debemos empezar dandole al modelo un `contexto` general al modelo. 

- Eres un asistente ...

Después, podemos darle un ejemplo de la posible entrada que tendría un usuario:

- ¿Quién gano el mundial de futbol?

Y finalmente, podemos darle un ejemplo de la respuesta que va a dar el modelo siendo él el `assistant`:

- El mundial de 2022 lo ganó Argentina.

Adicionalmente, podemos leer la documentación del API [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
para leer los demás parámetros del modelo. Aunque muchos son los mismos que ya hemos visto anteriormente.
Como resumen presentamos la siguiente diapositiva.

![12.png](ims%2F1%2F12.png)

Y finalmente, si te preguntas cuando debo usar Completion vs ChatCompletion
![11.png](ims%2F1%2F11.png)

Podemos ver que en general Completion sirve con todos los modelos, pero solamente está esperando generar
una sola `completion` no tener un diálogo profundo o conversación de multiples turno.

Adicionalmente, utilizar `ChatCompletion` nos permite dar un formato de Contexto, Entrada, Respuesta esperada, lo cuál es 
muy útil para un sin fin de aplicaciones. 


## 1.8 Actualizaciones de la API de OpenAI: GPT-4 disponible y modelos deprecados

OpenAI y su API es un producto que está en constante actualización por la rápida innovación que actualmente tienen las tecnologías de IA.

⚠️Recuerda estar al pendiente de este curso porque seguirá actualizándose de acuerdo a los cambios en OpenAI. Cada vez que haya una actualización importante podrás enterarte en tu correo electrónico y en un agregado en esta clase de lectura.

### Disponibilidad general de la API de GPT-4

GPT-4 es el modelo más capaz de OpenAI. En la última actualización de la API, todo mundo puede acceder libremente a la API de GPT-4 con un contexto de 8K. Ya no es necesario estar dentro de alguna beta.

Se planea que este modelo de Chat esté disponible en todas sus versiones completando el mes de julio de 2023 según la disponibilidad de recursos informáticos. Pero es muy posible que ya puedas utilizarlo en el modo ChatCompletion.

📣 Se está trabajando en habilitar de manera segura fine-tuning para GPT-4 y GPT-3.5 Turbo. Se espera que esta función esté disponible más adelante este año.

Fuente: [GPT-4 API general availability and deprecation of older models in the Completions API (openai.com)](https://openai.com/blog/gpt-4-api-general-availability)

### Pasando de text completions a chat completions (modelos deprecados)

La API de Completions fue introducida en junio de 2020 para proporcionar una solicitud de texto de forma libre para interactuar con nuestros modelos de lenguaje. Desde entonces, se ha aprendido que podemos obtener mejores resultados con una interfaz de solicitud más estructurada.

El paradigma basado en chat ha demostrado ser poderoso, manejando la gran mayoría de los casos de uso anteriores y las nuevas necesidades de conversación, al tiempo que brinda una mayor flexibilidad y especificidad. En particular, la interfaz estructurada de la API de Chat Completions y las capacidades de conversación de múltiples turnos permiten crear experiencias conversacionales y una amplia gama de tareas de completado.

OpenAI planea seguir invirtiendo la mayor parte de esfuerzos en Chat Completions, ya que ofrecerá una experiencia cada vez más capaz y fácil de usar. Es por ello que para enero de 2024 retirarán algunos modelos anteriores de Text Completions. **Si bien esta API seguirá siendo accesible, a partir de esta actualización está etiquetada como “legacy” en la plataforma.** No hay planes de lanzar nuevos modelos utilizando la API de Text Completions.

**A partir del 4 de enero de 2024, los modelos antiguos de text completions ya no estarán disponibles y serán reemplazados por los siguientes modelos:**

![13.png](ims%2F1%2F13.png)

⚠️Estos nuevos modelos estarán disponibles en las próximas semanas para pruebas tempranas al especificar los siguientes nombres de modelos en las llamadas a la API:

- ada-002
- babbage-002
- curie-002
- davinci-002
- gpt-3.5-turbo-instruct

- Te sugerimos usar estos nuevos modelos para aplicarles fine-tuning en las siguientes clases de este curso, si es que ya los tienes disponibles en tu cuenta de OpenAI.

Fuente: [GPT-4 API general availability and deprecation of older models in the Completions API (openai.com)](https://openai.com/blog/gpt-4-api-general-availability)


## Quiz de OpenAI API

![14.png](ims%2F1%2F14.png)

![15.png](ims%2F1%2F15.png)

![16.png](ims%2F1%2F16.png)

![17.png](ims%2F1%2F17.png)

![18.png](ims%2F1%2F18.png)


# 2 Fine-tuning de modelos de OpenAI

## 2.1 ¿Por qué hacer fine-tuning a modelos de OpenAI?

Cuando hablamos de fine-tunining nos referimos a refinamiento de un modelo pre-entrenado para poder cumplir con tareas 
más específicas para las que originalmente no fue necesariamente entrenado.

![1.png](ims%2F2%2F1.png)

Algunos de los beneficios que podemos mencionar sobre hacer fine-tuning a modelos de DL son:

**1. Adaptación a tareas específicas:** El fine-tuning permite ajustar el modelo preentrenado para tareas específicas o dominios particulares. Esto es especialmente útil cuando se necesita que el modelo realice una tarea específica, como la generación de respuestas en un servicio de atención al cliente, la traducción de texto en un idioma particular o la redacción de contenido especializado.

**2. Mejora del rendimiento:** Al ajustar el modelo a una tarea específica, el fine-tuning puede mejorar significativamente su rendimiento y precisión en esa tarea en comparación con usar el modelo preentrenado directamente.

**3. Reducción del tiempo de entrenamiento:** El preentrenamiento inicial de modelos de lenguaje como ChatGPT es una tarea intensiva en recursos y tiempo. Sin embargo, una vez que el modelo está preentrenado, el fine-tuning es un proceso más rápido y eficiente en comparación con el entrenamiento completo desde cero.

**4. Control y personalización:** Fine-tuning permite a los desarrolladores tener un mayor control sobre el modelo, lo que les permite ajustar y personalizar su comportamiento según las necesidades específicas de su aplicación.

**5. Adquisición de conocimiento específico:** Al entrenar el modelo en datos específicos de una tarea, el modelo puede adquirir conocimiento relevante y especializado para esa tarea, lo que lo hace más eficaz en su desempeño.

**6. Adaptación a cambios en datos o requisitos:** Si los datos o las necesidades de una tarea cambian con el tiempo, el modelo puede ser sometido a un nuevo proceso de fine-tuning para adaptarse a esos cambios sin tener que volver a preentrenar desde cero.

Algunos problemas más populares por los cuales decidimos hacer fine-tuning son:

- **Clasificación**:
  - ¿El Modelo está haciendo declaraciones falsas?
  - Análisis de sentimientos
  - Categorización de correo electrónico(clasificación de spam).
- **Generación condicional**(Crear conocimiento a partir de otro ya creado):
  - Ejemplo: tomar un texto de wiki y crear uno nuevo a partir de este.
  - Extracción de entidades(ver contexto del texto).
  - Chatbot de atención al cliente.
  - Descripción basada en una lista técnica de propiedades.(Dar descripciones de productos basadas en requermientos)


## 2.2 Costos de uso de OpenAI: tokenización de texto

Uno de los procesos principales que utiliza el API de OpenAI para procesar el texto, es la tokenización del mismo. En el contexto de ChatGPT, los tokens son las unidades más pequeñas en las que el texto se divide para que el modelo pueda procesarlo. Un token puede ser tan corto como un carácter o tan largo como una palabra completa, y el proceso de dividir el texto en estos tokens se conoce como tokenización.
Sin embargo, la relación Palabra a Token NO es exactamente 1:1 de hecho, una regla de dedo es que aproximadamente 100 tokens representan 75 palabras en el idioma inglés, o 
también podemos verlo como que la cantidad de palabras multiplicada por 1.33 es aproximadamente igual al número de tokens que se van a utilizar.

Podemos utilizar [Tokenizer](https://platform.openai.com/tokenizer) para darnos una idea de como OpenAI procesa la cantidad de tokens por oración.

![2.png](ims%2F2%2F2.png)

Esta información es indispensable tenerla en cuenta, porque cada modelo tiene un precio diferente tanto por procesar la información de entrada
como por generar la respuesta de salida.

![3.png](ims%2F2%2F3.png)

Esta información es bastante interesante cuando vamos a la sección de **Fine-tuning models:**

![4.png](ims%2F2%2F4.png)

- Ada-(La más rápida)
$0.0004 -> $0.0016 / 1K tokens 
Es el más rápido de los modelos enumerados, es una opción rentable para aplicaciones donde la velocidad es un factor crítico, como en aplicaciones de servicio al cliente o chatbot.

- Babbage
$0.0006 -> $0.0024 / 1K tokens
Es un poco más lento que Ada, pero aun así ofrece una opción rápida y eficiente para tareas de procesamiento de lenguaje natural.

- Curie
$0.0030 -> $0.0120 / 1K tokens
Es más caro que Ada y Babbage. Sin embargo, ofrece capacidades más avanzadas que los modelos más rápidos, lo que lo convierte en una buena opción para aplicaciones que requieren un procesamiento más complejo.

- Davinci (el más poderoso)
$0.0300 -> $0.120 / 1K tokens 
El modelo más poderoso de la lista es Davinci, que ofrece las capacidades más avanzadas para tareas de procesamiento de lenguaje natural. Sin embargo, es la opción más cara de la lista. Es ideal para aplicaciones donde la precisión y las respuestas matizadas son fundamentales, como en escenarios complejos de atención al cliente o proyectos de investigación.

## 2.3 Configuración de entorno local de OpenAI con Anaconda

Te guiaré a través del proceso de configuración de un entorno local para usar la API de OpenAI con Python. Empecemos con los requisitos previos.

### ✅ Requisitos previos
Antes de comenzar, asegúrate de tener los siguientes requisitos instalados en tu computadora:

1. Python versión 3.9 o superior: La API de OpenAI se integra estrechamente con Python, por lo que necesitarás tenerlo instalado en tu sistema operativo de preferencia. Puedes descargar la última versión desde su sitio web oficial.

2. VSCode: En este editor de código crearemos los scripts de Python para nuestra aplicación. Puedes descargar la última versión en el sitio oficial de VSCode.

3. Extensión de Python de VSCode: Para facilitar el uso de Python en el editor, instala la extensión desde su sitio oficial en el marketplace .

4. Anaconda: Es una herramienta popular para crear entornos de desarrollo para ciencia de datos y machine learning con Python. Asegúrate de tener la última versión de Anaconda instalada en tu sistema. Puedes descargarla desde el sitio web oficial de Anacondal.

### 🐍 Creación de entorno virtual con Anaconda
Es una buena práctica utilizar entornos virtuales para aislar tus proyectos y dependencias. Sigue estos pasos para crear un entorno virtual con Anaconda:

1. Abre una terminal en tu computadora.

2. Ejecuta el siguiente comando para crear un nuevo entorno virtual llamado “NAME” (puedes reemplazar “NAME” con el nombre que desees, por ejemplo “curso_openai”):

```commandline
conda create -n NAME python==3.9
```

3. Una vez que se haya creado el entorno virtual, actívalo con el siguiente comando:

```commandline
conda activate NAME
```

4. Ahora estás trabajando dentro del entorno virtual “NAME” y puedes instalar las bibliotecas necesarias sin afectar tu instalación principal de Python. Instala las librerías necesarias con el siguiente comando*:

```commandline
conda install numpy pandas openai requests
```

> *Para instalar las librerías en el entorno “NAME” deberá estar activado con el paso anterior.

5. Cuando hayas terminado de trabajar en tu proyecto, puedes desactivar el entorno virtual con el siguiente comando:

```commandline
conda deactivate
```

Si quieres aprender a detalle a usar Anaconda y Jupyter Notebooks, puedes tomar el Curso de Entorno de Trabajo para Ciencia de Datos con Jupyter Notebooks y Anaconda 🐍

### 🔐 Creación de API Key
Antes de utilizar la API de OpenAI necesitarás una clave de API para acceder a los modelos. Sigue estos pasos para crear tu propia clave:

1. Visita la página de OpenAI para gestionar tus claves de API: https://platform.openai.com/account/api-keys.

2. Si aún no tienes una cuenta de OpenAI, regístrate y crea una nueva cuenta.

3. Una vez que hayas iniciado sesión en tu cuenta de OpenAI, ve a la sección de API Keys y crea una nueva clave.

4. Copia tu API Key e impórtala como variable de entorno de tu sistema operativo. Guárdala como “OPENAI_API_KEY” de forma permanente. Esto ayudará a garantizar la seguridad de tu API Key al utilizarla de esta manera en tu código. Puedes ver esta clase para configurarla en Ubuntu o macOS si no conoces el proceso.

5. Recuerda que para comenzar a utilizar la API, necesitarás tener al menos $5 dólares de crédito en tu tarjeta bancaria para utilizar los modelos y funcionalidades de la API.

### ⚠️💵 Costo del fine-tuning del modelo Davinci y alternativas
Es importante tener en cuenta que el fine-tuning del modelo Davinci de OpenAI puede tener un costo significativo. La tarifa de uso de Davinci se basa en el número de tokens de entrada y salida utilizados durante el proceso de fine-tuning. Te recomiendo revisar la documentación oficial de OpenAI para obtener información actualizada sobre los precios y las políticas de uso en el siguiente enlace: https://openai.com/pricing

Para el proyecto de este curso el costo estimado por el fine-tuning ronda los $45 dólares, utilizando un dataset con 1800 registros y el modelo Davinci.

Si deseas reducir el costo del fine-tuning, considera las siguientes alternativas:

1.  Reducir el tamaño del dataset: Limitar la cantidad de registros utilizados en el proceso de fine-tuning puede ayudar a reducir los costos. Evalúa la posibilidad de seleccionar una muestra representativa de tus datos en lugar de utilizar todo el dataset.

2. Explorar modelos más accesibles: Además del modelo Davinci, OpenAI ofrece modelos como Ada o Babbage, que son más accesibles en términos de costo. Puedes explorar la opción de utilizar modelos alternativos según tus necesidades y presupuesto.

## 2.4 Formato de datos para fine-tuning



## 2.5 Preparar datos para fine-tuning

## 2.6 Fine-tuning de modelo de OpenAI

## 2.7 ¿Cómo usar PlayGround de OpenAI para probar modelos?

## 2.8 Pruebas al modelo con fine-tuning

## 2.9 Optimizar el modelo: ajuste de parámetros en Playground

## 2.10 Validación de modelos fine-tuned de OpenAI

## Quiz de fine-tuning de modelos de OpenAI

# 3 Integración de modelo a aplicación de chat

## 3.1 ¿Cómo crear un chatbot con Telegram?

## 3.2 Procesando la entrada del usuario para el chatbot

## 3.3 Prueba de envío de mensajes del chatbot

## 3.4 Función main() del chatbot

## 3.5 Integración del modelo de OpenAI a Telegram

## 3.6 Manejo de errores y excepciones de la API de OpenAI

## Quiz de integración de LLM a chat

# 4 Conclusión

## 4.1 Recomendaciones finales y proyectos alternativos con el API de OpenAI