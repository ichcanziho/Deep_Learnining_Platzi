# Curso de Desarrollo de Aplicaciones de IA con LangChain: Chatbots

En este curso dominarás LangChain, el framework para el manejo de LLM. Construye aplicaciones fascinantes que integren inteligencia artificial, como chatbots con control de memoria y consulta de datos específicos de una organización.

- Carga documentos para consultarlos a través de un LLM.
- Integra memoria y cadenas para desarrollar aplicaciones de IA robustas.
- Integra fácilmente LLM con LangChain usando prompts y cadenas.


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
> - [12: Curso de Desarrollo de Chatbots con OpenAI](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/12%20Desarrollo%20ChatBot)
> 
> Este Curso es el Número 13 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso


# ÍNDICE:

# Índice

- [1 Introducción a LangChain](#1-introducción-a-langchain)
  - [1.1 Desarrollo de aplicaciones con LLM utilizando LangChain](#11-desarrollo-de-aplicaciones-con-llm-utilizando-langchain)
  - [1.2 Estructura y módulos de LangChain](#12-estructura-y-módulos-de-langchain)
  - [1.3 Uso de modelos Open Source de Hugging Face](#13-uso-de-modelos-open-source-de-hugging-face)
  - [1.4 Uso de modelos de OpenAI API](#14-uso-de-modelos-de-openai-api)
  - [1.5 Prompt templates de LangChain](#15-prompt-templates-de-langchain)
  - [1.6 Cadenas en LangChain](#16-cadenas-en-langchain)
  - [1.7 Utility Chains](#17-utility-chains)
  - [1.8 RetrievalQA Chain](#18-retrievalqa-chain)
  - [1.9 Foundational Chains](#19-foundational-chains)
  - [1.10 Quizz Introducción a LangChain](#110-quizz-introducción-a-langchain)

- [2 Casos de uso de LangChain](#2-casos-de-uso-de-langchain)
  - [2.1 Casos de uso de LangChain](#21-casos-de-uso-de-langchain)
  - [2.2 ¿Cómo utilizar LangChain en mi equipo?](#22-cómo-utilizar-langchain-en-mi-equipo)

- [3 Manejo de documentos con índices](#3-manejo-de-documentos-con-índices)
  - [3.1 ¿Cómo manejar documentos con índices en LangChain?](#31-cómo-manejar-documentos-con-índices-en-langchain)
  - [3.2 La clase Document](#32-la-clase-document)
  - [3.3 Document Loaders: PDF](#33-document-loaders-pdf)
  - [3.4 Document Loaders: CSV con Pandas DataFrames](#34-document-loaders-csv-con-pandas-dataframes)
  - [3.5 Document Loaders: JSONL](#35-document-loaders-jsonl)
  - [3.6 Document Transformers: TextSplitters](#36-document-transformers-textsplitters)
  - [3.7 Proyecto de ChatBot: configuración de entorno para LangChain y obtención de datos](#37-proyecto-de-chatbot-configuración-de-entorno-para-langchain-y-obtención-de-datos)
  - [3.8 Proyecto de Chatbot: creación de documentos de Hugging Face](#38-proyecto-de-chatbot-creación-de-documentos-de-hugging-face)
  - [3.9 Quiz manejo de documentación con índices](#39-quiz-manejo-de-documentación-con-índices)

- [4 Embeddings y bases de datos vectoriales](#4-embeddings-y-bases-de-datos-vectoriales)
  - [4.1 Uso de embeddings y bases de datos vectoriales con LangChain](#41-uso-de-embeddings-y-bases-de-datos-vectoriales-con-langchain)
  - [4.2 ¿Cómo usar embeddings de OpenAI en LangChain?](#42-cómo-usar-embeddings-de-openai-en-langchain)
  - [4.3 ¿Cómo usar embeddings de Hugging Face en LangChain?](#43-cómo-usar-embeddings-de-hugging-face-en-langchain)
  - [4.4 Chroma vector store en LangChain](#44-chroma-vector-store-en-langchain)
  - [4.5 Proyecto de ChatBot: ingesta de documentos en Chroma](#45-proyecto-de-chatbot-ingesta-de-documentos-en-chroma)
  - [4.6 RetrievalQA: cadena para preguntar](#46-retrievalqa-cadena-para-preguntar)
  - [4.7 Proyecto de ChatBot: RetrievalQA chain](#47-proyecto-de-chatbot-retrievalqa-chain)
  - [4.8 Quiz de embeddings y bases de datos vectoriales](#48-quiz-de-embeddings-y-bases-de-datos-vectoriales)

- [5 Chats y memoria con LangChain](#5-chats-y-memoria-con-langchain)
  - [5.1 ¿Para qué sirve la memoria en cadenas y chats?](#51-para-qué-sirve-la-memoria-en-cadenas-y-chats)
  - [5.2 Uso de modelos de chat con LangChain](#52-uso-de-modelos-de-chat-con-langchain)
  - [5.3 Chat prompt templates](#53-chat-prompt-templates)
  - [5.4 ConversationBufferMemory](#54-conversationbuffermemory)
  - [5.5 ConversationBufferWindowMemory](#55-conversationbufferwindowmemory)
  - [5.6 ConversationSummaryMemory](#56-conversationsummarymemory)
  - [5.7 ConversationSummaryBufferMemory](#57-conversationsummarybuffermemory)
  - [5.8 Entity Memory](#58-entity-memory)
  - [5.9 Proyecto de ChatBot: chat history con ConversationalRetrievalChain](#59-proyecto-de-chatbot-chat-history-con-conversationalretrievalchain)
  - [5.10 Quiz de chats y memoria con LangChain](#510-quiz-de-chats-y-memoria-con-langchain)

- [6 Conclusiones](#6-conclusiones)
  - [6.1 LangChain y LLM en evolución constante](#61-langchain-y-llm-en-evolución-constante)


# 1 Introducción a LangChain

## 1.1 Desarrollo de aplicaciones con LLM utilizando LangChain

Hoy en día tenemos accedo a cada vez más Modelos de Lenguaje Grandes (LLMs). Sin embargo, no todo es miel sobre hojuelas.
Tenemos algunas preguntas a resolver. 

![1.png](ims%2F1%2F1.png)

- **Primero**: ¿Cómo hacemos llegar la información que requieren estos modelos de lenguaje para que nos resuelvan nuestras preguntas?
Es muy probable que esta información no se encuentre con los datos con los que se entrenó el modelo. 

- **Segundo**: ¿Cómo hacemos para crear un proceso en el que pasemos de Base de Datos hasta la respuesta del modelo de lenguaje?

Para esto **LangChain** propone una solución, crear cadenas que solucionan el proceso y retroalimentar con información al modelo 
de lenguaje para resolver las preguntas.

![2.png](ims%2F1%2F2.png)

Para resolver este problema, utilizamos diferentes Modelos de lenguaje como aquellos a los que podemos acceder en Hugging Face,
OpenAi, Cohere etc. Esto lo podemos integrar con bases de datos vectoriales, y más herramientas para crear `Indexes`.

Estos `Indexes` tiene como finalidad, encontrar los documentos que más tienen probabilidad de solucionar una pregunta que pudo haber hecho el cliente.

> Todo esto nos permite crear un flujo que nos permita responder preguntas sobre información personalizada.

![3.png](ims%2F1%2F3.png)

Un flujo muy interesante que podemos construir es el siguiente:

- **Modelo 1:** Transforma el texto en un embedding (representación vectorial) y busca este vector en una base de datos vectorial 
- **Modelo 2:** Resume la pregunta, extrayendo la información más relevante.
- **Modelo 3:** Busca la respuesta que mejor se adapte a la pregunta.

Vamos a empezar con un ejemplo muy práctico de como utilizar LangChain para responder preguntas a conocimiento nuevo que no éxiste en
la base de entrenamiento de ChatGPT.

> ## CASO DE USO:
> 
> Tenemos acceso a varios papers científicos de reciente publicación, y queremos analizarlos y poder formular preguntas que pueda responder ChatGPT por nosotros.
>
> ### Retos:
>
> - Los papers son tan recientes que el modelo ChatGPT no los tiene en su base de datos
> - El texto está contenido en documentos PDF
> 

Antes de empezar es recomendable que hagas un ambiente virtual, lo actives y sigas con los siguientes pasos de instalación de requisitos:

```bash
pip install langchain
pip install pypdf
pip install openai
pip install chromadb
pip install tiktoken
```

- **langchain**: creador de cadenas con conexiones lógicas entre uno o más LLMs
- **pypdf**: carga y extracción de texto de documentos pdf
- **openai**: acceso a la API de OpenAI para utilizar sus modelos de lenguaje
- **chromadb**: base de datos vectorial
- **tiktoken**: tokenizador de texto creado por openai 

Vamos a empezar partiendo de que tenemos los siguientes links a diferentes papers de actualidad:

```commandline
https://arxiv.org/pdf/2306.06031v1.pdf
https://arxiv.org/pdf/2306.12156v1.pdf
https://arxiv.org/pdf/2306.14289v1.pdf
https://arxiv.org/pdf/2305.10973v1.pdf
https://arxiv.org/pdf/2306.13643v1.pdf
```

![4.png](ims%2F1%2F4.png)

Cada uno de ellos contiene información sobre diferentes modelos de LLMs aplicado a diferentes sectores. Y tienen publicaciones muy recientes de 2023, por ende
ChatGPT no les conoce.

> ## Nota:
> El código lo puedes encontrar en: [1_hola_langchain.py](scripts%2F1_hola_langchain.py)

Para facilitar la lectura del código, la importación de bibliotecas será segmentada para explicar cada sección del mismo:

Vamos a empezar con la descarga de los documentos PDF, y su carga utilizando `PyPDFLoader`

```python
# --- Carga de documents
import os
import requests
from langchain.document_loaders import PyPDFLoader

urls = [
    'https://arxiv.org/pdf/2306.06031v1.pdf',
    'https://arxiv.org/pdf/2306.12156v1.pdf',
    'https://arxiv.org/pdf/2306.14289v1.pdf',
    'https://arxiv.org/pdf/2305.10973v1.pdf',
    'https://arxiv.org/pdf/2306.13643v1.pdf'
]
```
Ahora que ya tenemos importado `requests` y `PyPDFLoader` vamos a descargar cada uno de ellos y unirlos a una lista de documentos
llamada `ml_papers`

```python
ml_papers = []

for i, url in enumerate(urls):
    filename = f'paper{i+1}.pdf'

    # Verifico si el archivo no ha sido descargado previamente
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f'Descargado {filename}')
    else:
        print(f'{filename} ya existe, cargando desde el disco.')

    loader = PyPDFLoader(filename)
    data = loader.load()
    ml_papers.extend(data)

# Utiliza la lista ml_papers para acceder a los elementos de todos los documentos descargados
print('Contenido de ml_papers:')
print()

print(type(ml_papers), len(ml_papers), ml_papers[3])
```
Respuesta esperada:
```commandline
Descargado paper1.pdf
Descargado paper2.pdf
Descargado paper3.pdf
Descargado paper4.pdf
Descargado paper5.pdf
Contenido de ml_papers:

<class 'list'> 57 page_content='Figure 1: FinGPT Framework.\n4.1 Data Sources\nThe first stage of the FinGPT pipeline 
involves the collec-\ntion of extensive financial data from a wide array of online\nsources. These include, but are not 
limited to:\n•Financial news: Websites such as Reuters, CNBC, Yahoo\nFinance, among others, are rich sources of financial 
news\nand market updates. These sites provide valuable informa-\ntion on market trends, company earnings, macroeconomic
\nindicators, and other financial events.\n•Social media : Platforms such as Twitter, Facebook, Red-\ndit, Weibo, and 
others, offer a wealth of information in\nterms of public sentiment, trending topics, and immediate\nreactions to financial 
news and events.\n•Filings : Websites of financial regulatory authorities, such\nas the SEC in the United States, offer 
access to company\nfilings. These filings include annual reports, quarterly earn-\nings, insider trading reports, and other 
important company-\nspecific information. Official websites of stock exchanges\n(NYSE, NASDAQ, Shanghai Stock Exchange, etc.) 
pro-\nvide crucial data on stock prices, trading volumes, company\nlistings, historical data, and other related information.
\n•Trends : Websites like Seeking Alpha, Google Trends, and\nother finance-focused blogs and forums provide access to\nanalysts’ 
opinions, market predictions, the movement of\nspecific securities or market segments and investment ad-\nvice.•Academic 
datasets : Research-based datasets that offer cu-\nrated and verified information for sophisticated financial\nanalysis.
\nTo harness the wealth of information from these diverse\nsources, FinGPT incorporates data acquisition tools capable\nof 
scraping structured and unstructured data, including APIs,\nweb scraping tools, and direct database access where avail-
\nable. Moreover, the system is designed to respect the terms\nof service of these platforms, ensuring data collection is 
ethi-\ncal and legal.\nData APIs : In the FinGPT framework, APIs are used not\nonly for initial data collection but also 
for real-time data up-\ndates, ensuring the model is trained on the most current data.\nAdditionally, error handling and 
rate-limiting strategies are\nimplemented to respect API usage limits and avoid disrup-\ntions in the data flow.\n4.2 
Real-Time Data Engineering Pipeline for\nFinancial NLP\nFinancial markets operate in real-time and are highly sensi-\ntive 
to news and sentiment. Prices of securities can change\nrapidly in response to new information, and delays in pro-\ncessing 
that information can result in missed opportunities or\nincreased risk. As a result, real-time processing is essential 
in\nfinancial NLP.\nThe primary challenge with a real-time NLP pipeline is\nmanaging and processing the continuous inflow 
of data ef-\nficiently. The first step in the pipeline is to set up a system to' 

metadata={'source': 'paper1.pdf', 'page': 3}
```

Lo primero que podemos notar es que nuestra lista contiene información de 57 páginas extraídas de los 5 documentos PDF descargados.
Sin embargo, cuando abrimos el contenido de una sola página, nos percatamos que el contenido de texto es muy amplio, y no sería 
manejable fácilmente por `embeddings` entonces, tenemos que particionar estos datos en `chuncks` o grupos más pequeños que
sean manejables por los `embeddings`:

```python
# --- Split de documents

# Los documentos NO pueden ser procesados directamente por LLMs porque contienen demasiado texto, sin embargo, podemos
# particionarlo en conjuntos de texto más pequeños para entonces poder acceder a su información.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cada partición contendrá 1500 palabras, y tendrán una intersección de 200, de modo que la cadena 2 comparte 200
# palabras con la cadena 1 y con la cadena 3
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
    )

documents = text_splitter.split_documents(ml_papers)
# Ahora podemos revisar de nuevo la cantidad de `documentos` y ver un ejemplo del mismo
print(len(documents), documents[10])
```
Respuesta esperada:
```commandline
211 

page_content='highly volatile, changing rapidly in response to news events\nor market movements.\nTrends , often observable 
through websites like Seeking\nAlpha, Google Trends, and other finance-oriented blogs and\nforums, offer critical insights 
into market movements and in-\nvestment strategies. They feature:\n•Analyst perspectives: These platforms provide access 
to\nmarket predictions and investment advice from seasoned\nfinancial analysts and experts.\n•Market sentiment: The discourse 
on these platforms can\nreflect the collective sentiment about specific securities,\nsectors, or the overall market, 
providing valuable insights\ninto the prevailing market mood.\n•Broad coverage: Trends data spans diverse securities 
and\nmarket segments, offering comprehensive market coverage.\nEach of these data sources provides unique insights into\nthe 
financial world. By integrating these diverse data types,\nfinancial language models like FinGPT can facilitate a com-\nprehensive 
understanding of financial markets and enable ef-\nfective financial decision-making.\n3.2 Challenges in Handling Financial 
Data\nWe summarize three major challenges for handling financial\ndata as follows:\n•High temporal sensitivity : Financial 
data are character-\nized by their time-sensitive nature. Market-moving news or\nupdates, once released, provide a narrow 
window of oppor-\ntunity for investors to maximize their alpha (the measure of\nan investment’s relative return).•High 
dynamism : The financial landscape is perpetually' 

metadata={'source': 'paper1.pdf', 'page': 2}
```
Genial, ahora en lugar de tener **57 páginas tenemos 217 chunks** que es justo lo que queríamos, particiones de datos más
pequeñas y manejables. Ahora vamos a convertir estos `chunks` en su reprentación vectorial utilizando un modelo pre-entrenado
de OpenAI en este caso vamos a utilizar `ada-002`

```python
# --- Embeddings e ingesta a base de datos vectorial

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv


# leo el archivo keys.env y obtengo mi Api KEY de OpenAI
load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Es importante que quede seteado como una variable de entorno porque será utilizado más adelante
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Crea un objeto capaz de convertir el texto a un vector utilizando como base el modelo de ADA-002 de OpenAI
# En este punto es importante que hayas seteado tu OPENAI API KEY como variable de entorno, para que puedas acceder
# a este servicio
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Con ayuda de Chroma, creamos un objeto vectorstore para almacenar las representaciones vectoriales de los textos
# contenidos en `documents` una cadena de texto previamente generada

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)
```
Perfecto, hemos creado éxitosamente nuestro `vectorstore` que contiene los `chunks` vectorizados, ahora vamos a crear a nuestro
objeto `retriever` que nos permitirá obtener los mejores `candidatos` basados en la similitud entre sus `vectores`:

```python
# Una vez que hayas creado la Base de datos vectorial, el parámetro search_kwargs `k` me permite definir hasta cuantos
# vectores similares voy a buscar al momento de encontrar información para una pregunta. `retriever` será entonces
# nuestra base de datos de vectores que servirá para añadir información reciente a los LLMs con el fin de responder
# preguntas.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
    )
```
Ya estamos cerca de terminar nuestro `pipeline` de `LangChain` vamos a continuar instanciando un modelo LLM en este caso
vamos a usar como `chat_models` el `ChatOpenAI`. Especifica mente vamos a utilizar como `engine` a `gpt-3.5-turbo`:

```python
# --- Modelos de Chat y cadenas para consulta de información

from langchain.chat_models import ChatOpenAI

# Voy a crear un objeto `chat` de la clase ChatOpenAI indicando que el engine a utilizar será GPT 3.5 y cuya temperatura
# será 0 lo que signfica que tendrá respuestas muy restrictivas basadas únicamente en el texto que conoce y tendrá
# poca creatividad al momento de responder peticiones.
chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
```

Finalmente, vamos a enlazar a nuestro modelo LLM en este caso `chat gpt` a nuestro objeto `retriever` y especialmente lo vamos
a enlazar a través del sistema `RetrievalQA` que nos permitirá hacer preguntas y recibir respuestas de un texto:

```python
from langchain.chains import RetrievalQA
# Finalmente, creamos una cadena `chain` del tipo `Question Answer` pregunta-respuesta. Como LLM utilizará al objeto
# `chat` que es una instancia de ChatGPT 3.5, el tipo de cadena es `stuff` que significa que vamos a utilizar tanta
# información como quepa en el prompt, y finalmente el `retriever` será la base de datos vectoriales que hemos definido
# previamente.
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)
```
Por último, pongamos todo en marcha y hagamos un par de preguntas a nuestro modelo y conozcamos sus respuestas:

```python
# Vamos a poner a prueba nuestra cadena de preguntas y respuestas:

query = "qué es fingpt?"
print("--->", query)
print(qa_chain.run(query))

query = "qué hace complicado entrenar un modelo como el fingpt?"
print("--->", query)
print(qa_chain.run(query))

query = "qué es fast segment?"
print("--->", query)
print(qa_chain.run(query))

query = "cuál es la diferencia entre fast sam y mobile sam?"
print("--->", query)
print(qa_chain.run(query))

```
Respuesta esperada:
```commandline
---> qué es fingpt?

FinGPT es un modelo de lenguaje financiero de código abierto desarrollado por la comunidad AI4Finance. Está diseñado para 
aplicaciones en el campo de las finanzas y utiliza técnicas de procesamiento de lenguaje natural para analizar y comprender datos 
financieros. FinGPT adopta un enfoque centrado en los datos y utiliza métodos rigurosos de limpieza y preprocesamiento para 
garantizar la calidad de los datos. También ofrece un marco de trabajo de extremo a extremo con cuatro capas, que abarca desde 
la obtención de datos hasta la implementación de aplicaciones prácticas en finanzas. El objetivo de FinGPT es estimular la innovación, 
democratizar los modelos de lenguaje financiero y desbloquear nuevas oportunidades en el campo de las finanzas abiertas.

---> qué hace complicado entrenar un modelo como el fingpt?

Entrenar un modelo como FinGPT puede ser complicado por varias razones:

1. Requisitos computacionales intensivos: Modelos como BloombergGPT requieren una gran cantidad de recursos computacionales, 
como horas de GPU y costos asociados. Esto puede hacer que el entrenamiento sea costoso y lento.
2. Costo financiero: El entrenamiento de modelos como BloombergGPT puede ser extremadamente costoso, llegando a millones de 
dólares. Esto limita su accesibilidad y hace que sea menos viable para muchas organizaciones o individuos.
3. Actualizaciones y adaptabilidad: En el dominio financiero, es crucial tener modelos actualizados y adaptables debido a 
la naturaleza dinámica del mercado. Entrenar un modelo desde cero cada vez que se requiere una actualización puede ser ineficiente y llevar mucho tiempo.
4. Personalización y transparencia: Los modelos de lenguaje financiero deben ser personalizables para adaptarse a las necesidades 
específicas de los usuarios. Entrenar modelos desde cero no permite esta personalización, mientras que el ajuste fino de 
modelos preexistentes ofrece una solución más flexible y personalizable.

En resumen, entrenar modelos como FinGPT desde cero puede ser costoso, computacionalmente intensivo y menos adaptable a las 
necesidades individuales. Por lo tanto, el ajuste fino de modelos preexistentes se presenta como una alternativa más eficiente y accesible.

---> qué es fast segment?

Fast Segment es un método alternativo propuesto en el artículo "Fast Segment Anything" para acelerar el modelo Segment 
Anything (SAM) en tareas de visión por computadora. SAM es un modelo que puede segmentar cualquier objeto en una imagen y 
ha demostrado resultados prometedores en diversas tareas. Sin embargo, SAM tiene altos costos computacionales debido a su 
arquitectura Transformer en entradas de alta resolución. Fast Segment propone una forma más rápida de lograr resultados 
comparables al entrenar un método existente de segmentación de instancias utilizando solo una fracción del conjunto de datos 
utilizado por SAM. Con Fast Segment, se logra una velocidad de ejecución 50 veces más rápida que SAM sin comprometer significativamente el rendimiento.

---> cuál es la diferencia entre fast sam y mobile sam?

La diferencia entre FastSAM y MobileSAM se puede resumir en dos aspectos principales: tamaño y velocidad.
En cuanto al tamaño, MobileSAM es significativamente más pequeño que FastSAM. MobileSAM tiene menos de 10 millones de parámetros, 
mientras que FastSAM tiene 68 millones de parámetros.
En cuanto a la velocidad, MobileSAM es más rápido que FastSAM. En una GPU única, MobileSAM tarda solo 10 ms en procesar una imagen, 
mientras que FastSAM tarda 40 ms. Esto significa que MobileSAM es 4 veces más rápido que FastSAM.
Además de estas diferencias, también se menciona que MobileSAM tiene un rendimiento superior en términos de mIoU (mean Intersection over Union) 
en comparación con FastSAM. Esto sugiere que la predicción de máscaras de MobileSAM es más similar a la del SAM original que la de FastSAM.
En resumen, MobileSAM es más pequeño, más rápido y tiene un rendimiento mejorado en comparación con FastSAM.
```


## 1.2 Estructura y módulos de LangChain

## 1.3 Uso de modelos Open Source de Hugging Face

## 1.4 Uso de modelos de OpenAI API

## 1.5 Prompt templates de LangChain

## 1.6 Cadenas en LangChain

## 1.7 Utility Chains

## 1.8 RetrievalQA Chain

## 1.9 Foundational Chains

## 1.10 Quizz Introducción a LangChain

# 2 Casos de uso de LangChain

## 2.1 Casos de uso de LangChain

## 2.2 ¿Cómo utilizar LangChain en mi equipo?

# 3 Manejo de documentos con índices

## 3.1 ¿Cómo manejar documentos con índices en LangChain?

## 3.2 La clase Document

## 3.3 Document Loaders: PDF

## 3.4 Document Loaders: CSV con Pandas DataFrames

## 3.5 Document Loaders: JSONL

## 3.6 Document Transformers: TextSplitters

## 3.7 Proyecto de ChatBot: configuración de entorno para LangChain y obtención de datos

## 3.8 Proyecto de Chatbot: creación de documentos de Hugging Face

## 3.9 Quiz manejo de documentación con índices

# 4 Embeddings y bases de datos vectoriales

## 4.1 Uso de embeddings y bases de datos vectoriales con LangChain

## 4.2 ¿Cómo usar embeddings de OpenAI en LangChain?

## 4.3 ¿Cómo usar embeddings de Hugging Face en LangChain?

## 4.4 Chroma vector store en LangChain

## 4.5 Proyecto de ChatBot: ingesta de documentos en Chroma

## 4.6 RetrievalQA: cadena para preguntar

## 4.7 Proyecto de ChatBot: RetrievalQA chain

## 4.8 Quiz de embeddings y bases de datos vectoriales

# 5 Chats y memoria con LangChain

## 5.1 ¿Para qué sirve la memoria en cadenas y chats?

## 5.2 Uso de modelos de chat con LangChain

## 5.3 Chat prompt templates

## 5.4 ConversationBufferMemory

## 5.5 ConversationBufferWindowMemory

## 5.6 ConversationSummaryMemory

## 5.7 ConversationSummaryBufferMemory

## 5.8 Entity Memory

## 5.9 Proyecto de ChatBot: chat history con ConversationalRetrievalChain

## 5.10 Quiz de chats y memoria con LangChain

# 6 Conclusiones

## 6.1 LangChain y LLM en evolución constante
