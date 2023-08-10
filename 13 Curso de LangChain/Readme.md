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

En esta clase vamos a abordar el tema de **¿Cómo está estructurado LangChain?**

Podemos explicar su funcionamiento a partir de 3 claves principales:

1: Conexión con modelos
2: Conexión con datos
3: Encadenamiento de procesos

![5.png](ims%2F1%2F5.png)

El flujo es bastante auto-explicativo. El mismo empieza con la recepción y transformación de datos a formato vectorial (embedding)
estos a su vez son guardados en una base de datos. Una vez con esta información, podemos crear prompts con el cuál vamos a empezar
a hacer preguntas. Las preguntas son recibidas por un LLM que genera una respuesta con base en la información almacenada en nuestra
base de datos vectorial.
 
Sin embargo, entre la recepción de datos y la transformación vectorial de los datos puede o no haber otro flujo de información parecido al siguiente:

![6.png](ims%2F1%2F6.png)

 Este flujo intermedio se parece al flujo que seguimos en nuestro ejemplo de programación. Dado que los modelos NO pueden 
 vectorizar grandes conjuntos de texto, fue neceseario, primero, framgentar el texto en `chuncks` más pequeños para despues
vectorizarlos.
 
![7.png](ims%2F1%2F7.png)

Estos flujos adicionales de información también pueden estar presentes en la llamada a los LLMs y en otras secciones de nuestro flujo principal.

Una pregunta simple de hacerse sería: ¿Y dónde puedo encontrar información sobre cómo puedo utilizar LangChain?

![8.png](ims%2F1%2F8.png)

Debido a la naturaleza del proyecto que es tan reciente y que se actualiza rápidamente, lo más práctico es estar al pendiente
de su repositorio en github: https://github.com/langchain-ai/langchain

Esto nos permite tener información de funciones que quizá ni siquiera estén documentadas todavía. Sin embargo, eso no quita
que éxista una documentación oficial que nos permite conocer los conceptos básicos de LangChain y ejemplos de implementación 
en python: https://docs.langchain.com/docs/

![9.png](ims%2F1%2F9.png)

Algo sumamente interesante, es que cuenta con un buscador inteligente, que responde preguntas con lenguaje natural y ejemplos
basados en la documentación:

![10.png](ims%2F1%2F10.png)

Algo sumamente interesante es conocer las `integraciones` que tiene `LangChain`. Las mismas se dividen en:

- Document Loaders
- Vector Stores
- Embedding Models
- Chat Models
- LLMs
- Callbacks
- Tools
- Toolkits
- Message Histories

https://integrations.langchain.com/

![11.png](ims%2F1%2F11.png)

De esta forma podemos acceder información de como entrar un LLM en concreto como lo sería ChatGPT de OpenAI y nos brindará
información de cómo usarlo y ejemplos en código.


## 1.3 Uso de modelos Open Source de Hugging Face

> ## Nota: 
> En esta clase voy a compartir y explicar código, pero, la ejecución será dada por un NoteBook de Google Colab
> el código de ejemplo completo esta en: [2_falcon_example.py](scripts%2F2_falcon_example.py)
> 

### 1. Integración de LLMs en LangChain

En esta sección conocerás sobre los diferentes tipos de modelos que proporciona LangChain, sus ventajas y cómo utilizarlos para crear aplicaciones de IA potentes usando LLM.

**¿Qué es un modelo?**

Un modelo en LangChain es un modelo de aprendizaje automático pre-entrenado que se puede utilizar para realizar una tarea específica como generar texto, traducir idiomas o responder preguntas. Con LangChain puedes usar una variedad de modelos y utilizarlos para crear aplicaciones de IA sin tener que entrenar tus propios modelos desde cero.

**Ventajas de usar modelos de LangChain**

Hay varias ventajas de usar modelos de LangChain:

* **Consistencia:** Los modelos de LangChain proporcionan una interfaz consistente, independientemente de si está utilizando OpenAI o Hugging Face. Esto hace que sea más fácil aprender y usar los modelos de LangChain, y cambiar entre diferentes modelos si es necesario.
* **Eficiencia:** Los modelos de LangChain están pre-entrenados y alojados en la nube, lo que los hace mucho más rápidos de usar que entrenar sus propios modelos.
* **Flexibilidad:** Los modelos de LangChain se pueden utilizar para una variedad de tareas, como la comprensión del lenguaje natural, la traducción automática y el análisis de sentimientos.

**Ejemplos prácticos de uso de modelos de LangChain**

* **Generación de texto:** Utiliza un LLM para generar texto, como poemas, código, guiones, piezas musicales, correo electrónico, cartas, etc.
* **Traducción de idiomas:** Trauce un texto de un idioma a otro con modelos de lenguaje.
* **Escritura de diferentes tipos de contenido creativo:** Usa un modelo de lenguaje para escribir diferentes tipos de contenido creativo, como poemas, código, guiones, piezas musicales, correo electrónico, cartas, etc.
* **Respuesta a sus preguntas de forma informativa:** Utiliza un modelo de preguntas y respuestas para responder con información del modelo o que pueda consultar en otras fuentes, incluso si son abiertas, desafiantes o extrañas.

![12.png](ims%2F1%2F12.png)

LangChain tiene integraciones con varios modelos o plataformas de modelos, como el Hugging Face Hub. Con el tiempo, habrá disponibles más integraciones y modelos.

LangChain Models Docs: https://docs.langchain.com/docs/components/models/

### 1.1 Uso de modelos Open Source de Hugging Face

Vamos a empezar instalando un par de biblitecas necesarias:

```commandline
pip install langchain
pip install transformers
pip install einops 
pip install accelerate
```

Los modelos de Hugging Face requieren instalación de `einops`. Para utilizar `low_cpu_mem_usage=True` o `device_map` es necesario contar con `Accelerate` instalado: `pip install accelerate`.

Los modelos de Hugging Face requieren instalación de `einops`. Para utilizar `low_cpu_mem_usage=True` o `device_map` es necesario contar con `Accelerate` instalado: `pip install accelerate`.

Vamos a empezar descargando nuestro primer LLM opensource `Falcon-7b`:

```python
from transformers import AutoTokenizer, pipeline
import torch

# model = "tiiuae/falcon-40b-instruct"
# model = "stabilityai/stablelm-tuned-alpha-3b"
model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
```
Respuesta esperada:
```commandline

```
Vemos que en total hemos descargado más de 14G de información, y no la vamos a utilizar todo el tiempo, entonces es buena idea
correr todo este código en COLAB. 

Podemos enviar preguntas directamente al pipeline de Hugging Face para generar texto con nuestro modelo. Sin embargo, LangChain nos facilita la vida.

```python
print(type(pipeline))
```
Respuesta esperada:
```commandline
transformers.pipelines.text_generation.TextGenerationPipeline
```

Como ya tenemos el modelo descargado, vamos a hacer un pipeline que permita utilizar al mismo:

```python
from langchain import HuggingFacePipeline

llm_falcon = HuggingFacePipeline(
    pipeline = pipeline,
    model_kwargs = {
        'temperature': 0,
        'max_length': 200,
        'do_sample': True,
        'top_k': 10,
        'num_return_sequences':1,
        'eos_token_id': tokenizer.eos_token_id
    }
)
print(llm_falcon)
```
Respuesta esperada:
```commandline
HuggingFacePipeline(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, metadata=None, 
pipeline=<transformers.pipelines.text_generation.TextGenerationPipeline object at 0x7a7d4a5c0ca0>, model_id='gpt2', 
model_kwargs={'temperature': 0, 'max_length': 200, 'do_sample': True, 'top_k': 10, 'num_return_sequences': 1, 'eos_token_id': 11}, 
pipeline_kwargs=None)
```
Finalmente, podemos hacer uso del modelo que hemos instanciado:

```python
ans = llm_falcon("What is AI?")
print(ans)
```
Respuesta esperada:
```commandline
/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1270: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed soon, in a future version. Please use a generation configuration file (see https://huggingface.co/docs/transformers/main_classes/text_generation )
  warnings.warn(
Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1369: UserWarning: Using `max_length`'s default (20) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
\nAI stands for Artificial Intelligence. It is a branch of computer science that focuses
```
Podemos notar como específicamente `Falcon 7b` ha sido entrenado en el idioma inglés y utiliza como base un modelo `GPT2` entonces no podemos
esperar que tenga resultados tan buenos como `ChatGPT 3.5 o 4 de OpenAI`.

Los modelos de código abierto de Hugging Face son increíblemente poderosos. Sin embargo, al utilizarlos de esta manera, los descargamos y ejecutamos en nuestra propia máquina. Ahí es donde existen algunas complicaciones, ya que esto puede ser lento a menos que se cuente con el hardware adecuado.

Ahora piensa en modelos que provienen de API y servicios de OpenAI, Cohere y otros proveedores de modelos remotos (que normalmente no son de código abierto). La magia de estos modelos es que funcionan en sus servidores, no en nuestra máquina.

Es como si estuvieras invitado a una fiesta. Podrías hacer la fiesta en tu casa (como usar los modelos de Hugging Face en tu máquina), pero tendrías que hacer la limpieza antes y después, y preocuparte por la música, la comida, etc. En cambio, si la fiesta se celebra en un restaurante o salón dedicado a fiestas (como usar modelos de OpenAI o Cohere en sus servidores), solo tienes que llegar y disfrutar.

Por esto, vamos a seguir utilizando los modelos de la [API de OpenAI](https://platzi.com/cursos/openai). Todo lo que vamos a hacer a partir de ahora también se puede aplicar a los modelos descargados de Hugging Face.

## 1.4 Uso de modelos de OpenAI API

En este escenario vamos a ver un ejemplo de como utilizar los modelos de OpenAI a través de LangChain:

> ## Nota:
> El código de esta clase esta disponible en: [3_uso_modelos_openai.py](scripts%2F3_uso_modelos_openai.py)

Es necesario configurar la API Key de tu cuenta de OpenAI.

```python
import os
from dotenv import load_dotenv
from pprint import pprint

# leo el archivo keys.env y obtengo mi Api KEY de OpenAI
load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
```

Para usar al máximo los LLM con LangChain, tenemos que ajustar unas cuantas configuraciones básicas:

1. `model_name` - ¿Qué modelo vamos a usar?
    Por ejemplo, "text-davinci-003" (que es el valor por defecto) o "text-ada-001". Estos nombres cambian según quién haya creado el modelo, así que necesitas revisar la documentación de la API del proveedor que estás utilizando para encontrar el nombre específico de tu modelo.

2. `n` - La cantidad de respuestas distintas que vamos a generar para la pregunta dada (el número estándar es 1)

3. `streaming` - ¿Queremos que los resultados se transmitan poco a poco? (por defecto es "Falso").
    Esto es como decidir si preferimos escuchar una canción entera de una vez, o escucharla nota por nota. Esto es especialmente útil cuando estás armando una experiencia de chatbot y quieres que el texto aparezca línea por línea, en lugar de un solo bloque de respuesta.

4. `temperature` - Aquí ajustamos la "temperatura de muestreo" en un rango de 0 a 1.
    Imagina que la temperatura es como ajustar el nivel de improvisación de un solo de guitarra. Si la temperatura es 0, el LLM solo es "preciso" y siempre tocará las notas más probables. Siempre va a sonar igual para la misma canción. Pero si la temperatura es 1, el solo será "creativo" y va a tocar notas diferentes cada vez que lo escuches, a veces incluso notas que te sorprenden. El valor estándar es 0.7, lo que se considera lo suficientemente creativo pero no completamente aleatorio, como un solo de guitarra que sale un poco de los rieles, pero no tanto como para descarrilar la canción.

```python
from langchain.llms import OpenAI

llm_gpt3_5 = OpenAI(
    model_name="gpt-3.5-turbo",
    n=1,
    temperature=0.3
)
print(llm_gpt3_5)
print("*"*64)
```
Respuesta esperada:
```commandline
Params: {'model_name': 'gpt-3.5-turbo', 'n': 1, 'temperature': 0.3}
```

Una vez que el modelo LLM está configurado, podemos interactuar con él de la misma manera que lo haríamos con ChatGPT.
```python
ans = llm_gpt3_5("Cómo puedo lograr una clase más interactiva para estudiantes virtuales?")
print(ans)
```
Respuesta esperada:
```commandline
Aquí hay algunas sugerencias para lograr una clase más interactiva para estudiantes virtuales:

1. Utiliza herramientas de videoconferencia interactivas: Utiliza plataformas de videoconferencia que permitan a los estudiantes participar activamente a través de funciones como chat, levantar la mano virtualmente, compartir pantalla, etc. Esto les dará la oportunidad de hacer preguntas, compartir ideas y participar en discusiones.

2. Fomenta la participación activa: Anima a los estudiantes a participar activamente en la clase mediante preguntas, debates y actividades prácticas. Puedes asignar roles a los estudiantes, como líder de discusión o presentador, para que se sientan más involucrados en el proceso de aprendizaje.

3. Utiliza herramientas de colaboración en línea: Utiliza herramientas de colaboración en línea, como Google Docs o Padlet, para que los estudiantes puedan trabajar juntos en proyectos, compartir ideas y colaborar en tiempo real.

4. Incorpora actividades interactivas: Integra actividades interactivas en tu clase, como cuestionarios en línea, juegos educativos o simulaciones virtuales. Estas actividades ayudarán a mantener el interés de los estudiantes y les permitirán aplicar lo que están aprendiendo de manera práctica.

5. Proporciona retroalimentación constante: Proporciona retroalimentación constante a los estudiantes para que se sientan involucrados y motivados. Puedes utilizar herramientas de retroalimentación en línea, como comentarios en documentos compartidos o evaluaciones en línea, para brindarles retroalimentación inmediata sobre su desempeño.

6. Promueve la comunicación entre los estudiantes: Fomenta la comunicación entre los estudiantes mediante la creación de grupos de discusión en línea, foros de debate o chats grupales. Esto les permitirá interactuar entre ellos, compartir ideas y aprender de sus compañeros.

7. Varía tus métodos de enseñanza: Utiliza una variedad de métodos de enseñanza, como videos, presentaciones interactivas, actividades prácticas y debates, para mantener el interés de los estudiantes y adaptarte a diferentes estilos de aprendizaje.

Recuerda adaptar estas sugerencias a las necesidades y características específicas de tus estudiantes y materia de enseñanza.
```

Además, la función `generate` nos permite pasar una lista de entradas de prompts, lo cual produce una salida más detallada que incluye información como el uso de tokens. Esta información de uso de tokens puede ser útil para realizar un seguimiento de los tokens y estimar los costos.

```python
llm_davinci = OpenAI(
    model_name="text-davinci-003",
    n=2,
    temperature=0.3
    )

generacion = llm_davinci.generate(
    ["Dime un consejo de vida para alguien de 30 años", "Recomiendame libros similares a Hyperion Cantos"]
    )

pprint(generacion.generations)
```
Respuesta esperada:
```python
[[Generation(text='\n\nNo dejes que el pasado te detenga. Aprende de tus errores y sigue adelante. Vive el presente al máximo y construye tu futuro con esperanza y optimismo.', generation_info={'finish_reason': 'stop', 'logprobs': None}),
  Generation(text='\n\nAprovecha al máximo tu tiempo y energía para hacer las cosas que te hacen feliz. No te preocupes por lo que otros piensan de ti, sigue tu propio camino y no te detengas por los obstáculos. Aprende a decir no a lo que no te hace feliz y sí a lo que te hace sentir realizado.', generation_info={'finish_reason': 'stop', 'logprobs': None})],
 [Generation(text="\n\n1. The Fall of Hyperion de Dan Simmons\n2. Endymion de Dan Simmons\n3. Ilium de Dan Simmons\n4. Olympos de Dan Simmons\n5. Dune de Frank Herbert\n6. The Forever War de Joe Haldeman\n7. The Hitchhiker's Guide to the Galaxy de Douglas Adams\n8. The Foundation Trilogy de Isaac Asimov\n9. Snow Crash de Neal Stephenson\n10. The Culture Series de Iain M. Banks", generation_info={'finish_reason': 'stop', 'logprobs': None}),
  Generation(text="\n\n1. The Fall of Hyperion de Dan Simmons\n2. Endymion de Dan Simmons\n3. Ilium de Dan Simmons\n4. Olympos de Dan Simmons\n5. Dune de Frank Herbert\n6. The Forever War de Joe Haldeman\n7. The Mote in God's Eye de Larry Niven y Jerry Pournelle\n8. The Hitchhiker's Guide to the Galaxy de Douglas Adams\n9. The Foundation Trilogy de Isaac Asimov\n10. The Culture Series de Iain M. Banks", generation_info={'finish_reason': 'stop', 'logprobs': None})]]
```
Ahora podemos acceder a la información de la cantidad de tokens utilizados en esta petición:
```python
pprint(generacion.llm_output)
```
Respuesta esperada
```commandline
{'model_name': 'text-davinci-003',
 'token_usage': {'completion_tokens': 374,
                 'prompt_tokens': 32,
                 'total_tokens': 406}}
```

Otra función útil proporcionada por la clase LLM es `get_num_tokens`, que estima el número de tokens y fragmentos de texto contenidos en una entrada. Esta información es valiosa cuando se necesita limitar el número total de tokens o cumplir con un presupuesto específico.

```python
n_tokens_preview = llm_gpt3_5.get_num_tokens("mis jefes se van a preocupar si gasto mucho en openai")
print(n_tokens_preview)
```
Respuesta esperada:
```commandline
16
```

> ## Nota:
> El uso de método get_num_tokens necesita una bibliteca adicional:
>   > pip install tiktoken
> Tiktoken es opensource y fue desarrollada directamente por OpenAI
> 


## 1.5 Prompt templates de LangChain

Los prompt templates nos van a permitir ingresar la información que queremos que el modelo procese y luego nos dé un resultado.

Un 'prompt' o 'indicación' es como una receta que le proporcionamos a nuestro modelo de inteligencia artificial (IA). Esta receta contiene los ingredientes y las instrucciones que la IA necesita para cocinar la respuesta que estamos buscando.

1. **Instrucciones:** Esta es la parte donde le decimos a nuestro modelo de IA exactamente qué queremos que haga. Piensa en esto como cuando lees una receta de cocina. Por ejemplo, "corta las verduras", "sofríe los ingredientes", etc.

2. **Información externa o contexto:** Este es el ingrediente que añadimos a nuestra receta. Podría ser información que obtenemos de una base de datos, un cálculo que hemos hecho, etc. Esto le da a nuestro modelo un poco de sabor adicional y contexto sobre lo que estamos buscando.

3. **Entrada del usuario o consulta:** Este es el ingrediente principal de nuestra receta. Es el dato que el usuario introduce y en torno al cual queremos que nuestro modelo cocine la respuesta.

4. **Indicador de salida:** Piensa en esto como el momento en el que sabes que tu receta está lista. Para un modelo que genera código Python, podría ser la palabra 'import', que suele ser el comienzo de muchos scripts de Python. Para un chatbot, podría ser la frase 'Chatbot:', indicando que es hora de que el chatbot hable.

Por lo general, estos componentes se colocan en el orden en que los hemos descrito, igual que seguirías los pasos de una receta de cocina. Empezamos con las instrucciones, añadimos el contexto, luego la entrada del usuario, y finalmente, buscamos nuestro indicador de salida para saber que hemos terminado.

Agregamos estas cuatro recetas en el siguiente prompt que habla con estilo argentino.

> ## Nota:
> El código lo puedes encontrar en: [4_prompt_templates.py](scripts%2F4_prompt_templates.py)

```python
prompt_argentino = """Respondé la pregunta basándote en el contexto de abajo. Si la
pregunta no puede ser respondida usando la información proporcionada,
respondé con "Ni idea, che".

Contexto: Los Modelos de Lenguaje de Gran Escala (MLGEs) son lo último en modelos usados en el Procesamiento del Lenguaje Natural (NLP).
Su desempeño superior a los modelos más chicos los hizo increíblemente
útiles para los desarrolladores que arman aplicaciones con NLP. Estos modelos
se pueden acceder vía la librería `transformers` de Hugging Face, vía OpenAI
usando la librería `openai`, y vía Cohere usando la librería `cohere`.

Pregunta: ¿Qué librerías están cerca de Buenos Aires?

Respuesta (escribe como argentina informal): """

print(llm_gpt3_5(prompt_argentino))
```
Respuesta esperada:
```commandline
Ni idea, che.
```
Normalmente, no tenemos ni idea de lo que los usuarios van a preguntar de antemano. Así que, en lugar de escribir la pregunta directamente en el código, creamos un `PromptTemplate` (una plantilla de indicación) que tiene una casilla reservada para la pregunta. Es como tener una receta de cocina, pero en lugar de especificar 'pollo', tenemos un espacio en blanco que dice 'ingrediente principal'. De esta manera, los usuarios pueden poner lo que quieran en ese espacio, y el sistema adaptará su respuesta de acuerdo con lo que ellos introduzcan.

```python
from langchain import PromptTemplate

plantilla_colombiana = """Responde a la pregunta con base en el siguiente contexto, parce. Si la
pregunta no puede ser respondida con la información proporcionada, responde
con "No tengo ni idea, ome".

Contexto: Los Modelos de Lenguaje Grandes (LLMs) son los últimos modelos utilizados en PNL.
Su rendimiento superior sobre los modelos más pequeños los ha hecho increíblemente
útiles para los desarrolladores que construyen aplicaciones habilitadas para PNL. Estos modelos
pueden ser accedidos a través de la biblioteca `transformers` de Hugging Face, a través de OpenAI
usando la biblioteca `openai`, y a través de Cohere usando la biblioteca `cohere`.

Pregunta: {pregunta}

Respuesta (escribe como colombiano informal): """

prompt_plantilla_colombiana = PromptTemplate(
    input_variables=["pregunta"],
    template=plantilla_colombiana
)
```
Crearemos una cadena, más adelante conocerás exactamente qué significa esto. Por ahora, lo relevante es que nos permite unir nuestro prompt con un modelo.
```python
from langchain import LLMChain

llm_gpt3_5_chain = LLMChain(
    prompt=prompt_plantilla_colombiana,
    llm=llm_gpt3_5
)

pregunta = "Qué son los LLMs?"

ans = llm_gpt3_5_chain.run(pregunta)
print(ans)
```
Respuesta esperada:
```commandline
Los LLMs son los últimos modelos utilizados en PNL, parcero. Son más grandes y tienen un mejor rendimiento que los modelos más pequeños. Son muy útiles para los desarrolladores que construyen aplicaciones con PNL. Se pueden acceder a través de las bibliotecas `transformers` de Hugging Face, `openai` de OpenAI y `cohere` de Cohere.
```
```python
pregunta = "Qué son las RAFGSERS?"

ans = llm_gpt3_5_chain.run(pregunta)
print(ans)
```
Respuesta esperada:
```commandline
No tengo ni idea, ome.
```

## 1.6 Cadenas en LangChain

Una cadena conjunta información en un proceso, que es probable que culmine en un LLM respondiendo una pregunta. Por dar un ejemplo:
Podemos tener al inicio de la cadena un proceso de limpieza de datos y en la segunda parte de la cadena se podría recibir un promt e información del usuario para responder una pregunta.

Hay dos tipos de cadenas:

- **Utility:** Son cadenas que ya tienen un propósito muy específico 
  - Generar resúmenes a partir de texto
  - responder preguntas
  - Crear una conversación (con o sin) memoria

> Nota: Las cadenas de `Utility` están creadas a partir de cadenas `Foundational`

- **Foundational:** Estas están creadas a partir de cadenas fundacionales. 
  - LLM: Podemos agregar un `prompt` + un `modelo de lenguaje`
  - Transformation: Reciba un texto y lo limpie para insertarlo a otra cadena
  - Sequential: Sirve para unir cadenas, es un envoltorio.

## 1.7 Utility Chains

Las cadenas están en el corazon, y nombre de LangChain. Lo que hacemos es tener una secuencia de eslabones, en donde cada uno
representa un proceso diferente. La entrada es un texto, pasa por una secuencia de funciones y a la salida puede generar otro texto
que utilice un modelo de lenguaje. En este ejemplo vamos a enfocarnos en cadenas de `Utilidad` y `Funcional`.

Una cadena está compuesta por diferentes elementos, denominados eslabones, que pueden ser primitivas o incluso otras cadenas. Las primitivas, a su vez, pueden ser prompts, LLMs, utilidades, u otras cadenas.

Así que, en términos sencillos, una cadena no es más que una secuencia de operaciones que se llevan a cabo utilizando una mezcla específica de primitivas para procesar una entrada dada. Si lo visualizas de manera intuitiva, podrías pensar en una cadena como una especie de 'paso', que realiza un conjunto específico de operaciones en una entrada y produce un resultado. Estas operaciones pueden variar desde un prompt que pasa a través de un LLM, hasta la ejecución de una función de Python sobre un texto.

Las cadenas se agrupan en tres categorías principales:
- Cadenas de utilidad (Utility chain).
- Cadenas fundacionales (Foundational chains).
- Cadenas de combinación de documentos.

En este segmento, nos concentraremos en las primeras dos categorías, ya que la tercera es muy específica.

* Cadenas de utilidad (Utility chains): Este tipo de cadenas generalmente se emplean para extraer una respuesta específica de un LLM con un objetivo muy definido, y están listas para ser usadas sin modificaciones.

* Cadenas fundacionales (Foundational chains): Estas cadenas se usan como base para construir otras cadenas, sin embargo, a diferencia de las Cadenas de utilidad, las Cadenas fundacionales no pueden ser usadas tal cual sin formar parte de una cadena más compleja.

Importamos un texto para que trabajemos con él. En el módulo de índices aprenderemos más sobre lo que está ocurriendo.

Empecemos instalando algunas dependencias:
```bash
pip install unstructured pypdf chromadb
```

> ## Nota:
> El código completo esta en: [5_utility_chains.py](scripts%2F5_utility_chains.py)

Antes de continuar vamos a marcar los `antecedentes` que son necesarios para explicar las cosas nuevas que vamos a ver en este
código de ejemplo:
En resumen, seteamos como variable de entorno el APIKEY de OpenAI y creamos dos LLMs un GPT3.5 y un DAVINCI.
```python
# Antecedentes 1: Cargar el API KEY de OpenAI como una variable de sistema.
from dotenv import load_dotenv
load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Antecedentes 2: Instanciar dos LLMs de OpenAI un GPT3.5 y un Davinci
from langchain.llms import OpenAI
llm_gpt3_5 = OpenAI(
    model_name="gpt-3.5-turbo",
    n=1,
    temperature=0.3
)

llm_davinci = OpenAI(
    model_name="text-davinci-003",
    n=2,
    temperature=0.3
    )
```

Ahora sí, vamos a empezar por descargar un documento PDF y empezar a cargarlo a través de `PyPDFLoader`:

```python
url = 'https://www.cs.virginia.edu/~evans/greatworks/diffie.pdf'
filename = "public_key_cryptography.pdf"

# Descargamos el archivo PDF del Url con un nombre `filename`
if not os.path.exists(filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f'Descargado {filename}')
else:
    print(f'{filename} ya existe, cargando desde el disco.')

# Convertimos el archivo PDF a un documento legible `Loader`
loader = PyPDFLoader(filename)
data = loader.load()

# Creamos nuestro generador de `embeddings`(vectorizador de texto) utilizando OpenAi como modelo base.
embeddings = OpenAIEmbeddings()
# Chroma nos permitirá convertir los datos de texto que hay en `data` en una base de datos vectorial utilizando a
# `embeddings" como método para conversión de texto a vector.
vectorstore = Chroma.from_documents(data, embeddings)
```
Respuesta esperada:
```commandline
Descargado public_key_cryptography.pdf
```
Imagina que tienes un libro de 18 páginas delante de ti y alguien te pide un resumen. ¿Cómo lo haces? Seguramente no intentarías leer las 18 páginas de una sola vez y luego producir un resumen, ¿verdad? En su lugar, es probable que leyeras una página a la vez, escribieras un pequeño resumen, y luego lo harías para las siguientes páginas. Después de resumir todas las páginas de manera individual, podrías combinar esos resúmenes en uno más general.

¡Exactamente eso es lo que hace **`load_summarize_chain`**!

Esta cadena divide el texto en secciones más manejables (por ejemplo, por página) y luego llama al modelo de lenguaje para generar un resumen de cada sección. Una vez que todos los resúmenes individuales se han generado, los combina en un solo resumen. De esta manera, incluso un texto muy largo se puede resumir de manera efectiva.

Por tanto, si estás trabajando con un documento de investigación de 18 páginas, puedes dividirlo en 18 documentos separados, uno por página, y luego usar **`load_summarize_chain`** para generar un resumen general. ¿No es fantástico?

```python
# Esto es la cantidad de paǵinas en el documento
print("páginas:", len(data))
```
Respuesta esperada:
```commandline
páginas: 18
```

> Nota: hablaremos en un texto sobre los chain types.

Ahora veamos como podemos hacer un resumen con `load_summarize_chain` y el `chain_type map_reduce`:

```python
from langchain.chains.summarize import load_summarize_chain

# La cadena load_summarize busca crear resumenes, pero también utilizamos `map_reduce` para que cada página sea
# resumida antes de hacer el resumen general del documento.
cadena_que_resume_documentos = load_summarize_chain(
    llm_davinci,
    chain_type="map_reduce"
)

ans = cadena_que_resume_documentos.run(data)
print("*"*64)
print(ans)
```
Respuesta esperada:
```commandline
Public-key cryptography was discovered in 1975 and revolutionized communication security by allowing secure communication 
networks with hundreds of thousands of subscribers. It solves two problems: key distribution and digital signatures. 
Roger Needham of Cambridge University developed a system to protect computer passwords, which was then extended to public-key 
cryptography. In 1976, Marty and the author published a paper on multi-user cryptographic techniques, and Diffie discussed 
the difficulty of computing logarithms in a field. The RSA system is based on the difficulty of factoring large numbers and 
is proven to be secure. In the early 1980s, public-key cryptography was met with criticism from the cryptographic establishment, 
but this only added to the publicity of the discovery. In the mid-1980s, the NSA began feasibility studies for a new secure 
phone system, the STU-III, which used public key. Several companies dedicated to developing public-key technology were formed 
by academic cryptographers, and research in public-key cryptography has been motivated by application. Public-key cryptography 
has become a mainstay of cryptographic technology and is soon to be implemented in hundreds of thousands of secure telephones.
```

Si usamos una `chain_type` stuff entonces podemos incluir nuestro propio prompt/plantilla. Sin embargo, le cabe solo el máximo de tokens permitidos por el modelo y no documentos largos.

```python
# Sin embargo, `map_reduce`no es la única chain type, vamos a probar ahora con `stuff` porque esta nos permite
# integrar `prompts` a nuestras peticiones. Adicionalmente, nuestro `prompt` será enviado a partir de un
# `Prompt template` para tener más control del mismo.

from langchain import PromptTemplate
plantilla = """Escribe un resumen bien chido del siguiente rollo:

{text}

RESUMEN CORTO CON SLANG MEXICANO:"""

# Dada la plantilla definimos que solo tenemos una variable de entrada `text`
prompt = PromptTemplate(
    template=plantilla,
    input_variables=["text"]
)
```
Con el prompt definido, ahora vamos a concatenar el mismo a un LLM:
```python
# Concatenamos nuestra plantilla al LLM davinci
cadena_que_resume_con_slang = load_summarize_chain(
    llm=llm_davinci,
    chain_type="stuff",
    prompt=prompt,
    verbose=True  # Esto nos dará información por terminal, pero en producción no es necesario.
)
# Vamos a observar la respuesta que nos da de resumen al utilizar solo las primeras 2 hojas de contenido
ans = cadena_que_resume_con_slang.run(data[:2])
print("*"*64)
print(ans)
```
Respuesta esperada:
```commandline
> Entering new StuffDocumentsChain chain...


> Entering new LLMChain chain...
Prompt after formatting:
Escribe un resumen bien chido del siguiente rollo:

The First Ten Years of Public-Key 
Cryptography 
WH lTFl ELD DI FFlE 
Invited Paper 
Public-key cryptosystems separate the capacities for encryption 
and decryption so that 7) many people can encrypt messages in 
such a way that only one person can read them, or 2) one person 
can encrypt messages in such a way that many people can read 
them. This separation allows important improvements in the man- 
agement of cryptographic keys and makes it possible to ‘sign’ a 
purely digital message. 
Public key cryptography was discovered in the Spring of 1975 
and has followed a surprising course. Although diverse systems 
were proposed early on, the ones that appear both practical and 
secure today are all very closely related and the search for new and 
different ones has met with little success. Despite this reliance on 
a limited mathematical foundation public-key cryptography is rev- 
olutionizing communication security by making possible secure 
communication networks with hundreds of thousands of subscrib- 
ers. 
Equally important is the impact of public key cryptography on 
the theoretical side of communication security. It has given cryp- 
tographers a systematic means of addressing a broad range of 
security objectives and pointed the way toward a more theoretical 
approach that allows the development of cryptographic protocols 
with proven security characteristics. 
I. INITIAL DISCOVERIES 
Public key cryptography was born in May 1975, the child 
First came the problem of key distribution. If two peo- 
ple who have never met before are to communicate 
privately using conventional cryptographic means, 
they must somehow agree in advance on a key that will 
be known to themselves and to no one else. 
The second problem, apparently unrelated to the first, 
was the problem of signatures. Could a method be 
devised that would provide the recipient of a purely 
digital electronic message with a way of demonstrat- 
ing to other people that it had come from a particular 
person, just as awritten signature on a letter allows the 
recipient to hold the author to its contents? 
On the face of it, both problems seem to demand the 
impossible. In the first case, if two people could somehow 
communicate a secret key from one to the other without 
ever having met, why could they not communicate their 
Manuscript received January 19, 1988; revised March 25,1988. 
The author is with Bell-Northern Research, Mountain View, CA 
IEEE Log Number 8821645. of two problems and a misunderstanding. 
94039, USA. message in secret? The second is no better. To be effective, 
a signature must be hard to copy. How then can a digital 
message, which can be copied perfectly, bear a signature? 
The misunderstanding was mine and prevented me from 
rediscovering the conventional key distri bution center. The 
virtue of cryptography, I reasoned, was that, unlike any 
other known security technology, it did not require trust 
in any party not directly involved in the communication, 
only trust in the cryptographic systems. What good would 
it do to develop impenetrable cryptosystems, I reasoned, 
if their users were forced to share their keys with a key dis- 
tribution center that could be compromised by either bur- 
glary or subpoena. 
The discovery consisted not of a solution, but of the rec- 
ognition that the two problems, each of which seemed 
unsolvable by definition, could be solved at all and that the 
solutions to both problems came in one package. 
First to succumb was the signature problem. The con- 
ventional use of cryptography to authenticate messages had 
been joined in the 1950s by two new applications, whose 
functions when combined constitute a signature. 
Beginning in 1952, a group under the direction of Horst 
Feistel at the Air Force Cambridge Research Center began 
to apply cryptography to the military problem of distin- 
guishing friendly from hostile aircraft. In traditional Iden- 
tification Friend or Foe systems, a fire control radar deter- 
mines the identity of an aircraft by challenging it, much as 
a sentry challenges a soldier on foot. If the airplane returns 
the correct identifying information, it is judged to be 
friendly, otherwise it is thought to be hostile or at best neu- 
tral. To allow the correct response to remain constant for 
any significant period of time, however, is to invite oppo- 
nents to record a legitimate friendly response and play it 
back whenever they themselves are challenged. The 
approach taken by Feistel’s group, and now used in the MK 
XI1 IFF system, is to vary the exchange cryptographically 
from encounter to encounter. The radar sends a randomly 
selected challenge and judges the aircraft by whether it 
receives a correctly encrypted response. Because the chal- 
lenges are never repeated, previously recorded responses 
will not be judged correct by a challenging radar. 
Later in the decade, this novel authentication technique 
was joined by another, which seems first to have been 
560 001&9219/88/0500-0560$01.00 0 1988 IEEE 
PROCEEDINGS OF THE IEEE, VOL. 76, NO. 5, MAY 1988 

applied by Roger Needham of Cambridge University [112]. 
This timethe problem was protectingcomputer passwords. 
Access control systems often suffer from the extreme sen- 
sitivity of their password tables. The tables gather all of the 
passwards together in one place and anyone who gets 
access to this information can impersonate any of the sys- 
tem‘s users. To guard against this possibility, the password 
table is filled not with the passwords themselves, but with 
the images of the passwords under a one-way function. A 
one-way function is easy to compute, but difficult to invert. 
For any password, the correct table entry can be calculated 
easily. Given an output from the one-way function, how- 
ever, it is exceedingly difficult to find any input that will 
produce it. This reduces the value of the password table to 
an intruder tremendously, since its entries are not pass- 
words and are not acceptable to the password verification 
routine. 
Challenge and response identification and one-wayfunc- 
tions provide protection against two quite different sorts 
of threats. Challengeand response identification resists the 
efforts of an eavesdropper who can spy on the commu- 
nication channel. Since the challengevaries randomlyfrom 
event to event, the spy is unable to replay it and fool the 
challenging radar. There is, however, no protection against 
an opponent who captures the radar and learns its cryp- 
tographic keys. This opponent can use what he has learned 
to fool any other radar that is keyed the same. In contrast, 
the one-way function defeats the efforts of an intruder who 
captures the system password table (analogous to captur- 
ing the radar) but scuccumbs to anyone who intercepts the 
login message because the password does not change with 
time. 
I realized that the two goals might be achieved simul- 
taneously if the challenger could pose questions that it was 
unable to answer, but whose answers it could judge for cor- 
rectness. I saw the solution as a generalization of the one- 
way function: a trap-door one-way function that allowed 
someone in possession of secret information to go back- 
wards and compute the function’s inverse. The challenger 
would issue a value in the range of the one-way function 
and demand to know its inverse. Onlythe person who knew 
the trapdoor would be able to find the corresponding ele- 
ment in the domain, but the challenger, in possession of 
an algorithm for computing the one-way function, could 
readilychecktheanswer. In theapplicaticnsthat later came 
toseem most important, the roleof thechallengewas played 
by a message and the process took on the character of a 
signature, a digital signature. 
It did not take long to realize that the trap-door one-way 
function could also be applied to the baffling problem of 
key distribution. For someone in possession of the forward 
form of the one-way function to send a secret message to 
the person who knew the trapdoor, he had only to trans- 
form the message with the one-way function. Only the 
holder of the trap-door information would be able to invert 
the operation and recover the message. Because knowing 
the forward form of the function did not make it possible 
to compute the inverse, the function could be made freely 
available. It is this possibility that gave the field its name: 
public-ke y cryptography. 
The concept that emerges is that of a public-key cryp- 
tosystem: a cryptosystem in which keys come in inverse 
pairs [36] and each pair of keys has two properties. - Anything encrypted with one key can be decrypted 
with the other. 
Given one member of the pair, the public key, it is 
infeasible to discover the other, the secret key. 
This separation of encryption and decryption makes it 
possible for the subscribers to a communication system to 
list their public keys in a “telephone directory” along with 
their names and addresses. This done, the solutions to the 
original problems can be achieved by simple protocols. 
One subscriber can send a private message to another 
simply by looking up the addressee’s public key and 
using it to encrypt the message. Only the holder of the 
corresponding secret key can read such a message; 
even the sender, should he lose the plaintext, is inca- 
pable of extracting it from the ciphertext. 
A subscriber can sign a message by encrypting it with 
his own secret key. Anyone with access to the public 
key can verify that it must have been encrypted with 
the corresponding secret key, but this is of no help to 
him in creating (forging) a message with this property. 
The first aspect of public-key cryptography greatly sim- 
plifies the management of keys, especially in large com- 
munication networks. In order for a pair of subscribers to 
communicate privately using conventional end-to-end 
cryptography, they must both have copies of the same cryp- 
tographic key and this key must be kept secret from anyone 
they do not wish to take into their confidence. If a network 
has only a few subscribers, each person simply stores one 
key for every other subscriber against the day he will need 
it, but for a large network, this is impractical. 
In a network with n subscribers there are n(n - 1)/2 pairs, 
each of which may require a key. This amounts to five thou- 
sand keys in a network with only a hundred subscribers, 
half a million in a network with one thousand, and twenty 
million billion in a network the size of the North American 
telephone system. It is unthinkable to distribute this many 
keys in advance and undesirable to postpone secure com- 
munication while they are carried from one party to the 
other by courier. 
The second aspect makes its possible to conduct a much 
broader range of normal business practices over a tele- 
communication network. Theavailabilityof asignature that 
the receiver of a message cannot forge and the sender can- 
not readily disavow makes it possible to trust the network 
with negotiations and transactions of much higher value 
than would otherwise be possible. 
It must be noted that both problems can be solved with- 
out public-key cryptography, but that conventional solu- 
tions come at a great price. Centralized key distribution 
centers can on request provide a subscriber with a key for 
communicating with any other subscriber and protocols 
for this purpose will be discussed later on. The function of 
the signature can also beapproximated byacentral registry 
that records all transactions and bears witness in cases of 
dispute. Both mechanisms, however, encumber the net- 
work with the intrusion of a third party into many conver- 
sations, diminishing security and degrading performance. 
At the time public-key cryptography was discovered, I 
was working with Martin Hellman in the Electrical Engi- 
neering Department at Stanford University. It was our 
immediate reaction, and by no means ours alone, that the 
DIFFIE: TEN YEARS OF PUBLIC-KEY CRYPTOGRAPHY 561 

RESUMEN CORTO CON SLANG MEXICANO:

> Finished chain.
```
```commandline
Public-key cryptography fue descubierta en la primavera de 1975 y ha seguido un curso sorprendente. Esta criptografía separa 
la capacidad de encriptación y desencriptación para que muchas personas puedan encriptar mensajes de tal manera que solo una 
persona pueda leerlos, o una persona pueda encriptar mensajes de tal manera que muchas personas puedan leerlos. Esto permite 
mejoras importantes en la gestión de las claves criptográficas y hace posible firmar un mensaje digital. Esta criptografía de 
clave pública está revolucionando la seguridad de la comunicación al permitir redes de comunicación seguras con cientos de 
miles de suscriptores. Esto también tiene un impacto en el lado teórico de la seguridad de la comunicación. ¡La criptografía 
de clave pública
```

## 1.8 RetrievalQA Chain


> ## Nota:
> El código completo esta en: [5_utility_chains.py](scripts%2F5_utility_chains.py)

También podemos resolver preguntas con `RetrievalQA`
```python
# No siempre vamos a necesitar resúmenes del texto, que pasa si queremos responder alguna pregunta del texto?
from langchain.chains import RetrievalQA
# En ese sentido, NO vamos a usar un prompt template con el texto a analizar, sino que vamos a utilizar nuestra
# base de datos vectorial, y le vamos a pasar toda la información al `retriever`
cadena_que_resuelve_preguntas = RetrievalQA.from_chain_type(
    llm=llm_gpt3_5,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})  # kwargs k son los parámetros de la busqueda de cuales son los
  # fragmentos de texto relevantes para responder la pregunta, entre más fragmentos de texto mejor porque tiene más contexto el modelo,
  # pero NO van a entrar todas en un prompt. Entre mayor sea el texto a analizar también lo será el precio.
)
# Ahora ya podemos hacer prompts con preguntas a resolver.
ans = cadena_que_resuelve_preguntas.run("¿Cuál es la relevancia de la criptografía de llave pública?")
print("*"*64)
print(ans)
```
Respuesta esperada:
```commandline
The relevance of public-key cryptography is that it provides a secure method for key distribution and authentication in 
communication systems. It allows for secure communication between parties without the need for a trusted third party or the 
sharing of secret keys. Public-key cryptography has become a mainstay of cryptographic technology and is widely accepted 
and used in various applications. It offers improved security compared to conventional cryptography and has the potential 
to revolutionize data communications and electronic funds transfer technology. However, there are concerns about the narrow 
technological base and vulnerability to breakthroughs in factoring or discrete logarithms. Despite these concerns, public-key 
cryptography is considered a strong and indispensable tool in modern cryptography.
```

## 1.9 Foundational Chains

En esta clase veremos como funcionan las cadenas `Foundatinoal` y como podemos unir varias de ellas para llevar procesos 
más complejos a través de cadenas secuenciales `SequentialChain`.

Primero, vamos a construir una función personalizada para limpiar nuestros textos de URLs y emojis. Luego, utilizaremos esta función para crear una cadena en la que introduciremos nuestro texto y esperamos obtener un texto limpio como salida.

Debemos tener en cuenta que la función que hemos creado recibe como entrada un diccionario. En este diccionario, vamos a indicar los elementos que serán procesados por la cadena que estamos creando. El resultado que obtendremos de la cadena será el texto limpio.

Esto es el principio fundamental de las cadenas fundacionales, nos proporcionan un marco para llevar a cabo una serie de transformaciones de manera ordenada y estructurada.

Partimos de los antecedentes que ya conocemos:
```python
# Antecedentes 1: Cargar el API KEY de OpenAI como una variable de sistema.
import os
from dotenv import load_dotenv

load_dotenv("../secret/keys.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Antecedentes 2: Instanciar dos LLMs de OpenAI un GPT3.5 y un Davinci
from langchain.llms import OpenAI

llm_gpt3_5 = OpenAI(
    model_name="gpt-3.5-turbo",
    n=1,
    temperature=0.3
)
```
Vamos a empezar creando una función de limpieza de datos. La misma es una función que recibe un diccionario con una llave `texto`
que contiene el texto a limpiar y devuelve otro diccionario con la llave `texto_limpio`, el texto ya normalizado.

```python
def limpiar_texto(entradas: dict) -> dict:
    texto = entradas["texto"]

    # Eliminamos los emojis utilizando un amplio rango unicode
    # Ten en cuenta que esto podría potencialmente eliminar algunos caracteres válidos que no son en inglés
    patron_emoji = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticonos
        "\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # símbolos de transporte y mapas
        "\U0001F1E0-\U0001F1FF"  # banderas (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE,
    )
    texto = patron_emoji.sub(r'', texto)

    # Removemos las URLs
    patron_url = re.compile(r'https?://\S+|www\.\S+')
    texto = patron_url.sub(r'', texto)

    return {"texto_limpio": texto}
```
Con base en esta función de python podemos crear nuestro primer bloque de Cadena utilizando `TransformChain`:

```python
from langchain.chains import TransformChain

cadena_que_limpia = TransformChain(
    input_variables=["texto"],
    output_variables=["texto_limpio"],
    transform=limpiar_texto
)

clean = cadena_que_limpia.run('Chequen está página https://twitter.com/home 🙈')
print(clean)
```
Respuesta esperada:
```commandline
Chequen está página  
```

Ahora vamos a crear un par de cadenas más y finalmente las vamos a unir todas para un flujo de información completa.
Empecemos con una cadena de parafraseo de texto:

```python
from langchain import PromptTemplate
from langchain.chains import LLMChain

# Empezamos creando nuestro prompt template que recibe como parámetro un 'texto_limpio' (salida de de la cadena de limpieza)
# y lo parafrasea con un estilo informa de una persona (estilo).
plantilla_parafrasea = """Parafrasea este texto:

{texto_limpio}

En el estilo de una persona informal de {estilo}.

Parafraseado: """

# Dado que nuestro Template tiene 2 variables, debemos indicarlas en el parámetro `input_variables
prompt_parafraseo = PromptTemplate(
    input_variables=["texto_limpio", "estilo"],
    template=plantilla_parafrasea 
)

# Ahora solo falta crear la cadena que cambia estilo utilizando como LLM a GPT3.5, esta cadena terminará creando una variable
# a la salida llamada `texto_final`
cadena_que_cambia_estilo = LLMChain(
    llm=llm_gpt3_5,
    prompt=prompt_parafraseo,
    output_key='texto_final'
)
```
Ahora siguiendo la misma estructura lógica, vamos a crear una nueva `Chain` que se encargue de parafrasear un texto de entrada:

```python
# Texto_final es la variable de entrada, puesto que así la definimos en la cadena de parafraseo
plantilla_resumen = """Resume este texto:

{texto_final}

Resumen: """

prompt_resumen = PromptTemplate(
    input_variables=["texto_final"],
    template=plantilla_resumen
)

# Texto resumido será la variable final con la que termina nuestra secuencia de cadenas
cadena_que_resume = LLMChain(
    llm=llm_gpt3_5,
    prompt=prompt_resumen,
    output_key="texto_resumido"
)
```

Finalmente, para concluir vamos a unir todas nuestras cadenas entre ellas utilizando `SequentialChain`:

```python
from langchain.chains import SequentialChain

cadena_secuencial = SequentialChain(
    chains=[cadena_que_limpia, cadena_que_cambia_estilo, cadena_que_resume],
    input_variables=["texto", "estilo"],
    output_variables=["texto_resumido"]
)
```
> Nota: 
> Esta estructura de pensamiento es MUY similar a como PyTorch Organiza las capas de un modelo de DL.

Probemos entonces nuestra `SequentialChain`:

```python
texto_entrada = """
¡Monterrey es una ciudad impresionante! 🏙️
Es conocida por su impresionante paisaje de montañas ⛰️ y su vibrante cultura norteña.
¡No olvides visitar el famoso Museo de Arte Contemporáneo (MARCO)!
🖼️ Si eres fanático del fútbol, no puedes perderte un partido de los Rayados o de los Tigres. ⚽
Aquí te dejo algunos enlaces para que puedas conocer más sobre esta maravillosa ciudad:
https://visitamonterrey.com, https://museomarco.org, https://rayados.com, https://www.tigres.com.mx.
¡Monterrey te espera con los brazos abiertos! 😃🇲🇽

Monterrey es la capital y ciudad más poblada del estado mexicano de Nuevo León, además de la cabecera del 
municipio del mismo nombre. Se encuentra en las faldas de la Sierra Madre Oriental en la región noreste de 
México. La ciudad cuenta según datos del XIV Censo de Población y Vivienda del Instituto Nacional de 
Estadística y Geografía de México (INEGI) en 2020 con una población de 3 142 952 habitantes, por lo cual 
de manera individual es la 9.ª ciudad más poblada de México, mientras que la zona metropolitana de Monterrey 
cuenta con una población de 5 341 175 habitantes, la cual la convierte en la 2.ª área metropolitana más 
poblada de México, solo detrás de la Ciudad de México.8​

La ciudad fue fundada el 20 de septiembre de 1596 por Diego de Montemayor y nombrada así en honor al castillo 
de Monterrey en España. Considerada hoy en día una ciudad global, es el segundo centro de negocios y finanzas 
del país, así como una de sus ciudades más desarrolladas, cosmopolitas y competitivas. Sirve como el 
epicentro industrial, comercial y económico para el Norte de México.9​ Según un estudio de Mercer Human 
Resource Consulting, en 2019, fue la ciudad con mejor calidad de vida en México y la 113.ª en el mundo.10​ 
La ciudad de Monterrey alberga en su zona metropolitana la ciudad de San Pedro Garza García, la cual es el 
área con más riqueza en México y América Latina.11​
"""

ans = cadena_secuencial({'texto': texto_entrada, 'estilo': 'ciudad de méxico'})
print(ans)
```
Respuesta esperada:
```commandline
{'texto': 

'\n¡Monterrey es una ciudad impresionante! 🏙️\nEs conocida por su impresionante paisaje de montañas ⛰️ y su vibrante 
cultura norteña.\n¡No olvides visitar el famoso Museo de Arte Contemporáneo (MARCO)!\n🖼️ Si eres fanático del fútbol, no 
puedes perderte un partido de los Rayados o de los Tigres. ⚽\nAquí te dejo algunos enlaces para que puedas conocer más 
sobre esta maravillosa ciudad:\nhttps://visitamonterrey.com, https://museomarco.org, https://rayados.com, https://www.tigres.com.mx.
¡Monterrey te espera con los brazos abiertos! 😃🇲🇽\n\nMonterrey es la capital y ciudad más poblada del estado mexicano de 
Nuevo León, además de la cabecera del \nmunicipio del mismo nombre. Se encuentra en las faldas de la Sierra Madre Oriental en 
la región noreste de \nMéxico. La ciudad cuenta según datos del XIV Censo de Población y Vivienda del Instituto Nacional de 
\nEstadística y Geografía de México (INEGI) en 2020 con una población de 3 142 952 habitantes, por lo cual \nde manera 
individual es la 9.ª ciudad más poblada de México, mientras que la zona metropolitana de Monterrey \ncuenta con una población 
de 5 341 175 habitantes, la cual la convierte en la 2.ª área metropolitana más \npoblada de México, solo detrás de la Ciudad 
de México.8\u200b\n\nLa ciudad fue fundada el 20 de septiembre de 1596 por Diego de Montemayor y nombrada así en honor al 
castillo \nde Monterrey en España. Considerada hoy en día una ciudad global, es el segundo centro de negocios y finanzas 
\ndel país, así como una de sus ciudades más desarrolladas, cosmopolitas y competitivas. Sirve como el \nepicentro industrial, 
comercial y económico para el Norte de México.9\u200b Según un estudio de Mercer Human \nResource Consulting, en 2019, fue 
la ciudad con mejor calidad de vida en México y la 113.ª en el mundo.10\u200b \nLa ciudad de Monterrey alberga en su zona 
metropolitana la ciudad de San Pedro Garza García, la cual es el \nárea con más riqueza en México y América Latina.11\u200b\n', 

'estilo': 
'ciudad de méxico', 

'texto_resumido': '
Monterrey es una ciudad increíblemente hermosa y llena de vida, famosa por sus montañas y su cultura norteña. Es conocida 
por su Museo de Arte Contemporáneo y por los equipos de fútbol Rayados y Tigres. Es la capital y la ciudad más grande del 
estado de Nuevo León, con una población de más de 3 millones de habitantes. Es considerada una ciudad global y un importante 
centro de negocios y finanzas en México. También es una de las ciudades más desarrolladas y con mejor calidad de vida en 
el país. En resumen, Monterrey es una ciudad impresionante y llena de oportunidades.'}
```
Excelente, hemos podido aprender como unir varias cadenas entre ellas para crear un flujo de información efectivo a través
de `SequentialChains`

## 1.10 Quizz Introducción a LangChain

![1.png](ims%2Fq1%2F1.png)

![2.png](ims%2Fq1%2F2.png)

![3.png](ims%2Fq1%2F3.png)

![4.png](ims%2Fq1%2F4.png)

![5.png](ims%2Fq1%2F5.png)

# 2 Casos de uso de LangChain

## 2.1 Casos de uso de LangChain

LangChain es un framework muy versátil disponible en Python y JavaScript que ha tomado bastante importancia en la actualidad.
Una de las claves de su éxito es que siempre hemos visto a los LLMs como una especie de memoria, que conocía todos los
datos con los que fue entrenado, pero, ¿qué pasa con conocimiento que está por fuera de su memoria? 

¿Cómo podemos extraer información con datos recientes que no son conocidos por modelos de OpenAI por ejemplo? Es aquí dónde 
entra LangChain como una propuesta de solución, generar un proceso de una memoria de corto plazo. LangCHain es la respuesta
para brindar información privada o más reciente a modelos LLM.

**¿Cuáles son los principales usos que podemos darle a LangChain?**

**1. Summarization:** Podemos crear resúmenes de textos muy grandes, que de otro modo no podríamos insertar en un solo prompt.

**2. Question Answering:** Dada información muy precisa ser capaz de hacer preguntas y que el modelo responda con los datos extraídos de la nueva fuente de información.

**3. ChatBots:** Generar Bots que tengan memoria y pueda tomar en cuenta toda la conversación que se ha tenido.

**¿Existen casos de éxito donde se haya utilizado LangChain a Nivel Empresarial?**

- **Platzi:** Ha creado un ChatBot con la finalidad de responder preguntas a estudiantes y recomendar cursos que se adapten a las necesidades individuales de cada estudiante.
- **Duolingo:** Ha integrado LangChain para mejorar la experiencia de los estudiantes para aprender idiomas.
- **LangChain:** Ha implementado un asistente que permite resolver dudas acerca de la documentación de la propia empresa, LangChain.

**Mencionemos algunos beneficios de implementar LLMs:**

- Reducción de costos: podemos tener una atención a cliente más eficiente, y con una calidad similar a la que brindaría un humano.
- Ayuda en el proceso de aprendizaje: los llms pueden servir como mentores particulares que sean capaces de responder a nuestras preguntas al momento de aprender una nueva tecnología.
- Ayuda en la exploración de datos: con llms y LangChain podemos extraer fácilmente información de grandes documentos de texto a través de preguntas en lenguaje natural.



## 2.2 ¿Cómo utilizar LangChain en mi equipo?

La barrera de entrada es cada vez más pequeña, los costos actuales son cada vez menores. Se necesita de un ingenier@ de software
que sea capaz de programar un sistema que permita la concurrencia de muchos usuarios haciendo multiples peticiones al mismo tiempo.
Se necesita un equipo de software capaz de implementar de forma eficiente el cómo se utiliza un API.

**Respecto a los datos y seguridad:** si vamos a utilizar un proveedor (OPENAI, COHERE, etc.) hay que conocer la política de privacidad
del uso de los datos (pero realmente los datos sí los pueden conocer ellos y hay que ser conscientes de ello). Si la privacidad
de los datos es fundamental para tu empresa y no quieres que estén en un servidor, puedes utilizar modelos, embeddings y bases
de datos open source.

En el futuro serán vitales las personas que conozcan los sistemas open source que pueden ser utilizados de forma local,
como les podemos entrenar para los datos específicos de nuestro problema a resolver. El tiempo en que una empresa pueda
implementar LangChain y LLMs recae más en puestos de ingeniería de software más que en puestos de data scientist o ml engineer.

El punto es saber cómo conectar con las APIs, escalarlas y después conectar con un proveedor de nube, para que corra el modelo.
Se necesita pensar en:

- La privacidad de los datos
- La concurrencia de las peticiones
- ¿Qué vas a hacer con la información extraída?



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
