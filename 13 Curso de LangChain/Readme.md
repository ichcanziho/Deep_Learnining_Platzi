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

Los LLMs NO pueden generalizar sobre el mundo porque NO lo conocen, solo conocen los datos con los que fue entrenado.

Cuando interactuamos con un modelo de lenguaje, puede recordar información de 2 maneras. **La primera forma es:** a partir de los
datos con los que se entreno. Si los datos llegan hasta una fecha específica, o solo contiene información de un contexto
específico, no van a conocer información actual o externa al contexto, por ejemplo la documentación de un API moderno
o de tu empresa. **La segunda forma es:** `Indexes`, podemos hacer que un modelo recuerde información a partir de datos
ingresados en el prompt. La clave está en la forma en la que se debe ingresar esta información para que el modelo generé
la mejor respuesta a la pregunta del `user`. Los `Indexes` nos ayudarán a encontrar la información clave que necesitamos.

El proceso para crear un índice es:

- Acceder a información actual y propietaria y asi mismo recordar el contexto, son necesarios los `Indexes`
- Conectar desde cualquier tipo de documento, csv, json, excel, txt, word, pdf, etc
- Proceso de partir info, procesarla, transformar info, después ingresarlos con vector store, para buscar y poder contestar adecuadamente con relacion a la alimentacion de la info que le dimos. (fragmentos indexados)
- Los modelos actúan en función de los datos con los cuales ha sido entrenado

En las siguientes clases vamos a aprender a cargar diferentes tipos de documentos a LangChain a partir de los `Documents`.


## 3.2 La clase Document

Los `indexes` de LangChain nos permiten estructurar nuestra base de datos o nuestros documentos. Supongamos una novela de
muchas páginas, 300 por ejemplo, encontrar justamente un `quote` es muy complejo entre tanto texto. Sin embargo, si el libro
está divido en Capítulos, sub capítulos etc. es más factible encontrar información. Esta es la manera conceptual en la que
funcionan los `indexes` en LangChain. 

De esta manera cuando hagamos una pregunta, va a buscar qué pedazos de la gran base de datos son relevantes para resolver la
pregunta. Debajo de cada pedazo está la clase `Document`.

Los índices se refieren a las formas de estructurar documentos para que los Modelos de Lenguaje de Masivos (LLMs) puedan interactuar con ellos de la mejor manera posible. Esta es una tarea esencial para optimizar la eficiencia y velocidad de las operaciones de búsqueda y recuperación de información en sistemas de procesamiento de lenguaje natural.

Puedes pensar en los índices como en el índice de un libro. En un libro el índice te ayuda a localizar rápidamente un capítulo o sección específica sin tener que hojear todas las páginas. De manera similar, los índices en LangChain permiten a los LLMs encontrar rápidamente documentos o información relevantes sin tener que procesar todos los documentos disponibles.

### Índices y recuperación

El uso más común de los índices en las cadenas de procesamiento de datos es en un paso denominado **"recuperación"**. Este paso se refiere a tomar la consulta de un usuario y devolver los documentos más relevantes. Sin embargo, es importante hacer una distinción aquí porque:

1. Un índice puede utilizarse para otras cosas además de la recuperación.
2. La recuperación puede utilizar otras lógicas además de un índice para encontrar documentos relevantes.

La mayoría de las veces, cuando hablamos de índices y recuperación, nos referimos a la indexación y recuperación de datos no estructurados, como documentos de texto. En este contexto, "no estructurado" significa que los datos no siguen un formato fijo o predecible, como lo hace, por ejemplo, una tabla de base de datos. En cambio, los documentos de texto pueden variar ampliamente en términos de longitud, estilo, contenido, etc.

### Retriever en LangChain

El **Retriever** es un componente fundamental en el ecosistema de LangChain. Su responsabilidad principal es localizar y devolver documentos relevantes según una consulta específica. Imagínate un bibliotecario diligente que sabe exactamente dónde encontrar el libro que necesitas en una gran biblioteca; eso es lo que hace el Retriever en LangChain.

Para realizar esta tarea, el Retriever debe implementar el método `get_relevant_documents`. Aunque este método puede ser implementado de la forma que el usuario considere más conveniente, en LangChain se ha diseñado una estrategia para recuperar documentos lo más eficientemente posible. Esta estrategia se basa en el concepto de **Vectorstore**, por lo que vamos a centrarnos en el Retriever tipo Vectorstore en el resto de esta guía.

### Vectorstore y Vectorstore Retriever

Para entender qué es un **Retriever** tipo **Vectorstore**, primero debemos entender qué es un Vectorstore. Un Vectorstore es un tipo de base de datos especialmente diseñada para gestionar y manipular vectores de alta dimensionalidad, comúnmente utilizados para representar datos en aprendizaje automático y otras aplicaciones de inteligencia artificial.

En la analogía de la biblioteca mencionada anteriormente, si el Retriever es el bibliotecario, entonces el Vectorstore sería el sistema de clasificación y organización de la biblioteca que permite al bibliotecario encontrar exactamente lo que busca.

En LangChain, el sistema Vectorstore predeterminado que se utiliza es Chroma. Chroma se utiliza para indexar y buscar embeddings (vectores que representan documentos en el espacio multidimensional). Estos embeddings son una forma de condensar y representar la información de un documento para que pueda ser fácilmente comparable con otros documentos.

El Retriever tipo Vectorstore, por lo tanto, es un tipo de Retriever que utiliza una base de datos Vectorstore (como Chroma) para localizar documentos relevantes para una consulta específica. Primero transforma la consulta en un vector (a través de un proceso de incrustación (embedding)), luego busca en la base de datos Vectorstore los documentos cuyos vectores son más cercanos (en términos de distancia coseno u otras métricas de similitud) a la consulta vectorizada.


Vamos a empezar por instalar y observar qué versión de LangChain tenemos disponible:

```bash
pip install langchain
pip show langchain
```
Respuesta esperada:
```commandline
Name: langchain
Version: 0.0.245
Summary: Building applications with LLMs through composability
Home-page: https://www.github.com/hwchase17/langchain
Author: 
Author-email: 
License: MIT
Location: /home/ichcanziho/Documentos/programacion/Deep Learnining/venv/lib/python3.10/site-packages
Requires: aiohttp, async-timeout, dataclasses-json, langsmith, numexpr, numpy, openapi-schema-pydantic, pydantic, PyYAML, requests, SQLAlchemy, tenacity
Required-by: 
```

### 1. La clase Document

Esta clase es la base de cuando carguemos nuestros documentos. En LangChain se les llama schemas a estas clases base y se encuentran en langchain.schema. Así es el schema para Document:

```
class Document(Serializable):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)
```

> ## Nota:
> El código esta en: [7_clase_document.py](scripts%2F7_clase_document.py)

Los metadatos son información adicional que puede acompañar al texto. Puede ser el nombre del autor, fecha de publicación, lenguaje etc.
```python
from langchain.schema import Document

page_content = "Textooooooooolargoooooo ejemplo"
metadata = {'fuente': 'platzi', 'clase': 'langchain'}

doc = Document(
    page_content=page_content, metadata=metadata
)

print(doc.page_content)
print(doc)
```
Respuesta esperada:
```commandline
Textooooooooolargoooooo ejemplo
page_content='Textooooooooolargoooooo ejemplo' metadata={'fuente': 'platzi', 'clase': 'langchain'}
```
## 3.3 Document Loaders: PDF

La primera etapa en la indexación de documentos en LangChain implica cargar los datos en "Documentos". Este es el nombre de la clase con la que trabajaremos, ubicada en el directorio de esquemas en el repositorio de LangChain. Simplificando, un "Documento" es básicamente un fragmento de texto. El propósito del cargador de documentos es simplificar este proceso de carga.

### Document transformers

Los transformadores de carga son utilidades que convierten los datos desde un formato específico al formato "Documento". Por ejemplo, existen transformadores para los formatos CSV y SQL. En su mayoría, estos cargadores obtienen datos de archivos, pero a veces también de URLs.

Existen varios cargadores de documentos dependiendo de la fuente de nuestros datos. A continuación, se muestran algunos ejemplos (para más información, consulta la documentación):

- Airtable
- OpenAIWhisperParser
- CoNLL-U
- Copy Paste
- CSV
- Email
- EPub
- EverNote
- Microsoft Excel
- Facebook Chat
- File Directory
- HTML
- Images
- Jupyter Notebook
- JSON
- Markdown
- Microsoft PowerPoint
- Microsoft Word
- Open Document Format (ODT)
- Pandas DataFrame
- PDF

Al mismo tiempo, también puedes utilizar servicios como los datasets de Hugging Face, o incluso obtener datos de servicios como Slack, Snowflake, Spreedly, Stripe, 2Markdown, entre otros.

Cada mes se añaden nuevas fuentes y tipos de conjuntos de datos que podemos utilizar. Te recomendamos revisar la documentación con regularidad para mantenerte actualizado.

Vamos a utilizar el mismo documento que usamos en [Cadenas en langchin](#16-cadenas-en-langchain) ya lo hemos descargado en [public_key_cryptography.pdf](scripts%2Fpublic_key_cryptography.pdf)

> ## Nota:
> El código de esta sección está en [8_document_loaders.py](scripts%2F8_document_loaders.py)

Quizás el Document Loader más relevante es el unstructured pues se encuentra como la base de otros Document Loaders. Sirve por ejemplo para documentos de texto como .txt o .pdf.

```python
from langchain.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader("./public_key_cryptography.pdf")
data = loader.load()

print("tipo:", type(data), "tamaño:", len(data))
print("Ejemplo Metadata")
print(data[0].metadata)
print("Ejemplo Content")
print(data[0].page_content[:300])
```
Respuesta esperada:
```commandline
tipo: <class 'list'> tamaño: 1
Ejemplo Metadata
{'source': './public_key_cryptography.pdf'}
Ejemplo Content
The First Ten Years of Public-Key Cryptography

WH lTFl ELD DI FFlE

Invited Paper

Public-key cryptosystems separate the capacities for encryption and decryption so that 7) many people can encrypt messages in such a way that only one person can read them, or 2) one person can encrypt messages in su
```

Podemos observar como este Document es una lista de un solo elemento, y contiene todas las páginas del PDF en un solo compendio. Sin embargo,
también hay formas de que cada página sea un Document por sí mismo:

```python
from langchain.document_loaders import PyPDFLoader

print("*"*64)
loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()
print("tipo:", type(data), "tamaño:", len(data))
print("Ejemplo Metadata")
print(data[0].metadata)
print("Ejemplo Content")
print(data[0].page_content[:300])
```
Respuesta esperada:
```commandline
tipo: <class 'list'> tamaño: 18
Ejemplo Metadata
{'source': './public_key_cryptography.pdf', 'page': 0}
Ejemplo Content
The First Ten Years of Public-Key 
Cryptography 
WH lTFl ELD DI FFlE 
Invited Paper 
Public-key cryptosystems separate the capacities for encryption 
and decryption so that 7) many people can encrypt messages in 
such a way that only one person can read them, or 2) one person 
can encrypt messages i
```
Podemos observar como de igual manera es una lista, pero que contiene 18 elementos, en dónde cada elemento cuenta con su 
propio source y como metadata tiene el número de página. Adicionalmente, vemos como ha conservado el formato de columna del 
paper original.


## 3.4 Document Loaders: CSV con Pandas DataFrames

Uno de los usos más frecuentes de la biblioteca Pandas es la lectura de datos a partir de archivos CSV o hojas de cálculo (como Excel). Muchas empresas almacenan sus datos en estos formatos. En esta clase aprenderás cómo leer un archivo CSV y convertirlo en un DataFrame de Pandas, para después convertirlo a un Document de LangChain.

Para el seguimiento de la clase utiliza la Notebook 2 del curso en la sección CSV a Pandas DataFrame.

> ## Nota:
> El código completo está en: [9_csv_loader.py](scripts%2F9_csv_loader.py)

### Instalación de paquetes:

Primero, necesitaremos instalar la biblioteca de Pandas, si aún no está instalada en tu entorno. También vamos a instalar gdown para descargar docs de google drive:

```bash
pip install pandas
pip install gdown
```

Una vez que tengas instalada la biblioteca Pandas, podemos empezar a leer archivos CSV. Para esto, utilizaremos el método read_csv() proporcionado por Pandas.
Primero vamos a usar `gdown` para descargar el archivo csv:

```python
import gdown

file_url = 'https://drive.google.com/uc?id=1kihb-PiE0jLnlJicZ42yDCIpTo_D40Zc'
output_file = 'data/repos_cairo.csv'
gdown.download(file_url, output_file, quiet=False)
print(f"Archivo descargado como '{output_file}'")
```
Respuesta esperada:
```commandline
Downloading...
From: https://drive.google.com/uc?id=1kihb-PiE0jLnlJicZ42yDCIpTo_D40Zc
To: /home/ichcanziho/Documentos/programacion/Deep Learnining/13 Curso de LangChain/scripts/data/repos_cairo.csv
100%|██████████| 2.22k/2.22k [00:00<00:00, 18.3MB/s]
```

Podemos leer el archivo de la siguiente manera:

```python
import pandas as pd

df = pd.read_csv('data/repos_cairo.csv')
print(df.head())
```
Respuesta esperada:
```commandline
                          repo_name  ... repo_forks
0                 kkrt-labs/kakarot  ...         93
1                 ZeroSync/ZeroSync  ...         29
2   starknet-edu/starknet-cairo-101  ...        131
3         shramee/starklings-cairo1  ...        101
4  keep-starknet-strange/alexandria  ...         42

[5 rows x 6 columns]
```

A continuación, vamos a utilizar la clase DataFrameLoader de LangChain para cargar los datos de nuestro DataFrame.

```python
from langchain.document_loaders import DataFrameLoader

loader = DataFrameLoader(df, page_content_column="repo_name")
data = loader.load()
```
Finalmente, vamos a imprimir el tipo y la longitud de nuestros datos cargados.
```python
print(f"El archivo es de tipo {type(data)} y tiene una longitud de {len(data)} debido a la cantidad de observaciones en el CSV.")
```
Respuesta esperada:
```commandline
El archivo es de tipo <class 'list'> y tiene una longitud de 25 debido a la cantidad de observaciones en el CSV.
```

Y vamos a imprimir las primeras 5 líneas de nuestros datos.

```python
from pprint import pprint

pprint(data[:5])
```
Respuesta esperada:
```commandline
[Document(page_content='kkrt-labs/kakarot', metadata={'repo_owner': 'kkrt-labs', 'repo_updated_at': '2023-06-10T16:12:50Z', 'repo_created_at': '2022-10-04T14:33:18Z', 'repo_stargazers_count': 453, 'repo_forks': 93}),
 Document(page_content='ZeroSync/ZeroSync', metadata={'repo_owner': 'ZeroSync', 'repo_updated_at': '2023-06-09T15:19:11Z', 'repo_created_at': '2022-07-08T14:56:27Z', 'repo_stargazers_count': 290, 'repo_forks': 29}),
 Document(page_content='starknet-edu/starknet-cairo-101', metadata={'repo_owner': 'starknet-edu', 'repo_updated_at': '2023-06-07T23:08:37Z', 'repo_created_at': '2022-07-05T15:00:25Z', 'repo_stargazers_count': 259, 'repo_forks': 131}),
 Document(page_content='shramee/starklings-cairo1', metadata={'repo_owner': 'shramee', 'repo_updated_at': '2023-06-09T13:06:27Z', 'repo_created_at': '2023-01-05T10:04:40Z', 'repo_stargazers_count': 249, 'repo_forks': 101}),
 Document(page_content='keep-starknet-strange/alexandria', metadata={'repo_owner': 'keep-starknet-strange', 'repo_updated_at': '2023-06-09T09:17:59Z', 'repo_created_at': '2022-11-25T08:26:42Z', 'repo_stargazers_count': 115, 'repo_forks': 42})]
```

Como ves es bastante similar a cargar un PDF a LangChain. Te recuerdo que en la [documentación de Document Loaders de LangChain](https://python.langchain.com/docs/modules/data_connection/document_loaders/), puedes ver todas las integraciones para cargar diferentes tipos de archivos.


## 3.5 Document Loaders: JSONL

langChain nos permite cargar diferentes formatos de documentos, desde PDFs, CSVs, WORDs, etc. Sin embargo, qué tal si tus
datos NO está soportado directamente por LangChain o quizá simplemente quieres guardar la información de `metadata` de una forma
muy concreta que LangChain no soporta aún. Para estos casos podemos crear nuestros propios `Document Loaders`. En este ejercicio
veremos como leer un archivo en formato `JSONL` que aún no tiene soporte oficial de LangChain.

Veamos un caso más complejo. No tenemos una implementación directa de LangChain para importar JSONLs sin embargo es muy común tener que importar estos formatos.

El siguiente ejemplo muestra cómo importar un JSONL personalizado para nuestra base de datos de Transformers, pero aplica para otros formatos de datos que no necesariamente se encuentran entre los disponibles por LangChain. Nosotros creamos nuestros Document según lo que queramos asignar como page_content y metadata.

Empezamos instalando un paquete para leer jsonlines:

```bash
pip install jsonlines
```

Empecemos por conocer el documento que queremos cargar: [transformers_docs.jsonl](scripts%2Fdata%2Ftransformers_docs.jsonl)

```json
{
  "title": "accelerate.mdx", 
  "repo_owner": "huggingface", 
  "repo_name": "transformers", 
  "text": "<!--Copyright 2022 The HuggingFace Team. All rights resng larger models on limited ..."
}
```

Tiene 4 keys: `Title`, `Repo Owner`, `Repo Name` y `Text` es fácil observar que `Text` sería el `Page Content` y que todo lo demás sería
información de la `metadata`.

Vamos a construir nuestra propia clase `TransformerDocsJSONLLoader`

```python
from langchain.schema import Document
import jsonlines
from typing import List  # solo será útil para poner un formato de salida genérico


class TransformerDocsJSONLLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
```

Para este momento nuestra clase solo recibe como parámetro el path en donde se encuentra el archivo JSONL. Ahora vamos a implementar
el método `load` que nos permitirá cargar nuestros datos de forma sencilla:

```python
    def load(self) -> List[Document]:  # Solo ayuda a decir que la salida es una lista cuyos elementos son Documents
        with jsonlines.open(self.file_path) as reader:
            documents = []
            # Para cada entrada del documento JSONL
            for obj in reader:
                page_content = obj.get("text", "")
                metadata = {
                    'title': obj.get("title", ""),
                    'repo_owner': obj.get("repo_owner", ""),
                    'repo_name': obj.get("repo_name", ""),
                }
                # Añadimos el page content como text y metadata con title, repo_owner y repo_name
                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )
        return documents
```

Ahora vamos a poner a prueba nuestra clase instanciando un objeto:

```python
loader = TransformerDocsJSONLLoader("data/transformers_docs.jsonl")
data = loader.load()

for doc in data:
    print(doc)

```
Respuesta esperada:
```commandline
page_content='<!--Copyright 2022 The HuggingFace Team. All rights reserved.\n\nLicensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with\nthe License. You may obtain a copy of the License at\n\nhttp://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software distributed under the License is distributed on\nan "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the\nspecific language governing permissions and limitations under the License.\n-->\n\n# Distributed training with 🤗 Accelerate\n\nAs models get bigger, parallelism has emerged as a strategy for training larger models on limited hardware and accelerating training speed by several orders of magnitude. At Hugging Face, we created the [🤗 Accelerate](https://huggingface.co/docs/accelerate) library to help users easily train a 🤗 Transformers model on any type of distributed setup, whether it is multiple GPU\'s on one machine or multiple GPU\'s across several machines. In this tutorial, learn how to customize your native PyTorch training loop to enable training in a distributed environment.\n\n## Setup\n\nGet started by installing 🤗 Accelerate:\n\n```bash\npip install accelerate\n```\n\nThen import and create an [`~accelerate.Accelerator`] object. The [`~accelerate.Accelerator`] will automatically detect your type of distributed setup and initialize all the necessary components for training. You don\'t need to explicitly place your model on a device.\n\n```py\n>>> from accelerate import Accelerator\n\n>>> accelerator = Accelerator()\n```\n\n## Prepare to accelerate\n\nThe next step is to pass all the relevant training objects to the [`~accelerate.Accelerator.prepare`] method. This includes your training and evaluation DataLoaders, a model and an optimizer:\n\n```py\n>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(\n...     train_dataloader, eval_dataloader, model, optimizer\n... )\n```\n\n## Backward\n\nThe last addition is to replace the typical `loss.backward()` in your training loop with 🤗 Accelerate\'s [`~accelerate.Accelerator.backward`]method:\n\n```py\n>>> for epoch in range(num_epochs):\n...     for batch in train_dataloader:\n...         outputs = model(**batch)\n...         loss = outputs.loss\n...         accelerator.backward(loss)\n\n...         optimizer.step()\n...         lr_scheduler.step()\n...         optimizer.zero_grad()\n...         progress_bar.update(1)\n```\n\nAs you can see in the following code, you only need to add four additional lines of code to your training loop to enable distributed training!\n\n```diff\n+ from accelerate import Accelerator\n  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler\n\n+ accelerator = Accelerator()\n\n  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)\n  optimizer = AdamW(model.parameters(), lr=3e-5)\n\n- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")\n- model.to(device)\n\n+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(\n+     train_dataloader, eval_dataloader, model, optimizer\n+ )\n\n  num_epochs = 3\n  num_training_steps = num_epochs * len(train_dataloader)\n  lr_scheduler = get_scheduler(\n      "linear",\n      optimizer=optimizer,\n      num_warmup_steps=0,\n      num_training_steps=num_training_steps\n  )\n\n  progress_bar = tqdm(range(num_training_steps))\n\n  model.train()\n  for epoch in range(num_epochs):\n      for batch in train_dataloader:\n-         batch = {k: v.to(device) for k, v in batch.items()}\n          outputs = model(**batch)\n          loss = outputs.loss\n-         loss.backward()\n+         accelerator.backward(loss)\n\n          optimizer.step()\n          lr_scheduler.step()\n          optimizer.zero_grad()\n          progress_bar.update(1)\n```\n\n## Train\n\nOnce you\'ve added the relevant lines of code, launch your training in a script or a notebook like Colaboratory.\n\n### Train with a script\n\nIf you are running your training from a script, run the following command to create and save a configuration file:\n\n```bash\naccelerate config\n```\n\nThen launch your training with:\n\n```bash\naccelerate launch train.py\n```\n\n### Train with a notebook\n\n🤗 Accelerate can also run in a notebook if you\'re planning on using Colaboratory\'s TPUs. Wrap all the code responsible for training in a function, and pass it to [`~accelerate.notebook_launcher`]:\n\n```py\n>>> from accelerate import notebook_launcher\n\n>>> notebook_launcher(training_function)\n```\n\nFor more information about 🤗 Accelerate and it\'s rich features, refer to the [documentation](https://huggingface.co/docs/accelerate).' metadata={'title': 'accelerate.mdx', 'repo_owner': 'huggingface', 'repo_name': 'transformers'}
page_content='<!--Copyright 2020 The HuggingFace Team. All rights reserved.\n\nLicensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with\nthe License. You may obtain a copy of the License at\n\nhttp://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software distributed under the License is distributed on\nan "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the\n-->\n\n# How to add a model to 🤗 Transformers?\n\nThe 🤗 Transformers library is often able to offer new models thanks to community contributors. But this can be a challenging project and requires an in-depth knowledge of the 🤗 Transformers library and the model to implement. At Hugging Face, we\'re trying to empower more of the community to actively add models and we\'ve put together this guide to walk you through the process of adding a PyTorch model (make sure you have [PyTorch installed](https://pytorch.org/get-started/locally/)).\n\n<Tip>\n\nIf you\'re interested in implementing a TensorFlow model, take a look at the [How to convert a 🤗 Transformers model to TensorFlow](add_tensorflow_model) guide!\n\n</Tip>\n\nAlong the way, you\'ll:\n\n- get insights into open-source best practices\n- understand the design principles behind one of the most popular deep learning libraries\n- learn how to efficiently test large models\n- learn how to integrate Python utilities like `black`, `ruff`, and `make fix-copies` to ensure clean and readable code\n\nA Hugging Face team member will be available to help you along the way so you\'ll never be alone. 🤗 ❤️\n\nTo get started, open a [New model addition](https://github.com/huggingface/transformers/issues/new?assignees=&labels=New+model&template=new-model-addition.yml) issue for the model you want to see in 🤗 Transformers. If you\'re not especially picky about contributing a specific model, you can filter by the [New model label](https://github.com/huggingface/transformers/labels/New%20model) to see if there are any unclaimed model requests and work on it.\n\nOnce you\'ve opened a new model request, the first step is to get familiar with 🤗 Transformers if you aren\'t already!\n\n## General overview of 🤗 Transformers\n\nFirst, you should get a general overview of 🤗 Transformers. 🤗 Transformers is a very opinionated library, so there is a\nchance that you don\'t agree with some of the library\'s philosophies or design choices. From our experience, however, we\nfound that the fundamental design choices and philosophies of the library are crucial to efficiently scale 🤗\nTransformers while keeping maintenance costs at a reasonable level.\n\nA good first starting point to better understand the library is to read the [documentation of our philosophy](philosophy). As a result of our way of working, there are some choices that we try to apply to all models:\n\n- Composition is generally favored over-abstraction\n- Duplicating code is not always bad if it strongly improves the readability or accessibility of a model\n- Model files are as self-contained as possible so that when you read the code of a specific model, you ideally only\n  have to look into the respective `modeling_....py` file.\n\nIn our opinion, the library\'s code is not just a means to provide a product, *e.g.* the ability to use BERT for\ninference, but also as the very product that we want to improve. Hence, when adding a model, the user is not only the\nperson that will use your model, but also everybody that will read, try to understand, and possibly tweak your code.\n\nWith this in mind, let\'s go a bit deeper into the general library design.\n\n### Overview of models\n\nTo successfully add a model, it is important to understand the interaction between your model and its config,\n[`PreTrainedModel`], and [`PretrainedConfig`]. For exemplary purposes, we will\ncall the model to be added to 🤗 Transformers `BrandNewBert`.\n\nLet\'s take a look:\n\n<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_overview.png"/>\n\nAs you can see, we do make use of inheritance in 🤗 Transformers, but we keep the level of abstraction to an absolute\nminimum. There are never more than two levels of abstraction for any model in the library. `BrandNewBertModel`\ninherits from `BrandNewBertPreTrainedModel` which in turn inherits from [`PreTrainedModel`] and\nthat\'s it. As a general rule, we want to make sure that a new model only depends on\n[`PreTrainedModel`]. The important functionalities that are automatically provided to every new\nmodel are [`~PreTrainedModel.from_pretrained`] and\n[`~PreTrainedModel.save_pretrained`], which are used for serialization and deserialization. All of the\nother important functionalities, such as `BrandNewBertModel.forward` should be completely defined in the new\n`modeling_brand_new_bert.py` script. Next, we want to make sure that a model with a specific head layer, such as\n`BrandNewBertForMaskedLM` does not inherit from `BrandNewBertModel`, but rather uses `BrandNewBertModel`\nas a component that can be called in its forward pass to keep the level of abstraction low. Every new model requires a\nconfiguration class, called `BrandNewBertConfig`. This configuration is always stored as an attribute in\n[`PreTrainedModel`], and thus can be accessed via the `config` attribute for all classes\ninheriting from `BrandNewBertPreTrainedModel`:\n\n```python\nmodel = BrandNewBertModel.from_pretrained("brandy/brand_new_bert")\nmodel.config  # model has access to its config\n```\n\nSimilar to the model, the configuration inherits basic serialization and deserialization functionalities from\n[`PretrainedConfig`]. Note that the configuration and the model are always serialized into two\ndifferent formats - the model to a *pytorch_model.bin* file and the configuration to a *config.json* file. Calling\n[`~PreTrainedModel.save_pretrained`] will automatically call\n[`~PretrainedConfig.save_pretrained`], so that both model and configuration are saved.\n\n\n### Code style\n\nWhen coding your new model, keep in mind that Transformers is an opinionated library and we have a few quirks of our\nown regarding how code should be written :-)\n\n1. The forward pass of your model should be fully written in the modeling file while being fully independent of other\n   models in the library. If you want to reuse a block from another model, copy the code and paste it with a\n   `# Copied from` comment on top (see [here](https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/roberta/modeling_roberta.py#L160)\n   for a good example).\n2. The code should be fully understandable, even by a non-native English speaker. This means you should pick\n   descriptive variable names and avoid abbreviations. As an example, `activation` is preferred to `act`.\n   One-letter variable names are strongly discouraged unless it\'s an index in a for loop.\n3. More generally we prefer longer explicit code to short magical one.\n4. Avoid subclassing `nn.Sequential` in PyTorch but subclass `nn.Module` and write the forward pass, so that anyone\n   using your code can quickly debug it by adding print statements or breaking points.\n5. Your function signature should be type-annotated. For the rest, good variable names are way more readable and\n   understandable than type annotations.\n\n### Overview of tokenizers\n\nNot quite ready yet :-( This section will be added soon!\n\n## Step-by-step recipe to add a model to 🤗 Transformers\n\nEveryone has different preferences of how to port a model so it can be very helpful for you to take a look at summaries\nof how other contributors ported models to Hugging Face. Here is a list of community blog posts on how to port a model:\n\n1. [Porting GPT2 Model](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28) by [Thomas](https://huggingface.co/thomwolf)\n2. [Porting WMT19 MT Model](https://huggingface.co/blog/porting-fsmt) by [Stas](https://huggingface.co/stas)\n\nFrom experience, we can tell you that the most important things to keep in mind when adding a model are:\n\n-  Don\'t reinvent the wheel! Most parts of the code you will add for the new 🤗 Transformers model already exist\n  somewhere in 🤗 Transformers. Take some time to find similar, already existing models and tokenizers you can copy\n  from. [grep](https://www.gnu.org/software/grep/) and [rg](https://github.com/BurntSushi/ripgrep) are your\n  friends. Note that it might very well happen that your model\'s tokenizer is based on one model implementation, and\n  your model\'s modeling code on another one. *E.g.* FSMT\'s modeling code is based on BART, while FSMT\'s tokenizer code\n  is based on XLM.\n-  It\'s more of an engineering challenge than a scientific challenge. You should spend more time on creating an\n  efficient debugging environment than trying to understand all theoretical aspects of the model in the paper.\n-  Ask for help, when you\'re stuck! Models are the core component of 🤗 Transformers so that we at Hugging Face are more\n  than happy to help you at every step to add your model. Don\'t hesitate to ask if you notice you are not making\n  progress.\n\nIn the following, we try to give you a general recipe that we found most useful when porting a model to 🤗 Transformers.\n\nThe following list is a summary of everything that has to be done to add a model and can be used by you as a To-Do\nList:\n\n☐ (Optional) Understood the model\'s theoretical aspects<br>\n☐ Prepared 🤗 Transformers dev environment<br>\n☐ Set up debugging environment of the original repository<br>\n☐ Created script that successfully runs the `forward()` pass using the original repository and checkpoint<br>\n☐ Successfully added the model skeleton to 🤗 Transformers<br>\n☐ Successfully converted original checkpoint to 🤗 Transformers checkpoint<br>\n☐ Successfully ran `forward()` pass in 🤗 Transformers that gives identical output to original checkpoint<br>\n☐ Finished model tests in 🤗 Transformers<br>\n☐ Successfully added tokenizer in 🤗 Transformers<br>\n☐ Run end-to-end integration tests<br>\n☐ Finished docs<br>\n☐ Uploaded model weights to the Hub<br>\n☐ Submitted the pull request<br>\n☐ (Optional) Added a demo notebook\n\nTo begin with, we usually recommend to start by getting a good theoretical understanding of `BrandNewBert`. However,\nif you prefer to understand the theoretical aspects of the model *on-the-job*, then it is totally fine to directly dive\ninto the `BrandNewBert`\'s code-base. This option might suit you better, if your engineering skills are better than\nyour theoretical skill, if you have trouble understanding `BrandNewBert`\'s paper, or if you just enjoy programming\nmuch more than reading scientific papers.\n\n### 1. (Optional) Theoretical aspects of BrandNewBert\n\nYou should take some time to read *BrandNewBert\'s* paper, if such descriptive work exists. There might be large\nsections of the paper that are difficult to understand. If this is the case, this is fine - don\'t worry! The goal is\nnot to get a deep theoretical understanding of the paper, but to extract the necessary information required to\neffectively re-implement the model in 🤗 Transformers. That being said, you don\'t have to spend too much time on the\ntheoretical aspects, but rather focus on the practical ones, namely:\n\n-  What type of model is *brand_new_bert*? BERT-like encoder-only model? GPT2-like decoder-only model? BART-like\n  encoder-decoder model? Look at the [model_summary](model_summary) if you\'re not familiar with the differences between those.\n-  What are the applications of *brand_new_bert*? Text classification? Text generation? Seq2Seq tasks, *e.g.,*\n  summarization?\n-  What is the novel feature of the model making it different from BERT/GPT-2/BART?\n-  Which of the already existing [🤗 Transformers models](https://huggingface.co/transformers/#contents) is most\n  similar to *brand_new_bert*?\n-  What type of tokenizer is used? A sentencepiece tokenizer? Word piece tokenizer? Is it the same tokenizer as used\n  for BERT or BART?\n\nAfter you feel like you have gotten a good overview of the architecture of the model, you might want to write to the\nHugging Face team with any questions you might have. This might include questions regarding the model\'s architecture,\nits attention layer, etc. We will be more than happy to help you.\n\n### 2. Next prepare your environment\n\n1. Fork the [repository](https://github.com/huggingface/transformers) by clicking on the ‘Fork\' button on the\n   repository\'s page. This creates a copy of the code under your GitHub user account.\n\n2. Clone your `transformers` fork to your local disk, and add the base repository as a remote:\n\n```bash\ngit clone https://github.com/[your Github handle]/transformers.git\ncd transformers\ngit remote add upstream https://github.com/huggingface/transformers.git\n```\n\n3. Set up a development environment, for instance by running the following command:\n\n```bash\npython -m venv .env\nsource .env/bin/activate\npip install -e ".[dev]"\n```\n\nDepending on your OS, and since the number of optional dependencies of Transformers is growing, you might get a\nfailure with this command. If that\'s the case make sure to install the Deep Learning framework you are working with\n(PyTorch, TensorFlow and/or Flax) then do:\n\n```bash\npip install -e ".[quality]"\n```\n\nwhich should be enough for most use cases. You can then return to the parent directory\n\n```bash\ncd ..\n```\n\n4. We recommend adding the PyTorch version of *brand_new_bert* to Transformers. To install PyTorch, please follow the\n   instructions on https://pytorch.org/get-started/locally/.\n\n**Note:** You don\'t need to have CUDA installed. Making the new model work on CPU is sufficient.\n\n5. To port *brand_new_bert*, you will also need access to its original repository:\n\n```bash\ngit clone https://github.com/org_that_created_brand_new_bert_org/brand_new_bert.git\ncd brand_new_bert\npip install -e .\n```\n\nNow you have set up a development environment to port *brand_new_bert* to 🤗 Transformers.\n\n### 3.-4. Run a pretrained checkpoint using the original repository\n\nAt first, you will work on the original *brand_new_bert* repository. Often, the original implementation is very\n“researchy”. Meaning that documentation might be lacking and the code can be difficult to understand. But this should\nbe exactly your motivation to reimplement *brand_new_bert*. At Hugging Face, one of our main goals is to *make people\nstand on the shoulders of giants* which translates here very well into taking a working model and rewriting it to make\nit as **accessible, user-friendly, and beautiful** as possible. This is the number-one motivation to re-implement\nmodels into 🤗 Transformers - trying to make complex new NLP technology accessible to **everybody**.\n\nYou should start thereby by diving into the original repository.\n\nSuccessfully running the official pretrained model in the original repository is often **the most difficult** step.\nFrom our experience, it is very important to spend some time getting familiar with the original code-base. You need to\nfigure out the following:\n\n- Where to find the pretrained weights?\n- How to load the pretrained weights into the corresponding model?\n- How to run the tokenizer independently from the model?\n- Trace one forward pass so that you know which classes and functions are required for a simple forward pass. Usually,\n  you only have to reimplement those functions.\n- Be able to locate the important components of the model: Where is the model\'s class? Are there model sub-classes,\n  *e.g.* EncoderModel, DecoderModel? Where is the self-attention layer? Are there multiple different attention layers,\n  *e.g.* *self-attention*, *cross-attention*...?\n- How can you debug the model in the original environment of the repo? Do you have to add *print* statements, can you\n  work with an interactive debugger like *ipdb*, or should you use an efficient IDE to debug the model, like PyCharm?\n\nIt is very important that before you start the porting process, that you can **efficiently** debug code in the original\nrepository! Also, remember that you are working with an open-source library, so do not hesitate to open an issue, or\neven a pull request in the original repository. The maintainers of this repository are most likely very happy about\nsomeone looking into their code!\n\nAt this point, it is really up to you which debugging environment and strategy you prefer to use to debug the original\nmodel. We strongly advise against setting up a costly GPU environment, but simply work on a CPU both when starting to\ndive into the original repository and also when starting to write the 🤗 Transformers implementation of the model. Only\nat the very end, when the model has already been successfully ported to 🤗 Transformers, one should verify that the\nmodel also works as expected on GPU.\n\nIn general, there are two possible debugging environments for running the original model\n\n-  [Jupyter notebooks](https://jupyter.org/) / [google colab](https://colab.research.google.com/notebooks/intro.ipynb)\n-  Local python scripts.\n\nJupyter notebooks have the advantage that they allow for cell-by-cell execution which can be helpful to better split\nlogical components from one another and to have faster debugging cycles as intermediate results can be stored. Also,\nnotebooks are often easier to share with other contributors, which might be very helpful if you want to ask the Hugging\nFace team for help. If you are familiar with Jupyter notebooks, we strongly recommend you to work with them.\n\nThe obvious disadvantage of Jupyter notebooks is that if you are not used to working with them you will have to spend\nsome time adjusting to the new programming environment and that you might not be able to use your known debugging tools\nanymore, like `ipdb`.\n\nFor each code-base, a good first step is always to load a **small** pretrained checkpoint and to be able to reproduce a\nsingle forward pass using a dummy integer vector of input IDs as an input. Such a script could look like this (in\npseudocode):\n\n```python\nmodel = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")\ninput_ids = [0, 4, 5, 2, 3, 7, 9]  # vector of input ids\noriginal_output = model.predict(input_ids)\n```\n\nNext, regarding the debugging strategy, there are generally a few from which to choose from:\n\n- Decompose the original model into many small testable components and run a forward pass on each of those for\n  verification\n- Decompose the original model only into the original *tokenizer* and the original *model*, run a forward pass on\n  those, and use intermediate print statements or breakpoints for verification\n\nAgain, it is up to you which strategy to choose. Often, one or the other is advantageous depending on the original code\nbase.\n\nIf the original code-base allows you to decompose the model into smaller sub-components, *e.g.* if the original\ncode-base can easily be run in eager mode, it is usually worth the effort to do so. There are some important advantages\nto taking the more difficult road in the beginning:\n\n- at a later stage when comparing the original model to the Hugging Face implementation, you can verify automatically\n  for each component individually that the corresponding component of the 🤗 Transformers implementation matches instead\n  of relying on visual comparison via print statements\n- it can give you some rope to decompose the big problem of porting a model into smaller problems of just porting\n  individual components and thus structure your work better\n- separating the model into logical meaningful components will help you to get a better overview of the model\'s design\n  and thus to better understand the model\n- at a later stage those component-by-component tests help you to ensure that no regression occurs as you continue\n  changing your code\n\n[Lysandre\'s](https://gist.github.com/LysandreJik/db4c948f6b4483960de5cbac598ad4ed) integration checks for ELECTRA\ngives a nice example of how this can be done.\n\nHowever, if the original code-base is very complex or only allows intermediate components to be run in a compiled mode,\nit might be too time-consuming or even impossible to separate the model into smaller testable sub-components. A good\nexample is [T5\'s MeshTensorFlow](https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow) library which is\nvery complex and does not offer a simple way to decompose the model into its sub-components. For such libraries, one\noften relies on verifying print statements.\n\nNo matter which strategy you choose, the recommended procedure is often the same in that you should start to debug the\nstarting layers first and the ending layers last.\n\nIt is recommended that you retrieve the output, either by print statements or sub-component functions, of the following\nlayers in the following order:\n\n1. Retrieve the input IDs passed to the model\n2. Retrieve the word embeddings\n3. Retrieve the input of the first Transformer layer\n4. Retrieve the output of the first Transformer layer\n5. Retrieve the output of the following n - 1 Transformer layers\n6. Retrieve the output of the whole BrandNewBert Model\n\nInput IDs should thereby consists of an array of integers, *e.g.* `input_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]`\n\nThe outputs of the following layers often consist of multi-dimensional float arrays and can look like this:\n\n```\n[[\n [-0.1465, -0.6501,  0.1993,  ...,  0.1451,  0.3430,  0.6024],\n [-0.4417, -0.5920,  0.3450,  ..., -0.3062,  0.6182,  0.7132],\n [-0.5009, -0.7122,  0.4548,  ..., -0.3662,  0.6091,  0.7648],\n ...,\n [-0.5613, -0.6332,  0.4324,  ..., -0.3792,  0.7372,  0.9288],\n [-0.5416, -0.6345,  0.4180,  ..., -0.3564,  0.6992,  0.9191],\n [-0.5334, -0.6403,  0.4271,  ..., -0.3339,  0.6533,  0.8694]]],\n```\n\nWe expect that every model added to 🤗 Transformers passes a couple of integration tests, meaning that the original\nmodel and the reimplemented version in 🤗 Transformers have to give the exact same output up to a precision of 0.001!\nSince it is normal that the exact same model written in different libraries can give a slightly different output\ndepending on the library framework, we accept an error tolerance of 1e-3 (0.001). It is not enough if the model gives\nnearly the same output, they have to be the almost identical. Therefore, you will certainly compare the intermediate\noutputs of the 🤗 Transformers version multiple times against the intermediate outputs of the original implementation of\n*brand_new_bert* in which case an **efficient** debugging environment of the original repository is absolutely\nimportant. Here is some advice is to make your debugging environment as efficient as possible.\n\n- Find the best way of debugging intermediate results. Is the original repository written in PyTorch? Then you should\n  probably take the time to write a longer script that decomposes the original model into smaller sub-components to\n  retrieve intermediate values. Is the original repository written in Tensorflow 1? Then you might have to rely on\n  TensorFlow print operations like [tf.print](https://www.tensorflow.org/api_docs/python/tf/print) to output\n  intermediate values. Is the original repository written in Jax? Then make sure that the model is **not jitted** when\n  running the forward pass, *e.g.* check-out [this link](https://github.com/google/jax/issues/196).\n- Use the smallest pretrained checkpoint you can find. The smaller the checkpoint, the faster your debug cycle\n  becomes. It is not efficient if your pretrained model is so big that your forward pass takes more than 10 seconds.\n  In case only very large checkpoints are available, it might make more sense to create a dummy model in the new\n  environment with randomly initialized weights and save those weights for comparison with the 🤗 Transformers version\n  of your model\n- Make sure you are using the easiest way of calling a forward pass in the original repository. Ideally, you want to\n  find the function in the original repository that **only** calls a single forward pass, *i.e.* that is often called\n  `predict`, `evaluate`, `forward` or `__call__`. You don\'t want to debug a function that calls `forward`\n  multiple times, *e.g.* to generate text, like `autoregressive_sample`, `generate`.\n- Try to separate the tokenization from the model\'s *forward* pass. If the original repository shows examples where\n  you have to input a string, then try to find out where in the forward call the string input is changed to input ids\n  and start from this point. This might mean that you have to possibly write a small script yourself or change the\n  original code so that you can directly input the ids instead of an input string.\n- Make sure that the model in your debugging setup is **not** in training mode, which often causes the model to yield\n  random outputs due to multiple dropout layers in the model. Make sure that the forward pass in your debugging\n  environment is **deterministic** so that the dropout layers are not used. Or use *transformers.utils.set_seed*\n  if the old and new implementations are in the same framework.\n\nThe following section gives you more specific details/tips on how you can do this for *brand_new_bert*.\n\n### 5.-14. Port BrandNewBert to 🤗 Transformers\n\nNext, you can finally start adding new code to 🤗 Transformers. Go into the clone of your 🤗 Transformers\' fork:\n\n```bash\ncd transformers\n```\n\nIn the special case that you are adding a model whose architecture exactly matches the model architecture of an\nexisting model you only have to add a conversion script as described in [this section](#write-a-conversion-script).\nIn this case, you can just re-use the whole model architecture of the already existing model.\n\nOtherwise, let\'s start generating a new model. You have two choices here:\n\n- `transformers-cli add-new-model-like` to add a new model like an existing one\n- `transformers-cli add-new-model` to add a new model from our template (will look like BERT or Bart depending on the type of model you select)\n\nIn both cases, you will be prompted with a questionnaire to fill the basic information of your model. The second command requires to install `cookiecutter`, you can find more information on it [here](https://github.com/huggingface/transformers/tree/main/templates/adding_a_new_model).\n\n**Open a Pull Request on the main huggingface/transformers repo**\n\nBefore starting to adapt the automatically generated code, now is the time to open a “Work in progress (WIP)” pull\nrequest, *e.g.* “[WIP] Add *brand_new_bert*”, in 🤗 Transformers so that you and the Hugging Face team can work\nside-by-side on integrating the model into 🤗 Transformers.\n\nYou should do the following:\n\n1. Create a branch with a descriptive name from your main branch\n\n```bash\ngit checkout -b add_brand_new_bert\n```\n\n2. Commit the automatically generated code:\n\n```bash\ngit add .\ngit commit\n```\n\n3. Fetch and rebase to current main\n\n```bash\ngit fetch upstream\ngit rebase upstream/main\n```\n\n4. Push the changes to your account using:\n\n```bash\ngit push -u origin a-descriptive-name-for-my-changes\n```\n\n5. Once you are satisfied, go to the webpage of your fork on GitHub. Click on “Pull request”. Make sure to add the\n   GitHub handle of some members of the Hugging Face team as reviewers, so that the Hugging Face team gets notified for\n   future changes.\n\n6. Change the PR into a draft by clicking on “Convert to draft” on the right of the GitHub pull request web page.\n\nIn the following, whenever you have done some progress, don\'t forget to commit your work and push it to your account so\nthat it shows in the pull request. Additionally, you should make sure to update your work with the current main from\ntime to time by doing:\n\n```bash\ngit fetch upstream\ngit merge upstream/main\n```\n\nIn general, all questions you might have regarding the model or your implementation should be asked in your PR and\ndiscussed/solved in the PR. This way, the Hugging Face team will always be notified when you are committing new code or\nif you have a question. It is often very helpful to point the Hugging Face team to your added code so that the Hugging\nFace team can efficiently understand your problem or question.\n\nTo do so, you can go to the “Files changed” tab where you see all of your changes, go to a line regarding which you\nwant to ask a question, and click on the “+” symbol to add a comment. Whenever a question or problem has been solved,\nyou can click on the “Resolve” button of the created comment.\n\nIn the same way, the Hugging Face team will open comments when reviewing your code. We recommend asking most questions\non GitHub on your PR. For some very general questions that are not very useful for the public, feel free to ping the\nHugging Face team by Slack or email.\n\n**5. Adapt the generated models code for brand_new_bert**\n\nAt first, we will focus only on the model itself and not care about the tokenizer. All the relevant code should be\nfound in the generated files `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` and\n`src/transformers/models/brand_new_bert/configuration_brand_new_bert.py`.\n\nNow you can finally start coding :). The generated code in\n`src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` will either have the same architecture as BERT if\nit\'s an encoder-only model or BART if it\'s an encoder-decoder model. At this point, you should remind yourself what\nyou\'ve learned in the beginning about the theoretical aspects of the model: *How is the model different from BERT or\nBART?*". Implement those changes which often means to change the *self-attention* layer, the order of the normalization\nlayer, etc… Again, it is often useful to look at the similar architecture of already existing models in Transformers to\nget a better feeling of how your model should be implemented.\n\n**Note** that at this point, you don\'t have to be very sure that your code is fully correct or clean. Rather, it is\nadvised to add a first *unclean*, copy-pasted version of the original code to\n`src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` until you feel like all the necessary code is\nadded. From our experience, it is much more efficient to quickly add a first version of the required code and\nimprove/correct the code iteratively with the conversion script as described in the next section. The only thing that\nhas to work at this point is that you can instantiate the 🤗 Transformers implementation of *brand_new_bert*, *i.e.* the\nfollowing command should work:\n\n```python\nfrom transformers import BrandNewBertModel, BrandNewBertConfig\n\nmodel = BrandNewBertModel(BrandNewBertConfig())\n```\n\nThe above command will create a model according to the default parameters as defined in `BrandNewBertConfig()` with\nrandom weights, thus making sure that the `init()` methods of all components works.\n\nNote that all random initialization should happen in the `_init_weights` method of your `BrandnewBertPreTrainedModel`\nclass. It should initialize all leaf modules depending on the variables of the config. Here is an example with the\nBERT `_init_weights` method:\n\n```py\ndef _init_weights(self, module):\n    """Initialize the weights"""\n    if isinstance(module, nn.Linear):\n        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)\n        if module.bias is not None:\n            module.bias.data.zero_()\n    elif isinstance(module, nn.Embedding):\n        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)\n        if module.padding_idx is not None:\n            module.weight.data[module.padding_idx].zero_()\n    elif isinstance(module, nn.LayerNorm):\n        module.bias.data.zero_()\n        module.weight.data.fill_(1.0)\n```\n\nYou can have some more custom schemes if you need a special initialization for some modules. For instance, in\n`Wav2Vec2ForPreTraining`, the last two linear layers need to have the initialization of the regular PyTorch `nn.Linear`\nbut all the other ones should use an initialization as above. This is coded like this:\n\n```py\ndef _init_weights(self, module):\n    """Initialize the weights"""\n    if isinstnace(module, Wav2Vec2ForPreTraining):\n        module.project_hid.reset_parameters()\n        module.project_q.reset_parameters()\n        module.project_hid._is_hf_initialized = True\n        module.project_q._is_hf_initialized = True\n    elif isinstance(module, nn.Linear):\n        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)\n        if module.bias is not None:\n            module.bias.data.zero_()\n```\n\nThe `_is_hf_initialized` flag is internally used to make sure we only initialize a submodule once. By setting it to\n`True` for `module.project_q` and `module.project_hid`, we make sure the custom initialization we did is not overridden later on,\nthe `_init_weights` function won\'t be applied to them.\n\n**6. Write a conversion script**\n\nNext, you should write a conversion script that lets you convert the checkpoint you used to debug *brand_new_bert* in\nthe original repository to a checkpoint compatible with your just created 🤗 Transformers implementation of\n*brand_new_bert*. It is not advised to write the conversion script from scratch, but rather to look through already\nexisting conversion scripts in 🤗 Transformers for one that has been used to convert a similar model that was written in\nthe same framework as *brand_new_bert*. Usually, it is enough to copy an already existing conversion script and\nslightly adapt it for your use case. Don\'t hesitate to ask the Hugging Face team to point you to a similar already\nexisting conversion script for your model.\n\n- If you are porting a model from TensorFlow to PyTorch, a good starting point might be BERT\'s conversion script [here](https://github.com/huggingface/transformers/blob/7acfa95afb8194f8f9c1f4d2c6028224dbed35a2/src/transformers/models/bert/modeling_bert.py#L91)\n- If you are porting a model from PyTorch to PyTorch, a good starting point might be BART\'s conversion script [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py)\n\nIn the following, we\'ll quickly explain how PyTorch models store layer weights and define layer names. In PyTorch, the\nname of a layer is defined by the name of the class attribute you give the layer. Let\'s define a dummy model in\nPyTorch, called `SimpleModel` as follows:\n\n```python\nfrom torch import nn\n\n\nclass SimpleModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.dense = nn.Linear(10, 10)\n        self.intermediate = nn.Linear(10, 10)\n        self.layer_norm = nn.LayerNorm(10)\n```\n\nNow we can create an instance of this model definition which will fill all weights: `dense`, `intermediate`,\n`layer_norm` with random weights. We can print the model to see its architecture\n\n```python\nmodel = SimpleModel()\n\nprint(model)\n```\n\nThis will print out the following:\n\n```\nSimpleModel(\n  (dense): Linear(in_features=10, out_features=10, bias=True)\n  (intermediate): Linear(in_features=10, out_features=10, bias=True)\n  (layer_norm): LayerNorm((10,), eps=1e-05, elementwise_affine=True)\n)\n```\n\nWe can see that the layer names are defined by the name of the class attribute in PyTorch. You can print out the weight\nvalues of a specific layer:\n\n```python\nprint(model.dense.weight.data)\n```\n\nto see that the weights were randomly initialized\n\n```\ntensor([[-0.0818,  0.2207, -0.0749, -0.0030,  0.0045, -0.1569, -0.1598,  0.0212,\n         -0.2077,  0.2157],\n        [ 0.1044,  0.0201,  0.0990,  0.2482,  0.3116,  0.2509,  0.2866, -0.2190,\n          0.2166, -0.0212],\n        [-0.2000,  0.1107, -0.1999, -0.3119,  0.1559,  0.0993,  0.1776, -0.1950,\n         -0.1023, -0.0447],\n        [-0.0888, -0.1092,  0.2281,  0.0336,  0.1817, -0.0115,  0.2096,  0.1415,\n         -0.1876, -0.2467],\n        [ 0.2208, -0.2352, -0.1426, -0.2636, -0.2889, -0.2061, -0.2849, -0.0465,\n          0.2577,  0.0402],\n        [ 0.1502,  0.2465,  0.2566,  0.0693,  0.2352, -0.0530,  0.1859, -0.0604,\n          0.2132,  0.1680],\n        [ 0.1733, -0.2407, -0.1721,  0.1484,  0.0358, -0.0633, -0.0721, -0.0090,\n          0.2707, -0.2509],\n        [-0.1173,  0.1561,  0.2945,  0.0595, -0.1996,  0.2988, -0.0802,  0.0407,\n          0.1829, -0.1568],\n        [-0.1164, -0.2228, -0.0403,  0.0428,  0.1339,  0.0047,  0.1967,  0.2923,\n          0.0333, -0.0536],\n        [-0.1492, -0.1616,  0.1057,  0.1950, -0.2807, -0.2710, -0.1586,  0.0739,\n          0.2220,  0.2358]]).\n```\n\nIn the conversion script, you should fill those randomly initialized weights with the exact weights of the\ncorresponding layer in the checkpoint. *E.g.*\n\n```python\n# retrieve matching layer weights, e.g. by\n# recursive algorithm\nlayer_name = "dense"\npretrained_weight = array_of_dense_layer\n\nmodel_pointer = getattr(model, "dense")\n\nmodel_pointer.weight.data = torch.from_numpy(pretrained_weight)\n```\n\nWhile doing so, you must verify that each randomly initialized weight of your PyTorch model and its corresponding\npretrained checkpoint weight exactly match in both **shape and name**. To do so, it is **necessary** to add assert\nstatements for the shape and print out the names of the checkpoints weights. E.g. you should add statements like:\n\n```python\nassert (\n    model_pointer.weight.shape == pretrained_weight.shape\n), f"Pointer shape of random weight {model_pointer.shape} and array shape of checkpoint weight {pretrained_weight.shape} mismatched"\n```\n\nBesides, you should also print out the names of both weights to make sure they match, *e.g.*\n\n```python\nlogger.info(f"Initialize PyTorch weight {layer_name} from {pretrained_weight.name}")\n```\n\nIf either the shape or the name doesn\'t match, you probably assigned the wrong checkpoint weight to a randomly\ninitialized layer of the 🤗 Transformers implementation.\n\nAn incorrect shape is most likely due to an incorrect setting of the config parameters in `BrandNewBertConfig()` that\ndo not exactly match those that were used for the checkpoint you want to convert. However, it could also be that\nPyTorch\'s implementation of a layer requires the weight to be transposed beforehand.\n\nFinally, you should also check that **all** required weights are initialized and print out all checkpoint weights that\nwere not used for initialization to make sure the model is correctly converted. It is completely normal, that the\nconversion trials fail with either a wrong shape statement or wrong name assignment. This is most likely because either\nyou used incorrect parameters in `BrandNewBertConfig()`, have a wrong architecture in the 🤗 Transformers\nimplementation, you have a bug in the `init()` functions of one of the components of the 🤗 Transformers\nimplementation or you need to transpose one of the checkpoint weights.\n\nThis step should be iterated with the previous step until all weights of the checkpoint are correctly loaded in the\nTransformers model. Having correctly loaded the checkpoint into the 🤗 Transformers implementation, you can then save\nthe model under a folder of your choice `/path/to/converted/checkpoint/folder` that should then contain both a\n`pytorch_model.bin` file and a `config.json` file:\n\n```python\nmodel.save_pretrained("/path/to/converted/checkpoint/folder")\n```\n\n**7. Implement the forward pass**\n\nHaving managed to correctly load the pretrained weights into the 🤗 Transformers implementation, you should now make\nsure that the forward pass is correctly implemented. In [Get familiar with the original repository](#34-run-a-pretrained-checkpoint-using-the-original-repository), you have already created a script that runs a forward\npass of the model using the original repository. Now you should write an analogous script using the 🤗 Transformers\nimplementation instead of the original one. It should look as follows:\n\n```python\nmodel = BrandNewBertModel.from_pretrained("/path/to/converted/checkpoint/folder")\ninput_ids = [0, 4, 4, 3, 2, 4, 1, 7, 19]\noutput = model(input_ids).last_hidden_states\n```\n\nIt is very likely that the 🤗 Transformers implementation and the original model implementation don\'t give the exact\nsame output the very first time or that the forward pass throws an error. Don\'t be disappointed - it\'s expected! First,\nyou should make sure that the forward pass doesn\'t throw any errors. It often happens that the wrong dimensions are\nused leading to a *Dimensionality mismatch* error or that the wrong data type object is used, *e.g.* `torch.long`\ninstead of `torch.float32`. Don\'t hesitate to ask the Hugging Face team for help, if you don\'t manage to solve\ncertain errors.\n\nThe final part to make sure the 🤗 Transformers implementation works correctly is to ensure that the outputs are\nequivalent to a precision of `1e-3`. First, you should ensure that the output shapes are identical, *i.e.*\n`outputs.shape` should yield the same value for the script of the 🤗 Transformers implementation and the original\nimplementation. Next, you should make sure that the output values are identical as well. This one of the most difficult\nparts of adding a new model. Common mistakes why the outputs are not identical are:\n\n- Some layers were not added, *i.e.* an *activation* layer was not added, or the residual connection was forgotten\n- The word embedding matrix was not tied\n- The wrong positional embeddings are used because the original implementation uses on offset\n- Dropout is applied during the forward pass. To fix this make sure *model.training is False* and that no dropout\n  layer is falsely activated during the forward pass, *i.e.* pass *self.training* to [PyTorch\'s functional dropout](https://pytorch.org/docs/stable/nn.functional.html?highlight=dropout#torch.nn.functional.dropout)\n\nThe best way to fix the problem is usually to look at the forward pass of the original implementation and the 🤗\nTransformers implementation side-by-side and check if there are any differences. Ideally, you should debug/print out\nintermediate outputs of both implementations of the forward pass to find the exact position in the network where the 🤗\nTransformers implementation shows a different output than the original implementation. First, make sure that the\nhard-coded `input_ids` in both scripts are identical. Next, verify that the outputs of the first transformation of\nthe `input_ids` (usually the word embeddings) are identical. And then work your way up to the very last layer of the\nnetwork. At some point, you will notice a difference between the two implementations, which should point you to the bug\nin the 🤗 Transformers implementation. From our experience, a simple and efficient way is to add many print statements\nin both the original implementation and 🤗 Transformers implementation, at the same positions in the network\nrespectively, and to successively remove print statements showing the same values for intermediate presentations.\n\nWhen you\'re confident that both implementations yield the same output, verifying the outputs with\n`torch.allclose(original_output, output, atol=1e-3)`, you\'re done with the most difficult part! Congratulations - the\nwork left to be done should be a cakewalk 😊.\n\n**8. Adding all necessary model tests**\n\nAt this point, you have successfully added a new model. However, it is very much possible that the model does not yet\nfully comply with the required design. To make sure, the implementation is fully compatible with 🤗 Transformers, all\ncommon tests should pass. The Cookiecutter should have automatically added a test file for your model, probably under\nthe same `tests/models/brand_new_bert/test_modeling_brand_new_bert.py`. Run this test file to verify that all common\ntests pass:\n\n```bash\npytest tests/models/brand_new_bert/test_modeling_brand_new_bert.py\n```\n\nHaving fixed all common tests, it is now crucial to ensure that all the nice work you have done is well tested, so that\n\n- a) The community can easily understand your work by looking at specific tests of *brand_new_bert*\n- b) Future changes to your model will not break any important feature of the model.\n\nAt first, integration tests should be added. Those integration tests essentially do the same as the debugging scripts\nyou used earlier to implement the model to 🤗 Transformers. A template of those model tests is already added by the\nCookiecutter, called `BrandNewBertModelIntegrationTests` and only has to be filled out by you. To ensure that those\ntests are passing, run\n\n```bash\nRUN_SLOW=1 pytest -sv tests/models/brand_new_bert/test_modeling_brand_new_bert.py::BrandNewBertModelIntegrationTests\n```\n\n<Tip>\n\nIn case you are using Windows, you should replace `RUN_SLOW=1` with `SET RUN_SLOW=1`\n\n</Tip>\n\nSecond, all features that are special to *brand_new_bert* should be tested additionally in a separate test under\n`BrandNewBertModelTester`/``BrandNewBertModelTest`. This part is often forgotten but is extremely useful in two\nways:\n\n- It helps to transfer the knowledge you have acquired during the model addition to the community by showing how the\n  special features of *brand_new_bert* should work.\n- Future contributors can quickly test changes to the model by running those special tests.\n\n\n**9. Implement the tokenizer**\n\nNext, we should add the tokenizer of *brand_new_bert*. Usually, the tokenizer is equivalent or very similar to an\nalready existing tokenizer of 🤗 Transformers.\n\nIt is very important to find/extract the original tokenizer file and to manage to load this file into the 🤗\nTransformers\' implementation of the tokenizer.\n\nTo ensure that the tokenizer works correctly, it is recommended to first create a script in the original repository\nthat inputs a string and returns the `input_ids``. It could look similar to this (in pseudo-code):\n\n```python\ninput_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."\nmodel = BrandNewBertModel.load_pretrained_checkpoint("/path/to/checkpoint/")\ninput_ids = model.tokenize(input_str)\n```\n\nYou might have to take a deeper look again into the original repository to find the correct tokenizer function or you\nmight even have to do changes to your clone of the original repository to only output the `input_ids`. Having written\na functional tokenization script that uses the original repository, an analogous script for 🤗 Transformers should be\ncreated. It should look similar to this:\n\n```python\nfrom transformers import BrandNewBertTokenizer\n\ninput_str = "This is a long example input string containing special characters .$?-, numbers 2872 234 12 and words."\n\ntokenizer = BrandNewBertTokenizer.from_pretrained("/path/to/tokenizer/folder/")\n\ninput_ids = tokenizer(input_str).input_ids\n```\n\nWhen both `input_ids` yield the same values, as a final step a tokenizer test file should also be added.\n\nAnalogous to the modeling test files of *brand_new_bert*, the tokenization test files of *brand_new_bert* should\ncontain a couple of hard-coded integration tests.\n\n**10. Run End-to-end integration tests**\n\nHaving added the tokenizer, you should also add a couple of end-to-end integration tests using both the model and the\ntokenizer to `tests/models/brand_new_bert/test_modeling_brand_new_bert.py` in 🤗 Transformers.\nSuch a test should show on a meaningful\ntext-to-text sample that the 🤗 Transformers implementation works as expected. A meaningful text-to-text sample can\ninclude *e.g.* a source-to-target-translation pair, an article-to-summary pair, a question-to-answer pair, etc… If none\nof the ported checkpoints has been fine-tuned on a downstream task it is enough to simply rely on the model tests. In a\nfinal step to ensure that the model is fully functional, it is advised that you also run all tests on GPU. It can\nhappen that you forgot to add some `.to(self.device)` statements to internal tensors of the model, which in such a\ntest would show in an error. In case you have no access to a GPU, the Hugging Face team can take care of running those\ntests for you.\n\n**11. Add Docstring**\n\nNow, all the necessary functionality for *brand_new_bert* is added - you\'re almost done! The only thing left to add is\na nice docstring and a doc page. The Cookiecutter should have added a template file called\n`docs/source/model_doc/brand_new_bert.mdx` that you should fill out. Users of your model will usually first look at\nthis page before using your model. Hence, the documentation must be understandable and concise. It is very useful for\nthe community to add some *Tips* to show how the model should be used. Don\'t hesitate to ping the Hugging Face team\nregarding the docstrings.\n\nNext, make sure that the docstring added to `src/transformers/models/brand_new_bert/modeling_brand_new_bert.py` is\ncorrect and included all necessary inputs and outputs. We have a detailed guide about writing documentation and our docstring format [here](writing-documentation). It is always to good to remind oneself that documentation should\nbe treated at least as carefully as the code in 🤗 Transformers since the documentation is usually the first contact\npoint of the community with the model.\n\n**Code refactor**\n\nGreat, now you have added all the necessary code for *brand_new_bert*. At this point, you should correct some potential\nincorrect code style by running:\n\n```bash\nmake style\n```\n\nand verify that your coding style passes the quality check:\n\n```bash\nmake quality\n```\n\nThere are a couple of other very strict design tests in 🤗 Transformers that might still be failing, which shows up in\nthe tests of your pull request. This is often because of some missing information in the docstring or some incorrect\nnaming. The Hugging Face team will surely help you if you\'re stuck here.\n\nLastly, it is always a good idea to refactor one\'s code after having ensured that the code works correctly. With all\ntests passing, now it\'s a good time to go over the added code again and do some refactoring.\n\nYou have now finished the coding part, congratulation! 🎉 You are Awesome! 😎\n\n**12. Upload the models to the model hub**\n\nIn this final part, you should convert and upload all checkpoints to the model hub and add a model card for each\nuploaded model checkpoint. You can get familiar with the hub functionalities by reading our [Model sharing and uploading Page](model_sharing). You should work alongside the Hugging Face team here to decide on a fitting name for each\ncheckpoint and to get the required access rights to be able to upload the model under the author\'s organization of\n*brand_new_bert*. The `push_to_hub` method, present in all models in `transformers`, is a quick and efficient way to push your checkpoint to the hub. A little snippet is pasted below:\n\n```python\nbrand_new_bert.push_to_hub("brand_new_bert")\n# Uncomment the following line to push to an organization.\n# brand_new_bert.push_to_hub("<organization>/brand_new_bert")\n```\n\nIt is worth spending some time to create fitting model cards for each checkpoint. The model cards should highlight the\nspecific characteristics of this particular checkpoint, *e.g.* On which dataset was the checkpoint\npretrained/fine-tuned on? On what down-stream task should the model be used? And also include some code on how to\ncorrectly use the model.\n\n**13. (Optional) Add notebook**\n\nIt is very helpful to add a notebook that showcases in-detail how *brand_new_bert* can be used for inference and/or\nfine-tuned on a downstream task. This is not mandatory to merge your PR, but very useful for the community.\n\n**14. Submit your finished PR**\n\nYou\'re done programming now and can move to the last step, which is getting your PR merged into main. Usually, the\nHugging Face team should have helped you already at this point, but it is worth taking some time to give your finished\nPR a nice description and eventually add comments to your code, if you want to point out certain design choices to your\nreviewer.\n\n### Share your work!!\n\nNow, it\'s time to get some credit from the community for your work! Having completed a model addition is a major\ncontribution to Transformers and the whole NLP community. Your code and the ported pre-trained models will certainly be\nused by hundreds and possibly even thousands of developers and researchers. You should be proud of your work and share\nyour achievement with the community.\n\n**You have made another model that is super easy to access for everyone in the community! 🤯**\n' metadata={'title': 'add_new_model.mdx', 'repo_owner': 'huggingface', 'repo_name': 'transformers'}
page_content='<!--Copyright 2020 The HuggingFace Team. All rights reserved.\n\nLicensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with\nthe License. You may obtain a copy of the License at\n\nhttp://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software distributed under the License is distributed on\nan "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the\n-->\n\n# How to create a custom pipeline?\n\nIn this guide, we will see how to create a custom pipeline and share it on the [Hub](hf.co/models) or add it to the\n🤗 Transformers library.\n\nFirst and foremost, you need to decide the raw entries the pipeline will be able to take. It can be strings, raw bytes,\ndictionaries or whatever seems to be the most likely desired input. Try to keep these inputs as pure Python as possible\nas it makes compatibility easier (even through other languages via JSON). Those will be the `inputs` of the\npipeline (`preprocess`).\n\nThen define the `outputs`. Same policy as the `inputs`. The simpler, the better. Those will be the outputs of\n`postprocess` method.\n\nStart by inheriting the base class `Pipeline` with the 4 methods needed to implement `preprocess`,\n`_forward`, `postprocess`, and `_sanitize_parameters`.\n\n\n```python\nfrom transformers import Pipeline\n\n\nclass MyPipeline(Pipeline):\n    def _sanitize_parameters(self, **kwargs):\n        preprocess_kwargs = {}\n        if "maybe_arg" in kwargs:\n            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]\n        return preprocess_kwargs, {}, {}\n\n    def preprocess(self, inputs, maybe_arg=2):\n        model_input = Tensor(inputs["input_ids"])\n        return {"model_input": model_input}\n\n    def _forward(self, model_inputs):\n        # model_inputs == {"model_input": model_input}\n        outputs = self.model(**model_inputs)\n        # Maybe {"logits": Tensor(...)}\n        return outputs\n\n    def postprocess(self, model_outputs):\n        best_class = model_outputs["logits"].softmax(-1)\n        return best_class\n```\n\nThe structure of this breakdown is to support relatively seamless support for CPU/GPU, while supporting doing\npre/postprocessing on the CPU on different threads\n\n`preprocess` will take the originally defined inputs, and turn them into something feedable to the model. It might\ncontain more information and is usually a `Dict`.\n\n`_forward` is the implementation detail and is not meant to be called directly. `forward` is the preferred\ncalled method as it contains safeguards to make sure everything is working on the expected device. If anything is\nlinked to a real model it belongs in the `_forward` method, anything else is in the preprocess/postprocess.\n\n`postprocess` methods will take the output of `_forward` and turn it into the final output that was decided\nearlier.\n\n`_sanitize_parameters` exists to allow users to pass any parameters whenever they wish, be it at initialization\ntime `pipeline(...., maybe_arg=4)` or at call time `pipe = pipeline(...); output = pipe(...., maybe_arg=4)`.\n\nThe returns of `_sanitize_parameters` are the 3 dicts of kwargs that will be passed directly to `preprocess`,\n`_forward`, and `postprocess`. Don\'t fill anything if the caller didn\'t call with any extra parameter. That\nallows to keep the default arguments in the function definition which is always more "natural".\n\nA classic example would be a `top_k` argument in the post processing in classification tasks.\n\n```python\n>>> pipe = pipeline("my-new-task")\n>>> pipe("This is a test")\n[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}, {"label": "3-star", "score": 0.05}\n{"label": "4-star", "score": 0.025}, {"label": "5-star", "score": 0.025}]\n\n>>> pipe("This is a test", top_k=2)\n[{"label": "1-star", "score": 0.8}, {"label": "2-star", "score": 0.1}]\n```\n\nIn order to achieve that, we\'ll update our `postprocess` method with a default parameter to `5`. and edit\n`_sanitize_parameters` to allow this new parameter.\n\n\n```python\ndef postprocess(self, model_outputs, top_k=5):\n    best_class = model_outputs["logits"].softmax(-1)\n    # Add logic to handle top_k\n    return best_class\n\n\ndef _sanitize_parameters(self, **kwargs):\n    preprocess_kwargs = {}\n    if "maybe_arg" in kwargs:\n        preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]\n\n    postprocess_kwargs = {}\n    if "top_k" in kwargs:\n        postprocess_kwargs["top_k"] = kwargs["top_k"]\n    return preprocess_kwargs, {}, postprocess_kwargs\n```\n\nTry to keep the inputs/outputs very simple and ideally JSON-serializable as it makes the pipeline usage very easy\nwithout requiring users to understand new kind of objects. It\'s also relatively common to support many different types\nof arguments for ease of use (audio files, can be filenames, URLs or pure bytes)\n\n\n\n## Adding it to the list of supported tasks\n\nTo register your `new-task` to the list of supported tasks, you have to add it to the `PIPELINE_REGISTRY`:\n\n```python\nfrom transformers.pipelines import PIPELINE_REGISTRY\n\nPIPELINE_REGISTRY.register_pipeline(\n    "new-task",\n    pipeline_class=MyPipeline,\n    pt_model=AutoModelForSequenceClassification,\n)\n```\n\nYou can specify a default model if you want, in which case it should come with a specific revision (which can be the name of a branch or a commit hash, here we took `"abcdef"`) as well as the type:\n\n```python\nPIPELINE_REGISTRY.register_pipeline(\n    "new-task",\n    pipeline_class=MyPipeline,\n    pt_model=AutoModelForSequenceClassification,\n    default={"pt": ("user/awesome_model", "abcdef")},\n    type="text",  # current support type: text, audio, image, multimodal\n)\n```\n\n## Share your pipeline on the Hub\n\nTo share your custom pipeline on the Hub, you just have to save the custom code of your `Pipeline` subclass in a\npython file. For instance, let\'s say we want to use a custom pipeline for sentence pair classification like this:\n\n```py\nimport numpy as np\n\nfrom transformers import Pipeline\n\n\ndef softmax(outputs):\n    maxes = np.max(outputs, axis=-1, keepdims=True)\n    shifted_exp = np.exp(outputs - maxes)\n    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)\n\n\nclass PairClassificationPipeline(Pipeline):\n    def _sanitize_parameters(self, **kwargs):\n        preprocess_kwargs = {}\n        if "second_text" in kwargs:\n            preprocess_kwargs["second_text"] = kwargs["second_text"]\n        return preprocess_kwargs, {}, {}\n\n    def preprocess(self, text, second_text=None):\n        return self.tokenizer(text, text_pair=second_text, return_tensors=self.framework)\n\n    def _forward(self, model_inputs):\n        return self.model(**model_inputs)\n\n    def postprocess(self, model_outputs):\n        logits = model_outputs.logits[0].numpy()\n        probabilities = softmax(logits)\n\n        best_class = np.argmax(probabilities)\n        label = self.model.config.id2label[best_class]\n        score = probabilities[best_class].item()\n        logits = logits.tolist()\n        return {"label": label, "score": score, "logits": logits}\n```\n\nThe implementation is framework agnostic, and will work for PyTorch and TensorFlow models. If we have saved this in\na file named `pair_classification.py`, we can then import it and register it like this:\n\n```py\nfrom pair_classification import PairClassificationPipeline\nfrom transformers.pipelines import PIPELINE_REGISTRY\nfrom transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification\n\nPIPELINE_REGISTRY.register_pipeline(\n    "pair-classification",\n    pipeline_class=PairClassificationPipeline,\n    pt_model=AutoModelForSequenceClassification,\n    tf_model=TFAutoModelForSequenceClassification,\n)\n```\n\nOnce this is done, we can use it with a pretrained model. For instance `sgugger/finetuned-bert-mrpc` has been\nfine-tuned on the MRPC dataset, which classifies pairs of sentences as paraphrases or not.\n\n```py\nfrom transformers import pipeline\n\nclassifier = pipeline("pair-classification", model="sgugger/finetuned-bert-mrpc")\n```\n\nThen we can share it on the Hub by using the `save_pretrained` method in a `Repository`:\n\n```py\nfrom huggingface_hub import Repository\n\nrepo = Repository("test-dynamic-pipeline", clone_from="{your_username}/test-dynamic-pipeline")\nclassifier.save_pretrained("test-dynamic-pipeline")\nrepo.push_to_hub()\n```\n\nThis will copy the file where you defined `PairClassificationPipeline` inside the folder `"test-dynamic-pipeline"`,\nalong with saving the model and tokenizer of the pipeline, before pushing everything in the repository\n`{your_username}/test-dynamic-pipeline`. After that anyone can use it as long as they provide the option\n`trust_remote_code=True`:\n\n```py\nfrom transformers import pipeline\n\nclassifier = pipeline(model="{your_username}/test-dynamic-pipeline", trust_remote_code=True)\n```\n\n## Add the pipeline to 🤗 Transformers\n\nIf you want to contribute your pipeline to 🤗 Transformers, you will need to add a new module in the `pipelines` submodule\nwith the code of your pipeline, then add it in the list of tasks defined in `pipelines/__init__.py`.\n\nThen you will need to add tests. Create a new file `tests/test_pipelines_MY_PIPELINE.py` with example with the other tests.\n\nThe `run_pipeline_test` function will be very generic and run on small random models on every possible\narchitecture as defined by `model_mapping` and `tf_model_mapping`.\n\nThis is very important to test future compatibility, meaning if someone adds a new model for\n`XXXForQuestionAnswering` then the pipeline test will attempt to run on it. Because the models are random it\'s\nimpossible to check for actual values, that\'s why there is a helper `ANY` that will simply attempt to match the\noutput of the pipeline TYPE.\n\nYou also *need* to implement 2 (ideally 4) tests.\n\n- `test_small_model_pt` : Define 1 small model for this pipeline (doesn\'t matter if the results don\'t make sense)\n  and test the pipeline outputs. The results should be the same as `test_small_model_tf`.\n- `test_small_model_tf` : Define 1 small model for this pipeline (doesn\'t matter if the results don\'t make sense)\n  and test the pipeline outputs. The results should be the same as `test_small_model_pt`.\n- `test_large_model_pt` (`optional`): Tests the pipeline on a real pipeline where the results are supposed to\n  make sense. These tests are slow and should be marked as such. Here the goal is to showcase the pipeline and to make\n  sure there is no drift in future releases.\n- `test_large_model_tf` (`optional`): Tests the pipeline on a real pipeline where the results are supposed to\n  make sense. These tests are slow and should be marked as such. Here the goal is to showcase the pipeline and to make\n  sure there is no drift in future releases.\n' metadata={'title': 'add_new_pipeline.mdx', 'repo_owner': 'huggingface', 'repo_name': 'transformers'}

```
## 3.6 Document Transformers: TextSplitters

### Text Splitters

Imagina que estás trabajando con un libro muy grueso y necesitas pasarlo por una ventana muy estrecha. ¿Qué harías? Probablemente, lo cortarías en secciones más manejables y las pasarías una por una. Ahora, cambia el libro por un documento largo y la ventana por el modelo de procesamiento de lenguaje natural que estás utilizando. Este escenario es exactamente por qué necesitamos los separadores de texto en el campo de la inteligencia artificial.

LangChain, comprendiendo este desafío, tiene incorporados varios separadores de texto para facilitar la división, combinación, filtrado y manipulación de los documentos. De este modo, puedes transformarlos para que se adapten mejor a tu aplicación.

Cuando nos enfrentamos a textos largos, es imprescindible dividirlos en fragmentos. Aunque esto suena sencillo, no es tan simple como parece. Queremos mantener las partes del texto que están semánticamente relacionadas juntas. Y esto de "semánticamente relacionado" puede variar dependiendo del tipo de texto con el que estés trabajando.

Piensa en el texto como un rompecabezas, cada pieza (o fragmento) tiene su propio significado, pero también contribuye a la imagen general (o el contexto). Queremos separar el rompecabezas en piezas, pero sin perder el sentido de la imagen completa.

Entonces, ¿cómo funcionan exactamente los separadores de texto?

1. Primero, dividen el texto en fragmentos pequeños y semánticamente significativos (a menudo oraciones).
2. Luego, comienzan a combinar estos fragmentos pequeños en un fragmento más grande hasta que alcanzan un tamaño determinado (medido por alguna función).
3. Una vez que alcanzan ese tamaño, hacen de ese fragmento su propio texto y luego comienzan a crear un nuevo fragmento de texto con cierta superposición. Esto es para mantener el contexto entre fragmentos.

En este proceso, puedes personalizar tu separador de texto en dos aspectos: cómo se divide el texto y cómo se mide el tamaño del fragmento.

### RecursiveCharacterTextSplitter

Para facilitar las cosas, LangChain ofrece un separador de texto por defecto: el `RecursiveCharacterTextSplitter`. Este separador de texto toma una lista de caracteres y trata de crear fragmentos basándose en la división del primer carácter. Pero, si algún fragmento resulta demasiado grande, pasa al siguiente carácter, y así sucesivamente. Los caracteres que intenta dividir son ["\n\n", "\n", " ", ""]

El `RecursiveCharacterTextSplitter` ofrece una ventaja importante: intenta preservar tanto contexto semántico como sea posible manteniendo intactos los párrafos, las oraciones y las palabras. Estas unidades de texto suelen tener fuertes relaciones semánticas, lo que significa que las palabras dentro de ellas a menudo están estrechamente relacionadas en significado. Esta es una característica sumamente beneficiosa para muchas tareas de procesamiento del lenguaje natural.

Piensa en una conversación cotidiana, es más fácil entender una idea cuando escuchas la oración completa en lugar de palabras o frases sueltas. Esta misma lógica se aplica a los modelos de procesamiento de lenguaje natural. Al mantener intactos los párrafos, oraciones y palabras, se preserva el 'flujo de conversación' en el texto, lo que puede mejorar la eficacia del modelo al interpretar y comprender el texto.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [11_text_splitters.py](scripts%2F11_text_splitters.py)

A partir de nuestros `Document` podemos crear más `Document` con `RecursiveCharacterTextSplitter`, es decir, podemos partirlos manteniendo nuestra metadata.

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./public_key_cryptography.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    length_function=len,
    chunk_overlap=200
)

documents = text_splitter.split_documents(data)

print(len(documents))

print(documents[20])
```
Respuesta esperada:
```commandline
142
page_content='stances. Suppose, for example, that the plaintext of each \npuzzle is 96 bits, consisting of 64 bits of key together with \nathirty-two bit block of zeros that enables Bob to recognize \nthe right solution. The puzzle is constructed by encrypting \nthis plaintext using a block cipher with 20 bits of key. Alice produces a million of these puzzles and Bob requires about \nhalf a million tests to solve one. The bandwidth and com- \nputing power required to make this feasible are large but \nnot inaccessible. On a DSI (1.544 Mbit) channel it would \nrequire about a minute to communicate the puzzles. If keys \ncan be tried on the selected puzzle at about ten-thousand \nper second, it will take Bob another minute to solve it. \nFinally, it will take a similar amount of time for Alice to figure \nout, from the test message, which key has been chosen. \nThe intruder can expect to have to solve half a million \npuzzles at half a million tries apiece. With equivalent com-' metadata={'source': './public_key_cryptography.pdf', 'page': 2}
```

### Tamaño del fragmento y superposición

Imagina que estás trabajando con un rompecabezas de palabras, donde cada pieza es una porción de texto. Para que este rompecabezas sea manejable, necesitas asegurarte de que las piezas son del tamaño correcto y se superponen adecuadamente. En el mundo del procesamiento de texto, estas "piezas" son los fragmentos de texto, y su tamaño y superposición pueden ser esenciales para el rendimiento de tus modelos de aprendizaje automático.

En primer lugar, hablemos del tamaño del fragmento. La pregunta que podrías hacerte es, ¿cuán grande debe ser cada fragmento de texto? Bien, la respuesta depende del modelo de embedding de texto que estés utilizando. Un "modelo de embedding" puede parecer un término intimidante, pero es simplemente una herramienta que convertimos palabras, oraciones o documentos completos en vectores numéricos que las máquinas pueden entender.

Por ejemplo, el modelo de incrustación `text-embedding-ada-002` de OpenAI es excelente para muchas aplicaciones, pero puede manejar hasta 8191 tokens. Ahora, podrías preguntarte, ¿qué es un 'token'? Un token no es lo mismo que un carácter. Un token puede ser una palabra o incluso un signo de puntuación. Por lo tanto, un token podría tener desde un solo carácter hasta una decena de ellos. De esta manera, tu fragmento de texto podría tener miles de caracteres, pero debes asegurarte de que no contenga más de 8191 tokens.

Mantener los fragmentos entre 500 y 1000 caracteres suele ser un buen equilibrio. Este tamaño asegura que el contenido semántico es preservado sin sobrepasar el límite de tokens del modelo.

En cuanto a la superposición, este parámetro decide cuánto texto queremos repetir entre fragmentos. ¿Por qué querríamos hacer esto? Bueno, la superposición ayuda a mantener el contexto entre fragmentos contiguos. Es como tener una pequeña ventana de memoria que se traslada de un fragmento a otro. Generalmente, se recomienda ajustar la superposición al 10-20% del tamaño del fragmento. Esto asegura cierta conexión entre los fragmentos sin causar demasiada repetición. Si la superposición es demasiado grande, puede ralentizar el proceso y aumentar los costos de procesamiento.

Por lo tanto, si estás lidiando con textos relativamente largos, esta es la configuración que podrías utilizar.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
)

# o

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap  = 100,
#     length_function = len,
# )

documents = text_splitter.split_documents(data)
print("tamaño:", len(documents))
print("tipo:", type(documents))
print("tipo doc", type(documents[0]))
print("Contenido pag 0:")
print(documents[0].page_content)
```
Respuesta esperada:
```commandline
tamaño: 241
tipo: <class 'list'>
tipo doc <class 'langchain.schema.document.Document'>
Contenido pag 0:
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
```

## 3.7 Proyecto de ChatBot: configuración de entorno para LangChain y obtención de datos

De ahora en adelante, las clases estarán enfocadas a llevar adelante la creación de un ChatBot que nos permita responder
preguntas sobre la documentación de algunas bibliotecas de HuggingFace. Dentro de la información que vamos a obtener se encuentran
los siguiente respositorios:


- https://github.com/huggingface/blog
- https://github.com/huggingface/transformers/tree/main/docs/source/en
- https://github.com/huggingface/peft/tree/main/docs/source
- https://github.com/huggingface/accelerate/tree/main/docs/source

El objetivo será obtener toda la documentación de los repositorios, descargando sus archivos de `readme` en formato 
`md` o `mdx`.

Vamos a empezar clonando el repositorio principal del proyecto: https://github.com/platzi/curso-langchain.git

```bash
mkdir proyecto
cd proyecto
git clone https://github.com/platzi/curso-langchain.git
cd curso-langchain
```

Ahora creamos un entorno virtual: `asegurate de tener python 3.9 instalado`

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9
python3.9 -V
```

```bash
python3.9 -m venv hashira_env
source hashira_env/bin/activate
```

Actualizamos nuestra versión de PIP:

```bash
pip install --upgrade pip
```

Y finalmente, instalamos las dependencias que serán utilizadas en este proyecto: [requirements.txt](proyecto%2Fcurso-langchain%2Frequirements.txt)

```bash
pip install -r requirements.txt
```

Es importante tener en menta los scripts: [utils.py](proyecto%2Fcurso-langchain%2Fhashira%2Futils.py)

Y tambien: [text_extractor.py](proyecto%2Fcurso-langchain%2Fhashira%2Ftext_extractor.py) que al ejectuarlo nos ha descargado
y creado el documento: [docs_en_2023_06_29.jsonl](proyecto%2Fcurso-langchain%2Fdata%2Fdocs_en_2023_06_29.jsonl)

Que es con el que estaremos trabajando y contiene la documentación de HuggingFace en formato `JSONL`

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
