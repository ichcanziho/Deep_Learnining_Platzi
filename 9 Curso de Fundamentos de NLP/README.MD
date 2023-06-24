# Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK

Aprende cómo los algoritmos pueden aprender a procesar el lenguaje humano con Python y NLTK y entrena tus primeros modelos 
de procesamiento de lenguaje natural

- Dominar las estadísticas básicas en Procesamiento de Lenguaje Natural
- Entender la evolución del Procesamiento de Lenguaje
- Entrenar modelos de lenguaje natural


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
> 
> Este Curso es el Número 9 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice

- [1 Introducción al Procesamiento de Lenguaje Natural](#1-introducción-al-procesamiento-de-lenguaje-natural)
  - [1.1 Introducción al Procesamiento de Lenguaje Natural](#11-introducción-al-procesamiento-de-lenguaje-natural)
  - [1.2 Evolución del NLP](#12-evolución-del-nlp)
  - [1.3 Conceptos básicos de NLP](#13-conceptos-básicos-de-nlp)
- [2 Fundamentos con NLTK](#2-fundamentos-con-nltk)
  - [2.1 Configurar ambiente de trabajo](#21-configurar-ambiente-de-trabajo)
  - [2.2 Palabras, textos y vocabularios](#22-palabras-textos-y-vocabularios)
  - [2.3 Tokenizacion con Expresiones Regulares](#23-tokenizacion-con-expresiones-regulares)
  - [2.4 Estadísticas básicas del lenguaje](#24-estadísticas-básicas-del-lenguaje)
  - [2.5 Distribuciones de frecuencia de palabras](#25-distribuciones-de-frecuencia-de-palabras)
  - [2.6 Refinamiento y visualización de cuerpos de texto](#26-refinamiento-y-visualización-de-cuerpos-de-texto)
  - [2.7 N-gramas y Colocaciones del lenguaje](#27-n-gramas-y-colocaciones-del-lenguaje)
  - [2.8 ¿Cómo extraer n-gramas de un texto en Python?](#28-cómo-extraer-n-gramas-de-un-texto-en-python)
  - [2.9 Colocaciones en Python](#29-colocaciones-en-python)
  - [2.10 Colocaciones en gráficos de dispersión](#210-colocaciones-en-gráficos-de-dispersión)
  - [2.11 Filtros y colocaciones en NLTK](#211-filtros-y-colocaciones-en-nltk)
  - [2.12 Introducción a los recursos léxicos](#212-introducción-a-los-recursos-léxicos)
  - [2.13 Recursos léxicos en NLTK](#213-recursos-léxicos-en-nltk)
  - [2.14 NLTK para traducción de palabras](#214-nltk-para-traducción-de-palabras)
  - [2.15 Introducción a WordNet](#215-introducción-a-wordnet)
  - [2.16 Explorando WordNet](#216-explorando-wordnet)
  - [2.17 Similitud Semántica con WordNet](#217-similitud-semántica-con-wordnet)
  - [2.18 Procesamiento de texto plano desde Web](#218-procesamiento-de-texto-plano-desde-web)
  - [2.19 Usando código estructurado: conexión de Drive a Google Colab](#219-usando-código-estructurado-conexión-de-drive-a-google-colab)
  - [2.20 Usando código estructurado: Funciones externas](#220-usando-código-estructurado-funciones-externas)
- [3 Perspectivas de lo que viene](#3-perspectivas-de-lo-que-viene)
  - [3.1 Continúa con el Curso de Algoritmos de Clasificación de Texto](#31-continúa-con-el-curso-de-algoritmos-de-clasificación-de-texto)


# 1 Introducción al Procesamiento de Lenguaje Natural

## 1.1 Introducción al Procesamiento de Lenguaje Natural


El Procesamiento del Lenguaje Natural `(NLP)` se refiere a la capacidad de una máquina para comprender, interpretar y generar 
lenguaje humano de manera automática. El objetivo principal del `NLP` es permitir a las máquinas interactuar y comunicarse 
con los humanos en su propio idioma. Incluye tareas como la traducción automática, el reconocimiento de voz, el resumen 
automático de textos, la clasificación de textos, la extracción de información, entre otros. El `NLP` se basa en técnicas de 
aprendizaje automático y utiliza enfoques como el procesamiento de lenguaje natural estadístico y el aprendizaje profundo 
(deep learning).

![1.png](ims%2F1%2F1.png)

El Entendimiento del Lenguaje Natural `(NLU)` se enfoca en la comprensión profunda y precisa del lenguaje humano en un contexto 
específico. Va más allá de la simple comprensión superficial del lenguaje y se esfuerza por capturar el significado y la 
intención detrás de las palabras. El `NLU` implica la capacidad de extraer información relevante de un texto, identificar 
entidades y relaciones, comprender el contexto y responder de manera inteligente. Es una forma más avanzada de `NLP` y requiere 
técnicas más sofisticadas de aprendizaje automático, como el procesamiento del lenguaje natural basado en el conocimiento 
y el procesamiento del lenguaje natural basado en modelos.


### Algunos de los usos actuales más destacados del `NLP` son:

![2.png](ims%2F1%2F2.png)

- `Asistentes virtuales y chatbots:` Los asistentes virtuales como Siri, Google Assistant y Alexa, así como los chatbots utilizados en sitios web y aplicaciones, emplean NLP para comprender y responder a las consultas y comandos en lenguaje natural.

- `Traducción automática:` La traducción automática utiliza técnicas de NLP para traducir texto de un idioma a otro de manera automática. Algunas plataformas populares, como Google Translate, se basan en algoritmos de NLP para ofrecer traducciones.

- `Análisis de sentimientos:` El NLP se utiliza para analizar y clasificar la polaridad de las opiniones y emociones expresadas en texto, permitiendo a las empresas recopilar información valiosa sobre la percepción de los clientes en redes sociales, reseñas de productos, comentarios, etc.

- `Extracción de información:` El NLP permite extraer información estructurada y relevante de grandes volúmenes de texto, como la identificación de entidades (por ejemplo, nombres de personas, organizaciones, ubicaciones) y la extracción de relaciones entre ellas.

- `Resumen automático de textos:` Los sistemas de resumen automático utilizan técnicas de NLP para analizar y condensar grandes cantidades de texto en resúmenes más breves, facilitando la comprensión y el acceso a la información relevante.

- `Clasificación de textos:` El NLP se utiliza para clasificar automáticamente textos en categorías predefinidas, lo que tiene aplicaciones en áreas como filtrado de spam, análisis de sentimiento, categorización de noticias, entre otros.

- `Recuperación de información:` Los motores de búsqueda y los sistemas de recuperación de información utilizan técnicas de NLP para comprender las consultas de los usuarios y recuperar documentos relevantes en función de esas consultas.

- `Generación de texto:` El NLP también se utiliza para generar texto automáticamente, como en la redacción de noticias, la generación de contenido para redes sociales y la creación de respuestas automáticas.


## 1.2 Evolución del NLP

![3.png](ims%2F1%2F3.png)

- **Sistemas basados en reglas (1950s - 1990s):**
En sus primeras etapas, el NLP se basaba en sistemas construidos mediante reglas gramaticales y lingüísticas. Estos sistemas dependían de conocimientos y reglas explícitas codificadas por expertos para realizar tareas de procesamiento del lenguaje. El enfoque se centraba en el análisis sintáctico y la descomposición de oraciones en estructuras gramaticales.

- **Estadística de Corpus (1990s - 2000s):**
A medida que aumentaba la disponibilidad de grandes volúmenes de datos de texto, surgieron enfoques basados en estadísticas de corpus. En lugar de depender de reglas predefinidas, estos enfoques utilizaban técnicas de procesamiento del lenguaje natural estadístico para extraer patrones y probabilidades a partir de grandes conjuntos de texto. Esto permitió mejoras en áreas como la traducción automática y la corrección ortográfica.

- **Machine learning (2000s - 2014):**
Con los avances en el aprendizaje automático, especialmente en algoritmos como las máquinas de vectores de soporte (SVM), los modelos de lenguaje estadístico y el aprendizaje profundo, el NLP comenzó a beneficiarse del enfoque de machine learning. Los modelos de aprendizaje automático se entrenaban en grandes conjuntos de datos etiquetados para tareas específicas de procesamiento del lenguaje, como la clasificación de texto, el análisis de sentimientos y la extracción de información. Esto permitió una mayor precisión y capacidad de generalización en diversas tareas.

- **Deep Learning (2014 - Actualidad):**
En los últimos años, el NLP ha experimentado un avance significativo gracias al uso generalizado de las redes neuronales profundas (deep learning). Las arquitecturas de redes neuronales como las redes neuronales recurrentes (RNN) y las redes neuronales convolucionales (CNN) se aplicaron al procesamiento del lenguaje, y modelos como las redes neuronales de transformadores (como BERT y GPT) revolucionaron la forma en que se abordan tareas como la traducción automática, la generación de texto y el entendimiento del lenguaje. Estos modelos aprovechan grandes cantidades de datos no etiquetados y técnicas de aprendizaje no supervisado para aprender representaciones de lenguaje altamente contextuales y capturar relaciones complejas en el texto.

![4.png](ims%2F1%2F4.png)

EL NLP ha tenido 2 vertientes de progreso: Entendimiento de texto (bajo nivel) y Aprendizaje de representaciones.

1. Entendimiento de texto (bajo nivel)

   - **Morfología:** En el campo del NLP, la morfología se refiere al estudio de la estructura y formación de palabras. Los avances en este aspecto han incluido el desarrollo de algoritmos y técnicas para el análisis morfológico, que permite identificar y descomponer palabras en sus componentes más pequeños, como raíces, prefijos y sufijos. Esto es útil para tareas como lematización, reconocimiento de entidades nombradas y generación de formas flexionadas.

   - **Sintaxis:** La sintaxis se centra en el análisis de las estructuras gramaticales de las oraciones. Los avances en sintaxis han permitido el desarrollo de modelos y algoritmos para el análisis sintáctico automático, que consiste en etiquetar y analizar la función gramatical de cada palabra en una oración. Esto es fundamental para tareas como el análisis de dependencias y el análisis sintáctico en árboles.

   - **Semántica:** La semántica se ocupa del significado del lenguaje. Los avances en este ámbito han incluido el desarrollo de modelos y enfoques para capturar y representar el significado de las palabras y las oraciones. Esto ha permitido la creación de modelos semánticos que pueden realizar tareas como la desambiguación del sentido de las palabras, el análisis de sentimientos y la respuesta a preguntas basada en el significado.

2. Aprendizaje de representaciones

   - **Vectores de palabras:** Los vectores de palabras (también conocidos como word embeddings) son representaciones numéricas densas que capturan el significado y la relación entre las palabras. Los avances en este campo han incluido modelos como Word2Vec y GloVe, que utilizan técnicas de aprendizaje no supervisado para generar vectores de palabras a partir de grandes corpus de texto. Estos vectores permiten la representación semántica de las palabras y se utilizan en diversas tareas de NLP, como la similitud de palabras, la clasificación de textos y la traducción automática.

   - **Vectores de frases:** Los vectores de frases (también conocidos como sentece embeddings) buscan representar oraciones y textos completos en un espacio vectorial. Estos avances han permitido el desarrollo de modelos como Doc2Vec e InferSent, que generan representaciones vectoriales de mayor nivel para textos más largos. Estos vectores de frases se utilizan para tareas como la clasificación de documentos, el resumen automático y la búsqueda semántica de documentos.

   - **Mecanismos de atención:** Los mecanismos de atención son avances fundamentales en el procesamiento del lenguaje natural. Permiten a los modelos de NLP prestar atención selectiva a partes específicas de una secuencia de palabras durante el procesamiento. Esto ha impulsado el desarrollo de modelos como los transformers, que utilizan la atención para capturar relaciones y dependencias entre palabras en un contexto más amplio. Los transformers han logrado avances significativos en tareas como la traducción automática, el procesamiento del lenguaje natural basado en modelos (como BERT) y la generación de texto coherente.

![5.png](ims%2F1%2F5.png)

1. **LSTM (Long Short-Term Memory)**: LSTM es una arquitectura de red neuronal recurrente que se introdujo en el campo del NLP. A diferencia de las redes neuronales recurrentes tradicionales, las LSTM están diseñadas para manejar de manera más efectiva el problema del desvanecimiento y la explosión del gradiente. Esto permite a las LSTM capturar dependencias a largo plazo en secuencias de texto y ha sido ampliamente utilizado en tareas como el etiquetado de partes del discurso, el análisis de sentimientos y la generación de texto coherente.

2. **BiLSTM (Bidirectional LSTM)**: Las redes neuronales BiLSTM son una extensión de las LSTM que permiten capturar información contextual tanto de izquierda a derecha como de derecha a izquierda en una secuencia de texto. Esto significa que la red puede tomar en cuenta tanto el contexto anterior como el posterior de cada palabra, lo que ha demostrado ser beneficioso en tareas como el reconocimiento de entidades nombradas, la desambiguación del sentido de las palabras y la traducción automática.

3. **Transformer:** El Transformer es una arquitectura de red neuronal que revolucionó el campo del NLP. Introducida en el artículo "Attention is All You Need" de Vaswani et al. (2017), el Transformer se basa en mecanismos de atención para capturar relaciones entre palabras en una secuencia de manera más efectiva. Esta arquitectura se ha convertido en la base de modelos de vanguardia en NLP, como BERT (Bidirectional Encoder Representations from Transformers) y GPT (Generative Pre-trained Transformer). Los modelos Transformer han mejorado significativamente el desempeño en tareas como la traducción automática, el entendimiento del lenguaje y la generación de texto coherente.

4. **Reformer:** El Reformer es una variante del Transformer que se enfoca en mejorar la eficiencia computacional y el manejo de secuencias largas. Utiliza técnicas como la atención esparsa y la compresión de datos para reducir la complejidad de los cálculos en las redes Transformer. Esto permite manejar secuencias más largas y entrenar modelos más grandes con una mayor eficiencia. El Reformer ha sido utilizado en tareas como el procesamiento de documentos extensos, la traducción automática de larga distancia y la generación de texto a gran escala.


A lo largo del curso vamos a seguir el siguiente learning path:

![6.png](ims%2F1%2F6.png)


## 1.3 Conceptos básicos de NLP

El lenguaje es un sistema de comunicación utilizado por los seres humanos para expresar y transmitir ideas, pensamientos, emociones y conocimientos. Es una facultad exclusiva de los seres humanos y se considera una de las principales características que nos distingue de otras especies.

![7.png](ims%2F1%2F7.png)

Antes de empezar a trabajar con texto es necesario conocer algunos de los conceptos más fundamentales que manejamos en el tema de NLP.

![8.png](ims%2F1%2F8.png)

La normalización de texto consiste, en principio, en varios procesos de limpieza y transformación de los cuales podemos mencionar: tokenización, lematización, segmentación. 

- **Tokenización:** La tokenización es el proceso de dividir un texto en unidades más pequeñas llamadas "tokens". Estos tokens pueden ser palabras, frases, símbolos de puntuación o incluso caracteres individuales, dependiendo del nivel de granularidad deseado. La tokenización es un paso fundamental en el procesamiento del lenguaje natural, ya que permite trabajar con unidades discretas y facilita el análisis y procesamiento posterior del texto.

- **Lematización:** La lematización es el proceso de reducir las palabras a su forma base o "lema". Un lema es la forma canónica o raíz de una palabra, y la lematización busca identificar esa forma base para cada palabra en un texto. Por ejemplo, la lematización convertiría las palabras "corriendo", "corre" y "corrió" al lema "correr". La lematización es útil para normalizar las palabras y reducir la dimensionalidad del vocabulario, lo que ayuda a mejorar la precisión y eficiencia de los modelos de procesamiento del lenguaje.

- **Segmentación:** La segmentación se refiere al proceso de dividir un texto en unidades más pequeñas, como oraciones o párrafos. En el contexto de la normalización de texto, la segmentación se centra principalmente en la división de un texto en oraciones. Esto es importante para tareas como el análisis de sentimiento, la traducción automática y el resumen automático, ya que muchas técnicas y modelos de procesamiento del lenguaje operan a nivel de oración. La segmentación puede implicar el uso de reglas gramaticales, puntuación o incluso modelos de aprendizaje automático entrenados específicamente para esta tarea.

![9.png](ims%2F1%2F9.png)

En el ámbito del procesamiento del lenguaje natural (NLP), el término "corpus" se refiere a una colección o conjunto de textos escritos o hablados que se utilizan como recurso lingüístico para llevar a cabo investigaciones, análisis o entrenar modelos de lenguaje. Un corpus es una muestra representativa de datos lingüísticos recopilados y organizados de manera sistemática.

Un corpus puede ser compilado de diversas fuentes, como libros, artículos de periódicos, transcripciones de conversaciones, páginas web, documentos legales, entre otros. Puede ser diseñado para cubrir un dominio específico, como el corpus médico, el corpus legal o el corpus literario, o puede ser más general y abarcar un amplio rango de textos de diferentes temas y géneros.

El término "corpora" se utiliza para hacer referencia al plural de "corpus". Por lo tanto, cuando se habla de "corpora", se hace referencia a múltiples colecciones o conjuntos de textos utilizados en NLP o lingüística.

Los corpora desempeñan un papel fundamental en el desarrollo y avance del campo del procesamiento del lenguaje natural, ya que proporcionan los datos necesarios para entrenar y evaluar modelos de lenguaje, y permiten realizar investigaciones empíricas sobre el lenguaje humano en diversos contextos.

# 2 Fundamentos con NLTK

(Natural Language Toolkit) es una biblioteca de Python muy popular y ampliamente utilizada para el procesamiento del lenguaje natural (NLP, por sus siglas en inglés). Proporciona una amplia gama de herramientas y recursos para trabajar con texto y datos de lenguaje humano.

![1.png](ims%2F2%2F1.png)

Entre los usos más populares para los cuales podemos emplear NLTK podemos mencionar:

- **Preprocesamiento de texto:** NLTK ofrece una variedad de funciones y métodos para realizar tareas de preprocesamiento de texto, como tokenización (división de texto en palabras o frases), eliminación de puntuación, normalización de texto, lematización y etiquetado gramatical.

- **Colecciones de corpus:** NLTK proporciona una amplia gama de corpus (conjuntos de datos de texto) en diferentes idiomas y dominios. Estos corpus incluyen textos de libros, discursos, chats, etiquetas gramaticales, entre otros. También ofrece herramientas para trabajar con corpus personalizados.

- **Modelos de aprendizaje automático:** NLTK facilita la construcción y el entrenamiento de modelos de aprendizaje automático para tareas de procesamiento del lenguaje natural. Proporciona algoritmos para clasificación, análisis de sentimientos, extracción de información, segmentación de texto y más. Además, ofrece interfaces para integrar modelos de bibliotecas externas, como scikit-learn.

- **Visualización de datos:** NLTK incluye funciones para visualizar datos de lenguaje natural, como distribuciones de frecuencia de palabras, nubes de palabras, gráficos de dispersión y árboles sintácticos. Estas visualizaciones pueden ayudar en el análisis y comprensión de los datos de texto.

- **Recursos léxicos:** NLTK proporciona acceso a varios recursos léxicos, como WordNet, que es una base de datos léxica ampliamente utilizada que contiene sinónimos, antónimos, hiperónimos y otras relaciones semánticas entre las palabras. Estos recursos pueden ser útiles para tareas de desambiguación, expansión de consultas y generación de palabras clave.



## 2.1 Configurar ambiente de trabajo

Para empezar, primero debemos instalar la biblioteca [NLTK](https://www.nltk.org/)

```bash
pip install nltk
```
Respuesta esperada:
```commandline
Collecting nltk
  Downloading nltk-3.8.1-py3-none-any.whl (1.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 2.8 MB/s eta 0:00:00
Requirement already satisfied: click in ./venv/lib/python3.10/site-packages (from nltk) (8.1.3)
Requirement already satisfied: joblib in ./venv/lib/python3.10/site-packages (from nltk) (1.2.0)
Requirement already satisfied: regex>=2021.8.3 in ./venv/lib/python3.10/site-packages (from nltk) (2023.3.23)
Requirement already satisfied: tqdm in ./venv/lib/python3.10/site-packages (from nltk) (4.65.0)
Installing collected packages: nltk
Successfully installed nltk-3.8.1
```

> ## Nota:
> El código de esta sección lo puedes encontrar en: [1_configurar_ambiente_de_trabajo.py](scripts%2F1_configurar_ambiente_de_trabajo.py)

Primero vamos a empezar por importar la biblioteca y descargar nuestro primer ejemplo de corpus de texto en español
el cual es [cess_esp](https://mailman.uib.no/public/corpora/2007-October/005448.html)

```python
import nltk

# Descargamos un corpus en español
nltk.download('cess_esp')
```
Respuesta esperada:
```commandline
[nltk_data] Downloading package cess_esp to
[nltk_data]     /home/ichcanziho/nltk_data...
[nltk_data]   Unzipping corpora/cess_esp.zip.
```

Inspeccionamos nuestro corpus descargado

```python
# Cargamos las oraciones en la variable corpus
corpus = nltk.corpus.cess_esp.sents()
print(corpus)
print(len(corpus))
```
Respuesta esperada:
```commandline
[['El', 'grupo', 'estatal', 'Electricité_de_France', '-Fpa-', 'EDF', '-Fpt-', 'anunció', 'hoy', ',', 'jueves', ',', 'la', 'compra', 'del', '51_por_ciento', 'de', 'la', 'empresa', 'mexicana', 'Electricidad_Águila_de_Altamira', '-Fpa-', 'EAA', '-Fpt-', ',', 'creada', 'por', 'el', 'japonés', 'Mitsubishi_Corporation', 'para', 'poner_en_marcha', 'una', 'central', 'de', 'gas', 'de', '495', 'megavatios', '.'], ['Una', 'portavoz', 'de', 'EDF', 'explicó', 'a', 'EFE', 'que', 'el', 'proyecto', 'para', 'la', 'construcción', 'de', 'Altamira_2', ',', 'al', 'norte', 'de', 'Tampico', ',', 'prevé', 'la', 'utilización', 'de', 'gas', 'natural', 'como', 'combustible', 'principal', 'en', 'una', 'central', 'de', 'ciclo', 'combinado', 'que', 'debe', 'empezar', 'a', 'funcionar', 'en', 'mayo_del_2002', '.'], ...]
6030
```
Podemos observar como nuestro corpus es una lista de oraciones en donde cada oración es en si misma otra lista con las palabras
ya tokenizadas. Podemos observar como nuestro corpus tiene `6030` oraciones. Sin embargo, ahora nos interesa tener una sola lista con todas las palabras disponibles no una lista de listas, entonces
vamos a hacer `plana` nuestra lista.

```python
# Convertimos nuestro arreglo bidimensional en unidimensional
flatten = [item for sublist in corpus for item in sublist]
print(flatten[0:20])
print(len(flatten))
```
Respuesta esperada:
```commandline
['El', 'grupo', 'estatal', 'Electricité_de_France', '-Fpa-', 'EDF', '-Fpt-', 'anunció', 'hoy', ',', 'jueves', ',', 'la', 'compra', 'del', '51_por_ciento', 'de', 'la', 'empresa', 'mexicana']
192686
```
Genial ahora tenemos todas las palabras dentro de una sola lista y observamos que en total tenemos `192,686` palabras, más adeltante
veremos métricas que nos ayuden a entender un poco más la riqueza de este corpus.

## 2.2 Palabras, textos y vocabularios

En esta clase vamos a ver un par de ejemplos de como podemos utilizar la biblioteca de `re` para utilizar `expresiones regulares` en python.

> Las expresiones regulares (regexp o regex) son secuencias de caracteres que forman un patrón de búsqueda. Se utilizan para buscar y manipular texto de manera eficiente. Las expresiones regulares permiten realizar tareas como encontrar coincidencias de patrones, validar y extraer información específica de un texto.

En `python` podemos utilizar la biblioteca `re`para utilizar estas expresiones regulares.
Un ejemplo básico de uso de `re` es el siguiente:

```python
import re

texto = "Hola, ¿cómo estás?"
patron = r"estás"  # El prefijo 'r' indica una cadena de texto cruda (raw string)

coincidencia = re.search(patron, texto)
if coincidencia:
    print("Se encontró una coincidencia")
else:
    print("No se encontraron coincidencias")
```

Ahora con base en el código de la clase pasada veamos un par de expresiones regulares que nos pueden ser útiles para crear
nuestros vocabularios limpios.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [2_palabras_textos_vocabularios.py](scripts%2F2_palabras_textos_vocabularios.py)

De la clase anterior tenemos:

```python
# Clase 1
import nltk
corpus = nltk.corpus.cess_esp.sents()
flatten = [item for sublist in corpus for item in sublist]
```

Continuemos con unas búsquedas muy simples. Como el ejemplo anterior, si queremos buscar que una palabra o una secuencia de texto se encuentre dentro
de otra podemos utilizar `re.search(patron, texto)`

```python
# Clase 2
import re

# Meta-caracteres básicos
arr = [w for w in flatten if re.search('es', w)]
print(arr[:5])
```
Respuesta esperada:
```commandline
['estatal', 'jueves', 'empresa', 'centrales', 'francesa']
```
El código nos permitio encontrar 5 palabras que contengan la secuencia `es` dentro de si mismas.

Pero algo un poco más avanzado sería preguntar por las palabras que inicien o terminen con la secuencia `es` para ello hacemos:
```python
# Utilizamos $ para buscar palabras que terminen con un patron 
arr = [w for w in flatten if re.search('es$', w)]
print(arr[:5])

# Utilizamos ^ para buscar palabras que inicien con un patron 
arr = [w for w in flatten if re.search('^es', w)]
print(arr[:5])
```
Respuesta esperada:
```commandline
['jueves', 'centrales', 'millones', 'millones', 'dólares']
['estatal', 'es', 'esta', 'esta', 'eso']
```
Ahora un reto un poco más interesante. Podemos combinar los comandos anteriores `$` y `^` para buscar palabras que inicien
y terminen de una forma concreta, sin embargo, podemos usar una secuencia `comodin` para indicar que puede haber cualquier 
cantidad de caracteres presentes usando `..`.

```python
arr = [w for w in flatten if re.search('^..j..t..$', w)]
print(arr)
```
Respuesta esperada:
```commandline
['tajantes']
```
La palabra `tajantes` cumple con nuestra expresión regular, porque la expresión se puede leer así:

- `^..` > una palabra que inicie con cualquier letra y con cualquier cantidad de ellas
- `j` > que eventualmente encuentre una letra `j`
- `..` > seguido de cualquier letra cualquier cantidad de veces
- `t` > que eventualmente encuentre una letra `t`
- `..$` > y que finalmente termine con cualquier letra cualquier cantidad de veces

Algo interesante es que podemos utilizar `rangos` para buscar secuencias específicas de letras entre una lista de letras disponibles.

```python
# Rangos [a-z], [A-Z], [0-9]
arr = [w for w in flatten if re.search('^[ghi][mno][jlk][def]$', w)]
print(arr)
```
Respuesta esperada:
```commandline
['golf', 'golf']
```
El código anterior permite encontrar una palabra de 4 letras donde la primer letra puede ser cualquiera entre `(g, h, i)` y la siguiente
puede ser entre `(m, n, o)` de esta manera una palabra que cumple con esta expresión es `golf`

Finalmente, podemos utilizar `Kleene closures` para decir si nos interesa que aparezca un patron 0 o más veces o al menos 1 vez:

```python
# Clausuras *, * (Kleene closures)
# El comando * es para indicar un 0 o más veces (osea que NO es indispensable que aparezca el patron)
arr = [w for w in flatten if re.search('^(no)*', w)]
print(arr[:10])
# El comando + es para indicar que necesita aparecer al menos 1 vez
arr = [w for w in flatten if re.search('(no)+', w)]
print(arr[:10])
```
Respuesta esperada:
```commandline
['El', 'grupo', 'estatal', 'Electricité_de_France', '-Fpa-', 'EDF', '-Fpt-', 'anunció', 'hoy', ',']
['norte', 'no', 'no', 'noche', 'no', 'no', 'gobierno', 'notificación', 'Unión_Fenosa_Inversiones', 'italiano']
```

Como ejercicio extra, una vez que hemos encontrado un cierto patron en el texto podemos utilizar group() para extraer ese texto
e incluso agruparlo en caso de que el patron contenga varias secuencias.

```python
texto = "Mi número de teléfono es 123-456-7890"
patron = r"(\d{3})-(\d{3})-(\d{4})"

coincidencia = re.search(patron, texto)
if coincidencia:
    numero_telefono = coincidencia.group()  # Obtiene la coincidencia completa
    area = coincidencia.group(1)  # Obtiene el grupo de captura 1 (código de área)
    print("Número de teléfono:", numero_telefono)
    print("Código de área:", area)
```
Respuesta esperada:
```commandline
Número de teléfono: 123-456-7890
Código de área: 123
```

## 2.3 Tokenizacion con Expresiones Regulares

> **Normalización de Texto** (como aplicación de las expresiones regulares)
**Tokenización:** Es el proceso mediante el cual se sub-divide una cadena de texto en unidades linguísticas minimas (palabras)

Como ya conocemos un poco de `regexp` podemos utilizar este conocimiento para crear un sistema de tokenizado que nos permita
separar las palabras de una oración.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [3_tokenizado_con_regexp.py](scripts%2F3_tokenizado_con_regexp.py)

Empecemos importando bibliotecas y descargando paquetes que vamos a necesitar:

```python
import re
import nltk

nltk.download('wordnet')
print("="*64)
nltk.download('punkt')
print("="*64)
```
Respuesta esperada:
```commandline
[nltk_data] Downloading package wordnet to
[nltk_data]     /home/ichcanziho/nltk_data...
[nltk_data] Downloading package punkt to /home/ichcanziho/nltk_data...
================================================================
[nltk_data]   Unzipping tokenizers/punkt.zip.
================================================================
```

Ahora para probar nuestros tokenizadores veamos un texto de ejemplo:

```python
texto = """ Cuando sea el rey del mundo  (imaginaba él en su cabeza) no tendré que  preocuparme por estas bobadas.
            Era solo un niño de 7 años, pero pensaba que podría ser cualquier cosa que su imaginación le permitiera visualizar en su cabeza ..."""

print(texto)
```
respuesta esperada:
```commandline
 Cuando sea el rey del mundo  (imaginaba él en su cabeza) no tendré que  preocuparme por estas bobadas.
            Era solo un niño de 7 años, pero pensaba que podría ser cualquier cosa que su imaginación le permitiera visualizar en su cabeza ...
```
Quizá lo más lógico es partir por la idea de que las palabras se separan por espacios entonces hacer un split por ' ' no parece
una mala idea.

```python
# Caso 1: tokenizacion más simple: por espacios vacios !
print(re.split(r' ', texto))
```
Respuesta esperada:
```commandline
['', 'Cuando', 'sea', 'el', 'rey', 'del', 'mundo', '', '(imaginaba', 'él', 'en', 'su', 'cabeza)', 'no', 'tendré', 'que', '', 'preocuparme', 'por', 'estas', 'bobadas.\n', '', '', '', '', '', '', '', '', '', '', '', 'Era', 'solo', 'un', 'niño', 'de', '7', 'años,', 'pero', 'pensaba', 'que', 'podría', 'ser', 'cualquier', 'cosa', 'que', 'su', 'imaginación', 'le', 'permitiera', 'visualizar', 'en', 'su', 'cabeza', '...']
```
Observamos que no es perfecto porque hay palabras que tienen más de un espacio seguido o que tienen saltos de línea `\n` y este tokenizador no lidia con esos problemas,
Podemos utilizar una variación ligeramente más inteligente que es `\s` que indica cualquier separador no solo espacios.

```python
# Caso 2: tokenización usando expresiones regulares
print(re.split(r'[ \s]+', texto))
```
Respuesta esperada:
```commandline
['', 'Cuando', 'sea', 'el', 'rey', 'del', 'mundo', '(imaginaba', 'él', 'en', 'su', 'cabeza)', 'no', 'tendré', 'que', 'preocuparme', 'por', 'estas', 'bobadas.', 'Era', 'solo', 'un', 'niño', 'de', '7', 'años,', 'pero', 'pensaba', 'que', 'podría', 'ser', 'cualquier', 'cosa', 'que', 'su', 'imaginación', 'le', 'permitiera', 'visualizar', 'en', 'su', 'cabeza', '...']
```
Mejoro respecto al pasado, pero aún tiene problemas con identificar palabras por ejemplo: `'(imaginaba'` o `cabeza)`
Entonces podemos utilizar la expresión `\W\s` letras sin contar signos de puntuación o números.

```python
# RegEx reference: \W -> all characters other than letters, digits or underscore
print(re.split(r'[ \W\s]+', texto))
```
Respuesta esperada:
```commandline
['', 'Cuando', 'sea', 'el', 'rey', 'del', 'mundo', 'imaginaba', 'él', 'en', 'su', 'cabeza', 'no', 'tendré', 'que', 'preocuparme', 'por', 'estas', 'bobadas', 'Era', 'solo', 'un', 'niño', 'de', '7', 'años', 'pero', 'pensaba', 'que', 'podría', 'ser', 'cualquier', 'cosa', 'que', 'su', 'imaginación', 'le', 'permitiera', 'visualizar', 'en', 'su', 'cabeza', '']
```
Ese código mejora los resultados, pero sigue siendo un tokenizador muy simple que falla con textos un poco más complejos, por
ejemplo tomemos el siguiente caso:

```python
# nuestra antigua regex no funciona en este caso:
texto = 'En los E.U. esa postal vale $15.50 ...'

print(re.split(r'[ \W\s]+', texto))
```
Respuesta esperada:
```commandline
['En', 'los', 'E', 'U', 'esa', 'postal', 'vale', '15', '50', '']
```
Nuestro tokenizador simple no logra entender siglas o dinero con centavos.

Podemos utilizar un patron mucho más inteligente y completo y usarlo junto con `nltk.regexp_tokenize` para justamente tener nuestro
tokenizador.

```python
pattern = r'''(?x)                 # set flag to allow verbose regexps
              (?:[A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*       # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
              | \.\.\.             # ellipsis
              | [][.,;"'?():-_`]   # these are separate tokens; includes ], [
'''

print(nltk.regexp_tokenize(texto, pattern))
```
Respuesta esperada:
```commandline
['En', 'los', 'E.U.', 'esa', 'postal', 'vale', '$15.50', '...']
```
Esta respuesta es muy limpia e interesante. Sin embargo, también podemos utilizar un tokenizador en español pre-entrenado.

```python
from nltk import word_tokenize

print(word_tokenize(texto, language="spanish"))
```
Respuesta esperada:
```commandline
['En', 'los', 'E.U', '.', 'esa', 'postal', 'vale', '$', '15.50', '...']
```

### Normalizando las palabras

En esta breve sección vamos a explicar los conceptos de `Stemming` y `Lemmatization`.

Tanto los stemmers como los lematizadores son herramientas utilizadas en el procesamiento del lenguaje natural (NLP) para reducir las palabras a su forma base o raíz, lo que ayuda a normalizar el texto y reducir la variación morfológica. Aunque tienen objetivos similares, se diferencian en su enfoque y resultados.

**Stemming (Stemmer):**

El stemming (o el stemmer) es un proceso de reducción de palabras a su raíz o "stem" eliminando afijos o sufijos comunes. El resultado a menudo no es una palabra real o legible en el lenguaje natural, pero aún puede ser útil para ciertos análisis de texto. El stemming es un enfoque basado en reglas heurísticas simples y no tiene en cuenta el contexto de las palabras.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runs", "ran"]

for word in words:
    stemmed_word = stemmer.stem(word)
    print(word, "->", stemmed_word)
```
Respuesta esperada:
```commandline
running -> run
runs -> run
ran -> ran
```
En el ejemplo, el stemmer de Porter de NLTK reduce las palabras a su forma base eliminando sufijos comunes.

**Lemmatization (Lemmatizer):**

La lematización (o el lematizador) es un enfoque más avanzado que busca la forma base de una palabra, conocida como "lematización", conservando la legibilidad y validez en el lenguaje natural. A diferencia del stemming, la lematización tiene en cuenta el contexto y la morfología de las palabras. Los lematizadores utilizan reglas lingüísticas y bases de datos léxicas (como WordNet) para obtener el lema de una palabra.
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran"]

for word in words:
    lemmatized_word = lemmatizer.lemmatize(word)
    print(word, "->", lemmatized_word)
```
Respuesta esperada:
```commandline
running -> running
runs -> run
ran -> ran
```

En este ejemplo, el lematizador de NLTK produce lemas válidos para las palabras en contexto, teniendo en cuenta las diferentes formas gramaticales.

## 2.4 Estadísticas básicas del lenguaje

Empecemos importando las bibliotecas y datos necesarios:

```python
import nltk
nltk.download('book')
from nltk.book import *
```
Respuesta esperada:
```commandline
*** Introductory Examples for the NLTK Book ***
Loading text1, ..., text9 and sent1, ..., sent9
Type the name of the text or sentence to view it.
Type: 'texts()' or 'sents()' to list the materials.
text1: Moby Dick by Herman Melville 1851
text2: Sense and Sensibility by Jane Austen 1811
text3: The Book of Genesis
text4: Inaugural Address Corpus
text5: Chat Corpus
text6: Monty Python and the Holy Grail
text7: Wall Street Journal
text8: Personals Corpus
text9: The Man Who Was Thursday by G . K . Chesterton 1908
```
Ahora sabemos que tenemos 9 libros a nuestra disposición, vamos a analizar el primero de ellos
`Moby Dick`:

```python
# escogemos text1 que es el famoso libro Moby Dick
print(text1)

# Vemos que el texto ya viene tokenizado incluyendo caracteres especiales ....
print(text1.tokens[:10])

# ¿Cuantos tokens tiene el libro?
print(len(text1))
```
Respuesta esperada:
```commandline
<Text: Moby Dick by Herman Melville 1851>
['[', 'Moby', 'Dick', 'by', 'Herman', 'Melville', '1851', ']', 'ETYMOLOGY', '.']
260819
```
Ahora sabemos que hay un total de 260,819 palabras en el texto, pero nos interesa obtener la cantidad de palabras únicas
y con ello construir nuestro vocabulario. Con esta información podemos definir: **Medida de riqueza lexica en un texto:**

![formula](https://latex.codecogs.com/gif.download?%5Cdpi%7B120%7D%20%5Cbg_white%20R_%7Bl%7D%20%3D%20%5Cfrac%7B%5Ctext%7Btotal%20de%20palabras%20unicas%7D%7D%7B%5Ctext%7Btotal%20de%20palabras%7D%7D%20%3D%20%5Cfrac%7B%5Ctext%7Blongitud%20del%20vocabulario%7D%7D%7B%5Ctext%7Blongitud%20del%20texto%7D%7D)

```python
# Primero realizamos la construcción de un vocabulario (identificamos las palabras unicas que hay en el libro)
vocabulario = sorted(set(text1))
print(len(vocabulario))
print(vocabulario[1000:1050])

# luego definimos la medida de riqueza léxica:
rl = len(set(text1)) / len(text1)
print(rl)
```
Respuesta esperada:

```commandline
19317
['Crew', 'Crish', 'Crockett', 'Cross', 'Crossed', 'Crossing', 'Crotch', 'Crowding', 'Crown', 'Crozetts', 'Cruelty', 'Cruising', 'Cruppered', 'Crusaders', 'Crushed', 'Crying', 'Cuba', 'Curious', 'Curse', 'Cursed', 'Curses', 'Cussed', 'Customs', 'Cut', 'Cutter', 'Cutting', 'Cuvier', 'Cyclades', 'Czar', 'D', 'DAGGOO', 'DAM', 'DANCE', 'DANCING', 'DANIEL', 'DANISH', 'DARKENS', 'DARWIN', 'DAVENANT', 'DEAD', 'DEATH', 'DEBELL', 'DECK', 'DEL', 'DESTROYED', 'DEVIL', 'DICTIONARY', 'DID', 'DIGNITY', 'DISCOVERS']
0.07406285585022564
```

De esta forma podemos observar que el libro, sin una limpieza en las palabras, cuenta con 19,317 palabras diferentes, lo cuáñ
representa una riqueza léxica del 7.4%.

```python
# podemos definir funciones en python para estas medidas léxicas:
def riqueza_lexica(texto):
    return len(set(texto)) / len(texto)


def porcentaje_palabra(palabra, texto):
    c = texto.count(palabra)
    return 100 * c / len(texto), c


print(riqueza_lexica(text1))
print(porcentaje_palabra("monster", text1))
```
Respuesta esperada:
```commandline
0.07406285585022564
(0.018786974875296663, 49)
```

## 2.5 Distribuciones de frecuencia de palabras

> ## Nota:
> El código completo de esta sección lo puedes encontrar en: [5_distribucion_palabras.py](scripts%2F5_distribucion_palabras.py)

Los cálculos estadísticos más simples que se pueden efectuar sobre un texto o un corpus son los relacionados con frecuencia de aparición de palabras.

* Podemos construir un diccionario en Python donde las llaves sean las palabras y los valores sean las frecuencias de ocurrencias de esas palabras.

* ejemplo `dic = {'monster': 49 ,  'boat': 54,  ...}`

El método más simple que se nos puede ocurrir para esta tarea sería recorrer cada una de las palabras diferentes dentro de nuestro
vocabulario y contar cada una de ellas y almacenar en un diccionario la palabra como `key` y la frecuencia como `value`:

```python
start = time()
# METODO NO recomendable para conjuntos muy grandes
dic = {}
for palabra in set(text1):
    # dic[palabra] = porcentaje_palabra(palabra, text1)
    dic[palabra] = text1.count(palabra)

print("Execution's time:", time()-start, "seconds")
print(dic)
```
Respuesta esperada:
```commandline
Execution's time: 30.865009546279907 seconds
{'claret': 1, 'trowsers': 13, 'observation': 5, 'Won': 1, 'arises': 1, 'panting': 3, 'Sebastian': 9, 'dedicates': 1, 'expediency': 1, 'coined': 1, 'Dusk': 1, 'Sub': 7, 'goggling': 1, 'screamed': 1, 'whistled': 1, 'fibrous': 1, 'wander': 1, 'opening': 14, 'anchor': 21, 'BLOODY': 1, 'extant': 3, 'BLACK': 2,...}
```
Sin embargo, este método es muy ineficiente y NLTK proporciona un método mucho más simple y eficaz.
```python
start = time()
# NLTK tiene un metodo muy eficiente
fdist = FreqDist(text1)
print("Execution's time:", time()-start, "seconds")
print(fdist.most_common(20))
```
Respuesta esperada:
```commandline
Execution's time: 0.0629119873046875 seconds
[(',', 18713), ('the', 13721), ('.', 6862), ('of', 6536), ('and', 6024), ('a', 4569), ('to', 4542), (';', 4072), ('in', 3916), ('that', 2982), ("'", 2684), ('-', 2552), ('his', 2459), ('it', 2209), ('I', 2124), ('s', 1739), ('is', 1695), ('he', 1661), ('with', 1659), ('was', 1632)]
```
Adicinoalmente el objeto `fdist` tiene un método `plot`:
```python
fdist.plot(20)
plt.savefig("i1.png")
plt.close()
```
Respuesta esperada:
![i1.png](scripts%2Fi1.png)

## 2.6 Refinamiento y visualización de cuerpos de texto

> ## Nota:
> El código completo de esta sección lo puedes encontrar en: [6_refinamiento.py](scripts%2F6_refinamiento.py)

En este código veremos una forma más detallada de limpiar y normalizar nuestro texto para gráficar las 20 palabras más comunes.

Empezamos importando nuestras bibliotecas:

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from nltk.book import text1
import string
import re
```
Ahora creamos un `lematizador` y un conjunto de `stop words` que eliminaremos del texto original:

```python
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
```

Ahora creamos una función que limpie el texto por nosotros utilizando las variables anteriormente definidas.

```python
def clean_corpus(corpus, sw, lem, min_words, most_common):
    """
    Limpia un corpus de texto aplicando diferentes filtros.

    Args:
        corpus (list): Corpus de texto a limpiar.
        sw (set): Conjunto de palabras vacías (stop words) a eliminar.
        lem (WordNetLemmatizer): Objeto lematizador de palabras.
        min_words (int): Mínimo número de palabras en una palabra filtrada.
        most_common (int): Número de palabras más comunes a extraer.

    Returns:
        tuple: Una tupla que contiene el diccionario de palabras filtradas y una lista de las palabras más comunes.
    """
    # empezamos minimizando todas las palabras y eliminando las palabras que sean stopwords
    filtered_corpus = [word.lower() for word in corpus if word.lower() not in sw]
    # eliminamos signos de puntación
    filtered_corpus = [word for word in filtered_corpus if word not in string.punctuation]
    # eliminamos palabras que tengan una longitud menor o igual a 5 letras
    filtered_corpus = [word for word in filtered_corpus if len(word) > 5]
    # eliminamos letras que no sean texto
    filtered_corpus = [word for word in filtered_corpus if not re.match(r'\d', word)]
    # lematizamos cada palabra
    filtered_corpus = [lem.lemmatize(word) for word in filtered_corpus]
    # obtenemos un contador de las palabras en el corpus
    filtered_corpus = Counter(filtered_corpus)
    # guardamos las palabras más comunes
    mc = filtered_corpus.most_common(most_common)
    # eliminamos palabras cuya frecuencia sea menor a un umbral
    elementos_filtrados = {clave: valor for clave, valor in filtered_corpus.items() if valor > min_words}
    # regresamos el contador y las palabras más comunes
    return elementos_filtrados, mc
```

Finalmente, creamos una función para graficar nuestra lista de palabras más comunes:

```python
def plot_freq(elementos):
    """
       Grafica la frecuencia de las palabras en un diccionario.

       Args:
           elementos (list): Lista de tuplas que contienen palabras y sus frecuencias.
       """
    palabras = [elemento[0] for elemento in elementos]
    frecuencias = [elemento[1] for elemento in elementos]
    plt.plot(palabras, frecuencias)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de palabras')
    plt.tight_layout()
    plt.savefig("i2.png")
```

Y ahora mandamos a llamar a nuestras funciones definidas:

```python
if __name__ == '__main__':

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    vocabulario_filtrado, vocabulario_popular = clean_corpus(text1, stop_words, lemmatizer, 5, 20)
    print(vocabulario_filtrado)
    print(vocabulario_popular)
    plot_freq(vocabulario_popular)
```
Respuesta esperada:
```commandline
{'supplied': 12, 'school': 18, 'nation': 17, 'somehow': 44, 'mildly': 10, 'others': 39, 'called': 116, 'tongue': 14, 'leaving': 39, 'ignorance': 11, 'letter': 26, 'almost': 195, 'animal': 22, 'rolling': 36, 'arched': 11, 'dictionary': 6, 'immediately': 17, 'english': 49, 'french': 23, 'spanish': 12, 'extract': 7, 'appears': 7, 'street': 28, 'whatever': 46, 'allusion': 12, 'whale': 268, 'sacred': 10, 'therefore': 67, 'statement': 8, 'however': 95, 'cetology': 10, 'touching': 46, 'ancient': 28, 'author': 12, 'generally': 31, 'solely': 6, 'valuable': 11, 'glancing': 14, 'thought': 184, 'fancied': 9, 'leviathan': 113, 'generation': 6, 'including': 8, 'hopeless': 8, 'strong': 36, 'sometimes': 87, 'devilish': 9, 'glass': 10, 'altogether': 29, 'please': 13, 'heart': 29, 'friend': 43, 'clearing': 6, 'heaven': 68, 'making': 47, 'gabriel': 20, 'coming': 54, 'strike': 37, 'splintered': 7, 'together': 64, 'created': 7, 'prepared': 12, 'swallow': 15, 'therein': 6, 'serpent': 8, 'crooked': 8, 'dragon': 7, 'besides': 50, 'within': 79, 'monster': 58, 'bottomless': 7, 'holland': 7, 'indian': 61, 'length': 82, 'scarcely': 9, 'proceeded': 6, 'sunrise': 11, 'appeared': 10, 'former': 20, 'monstrous': 11, 'towards': 114, 'beating': 11, 'history': 25, 'visited': 7, 'country': 29, 'catching': 7, 'brought': 38, 'killed': 35, 'narrative': 12, 'whereas': 18, 'thing': 132, 'whether': 91, 'vessel': 81, 'dreadful': 6, 'swallowed': 13, 'sleep': 7, 'described': 8, 'prophet': 19, 'patient': 6, 'boiling': 15, 'received': 26, 'nothing': 115, 'certain': 89, 'exceeding': 12, 'incredible': 9, 'quantity': 13, 'secure': 12, 'breast': 8, 'wounded': 9, 'immense': 16, 'motion': 29, 'body': 9, 'peaceful': 6, 'trouble': 11, 'learned': 25, 'thirty': 27, 'plainly': 39, 'browne': 6, 'modern': 19, 'ponderous': 13, 'battle': 32, 'summer': 18, 'island': 51, 'opening': 15, 'without': 156, 'stream': 15, 'living': 73, 'creature': 76, 'stretched': 10, 'promontory': 6, 'moving': 15, 'breath': 25, 'spout': 16, 'mighty': 47, 'swimming': 30, 'behind': 50, 'attend': 10, 'chance': 54, 'mistake': 11, 'floating': 26, 'twelve': 16, 'voyage': 121, 'nature': 44, 'placed': 37, 'shoulder': 25, 'africa': 8, 'forced': 19, 'proceed': 15, 'frequently': 14, 'barrel': 21, 'herring': 6, 'harpooneer': 137, 'caught': 45, 'greenland': 36, 'several': 47, 'afford': 9, 'weight': 17, 'baleen': 11, 'garden': 9, 'master': 25, 'swiftness': 6, 'southern': 12, 'hundred': 58, 'captain': 353, 'attended': 6, 'america': 13, 'special': 15, 'important': 19, 'charge': 11, 'compare': 7, 'respect': 24, 'magnitude': 22, 'appear': 6, 'comparison': 6, 'doubtless': 13, 'largest': 19, 'creation': 8, 'little': 249, 'afternoon': 11, 'supposed': 15, 'towing': 14, 'ashore': 34, 'seemed': 283, 'endeavor': 8, 'larger': 9, 'seldom': 25, 'attack': 12, 'afraid': 15, 'mention': 16, 'article': 16, 'prevent': 6, 'approach': 6, 'active': 8, 'fierce': 11, 'requires': 7, 'fisherman': 35, 'reference': 14, 'nantucket': 96, 'fishery': 69, 'stranded': 12, 'shore': 7, 'somewhere': 24, 'branch': 9, 'ordinary': 19, 'consideration': 14, 'pirate': 16, 'either': 40, 'thrown': 25, 'property': 8, 'barbed': 6, 'bright': 25, 'spire': 6, 'driven': 10, 'around': 38, 'spouted': 6, 'express': 9, 'london': 14, 'fifteen': 11, 'gallon': 7, 'stroke': 18, 'velocity': 10, 'hunter': 48, 'account': 43, 'roaring': 9, 'passage': 35, 'inferior': 11, 'cuvier': 11, 'degree': 31, 'covered': 11, 'purpose': 43, 'extending': 8, 'spermaceti': 19, 'element': 19, 'beneath': 76, 'colour': 21, 'language': 7, 'cannot': 69, 'mariner': 45, 'million': 9, 'gather': 6, 'shoal': 8, 'mysterious': 11, 'instinct': 9, 'region': 9, 'though': 384, 'shark': 51, 'people': 46, 'atlantic': 20, 'person': 34, 'observing': 8, 'spouting': 12, 'observed': 18, 'pointing': 19, 'pasture': 7, 'child': 18, 'setting': 13, 'pacific': 36, 'answered': 22, 'pretty': 29, 'christian': 25, 'fellow': 65, 'cooper': 10, 'paper': 10, 'introduced': 6, 'conversation': 6, 'matter': 100, 'finally': 27, 'destroyed': 10, 'shroud': 18, 'moonlight': 8, 'engaged': 18, 'capture': 14, 'nearly': 34, 'tremendous': 9, 'distance': 41, 'scoresby': 10, 'enormous': 26, 'everything': 20, 'utterly': 13, 'astonishment': 6, 'interesting': 8, 'entirely': 29, 'excited': 8, 'curiosity': 10, 'numerous': 16, 'possessed': 7, 'convenient': 11, 'opportunity': 6, 'better': 64, 'weapon': 17, 'extremity': 11, 'manner': 38, 'regarded': 24, 'dangerous': 13, 'specie': 31, 'whaling': 131, 'demanded': 11, 'point': 14, 'steady': 29, 'breach': 9, 'thunder': 34, 'lightning': 31, 'occurred': 10, 'belonged': 13, 'hussey': 17, 'pursued': 10, 'assault': 9, 'furious': 7, 'rushed': 15, 'comrade': 19, 'leaping': 14, 'striking': 24, 'peculiar': 56, 'portion': 7, 'interest': 20, 'thousand': 67, 'largely': 8, 'report': 9, 'application': 6, 'directly': 20, 'probably': 15, 'moment': 117, 'whaleman': 46, 'adventure': 11, 'gathered': 8, 'homeward': 7, 'cruise': 11, 'commodore': 12, 'replied': 11, 'samuel': 11, 'brother': 14, 'another': 115, 'northern': 11, 'possible': 30, 'discover': 7, 'object': 50, 'forward': 57, 'laying': 6, 'whalemen': 72, 'indirectly': 13, 'mystic': 17, 'something': 123, 'impossible': 17, 'struck': 49, 'appearance': 15, 'eagerly': 8, 'different': 24, 'regular': 24, 'current': 10, 'vicinity': 20, 'elsewhere': 13, 'upright': 12, 'entrance': 8, 'perhaps': 89, 'arctic': 7, 'returned': 14, 'pursuit': 26, 'bloody': 9, 'possession': 13, 'savage': 65, 'taking': 53, 'american': 40, 'return': 15, 'departed': 14, 'suddenly': 50, 'perpendicularly': 8, 'coffin': 59, 'harpooned': 8, 'bethink': 9, 'chapter': 181, 'occasion': 15, 'slowly': 52, 'naturalist': 15, 'exclaimed': 15, 'turning': 53, 'instant': 77, 'destruction': 7, 'killer': 7, 'boundless': 6, 'ishmael': 20, 'precisely': 27, 'particular': 58, 'watery': 25, 'driving': 9, 'whenever': 15, 'growing': 9, 'involuntarily': 12, 'pausing': 10, 'bringing': 10, 'especially': 44, 'principle': 13, 'deliberately': 8, 'stepping': 7, 'knocking': 8, 'pistol': 8, 'quietly': 19, 'surprising': 7, 'cherish': 9, 'feeling': 34, 'commerce': 6, 'extreme': 14, 'breeze': 24, 'previous': 35, 'thence': 7, 'silent': 26, 'mortal': 50, 'revery': 7, 'leaning': 28, 'seated': 27, 'looking': 70, 'bulwark': 37, 'rigging': 34, 'striving': 10, 'landsman': 18, 'nailed': 11, 'field': 8, 'pacing': 11, 'straight': 46, 'seemingly': 9, 'strange': 98, 'content': 25, 'yonder': 18, 'possibly': 30, 'falling': 12, 'league': 12, 'magnetic': 7, 'virtue': 22, 'needle': 22, 'compass': 27, 'thither': 10, 'carry': 16, 'leaf': 19, 'absent': 10, 'minded': 6, 'plunged': 9, 'deepest': 6, 'infallibly': 11, 'desert': 10, 'happen': 8, 'desire': 12, 'romantic': 7, 'valley': 11, 'hollow': 23, 'meadow': 9, 'cattle': 6, 'sleepy': 8, 'distant': 21, 'reaching': 12, 'mountain': 25, 'picture': 35, 'unless': 29, 'prairie': 18, 'wanting': 11, 'travel': 9, 'receiving': 13, 'handful': 10, 'silver': 13, 'needed': 7, 'invest': 6, 'passenger': 11, 'mystical': 9, 'vibration': 6, 'persian': 7, 'separate': 20, 'surely': 7, 'meaning': 21, 'deeper': 19, 'fountain': 13, 'drowned': 12, 'ocean': 16, 'phantom': 22, 'conscious': 8, 'night': 11, 'general': 69, 'distinction': 7, 'office': 10, 'honourable': 7, 'schooner': 6, 'confess': 8, 'considerable': 26, 'officer': 35, 'salted': 6, 'egyptian': 11, 'house': 9, 'pyramid': 10, 'simple': 11, 'sailor': 133, 'forecastle': 38, 'rather': 72, 'enough': 76, 'honour': 26, 'particularly': 8, 'family': 14, 'putting': 16, 'schoolmaster': 8, 'order': 24, 'amount': 7, 'scale': 6, 'archangel': 9, 'think': 20, 'anything': 43, 'instance': 49, 'knowing': 16, 'everybody': 7, 'served': 10, 'universal': 8, 'passed': 42, 'blade': 8, 'always': 80, 'single': 39, 'contrary': 12, 'difference': 19, 'activity': 9, 'really': 30, 'marvellous': 15, 'considering': 31, 'earnestly': 7, 'believe': 27, 'earthly': 16, 'astern': 29, 'quarter': 52, 'atmosphere': 6, 'second': 60, 'breathes': 6, 'leader': 7, 'suspect': 6, 'wherefore': 13, 'merchant': 20, 'invisible': 16, 'influence': 16, 'unaccountable': 16, 'answer': 28, 'formed': 22, 'united': 9, 'state': 7, 'exactly': 24, 'recall': 6, 'circumstance': 52, 'spring': 38, 'presented': 19, 'various': 38, 'judgment': 12, 'portentous': 6, 'rolled': 43, 'nameless': 18, 'peril': 24, 'marvel': 10, 'sight': 12, 'sound': 12, 'helped': 15, 'tormented': 14, 'everlasting': 9, 'remote': 11, 'coast': 7, 'perceive': 8, 'horror': 17, 'social': 12, 'reason': 75, 'welcome': 6, 'wonder': 49, 'conceit': 17, 'floated': 18, 'endless': 15, 'procession': 6, 'hooded': 7, 'carpet': 9, 'started': 27, 'quitting': 6, 'arrived': 14, 'bedford': 18, 'learning': 6, 'already': 32, 'sailed': 38, 'following': 24, 'related': 6, 'connected': 12, 'famous': 37, 'pleased': 9, 'gradually': 10, 'business': 68, 'original': 21, 'canoe': 8, 'partly': 14, 'harpoon': 108, 'bowsprit': 8, 'destined': 7, 'became': 24, 'meanwhile': 28, 'dismal': 8, 'anxious': 9, 'sounded': 8, 'pocket': 20, 'piece': 18, 'wherever': 6, 'middle': 37, 'dreary': 6, 'darkness': 32, 'conclude': 8, 'crossed': 15, 'looked': 64, 'window': 21, 'everywhere': 7, 'inch': 20, 'remorseless': 6, 'service': 11, 'miserable': 9, 'stopping': 9, 'followed': 26, 'block': 7, 'blackness': 11, 'candle': 16, 'proved': 16, 'presently': 19, 'proceeding': 10, 'entering': 8, 'flying': 30, 'picked': 11, 'hearing': 13, 'pushed': 15, 'opened': 9, 'interior': 12, 'sitting': 22, 'turned': 66, 'beyond': 24, 'pulpit': 18, 'church': 20, 'muttered': 17, 'wretched': 8, 'swinging': 13, 'painting': 9, 'representing': 7, 'spouter': 8, 'connexion': 9, 'common': 51, 'suppose': 37, 'wooden': 27, 'district': 6, 'stricken': 16, 'coffee': 6, 'corner': 20, 'howling': 13, 'tossed': 31, 'nevertheless': 50, 'pleasant': 16, 'posse': 7, 'outside': 7, 'thrust': 20, 'universe': 6, 'pillow': 7, 'shaking': 9, 'shivering': 8, 'afterwards': 25, 'light': 10, 'oriental': 13, 'privilege': 9, 'holding': 37, 'sumatra': 6, 'lengthwise': 12, 'equator': 11, 'wonderful': 22, 'iceberg': 8, 'moored': 9, 'frozen': 10, 'society': 10, 'orphan': 6, 'plenty': 18, 'scrape': 6, 'fashioned': 12, 'careful': 18, 'inquiry': 6, 'mass': 8, 'shade': 6, 'shadow': 30, 'england': 20, 'earnest': 10, 'repeated': 21, 'throwing': 11, 'conclusion': 6, 'hovering': 10, 'centre': 23, 'perpendicular': 11, 'nervous': 9, 'fairly': 16, 'midnight': 27, 'unnatural': 8, 'blasted': 8, 'winter': 12, 'breaking': 16, 'resemblance': 7, 'gigantic': 20, 'design': 6, 'opinion': 16, 'subject': 26, 'visible': 32, 'exasperated': 6, 'opposite': 18, 'heathenish': 6, 'spear': 8, 'glittering': 8, 'shaped': 8, 'handle': 17, 'sweeping': 12, 'cannibal': 30, 'implement': 6, 'lance': 23, 'broken': 50, 'wildly': 9, 'sunset': 10, 'entered': 14, 'crossing': 18, 'central': 12, 'chimney': 9, 'wrinkled': 15, 'plank': 31, 'furiously': 6, 'cracked': 7, 'filled': 22, 'remotest': 10, 'projecting': 10, 'stand': 29, 'attempt': 13, 'ranged': 8, 'bottle': 9, 'cursed': 12, 'indeed': 89, 'delirium': 6, 'downwards': 12, 'bottom': 55, 'parallel': 7, 'meridian': 6, 'measure': 18, 'number': 31, 'seaman': 56, 'specimen': 8, 'sought': 18, 'landlord': 34, 'telling': 19, 'desired': 11, 'forehead': 40, 'blanket': 14, 'bitter': 9, 'decent': 6, 'supper': 30, 'settle': 8, 'carved': 14, 'diligently': 7, 'working': 15, 'trying': 17, 'winding': 6, 'button': 8, 'monkey': 16, 'jacket': 27, 'finger': 27, 'dumpling': 7, 'addressed': 7, 'direful': 10, 'whispered': 11, 'company': 39, 'resolved': 13, 'evening': 18, 'starting': 11, 'morning': 75, 'hurrah': 10, 'shaggy': 7, 'muffled': 9, 'woollen': 7, 'landed': 12, 'standing': 82, 'weather': 36, 'mounted': 14, 'somewhat': 27, 'shipmate': 40, 'become': 28, 'sleeping': 22, 'concerned': 8, 'description': 7, 'height': 19, 'deeply': 6, 'contrast': 12, 'announced': 6, 'companion': 6, 'slipped': 8, 'minute': 38, 'raised': 20, 'bulkington': 8, 'darted': 38, 'seeming': 8, 'private': 13, 'unknown': 36, 'stranger': 55, 'anybody': 10, 'bachelor': 9, 'apartment': 6, 'hammock': 34, 'certainly': 29, 'finest': 6, 'getting': 26, 'changed': 10, 'carpenter': 54, 'saying': 40, 'grinning': 10, 'shaving': 7, 'narrow': 12, 'higher': 23, 'interval': 57, 'draught': 7, 'immediate': 7, 'inside': 18, 'violent': 15, 'seeing': 37, 'spending': 6, 'awhile': 6, 'dropping': 18, 'bedfellow': 7, 'pretend': 7, 'actually': 16, 'blessed': 11, 'sunday': 7, 'market': 6, 'shouted': 17, 'calmly': 13, 'passion': 9, 'understand': 17, 'belongs': 8, 'story': 7, 'confidential': 9, 'highest': 8, 'demand': 6, 'thereby': 18, 'render': 6, 'zealand': 7, 'morrow': 12, 'wanted': 9, 'stopped': 8, 'string': 8, 'otherwise': 26, 'mystery': 14, 'showed': 16, 'stayed': 7, 'fluke': 37, 'dreaming': 6, 'pitched': 10, 'lighted': 13, 'anchor': 28, 'considered': 14, 'stair': 10, 'furnished': 16, 'prodigious': 18, 'placing': 7, 'double': 16, 'comfortable': 9, 'eyeing': 27, 'disappeared': 15, 'counterpane': 7, 'glanced': 6, 'belonging': 8, 'properly': 9, 'lashed': 11, 'containing': 8, 'likewise': 22, 'outlandish': 6, 'concerning': 40, 'uncommonly': 7, 'commenced': 7, 'thinking': 30, 'beginning': 19, 'jumped': 11, 'blowing': 6, 'tumbled': 6, 'infernal': 8, 'perfectly': 6, 'spoken': 13, 'identical': 7, 'eagerness': 8, 'employed': 15, 'accomplished': 7, 'yellow': 23, 'square': 22, 'terrible': 22, 'surgeon': 11, 'chanced': 13, 'sticking': 6, 'cheek': 7, 'remembered': 6, 'tattooed': 8, 'concluded': 19, 'course': 47, 'similar': 25, 'honest': 14, 'unearthly': 12, 'complexion': 6, 'completely': 33, 'independent': 9, 'tattooing': 6, 'produced': 6, 'extraordinary': 7, 'effect': 17, 'passing': 23, 'noticed': 11, 'difficulty': 6, 'pulled': 13, 'tomahawk': 19, 'singing': 12, 'surprise': 10, 'twisted': 9, 'bolted': 7, 'dinner': 18, 'coward': 21, 'purple': 6, 'rascal': 7, 'inexplicable': 7, 'continued': 14, 'escaped': 11, 'marked': 18, 'running': 46, 'shipped': 12, 'aboard': 21, 'shuddering': 9, 'attention': 9, 'previously': 17, 'curious': 54, 'embalmed': 6, 'polished': 7, 'removing': 6, 'backed': 9, 'chapel': 11, 'screwed': 8, 'hidden': 23, 'meantime': 23, 'follow': 20, 'place': 20, 'carefully': 11, 'biscuit': 11, 'snatch': 7, 'whereby': 9, 'succeeded': 9, 'drawing': 37, 'accompanied': 6, 'increased': 7, 'symptom': 8, 'concluding': 11, 'operation': 11, 'cloud': 12, 'tobacco': 6, 'sprang': 13, 'giving': 28, 'sudden': 43, 'angel': 9, 'scattered': 9, 'queequeg': 252, 'clothes': 11, 'smoking': 19, 'waking': 6, 'daylight': 9, 'loving': 8, 'coloured': 8, 'figure': 26, 'precise': 19, 'keeping': 13, 'hardly': 33, 'explain': 8, 'remember': 17, 'reality': 8, 'cutting': 37, 'sending': 8, 'mother': 24, 'dragged': 17, 'sheet': 10, 'sixteen': 7, 'entire': 55, 'resurrection': 7, 'voice': 7, 'dressed': 7, 'softly': 11, 'conscientious': 6, 'greatest': 12, 'subsequent': 13, 'troubled': 15, 'dream': 8, 'wrapped': 9, 'instantly': 23, 'supernatural': 7, 'closely': 14, 'daring': 8, 'consciousness': 6, 'glided': 14, 'month': 17, 'experienced': 11, 'event': 15, 'naught': 14, 'strove': 8, 'slight': 9, 'hatchet': 6, 'pickle': 6, 'serious': 10, 'character': 15, 'civilized': 16, 'delicacy': 7, 'treated': 7, 'staring': 6, 'watching': 16, 'unusual': 12, 'trowsers': 13, 'hunted': 19, 'movement': 12, 'straining': 9, 'neither': 16, 'completed': 6, 'crushed': 8, 'commanded': 24, 'amazement': 6, 'behold': 16, 'begin': 11, 'vigorous': 9, 'scraping': 6, 'vengeance': 12, 'exceedingly': 15, 'breakfast': 14, 'quickly': 21, 'descending': 10, 'malice': 12, 'scarce': 12, 'proper': 32, 'backward': 8, 'blacksmith': 24, 'brawny': 6, 'tropic': 9, 'slightly': 13, 'western': 8, 'contrasting': 9, 'flinging': 7, 'traveller': 12, 'parlor': 7, 'solitary': 21, 'stomach': 7, 'anywhere': 14, 'reflection': 10, 'profound': 20, 'silence': 20, 'slightest': 20, 'boarded': 6, 'calling': 13, 'warrior': 7, 'jeopardy': 6, 'coolly': 7, 'peculiarity': 18, 'applied': 8, 'glimpse': 15, 'individual': 15, 'foreign': 8, 'mediterranean': 9, 'affrighted': 6, 'lady': 13, 'native': 20, 'mentioned': 18, 'mostly': 11, 'whence': 17, 'bumpkin': 6, 'distinguished': 7, 'canvas': 14, 'throat': 6, 'tempest': 9, 'condition': 6, 'planted': 14, 'question': 36, 'hither': 14, 'father': 31, 'daughter': 6, 'porpoise': 26, 'wedding': 6, 'recklessly': 6, 'august': 8, 'beautiful': 8, 'tapering': 12, 'flower': 6, 'refuse': 7, 'sunlight': 6, 'seventh': 9, 'instead': 24, 'shortly': 10, 'returning': 8, 'congregation': 8, 'reigned': 6, 'shriek': 8, 'steadfastly': 6, 'marble': 18, 'tablet': 10, 'memory': 15, 'eighteen': 7, 'overboard': 27, 'sister': 6, 'forming': 14, 'ground': 67, 'sideways': 29, 'affected': 8, 'countenance': 12, 'present': 87, 'notice': 11, 'reading': 8, 'accident': 12, 'caused': 12, 'buried': 14, 'despair': 9, 'deadly': 22, 'being': 9, 'perished': 6, 'mankind': 13, 'included': 11, 'secret': 30, 'yesterday': 10, 'significant': 11, 'infidel': 7, 'immortal': 16, 'eternal': 19, 'trance': 6, 'century': 22, 'unspeakable': 9, 'doubt': 6, 'eternity': 9, 'methinks': 10, 'mistaken': 8, 'substance': 33, 'spiritual': 11, 'venerable': 10, 'sufficiently': 9, 'mapple': 9, 'wrinkle': 15, 'verdure': 7, 'utmost': 15, 'carried': 30, 'removed': 10, 'ladder': 13, 'provided': 6, 'headed': 25, 'contrivance': 10, 'grasping': 7, 'upwards': 19, 'ascending': 7, 'usually': 6, 'unnecessary': 6, 'gaining': 10, 'furthermore': 7, 'unseen': 14, 'outward': 11, 'faithful': 7, 'feature': 16, 'gallant': 16, 'breaker': 7, 'distinct': 16, 'inserted': 17, 'likeness': 7, 'foremost': 11, 'descried': 22, 'complete': 25, 'authority': 13, 'ordered': 8, 'starboard': 17, 'gangway': 6, 'larboard': 9, 'paused': 16, 'folded': 8, 'across': 31, 'uplifted': 10, 'closed': 10, 'offered': 12, 'prayer': 10, 'prolonged': 11, 'solemn': 12, 'continual': 14, 'terror': 42, 'sorrow': 7, 'plunging': 6, 'relief': 6, 'dolphin': 7, 'record': 8, 'joined': 7, 'swelled': 6, 'smallest': 10, 'depth': 10, 'lesson': 14, 'billow': 25, 'water': 69, 'wilful': 7, 'command': 31, 'wherein': 8, 'seeking': 12, 'tarshish': 7, 'hitherto': 17, 'strait': 19, 'worthy': 8, 'slouched': 8, 'shipping': 6, 'suspicion': 9, 'touched': 13, 'fugitive': 6, 'search': 8, 'hoisting': 12, 'confidence': 6, 'innocent': 6, 'whisper': 8, 'missing': 16, 'murderer': 12, 'apprehension': 8, 'custom': 8, 'harmless': 6, 'sooner': 13, 'glance': 35, 'intently': 9, 'swiftly': 19, 'written': 11, 'expose': 6, 'freely': 12, 'thrice': 8, 'flight': 12, 'prudent': 7, 'contains': 6, 'allowed': 6, 'locked': 16, 'ceiling': 6, 'resting': 12, 'bowel': 12, 'obvious': 11, 'conscience': 13, 'chamber': 6, 'anguish': 8, 'prodigy': 6, 'misery': 6, 'wicked': 11, 'burden': 7, 'hideous': 10, 'timber': 11, 'asleep': 6, 'frightened': 6, 'sleeper': 15, 'startled': 7, 'stagger': 6, 'finding': 8, 'afloat': 12, 'overhead': 6, 'downward': 10, 'attitude': 11, 'casting': 6, 'discovered': 10, 'receive': 8, 'straightway': 13, 'dropped': 27, 'float': 11, 'smooth': 9, 'whirling': 7, 'commotion': 12, 'awaiting': 6, 'observe': 11, 'direct': 11, 'temple': 13, 'conduct': 12, 'speaking': 23, 'heaved': 12, 'silently': 15, 'motionless': 8, 'leaned': 13, 'bowing': 7, 'aspect': 42, 'greater': 13, 'hatch': 10, 'listen': 9, 'nineveh': 6, 'escape': 25, 'reached': 19, 'fathom': 15, 'delight': 20, 'bidding': 7, 'castaway': 7, 'lifting': 9, 'heavenly': 6, 'treacherous': 6, 'chiefly': 19, 'waving': 7, 'remained': 34, 'hearth': 7, 'peering': 8, 'gently': 10, 'spirit': 28, 'devil': 17, 'bearing': 26, 'excellent': 12, 'popular': 8, 'regularly': 18, 'george': 10, 'heeded': 6, 'presence': 6, 'wholly': 27, 'occupied': 9, 'indifference': 7, 'advance': 27, 'circle': 37, 'singular': 9, 'sublime': 8, 'twenty': 34, 'planet': 6, 'serenity': 6, 'philosopher': 9, 'lonely': 12, 'burning': 12, 'intensity': 9, 'swell': 11, 'sensible': 9, 'lurked': 8, 'producing': 6, 'naturally': 12, 'pressed': 9, 'phrase': 6, 'dollar': 9, 'mechanically': 8, 'dividing': 7, 'worship': 13, 'pagan': 6, 'couple': 7, 'future': 16, 'position': 15, 'bodily': 26, 'quality': 12, 'merely': 19, 'crystal': 6, 'except': 31, 'elastic': 7, 'serene': 8, 'hanging': 20, 'scene': 11, 'skeleton': 39, 'whaler': 38, 'priest': 15, 'harbor': 21, 'paddle': 9, 'gliding': 14, 'gained': 14, 'chain': 18, 'suspended': 30, 'wrist': 6, 'desperate': 12, 'prince': 6, 'coronation': 7, 'consider': 23, 'fearful': 12, 'throne': 9, 'vocation': 11, 'intention': 9, 'boldly': 10, 'world': 7, 'ignorant': 10, 'settled': 12, 'alarmed': 7, 'hinted': 25, 'assured': 6, 'inland': 6, 'furnish': 9, 'owner': 20, 'fragrant': 6, 'commander': 22, 'gentleman': 58, 'majesty': 6, 'unlike': 8, 'immemorial': 7, 'proceeds': 8, 'perilous': 15, 'nostril': 9, 'pointed': 29, 'homage': 9, 'intense': 18, 'strength': 24, 'bursting': 12, 'yelled': 6, 'expression': 11, 'roared': 13, 'happened': 16, 'strain': 13, 'parted': 8, 'handled': 8, 'snatching': 9, 'madness': 17, 'splinter': 7, 'capable': 6, 'consternation': 11, 'crawling': 6, 'secured': 15, 'stripped': 11, 'shooting': 8, 'dragging': 7, 'surrounded': 8, 'table': 6, 'wondrous': 42, 'infant': 6, 'direction': 24, 'nantucketers': 10, 'captured': 17, 'season': 23, 'declared': 14, 'mightiest': 6, 'animated': 7, 'unconscious': 6, 'fearless': 8, 'malicious': 9, 'power': 7, 'blazing': 7, 'nantucketer': 22, 'emperor': 11, 'empire': 7, 'smell': 12, 'strangely': 29, 'chowder': 12, 'moreover': 20, 'inhabitant': 6, 'painted': 12, 'impression': 15, 'remaining': 11, 'carrying': 15, 'affair': 23, 'spread': 26, 'hurried': 7, 'leading': 12, 'apparently': 9, 'prospect': 9, 'butter': 11, 'appetite': 6, 'resumed': 6, 'vertebra': 12, 'superior': 16, 'feeding': 10, 'marching': 7, 'nearest': 10, 'variety': 8, 'concern': 11, 'strongly': 11, 'inasmuch': 8, 'forgotten': 9, 'fitted': 12, 'accordingly': 8, 'determined': 6, 'rushing': 17, 'energy': 8, 'ramadan': 13, 'pequod': 173, 'japanese': 10, 'typhoon': 11, 'bearded': 6, 'antiquity': 7, 'retired': 6, 'material': 10, 'barbaric': 11, 'chased': 14, 'hempen': 7, 'tiller': 13, 'curiously': 6, 'hereditary': 7, 'helmsman': 11, 'steered': 8, 'melancholy': 8, 'wigwam': 9, 'temporary': 11, 'fibre': 6, 'carving': 6, 'heavily': 7, 'quaker': 13, 'sailing': 35, 'windward': 22, 'muscle': 7, 'advancing': 20, 'damned': 6, 'humorous': 6, 'hailed': 21, 'vineyard': 8, 'bildad': 76, 'nearer': 15, 'hearty': 8, 'exclamation': 9, 'inclined': 9, 'indispensable': 11, 'experience': 15, 'perceived': 14, 'obliquely': 8, 'horizon': 19, 'squall': 25, 'uncommon': 16, 'originally': 17, 'retain': 7, 'fashion': 13, 'audacious': 6, 'greatly': 6, 'natural': 36, 'watch': 12, 'virgin': 11, 'advantage': 6, 'result': 6, 'deemed': 14, 'according': 26, 'lovely': 11, 'spilled': 8, 'religion': 10, 'practical': 13, 'rising': 32, 'header': 8, 'hearted': 12, 'clutch': 6, 'hammer': 37, 'beside': 6, 'spectacle': 11, 'volume': 15, 'knowledge': 12, 'unconsciously': 7, 'mumbling': 7, 'profit': 7, 'importance': 6, 'splice': 7, 'eventually': 8, 'management': 6, 'vainly': 6, 'knowest': 6, 'seventy': 22, 'corrupt': 6, 'doubloon': 27, 'solemnly': 8, 'advice': 6, 'steadily': 11, 'insult': 12, 'flame': 16, 'sliding': 10, 'temporarily': 9, 'stepped': 8, 'withdrawing': 8, 'leeward': 29, 'significance': 6, 'foolish': 17, 'accursed': 10, 'happens': 12, 'walked': 6, 'incidentally': 7, 'revealed': 18, 'notion': 6, 'knocked': 6, 'fastened': 11, 'landlady': 6, 'abroad': 6, 'murder': 10, 'handed': 10, 'doctor': 6, 'talking': 6, 'seized': 18, 'dashed': 22, 'pushing': 6, 'change': 8, 'intolerable': 9, 'probability': 7, 'pudding': 6, 'becomes': 12, 'frantic': 15, 'health': 6, 'useless': 6, 'sagacious': 6, 'hereafter': 10, 'cooked': 6, 'inference': 6, 'mouth': 9, 'remark': 7, 'simply': 10, 'walking': 9, 'loudly': 7, 'member': 9, 'meeting': 15, 'thyself': 24, 'belong': 11, 'belief': 6, 'quohog': 9, 'glistening': 7, 'preliminary': 10, 'entitled': 10, 'latter': 15, 'remain': 23, 'domestic': 10, 'thrusting': 10, 'levelled': 9, 'ribbed': 7, 'rapidly': 19, 'laughed': 7, 'elijah': 15, 'intent': 15, 'shrouded': 7, 'relieved': 10, 'knife': 10, 'period': 21, 'steward': 18, 'charity': 9, 'safety': 7, 'comfort': 8, 'longer': 18, 'whalebone': 11, 'hatchway': 10, 'rigger': 6, 'expected': 7, 'necessary': 8, 'involved': 9, 'uncertain': 10, 'twilight': 6, 'hailing': 7, 'scuttle': 24, 'slumber': 6, 'alluded': 6, 'heaving': 15, 'vapour': 19, 'starbuck': 198, 'lively': 14, 'hauled': 9, 'issued': 8, 'steering': 13, 'capstan': 13, 'handspike': 8, 'approaching': 10, 'windlass': 22, 'apparition': 9, 'whisker': 8, 'elephant': 30, 'eternally': 7, 'remains': 24, 'ranging': 7, 'alongside': 27, 'invested': 21, 'encounter': 20, 'continent': 7, 'coiling': 7, 'grasped': 6, 'lantern': 17, 'gazing': 23, 'murmured': 12, 'forget': 11, 'locker': 6, 'beware': 14, 'cheese': 9, 'screaming': 6, 'blindly': 7, 'encountered': 16, 'shudder': 6, 'infinite': 8, 'hereby': 9, 'accounted': 9, 'profession': 7, 'procedure': 6, 'invariably': 11, 'slippery': 6, 'compared': 8, 'abounding': 9, 'taper': 6, 'admiral': 7, 'expense': 6, 'manned': 13, 'remarkable': 13, 'archipelago': 6, 'musket': 12, 'distinctly': 8, 'discovery': 12, 'declare': 6, 'shiver': 7, 'darting': 14, 'imperial': 10, 'capital': 6, 'conspicuous': 7, 'dignity': 20, 'possibility': 6, 'precious': 18, 'surmise': 6, 'supply': 10, 'knight': 7, 'endure': 10, 'latitude': 28, 'vitality': 8, 'lingering': 8, 'action': 11, 'superstition': 9, 'intelligence': 7, 'portent': 6, 'welded': 11, 'evinced': 21, 'vicissitude': 8, 'courage': 9, 'lowering': 20, 'critical': 14, 'reasonable': 8, 'withstand': 9, 'terrific': 14, 'enraged': 6, 'sparkling': 9, 'circumference': 7, 'divine': 8, 'hammered': 8, 'valiant': 10, 'indifferent': 11, 'loaded': 13, 'personally': 6, 'majestic': 7, 'danger': 7, 'encountering': 6, 'divided': 10, 'wrought': 6, 'battering': 6, 'headsman': 15, 'descend': 12, 'tashtego': 57, 'village': 10, 'believed': 6, 'daggoo': 37, 'golden': 22, 'securing': 8, 'heedful': 6, 'retained': 10, 'military': 6, 'brain': 8, 'alabama': 6, 'tambourine': 11, 'retreat': 8, 'heightened': 6, 'subtle': 11, 'emotion': 11, 'calculated': 7, 'readily': 13, 'taffrail': 8, 'perseus': 11, 'scorched': 6, 'slender': 6, 'throughout': 10, 'manxman': 12, 'tradition': 9, 'popularly': 8, 'elevated': 11, 'pitching': 6, 'continually': 17, 'cruising': 27, 'dancing': 9, 'velvet': 6, 'potency': 6, 'noiseless': 7, 'agency': 7, 'steersman': 7, 'repose': 12, 'mainmast': 11, 'cannon': 6, 'speechless': 6, 'violently': 9, 'advanced': 11, 'powder': 10, 'riddle': 6, 'kicked': 9, 'binnacle': 16, 'narwhale': 13, 'bubble': 14, 'sinking': 13, 'fright': 8, 'eating': 12, 'lifted': 17, 'halloa': 10, 'hereabouts': 6, 'launched': 6, 'leviathanic': 8, 'research': 7, 'covering': 6, 'science': 9, 'ultimate': 7, 'scientific': 16, 'outline': 6, 'promise': 8, 'foundation': 7, 'system': 12, 'argument': 9, 'external': 14, 'horizontal': 16, 'smaller': 11, 'lurking': 7, 'octavo': 12, 'duodecimo': 8, 'german': 18, 'obtained': 7, 'derived': 10, 'bestowed': 8, 'commonly': 7, 'specially': 8, 'isolated': 6, 'surface': 42, 'sullen': 6, 'notwithstanding': 6, 'detached': 10, 'regard': 11, 'structure': 8, 'irregular': 6, 'example': 22, 'diving': 7, 'embrace': 8, 'breathing': 6, 'recognised': 9, 'corresponding': 7, 'average': 6, 'blubber': 34, 'exceed': 7, 'strictly': 7, 'clumsy': 10, 'correct': 6, 'tossing': 12, 'fourth': 15, 'beholding': 11, 'delicate': 7, 'lowered': 20, 'existence': 7, 'attached': 21, 'lodged': 9, 'hunting': 9, 'vigilance': 7, 'grandeur': 8, 'obedience': 8, 'usage': 6, 'intended': 9, 'superiority': 6, 'observation': 7, 'reckoning': 6, 'reserved': 9, 'swing': 8, 'dexterous': 6, 'customary': 14, 'problem': 9, 'intelligent': 7, 'derive': 6, 'presumed': 9, 'lounging': 6, 'mouthful': 6, 'insanity': 6, 'belly': 9, 'tumultuous': 9, 'blind': 10, 'opposing': 6, 'hearse': 15, 'colossal': 7, 'comparatively': 8, 'produce': 6, 'shattered': 6, 'murderous': 10, 'right': 6, 'sucking': 6, 'simultaneously': 14, 'stander': 6, 'formation': 6, 'summit': 11, 'tackle': 28, 'column': 8, 'hercules': 6, 'excitement': 7, 'envelope': 6, 'additional': 12, 'drawer': 7, 'trumpet': 7, 'occasionally': 14, 'widely': 7, 'revolving': 9, 'lightly': 7, 'sunken': 6, 'vision': 8, 'pervading': 6, 'uprising': 6, 'flitting': 6, 'enchanted': 12, 'rocking': 8, 'inscrutable': 8, 'transparent': 8, 'fiercely': 6, 'mechanical': 7, 'raise': 6, 'separately': 7, 'specific': 8, 'wrenched': 6, 'measureless': 8, 'closer': 8, 'guinea': 6, 'turkish': 6, 'foremast': 6, 'subterranean': 6, 'warning': 10, 'innermost': 7, 'necessity': 10, 'seizing': 16, 'socket': 9, 'bearer': 6, 'shivered': 6, 'sidelong': 7, 'flashing': 6, 'swerve': 6, 'insufferable': 6, 'horrible': 15, 'consequence': 9, 'carcase': 7, 'frigate': 8, 'brace': 6, 'copper': 8, 'pulling': 24, 'cluster': 6, 'tahiti': 6, 'violence': 6, 'revenge': 8, 'stretch': 7, 'cunning': 15, 'fatality': 8, 'rumor': 8, 'smitten': 7, 'contact': 11, 'longitude': 9, 'superstitious': 10, 'recalled': 6, 'oftentimes': 6, 'detail': 8, 'sounding': 20, 'flank': 7, 'imagination': 6, 'disaster': 8, 'monomaniac': 11, 'probable': 6, 'darker': 6, 'profundity': 6, 'appalling': 10, 'chasing': 7, 'whiteness': 28, 'beauty': 8, 'symbol': 12, 'forked': 6, 'witness': 6, 'fleece': 13, 'beheld': 21, 'albatross': 10, 'horse': 11, 'peculiarly': 6, 'potent': 6, 'pallor': 6, 'ghost': 7, 'evangelist': 6, 'earthquake': 6, 'phenomenon': 6, 'comparative': 6, 'buffalo': 12, 'cabaco': 6, 'bucket': 20, 'occasional': 9, 'chart': 7, 'circular': 7, 'expand': 6, 'breadth': 10, 'preceding': 6, 'equatorial': 7, 'trade': 7, 'resuming': 6, 'vulture': 6, 'adequately': 7, 'incident': 6, 'historical': 8, 'killing': 6, 'hieroglyphic': 7, 'plague': 6, 'dashing': 8, 'directed': 6, 'suffering': 6, 'amazing': 6, 'reeled': 6, 'hurled': 6, 'solomon': 8, 'easily': 9, 'swaying': 6, 'playing': 6, 'weaving': 7, 'lashing': 10, 'shuttle': 6, 'thread': 9, 'perched': 8, 'heading': 10, 'basket': 6, 'gunwale': 29, 'technically': 7, 'coiled': 11, 'manilla': 6, 'fedallah': 27, 'leaped': 13, 'rowing': 8, 'ginger': 16, 'oarsman': 32, 'indolent': 6, 'jerking': 6, 'loggerhead': 7, 'firmly': 12, 'betrayed': 6, 'feather': 6, 'plate': 7, 'curling': 7, 'headlong': 7, 'worked': 8, 'stretching': 6, 'abandoned': 7, 'published': 8, 'involve': 6, 'silvery': 6, 'buoyant': 7, 'eastward': 11, 'shower': 8, 'dripping': 10, 'storm': 8, 'equally': 8, 'englishman': 7, 'skull': 6, 'sebastian': 9, 'radney': 22, 'steelkilt': 40, 'lakeman': 24, 'rigged': 7, 'hoisted': 30, 'canallers': 8, 'signal': 10, 'corpse': 14, 'crowding': 6, 'bowsman': 8, 'strained': 9, 'blinding': 6, 'slackened': 6, 'hindoo': 6, 'tender': 8, 'wrecked': 8, 'highly': 7, 'engraving': 10, 'drooping': 7, 'profile': 6, 'ninety': 7, 'separated': 6, 'murdered': 6, 'gentle': 7, 'thickness': 7, 'chock': 6, 'vibrating': 6, 'clinging': 9, 'spiracle': 8, 'churning': 6, 'contracting': 7, 'crotch': 10, 'freighted': 9, 'forged': 6, 'unctuous': 7, 'considerably': 9, 'diminished': 6, 'spade': 10, 'snapped': 8, 'shred': 6, 'beheaded': 6, 'jeroboam': 10, 'mayhew': 8, 'burned': 6, 'margin': 6, 'submerged': 8, 'lucifer': 6, 'parsee': 28, 'proportion': 6, 'heidelburgh': 8, 'combined': 9, 'curled': 6, 'cavity': 6, 'spinal': 8, 'derick': 12, 'feeder': 9, 'yarman': 7, 'adrift': 6, 'vishnoo': 8, 'pitchpoling': 6, 'spoutings': 7, 'masonry': 6, 'smiting': 6, 'hovered': 6, 'drugged': 7, 'measured': 8, 'frenchman': 8, 'plaintiff': 6, 'ambergris': 13, 'bouton': 6, 'guernsey': 11, 'monsieur': 9, 'squeeze': 9, 'enderby': 7, 'bunger': 13, 'weaver': 6, 'burton': 6, 'fossil': 7, 'sneeze': 8, 'quadrant': 8, 'corpusants': 6, 'rachel': 6, 'cherry': 6}
[('though', 384), ('captain', 353), ('seemed', 283), ('whale', 268), ('queequeg', 252), ('little', 249), ('starbuck', 198), ('almost', 195), ('thought', 184), ('chapter', 181), ('pequod', 173), ('without', 156), ('harpooneer', 137), ('sailor', 133), ('thing', 132), ('whaling', 131), ('something', 123), ('voyage', 121), ('moment', 117), ('called', 116)]
```
![i2.png](scripts%2Fi2.png)


## 2.7 N-gramas y Colocaciones del lenguaje

**¿Qué es un N-grama?**

Los n-gramas son secuencias de n elementos (ya sean caracteres o palabras) extraídos de un texto. Un n-grama puede ser un unigrama (n = 1), un bigrama (n = 2), un trigrama (n = 3) o cualquier otro número de elementos. Estos n-gramas se utilizan para analizar la estructura y el contenido de un texto.

![2.png](ims%2F2%2F2.png)

Cuando se trabaja con palabras, los n-gramas son secuencias consecutivas de n palabras en un texto. Por ejemplo, si tenemos la 
oración "Estoy aprendiendo cosas increíbles", los unigramas serían ["Estoy", "aprendiendo", "cosas" "increíbles"], los bigramas 
serían ["Estoy aprendiendo", "aprendiendo cosas", "cosas increíbles"], y los trigramas serían ["estoy aprendiendo cosas", 
"aprendiendo cosas increíbles"].

Los n-gramas son útiles en NLP porque capturan información sobre la co-ocurrencia de elementos en un texto y pueden revelar patrones y estructuras lingüísticas. Se utilizan en diversas aplicaciones, como el análisis de sentimientos, la generación de texto predictivo, la traducción automática y la detección de spam, entre otros.

Al calcular la frecuencia de los n-gramas en un corpus de texto, se pueden obtener estadísticas útiles, como la frecuencia de palabras individuales o la probabilidad condicional de una palabra dada su historia (usando n-gramas de orden superior). Estas estadísticas son utilizadas por modelos de lenguaje y otras técnicas de procesamiento de texto en NLP.

**Por otro lado tenemos a las `colocaciones`**

Las colocaciones son combinaciones de palabras que tienden a aparecer juntas con una frecuencia superior a la esperada por azar. En otras palabras, son secuencias de palabras que tienen una asociación léxica o semántica fuerte y son utilizadas de manera idiomática por los hablantes nativos de un idioma.

![3.png](ims%2F2%2F3.png)

Por ejemplo:

![4.png](ims%2F2%2F4.png)

Gramaticalmente hablando ambas oraciones tienen sentido y están bien escritas, sin embargo, es más natural para nosotros en español
pensar en "le dieron ganas" a "le introdujeron ganas" pese a que ambas están bien dichas.

Otra forma de interpretar a las `colocaciones` es pensar en ellas como conexiones de palabras recurrentes cuyo significado
es más que la suma de sus palabras individuales. Por ejemplo, en inglés, las colocaciones comunes son "strong coffee" (café fuerte), "fast food" (comida rápida) o "catch a bus" (tomar un autobús). Estas expresiones son consideradas colocaciones porque son frecuentes y su significado no puede deducirse de forma directa a partir del significado individual de las palabras que las componen.


> Las colocaciones son relevantes en NLP porque capturan la naturaleza idiomática y la semántica de las combinaciones de palabras en un idioma. Son utilizadas en diversas tareas, como la traducción automática, la generación de lenguaje natural, la desambiguación léxica y la recuperación de información, entre otras. El reconocimiento y el manejo adecuado de las colocaciones son importantes para comprender y generar texto de manera más precisa y natural.
>


## 2.8 ¿Cómo extraer n-gramas de un texto en Python?

> ## Nota:
> El código completo de esta sección lo puedes encontrar en: [7_extraer_ngramas.py](scripts%2F7_extraer_ngramas.py)

Primero partimos de una base de código de limpieza muy similar al visto en [6_refinamiento.py](scripts%2F6_refinamiento.py)

```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from nltk.book import text1
import string
import re


def filter_corpus(corpus, sw, lem):
    filtered_corpus = [word.lower() for word in corpus if word.lower() not in sw]
    filtered_corpus = [word for word in filtered_corpus if word not in string.punctuation]
    filtered_corpus = [word for word in filtered_corpus if len(word) > 5]
    filtered_corpus = [word for word in filtered_corpus if not re.match(r'\d', word)]
    filtered_corpus = [lem.lemmatize(word) for word in filtered_corpus]
    return filtered_corpus


def plot_freq(elementos, figname):
    palabras = [elemento[0] for elemento in elementos]
    frecuencias = [elemento[1] for elemento in elementos]
    plt.plot(palabras, frecuencias)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de palabras')
    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.close()
```
Pero vamos a añadir una nueva biblioteca que hará los bi-gramas y n-gramas por nosotros:
```python
from nltk import bigrams, ngrams
```

Empezamos por los bi-gramas vamos a limpiar el `text1` para ver como lucen cuando normalizamos el texto:

```python
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text1_filtered = filter_corpus(text1, stop_words, lemmatizer)
    md_bigrams = list(bigrams(text1_filtered))
    print(md_bigrams[:10])
```
Respuesta esperada:
```commandline
[('herman', 'melville'), ('melville', 'etymology'), ('etymology', 'supplied'), ('supplied', 'consumptive'), ('consumptive', 'grammar'), ('grammar', 'school'), ('school', 'threadbare'), ('threadbare', 'dusting'), ('dusting', 'lexicon'), ('lexicon', 'grammar')]
```
Ahora podemos utilizar `Counter` y `most_common` para extraer los 20 bi-gramas más utilizados:

```python
    bigrams_counter = Counter(md_bigrams)
    mc_bigrams = bigrams_counter.most_common(20)
    # aqui simplemente convertimos una lista de palabras en un texto que contenga todas sus palabras
    mc_bigrams = [(" ".join(grams), freq) for grams, freq in mc_bigrams]
    print(mc_bigrams)
    plot_freq(mc_bigrams, "i3")
```
Respuesta esperada:
```commandline
[('captain bildad', 19), ('whaling voyage', 17), ('cruising ground', 13), ('though indeed', 11), ('almost thought', 10), ('captain captain', 10), ('beneath surface', 10), ('queequeg queequeg', 9), ('father mapple', 9), ('whaling vessel', 8), ('business whaling', 8), ('nothing nothing', 8), ('hundred seventy', 8), ('harpoon lance', 8), ('latitude longitude', 8), ('almost seemed', 8), ('specie leviathan', 7), ('stranger captain', 7), ('chapter pequod', 7), ('monkey jacket', 6)]
```
![i3.png](scripts%2Fi3.png)

Ahora vamos a repetir pero usando `n-grams`:

```python
    # aquí he decido NO limpiar el text1 para observar las diferencias :)
    md_trigrams = list(ngrams(text1, 3))
    print(md_trigrams[:10])
```
Respuesta esperada:
```commandline
[('[', 'Moby', 'Dick'), ('Moby', 'Dick', 'by'), ('Dick', 'by', 'Herman'), ('by', 'Herman', 'Melville'), ('Herman', 'Melville', '1851'), ('Melville', '1851', ']'), ('1851', ']', 'ETYMOLOGY'), (']', 'ETYMOLOGY', '.'), ('ETYMOLOGY', '.', '('), ('.', '(', 'Supplied')]
```
Creamos el contador y gráficamos los `tri-gramas` más populares:

```python
    trigrams_counter = Counter(md_trigrams)
    mc_trigrams = trigrams_counter.most_common(20)
    mc_trigrams = [(" ".join(grams), freq) for grams, freq in mc_trigrams]
    print(mc_trigrams)
    plot_freq(mc_trigrams, "i4")
```
Respuesta esperada:
```commandline
[(', and the', 187), ("don ' t", 103), ('of the whale', 101), (', in the', 93), (', then ,', 87), ("whale ' s", 81), ('. It was', 81), ("ship ' s", 80), ('the Sperm Whale', 77), (', as if', 76), ("he ' s", 76), ("Ahab ' s", 75), ('. Now ,', 74), ("' s a", 73), ("' s the", 72), ("that ' s", 69), (', as the', 68), ('the sea ,', 67), ("it ' s", 67), (', and then', 67)]
```
![i4.png](scripts%2Fi4.png)

Observamos como aquí la frecuencia es mucho mayor a pesar de que son `tri-gramas` esto es debido a que en el ejemplo anterior eliminamos
tanto los signos de puntuación como los stopwords, aquí vemos que la mayoría de los `tri-gramas` que fueron encontrados usan stopwords
y signos de puntuación.

## 2.9 Colocaciones en Python

El PMI (Pointwise Mutual Information) es una métrica basada en la teoría de la información que se utiliza para medir la asociación entre dos elementos en un corpus de texto, como palabras o n-gramas. Es especialmente útil para identificar collocations o colocaciones, que son combinaciones de palabras que tienden a aparecer juntas con una frecuencia mayor de la esperada por azar.

El PMI se calcula utilizando la probabilidad de ocurrencia conjunta de dos elementos y las probabilidades marginales de cada elemento por separado. La fórmula general del PMI para dos elementos A y B es la siguiente:

PMI(A, B) = log2(P(A, B) / (P(A) * P(B)))

Donde:

- P(A, B) es la probabilidad de ocurrencia conjunta de A y B.
- P(A) es la probabilidad de ocurrencia de A.
- P(B) es la probabilidad de ocurrencia de B.

- El PMI mide la desviación de la ocurrencia conjunta de A y B con respecto a lo que se esperaría si A y B fueran eventos independientes. Un valor de PMI alto indica que la ocurrencia conjunta es mayor de lo esperado y sugiere una asociación fuerte entre los dos elementos.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [8_colocaciones.py](scripts%2F8_colocaciones.py)

Para este ejemplo vamos a usar los `bi-gramas` que entendimos en la clase pasada y vamos a usar un método de limpieza de
bi-gramas sumamente sencillo y es que cada una de las palabras que compone el bigrama debe tener al menos más de 2 letras, de
esta forma eliminamos signos de puntuación y una que otra stopword.

```python
import numpy as np
from nltk.book import text1
from nltk import bigrams, FreqDist
import pandas as pd
# esta biblioteca me permite hacer gráficos interactivos
import plotly.express as px

md_bigrams = list(bigrams(text1))

threshold = 2
# Obtenemos los bigramas donde cada palabra tenga más de 2 letras
filtered_bigrams = [bigram for bigram in md_bigrams if len(bigram[0]) > threshold and len(bigram[1]) > threshold]
# Usamos un contador para obtener la frecuencia de los bigramas
filtered_bigram_dist = FreqDist(filtered_bigrams)
# Obtenemos las palabras individuales que tengan más de 2 letras
filtered_words = [word for word in text1 if len(word) > threshold]
# Usamos un contador para obtener la frecuencia de las palabras (unigramas)
filtered_word_dist = FreqDist(filtered_words)
```
Ahora vamos a gener un `DataFrame` de pandas que contenga las principales variables que vamos a utilizar para el calculo del 
PMI (solo que en lugar de utilizar directamente las probabilidades, vamos a utilizar la frecuencia de las palabras)

```python
# Para entender mejor las variables que vamos a utilizar vamos a construir un DataFrame de pandas
df = pd.DataFrame()
df['bi_gram'] = list(set(filtered_bigrams))
df['word_0'] = df['bi_gram'].apply(lambda x: x[0])
df['word_1'] = df['bi_gram'].apply(lambda x: x[1])
# Aquí estamos usando los diccionarios de filtered_bigram_dist y filtered_word_dist para obtener la frecuencia de una
# palabra con base justamente en esa palabra
df['bi_gram_freq'] = df['bi_gram'].apply(lambda x: filtered_bigram_dist[x])
df['word_0_freq'] = df['word_0'].apply(lambda x: filtered_word_dist[x])
df['word_1_freq'] = df['word_1'].apply(lambda x: filtered_word_dist[x])
print(df)
```
Respuesta esperada:
```commandline
                 bi_gram  word_0  ... word_0_freq  word_1_freq
0            (very, act)    very  ...         311           30
1         (object, here)  object  ...          39          272
2            (the, sons)     the  ...       13721            4
3             (!--, see)     !--  ...         141          253
4           (the, fires)     the  ...       13721            4
...                  ...     ...  ...         ...          ...
67937       (thou, must)    thou  ...         232          282
67938         (the, wit)     the  ...       13721            2
67939   (the, direction)     the  ...       13721           15
67940  (lithe, swayings)   lithe  ...           2            1
67941   (Slowly, wading)  Slowly  ...           3            2

[67942 rows x 6 columns]
```

Finalmente, calculamos el PMI, y el log de la frecuencia de los bigramas:

```python
# Calculamos un pseudo PMI (NO es directamente la probabilidad de las palabras) pero es similar usar
# la frecuencia de las palabras
df['PMI'] = np.log2(df['bi_gram_freq'] / (df['word_0_freq'] * df['word_1_freq']))
df['log(bi_gram_freq)'] = np.log2(df['bi_gram_freq'])
df.sort_values(by='PMI', ascending=False, inplace=True)
print(df)
```
Respuesta esperada:
```commandline
                        bi_gram    word_0  ...        PMI  log(bi_gram_freq)
20874         (balmy, autumnal)     balmy  ...   0.000000                0.0
10699     (corporal, animosity)  corporal  ...   0.000000                0.0
48366    (minor, contingencies)     minor  ...   0.000000                0.0
62833           (WAS, ATTACKED)       WAS  ...   0.000000                0.0
56515  (seemly, correspondence)    seemly  ...   0.000000                0.0
...                         ...       ...  ...        ...                ...
47831                (man, the)       man  ... -22.732783                0.0
11090               (some, the)      some  ... -22.919024                0.0
30770                (one, the)       one  ... -23.540138                0.0
26085                (the, not)       the  ... -23.851315                0.0
64019                (the, but)       the  ... -23.864336                0.0

[67942 rows x 8 columns]
```

Ahora vamos a graficar esta distribución (se ve mucho mejor en el navegador)

```python
# Finalmente, hacemos un scatter de la distrbiución de los bigramas
fig = px.scatter(x=df['PMI'].values, y=df['log(bi_gram_freq)'].values, color=df['PMI'] + df['log(bi_gram_freq)'],
                 size=(df['PMI'] + df['log(bi_gram_freq)']).apply(lambda x: 1 / (1 + abs(x))).values,
                 hover_name=df['bi_gram'].values, width=600, height=600,
                 labels={'x': 'PMI', 'y': 'Log(Bigram Frequency)'})
fig.show()
```
Respuesta esperada:

![i5.png](scripts%2Fi5.png)

## 2.10 Colocaciones en gráficos de dispersión

Explicación del gráfico de PMI:

![5.png](ims%2F2%2F5.png)

Recordemos que solamente el valor del PMI no es suficiente informativo para saber si las combinaciones de palabras son o no
colocaciones porque si el PMI es más cercano a 0 da la apariencia de que se trata de un PMI, sin embargo, este valor es fácilmente
obtenible si la frecuencia del bi-grama y de sus palabras individuales es muy pequeña.

Para tomar en cuenta esta desventaja es que hacemos un gráfico de PMI vs Frecuencia, sin embargo, como el PMI está en escala 
logarítmica es buena idea pasar la frecuencia del bi-grama a la misma escala para que los ejes X & Y sean proporcionales.

![6.png](ims%2F2%2F6.png)

La verdadera interpretación de colaciones son palabras que, en este caso, bi-gramas que tienen un PMI cercano a 0 y una Frecuencia 
más alta. Si la frecuencia es muy alta, pero el PMI es bajo entonces no son colocaciones.

![7.png](ims%2F2%2F7.png)

Tampoco son colocaciones PMI muy bajas si tienen una frecuencia de aparición muy baja.


## 2.11 Filtros y colocaciones en NLTK

Con NLTK podemos realizar el proceso de búsqueda de colocaciones de forma más sistemática y eficiente. En esta clase veremos
cómo utilizar NLTK para encontrar las colocaciones sin tener que utilizar gráficos adicionales.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [9_colocaciones_nltk.py](scripts%2F9_colocaciones_nltk.py)

Para esta sección vamos a apoyarnos principalemnte de 2 métodos de `nltk.collocations` los cuales son:

- BigramAssocMeasures: de aquí vamos a obtener la métrica de PMI
- BigramCollocationFinder: con esto podemos buscar las colocaciones

```python
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import nltk
nltk.download('cess_esp')  # Esta descarga es para tener ejemplos de texto en español
from nltk.book import text1
from pprint import pprint  # solo es para imprimir texto de forma más "linda"
```
Ahora vamos a obtener las colocaciones:
```python
bigram_measures = BigramAssocMeasures()  # Creamos un objeto bigram_measures que tengas la métrica de PMI
finder = BigramCollocationFinder.from_words(text1)  # Empezamos a buscar los bigramas del corpus text1
finder.apply_freq_filter(20)  # nos quedamos con aquellos que tengan al menos una frecuencia de 20
ans = finder.nbest(bigram_measures.pmi, 10)  # obtenemos los 10 mejores ejemplos usando como base el PMI
pprint(ans)
```
Respuesta esperada:
```commandline
[('Moby', 'Dick'),
 ('Sperm', 'Whale'),
 ('White', 'Whale'),
 ('Right', 'Whale'),
 ('Captain', 'Peleg'),
 (',"', 'said'),
 ('never', 'mind'),
 ('!"', 'cried'),
 ('no', 'means'),
 ('each', 'other')]
```
Ahora veamos un ejemplo adicional pero con un corpus de texto en español:
```python
corpus = nltk.corpus.cess_esp.sents()
print(corpus[:2])
flatten_corpus = [w for l in corpus for w in l]  # Descomprimimos la lista de listas en una sola lista
print(flatten_corpus[:50])
```
Respuesta esperada:
```commandline
[['El', 'grupo', 'estatal', 'Electricité_de_France', '-Fpa-', 'EDF', '-Fpt-', 'anunció', 'hoy', ',', 'jueves', ',', 'la', 'compra', 'del', '51_por_ciento', 'de', 'la', 'empresa', 'mexicana', 'Electricidad_Águila_de_Altamira', '-Fpa-', 'EAA', '-Fpt-', ',', 'creada', 'por', 'el', 'japonés', 'Mitsubishi_Corporation', 'para', 'poner_en_marcha', 'una', 'central', 'de', 'gas', 'de', '495', 'megavatios', '.'], ['Una', 'portavoz', 'de', 'EDF', 'explicó', 'a', 'EFE', 'que', 'el', 'proyecto', 'para', 'la', 'construcción', 'de', 'Altamira_2', ',', 'al', 'norte', 'de', 'Tampico', ',', 'prevé', 'la', 'utilización', 'de', 'gas', 'natural', 'como', 'combustible', 'principal', 'en', 'una', 'central', 'de', 'ciclo', 'combinado', 'que', 'debe', 'empezar', 'a', 'funcionar', 'en', 'mayo_del_2002', '.']]
['El', 'grupo', 'estatal', 'Electricité_de_France', '-Fpa-', 'EDF', '-Fpt-', 'anunció', 'hoy', ',', 'jueves', ',', 'la', 'compra', 'del', '51_por_ciento', 'de', 'la', 'empresa', 'mexicana', 'Electricidad_Águila_de_Altamira', '-Fpa-', 'EAA', '-Fpt-', ',', 'creada', 'por', 'el', 'japonés', 'Mitsubishi_Corporation', 'para', 'poner_en_marcha', 'una', 'central', 'de', 'gas', 'de', '495', 'megavatios', '.', 'Una', 'portavoz', 'de', 'EDF', 'explicó', 'a', 'EFE', 'que', 'el', 'proyecto']
```
Buscamos las colocaciones con la misma metodología anteriormente utilizada:

```python
# repetimos el procedimiento anterior pero con el corpus en español
finder = BigramCollocationFinder.from_documents(corpus)
finder.apply_freq_filter(10)
ans = finder.nbest(bigram_measures.pmi, 10)
pprint(ans)
```
Respuesta esperada:
```commandline
[('señora', 'Aguirre'),
 ('secretario', 'general'),
 ('elecciones', 'generales'),
 ('campaña', 'electoral'),
 ('quiere', 'decir'),
 ('Se', 'trata'),
 ('segunda', 'vuelta'),
 ('director', 'general'),
 ('primer', 'ministro'),
 ('primer', 'lugar')]
```
Podemos observar como `campaña electoral` es un excelente ejemplo de una colocación puesto que el significado de ambas palabras
es más que la suma de las palabras aisladas.

## 2.12 Introducción a los recursos léxicos

![8.png](ims%2F2%2F8.png)

Un recurso léxico es una colección de palabras o términos que se utilizan como referencia para diversas tareas lingüísticas. Estos recursos léxicos suelen incluir información adicional asociada a las palabras, como su definición, categoría gramatical, sinónimos, antónimos, relaciones semánticas, información morfológica, frecuencia de uso, entre otros.

![9.png](ims%2F2%2F9.png)

En el ejemplo de esta imagen, el recurso léxico se compone de una `entrada léxica`, `categoría léxica`, `significado`
Los recursos léxicos son de gran utilidad en NLP, ya que proporcionan información valiosa sobre las palabras y permiten enriquecer el análisis lingüístico.

## 2.13 Recursos léxicos en NLTK

Para este momento ya hemos utilizado algunos recursos lexicos, los más simples son el vocabulario y una FreqDist. 

> ## Nota:
> El código completo esta en: [10_recursos_lexicos.py](scripts%2F10_recursos_lexicos.py)

```python
from nltk.book import text1
from nltk.corpus import stopwords
from nltk import FreqDist

# Vocabularios: palabras únicas en un corpus
vocab = sorted(set(text1))
# Esto es un discurso Léxico NO enriquecido
print(vocab[:10])
# Distribuciones: frecuencia de aparición
word_freq = FreqDist(text1)
print(word_freq)
# Esto si tiene información enriquecida (la frecuencia de uso de la palabra)
```
Respuesta esperada:
```commandline
['!', '!"', '!"--', "!'", '!\'"', '!)', '!)"', '!*', '!--', '!--"']
<FreqDist with 19317 samples and 260819 outcomes>
```
Adicionalmente, hemos usado el concepto de stopwords anteriormente pero no se había explicado explícitamente en el curso, 
en nltk podemos observar algunos stopwords de la siguiente manera:

> Las "stopwords" (palabras vacías o de parada, en español) son palabras comunes y frecuentes que se consideran irrelevantes para el análisis de texto debido a que no aportan un significado sustancial. Estas palabras suelen ser preposiciones, conjunciones, artículos y otros términos muy utilizados en el lenguaje, pero que aportan poco valor semántico en un contexto específico.
>
> Ejemplos comunes de stopwords en el idioma inglés incluyen palabras como "a", "the", "is", "and", "in", "of", "to", entre otros. En otros idiomas, las stopwords pueden variar.

```python
print(stopwords.words('spanish')[:10])
```
Respuesta esperada:
```commandline
['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se']
```
Un número que nos ayuda a entender la distribución de stopwords es el porcentaje del mismo, cuantas palabras del total de palabras
son stopwords, eso lo podemos hacer de la siguiente manera:

```python
def stopwords_percentage(text):
    """
    aqui usamos un recurso léxico (stopwords) para filtrar un corpus
    """
    stopwd = stopwords.words('english')
    content = [w for w in text if w.lower() in stopwd]
    return len(content) / len(text)


swp = stopwords_percentage(text1)
print(swp)
```
Respuesta esperada:
```commandline
0.4137045230600531
```
Poco más del 41% de todo el texto de Moby Dick son Stopwords.

## 2.14 NLTK para traducción de palabras

Una de las herramientas que nos proporciona nltk es:  `swadesh` una clase que nos permite hacer traducción simple entre palabras
de diferentes idiomas, primero veamos los idiomas disponibles:

> ## Nota:
> El código completo esta en: [11_traduccion_palabras.py](scripts%2F11_traduccion_palabras.py)

```python
from nltk.corpus import swadesh

# idiomas disponibles
print(swadesh.fileids())
```
Respuesta esperada:
```commandline
['be', 'bg', 'bs', 'ca', 'cs', 'cu', 'de', 'en', 'es', 'fr', 'hr', 'it', 'la', 'mk', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sw', 'uk']
```
Perfecto, son varios, veamos algunas de las palabras disponible que tiene en Inglés:
```python
print(swadesh.words('en'))
```
Respuesta esperada:
```commandline
['I', 'you (singular), thou', 'he', 'we', 'you (plural)', 'they', 'this', 'that', 'here', 'there', 'who', 'what', 'where', 'when', 'how', 'not', 'all', 'many', 'some', 'few', 'other', 'one', 'two', 'three', 'four', 'five', 'big', 'long', 'wide', 'thick', 'heavy', 'small', 'short', 'narrow', 'thin', 'woman', 'man (adult male)', 'man (human being)', 'child', 'wife', 'husband', 'mother', 'father', 'animal', 'fish', 'bird', 'dog', 'louse', 'snake', 'worm', 'tree', 'forest', 'stick', 'fruit', 'seed', 'leaf', 'root', 'bark (from tree)', 'flower', 'grass', 'rope', 'skin', 'meat', 'blood', 'bone', 'fat (noun)', 'egg', 'horn', 'tail', 'feather', 'hair', 'head', 'ear', 'eye', 'nose', 'mouth', 'tooth', 'tongue', 'fingernail', 'foot', 'leg', 'knee', 'hand', 'wing', 'belly', 'guts', 'neck', 'back', 'breast', 'heart', 'liver', 'drink', 'eat', 'bite', 'suck', 'spit', 'vomit', 'blow', 'breathe', 'laugh', 'see', 'hear', 'know (a fact)', 'think', 'smell', 'fear', 'sleep', 'live', 'die', 'kill', 'fight', 'hunt', 'hit', 'cut', 'split', 'stab', 'scratch', 'dig', 'swim', 'fly (verb)', 'walk', 'come', 'lie', 'sit', 'stand', 'turn', 'fall', 'give', 'hold', 'squeeze', 'rub', 'wash', 'wipe', 'pull', 'push', 'throw', 'tie', 'sew', 'count', 'say', 'sing', 'play', 'float', 'flow', 'freeze', 'swell', 'sun', 'moon', 'star', 'water', 'rain', 'river', 'lake', 'sea', 'salt', 'stone', 'sand', 'dust', 'earth', 'cloud', 'fog', 'sky', 'wind', 'snow', 'ice', 'smoke', 'fire', 'ashes', 'burn', 'road', 'mountain', 'red', 'green', 'yellow', 'white', 'black', 'night', 'day', 'year', 'warm', 'cold', 'full', 'new', 'old', 'good', 'bad', 'rotten', 'dirty', 'straight', 'round', 'sharp', 'dull', 'smooth', 'wet', 'dry', 'correct', 'near', 'far', 'right', 'left', 'at', 'in', 'with', 'and', 'if', 'because', 'name']
```
Ahora vamos a crear un traductor de palabras entre Francés y Español:

```python
fr2es = swadesh.entries(['fr', 'es'])
print(fr2es)
```
Respuesta esperada:
```commandline
[('je', 'yo'), ('tu, vous', 'tú, usted'), ('il', 'él'), ('nous', 'nosotros'), ('vous', 'vosotros, ustedes'), ('ils, elles', 'ellos, ellas'), ('ceci', 'este'), ('cela', 'ese, aquel'), ('ici', 'aquí, acá'), ('là', 'ahí, allí, allá'), ('qui', 'quien'), ('quoi', 'que'), ('où', 'donde'), ('quand', 'cuando'), ('comment', 'como'), ('ne...pas', 'no'), ('tout', 'todo'), ('plusieurs', 'muchos'), ('quelques', 'algunos, unos'), ('peu', 'poco'), ('autre', 'otro'), ('un', 'uno'), ('deux', 'dos'), ('trois', 'tres'), ('quatre', 'cuatro'), ('cinq', 'cinco'), ('grand', 'grande'), ('long', 'largo'), ('large', 'ancho'), ('épais', 'gordo'), ('lourd', 'pesado'), ('petit', 'pequeño'), ('court', 'corto'), ('étroit', 'estrecho, angosto'), ('mince', 'delgado, flaco'), ('femme', 'mujer'), ('homme', 'hombre'), ('homme', 'hombre'), ('enfant', 'niño'), ('femme, épouse', 'esposa, mujer'), ('mari, époux', 'esposo, marido'), ('mère', 'madre'), ('père', 'padre'), ('animal', 'animal'), ('poisson', 'pez, pescado'), ('oiseau', 'ave, pájaro'), ('chien', 'perro'), ('pou', 'piojo'), ('serpent', 'serpiente, culebra'), ('ver', 'gusano'), ('arbre', 'árbol'), ('forêt', 'bosque'), ('bâton', 'palo'), ('fruit', 'fruta'), ('graine', 'semilla'), ('feuille', 'hoja'), ('racine', 'raíz'), ('écorce', 'corteza'), ('fleur', 'flor'), ('herbe', 'hierba, pasto'), ('corde', 'cuerda'), ('peau', 'piel'), ('viande', 'carne'), ('sang', 'sangre'), ('os', 'hueso'), ('graisse', 'grasa'), ('œuf', 'huevo'), ('corne', 'cuerno'), ('queue', 'cola'), ('plume', 'pluma'), ('cheveu', 'cabello, pelo'), ('tête', 'cabeza'), ('oreille', 'oreja'), ('œil', 'ojo'), ('nez', 'nariz'), ('bouche', 'boca'), ('dent', 'diente'), ('langue', 'lengua'), ('ongle', 'uña'), ('pied', 'pie'), ('jambe', 'pierna'), ('genou', 'rodilla'), ('main', 'mano'), ('aile', 'ala'), ('ventre', 'barriga, vientre, panza'), ('entrailles', 'entrañas, tripas'), ('cou', 'cuello'), ('dos', 'espalda'), ('sein, poitrine', 'pecho, seno'), ('cœur', 'corazón'), ('foie', 'hígado'), ('boire', 'beber, tomar'), ('manger', 'comer'), ('mordre', 'morder'), ('sucer', 'chupar'), ('cracher', 'escupir'), ('vomir', 'vomitar'), ('souffler', 'soplar'), ('respirer', 'respirar'), ('rire', 'reír'), ('voir', 'ver'), ('entendre', 'oír'), ('savoir', 'saber'), ('penser', 'pensar'), ('sentir', 'oler'), ('craindre, avoir peur', 'temer'), ('dormir', 'dormir'), ('vivre', 'vivir'), ('mourir', 'morir'), ('tuer', 'matar'), ('se battre', 'pelear'), ('chasser', 'cazar'), ('frapper', 'golpear'), ('couper', 'cortar'), ('fendre', 'partir'), ('poignarder', 'apuñalar'), ('gratter', 'arañar, rascar'), ('creuser', 'cavar'), ('nager', 'nadar'), ('voler', 'volar'), ('marcher', 'caminar'), ('venir', 'venir'), ("s'étendre", 'echarse, acostarse, tenderse'), ("s'asseoir", 'sentarse'), ('se lever', 'estar de pie'), ('tourner', 'voltear'), ('tomber', 'caer'), ('donner', 'dar'), ('tenir', 'sostener'), ('serrer', 'apretar'), ('frotter', 'frotar'), ('laver', 'lavar'), ('essuyer', 'limpiar'), ('tirer', 'tirar'), ('pousser', 'empujar'), ('jeter', 'tirar'), ('lier', 'atar'), ('coudre', 'coser'), ('compter', 'contar'), ('dire', 'decir'), ('chanter', 'cantar'), ('jouer', 'jugar'), ('flotter', 'flotar'), ('couler', 'fluir'), ('geler', 'helar'), ('gonfler', 'hincharse'), ('soleil', 'sol'), ('lune', 'luna'), ('étoile', 'estrella'), ('eau', 'agua'), ('pluie', 'lluvia'), ('rivière', 'río'), ('lac', 'lago'), ('mer', 'mar'), ('sel', 'sal'), ('pierre', 'piedra'), ('sable', 'arena'), ('poussière', 'polvo'), ('terre', 'tierra'), ('nuage', 'nube'), ('brouillard', 'niebla'), ('ciel', 'cielo'), ('vent', 'viento'), ('neige', 'nieve'), ('glace', 'hielo'), ('fumée', 'humo'), ('feu', 'fuego'), ('cendres', 'cenizas'), ('brûler', 'quemar'), ('route', 'camino'), ('montagne', 'montaña'), ('rouge', 'rojo'), ('vert', 'verde'), ('jaune', 'amarillo'), ('blanc', 'blanco'), ('noir', 'negro'), ('nuit', 'noche'), ('jour', 'día'), ('an, année', 'año'), ('chaud', 'cálido, tibio'), ('froid', 'frío'), ('plein', 'lleno'), ('nouveau', 'nuevo'), ('vieux', 'viejo'), ('bon', 'bueno'), ('mauvais', 'malo'), ('pourri', 'podrido'), ('sale', 'sucio'), ('droit', 'recto'), ('rond', 'redondo'), ('tranchant, pointu, aigu', 'afilado'), ('émoussé', 'desafilado'), ('lisse', 'suave, liso'), ('mouillé', 'mojado'), ('sec', 'seco'), ('juste, correct', 'correcto'), ('proche', 'cerca'), ('loin', 'lejos'), ('à droite', 'derecha'), ('à gauche', 'izquierda'), ('à', 'a, en, ante'), ('dans', 'en'), ('avec', 'con'), ('et', 'y'), ('si', 'si'), ('parce que', 'porque'), ('nom', 'nombre')]
```
Pongamos a prueba el traductor:

```python
translate = dict(fr2es)
print(translate['chien'])

print(translate['jeter'])
```
Respuesta esperada:
```commandline
perro
tirar
```

## 2.15 Introducción a WordNet

![10.png](ims%2F2%2F10.png)

WordNet: WordNet es una base de datos léxica y un recurso léxico ampliamente utilizado en el campo del procesamiento del lenguaje natural (NLP). Proporciona un conjunto organizado de palabras en inglés agrupadas en conjuntos de sinónimos, conocidos como "synsets" (conjuntos de sinónimos). WordNet agrupa las palabras en categorías semánticas y establece relaciones entre ellas, lo que permite explorar y comprender las relaciones entre las palabras.

![11.png](ims%2F2%2F11.png)

Synset: Un synset (conjunto de sinónimos) en WordNet es un conjunto de palabras que tienen un significado semántico similar o relacionado. Cada synset está asociado con una palabra clave o "lemmainform" y contiene sinónimos (palabras que comparten un significado similar), definiciones, ejemplos de uso y otras relaciones semánticas. Por ejemplo, el synset para el término "car" (automóvil) en WordNet incluye palabras como "automobile", "motorcar", "machine", etc.

![12.png](ims%2F2%2F12.png)


Hiperónimos e Hipónimos: En WordNet, las relaciones hiperónimo-hipónimo (también conocidas como relaciones de "es un") establecen la jerarquía semántica entre los synsets. Un hiperónimo es un término más general o una categoría que engloba a otros términos más específicos llamados hipónimos. Por ejemplo, en el synset "automobile" (automóvil), "vehicle" (vehículo) es su hiperónimo, ya que un automóvil es un tipo de vehículo. A su vez, "sedan" (sedán) podría ser un hipónimo de "automobile", ya que un sedán es un tipo específico de automóvil.

Estas relaciones jerárquicas de hiperónimos e hipónimos en WordNet permiten organizar y estructurar el vocabulario en niveles de generalidad y especificidad, lo que facilita la comprensión de las relaciones semánticas entre las palabras y su uso en diversas aplicaciones de procesamiento del lenguaje natural, como la búsqueda de información, el análisis de sentimientos, la desambiguación léxica, entre otros.

## 2.16 Explorando WordNet

En esta sección vamos a ver ejemplos de Synsets y sus Hiperónimos e Hipónimos.

> ## Nota: 
> El código de esta sección está en: [12_explorando_wordnet.py](scripts%2F12_explorando_wordnet.py)

Importamos las bibliotecas necesarias:
```python
import nltk
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from pprint import pprint

```

Escribamos una palabra de ejemplo y veamos sus synsets:

```python
ss = wn.synsets('carro', lang='spa')
pprint(ss)
```
Respuesta esperada:
```commandline
[Synset('car.n.01'),
 Synset('carriage.n.04'),
 Synset('carrier.n.02'),
 Synset('cart.n.01'),
 Synset('chariot.n.02'),
 Synset('cartload.n.01')]
```
Para cada synset de la palabra Carro podemos observar cuál es su nombre, definición y al mismo tiempo las palabras asociadas al mismo:

```python
# Explorando los synsets
for syn in ss:
    print(syn.name(), ': ', syn.definition())
    for name in syn.lemma_names():
        print(' * ', name)
```
Respuesta esperada:
```commandline
car.n.01 :  a motor vehicle with four wheels; usually propelled by an internal combustion engine
 *  car
 *  auto
 *  automobile
 *  machine
 *  motorcar
carriage.n.04 :  a machine part that carries something else
 *  carriage
carrier.n.02 :  a self-propelled wheeled vehicle designed specifically to carry something
 *  carrier
cart.n.01 :  a heavy open wagon usually having two wheels and drawn by an animal
 *  cart
chariot.n.02 :  a two-wheeled horse-drawn battle vehicle; used in war and races in ancient Egypt and Greece and Rome
 *  chariot
cartload.n.01 :  the quantity that a cart holds
 *  cartload
```
Ahora veamos ejemplos de `Hipónimos` e `Hiperónimos`:
```python
# Palabras más específicas
pprint(ss[0].hyponyms())

# Palabras más generales
pprint(ss[0].hypernyms())
```
Respuesta esperada:
```commandline
[Synset('ambulance.n.01'),
 Synset('beach_wagon.n.01'),
 Synset('bus.n.04'),
 Synset('cab.n.03'),
 Synset('compact.n.03'),
 Synset('convertible.n.01'),
 Synset('coupe.n.01'),
 Synset('cruiser.n.01'),
 Synset('electric.n.01'),
 Synset('gas_guzzler.n.01'),
 Synset('hardtop.n.01'),
 Synset('hatchback.n.01'),
 Synset('horseless_carriage.n.01'),
 Synset('hot_rod.n.01'),
 Synset('jeep.n.01'),
 Synset('limousine.n.01'),
 Synset('loaner.n.02'),
 Synset('minicar.n.01'),
 Synset('minivan.n.01'),
 Synset('model_t.n.01'),
 Synset('pace_car.n.01'),
 Synset('racer.n.02'),
 Synset('roadster.n.01'),
 Synset('sedan.n.01'),
 Synset('sport_utility.n.01'),
 Synset('sports_car.n.01'),
 Synset('stanley_steamer.n.01'),
 Synset('stock_car.n.01'),
 Synset('subcompact.n.01'),
 Synset('touring_car.n.01'),
 Synset('used-car.n.01')]
```
```commandline
[Synset('motor_vehicle.n.01')]
```

Ahora vamos a graficar esta información. Creamos un par de funciones que dejare comentadas con Docstrings:

```python
import networkx as nx
import matplotlib.pyplot as plt


def closure_graph(synset, fn):
    """
       Crea un grafo de cierre a partir de un synset dado y una función.

       Args:
           synset (nltk.corpus.reader.wordnet.Synset): Synset de inicio.
           fn (function): Función que toma un Synset y devuelve una lista de Synsets.

       Returns:
           tuple: Tupla que contiene el grafo y un diccionario de etiquetas.
       """
    seen = set()
    graph = nx.DiGraph()
    labels = {}

    def recurse(s):
        """
                Función recursiva para construir el grafo de cierre.

                Args:
                    s (nltk.corpus.reader.wordnet.Synset): Synset actual.
                """
        if not s in seen:
            seen.add(s)
            labels[s.name] = s.name().split('.')[0]
            graph.add_node(s.name)
            for s1 in fn(s):
                graph.add_node(s1.name)
                graph.add_edge(s.name, s1.name)
                recurse(s1)

    recurse(synset)
    return graph, labels


def draw_text_graph(G, labels, filename):
    """
        Dibuja un grafo de texto utilizando NetworkX y Matplotlib.

        Args:
            G (networkx.DiGraph): Grafo dirigido.
            labels (dict): Diccionario de etiquetas de nodos.
            filename (str): Nombre del archivo a guardar
        """
    plt.figure(figsize=(18, 12))
    pos = nx.planar_layout(G, scale=18)
    nx.draw_networkx_nodes(G, pos, node_color="red", linewidths=0, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=20, labels=labels)
    nx.draw_networkx_edges(G, pos)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    plt.close()
```

Ahora podemos graficar un par de ejemplos:
```python
# Conceptos que son más específicos que la palabra raíz de la cual derivan.
print(ss[0].name())
G, labels = closure_graph(ss[0], fn=lambda s: s.hyponyms())
draw_text_graph(G, labels, "carro_hypo.png")
```
Respuesta esperada:
![carro_hypo.png](scripts%2Fcarro_hypo.png)

```python
# Conceptos que son más generales.
print(ss[0].name())
G, labels = closure_graph(ss[0], fn=lambda s: s.hypernyms())
draw_text_graph(G, labels, "carro_hyper.png")
```
Respuesta esperada:
![carro_hyper.png](scripts%2Fcarro_hyper.png)


## 2.17 Similitud Semántica con WordNet

Podemos utilizar la estructura de nodos para saber cuál es la distancia semántica entre dos palabras, la cuál puede ser medida
como la cantidad de nodos que separa a dos palabras, pero el propio wordnet ya incluye una función llamada:
`path_similarity` que nos ayuda con este trabajo:

> ## Nota:
> El código de está sección lo puedes encontrar en: [13_similitud_semantica.py](scripts%2F13_similitud_semantica.py)

Este código lo vimos en la sección pasada, ahora lo pondremos en forma de función:

```python
from nltk.corpus import wordnet as wn


def show_syns(word):
    # Obtener los synsets (conjunto de sinónimos) para la palabra proporcionada
    ss = wn.synsets(word, lang='spa')

    for syn in ss:
        # Imprimir el nombre del synset y su definición
        print(syn.name(), ': ', syn.definition())

        for name in syn.lemma_names():
            # Imprimir cada nombre de lema (sinónimo)
            print(' * ', name)

    return ss
```

Obtenemos un par de synsets para algunas palabras:
```python
# Obtener los synsets para la palabra 'perro' y mostrarlos
ss = show_syns('perro')
```
Respuesta esperada:
```commandline
dog.n.01 :  a member of the genus Canis (probably descended from the common wolf) that has been domesticated by man since prehistoric times; occurs in many breeds
 *  dog
 *  domestic_dog
 *  Canis_familiaris
rotter.n.01 :  a person who is deemed to be despicable or contemptible
 *  rotter
 *  dirty_dog
 *  rat
 *  skunk
 *  stinker
 *  stinkpot
 *  bum
 *  puke
 *  crumb
 *  lowlife
 *  scum_bag
 *  so-and-so
 *  git
```
Synsets de Gato
```python
# Obtener los synsets para la palabra 'gato' y mostrarlos
ss2 = show_syns('gato')
```
Respuesta esperada:
```commandline
cat.n.01 :  feline mammal usually having thick soft fur and no ability to roar: domestic cats; wildcats
 *  cat
 *  true_cat
tom.n.02 :  male cat
 *  tom
 *  tomcat
dodger.n.01 :  a shifty deceptive person
 *  dodger
 *  fox
 *  slyboots
```
Synsets de Animal:
```python
# Obtener los synsets para la palabra 'animal' y mostrarlos
ss3 = show_syns('animal')
```
Respuesta esperada:
```commandline
animal.n.01 :  a living organism characterized by voluntary movement
 *  animal
 *  animate_being
 *  beast
 *  brute
 *  creature
 *  fauna
beast.n.02 :  a cruelly rapacious person
 *  beast
 *  wolf
 *  savage
 *  brute
 *  wildcat
dunce.n.01 :  a stupid person; these words are used to express a low opinion of someone's intelligence
 *  dunce
 *  dunderhead
 *  numskull
 *  blockhead
 *  bonehead
 *  lunkhead
 *  hammerhead
 *  knucklehead
 *  loggerhead
 *  muttonhead
 *  shithead
 *  dumbass
 *  fuckhead
```

Comparación entre palabras (distancia semántica):

```python
# Obtener el primer synset de 'perro'
perro = ss[0]

# Obtener el primer synset de 'gato'
gato = ss2[0]

# Obtener el primer synset de 'animal'
animal = ss3[0]

# Calcular y mostrar la similitud de ruta entre 'animal' y 'perro'
print(animal.path_similarity(perro))  # 0.3333333333333333

# Calcular y mostrar la similitud de ruta entre 'animal' y 'gato'
print(animal.path_similarity(gato))  # 0.125

# Calcular y mostrar la similitud de ruta entre 'perro' y 'gato'
print(perro.path_similarity(gato))  # 0.2

# Calcular y mostrar la similitud de ruta entre 'perro' y 'perro' (misma palabra)
print(perro.path_similarity(perro))  # 1.0
```
Respuesta esperada:
```commandline
0.3333333333333333
0.125
0.2
1.0
```

## 2.18 Procesamiento de texto plano desde Web

En esa clase vamos a ver como podemos acceder fácilmente a texto plano de páginas web descargando a través de `urllib` 
y como incluso podemos descargar el texto plano de una página web utilizando `BeautifulSoup`para después limpiarlo.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [14_procesamiento_texto_web.py](scripts%2F14_procesamiento_texto_web.py) 

Importamos las bibliotecas necesarias:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from urllib import request
```
Empecemos con algo muy simple, acceder a texto plano de una página web:

```python
# Texto plano desde web
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')

print(raw[:200])
```
Respuesta esperada:
```commandline
﻿The Project Gutenberg eBook of Crime and Punishment, by Fyodor Dostoevsky

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with a ..
```
Observamos como la página contiene el libro de `Crimen y Castigo` de `Fyodor Dostoevsky`.
Ahora podemos tokenizar las palabras con `word_tokenize`

```python
tokens = word_tokenize(raw)
print(tokens[:20])
```
Respuesta esperada:
```commandline
['\ufeffThe', 'Project', 'Gutenberg', 'eBook', 'of', 'Crime', 'and', 'Punishment', ',', 'by', 'Fyodor', 'Dostoevsky', 'This', 'eBook', 'is', 'for', 'the', 'use', 'of', 'anyone']
```
Podemos transformar este lista de palabras en un `Text` de `NLTK`con el fin de acceder a otros métodos interesantes como lo son
las colocaciones.

```python
text = nltk.Text(tokens)
a = text.collocations()
print(a)
```
Respuesta esperada:
```commandline
Katerina Ivanovna; Pyotr Petrovitch; Pulcheria Alexandrovna; Avdotya
Romanovna; Rodion Romanovitch; Marfa Petrovna; Sofya Semyonovna; old
woman; Project Gutenberg-tm; Porfiry Petrovitch; Amalia Ivanovna;
great deal; young man; Nikodim Fomitch; Project Gutenberg; Ilya
Petrovitch; Andrey Semyonovitch; Hay Market; Dmitri Prokofitch; Good
heavens
```
Es fácil observar que varias de las colocaciones son nombres propios de los personajes de la obra.

Ahora veamos como podemos obtener el texto de una página HTML y después limpiarlo para proceder a tokenizarlo.
Importamos bibliotecas:
```python
# Procesar HTML
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
```
Hacemos un request a una página web para obtener todo su código HTML:
```python
url = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'
r = requests.get(url)

html = r.text
print(html[:200])
```
Respuesta esperada:
```commandline
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-
```
Ahora debemos hacer un `parse` de este texto html con Beautiful soup y después podemos tokenizar las palabras:
```python
soup = BeautifulSoup(html, 'html.parser')
# print(soup)

text = soup.get_text()
tokens = re.findall('\w+', text)
print(tokens[:10])
```
Respuesta esperada:
```commandline
['The', 'Project', 'Gutenberg', 'eBook', 'of', 'Moby', 'Dick', 'Or', 'the', 'Whale']
```
El tokenizador empleado fue sumamente sencillo simplemente fue quedarse con los caracteres de palabra.
Una alternativa sería:
```python
tokenizer = RegexpTokenizer('\w+')
tokens = tokenizer.tokenize(text)
tokens = [token.lower() for token in tokens]
print(tokens[:10])
```
Respuesta esperada:
```commandline
['the', 'project', 'gutenberg', 'ebook', 'of', 'moby', 'dick', 'or', 'the', 'whale']
```
De igual forma podemos transformar los tokens a un `Text`:
```python
text = nltk.Text(tokens)
a = text.collocations()
print(a)
```
Respuesta esperada:
```commandline
sperm whale; moby dick; project gutenberg; white whale; old man; mast
head; mast heads; captain ahab; right whale; quarter deck; united
states; aye aye; captain peleg; new bedford; literary archive; mrs
hussey; dough boy; chief mate; gutenberg electronic; try works
None
```

## 2.19 Usando código estructurado: conexión de Drive a Google Colab

Esta clase solo introdujo como se puede acceder a Drive desde Colab. Realmente por como tengo estructurado 
este repositorio tiene poco sentido que lo explique, sin embargo, te comparto el código por si no sabes cómo hacer eso:

```python
from google.colab import drive
drive.mount('/content/drive')
filepath = '/content/drive/My Drive/Colab Notebooks/NLP_course_resources/'
```
Cuando corras el código desde Google Colab te va a pedir que ingreses una contraseña, pero esta se va a generar directamente
desde el navegador.


## 2.20 Usando código estructurado: Funciones externas

De nuevo esta clase no tiene mucho sentido si no se ejecuta desde Colab, pero el objetivo de la misma fue hacer códigos python (.py)
e importarlos desde Colab para usarlos como si de librerías se trataran.

En colab la forma de permitirle a python acceder a archivos .py ya que han sido montados en drive es la siguiente:

Supongamos la existencia de un archivo `read.py` que se encuentra en la siguiente dirección:

```python
filepath = '/content/drive/My Drive/Colab Notebooks/NLP_course_resources/'
```

El archivo `read.py` tiene el siguiente código en su interior:
```python
import re

# la funcion la podemos definir en el notebook y usar directamente
def get_text(file):
  '''Read Text from file'''
  text = open(file).read()
  text = re.sub(r'<.*?>', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  return text
```

Entonces para importarla como un modelo en Colab y poder usar la función `get_text()` podemos hacer:

```python
import sys
sys.path.append(filepath)
import read  # este es el nomber del archivo .py
read.get_text(filepath+'book.txt')
```
Respuesta esperada:
```commandline
'\ufeffThe Project Gutenberg EBook of Crime and Punishment, by Fyodor Dostoevsky This eBook is for the use of anyone anywhere at no cost and with almost no restrictions
```

# 3 Perspectivas de lo que viene

## 3.1 Continúa con el Curso de Algoritmos de Clasificación de Texto

En los siguientes cursos veremos como usar spaCy, y veremos un poco sobre Como llevar a producción los modelos.



