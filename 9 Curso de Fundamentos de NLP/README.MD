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

## 2.1 Configurar ambiente de trabajo

## 2.2 Palabras, textos y vocabularios

## 2.3 Tokenizacion con Expresiones Regulares

## 2.4 Estadísticas básicas del lenguaje

## 2.5 Distribuciones de frecuencia de palabras

## 2.6 Refinamiento y visualización de cuerpos de texto

## 2.7 N-gramas y Colocaciones del lenguaje

## 2.8 ¿Cómo extraer n-gramas de un texto en Python?

## 2.9 Colocaciones en Python

## 2.10 Colocaciones en gráficos de dispersión

## 2.11 Filtros y colocaciones en NLTK

## 2.12 Introducción a los recursos léxicos

## 2.13 Recursos léxicos en NLTK

## 2.14 NLTK para traducción de palabras

## 2.15 Introducción a WordNet

## 2.16 Explorando WordNet

## 2.17 Similitud Semántica con WordNet

## 2.18 Procesamiento de texto plano desde Web

## 2.19 Usando código estructurado: conexión de Drive a Google Colab

## 2.20 Usando código estructurado: Funciones externas

# 3 Perspectivas de lo que viene

## 3.1 Continúa con el Curso de Algoritmos de Clasificación de Texto
