# Curso de Fundamentos de Procesamiento de Lenguaje Natural con Python y NLTK

El procesamiento de lenguaje natural (NLP) nos permite comprender el lenguaje humano, uno de sus usos más comunes es realizar 
análisis de sentimientos. Comprende conceptos como la desambiguación, domina el etiquetado de palabras y aprende a implementar 
algoritmos de clasificación de texto desde cero usando Python yNLTK.

- Descubrir las aplicaciones de la clasificación de texto y el procesamiento de lenguaje natural (NLP)
- Entender tareas del NLP como la desambiguación y el etiquetado de palabras
- Comprender y usar los algoritmos de clasificación de texto con Python y NLTK
- Implementar tu propia versión del Modelo Markoviano de Máxima Entropía (MMME)


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
> 
> Este Curso es el Número 9 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice

- [1 Desambiguación y etiquetado de palabras](#1-desambiguación-y-etiquetado-de-palabras)
  - [1.1 Introducción a la desambiguación](#11-introducción-a-la-desambiguación)
  - [1.2 Etiquetado rápido en Python: español e inglés](#12-etiquetado-rápido-en-python-español-e-inglés)
  - [1.3 Etiquetado rápido en Python: Stanza (Stanford NLP)](#13-etiquetado-rápido-en-python-stanza-stanford-nlp)
- [2 Modelos Markovianos Latentes (HMM)](#2-modelos-markovianos-latentes-hmm)
  - [2.1 Cadenas de Markov](#21-cadenas-de-markov)
  - [2.2 Modelos Markovianos latentes (HMM)](#22-modelos-markovianos-latentes-hmm)
  - [2.3 Entrenando un HMM](#23-entrenando-un-hmm)
  - [2.4 Fases de entrenamiento de un HMM](#24-fases-de-entrenamiento-de-un-hmm)
  - [2.5 Entrenando un HMM en Python](#25-entrenando-un-hmm-en-python)
- [3 Algoritmo de Viterbi](#3-algoritmo-de-viterbi)
  - [3.1 El algoritmo de Viterbi](#31-el-algoritmo-de-viterbi)
  - [3.2 Cálculo de las probabilidades de Viterbi](#32-cálculo-de-las-probabilidades-de-viterbi)
  - [3.3 Carga del modelo HMM y distribución inicial](#33-carga-del-modelo-hmm-y-distribución-inicial)
  - [3.4 Implementación de algoritmo de Viterbi en Python](#34-implementación-de-algoritmo-de-viterbi-en-python)
  - [3.5 Entrenamiento directo de HMM con NLTK](#35-entrenamiento-directo-de-hmm-con-nltk)
- [4 Modelos Markovianos de máxima entropía (MEMM)](#4-modelos-markovianos-de-máxima-entropía-memm)
  - [4.1 Modelos Markovianos de máxima entropia (MEMM)](#41-modelos-markovianos-de-máxima-entropia-memm)
  - [4.2 Algoritmo de Viterbi para MEMM](#42-algoritmo-de-viterbi-para-memm)
  - [4.3 Reto: construye un MEMM en Python](#43-reto-construye-un-memm-en-python)
- [5 Clasificación de texto con NLTK](#5-clasificación-de-texto-con-nltk)
  - [5.1 El problema general de la clasificación de texto](#51-el-problema-general-de-la-clasificación-de-texto)
  - [5.2 Tareas de clasificación con NLTK](#52-tareas-de-clasificación-con-nltk)
  - [5.3 Modelos de clasificación en Python: nombres](#53-modelos-de-clasificación-en-python-nombres)
  - [5.4 Modelos de clasificación en Python: documentos](#54-modelos-de-clasificación-en-python-documentos)
- [6 Implementación de un modelo de clasificación de texto](#6-implementación-de-un-modelo-de-clasificación-de-texto)
  - [6.1 Naive Bayes](#61-naive-bayes)
  - [6.2 Naive Bayes en Python: preparación de los datos](#62-naive-bayes-en-python-preparación-de-los-datos)
  - [6.3 Naive Bayes en Python: construcción del modelo](#63-naive-bayes-en-python-construcción-del-modelo)
  - [6.4 Naive Bayes en Python: ejecución del modelo](#64-naive-bayes-en-python-ejecución-del-modelo)
  - [6.5 Métricas para algoritmos de clasificación](#65-métricas-para-algoritmos-de-clasificación)
  - [6.6 Reto final: construye un modelo de sentimientos](#66-reto-final-construye-un-modelo-de-sentimientos)


# 1 Desambiguación y etiquetado de palabras

## 1.1 Introducción a la desambiguación

La desambiguación de palabras es una tarea desafiante en el procesamiento del lenguaje natural (NLP, por sus siglas en inglés) que implica identificar el significado correcto de una palabra o frase en un determinado contexto. 

![1.png](imgs%2F1%2F1.png)

En la primera oración, "Debo ir al banco para retirar dinero", la palabra "banco" se refiere a una institución financiera 
donde las personas pueden realizar transacciones bancarias, como depositar, retirar o transferir dinero. Sin embargo, en 
la segunda oración, "Te puedes sentar en ese banco para descansar", "banco" se refiere a un asiento o superficie para sentarse.

La desambiguación léxica es un problema importante en el NLP porque las palabras a menudo tienen múltiples significados y el contexto circundante es crucial para determinar cuál es el significado correcto en un contexto dado. La tarea de desambiguación léxica puede abordarse de diferentes formas.

**Algunos ejemplos de ambigüedad:**

![3.png](imgs%2F1%2F3.png)

![4.png](imgs%2F1%2F4.png)

Este ejemplo es particularmente complejo porque en este escenario el texto es idéntico en ambas partes, sin embargo, la primera
oración hace referencia a que el animal está a punto de comer mientras que la segunda hace referencia a que está a punto de ser comido.

![5.png](imgs%2F1%2F5.png)

Algunas de las formas de lidiar con la desambiguación léxica son:

Basado en reglas: En este enfoque, se utilizan reglas lingüísticas predefinidas para determinar el significado de una palabra en función de su contexto. Estas reglas pueden ser diseñadas manualmente o aprenderse automáticamente a partir de un corpus de texto anotado. Por ejemplo, una regla podría establecer que si la palabra "banco" aparece junto a palabras como "retirar", "depositar" o "transacciones", probablemente se refiere a una institución financiera.

Basado en aprendizaje automático: En este enfoque, se utilizan técnicas de aprendizaje automático para entrenar modelos que puedan predecir el significado de una palabra en función de su contexto. Estos modelos se entrenan utilizando grandes conjuntos de datos anotados, donde se especifica el significado correcto de las palabras en diferentes oraciones. El modelo aprende patrones y relaciones entre las palabras y sus contextos y luego puede aplicarse para desambiguar nuevas oraciones. En este caso, el modelo podría aprender que en la primera oración, "banco" se refiere a una institución financiera porque se encuentra junto a palabras relacionadas con transacciones bancarias.

En este curso vamos a empezar por entender cómo debe ser etiquetada una palabra de forma correcta.

Conozcamos un ejemplo de google: https://cloud.google.com/natural-language?hl=es

![6.png](imgs%2F1%2F6.png)

Nos permite extraer información sobre cada una de las palabras que componen una oración tomando en consideración el contexto
que rodea a la palabra.

**Algunas de las aplicaciones del NLP hoy en día son:**

- Mejoras en motores de búsqueda, e-commerce y web.
- Automatización en manejo de CRMs.
- Censura en redes sociales.
-Orden de datos no-estructurados.

## 1.2 Etiquetado rápido en Python: español e inglés

En esta clase veremos como podemos etiquetar cada una de las palabras de una oración y entender su significado de acuerdo
al contexto de la palabra, de una forma similar a como lo hizo el ejemplo de Google.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [1_etiquetado_rapido.py](scripts%2F1_etiquetado_rapido.py)

Primero importamos las bibliotecas necesarias:
```python
# Dependencias previas
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
nltk.download('cess_esp')
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
```
Este código importa las bibliotecas necesarias para realizar el etiquetado gramatical. NLTK (Natural Language Toolkit) es una biblioteca popular para el procesamiento del lenguaje natural en Python. Aquí, se descargan los recursos necesarios, como los modelos de etiquetado y los datos de entrenamiento.

El siguiente fragmento muestra un ejemplo sencillo de etiquetado gramatical utilizando NLTK. La función word_tokenize se utiliza para dividir el texto en palabras individuales. Luego, pos_tag se aplica a la lista de palabras para etiquetar cada una con su categoría gramatical.
```python
# Etiquetado en una línea
text = word_tokenize("And now here I am enjoying today")
print(nltk.pos_tag(text))
```
Respuesta esperada:
```commandline
[('And', 'CC'), ('now', 'RB'), ('here', 'RB'), ('I', 'PRP'), ('am', 'VBP'), ('enjoying', 'VBG'), ('today', 'NN')]
```

En este código, se descargan los conjuntos de etiquetas gramaticales definidos por el proyecto Penn Treebank. Luego, se itera sobre las etiquetas gramaticales generadas anteriormente por pos_tag. Para cada etiqueta, se utiliza la función help.upenn_tagset para obtener una descripción de la etiqueta gramatical correspondiente. Esto ayuda a comprender el significado de cada etiqueta.
```python
# Categoria gramatical de cada etiqueta
nltk.download('tagsets')
for _, tag in nltk.pos_tag(text):
    print(nltk.help.upenn_tagset(tag))
```
Respuesta esperada:
```commandline
CC: conjunction, coordinating
    & 'n and both but either et for less minus neither nor or plus so
    therefore times v. versus vs. whether yet
None
RB: adverb
    occasionally unabatingly maddeningly adventurously professedly
    stirringly prominently technologically magisterially predominately
    swiftly fiscally pitilessly ...
None
RB: adverb
    occasionally unabatingly maddeningly adventurously professedly
    stirringly prominently technologically magisterially predominately
    swiftly fiscally pitilessly ...
None
PRP: pronoun, personal
    hers herself him himself hisself it itself me myself one oneself ours
    ourselves ownself self she thee theirs them themselves they thou thy us
None
VBP: verb, present tense, not 3rd person singular
    predominate wrap resort sue twist spill cure lengthen brush terminate
    appear tend stray glisten obtain comprise detest tease attract
    emphasize mold postpone sever return wag ...
None
VBG: verb, present participle or gerund
    telegraphing stirring focusing angering judging stalling lactating
    hankerin' alleging veering capping approaching traveling besieging
    encrypting interrupting erasing wincing ...
None
NN: noun, common, singular or mass
    common-carrier cabbage knuckle-duster Casino afghan shed thermostat
    investment slide humour falloff slick wind hyena override subhumanity
    machinist ...
None
```
Este fragmento muestra otro ejemplo de etiquetado gramatical utilizando NLTK. Aquí se etiquetan las palabras de una oración y se imprime el resultado en la consola. Es interesante observar cómo NLTK maneja las palabras homónimas en función del contexto.
```python
# Palabras homónimas
text = word_tokenize("They do not permit other people to get residence permit")
print(nltk.pos_tag(text))
```
Respuesta esperada:
```commandline
[('They', 'PRP'), ('do', 'VBP'), ('not', 'RB'), ('permit', 'VB'), ('other', 'JJ'), ('people', 'NNS'), ('to', 'TO'), ('get', 'VB'), ('residence', 'NN'), ('permit', 'NN')]
```
Este código muestra el etiquetado gramatical en español utilizando un enfoque basado en unigramas. Los datos etiquetados en español se cargan desde el corpus "cess_esp". Luego, se divide el corpus en un conjunto de entrenamiento y un conjunto de prueba. Se entrena el etiquetador unigrama utilizando el conjunto de entrenamiento y se evalúa su rendimiento en el conjunto de prueba. Finalmente, se muestra el rendimiento del etiquetador y se etiqueta una frase en español utilizando el modelo entrenado.
```python
# Etiquetado en Español
# Entrenamiendo del tagger por unigramas
cess_sents = cess.tagged_sents()
fraction = int(len(cess_sents)*90/100)
uni_tagger = ut(cess_sents[:fraction])
p = uni_tagger.evaluate(cess_sents[fraction+1:])
print(p)
print(uni_tagger.tag("Yo soy una persona muy amable".split(" ")))
```
Respuesta esperada:
```commandline
0.8069484240687679
[('Yo', 'pp1csn00'), ('soy', 'vsip1s0'), ('una', 'di0fs0'), ('persona', 'ncfs000'), ('muy', 'rg'), ('amable', None)]
```
Este fragmento de código es similar al anterior, pero utiliza un etiquetador basado en bigramas en lugar de unigramas. Se realiza el entrenamiento del etiquetador bigrama y se evalúa su rendimiento en el conjunto de prueba. Luego, se muestra el rendimiento del etiquetador y se etiqueta una frase en español utilizando el modelo entrenado.
```python
fraction = int(len(cess_sents)*90/100)
bi_tagger = bt(cess_sents[:fraction])
p = bi_tagger.evaluate(cess_sents[fraction+1:])
print(p)

print(bi_tagger.tag("Yo soy una persona muy amable".split(" ")))
```
Respuesta esperada:
```commandline
0.1095272206303725
[('Yo', 'pp1csn00'), ('soy', 'vsip1s0'), ('una', None), ('persona', None), ('muy', None), ('amable', None)]
```


## 1.3 Etiquetado rápido en Python: Stanza (Stanford NLP)

Stanza, anteriormente conocido como Stanford NLP, es una biblioteca de procesamiento del lenguaje natural (NLP, por sus siglas en inglés) desarrollada por el Grupo de Procesamiento del Lenguaje Natural de la Universidad de Stanford. Proporciona una amplia gama de herramientas y modelos para tareas de procesamiento del lenguaje natural, como el etiquetado gramatical, el análisis sintáctico, el reconocimiento de entidades nombradas, el análisis de sentimientos, la desambiguación del sentido de las palabras y mucho más.

A continuación algunos aspectos destacados y características clave de Stanza:

- **Procesamiento en varios idiomas:** Stanza admite una amplia variedad de idiomas y proporciona modelos y recursos preentrenados para más de 70 idiomas diferentes, lo que permite el procesamiento del lenguaje natural en un contexto multilingüe.

- **Pipeline de procesamiento:** Stanza sigue un enfoque de pipeline modular, donde cada tarea de procesamiento, como el tokenizado, el etiquetado gramatical y el análisis sintáctico, se realiza de manera secuencial. Esto permite un flujo de trabajo flexible y la posibilidad de personalizar el pipeline según las necesidades del proyecto.

- **Modelos de aprendizaje automático:** Stanza utiliza técnicas de aprendizaje automático y modelos neuronales para realizar diversas tareas de procesamiento del lenguaje natural. Estos modelos se han entrenado en grandes conjuntos de datos anotados y proporcionan un rendimiento preciso y robusto en las tareas de NLP.

- **Integración con Python:** Stanza está diseñado para ser utilizado con Python, lo que facilita la integración en proyectos de NLP existentes. Proporciona una API intuitiva y de fácil uso para acceder a las funcionalidades de procesamiento del lenguaje natural.

- **Recursos lingüísticos:** Stanza incluye recursos lingüísticos, como modelos de lenguaje, etiquetas gramaticales y datos anotados, que se pueden descargar y utilizar para entrenar y aplicar modelos de procesamiento del lenguaje natural.

Stanza es una biblioteca muy popular y ampliamente utilizada en la comunidad de procesamiento del lenguaje natural debido a su precisión, rendimiento y soporte multilingüe. Puede ser utilizado para una variedad de aplicaciones, como análisis de texto, extracción de información, traducción automática, generación de resúmenes y más.

Conociendo un poco sobre lo que es Stanza veamos un sencillo ejemplo en código:

Antes de empezar necesitamos instalar Stanza:
```bash
pip install stanza
```

> ## Nota:
> El código completo está en: [2_etiquerado_stanza.py](scripts%2F2_etiquerado_stanza.py)

Ahora sí, empezamos con el código:
```python
import stanza
stanza.download('es')
```
Respuesta esperada:
```commandline
Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/master/resources_1.0.0.json: 115kB [00:00, 10.1MB/s]
2020-07-13 01:50:46 INFO: Downloading default packages for language: es (Spanish)...
Downloading http://nlp.stanford.edu/software/stanza/1.0.0/es/default.zip: 100%|██████████| 583M/583M [01:48<00:00, 5.36MB/s]
2020-07-13 01:52:48 INFO: Finished downloading models and saved to /root/stanza_resources.
```
El código anterior importa la biblioteca Stanza y descarga los modelos pre-entrenados para el procesamiento del lenguaje natural en español. Se utiliza la función stanza.download('es') para descargar los recursos necesarios.
```python
nlp = stanza.Pipeline('es', processors='tokenize,pos')
doc = nlp('yo soy una persona muy amable')
print(doc)
```
Respuesta esperada:
```commandline
2020-07-13 01:53:05 INFO: Loading these models for language: es (Spanish):
=======================
| Processor | Package |
-----------------------
| tokenize  | ancora  |
| pos       | ancora  |
=======================

2020-07-13 01:53:06 INFO: Use device: cpu
2020-07-13 01:53:06 INFO: Loading: tokenize
2020-07-13 01:53:06 INFO: Loading: pos
2020-07-13 01:53:07 INFO: Done loading processors!
```
El código anterior inicializa un objeto nlp que representa el pipeline de procesamiento de texto en Stanza. Se utiliza el modelo en español ('es') y se especifican los procesadores necesarios, en este caso, la tokenización y el etiquetado gramatical ('tokenize,pos'). Luego, se procesa el texto "yo soy una persona muy amable" utilizando el objeto nlp y se guarda el resultado en la variable doc. Por último, se imprime el contenido de doc

```python
for sentence in doc.sentences:
  for word in sentence.words:
    print(word.text, word.pos)
```
Respuesta esperada:
```commandline
yo PRON
soy AUX
una DET
persona NOUN
muy ADV
amable ADJ
```
El fragmento de código anterior muestra cómo acceder a las palabras etiquetadas gramaticalmente en el documento procesado por Stanza. Se itera sobre las oraciones (sentence) en doc.sentences y luego se itera sobre las palabras (word) en cada oración. Se imprime el texto de cada palabra (word.text) y su etiqueta gramatical (word.pos).

# 2 Modelos Markovianos Latentes (HMM)

## 2.1 Cadenas de Markov

En clases pasadas observamos que al utilizar nltk teníamos que descargar un par de cosas:

![1.png](imgs%2F2%2F1.png)

**PUNKT** es un módulo y una clase en NLTK que se utiliza para el tokenizado de texto. El tokenizado es el proceso de dividir un texto en unidades más pequeñas, como palabras o frases, llamadas "tokens". PUNKT es un tokenizador no supervisado que implementa un algoritmo para dividir el texto en oraciones y palabras. Utiliza modelos previamente entrenados para identificar los límites de las oraciones y las palabras en un texto en inglés.

**AVERAGED_PERCEPTRON**, por otro lado, es un algoritmo de aprendizaje automático utilizado en NLTK para el etiquetado gramatical o etiquetado POS (part-of-speech). El etiquetado gramatical es el proceso de asignar etiquetas de partes del discurso a cada palabra en un texto. AVERAGED_PERCEPTRON es un clasificador basado en perceptrón promediado, que se entrena utilizando un conjunto de características extraídas del texto para predecir las etiquetas gramaticales de las palabras.

Con base en ello podemos definir una escalera de modelos, donde los modelos markovianos latentes serán nuestra base para entender los siguientes.

![2.png](imgs%2F2%2F2.png)

HMM está basado en el concepto de cadenas de Markov

![3.png](imgs%2F2%2F3.png)

Las cadenas de Markov son un tipo de modelo estadístico que se utiliza para describir la evolución de un sistema en el tiempo, donde el sistema puede estar en diferentes estados discretos. Estos modelos se basan en la propiedad de Markov, que establece que la probabilidad de que el sistema pase a un estado futuro depende únicamente del estado actual y no de los estados anteriores.

Piensa en un sistema con un conjunto finito de estados (conjunto determinado de categorías que puedes contar, como el clima en un dia).

![4.png](imgs%2F2%2F4.png)

El diagrama mide la probabilidad de que a un dia frio le siga otro dia frio, o que de un dia frio le siga un dia caliente, etc.

Todas las flechas definen una transición, en conjunto definen todas las transiciones.

Una `cadena de Markov` define la `probabilidad de transición` entre los posibles estados que un sistema puede tener.

Haciendo la lógica para las posibles transiciones obtenemos la matriz de transición.

![5.png](imgs%2F2%2F5.png)

En el caso de nuestro ejemplo, consideraremos tres estados climáticos: "frío", "caliente" y "tibio". Una cadena de Markov para este sistema implicaría definir las probabilidades de transición entre estos estados. Por ejemplo, supongamos que tenemos las siguientes probabilidades de transición:

- Si hoy es "frío", la probabilidad de que mañana sea "frío" es del 40%, la probabilidad de que sea "caliente" es del 30% y la probabilidad de que sea "tibio" es del 30%.
- Si hoy es "caliente", la probabilidad de que mañana sea "frío" es del 20%, la probabilidad de que sea "caliente" es del 50% y la probabilidad de que sea "tibio" es del 30%.
- Si hoy es "tibio", la probabilidad de que mañana sea "frío" es del 50%, la probabilidad de que sea "caliente" es del 10% y la probabilidad de que sea "tibio" es del 40%.

|          | Frio | Caliente | Tibio |
|----------|------|----------|-------|
| Frio     | 0.4  | 0.3      | 0.3   |
| Caliente | 0.2  | 0.5      | 0.3   |
| Tibio    | 0.5  | 0.1      | 0.4   |


En esta matriz, cada fila de la matriz representa un estado inicial y cada columna representa un estado terminal. Los valores en la matriz representan las probabilidades de transición entre los estados. Por ejemplo, el valor en la fila "Frío" y la columna "Caliente" es 0.3, lo que significa que hay una probabilidad del 30% de pasar del estado "frío" al estado "caliente".

Otro componente importante es la distribución inicial de estados (letra PI).

![6.png](imgs%2F2%2F6.png)

PI tiene 3 componentes, que definen la probabilidad inicial de que un dia sea frio o caliente o tibio.

Lo que sucede es que nosotros multiplicamos nuestra matriz de transición por nuestro estado inicial PI(0), ese resultado nos dará el siguiente vector de estados PI(1). Eso nos indica la manera en que las probabilidades van cambiando a medida que el sistema evoluciona.


## 2.2 Modelos Markovianos latentes (HMM)

Las cadenas de markov son la base para entender como funcionan los etiquetadores de palabras.

En una cadena de markov tenemos dos ingredientes principales:

- `Matriz de transición` (cada elemento representa la probabilidad de transición al siguiente estado)

- `Distribución de estados`

![7.png](imgs%2F2%2F7.png)

Tomemos un ejemplo sencillo, queremos calcular cuál es la probabilidad de que el día de mañana haya cierta temperatura
tomando como base la temperatura del día de hoy. Como tal las variables de las cadenas de Markov DEBEN ser `discretas` esto
quiere decir que son clases no variables continuas o numéricas. Entonces en lugar de definir a la temperatura como una
variable continua vamos a proponer 3 clases: 1 Frío, 2 Caliente, 3 Tibio.

Primero vamos a definir un par de cosas, debemos haber muestreado el comportamiento del cambio de temperatura en un intervalo
por ejemplo 5 días:

{d1, d2, d3, d4, d5, d6} -> {f, f, t, c, c, c}

Excelente ahora ya tenemos una pequeña muestra del comportamiento de la temperatura a lo largo del tiempo, vamos a normalizar
la temperatura con su variable numérica en lugar de su letra representativa.

{d1, d2, d3, d4, d5, d6} -> {1, 1, 3, 2, 2, 2}

Ahora, nos podemos hacer una pregunta de probabilidad condicional, dado que yo sé que hoy la temperatura es Tibia (3) cuál
es la probabilidad de que el día de mañana la temperatura sea caliente (2)? Antes de continuar repasemos rápidamente los conceptos
de probabilidad condicional:

La probabilidad condicional es una medida de la probabilidad de que ocurra un evento A, dado que otro evento B ha ocurrido. Se representa como P(A|B), donde "|" denota "dado" o "sujeto a". En otras palabras, la probabilidad condicional se refiere a la probabilidad de que ocurra A, dado que ya sabemos que B ha ocurrido.

La fórmula general para la probabilidad condicional es:

P(A|B) = P(A ∩ B) / P(B)

Donde:

- `P(A|B)` es la probabilidad condicional de A dado B.
- `P(A ∩ B)` es la probabilidad de que ocurran tanto A como B (intersección de A y B).
- `P(B)` es la probabilidad de que ocurra B.

Teniendo todo esto en cuenta podemos empezar a obtener la probabilidad de que mañana sea caliente dado que hoy es tibio.

Primero planteamos la ecuación que queremos resolver:

P(2|3) = P(2 ∩ 3) / P(3)

Busquemos entonces primero a: P(2 ∩ 3), podemos notar que en nuestras observaciones de tiempo: {1, 1, 3, 2, 2, 2}
tenemos 5 transiciones (1 -> 1, 1 -> 3, 3 -> 2, 2 -> 2, 2 -> 2) (no contamos el día 6 porque ese no provoco una transición)
de nuestro total de observaciones vemos como hay 1 en la que se cumple la transición (3 -> 2) que es la que buscamos, de un total
de 5 transiciones, entonces:

P(2 ∩ 3) = 1/5

Ahora busquemos a P(3) este es más fácil es contar cuantos días han sido tibios del total de observaciones y vemos que a pesar
de tener 6 observaciones vamos a contar 5 porque son las que generan transición, entonces un día ha sido tibio de 5 días vistos:

P(3) = 1/5

Finalmente, entonces: P(2|3) = (1/5) / (1/5) = 1.

![8.png](imgs%2F2%2F8.png)

Ahora podemos repetir este pensamiento para las demas combinaciones de tal manera que hacemos nuestra matriz de transición:

![9.png](imgs%2F2%2F9.png)

Nota dado que nuestra busqueda fue cuál es la probabilidad de que mañana sea caliente dado que HOY es tibio, entonces la variable
tibio es la variable que representa una fila en la matriz de transición y la variable caliente representa una columna, esto
se puede denotar de la siguiente manera C32 (Probabilidad de mañana caliente dado hoy tibio).

Podemos seguir construyendo la matriz y hay algunas observaciones que son muy obvias por ejemplo cuál es la probabilidad de que
el día de mañana sea tibio dado que hoy es tibio C33 sería 0 porque en nuestras muestras nunca vimos ese caso entonces
P(3 ∩ 3) = 0 y, por tanto, P(3|3) también sería 0.

Ahora con toda la explicación anterior hemos construido la `matriz de transición` pero nos falta un elemento más nos falta
la `Distribución de estados`, que de forma sencilla puede ser explicada como las condiciones iniciales. Supongamos que 
para el día de hoy las probabilidades son 40% frío, 20% caliente y 40% tibio, a esto lo llamamos (PI)

Ahora solo falta multiplicar la matriz de transición por el vector de distribución de estados y tenemos la fórmula general de una cadena de Markov.

![10.png](imgs%2F2%2F10.png)

Los resultados de las probabilidades de clima de mañana son entonces:

![11.png](imgs%2F2%2F11.png)

20% frio, 60% caliente y 20% tibio.

Ahora que vimos las bases de una cadena de markov y sus ingredientes esenciales veamos de manera rápida com se expande esto a una la HMM o Modelo Markoviano Latente (Hidden Markov Model).
Pensemos ahora que queremos descubrir la secuencia de etiquetas de una palabra dada la secuencia de palabras:

![12.png](imgs%2F2%2F12.png)

Pensemos en secuencia de palabras y secuencia de etiquetas de cada una de esas palabras.

(pedro, es, ingeniero) | (sustantivo, Verbo, Sustantivo)

Esto es una cadena latente (oculta)y el propósito del modelo es descubrir o encontrar cuál es esa cadena.

## 2.3 Entrenando un HMM

> El objetivo de una cadena de markov latente es encontrar dada una secuencia de palabras cuál es la secuencia de etiquetas que le corresponde con mayor probabilidad
> 

La idea en un HMM es la siguiente.

![13.png](imgs%2F2%2F13.png)

Dado que ahora tenemos tenemos una cadena de markov sobre unos estados, que aunque son latentes son estados, sobre ellos podemos definir probabilidades de transición, los cuales están en el recuadro verde Cij los cuales serán la probabilidad de cambiar de una categoría gramatical a que en la siguiente palabra haya otra categoría gramatical.

Ahora las probabilidades serán, **dada una categoría gramatical cuál es la probabilidad de que esa categoría le corresponda a una cierta palabra** como en el ejemplo en el recuadro blanco.

El nodo 1 que es sustantivo puede corresponder a varias palabras, puede ser "pedro", puede ser "es", o en algunos casos no aplica, pero entonces ahí las probabilidades son cero, y eso se calcula con un corpus de palabras, 
**esas probabilidades que son dadas una categoría gramatical cuál es la probabilidad de que le corresponda una cierta palabra las 
llamamos probabilidades de emisión** y podemos calcular un conjunto de probabilidades de emisión para cada uno de los nodos 
del mapa de cadena markoviana y eso junto todo corresponde a un **modelo markoviano latente** de manera que ahora tenemos 3
ingredientes para un HMM:

- Matriz de transición
- Distribución inicial de estados
- Probabilidades de emisión.

![14.png](imgs%2F2%2F14.png)

Con estos tres ingredientes el objetivo de una cadena de markov latente es encontrar **dada una secuencia de palabras cuál es la secuencia de etiquetas que le corresponde por mayor probabilidad**.

Ejemplo:

Tenemos un modelo caracterizado por dos objetos, una matriz de transiciones (A) y otro elemento que contiene las probabilidades de emisión (B).

La entrada de un modelo serán observaciones (O), secuencias de palabras observadas (q).

Cuál es la probabilidad (P) de que dada una secuencia de palabras (w^n) cuál será la probabilidad de que una secuencia de 
etiquetas o tags (t^n) le sea asignada, nuestro objetivo es encontrar la cantidad maxima t~^n

![15.png](imgs%2F2%2F15.png)

Para calcular esta probabilidad usaremos la regla de Bayes.

La probabilidad de tener una etiqueta dada una palabra (por teorema de bayes) es igual a la probabilidad de tener una
palabra dada una etiqueta por la probabilidad de la etiqueta entre la probabilidad de la palabra.

![16.png](imgs%2F2%2F16.png)

Sin embargo, vamos a tomar un leve atajo matemático, y es que si nos damos cuenta estamos calculando multiples veces las
probabilidades de obtener una etiqueta dada una palabra para secuencias de texto y cada vez que lo hago siempre termino
dividiendo entre la probabilidad de aparición de dicha palabra P(w) sin embargo, esta probabilidad se obtiene para un corpus de texto
y no se modifica una vez calculada, entonces podemos quitar el denominador en todas las operaciones de tal manera de reducir
la cantidad de operaciones. Esto NO nos genera tal cual la probabildiad pero como la relación matemática se sigue manteniendo
es algo que podemos hacer.

![17.png](imgs%2F2%2F17.png)

En resumen:

![18.png](imgs%2F2%2F18.png)

El máximo de la probabilidad a la izquierda se traduce en el producto de las dos probabilidades de la derecha. Para ello ahora es necesario introducir dos hipótesis

La **hipótesis de independencia y la hipótesis markoviana**

- **En la hipótesis de independencia**, quiere decir que las probabilidades de palabras a etiquetas solamente dependen de la 
misma posición de la palabra en cuestión.
- **En la hipótesis Markoviana** el estado actual (etiqueta de la palabra) depende del estado inmediato anterior (la etiqueta de la que viene)

![19.png](imgs%2F2%2F19.png)

Con esto la fórmula que sigue un HMM es:

![20.png](imgs%2F2%2F20.png)

Entonces ¿Qué significa entrenar a un HMM? Pues básicamente se resume en aprender las probabilidades de (A) la matriz de transición
y las (B) probabilidades de emisión. Esto lo podemos hacer mediante la frecuencia de distribución de palabras y bi-gramas
cosas que ya hemos visto en el curso anterior de NLP.


## 2.4 Fases de entrenamiento de un HMM

Para estas próximas 2 clases vamos a utilizar una base de datos en español que contiene oraciones y cada palabra su respectivo POS
esto con el objetivo de poner en práctica todo lo que hemos visto sobre HMM. Para eso vamos a utilizar base de datos [AnCora](https://github.com/UniversalDependencies/UD_Spanish-AnCora)

> UD_Spanish-AnCora es un corpus anotado en el marco Universal Dependencies (UD) que se centra en el idioma español. El proyecto Universal Dependencies tiene como objetivo establecer una representación gramatical consistente y universalmente aplicable para diferentes idiomas. El corpus UD_Spanish-AnCora sigue estas directrices y proporciona un conjunto de anotaciones gramaticales coherentes para el análisis sintáctico del español.

> El corpus UD_Spanish-AnCora está compuesto por oraciones en español extraídas de diferentes fuentes, como periódicos, revistas y textos literarios. Cada oración del corpus se ha anotado manualmente con información gramatical y estructural, lo que incluye etiquetas de partes del discurso, relaciones sintácticas, dependencias entre palabras y otras características lingüísticas relevantes.

Es importante tomar en cuenta que AnCora es una base de datos en formato `Conllu`

> En cuanto a Conllu, es un formato de archivo utilizado para representar y almacenar datos anotados en el marco Universal Dependencies. Conllu significa "Universal Dependencies in CoNLL-U format". CoNLL-U se refiere a la Conferencia sobre Aprendizaje Computacional y Lenguaje Natural (Conference on Computational Natural Language Learning) y al formato Unversal (Universal), que fue adoptado por el proyecto Universal Dependencies.

> El formato Conllu es un formato de archivo de texto tabular que organiza las anotaciones gramaticales en columnas. Cada palabra de una oración se representa en una línea separada, y las diferentes propiedades de la palabra, como la forma, el lema, la etiqueta de parte del discurso y las dependencias sintácticas, se representan en diferentes columnas. Esto permite una representación estructurada y legible por máquina de las anotaciones gramaticales en el contexto del marco Universal Dependencies.

## Corpus de español: 

* AnCora | Github: https://github.com/UniversalDependencies/UD_Spanish-AnCora

* usamos el conllu parser para leer el corpus: https://pypi.org/project/conllu/

* Etiquetas Universal POS (Documentación): https://universaldependencies.org/u/pos/

Vamos a empezar instalando Conllu en Python

```bash
pip install conllu
```
Respuesta esperada:
```commandline
Collecting conllu
  Downloading conllu-4.5.3-py2.py3-none-any.whl (16 kB)
Installing collected packages: conllu
Successfully installed conllu-4.5.3
```

Ahora clonamos el repositorio de AnCora
```bash
git clone https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
```
Respuesta esperada:
```commandline
Clonando en 'UD_Spanish-AnCora'...
remote: Enumerating objects: 1262, done.
remote: Counting objects: 100% (278/278), done.
remote: Compressing objects: 100% (68/68), done.
remote: Total 1262 (delta 219), reused 269 (delta 210), pack-reused 984
Recibiendo objetos: 100% (1262/1262), 420.44 MiB | 7.22 MiB/s, listo.
Resolviendo deltas: 100% (912/912), listo.
```

> ## Nota:
> El código de esta sección lo puedes encontrar en: [3_fases_entrenamiento.py](scripts%2F3_fases_entrenamiento.py)

Ahora vamos a conocer brevemente a AnCora y el formato Conllu:

Importamos bibliotecas necesarias:
```python
from conllu import parse_incr
from pprint import pprint
```
Y ahora vamos a abrir el archivo es_ancora-ud-dev.conllu y observar como luce por dentro:

```python
data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
for tokenlist in parse_incr(data_file):
    print(tokenlist.serialize())
    break
```
Respuesta esperada:
```commandline
 global.Entity = eid-etype-head-other
# newdoc id = 3LB-CAST-111_C-2
# sent_id = 3LB-CAST-111_C-2-s1
# text = El gobernante, con ganada fama desde que llegó hace 16 meses al poder de explotar al máximo su oratoria y acusado por sus detractores de incontinencia verbal, enmudeció desde el momento en el que el Tribunal Supremo de Justicia (TSJ) decidió suspender temporalmente los comicios múltiples ante la imposibilidad "técnica" de celebrarlos el 28 de mayo.
1	El	el	DET	da0ms0	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	2	det	2:det	_
2	gobernante	gobernante	NOUN	ncms000	Gender=Masc|Number=Sing	32	nsubj	32:nsubj	SpaceAfter=No|ArgTem=arg1:tem
3	,	,	PUNCT	fc	PunctType=Comm	6	punct	6:punct	_
4	con	con	ADP	sps00	_	6	case	6:case	_
5	ganada	ganado	ADJ	aq0fsp	Gender=Fem|Number=Sing|VerbForm=Part	6	amod	6:amod	_
6	fama	fama	NOUN	ncfs000	Gender=Fem|Number=Sing	2	nmod	2:nmod	_
7	desde	desde	ADP	sps00	_	9	mark	9:mark	_
8	que	que	SCONJ	cs	_	9	mark	9:mark	_
...
60	celebrar	celebrar	VERB	vmn0000	VerbForm=Inf	55	acl	55:acl	_
61	los	él	PRON	_	Case=Acc|Gender=Masc|Number=Plur|Person=3|PrepCase=Npr|PronType=Prs	60	obj	60:obj	_
62	el	el	DET	da0ms0	Definite=Def|Gender=Masc|Number=Sing|PronType=Art	63	det	63:det	Entity=(NOCOREF:Spec.date-time-2-gstype:spec
63	28	28	NUM	_	NumForm=Digit|NumType=Card	60	obl	60:obl	MWE=28_de_mayo|MWEPOS=NOUN|ArgTem=argM:tmp
64	de	de	ADP	_	_	65	case	65:case	_
65	mayo	mayo	NOUN	_	_	63	compound	63:compound	SpaceAfter=No|Entity=NOCOREF:Spec.date)
66	.	.	PUNCT	fp	PunctType=Peri	32	punct	32:punct	_
```
Podemos observar como a una oración nos otorga cada palabra y mucha información de la misma y lo que nos importa a nosotros
nos devuelve su POS.

Veamos la información correspondiente a la palabra [1] de la oración 1 correspondiente a la "gobernante":

```python
pprint(tokenlist[1])
```
Respuesta esperada:
```commandline
{'deprel': 'nsubj',
 'deps': [('nsubj', 32)],
 'feats': {'Gender': 'Masc', 'Number': 'Sing'},
 'form': 'gobernante',
 'head': 32,
 'id': 2,
 'lemma': 'gobernante',
 'misc': {'ArgTem': 'arg1:tem', 'SpaceAfter': 'No'},
 'upos': 'NOUN',
 'xpos': 'ncms000'}
```
Podemos observar toda la información relacionada a la palabra `gobernante` y vemos como lo que nos importa está en las claves:
`form` (refiriendose a la palabra) y `upos` relacionado al POS.

Para acceder a esta información es tan fácil como:
```python
print(tokenlist[1]['form']+'|'+tokenlist[1]['upos'])
```
Respuesta esperada:
```commandline
gobernante|NOUN
```
De esta manera vamos a ser capaces de construir nuestro HMM contando frecuencias de tags dadas palabras, de tags, y de tags dados tags anteriores.


## 2.5 Entrenando un HMM en Python

> ## Nota: 
> El código de esta sección lo puedes encontrar en: [4_entrenando_hmm.py](scripts%2F4_entrenando_hmm.py)

Empezamos importando bibliotecas:

```python
from conllu import parse_incr
import numpy as np
# esta solo será para imprimir un diccionario de forma más estética
import itertools
```
Leemos los datos de entrada
```python
data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
```
Obtenemos los contadores necesarios para crear las matrices de emsion y transicion:

```python
def get_dicts(data):
    tagCountDict = {}
    emissionDict = {}
    transitionDict = {}

    for sentence in parse_incr(data):
        prev_tag = None
        for word in sentence:
            # Esto representa la etiqueta asociada a la palabra
            current_tag = word['upos']
            # Si la etiqueta no existe aun en el diccionario de etiquetas
            if current_tag not in tagCountDict:
                # Quiere decir que es la primera vez que la vemos, entonces le ponemos un contador de 1
                tagCountDict[current_tag] = 1
            else:
                # Si ya existía significa que entonces al menos tiene un 1 y puedo actualizar su valor con += 1
                tagCountDict[current_tag] += 1
```

Hasta aquí ya he construido mi diccionario que cuenta cuantas veces ha aparecido cada TAG

Ahora vamos a repetir el proceso pero para las probabilidades de emision

Recordemos que la emisión es: Dada una palabra que probabilidad tiene de que tenga cierto tag
            
```python
            current_word = word['form'].lower()

            # De forma simple es adjuntar la etiqueta de la palabra a la palabra
            emission = current_word + "|" + current_tag
            # Esto como en un ejemplo pasado podría devolver "gobernante|NOUN"

            # Esta técnica de contador es la misma que en el ejemplo anterior
            if emission not in emissionDict:
                emissionDict[emission] = 1
            else:
                emissionDict[emission] += 1
```

Con esto ya tenemos el diccionario que cuenta cuantas veces se ha asignado una etiqueta a una palabra.

Ahora vamos a continuar calculando la probabilidad de transición.

Recordemos que la probabilidad de transición es: Dada una etiqueta que probablilidad tiene de que tenga otra eitiqueta seguida.

Esto básicamente quiere decir, si yo soy NOUN cuál es la etiqueta más probable que pueda tener a continuación.

Específicamente para este diccionario se necesita comparar la etiqueta actual y la anterior, entonces debo pasar
al menos una iteración para que en la segunda vuelta ya exista mi primer etiqueta anterior


```python
            if prev_tag is None:
                # si la etiqueta anterior no ha sido asignada significa que estoy en la primera vuelta
                # por ende la etiqueta anterior será la actual
                prev_tag = current_tag
                # y utilizo continue para saltarme la siguiente parte de código y continuar con la iteración
                continue

            # si llegamos a esta parte es porque forzosamente estamos al menos en la segunda palabra, lo cual significa
            # que ya existe una prev_tag, por ende ya podemos hacer la transition.
            transition = current_tag + "|" + prev_tag

            # Misma técnica del contador en diccionario
            if transition not in transitionDict:
                transitionDict[transition] = 1
            else:
                transitionDict[transition] += 1
            # Como ya hemos terminado un ciclo ahora podemos decir que la etiqueta anterior fue la actual del inicio
            prev_tag = current_tag
```
Finalmente regresamos los contadores:
```python
    return tagCountDict, emissionDict, transitionDict
```

Continuando con nuestro código principal:

```python
    # Leemos los datos de entrada
    data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
    # Obtenemos los contadores necesarios
    tags, emissions, transitions = get_dicts(data_file)
    # Por fines estéticos solo imprimimos parte de los diccionarios.
    print(tags)
    print(dict(itertools.islice(emissions.items(), 5)))
    print(dict(itertools.islice(transitions.items(), 5)))
```
Respuesta esperada:
```commandline
Ejemplo de contador de TAGS: {'DET': 8057, 'NOUN': 9596, 'PUNCT': 6303, 'ADP': 8523, 'ADJ': 3541, 'SCONJ': 1136, 'VERB': 4563, 'NUM': 875, '_': 1262, 'CCONJ': 1454, 'PRON': 3154, 'PROPN': 4048, 'ADV': 1710, 'AUX': 1200, 'INTJ': 10, 'PART': 14, 'SYM': 36}
Ejemplo de contador de EMISIONES: {'el|DET': 2763, 'gobernante|NOUN': 2, ',|PUNCT': 2845, 'con|ADP': 443, 'ganada|ADJ': 1}
Ejemplo de contador de TRANSICIONES: {'NOUN|DET': 5683, 'PUNCT|NOUN': 2086, 'ADP|PUNCT': 636, 'ADJ|ADP': 106, 'NOUN|ADJ': 764}
```

Ahora como ya tenemos todos los contadores podemos utilizar las siguientes fórmulas para obtener las matrices correspondientes
de emisión y transición:

![21.png](imgs%2F2%2F21.png)

```python
def get_matrices(tagCountDict, emissionDict, transitionDict):
    transitionProbDict = {}  # matriz A
    emissionProbDict = {}  # matriz B

    # Recordemos que transition es TAG | TAG
    for key in transitionDict.keys():
        # Cada Key de mi transitionDict tiene 2 elementos que puedo separar por |
        tag, prevtag = key.split('|')
        if tagCountDict[prevtag] > 0:
            # esta es la formula que básicamente es la cantidad de aparición de Ambos TAG juntos entre la cantidad de aparición del TAG anterior
            transitionProbDict[key] = transitionDict[key] / (tagCountDict[prevtag])

    # La lógica en la matriz de emisión es muy similar
    for key in emissionDict.keys():
        word, tag = key.split('|')
        if emissionDict[key] > 0:
            # La fórmula es: la cantidad de apariciones de una palabra dado un tag entre las apariciones de dicho Tag
            emissionProbDict[key] = emissionDict[key] / tagCountDict[tag]
    # Regresamos las matrices
    return transitionProbDict, emissionProbDict
```

Volvemos a nuestro código principal y llamamos la función que acabamos de crear:

```python
    # Dado los contadores, podemos obtener las matrices de transición y emisión
    transition_matrix, emission_matrix = get_matrices(tags, emissions, transitions)
    print(dict(itertools.islice(transition_matrix.items(), 5)))
    print(dict(itertools.islice(emission_matrix.items(), 5)))
```
Respuesta esperada:
```commandline
Ejemplo de matriz de transicion: {'NOUN|DET': 0.7053493856274048, 'PUNCT|NOUN': 0.2173822426010838, 'ADP|PUNCT': 0.10090433127082342, 'ADJ|ADP': 0.012436935351402088, 'NOUN|ADJ': 0.21575826037842416}
Ejemplo de matriz de emision: {'el|DET': 0.34293161226262875, 'gobernante|NOUN': 0.00020842017507294707, ',|PUNCT': 0.45137236236712674, 'con|ADP': 0.051977003402557787, 'ganada|ADJ': 0.0002824060999717594}
```

Finalmente vamos a crear 2 funciones que nos ayuden a guardar y cargar estas matrices que acabamos de crear:

```python
def save_matrices(transitionProbDict, emissionProbDict):
    np.save('outputs/transitionHMM.npy', transitionProbDict)
    np.save('outputs/emissionHMM.npy', emissionProbDict)


def load_matrices(transition_file, emission_file):
    a = np.load(transition_file, allow_pickle='TRUE').item()
    b = np.load(emission_file, allow_pickle='TRUE').item()
    return a, b
```

Regresamos a nuestro código principal y podemos utilizar estas funciones para comprobar la persistencia de nuestro modelo:
```python
 # Utilizamos numpy para guardar las matrices en formato npu
    save_matrices(transition_matrix, emission_matrix)
    # Cargamos las matrices que habiamos guardado
    transition_load, emission_load = load_matrices('outputs/transitionHMM.npy', 'outputs/emissionHMM.npy')
    # Predecimos por ejemplo la probabilidad de la transición ADJ dado ADJ
    prob = transition_load['ADJ|ADJ']
    print(prob)
```
Respuesta esperada:
```commandline
0.030217452696978255
```

# 3 Algoritmo de Viterbi

## 3.1 El algoritmo de Viterbi

En la clase pasada entrenamos un modelo Markoviano latente calculando las probabilidades de transición y emisión a partir de un corpus de textos en español, la idea es ahora entender como podemos usar este modelo para hacer predicciones.

La pregunta es ¿qué tipo de predicciones hace un modelo markoviano latente HMM)?

![1.png](imgs%2F3%2F1.png)

Una vez entrenado, el proceso que denominaremos **Decodificación** consiste en que dada una secuencia de palabras podamos identificar la secuencia de etiquetas gramaticales mas probable que le corresponda, y esto se hace mediante el algoritmo de **Viterbi**.

Hay otras alternativas, pero primero enfoquemos en este.

La parte de entrenamiento que programamos consiste en encontrar la **matriz A** con sus **coeficientes C** y luego las **probabilidades Emisión** que son los **B** dados las probabilidades  condicionales (word | tag), luego viene el **algoritmo de Viterbi** que **se va a encargar de encontrar de entre un montón de  secuencias la secuencia mas probable** esto lo hace asignándole una probabilidad a cada secuencia que llamaremos **probabilidad de Viterbi**, luego dentro de ese espacio de probabilidades escogemos la mayor y esa seria la que el algoritmo va a retornar como la mas probable y por lo tanto la que debería ser las etiquetas correctas de la secuencia de palabras.

El algoritmo de Viterbi funciona de la siguiente manera.

![2.png](imgs%2F3%2F2.png)

Cada columna son todas las posibles etiquetas que una palabra va a tener, castillo es una persona no un edificio un sustantivo, cada circulo corresponde a una posible categoría gramatical, los círculos en gris tienen una probabilidad cero.

¿Cómo esto nos ayuda a entender el algoritmo de Viterbi? De la siguiente manera.

![3.png](imgs%2F3%2F3.png)

Considerando todas las posibles categorías gramaticales de cada palabra vamos creando caminos creando las etiquetas posteriores, cada camino es recorrer la primer etiqueta posible hasta la siguiente posible  y asi hasta llegar a la ultima palabra que contiene la secuencia. De todos esos caminos hay que calcular el mas probable, eso lo hacemos calculando un numero probabilistico que me diga que tan probable es que sea uno de esos caminos y escoger el mayor, eso lo hacemos de la siguiente manera.

![4.png](imgs%2F3%2F4.png)

El círculo de sustantivo propio está en color verde, significa que vamos a analizar lo que sucede en este nodo en particular, vamos a denotar con la letra griega NU, que parece una letra *V* paréntesis prop estilizada probabilidad de viterbi de que la categoría gramatical de castillo sea  "PROP" y eso es igual al "producto de la probabilidad inicial" este multiplicado por una "probabilidad condicional" de que ya que la categoría Inicial es PROP la palabra que este ubicada ahi es castillo, ese calculo lo hacemos para cada una de las celdas de esta columna, de esta manera calculamos la probabilidad de Viterbi para cada uno de los nodos de la primera columna.

![5.png](imgs%2F3%2F5.png)

Luego vamos a la segunda columna

![6.png](imgs%2F3%2F6.png)

El único nodo que tiene probabilidad no nula en la segunda columna es  el de categoría determinante **(DET)**, para calcular la probabilidad de este nodo lo que vamos a hacer es considerar todos los posibles caminos que pasan por ese nodo, vemos que son dos NOUN-DET Y PROP-DET, cada uno de los números tiene una probabilidad asignada que consiste en tomar la probabilidad del estado anterior (*V1*(PROP)), multiplicar por la probabilidad condicional de la etiqueta anterior y cual sera la etiqueta siguiente que en este caso sera de que dado prop la siguiente sea *DET* y esto multiplicado por una probabilidad de emisión, que dado que la categoría es **DET** cual sera la palabra el, y que tan probable es eso, aplicamos lo mismo a otra ruta con NOUN, y de esos dos números calculamos la probabilidad y tomamos esa  como la nueva probabilidad de Viterbi del nodo que esta en verde en este momento, y asi subsecuentemente para todos los nodos en la columna, el proceso estará completo cuando hayamos calculado las probabilidades de  cada uno de los elementos de esta matriz.

## 3.2 Cálculo de las probabilidades de Viterbi

En la clase anterior vimos como funciona el algoritmo de Viterbi, este **asigna probabilidades a cada elemento en una matriz que combina filas de categorías gramaticales y columnas de palabras en una secuencia**, la idea es que con este algoritmo que con estas probabilidades el puede determinar cual es la secuencia de etiquetas mas probable para esa secuencia que nosotros le estamos dando al modelo como entrada. Veamos ahora como en profundidad son estos cálculos de probabilidades y entender la matriz mencionada.

Retomamos el ejemplo anterior donde castillo es el apellido de una persona, donde los nodos contienen las etiquetas gramaticales, pero solo los azules tienen probabilidad **no nula**, de esta manera en la primer columna solo vamos a calcular 2 probabilidades que corresponden con los nodos que indican la categoría de sustantivo (noun) o sustantivo propio (prop).

![7.png](imgs%2F3%2F7.png)

Una vez calculadas las probabilidades de cada columna, podemos empezar a visualizar la matriz, donde cada palabra de la frase sea una columna, y las filas serán las etiquetas gramaticales.
> En la imagen siguiente solo se muestra las etiquetas de ADJ, PRON, DET, NOUN, PROP para simplificar, porque en la imagen anterior
> vimos que hay etiquetas que no aparecen en ninguna palabra (VERB, ADV, ADP)

![8.png](imgs%2F3%2F8.png)

El siguiente paso es asignar las probabilidades a la columna 2 primer elemento en sus dos caminos para llegar al nodo

![9.png](imgs%2F3%2F9.png)

La probabilidad **a** sería igual a Probabilidad de viterbi de la etiqueta por la probabilidad de transición, por la probabilidad de emisión.

![10.png](imgs%2F3%2F10.png)

Mientras que la probabilidad de Viterbi *V2* será la mayor entre a1 y a2, lo siguiente es aplicar el mismo concepto al nodo **PRON**, hecho esto asignamos los valores a la matriz

![11.png](imgs%2F3%2F11.png)

El algoritmo de viterbi termina cuando hemos calculado las probabilidades de todos los elementos de esta matriz.

En resumen el algoritmo de Viterbi mediante la búsqueda de posibles caminos de etiquetas calcula una probabilidad a cada elemento de una matriz donde esas probabilidades las llamamos probabilidades de Viterbi, el objetivo de encontrar la secuencia mas probable consiste en encontrar el camino cuyas probabilidades de Viterbi son mas grandes.

![12.png](imgs%2F3%2F12.png)

Una vez calculadas todas las probabilidades de viterbi de todas las etiquetas posibles la matriz nos queda de la siguiente manera.

![13.png](imgs%2F3%2F13.png)


![14.png](imgs%2F3%2F14.png)

## 3.3 Carga del modelo HMM y distribución inicial

> ## Nota:
> El código de esta sección lo puedes encontrar en: [5_carga_hmm.py](scripts%2F5_carga_hmm.py)

Para empezar, vamos a cargar nuestras matrices de transición y emisión pre-entrenadas

```python
import numpy as np

transitionProbdict = np.load('outputs/transitionHMM.npy', allow_pickle='TRUE').item()
emissionProbdict = np.load('outputs/emissionHMM.npy', allow_pickle='TRUE').item()
```

Vamos a necesitar conocer, y tener enumeradas las diferentes etiquetas (TAG) que puede tener cada palabra.
Empecemos creando un set de las etiquetas únicas

```python
stateSet = set([w.split('|')[1] for w in list(emissionProbdict.keys())])
print(stateSet)
```
Respuesta esperada:
```commandline
{'ADV', 'CCONJ', 'AUX', 'NOUN', '_', 'DET', 'SCONJ', 'PROPN', 'INTJ', 'PART', 'ADP', 'SYM', 'PUNCT', 'NUM', 'ADJ', 'VERB', 'PRON'}
```

Ahora vamos a asignarles un ID a cada TAG (esto nos servirá más adelante para identificar con mayor facilidad a los
tag como columnas de un frame)

```python
tagStateDict = {state: i for i, state in enumerate(stateSet)}
print(tagStateDict)
```
Respuesta esperada:
```commandline
{'ADV': 0, 'CCONJ': 1, 'AUX': 2, 'NOUN': 3, '_': 4, 'DET': 5, 'SCONJ': 6, 'PROPN': 7, 'INTJ': 8, 'PART': 9, 'ADP': 10, 'SYM': 11, 'PUNCT': 12, 'NUM': 13, 'ADJ': 14, 'VERB': 15, 'PRON': 16}
```

### Distribución inicial de estados latentes.

En este punto lo que nos interesa comprobar es, cuál es la probabilidad de que un TAG sea el inicial de una oración.
Para resolver esta tarea es relativamente sencillo. Con en dataset Ancora vamos a recorrer cada oración del mismo.
Apuntamos a la primera palabra y obtenemos su tag (UPOS) creamos un diccionario que cuente la frecuencia de a cada TAG
cuantas veces fue asignado como inicio de oración. Finalmente, como es una probabilidad, debemos dividir entre el
total de apariciones, que corresponde con el total de oraciones disponibles en AnCora.

```python
from conllu import parse_incr

data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-dev.conllu", "r", encoding="utf-8")
data = parse_incr(data_file)
len_sentences = 0
initTagStateProb = {}  # \rho_i^{(0)}
# primero creamos el contador
for token_list in data:
    len_sentences += 1
    tag = token_list[0]['upos']
    if tag in initTagStateProb.keys():
        initTagStateProb[tag] += 1
    else:
        initTagStateProb[tag] = 1
# Ahora noramlizamos dividiendo entre el total de oraciones
for key in initTagStateProb.keys():
    initTagStateProb[key] /= len_sentences

print(initTagStateProb)
```
Respuesta esperada:
```commandline
{'DET': 0.36275695284159615, 'PROPN': 0.1124546553808948, 'ADP': 0.15538089480048367, 'PRON': 0.06348246674727932, 'SCONJ': 0.02418379685610641, 'ADV': 0.056831922611850064, 'PUNCT': 0.08222490931076179, 'VERB': 0.02418379685610641, 'ADJ': 0.010882708585247884, 'CCONJ': 0.032648125755743655, 'NOUN': 0.02720677146311971, '_': 0.009068923821039904, 'INTJ': 0.0006045949214026602, 'AUX': 0.016324062877871828, 'NUM': 0.01995163240628779, 'PART': 0.0018137847642079807}
```
Finalmente, vamos a corroborar que la suma de probabilidades de cada etiqueta sea 1
```python
print(sum(initTagStateProb.values()))
```
Respuesta esperada:
```commandline
0.9999999999999999
```

Finalmente, guardamos las nuevas matrices:
```python
def save_matrices(initTagStateProb, tagStateDict):
    np.save('outputs/initTagStateProbHMM.npy', initTagStateProb)
    np.save('outputs/tagStateDictHMM.npy', tagStateDict)


save_matrices(initTagStateProb, tagStateDict)
```


## 3.4 Implementación de algoritmo de Viterbi en Python

En esta clase vamos a ver la implementación en código del algoritmo de Viterbi:

![1.png](imgs%2F4%2F1.png)

Aclaramos que esta implementación es meramente académica y NO tiene propósito alguno de ser llevada a un entorno de producción.
Es simplemente para entender conceptualmente al algoritmo de Viterbi.

> ## Nota:
> El código de esta sección lo puedes encontrar en: [6_viterbi_1.py](scripts%2F6_viterbi_1.py)

Importamos el módulo nltk y descargamos el tokenizador punkt para utilizar la función word_tokenize más adelante.

```python
import numpy as np
from nltk import word_tokenize
```

Creamos una función para cargar las matrices previamente generadas:
```python
def load_matrices(transition_file, emission_file, tag_state_file, init_tag_state_file):
    a = np.load(transition_file, allow_pickle='TRUE').item()
    b = np.load(emission_file, allow_pickle='TRUE').item()
    c = np.load(tag_state_file, allow_pickle='TRUE').item()
    d = np.load(init_tag_state_file, allow_pickle='TRUE').item()
    return a, b, c, d
```
Cargamos los datos necesarios:
```python
# Cargamos los datos necesarios
    a = 'outputs/transitionHMM.npy'
    b = 'outputs/emissionHMM.npy'
    c = 'outputs/tagStateDictHMM.npy'
    d = 'outputs/initTagStateProbHMM.npy'
    transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb = load_matrices(a, b, c, d)
```
Definimos la función ViterbiMatrix que implementa el algoritmo de Viterbi. Los parámetros transitionProbdict,
emissionProbdict, tagStateDict e initTagStateProb son diccionarios que contienen las probabilidades de
transición, las probabilidades de emisión, los estados de etiquetas y las probabilidades iniciales de los
estados de etiquetas, respectivamente.
```python
def viterbi_matrix(secuencia, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):
```

Tokenizamos la secuencia de entrada utilizando word_tokenize y almacenamos los tokens en la variable seq.

```python
def viterbi_matrix(secuencia, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):
    seq = word_tokenize(secuencia)
```
Creamos una matriz viterbiProb de dimensiones (17, len(seq)) para almacenar las probabilidades del algoritmo de
Viterbi. El número 17 representa las categorías posibles para las etiquetas (upos).
```python
def viterbi_matrix(secuencia, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):
    seq = word_tokenize(secuencia)
    viterbiProb = np.zeros((len(tagStateDict), len(seq)))  # upos tiene 17 categorias
```

Inicializamos la primera columna de la matriz viterbiProb. Para cada etiqueta en tagStateDict, calculamos la
etiqueta de la palabra (word_tag) y verificamos si existe una probabilidad de emisión para esa etiqueta de palabra
en emissionProbdict. Si existe, multiplicamos la probabilidad inicial de la etiqueta (initTagStateProb[key]) por
la probabilidad de emisión y la asignamos a viterbiProb[tag_row, 0].

```python
def viterbi_matrix(secuencia, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):
    seq = word_tokenize(secuencia)
    viterbiProb = np.zeros((len(tagStateDict), len(seq)))  # upos tiene 17 categorias
    for key in tagStateDict.keys():
        tag_row = tagStateDict[key]
        word_tag = seq[0].lower() + '|' + key
        if word_tag in emissionProbdict.keys():
            viterbiProb[tag_row, 0] = initTagStateProb[key] * emissionProbdict[word_tag]
```

Calculamos las siguientes columnas de la matriz viterbiProb. Para cada columna (col) a partir de la segunda
columna, y para cada etiqueta en tagStateDict, calculamos la etiqueta de la palabra (word_tag) y verificamos si
existe una probabilidad de emisión para esa etiqueta de palabra en emissionProbdict. Si existe, recorremos todas
las etiquetas en tagStateDict para obtener las probabilidades posibles en la columna anterior. Si hay una
probabilidad mayor a cero en la columna anterior (viterbiProb[tag_row2, col-1] > 0), multiplicamos esa
probabilidad por la probabilidad de transición y la probabilidad de emisión, y las agregamos a la lista
possible_probs. Finalmente, asignamos a viterbiProb[tag_row, col] el máximo valor de possible_probs.

```python
def viterbi_matrix(secuencia, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):
    seq = word_tokenize(secuencia)
    viterbiProb = np.zeros((len(tagStateDict), len(seq)))  # upos tiene 17 categorias
    for key in tagStateDict.keys():
        tag_row = tagStateDict[key]
        word_tag = seq[0].lower() + '|' + key
        if word_tag in emissionProbdict.keys():
            viterbiProb[tag_row, 0] = initTagStateProb[key] * emissionProbdict[word_tag]
    
    for col in range(1, len(seq)):
          for key in tagStateDict.keys():
              tag_row = tagStateDict[key]
              word_tag = seq[col].lower() + '|' + key
              if word_tag in emissionProbdict.keys():
                  # Miramos los estados de la columna anterior
                  possible_probs = []
                  for key2 in tagStateDict.keys():
                      tag_row2 = tagStateDict[key2]
                      tag_prevtag = key + '|' + key2
                      if tag_prevtag in transitionProbdict.keys():
                          if viterbiProb[tag_row2, col - 1] > 0:
                              possible_probs.append(
                                  viterbiProb[tag_row2, col - 1] * transitionProbdict[tag_prevtag] * emissionProbdict[
                                      word_tag])
                  viterbiProb[tag_row, col] = max(possible_probs)
    # Devolvemos la matriz viterbiProb.
    return viterbiProb
```

Usaremos de ejemplo la oración `el mundo es pequeño`:

```python
matrix = viterbi_matrix('el mundo es pequeño', transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb)
print(matrix)
```
Respuesta esperada:
```commandline
[[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 1.48925188e-10]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 6.86297327e-09 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 2.01168511e-04 3.97604151e-10 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.24400827e-01 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 6.76082166e-07 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 3.71508159e-05 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
```

## 3.5 Entrenamiento directo de HMM con NLTK

Para termina esta sección, lo único que necesitamos es convertir la matriz anterior en una lista que contenga el nombre 
de los TAGS asignados a cada palabra. Cada una de las columnas de la matriz representa una palabra, y cada una de las filas
representa un TAG diferente, entonces lo único que debemos hacer es obtener el `argmax` de cada columna y con base en ello
guardamos la palabra y su TAG.

> ## Nota:
> El código los puedes encontrar en: [7_viterbi_tags.py](scripts%2F7_viterbi_tags.py)

Básicamente lo único que vamos a hacer es agregar una nueva función al código que vimos la clase pasada.

```python
def get_viterbi_tags(seq, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb):

    viterbi_prob, seq = viterbi_matrix(seq, transitionProbdict, emissionProbdict,
                  tagStateDict, initTagStateProb)
    res = []
    # enumeramos cada palabra de la oración y recorremos todas ellas
    for i, p in enumerate(seq):
        # empezamos a recorrer todos los tags disponibles
        for tag in tagStateDict.keys():
            # si la probabilidad actual de este tag es ARGMAX de la columna,
            if tagStateDict[tag] == np.argmax(viterbi_prob[:, i]):
                # entonces en el resultado, adjunto la palabra y el tag
                res.append((p, tag))

    return res
```

De esta simple manera, ya tenemos nuestro código completo, y podemos hacer pruebas con él para obtener el UPOS de cada palabra
dentro de una oración:

```python
    msg = "el mundo es muy pequeño"
    response = get_viterbi_tags(msg, transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb)
    pprint(response)
    print()
    msg = "estos instrumentos han de rasgar"
    response = get_viterbi_tags(msg, transitionProbdict, emissionProbdict, tagStateDict, initTagStateProb)
    pprint(response)
```
Respuesta esperada:

```commandline
[('el', 'DET'),
 ('mundo', 'NOUN'),
 ('es', 'AUX'),
 ('muy', 'ADV'),
 ('pequeño', 'ADJ')]

[('estos', 'DET'),
 ('instrumentos', 'NOUN'),
 ('han', 'AUX'),
 ('de', 'ADP'),
 ('rasgar', 'VERB')]
```

Ahora, como es de esperarse este algoritmo tiene una implementación en NLTK más robusta y menos académica que si podríamos 
poner en un ambiente de producción. Vamos a conocer como podemos resolver este mismo problema utilizando NLTK.

> ## Nota:
> El código lo puedes encontrar en: [8_viterbi_nltk.py](scripts%2F8_viterbi_nltk.py)

Empecemos importando un par de bibliotecas, parse_incr que ya conocemos para leer archivos conllu y adicionalmente vamos a usar
train_test_split de sklearn para crear nuestros conjuntos de train y test y tambien vamos a importar `hmm` de nltk.

```python
from sklearn.model_selection import train_test_split
from conllu import parse_incr
from nltk.tag import hmm
```
Creamos nuestra lista de oraciones, en donde cada oración es a su vez una lista de palabras etiquetadas con su TAG:

```python
data_file = open("../data/UD_Spanish-AnCora/es_ancora-ud-train.conllu", "r", encoding="utf-8")
data_array = []
for tokenlist in parse_incr(data_file):
    tokenized_text = []
    for token in tokenlist:
        tokenized_text.append((token['form'], token['upos']))
    data_array.append(tokenized_text)

print(data_array[:10])
print(len(data_array))
```
Respuesta esperada:
```commandline
[[('Las', 'DET'), ('reservas', 'NOUN'), ('de', 'ADP'), ('oro', 'NOUN'), ('y', 'CCONJ'), ('divisas', 'NOUN'), ('de', 'ADP'), ('Rusia', 'PROPN'), ('subieron', 'VERB'), ('800', 'NUM'), ('millones', 'NOUN'), ('de', 'ADP'), ('dólares', 'NOUN'), ('y', 'CCONJ'), ('el', 'DET'), ('26', 'NUM'), ('de', 'ADP'), ('mayo', 'NOUN'), ('equivalían', 'VERB'), ('a', 'ADP'), ('19.100', 'NUM'), ('millones', 'NOUN'), ('de', 'ADP'), ('dólares', 'NOUN'), (',', 'PUNCT'), ('informó', 'VERB'), ('hoy', 'ADV'), ('un', 'DET'), ('comunicado', 'NOUN'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('Banco', 'PROPN'), ('Central', 'PROPN'), ('.', 'PUNCT')], [('Según', 'ADP'), ('el', 'DET'), ('informe', 'NOUN'), (',', 'PUNCT'), ('el', 'DET'), ('19', 'NUM'), ('de', 'ADP'), ('mayo', 'NOUN'), ('las', 'DET'), ('reservas', 'NOUN'), ('de', 'ADP'), ('oro', 'NOUN'), ('y', 'CCONJ'), ('divisas', 'NOUN'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('Banco', 'PROPN'), ('Central', 'PROPN'), ('eran', 'AUX'), ('de', 'ADP'), ('18.300', 'NUM'), ('millones', 'NOUN'), ('de', 'ADP'), ('dólares', 'NOUN'), ('.', 'PUNCT')], [('Los', 'DET'), ('activos', 'NOUN'), ('en', 'ADP'), ('divisas', 'NOUN'), ('en', 'ADP'), ('poder', 'NOUN'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('Banco', 'PROPN'), ('Central', 'PROPN'), ('y', 'CCONJ'), ('el', 'DET'), ('Ministerio', 'PROPN'), ('de', 'ADP'), ('Finanzas', 'PROPN'), ('se', 'PRON'), ('calculan', 'VERB'), ('en', 'ADP'), ('dólares', 'NOUN'), ('estadounidenses', 'ADJ'), ('y', 'CCONJ'), ('su', 'DET'), ('valor', 'NOUN'), ('depende', 'VERB'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('cambio', 'NOUN'), ('oficial', 'ADJ'), ('rublo-dólar', 'NOUN'), ('que', 'PRON'), ('establece', 'VERB'), ('el', 'DET'), ('Banco', 'PROPN'), ('Central', 'PROPN'), ('.', 'PUNCT')], [('Las', 'DET'), ('reservas', 'NOUN'), ('en', 'ADP'), ('oro', 'NOUN'), ('se', 'PRON'), ('valoran', 'VERB'), ('en', 'ADP'), ('base', 'NOUN'), ('a', 'ADP'), ('300', 'NUM'), ('dólares', 'NOUN'), ('estadounidenses', 'ADJ'), ('por', 'ADP'), ('cada', 'DET'), ('onza', 'NOUN'), ('troy', 'NOUN'), ('de', 'ADP'), ('oro', 'NOUN'), ('.', 'PUNCT')], [('El', 'DET'), ('presidente', 'NOUN'), ('de', 'ADP'), ('la', 'DET'), ('Generalitat', 'PROPN'), (',', 'PUNCT'), ('Jordi', 'PROPN'), ('Pujol', 'PROPN'), (',', 'PUNCT'), ('cree', 'VERB'), ('necesario', 'ADJ'), ('que', 'SCONJ'), ('el', 'DET'), ('Gobierno', 'PROPN'), ('agilice', 'VERB'), ('los', 'DET'), ('permisos', 'NOUN'), ('de', 'ADP'), ('residencia', 'NOUN'), ('a', 'ADP'), ('los', 'DET'), ('inmigrantes', 'NOUN'), ('para', 'ADP'), ('poder', 'AUX'), ('responder', 'VERB'), ('a', 'ADP'), ('la', 'DET'), ('falta', 'NOUN'), ('de', 'ADP'), ('mano', 'NOUN'), ('de', 'ADP'), ('obra', 'NOUN'), ('en', 'ADP'), ('Cataluña', 'PROPN'), ('que', 'PRON'), (',', 'PUNCT'), ('según', 'ADP'), ('cálculos', 'NOUN'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('Departamento', 'PROPN'), ('de', 'ADP'), ('Trabajo', 'PROPN'), (',', 'PUNCT'), ('impide', 'VERB'), ('cubrir', 'VERB'), ('unos', 'PRON'), ('23.000', 'NUM'), ('empleos', 'NOUN'), ('en', 'ADP'), ('diversos', 'DET'), ('sectores', 'NOUN'), ('.', 'PUNCT')], [('Pujol', 'PROPN'), ('urgió', 'VERB'), ('la', 'DET'), ('necesidad', 'NOUN'), ('de', 'ADP'), ('esta', 'DET'), ('medida', 'NOUN'), ('legal', 'ADJ'), ('durante', 'ADP'), ('una', 'DET'), ('cena', 'NOUN'), ('organizada', 'ADJ'), ('anoche', 'ADV'), ('por', 'ADP'), ('la', 'DET'), ('patronal', 'NOUN'), ('de', 'ADP'), ('la', 'DET'), ('pequeña', 'ADJ'), ('y', 'CCONJ'), ('mediana', 'ADJ'), ('empresa', 'NOUN'), (',', 'PUNCT'), ('Pimec-Sefes', 'PROPN'), (',', 'PUNCT'), ('en', 'ADP'), ('la', 'DET'), ('que', 'PRON'), ('también', 'ADV'), ('advirtió', 'VERB'), ('a', 'ADP'), ('los', 'DET'), ('empresarios', 'NOUN'), (',', 'PUNCT'), ('sindicatos', 'NOUN'), ('y', 'CCONJ'), ('administraciones', 'NOUN'), ('a', 'ADP'), ('que', 'SCONJ'), ('cumplan', 'VERB'), ('con', 'ADP'), ('la', 'DET'), ('normativa', 'NOUN'), ('laboral', 'ADJ'), ('en', 'ADP'), ('materia', 'NOUN'), ('de', 'ADP'), ('prevención', 'NOUN'), ('de', 'ADP'), ('la', 'DET'), ('siniestralidad', 'NOUN'), ('.', 'PUNCT')], [('Según', 'ADP'), ('Pujol', 'PROPN'), (',', 'PUNCT'), ('"', 'PUNCT'), ('no', 'ADV'), ('puede', 'AUX'), ('ser', 'AUX'), ('que', 'SCONJ'), ('se', 'PRON'), ('tarde', 'VERB'), ('ocho', 'NUM'), ('meses', 'NOUN'), ('para', 'ADP'), ('dar', 'VERB'), ('un', 'DET'), ('permiso', 'NOUN'), ('de', 'ADP'), ('residencia', 'NOUN'), ('de', 'ADP'), ('trabajo', 'NOUN'), ('a', 'ADP'), ('un', 'DET'), ('ciudadano', 'NOUN'), ('polaco', 'ADJ'), ('"', 'PUNCT'), (',', 'PUNCT'), ('en', 'ADP'), ('un', 'DET'), ('momento', 'NOUN'), ('de', 'ADP'), ('falta', 'NOUN'), ('de', 'ADP'), ('mano', 'NOUN'), ('de', 'ADP'), ('obra', 'NOUN'), ('que', 'PRON'), ('actúa', 'VERB'), (',', 'PUNCT'), ('en', 'ADP'), ('su', 'DET'), ('opinión', 'NOUN'), (',', 'PUNCT'), ('como', 'SCONJ'), ('freno', 'NOUN'), ('al', '_'), ('a', 'ADP'), ('el', 'DET'), ('crecimiento', 'NOUN'), ('económico', 'ADJ'), ('.', 'PUNCT')], [('Esta', 'DET'), ('opinión', 'NOUN'), ('fue', 'AUX'), ('corroborada', 'VERB'), ('por', 'ADP'), ('el', 'DET'), ('presidente', 'NOUN'), ('de', 'ADP'), ('Pimec-Sefes', 'PROPN'), (',', 'PUNCT'), ('Josep', 'PROPN'), ('González', 'PROPN'), (',', 'PUNCT'), ('quien', 'PRON'), ('subrayó', 'VERB'), ('que', 'SCONJ'), ('esta', 'DET'), ('falta', 'NOUN'), ('de', 'ADP'), ('mano', 'NOUN'), ('de', 'ADP'), ('obra', 'NOUN'), (',', 'PUNCT'), ('tanto', 'ADV'), ('cualificada', 'ADJ'), ('como', 'SCONJ'), ('no', 'ADV'), (',', 'PUNCT'), ('comenzó', 'VERB'), ('en', 'ADP'), ('las', 'DET'), ('comarcas', 'NOUN'), ('de', 'ADP'), ('Lérida', 'PROPN'), ('pero', 'CCONJ'), ('ya', 'ADV'), ('ha', 'AUX'), ('llegado', 'VERB'), ('a', 'ADP'), ('todas', 'DET'), ('las', 'DET'), ('provincias', 'NOUN'), ('catalanas', 'ADJ'), ('.', 'PUNCT')], [('Por', 'ADP'), ('ello', 'PRON'), (',', 'PUNCT'), ('González', 'PROPN'), ('pidió', 'VERB'), ('una', 'DET'), ('reforma', 'NOUN'), ('"', 'PUNCT'), ('urgente', 'ADJ'), ('"', 'PUNCT'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('sistema', 'NOUN'), ('de', 'ADP'), ('contratación', 'NOUN'), ('de', 'ADP'), ('extranjeros', 'NOUN'), ('que', 'PRON'), ('permita', 'VERB'), ('un', 'DET'), ('sistema', 'NOUN'), ('"', 'PUNCT'), ('ágil', 'ADJ'), ('y', 'CCONJ'), ('rápido', 'ADJ'), ('"', 'PUNCT'), ('para', 'ADP'), ('insertar', 'VERB'), ('a', 'ADP'), ('los', 'DET'), ('inmigrantes', 'NOUN'), ('en', 'ADP'), ('el', 'DET'), ('mercado', 'NOUN'), ('laboral', 'ADJ'), ('.', 'PUNCT')], [('Ante', 'ADP'), ('unas', 'DET'), ('mil', 'NUM'), ('personas', 'NOUN'), (',', 'PUNCT'), ('entre', 'ADP'), ('ellas', 'PRON'), ('la', 'DET'), ('ministra', 'NOUN'), ('de', 'ADP'), ('Ciencia', 'PROPN'), ('y', 'CCONJ'), ('Tecnología', 'PROPN'), (',', 'PUNCT'), ('Anna', 'PROPN'), ('Birulés', 'PROPN'), (',', 'PUNCT'), ('el', 'DET'), ('alcalde', 'NOUN'), ('de', 'ADP'), ('Barcelona', 'PROPN'), (',', 'PUNCT'), ('Joan', 'PROPN'), ('Clos', 'PROPN'), (',', 'PUNCT'), ('la', 'DET'), ('Delegada', 'PROPN'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('Gobierno', 'PROPN'), (',', 'PUNCT'), ('Julia', 'PROPN'), ('García', 'PROPN'), ('Valdecasas', 'PROPN'), (',', 'PUNCT'), ('y', 'CCONJ'), ('una', 'DET'), ('nutrida', 'ADJ'), ('representación', 'NOUN'), ('del', '_'), ('de', 'ADP'), ('el', 'DET'), ('gobierno', 'NOUN'), ('catalán', 'ADJ'), (',', 'PUNCT'), ('Pujol', 'PROPN'), ('dio', 'VERB'), ('un', 'DET'), ('toque', 'NOUN'), ('de', 'ADP'), ('alerta', 'NOUN'), ('sobre', 'ADP'), ('el', 'DET'), ('aumento', 'NOUN'), ('de', 'ADP'), ('los', 'DET'), ('accidentes', 'NOUN'), ('laborales', 'ADJ'), ('.', 'PUNCT')]]
14287
```
Tenemos más de 14 mil oraciones, cada una con una cantidad de palabras etiquetadas
Ahora creamos 2 particiones para entrenar y probar a nuestro HMM:

```python
train_data, test_data = train_test_split(data_array, test_size=0.2, random_state=42)
print(len(train_data))
print(len(test_data))
```
Respuesta esperada:
```python
11429
2858
```

Ahora instanciamos el modelo y empezamos a entrenarlo en el conjunto de entrenamiento:
```python
tagger = hmm.HiddenMarkovModelTrainer().train_supervised(train_data)
print(tagger)
```
Respuesta esperada:
```commandline
<HiddenMarkovModelTagger 18 states and 34366 output symbols>
```
Finalmente, podemos poner a prueba el modelo en ambos conjuntos y conocer su precisión:
```python
train_acc = tagger.evaluate(train_data)
print("Accuracy over Train's partition:", train_acc)


test_acc = tagger.evaluate(test_data)
print("Accuracy over Test's partition:", test_acc)
```
Respuesta esperada:
```commandline
Accuracy over Train's partition: 0.9795440204266921
Accuracy over Test's partition: 0.5277194254577251
```
Es importante tener en cuenta que el modelo bajo datos no vistos obtuvo más del 50% de accuracy para un problema de más de 15 clases.

# 4 Modelos Markovianos de máxima entropía (MEMM)

## 4.1 Modelos Markovianos de máxima entropia (MEMM)

En este punto tenemos los conocimientos suficientes para entender diversos algoritmos de clasificación, específicamente de etiquetado  que tienen cierta afinidad o similitud con los modelos  markovianos latentes, la idea de esta sección es que el profesor propone un reto en el cual tendrás que usar todo lo aprendido hasta este momento pero desarrollando codigo basado en un nuevo modelo propuesto en esta clase, derivado de los HMM.

![2.png](imgs%2F4%2F2.png)

En un modelo Markoviano latente ya conocemos la fórmula, la secuencia más probable de etiquetas se calcula como aquella donde dada una secuencia de palabras cual es la secuencia de etiquetas mas probable, usábamos bayes para convertir probabilidad condicional en el producto de dos probabilidades que llamábamos probabilidades de transición y  probabilidades de emisión representados con flechas rojas en HMM

![3.png](imgs%2F4%2F3.png)

El nuevo modelo hace una diferencia, las etiquetas siguen teniendo flechas que van dirigidas desde la etiqueta anterior a la posterior, pero las flechas que conectan palabras y etiquetas van hacia arriba (de la palabra a la etiqueta).

![4.png](imgs%2F4%2F4.png)

En resumen un Modelo Markoviano de Maxima Entropia se define como idéntico a HMM, pero no usamos la regla de bayes para convertir esto en probabilidades de transición y emisión, sino que la probabilidad inicial es la única que vamos a  considerar directamente y vamos a crear nuevas dependencias.

En la formula inferior estamos calculando la probabilidad de que dada una palabra en la posición i, y una etiqueta en la posición i-1, cual sera la etiqueta en la posición i, sin embargo esta probabilidad no se descompone en transición - emisión como en el HMM, esta es una sutil pero gran diferencia, el performance del modelo sera distinto.

En el siguiente diagrama vemos la forma en que funciona un Modelo Markoviano de Maxima Entropia con dependencias tan arbitrarias como se desee.

![5.png](imgs%2F4%2F5.png)

En un MEMM estamos calculando la probabilidad de que dada una palabra le corresponda una cierta etiqueta (probabilidades posteriores en bayes) y esto lo descomponíamos en la probabilidad de transición (dada una etiqueta, cual sera la que le corresponde en la siguiente posición) y la probabilidad de emisión (dada una etiqueta cual sera la palabra que el corresponda), aquí debemos pensar de forma distinta con probabilidades posteriores, la probabilidad seria la siguiente:

>Dado un contexto al rededor de una etiqueta particular que yo quiero calcular o determinar cuál será la probabilidad de que esa categoría sea un verbo/sustantivo/etc.

Esto quiere decir que tomaremos información de las etiquetas anterior y posterior con las palabras que están antes, y después, con esto caemos en el concepto de las redes neuronales donde tenemos muchas entradas y la red debe inferir una salida, haciendo uso de modelos de clasificación.

La fórmula nos indica que la secuencia de palabras a la cual le vamos a asignar una secuencia de etiquetas donde la probabilidad dea maxima esta dado solo por esa probabilidad, donde dada una palabra actual, palabra posterior y palabra anterior, etiqueta posterior, y etiqueta anterior quiero saber la etiqueta actual, lo cual es un contexto completo.

#### Comparacion entre modelos

CASO HMM

![6.png](imgs%2F4%2F6.png)

Caso MEMM

![7.png](imgs%2F4%2F7.png)

Cuál es la probabilidad de que dada una palabra, y una etiqueta anterior, cuál es la probabilidad de que corresponda una etiqueta en la posición actual, deberá ser un conteo donde observo la palabra y las dos etiquetas consecutivas,  dividido entre el numero de veces que veo la palabra  junto con la etiqueta en la posición inmediatamente anterior.

Esa es la forma en que deberíamos calcular matemáticamente las probabilidades para el MEMM


## 4.2 Algoritmo de Viterbi para MEMM

En este slide vimos como construir un MEMM

![5.png](imgs%2F4%2F5.png)

Donde yo puedo predecir la categoría a la que pertenece cierta palabra considerando todo el contexto que rodea a esa categoría en términos de las categorías y palabras que se encuentran a los costados,y la palabra que corresponde a esa categoría.

Aquí el algoritmo de Viterbi se calcula con una pequeña modificación.

![8.png](imgs%2F4%2F8.png)

Teniendo en cuenta que solo calculamos probabilidades posteriores, dado un contexto de palabras y etiquetas cuál  es la probabilidad de que le corresponda una cierta etiqueta. Ya no utilizamos probabilidades de transición ni emisión solo una probabilidad posterior, y el algoritmo de Viterbi tiene que adaptarse a esa filosofía, la formula debajo del slide indica como seria: "la probabilidad de Viterbi para la columna *t*  para una categoría *j* es igual al máximo de todas las posibles probabilidades donde cada una de esas probabilidades es el producto de la probabilidad de Viterbi de la columna anterior *t-1* de una categoría *i* multiplicado por la probabilidad posterior, que dado ese contexto cual debe ser la categoría *j* que debería corresponder  a la palabra.

Veamos en el tablero las diferencias de HMM y MEMM

![9.png](imgs%2F4%2F9.png)

Si observas todo es muy similar al algoritmo y código anterior, donde eliminamos la probabilidad de transición y en lugar de la probabilidad de emisión calculamos la **probabilidad posterior de dado un contexto nos de la categoría**

## 4.3 Reto: construye un MEMM en Python

# 5 Clasificación de texto con NLTK

## 5.1 El problema general de la clasificación de texto

En general el ciclo básico de clasificación de texto, es poder convertir estas palabras en atributos, a través de un
extractor de características y con estas entrenar a un algoritmo de ML. Una vez se tiene entrenado al extractor de características
y al algoritmo de ML se puede predecir un nuevo documento.
![10.png](imgs%2F4%2F10.png)

A continuación en listo a mayor profundidad los pasos que componen un problema básico de clasificación de texto:

1. **Recopilación y preprocesamiento de datos:** El primer paso es recopilar un conjunto de datos etiquetados que contengan ejemplos de texto clasificados en categorías específicas. Estos datos pueden ser obtenidos de diversas fuentes, como bases de datos, redes sociales, páginas web, etc. Una vez que se tiene el conjunto de datos, se realiza un preprocesamiento para limpiar y normalizar el texto, lo cual puede incluir la eliminación de caracteres especiales, convertir el texto a minúsculas, eliminar palabras irrelevantes (stop words) y aplicar técnicas de lematización o stemming.

2. **División de datos en conjuntos de entrenamiento y prueba:** El conjunto de datos se divide en dos partes: uno para entrenamiento y otro para evaluar el rendimiento del modelo. Esto se hace para evitar el sobreajuste (overfitting) y tener una evaluación imparcial del modelo. Por lo general, se reserva un porcentaje (por ejemplo, 70-80%) de los datos para el entrenamiento y el resto se utiliza para la prueba.

3. **Extracción de características:** Antes de entrenar un modelo de clasificación, es necesario representar los textos como características numéricas que puedan ser utilizadas por el algoritmo de aprendizaje. Esto implica la extracción de características relevantes del texto, como la frecuencia de las palabras (TF-IDF), n-gramas, características semánticas, etc. Esta etapa convierte el texto en un formato que el modelo pueda entender.

4. **Elección y entrenamiento del modelo:** A continuación, se selecciona un modelo de clasificación adecuado para el problema de texto. Algunos de los modelos comúnmente utilizados son Naive Bayes, Support Vector Machines (SVM), Árboles de decisión, Redes Neuronales, entre otros. El modelo se entrena utilizando el conjunto de entrenamiento y las características extraídas. Durante el entrenamiento, el modelo aprende los patrones y características que ayudan a distinguir las diferentes clases de texto.

5. **Ajuste de hiperparámetros:** Muchos modelos de clasificación tienen hiperparámetros que deben ser ajustados para obtener un mejor rendimiento. Estos hiperparámetros controlan aspectos como la complejidad del modelo, la regularización, el tamaño de la ventana de contexto, etc. Es importante ajustar estos hiperparámetros de manera adecuada utilizando técnicas como la validación cruzada para evitar el sobreajuste y mejorar la generalización del modelo.

6. **Evaluación del modelo:** Una vez que el modelo ha sido entrenado y ajustado, se evalúa utilizando el conjunto de prueba que se reservó anteriormente. Se calculan métricas de evaluación, como la precisión, el recall, la puntuación F1, la matriz de confusión, entre otras, para medir el rendimiento del modelo en la clasificación de texto. Estas métricas ayudan a comprender la efectividad del modelo y su capacidad para generalizar a nuevos datos.

7. **Predicción de nuevas muestras:** Después de evaluar el modelo, se puede utilizar para realizar predicciones en nuevos textos sin etiquetar. El modelo aplicará las características aprendidas durante el entrenamiento para asignar una clase o categoría a cada texto nuevo basándose en su contenido.

Cuando hablamos de técnicas de clasificación podemos pensar en 3 principales vertientes:

1. **Basadas en teoría de la probabilidad:** Estas técnicas se basan en la teoría de la probabilidad y asumen que los textos pertenecen a diferentes clases con ciertas probabilidades. Un ejemplo común es el clasificador Naive Bayes, que utiliza el teorema de Bayes para calcular la probabilidad de que un documento pertenezca a una determinada clase dada su característica. El clasificador Naive Bayes asume independencia condicional entre las características (palabras) del documento, lo que hace que el cálculo de la probabilidad sea más eficiente. Esta técnica es fácil de implementar y suele funcionar bien en conjuntos de datos de texto de tamaño moderado.

2. **Basadas en teoría de la Información:** Estas técnicas se basan en la teoría de la información y utilizan medidas como la entropía y la ganancia de información para tomar decisiones de clasificación. Un ejemplo común es el algoritmo de árbol de decisión, que construye un árbol en el que cada nodo representa una característica y cada rama representa una posible clasificación basada en esa característica. El árbol de decisión elige la característica que proporciona la mayor ganancia de información para dividir los datos y realizar la clasificación. Esta técnica es interpretable y puede manejar tanto características categóricas como numéricas, pero puede ser propensa al sobreajuste si el árbol se construye de manera demasiado compleja.

3. **Basadas en espacios Vectoriales:** Estas técnicas representan los textos y las categorías como vectores en un espacio multidimensional. La representación más común es la frecuencia de términos (TF-IDF), que mide la importancia relativa de una palabra en un documento. Con estas técnicas, los documentos se representan como vectores TF-IDF y luego se utilizan algoritmos de aprendizaje automático, como Support Vector Machines (SVM) o clasificadores lineales, para encontrar fronteras de decisión que separen las diferentes categorías. Estas técnicas pueden manejar grandes conjuntos de datos y funcionar bien en problemas de clasificación de texto de alta dimensionalidad.

Cuando hablamos de palabras, podemos clasificar cada una de ellas en, por ejemplo:

- Asignar categorías. 
- Clasificarlas por genero.
- Etiquetas gramaticales

Y cuando hablamos de documentos:

- Análisis de sentimientos.
- Respecto al tópico, o tema de conversación.
- Priorización (spam no spam).

## 5.2 Tareas de clasificación con NLTK



## 5.3 Modelos de clasificación en Python: nombres

## 5.4 Modelos de clasificación en Python: documentos

# 6 Implementación de un modelo de clasificación de texto

## 6.1 Naive Bayes

## 6.2 Naive Bayes en Python: preparación de los datos

## 6.3 Naive Bayes en Python: construcción del modelo

## 6.4 Naive Bayes en Python: ejecución del modelo

## 6.5 Métricas para algoritmos de clasificación

## 6.6 Reto final: construye un modelo de sentimientos

