# Curso de Redes Neuronales con PyTorch

Trabaja con modelos de deep learning usando PyTorch. Suma esta herramienta a tus habilidades en machine learning y desarrolla los modelos de inteligencia artificial que están definiendo el futuro.

- Entrena y evalúa modelos de redes neuronales con PyTorch.
- Desarrolla modelos para clasificación de texto.
- Genera inferencia con el mejor modelo y almacénalo.
- Crea la estructura de una red neuronal con sus capas.


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
> 
> Este Curso es el Número 11 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso
> 

# ÍNDICE:

- [1 Fundamentos de PyTorch](#1-fundamentos-de-pytorch)
  - [1.1 ¿Qué necesitas para aprender PyTorch?](#11-qué-necesitas-para-aprender-pytorch)
  - [1.2 ¿Por qué usar PyTorch?](#12-por-qué-usar-pytorch)
  - [1.3 Hola mundo en PyTorch](#13-hola-mundo-en-pytorch)
  - [1.4 Creación de Tensores en PyTorch](#14-creación-de-tensores-en-pytorch)
  - [1.5 Debugging de operaciones con tensores](#15-debugging-de-operaciones-con-tensores)
  - [1.6 Conversión y operación de tensores con PyTorch](#16-conversión-y-operación-de-tensores-con-pytorch)
- [2 Estructura de modelo de deep learning en PyTorch](#2-estructura-de-modelo-de-deep-learning-en-pytorch)
  - [2.1 Generación y split de datos para entrenamiento del modelo](#21-generación-y-split-de-datos-para-entrenamiento-del-modelo)
  - [2.2 Estructura de modelo en PyTorch con torch.nn](#22-estructura-de-modelo-en-pytorch-con-torchnn)
  - [2.3 Entrenamiento, funciones de pérdida y optimizadores](#23-entrenamiento-funciones-de-pérdida-y-optimizadores)
  - [2.4 Entrenamiento y visualización de pérdida](#24-entrenamiento-y-visualización-de-pérdida)
  - [2.5 Predicción con un modelo de PyTorch entrenado](#25-predicción-con-un-modelo-de-pytorch-entrenado)
- [3 Redes neuronales con PyTorch](#3-redes-neuronales-con-pytorch)
  - [3.1 Datos para clasificación de texto](#31-datos-para-clasificación-de-texto)
  - [3.2 Procesamiento de datos: tokenización y creación de vocabulario](#32-procesamiento-de-datos-tokenización-y-creación-de-vocabulario)
  - [3.3 Procesamiento de datos: preparación de DataLoader()](#33-procesamiento-de-datos-preparación-de-dataloader)
  - [3.4 Creación de modelo de clasificación de texto con PyTorch](#34-creación-de-modelo-de-clasificación-de-texto-con-pytorch)
  - [3.5 Función para entrenamiento](#35-función-para-entrenamiento)
  - [3.6 Función para evaluación](#36-función-para-evaluación)
  - [3.7 Split de datos, pérdida y optimización](#37-split-de-datos-pérdida-y-optimización)
  - [3.8 Entrenamiento y evaluación de modelo de clasificación de texto](#38-entrenamiento-y-evaluación-de-modelo-de-clasificación-de-texto)
  - [3.9 Inferencia utilizando torch.compile(): el presente con PyTorch 2.X](#39-inferencia-utilizando-torchcompile-el-presente-con-pytorch-2x)
  - [3.10 Almacenamiento del modelo con torch.save() y state_dict()](#310-almacenamiento-del-modelo-con-torchsave-y-statedict)
  - [3.11 Sube tu modelo de PyTorch a Hugging Face](#311-sube-tu-modelo-de-pytorch-a-hugging-face)
  - [3.12 Carga de modelo de PyTorch con torch.load()](#312-carga-de-modelo-de-pytorch-con-torchload)
- [4 Cierre del curso](#4-cierre-del-curso)

# 1 Fundamentos de PyTorch

## 1.1 ¿Qué necesitas para aprender PyTorch?

Para este curso es fundamental tener conocimientos previos de python, entre los que podemos destacar:

- Programación Orientada Objetos
- NumPy y MatPlotLib
- Fundamentos de DeepLearning
- Conocimiento de Hugging Face

A lo largo del curso estaremos llevando a cabo un proyecto de clasificación de texto utilizando TorchText. PyTorch proporciona
diferentes bibliotecas para trabajar con estructuras de datos muy concretas como lo pueden ser: Audio, Texto o Imagen.

En este específico proyecto vamos a utilizar [dbpedia](https://pytorch.org/text/0.10.0/_modules/torchtext/datasets/dbpedia.html) un dataset de artículos en Inglés.
El objetivo de nuestro modelo de clasificación de texto será asignar una clase a cada artículo disponible. Estas clases serán
exploradas más a fondo a lo largo del curso, pero se trata de 14 clases diferentes que describen de qué trata el artículo.

## 1.2 ¿Por qué usar PyTorch?

En general `PyTorch` es un `framework` de `deep learning` desarrollado por `META` anteriormente `FaceBook` que fue liberado en 2016.
En la actualidad `PyTorch` es el `framework` de desarrollo de `deep learning` más utilizado por la comunidad científica. Algunas de las
ventajas principales que ofrece `PyTorch` sobre otros marcos de desarrollo son:

- `Sintaxis intuitiva:` PyTorch ofrece una sintaxis intuitiva y fácil de usar que permite a los desarrolladores escribir código limpio y legible. Esto facilita el proceso de creación y depuración de modelos de aprendizaje profundo.

- `Flexibilidad y dinamismo:` A diferencia de algunos otros marcos de trabajo de aprendizaje profundo, PyTorch es dinámico en su naturaleza, lo que significa que los gráficos computacionales se construyen y modifican durante la ejecución. Esto proporciona una mayor flexibilidad al modelar y ajustar los componentes del modelo.

- `Interactividad:` PyTorch es compatible con el modo interactivo, lo que permite a los desarrolladores experimentar y depurar su código línea por línea. Esto resulta especialmente útil durante la fase de desarrollo y pruebas, ya que facilita la comprensión del comportamiento del modelo en tiempo real.

- `Amplia comunidad y documentación:` PyTorch cuenta con una comunidad activa y una amplia base de usuarios. Esto significa que hay una abundancia de recursos disponibles, como tutoriales, documentación y ejemplos de código, que facilitan el aprendizaje y el desarrollo de proyectos.

- `Compatibilidad con GPU:` PyTorch está optimizado para aprovechar al máximo el poder de las Unidades de Procesamiento Gráfico (GPU). Esto permite un procesamiento eficiente y rápido de las operaciones de aprendizaje profundo, lo que es especialmente beneficioso para el entrenamiento de modelos en conjuntos de datos grandes.

A continuación una pequeña comparación entre `PyTorch` y `TensorFlow`:

![1.png](ims%2F1%2F1.png)

En modelos disponibles en la comunidad de `Hugging Face`:

![2.png](ims%2F1%2F2.png)

> ## Un último consejo:
> No te enamores de la solución, enamórate del problema. Utiliza el `framework` que más ventajas te de para solucionar 
> el problema que quieres resolver. Entre más herramientas conozcas más fácil será tomar una decisión más fundamentada
> de qué `framework` es más conveniente para una situación concreta.


## 1.3 Hola mundo en PyTorch

Con PyTorch puedes aprovechar los últimos avances en inteligencia artificial para crear soluciones innovadoras para una amplia gama de aplicaciones, desde visión por computadora hasta procesamiento de lenguaje natural y más.

También cuenta con el respaldo de una comunidad vibrante y solidaria de desarrolladores e investigadores. Esto significa que tendrás acceso a una gran cantidad de recursos, herramientas y conocimientos que lo ayudarán a mantenerse a la vanguardia de la investigación y el desarrollo de IA.

Podrás apreciar cómo se ve un modelo relativamente sencillo. Sin embargo, esta siempre será la estructura de un modelo, no importa que sea de visión, transformers para procesamiento de lenguaje natural o reinforcement learning.

Debajo de todo eso, por más complicado que suene, hay un modelo de este tipo. Y es lo que marca la creación del `machine learning moderno`.

En PyTorch, `nn.Module` es muy importante. Es una clase que te permite crear, almacenar y manipular todo tipo de capas y operaciones de redes neuronales.

¿Por qué es esto genial? Significa que puedes crear modelos complejos con muchas capas sin tener que preocuparte por los detalles de cada una de ellas. nn.Module se encarga de todo eso por ti, para que puedas centrarte en construir tu modelo y obtener resultados.

En resumen, si estás trabajando con PyTorch y construyendo redes neuronales, nn.Module será tu nuevo mejor amigo.

> ### Nota. El código de esta clase está en: [hello_pytorch.py](scripts%2F1%2Fhello_pytorch.py)

```python
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        # Capa de embedding para mapear índices de palabras a vectores de embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capa LSTM para procesar los embeddings y obtener estados ocultos
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)

        # Capa lineal para proyectar el último estado oculto en la salida final
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Obtener los embeddings de las palabras
        embedded = self.embedding(text)

        # Propagar los embeddings a través de la capa LSTM
        output, (hidden, cell) = self.rnn(embedded)

        # Seleccionar el último estado oculto como estado final
        final_hidden = hidden[-1]

        # Pasar el estado final a través de la capa lineal para obtener la salida final
        return self.fc(final_hidden)


# Parámetros del modelo
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2

# Crear instancia del modelo
model = TextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Imprimir la estructura del modelo
print(model)
```
Respuesta esperada:
```commandline
TextClassifier(
  (embedding): Embedding(10000, 100)
  (rnn): LSTM(100, 256, num_layers=2, batch_first=True)
  (fc): Linear(in_features=256, out_features=2, bias=True)
)

Process finished with exit code 0
```

### Expliquemos el código paso a paso:

1. Se importan las bibliotecas necesarias: `torch` para el uso de PyTorch y `torch.nn` para las capas y funciones de redes neuronales.
2. Se define la clase `TextClassifier`, que es una subclase de `nn.Module`, la cual es la base para definir modelos de PyTorch.
3. En el método `__init__` de la clase `TextClassifier`, se definen los componentes del modelo. Los parámetros que recibe son:
   - `vocab_size:` Tamaño del vocabulario, es decir, la cantidad de palabras únicas en el corpus de texto.
   - `embedding_dim:` Dimensión de los vectores de embeddings. Cada palabra será representada por un vector de esta dimensión.
   - `hidden_dim:` Dimensión de los estados ocultos de la capa LSTM.
   - `output_dim:` Dimensión de la salida del clasificador.
   - En el cuerpo del método, se definen tres capas:
   - `self.embedding:` Una capa de embedding (nn.Embedding) que mapea cada índice de palabra a un vector de embeddings de tamaño embedding_dim.
   - `self.rnn:` Una capa LSTM (nn.LSTM) que recibe los embeddings y produce secuencias de estados ocultos. Tiene num_layers=2 capas LSTM apiladas y batch_first=True indica que la entrada se proporcionará en el formato (batch_size, sequence_length, input_size).
   - `self.fc:` Una capa lineal (nn.Linear) que proyecta el último estado oculto de la capa LSTM (hidden_dim) en la dimensión de salida (output_dim).
4. En el método `forward` de la clase `TextClassifier`, se define cómo se propagan los datos a través del modelo. El parámetro text representa las secuencias de palabras de entrada.
   - Primero, los embeddings de las palabras se obtienen utilizando self.embedding y se almacenan en embedded.
   - A continuación, embedded se pasa a través de la capa LSTM (self.rnn) y se obtienen output, hidden y cell. output contiene los estados ocultos de todas las palabras de la secuencia, mientras que hidden y cell contienen los estados finales de la capa LSTM.
   - El último estado oculto hidden[-1] se selecciona como el estado oculto final y se pasa a través de la capa lineal (self.fc) para obtener la salida final del clasificador.
   - La salida final se devuelve.
5. A continuación, se definen los parámetros para construir una instancia del modelo. `vocab_size` se establece en 10000, `embedding_dim` en 100, `hidden_dim` en 256 y `output_dim` en 2.

6. Se crea una instancia del modelo `TextClassifier` utilizando los parámetros definidos.

7. Finalmente, se imprime el modelo creado. Esto mostrará la estructura del modelo, incluyendo las capas y sus dimensiones.

## 1.4 Creación de Tensores en PyTorch

## 1.5 Debugging de operaciones con tensores

## 1.6 Conversión y operación de tensores con PyTorch

# 2 Estructura de modelo de deep learning en PyTorch

## 2.1 Generación y split de datos para entrenamiento del modelo

## 2.2 Estructura de modelo en PyTorch con torch.nn

## 2.3 Entrenamiento, funciones de pérdida y optimizadores

## 2.4 Entrenamiento y visualización de pérdida
## 2.5 Predicción con un modelo de PyTorch entrenado

# 3 Redes neuronales con PyTorch

## 3.1 Datos para clasificación de texto

## 3.2 Procesamiento de datos: tokenización y creación de vocabulario

## 3.3 Procesamiento de datos: preparación de DataLoader()

## 3.4 Creación de modelo de clasificación de texto con PyTorch

## 3.5 Función para entrenamiento

## 3.6 Función para evaluación

## 3.7 Split de datos, pérdida y optimización

## 3.8 Entrenamiento y evaluación de modelo de clasificación de texto

## 3.9 Inferencia utilizando torch.compile(): el presente con PyTorch 2.X

## 3.10 Almacenamiento del modelo con torch.save() y state_dict()

## 3.11 Sube tu modelo de PyTorch a Hugging Face

## 3.12 Carga de modelo de PyTorch con torch.load()

# 4 Cierre del curso
