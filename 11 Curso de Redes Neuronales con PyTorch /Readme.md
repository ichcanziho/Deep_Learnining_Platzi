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

En el deep learning los tensores son como la gasolina de PyTorch. Básicamente, son una forma elegante de decir una matriz multidimensional que se puede usar para almacenar y manipular datos en PyTorch.

Los tensores son un componente clave para crear y entrenar modelos de machine learning, lo que permite realizar operaciones como la multiplicación de matrices y la multiplicación por elementos en grandes conjuntos de datos. También son útiles para tareas como el procesamiento de imágenes, donde puede representar una imagen como un tensor tridimensional de pixeles.

Pero aquí está la parte interesante, los tensores son súper flexibles y se pueden usar para todo tipo de datos, no solo para imágenes o números. Puedes usarlos para cosas como datos de texto, donde puede representar palabras como vectores o embeddings.

Al dominar los tensores, podrás abordar problemas de deep learning más complejos y desarrollar modelos más sofisticados. Entonces, prepárate para sumergirte profundamente en el mundo de los tensores y ¡continuemos nuestro viaje hacia el emocionante mundo de PyTorch!

> ## Nota: El código de esta sección lo puedes encontrar en [2_tensores.py](scripts%2F1%2F2_tensores.py)

Importa PyTorch, revisemos la versión de PyTorch que estamos usando:

```python
import torch

print(torch.__version__)
```
Respuesta esperada:
```commandline
2.0.0+cu117  # nota esto cambiará de acuerdo a la instalación que tengas de PyTorch
```

Los escalares, vectores, matrices y tensores son conceptos matemáticos que se utilizan en el deep learning y otros campos de la ciencia y la ingeniería.

Un escalar es un valor numérico único, como `3` o `5,7`.

Un vector es una matriz unidimensional de valores numéricos, como `[1, 2, 3]` o `[0.2, 0.5, 0.8]`.

Una matriz es una matriz bidimensional de valores numéricos, como `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` o `[[0.1, 0.2], [0.3, 0,4], [0,5, 0,6]]`.

Un tensor es una matriz multidimensional de valores numéricos, que se puede considerar como una generalización de vectores y matrices. 

Un escalar es un tensor de orden 0, un vector es un tensor de primer orden y una matriz es un tensor de segundo orden. Los tensores de orden superior, como un tensor de tercer orden o un tensor de cuarto orden, pueden representar estructuras de datos más complejas, como imágenes o videos.

Aquí hay una ilustración simple:

Escalar: `3`

Vector: `[1, 2, 3]`

Matriz: `[[1, 2], [3, 4], [5, 6]]`

Tensor: `[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]`

Podemos representar estas estructuras de datos con PyTorch. Podemos crear tensores usando diferentes tipos de valores. Por ejemplo, random, ceros, o unos.

```python
escalar = torch.randn(1)
vector = torch.zeros(1, 10)
matriz = torch.ones(2, 2)

print(escalar)
print(vector)
print(matriz)
```
Respuesta esperada:
```commandline
tensor([-1.4738])
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
tensor([[1., 1.],
        [1., 1.]])
```

Pero también podemos representar estructuras sin nombre común:

```python
t5 = torch.randn(5, 2, 3)

print(t5)
```
Respuesta esperada:
```commandline
tensor([[[ 1.6913,  1.2107, -0.0387],
         [ 1.7567, -0.7986, -0.2261]],

        [[-1.6772, -0.8611,  0.4250],
         [ 0.1645,  0.3996, -1.0157]],

        [[ 0.7795,  0.6061, -1.3222],
         [-2.1188, -0.2354, -0.5088]],

        [[ 0.6316,  0.3993,  0.2365],
         [ 1.9170, -1.3484,  0.5882]],

        [[-0.2356,  0.4167,  0.6396],
         [-1.1904, -0.3743, -1.9593]]])
```
Podemos crear tensores con los valores que queramos, no necesariamente aleatorios:
```python
t2 = torch.tensor([[2, 2], [3, 3]])

print(t2)
```
Respuesta esperada:
```commandline
tensor([[2, 2],
        [3, 3]])
```

## 1.5 Debugging de operaciones con tensores

> ## Nota: El código de esta sección lo puedes encontrar en [2_tensores.py](scripts%2F1%2F2_tensores.py)

Cuando trabajamos con tensores y las operaciones resultan no válidas tendremos estos tres problemas más comunes:

- El tamaño o forma.
- El datatype.
- El device (dispositivo) en el que se encuentra el tensor.

La forma (shape) te dice cómo están organizados los elementos dentro del tensor.

```python
print(f"La shape de la matriz es {matriz.shape}")
print(f"La shape de t5 es {t5.shape}")
```
Respuesta esperada:
```commandline
La shape de la matriz es torch.Size([2, 2])
La shape de t5 es torch.Size([5, 2, 3])
```
También podemos conocer la dimensión de un tensor.
```python
print(f"La shape de la matriz es {matriz.ndim}")
print(f"La shape de t5 es {t5.ndim}")
```
Respuesta esperada:
```commandline
La shape de la matriz es 2
La shape de t5 es 3
```
Los tensores pueden tener elementos con diferentes types. Es importante saber qué type estamos usando.
```python
print(f"El tensor matriz tiene elementos de tipo: {matriz.dtype}")
```
Respuesta esperada:
```commandline
El tensor matriz tiene elementos de tipo: torch.float32
```

Hay muchos tipos de datos de tensor diferentes disponibles en PyTorch.

![3.png](ims%2F1%2F3.png)

El type más común es `torch.float` o `torch.float32` (float de 32 bits). Cuando hablamos de bits estamos tratando con el tamaño de la información necesaria para representar un número. En machine learning trabajamos con miles de números, por lo que elegir el tamaño ideal es clave. 

**La regla es:** los tipos de datos de menores bits (es decir, de menor precisión) son más rápidos de calcular pero sacrifican precisión (más rápido de calcular, pero menos preciso).

Normalmente, cuando operamos entre tensores, PyTorch convierte los tensores a tipos compatibles pero es importante que tengamos presente el tipo de los tensores para evitar errores futuros.

```python
matriz_float32 = torch.tensor([[3.1, 3.2], [3.3, 3.4]])
matriz_uint64 = torch.tensor([[3, 3], [3, 3]])

print(matriz_float32.dtype, matriz_uint64.dtype)

print((matriz_float32 + matriz_uint64).dtype)
```
Respuesta esperada:
```commandline
torch.float32 torch.int64
torch.float32
```
Si es necesario, podemos usar `y = y.to(...)` para convertir los tensores a diferentes types.
```python
print(matriz_uint64.to(torch.float32))
```
Respuesta esperada:
```commandline
tensor([[3., 3.],
        [3., 3.]])
```
Tenemos que tener en cuenta el dispositivo para el que nuestro tensor está preparado. No podemos operar con un tensor diseñado para GPU (CUDA) y uno para CPU.
```python
print(matriz_uint64.device)
```
Respuesta esperada:
```commandline
cpu
```
Revisamos si tenemos un GPU disponible con `cuda.is_available()`.

CUDA es una plataforma de computación paralela y una interfaz de programación de aplicaciones (API) que nos permite aprovechar la potencia de las GPU para tareas de deep learning. Cuando usamos CUDA podemos realizar operaciones matemáticas complejas en paralelo en la GPU, lo que puede acelerar significativamente el entrenamiento y la inferencia de modelos de machine learning.

Al usar CUDA, podemos aprovechar las capacidades masivas de procesamiento paralelo de las GPU y entrenar modelos mucho más rápido de lo que podríamos usar solo la CPU. 
```python
print(torch.cuda.is_available())
```
Respuesta esperada:
```commandline
True
```
El siguiente código, revisaremos si tenemos CUDA disponible y, si sí, convertimos tensores de CPU a CUDA y viceversa, a la vez que también cambiamos el type.
```python
if torch.cuda.is_available():
    matriz_uint64_cuda = matriz_uint64.to(torch.device("cuda"))

    print(matriz_uint64_cuda, matriz_uint64_cuda.type())
    print(matriz_uint64_cuda.to("cpu", torch.float32))
```
Respuesta esperada:
```commandline
tensor([[3, 3],
        [3, 3]], device='cuda:0') torch.cuda.LongTensor
tensor([[3., 3.],
        [3., 3.]])
```
## 1.6 Conversión y operación de tensores con PyTorch

Convierte el tensor a NumPy.
```python
print(type(matriz.numpy()))
```
Respuesta esperada:
```commandline
<class 'numpy.ndarray'>
```
Nota que podemos convertir también de NumPy a PyTorch y el type se mantiene.
```python
vector = np.ones(5)
vector_torch = torch.from_numpy(vector)
print(vector_torch, vector_torch.dtype)
```
Respuesta esperada:
```python
tensor([1., 1., 1., 1., 1.], dtype=torch.float64) torch.float64
```

### Operaciones con Tensores

Primero creemos algunos tensores en PyTorch:

```python
# create a tensor of zeros with shape (3, 4)
zeros_tensor = torch.zeros((3, 4))

# create a tensor of ones with shape (3, 4)
ones_tensor = torch.ones((3, 4))

# create a tensor of random values with shape (2, 2)
random_tensor = torch.randn(4)

print(random_tensor, random_tensor.shape)
```
Respuesta esperada:
```commandline
tensor([ 1.5882, -0.0746,  0.9659,  0.8113]) torch.Size([4])
```
Hagamos operaciones `element-wise:`

```python
# add two tensors element-wise
added_tensor = zeros_tensor + ones_tensor

# subtract two tensors element-wise
subtracted_tensor = zeros_tensor - ones_tensor

# multiply two tensors element-wise
multiplied_tensor = zeros_tensor * ones_tensor

# divide two tensors element-wise
divided_tensor = random_tensor / ones_tensor

print(divided_tensor)
```
Respuesta esperada:
```commandline
tensor([[ 1.5882, -0.0746,  0.9659,  0.8113],
        [ 1.5882, -0.0746,  0.9659,  0.8113],
        [ 1.5882, -0.0746,  0.9659,  0.8113]])
```
> Nota: Esta respuesta se debe a que PyTorch hizo `broadcasting`, realmente random_tensor tenía un shape de (1, 4) mientras que el tensor de ones era de (3, 4),
> entonces para que se pudiera hacer esta división exitosamente se extendió el resultado de la division de los números aleatorios a las dimenciones de la matriz que dividio.
> De esta manera se pudo realizar la operación.

Multiplicación de matrices:

```python
# create two matrices
matrix1 = torch.randn(2,3)
matrix2 = torch.randn(3,2)

print(f"matrix1 shape: {matrix1.shape}")
print(f"matrix2 shape: {matrix2.shape}")

# perform matrix multiplication
matx1x2 = torch.matmul(matrix1, matrix2)
print(matx1x2, matx1x2.shape)
```
Respuesta esperada:
```commandline
matrix1 shape: torch.Size([2, 3])
matrix2 shape: torch.Size([3, 2])
tensor([[-1.2454,  2.2567],
        [-1.9022,  5.0964]]) torch.Size([2, 2])
```

# 2 Estructura de modelo de deep learning en PyTorch

## 2.1 Generación y split de datos para entrenamiento del modelo

Cubriremos los fundamentos de la creación de un modelo PyTorch, desde la creación de un objeto nn.Module hasta el entrenamiento del modelo y la adición de una función de pérdida.

Para empezar utilizaremos un ejemplo sencillo de regresión lineal para ilustrar estos conceptos. Al final de esta clase, tendrás una comprensión sólida de cómo funciona PyTorch y cómo crear y entrenar tus propios modelos. ¡Comencemos!

> ### Nota: el código de esta sección está en [1_bloques_de_creacion.py](scripts%2F2%2F1_bloques_de_creacion.py)

### Importar librerías

```python
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Comprobar la versión de PyTorch
print(torch.__version__)
```
Respuesta esperada:
```commandline
2.0.0+cu117
```

### Genera tus datos

Por agilidad, para este ejemplo, crearemos datos sintéticos.

Creamos un tensor unidimensional llamado `X` que contiene un rango de valores, utilizando la función arange. Los parámetros de entrada `inicio`, `final` y `step` especifican el inicio, final y tamaño del paso del rango respectivamente.

La función `unsqueeze` se utiliza para agregar una dimensión adicional al tensor, lo que convierte el tensor unidimensional en un tensor de columna (con una dimensión adicional al final).

En resumen, este código crea un tensor de columna con un rango de valores especificados.

```python
# Crea *nuevos* parámetros
volumen = 0.8
sesgo = 0.2

# Crea datos
inicio = 0
final = 1
step = 0.025
X = torch.arange(inicio, final, step).unsqueeze(dim=1)
print(f"Shape de X: {X.shape}")
y = volumen * X + sesgo
print(f"Shape de y: {y.shape}")

print(X[:10], y[:10])
```
Respuesta esperada:
```commandline
Shape de X: torch.Size([40, 1])
Shape de y: torch.Size([40, 1])
tensor([[0.0000],
        [0.0250],
        [0.0500],
        [0.0750],
        [0.1000],
        [0.1250],
        [0.1500],
        [0.1750],
        [0.2000],
        [0.2250]]) tensor([[0.2000],
        [0.2200],
        [0.2400],
        [0.2600],
        [0.2800],
        [0.3000],
        [0.3200],
        [0.3400],
        [0.3600],
        [0.3800]])
```

Necesitamos un **conjunto de prueba** y uno de **entrenamiento**.

Cada conjunto tiene un objetivo específico:

*   **Conjunto de entrenamiento:** El modelo aprende de los datos.
*   **Conjunto de prueba:** El modelo se evalúa con los datos para probar lo que ha aprendido.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train), len(X_test))
```
Respuesta esperada:
```commandline
28 12
```
Tenemos 28 muestras para entrenamiento (```X_ent``` y ```y_ent```) y 12 muestras de prueba (```X_prueb``` y ```y_prueb```) 
Visualizamos nuestros datos.

```python
def plot_predictions(datos_ent=X_train,
                     etiq_ent=y_train,
                     datos_prueba=X_test,
                     etiq_prueba=y_test,
                     predictions=None):
    """
    Traza datos de entrenamiento, datos de prueba y compara predicciones
    """
    plt.figure(figsize=(10, 10))

    # Traza datos de entrenamiento en verde
    plt.scatter(datos_ent, etiq_ent, c="b", s=6, label="Datos de entrenamiento")

    # Traza datos de prueba en amarillo
    plt.scatter(datos_prueba, etiq_prueba, c="y", s=6, label="Datos de prueba")

    if predictions is not None:
        # Traza las predicciones en rojo
        plt.scatter(datos_prueba, predictions, c="r", s=6, label="Predicciones")

    # Leyenda
    plt.legend(prop={"size": 12})
    plt.savefig("datos.png")
    plt.close()


plot_predictions()
```

Respuesta esperada:
![datos_1.png](scripts%2F2%2Fdatos_1.png)


## 2.2 Estructura de modelo en PyTorch con torch.nn

Construyamos un modelo de regresión lineal utilizando PyTorch.

`torch.nn` proporciona herramientas para construir redes neuronales, `torch.optim` para optimizar los modelos, `data.Dataset` para manejar los conjuntos de datos y `torch.utils.data.DataLoader` para cargar y transformar los datos. Estas herramientas son fundamentales para la construcción y entrenamiento de modelos de machine learning.

* `torch.nn`: es un módulo que proporciona clases y funciones para construir redes neuronales. Contiene una variedad de capas, como capas de convolución, capas de agrupación, capas de normalización, capas recurrentes y capas completamente conectadas, que se pueden combinar para construir una variedad de arquitecturas de redes neuronales.

* `torch.optim`: proporciona clases y funciones para optimizar los modelos de machine learning. Contiene una variedad de algoritmos de optimización, como SGD, Adam, Adagrad y Adadelta, que se utilizan para ajustar los parámetros de los modelos durante el entrenamiento.

* `torch.utils.data.Dataset`: es una clase que se utiliza para representar conjuntos de datos de machine learning. Proporciona una interfaz consistente para acceder a los datos y sus etiquetas. Se puede personalizar para trabajar con conjuntos de datos de diferentes formatos y tipos.

* `torch.utils.data.DataLoader`: es una clase que se utiliza para cargar y transformar datos de un conjunto en lotes para el entrenamiento de modelos. Se encarga de la asignación de los lotes, la mezcla de los datos y la carga de los datos en la GPU si es necesario, lo que facilita el procesamiento de grandes conjuntos de datos.

Por el momento, vamos a utilizar los dos primeros módulos. En el proyecto final usaremos `DataLoader`.

> ## Nota: el código de esta sección esta en [1_bloques_de_creacion.py](scripts%2F2%2F1_bloques_de_creacion.py)

El siguiente fragmento define una clase llamada `ModeloRegresionLineal` que hereda de la clase `nn.Module` en PyTorch. 

La clase tiene dos parámetros ajustables (`volumen` y `sesgo`) que se inicializan con valores aleatorios y se pueden optimizar durante el entrenamiento. 

La función `forward` realiza el cálculo del modelo, donde se multiplica el tensor de entrada `x` por el parámetro `volumen` y se le agrega el parámetro `sesgo`.

En resumen, este código define una clase de modelo de regresión lineal con dos parámetros ajustables y define la operación de cálculo del modelo. Esto se utiliza como una plantilla para definir y entrenar modelos de regresión lineal en PyTorch.

```python
# Crea una clase de modelo de regresión lineal
class ModeloRegresionLineal(nn.Module):
    def __init__(self):
        super().__init__()
        self.volumen = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.sesgo = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    # Define el cálculo en el modelo
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.volumen * x + self.sesgo
```

`torch.manual_seed(42)` establece la semilla del generador de números aleatorios en 42. Lo que significa que los mismos números aleatorios se generarán en cada ejecución del código que utilice tensores de PyTorch. 

Esto puede ser útil para reproducir resultados y garantizar la reproducibilidad de los experimentos de machine learning.

```python
torch.manual_seed(42)
# Crea un objeto instanciando la clase ModeloRegresionLineal

model_1 = ModeloRegresionLineal()

print(model_1)
```
Respuesta esperada:
```commandline
ModeloRegresionLineal()
```
`model_1.state_dict()` devolverá el diccionario que contiene los valores de todos los parámetros entrenables del modelo. Este diccionario puede ser útil para guardar y cargar modelos.
```python
print(model_1.state_dict())
```
Respuesta esperada:
```commandline
OrderedDict([('volumen', tensor([0.3367])), ('sesgo', tensor([0.1288]))])
```
Los valores de las variables volumen y sesgo aparecen como tensores aleatorios.

Iniciamos con parámetros aleatorios para luego hacer que el modelo los actualice hacia parámetros que se ajusten mejor a nuestros datos.

### Predicciones usando `torch.inference_mode()`

Para hacer predicciones con `torch.inference_mode()` podemos pasar los datos de prueba `X_prueb` para ver qué tan cerca pasan de `y_prueb`.

Cuando pasemos datos al modelo, pasarán por `forward()` produciendo un resultado con el cálculo que definimos anteriormente.

```python
# Hacer predicciones con el modelo
with torch.inference_mode():
    y_predc = model_1(X_test)
```

> `#torch.inference_mode` se usa para hacer inferencia (predicciones). Además, desactiva algunas opciones como el seguimiento del gradiente (necesario para el entrenamiento, NO para inferencia).

```python
# Comprueba las predicciones
print(y_predc)
```
Respuesta esperada:
```commandline
tensor([[0.2887],
        [0.2635],
        [0.2551],
        [0.3477],
        [0.1625],
        [0.2298],
        [0.4402],
        [0.3561],
        [0.4571],
        [0.1793],
        [0.3392],
        [0.2046]])
```
Hay un valor de predicción por muestra de prueba debido al tipo de datos implementados.



> En este caso, para una línea recta, un valor `X` se asigna a un valor `y`. Sin embargo, los modelos de aprendizaje automático son muy flexibles. Podemos asignar 80 valores de `X` para 10 valores `y`.

Visualicemos nuestros datos utilizando `plot_predictions()`

```python
plot_predictions(predictions=y_predc, name="2")
```
Respuesta esperada:
![datos_2.png](scripts%2F2%2Fdatos_2.png)

Recuerda que nuestro modelo solo usa valores de parámetros aleatorios para hacer predicciones, básicamente al azar. **Por ello, la predicción se ve mal.**

## 2.3 Entrenamiento, funciones de pérdida y optimizadores

Para arreglar los valores aleatorios de los parámetros del modelo podemos actualizar los parámetros internos de las variables `# volumen` y `# sesgo` para representar mejor los datos.

Para ello, crearemos una **función de pérdida** así como un **optimizador** con PyTorch.

> La `# función de pérdida` mide qué tan equivocadas están las predicciones del modelo `# y_predc`, en comparación con las etiquetas `# y_prueb`. PyTorch tiene muchas funciones de pérdida integradas en `#torch.nn`.


> El `#optimizador` le indica a los modelos cómo actualizar sus parámetros internos para reducir la pérdida. Podemos encontrar varias implementaciones en `#torch.optim`.

Dependiendo del tipo de problema que estemos trabajando vamos a emplear una determinada función de pérdida y optimización.

Para nuestro problema utilizaremos el **Error Cuadrático Medio (MAE)** como la función de pérdida `(torch.nn.L1Loss)` para medir la diferencia absoluta entre dos puntos y tomar la media en todos los ejemplos.

También usaremos **Stochastic Gradient Descent (SGD)** `(torch.optim.SGD(params, lr))` como nuestro optimizador, donde `# params` son los parámetros del modelo (volumen y sesgo) y `# lr` es la tasa de aprendizaje a la que desea que el optimizador actualice los parámetros.

Además, fijamos arbitrariamente una tasa de aprendizaje de `0.01

```python
# Crea función de pérdida
fn_perd = torch.nn.L1Loss()

# Crea el optimizador
optimizador = torch.optim.SGD(params=model_1.parameters(), lr=0.01)  # tasa de aprendizaje (cuánto debe cambiar el
# optimizador de parámetros en cada paso, más alto = más (menos estable), más bajo = menos (puede llevar mucho tiempo))

print(fn_perd)
```
Respuesta esperada:
```commandline
L1Loss()
```

Ya que tenemos una función de pérdida y un optimizador, vamos a crear un ciclo de entrenamiento y uno de prueba. Esto implica que el modelo pase por los datos de entrenamiento y aprenda la relación entre `features` y `labels`.

El ciclo de prueba implica revisar los datos de prueba y evaluar qué tan buenos son los patrones que el modelo aprendió de los datos de entrenamiento.

Para entrenar, vamos a escribir un bucle `for` de Python.


### Bucle de entrenamiento:

**Pasos a seguir:**

1.  El modelo pasa por todos los datos de entrenamiento nuevamente, realizando sus cálculos en funciones `forward ()`. *Código:* `modelo(X_ent)`
2. Las predicciones se comparan  y se evalúan para ver qué tan equivocadas están. *Código:* `perdida = fn_perd(y_predc, y_ent)`.
3. Los gradientes de los optimizadores se establecen en cero para que puedan recalcularse y dar paso al entrenamiento específico. *Código:* `optimizer.zero_grad()`.
4. Se calcula el gradiente de la pérdida con respecto a cada parámetro que se actualizará (retroprogramación). *Código:* `loss.backward()`.
5. Se actualizan los parámetros con `requires_grand=True` respecto a la pérdida para mejorarlos. *Código:* `optimizer.step()`.

```python
torch.manual_seed(42)

# Establezca cuántas veces el modelo pasará por los datos de entrenamiento
epocas = 100

# Cree listas de vacías para realizar un seguimiento de nuestro modelo
entrenamiento_loss = []
test_loss = []

for epoca in range(epocas):
    # Entrenamiento

    # Pon el modelo en modo entrenamiento
    model_1.train()

    # 1. Pase hacia adelante los datos usando el método forward()
    y_predc = model_1(X_train)

    # 2. Calcula la pérdida (Cuán diferentes son las predicciones de nuestros modelos)
    perdida = fn_perd(y_predc, y_train)

    # 3. Gradiente cero del optimizador
    optimizador.zero_grad()

    # 4. Pérdida al revés
    perdida.backward()

    # 5. Progreso del optimizador
    optimizador.step()

    # Función de prueba

    # Pon el modelo en modo evaluación
    model_1.eval()
```
Hasta este momento ya tenemos un código capaz de iterar `epoca` a `epoca` y mejorando continuamente
los resultados del modelo. En la siguiente clase vamos a ver como podemos ir guardando la perdida del modelo `epoca` a `epoca`
para ello volveremos a utilizar el `torch.inference_model()`

## 2.4 Entrenamiento y visualización de pérdida

Como el modelo ya se encuentra en `modo evaluación` entonces podemos empezar a hacer inferencias con el modelo entrenado en un
punto específico
```python
    with torch.inference_mode():

        # 1. Reenviar datos de prueba
        prueba_predc = model_1(X_test)

        # 2. Calcular la pérdida en datos de prueba
        prueb_perd = fn_perd(prueba_predc, y_test.type(torch.float))

        # Imprime lo que está pasando
        if epoca % 10 == 0:
            # usamos `.detach()` porque NO queremos almacenar el número como un `tensor` es más fácil trabajar con el como un valor de `numpy`
            entrenamiento_loss.append(perdida.detach().numpy())
            test_loss.append(prueb_perd.detach().numpy())
            print(f"Epoca: {epoca} | Entrenamiento pérdida: {perdida} | Test pérdida {prueb_perd}")

```
Respuesta esperada:
```commandline
Epoca: 0 | Entrenamiento pérdida: 0.29664039611816406 | Test pérdida 0.28563693165779114
Epoca: 10 | Entrenamiento pérdida: 0.17860610783100128 | Test pérdida 0.16613411903381348
Epoca: 20 | Entrenamiento pérdida: 0.11701875925064087 | Test pérdida 0.10174953937530518
Epoca: 30 | Entrenamiento pérdida: 0.09006500989198685 | Test pérdida 0.07594708353281021
Epoca: 40 | Entrenamiento pérdida: 0.07644685357809067 | Test pérdida 0.06794114410877228
Epoca: 50 | Entrenamiento pérdida: 0.0689956396818161 | Test pérdida 0.06417617946863174
Epoca: 60 | Entrenamiento pérdida: 0.06370030343532562 | Test pérdida 0.05965892970561981
Epoca: 70 | Entrenamiento pérdida: 0.058579545468091965 | Test pérdida 0.054845139384269714
Epoca: 80 | Entrenamiento pérdida: 0.0534588024020195 | Test pérdida 0.050031352788209915
Epoca: 90 | Entrenamiento pérdida: 0.04834337159991264 | Test pérdida 0.045158278197050095
```

Parece que nuestra pérdida fue disminuyendo con cada época, veamos graficamente:
```python
# Traza las curvas de pérdida
plt.plot(entrenamiento_loss, label="Perd entrenamiento")
plt.plot(test_loss, label="Perd prueba")
plt.ylabel("Pérdida")
plt.xlabel("Epoca")
plt.legend()
plt.savefig(f"datos_perdida.png")
plt.close()
```
Respuesta esperada:
![datos_perdida.png](scripts%2F2%2Fdatos_perdida.png)


## 2.5 Predicción con un modelo de PyTorch entrenado

Una vez entrenado el modelo, podemos hacer inferencia (predicciones) con él.

Hay tres aspectos que debemos recordar para hacer predicciones correctamente:

1. Configurar el modelo en modo de evaluación `(model.eval())`.
2. Realizar las predicciones utilizando el administrador de contexto del modo de inferencia `(with torch.inference_mode(): ...)`.
3. Todas las predicciones deben realizarse con objetos en el mismo dispositivo (datos y modelo solo en GPU o datos y modelo solo en CPU).

```python
# 1. Configura el modelo en modo de evaluación
model_1.eval()

# 2. Configura el administrador de contexto del modo de inferencia
with torch.inference_mode():
    # 3. Asegúrate de que los cálculos se realicen con el modelo y los datos en el mismo dispositivo en nuestro caso,
    # nuestros datos y modelo están en la CPU de forma predeterminada
    # model_1.to(device)
    # X_prueb = X_prueb.to(device)
    y_predc = model_1(X_test)
print(y_predc)
```
Respuesta esperada:
```commandline
tensor([[0.6694],
        [0.6829],
        [0.6965],
        [0.7100],
        [0.7236],
        [0.7371],
        [0.7506],
        [0.7642],
        [0.7777],
        [0.7913],
        [0.8048],
        [0.8184]])
```
Observa cómo se ve gráficamente el modelo entrenado:
```python
plot_predictions(predictions=y_predc, name="3")
```
![datos_3.png](scripts%2F2%2Fdatos_3.png)



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
