# Curso de Transfer Learning con Hugging Face

Eleva tus habilidades en deep learning aplicando transfer learning a modelos pre-entrenados. Aprende a utilizar todo el poder de la biblioteca de modelos de machine learning y datasets open-source de Hugging Face de la mano de sus expertos.

- Comparte tus modelos en el Hub de Hugging Face.
- Afina modelos de visión computarizada y procesamiento de lenguaje natural.
- Aplica transfer learning de forma sencilla y rápida.
- Utiliza el Hub de Hugging Face y su biblioteca de modelos.

> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
>
> Este Curso es el Número 4 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso


# Índice:
- [1 Introducción al Hub de Hugging Face](#1-introducción-al-hub-de-hugging-face)
  - [1 Introducción al transfer learning](#11-introducción-al-transfer-learning)
  - [1.1 Machine Learning abierto para todo el mundo](#12-machine-learning-abierto-para-todo-el-mundo)
  - [1.2 Las tasks en machine learning](#13-las-tasks-en-machine-learning)
  - [1.3 Quiz: introducción al hub de hugging face](#14-quiz--introducción-al-hub-de-hugging-face)
- [2 Primeros pasos con transfer learning y transformers](#2-primeros-pasos-con-transfer-learning-y-transformers)
  - [2.1 Tus primeros modelos pre-entrenados usando pipeline](#21-tus-primeros-modelos-pre-entrenados-usando-pipeline)
  - [2.2 Explorando los datasets](#22-explorando-los-datasets)
  - [2.3 Introducción a los Transformers](#23-introducción-a-los-transformers)
  - [2.4 Aplicando transfer learning](#24-aplicando-transfer-learning)
  - [2.5 Quiz: Primeros pasos con transfer learning](#25-quiz--primeros-pasos-con-transfer-learning)
- [3 Computer Vision](#3-computer-vision)
  - [3.1 Carga de dataset para computer vision](#31-carga-de-dataset-para-computer-vision)
  - [3.2 Procesamiento de dataset para computer vision](#32-procesamiento-de-dataset-para-computer-vision)
  - [3.3 Configurando un Trainer para computer vision](#33-configurando-un-trainer-para-computer-vision)
  - [3.4 Entrenamiento y evaluación de modelo de computer vision](#34-entrenamiento-y-evaluación-de-modelo-de-computer-vision)
  - [3.5 Quiz: Computer vision con Hugging Face](#35-quiz--computer-vision-con-hugging-face)
- [4 Natural Language Processing](#4-natural-language-processing)
  - [4.1 Carga de datasets para NLP](#41-carga-de-datasets-para-nlp)
  - [4.2 Procesamiento de dataset para NLP](#42-procesamiento-de-dataset-para-nlp)
  - [4.3 Configurando un Trainer para NLP](#43-configurando-un-trainer-para-nlp)
  - [4.4 Entrenamiento y evaluación de modelo de NLP](#44-entrenamiento-y-evaluación-de-modelo-de-nlp)
  - [4.5 Quiz: NLP con hugging face](#45-quiz--nlp-con-hugging-face)
- [5 Comparte en el HUB](#5-comparte-en-el-hub)
  - [5.1 El Hub como tu curriculum para machine learning](#51-el-hub-como-tu-curriculum-para-machine-learning)
  - [5.2 Compartir tu modelo en el HUB de Hugging Face](#52-compartir-tu-modelo-en-el-hub-de-hugging-face)

# 1 Introducción al Hub de Hugging Face

## 1.1 Introducción al transfer learning

**Una breve introducción a Hugging Face**

El Hub de Hugging Face es una plataforma en línea que permite a los desarrolladores de inteligencia artificial compartir, descubrir y usar modelos pre-entrenados de aprendizaje profundo y otros recursos de procesamiento de lenguaje natural (NLP). Hugging Face es una empresa de tecnología de inteligencia artificial que se especializa en herramientas y modelos de NLP, y su Hub es una parte importante de su plataforma.

![1.png](imgs%2F1%20Introducci%C3%B3n%2F1.png)

En el Hub de Hugging Face, los usuarios pueden encontrar una amplia variedad de modelos de NLP pre-entrenados en varios idiomas, incluyendo modelos para tareas como la traducción automática, el reconocimiento de entidades nombradas, la generación de lenguaje natural y mucho más. Los modelos están disponibles en diferentes tamaños y configuraciones, lo que permite a los usuarios encontrar el modelo adecuado para su caso de uso específico.

Además de los modelos pre-entrenados, el Hub también contiene recursos de datos, códigos y herramientas para ayudar a los desarrolladores a construir y mejorar sus propios modelos de NLP. La plataforma es de código abierto, lo que significa que los usuarios pueden contribuir y mejorar los recursos existentes para el beneficio de la comunidad en general.

**¿Qué aprenderemos en este curso?**

- Qué es y cómo hacer transfer learning?
- Qué es el Hub de Hugging Face y cómo usarlo
- Utilizar modelos modernos para nuestras propias aplicaciones.
- Afinar modelos de lenguaje y visión
- Utilizar datasets open source compartidos por la comunidad en el Hub

**Bases de Transfer Learning**

Transfer learning (aprendizaje transferido, en español) es una técnica de aprendizaje automático en la que se aprovecha 
el conocimiento adquirido de un modelo entrenado en una tarea para mejorar el desempeño de otro modelo en una tarea relacionada.

![2.png](imgs%2F1%20Introducci%C3%B3n%2F2.png)

En lugar de entrenar un modelo desde cero para una tarea específica, el aprendizaje transferido utiliza un modelo previamente 
entrenado en una tarea similar como punto de partida y ajusta sus parámetros para la tarea en cuestión. Esto se logra mediante 
la reutilización de algunas o todas las capas de la red neuronal del modelo previo, que contienen conocimientos generales 
sobre el lenguaje o las imágenes.

**¿Por qué deberíamos utilizar Transfer Learning?**

El beneficio de esta técnica es que puede reducir significativamente la cantidad de datos y el tiempo necesarios para entrenar 
un modelo para una nueva tarea, especialmente si la tarea en cuestión tiene una cantidad limitada de datos de entrenamiento 
disponibles. Además, el modelo pre-entrenado puede contener características generales útiles para tareas relacionadas, lo 
que puede mejorar la calidad de la predicción.

El aprendizaje transferido se ha demostrado que es efectivo en una amplia variedad de tareas de aprendizaje automático, 
incluyendo la clasificación de imágenes, el procesamiento del lenguaje natural, el reconocimiento de voz, entre otros.

## 1.2 Machine Learning abierto para todo el mundo

En la actualidad es MUY complejo e incluso inocente pensar que alguna empresa será capaz de "resolver la IA" por sí misma.
Esto debido al alto nivel de complejidad de la tarea. Este tipo de problemas requieren el esfuerzo de una gran cantidad de
personas trabajando por un bien común. Se requiere de una `comunidad`. Tener a más personas contribuyendo en los proyectos de
IA es la forma más efectiva y eficiente de generar mejores y más poderosos modelos.

Actualmente, `Hugging Face` es una comunidad de IA en donde podremos encontrar:

- más 50 mil modelos
- más de 5 mil datasets
- más de 5 mil Spaces (demos)

Los modelos son libres, y podemos usarlos como base para crear nuestras nuevas propuestas de modelos y solucionar problemas
cada vez más específicos.

Empecemos por conocer la página de Hugging Face: https://huggingface.co/

![3.png](imgs%2F1%20Introducci%C3%B3n%2F3.png)


De forma muy sencilla, podemos decir que dentro de Hugging Face, cada modelo es en sí mismo un repositorio de Git. Hugging Face
nos permite filtrar por una amplia variedad de parámetros, desde el tipo de problema: Multimodal, NLP, Audio, Computer Vision.
Hasta por el tipo de biblioteca que manejamos: Pytorch, TensorFlow, Keras, Scikit-learn etc. Por idioma, por licencia entre otros.


![4.png](imgs%2F1%20Introducci%C3%B3n%2F4.png)

Nuestros filtros no arrojaran que modelos cumplen con nuestros requisitos:

![5.png](imgs%2F1%20Introducci%C3%B3n%2F5.png)

Por ejemplo conozcamos a: [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

![6.png](imgs%2F1%20Introducci%C3%B3n%2F6.png)

Podemos observar como es una página bastante similar a GitHub. Incluso en su pestaña de `Files and versions` podemos encontrar
una estructura muy similar a Github.

![7.png](imgs%2F1%20Introducci%C3%B3n%2F7.png)

Varios modelos de Hugging Face me van a permitir usar su `Hosted inference API`
Una herramienta sumamente práctica que me va a dejar probar el funcionamiento del modelo de una forma muy simple.

Por ejemplo: busquemos [xlm-roberta-base](https://huggingface.co/xlm-roberta-base?text=Hola+soy+una+persona+muy+%3Cmask%3E+y+me+encanta%21)

![8.png](imgs%2F1%20Introducci%C3%B3n%2F8.png)

De forma sumamente rápida podemos ver el funcionamiento de este modelo. Inlcuso en la ventaja de `Use in Transformers`
nos dan un ejemplo de implementación simple en código:

![9.png](imgs%2F1%20Introducci%C3%B3n%2F9.png)


## 1.3 Las tasks en machine learning

En esta clase aprenderemos ¿Qué son los tasks? Y ¿Cómo elegir el mejor modelo para nuestra aplicación?

Una task es una aplicación específica para la que creas un modelo de Machine Learning. Por ejemplo, pensemos en las siguientes
áreas de aplicación de Deep Learning y mencionemos algunas de sus `tasks`:

- Computer Vision:
  - Image Classification
  - Image Segmentation
  - Image-to-image
  - Unconditional Image Generation
  - Object Detection
- Natural Language Processing
  - Translation
  - Token Classification
  - Sentence Similarity
  - Question Answering
  - Summarization
  - Zero-Shot Classification
  - Text Classification
  - Text2Text Generation
- Audio
  - Audio-Classification
  - Text-to-Speech
  - Audio-to-Audio
- Multimodal
  - Feature Extraction
  - Image-to-Text
  - Text-to-Image
- Tabular
  - Tabular Classification
  - Tabular Regression
- Reinforcement Learning
  - Reinforcement Learning

Todos estos y más `tasks` están disponibles en `Hugging Face`: https://huggingface.co/tasks

![10.png](imgs%2F1%20Introducci%C3%B3n%2F10.png)

Cada `task` nos brinda más información al respecto del mismo, modelos disponibles, bibliotecas compatibles y en algunos
casos incluso códigos de ejemplo en formato NoteBook e incluso Datasets. Por ejemplo veamos el task de `Text Classification`

![11.png](imgs%2F1%20Introducci%C3%B3n%2F11.png)



## 1.4 Quiz: introducción al hub de hugging face

![12.png](imgs%2F1%20Introducci%C3%B3n%2F12.png)

# 2 Primeros pasos con transfer learning y transformers

## 2.1 Tus primeros modelos pre-entrenados usando pipeline

> ## Nota:
> Puedes encontrar el notebook completo de esta clase: [Aquí](2%20Primeros%20pasos%20con%20transfer%20learning%20y%20transformers%2F1%20Primeros%20modelos%20pre-entrenados%20con%20pipeline%2FIntroducci%C3%B3n_al_Hub_de_Hugging_Face.ipynb)

En esta clase vamos a ver una forma sumamente simple de utilizar `pipeline` para utilizar modelos pre-entrenados de Hugging Face.
El código completo de todas las pruebas se encuentra en el notebook descrito anteriormente. Sin embargo, en esta sección 
solamente vamos a presentar un solo ejemplo, puesto que en general todos son básicamente lo mismo.

Primero debemos empezar instalando la biblioteca de `transformers` de `Hugging Face`

```bash
pip install transformers
```
Respuesta esperada:
```commandline
Successfully installed filelock-3.10.7 huggingface-hub-0.13.3 pyyaml-6.0 regex-2023.3.23 tokenizers-0.13.2 transformers-4.27.3
```

Imaginemos que queremos clasificar una imagen y qué un modelo nos diga que entidad es capaz de reconocer en dicha imagen.
Por ejemplo la siguiente:

![coche.png](2%20Primeros%20pasos%20con%20transfer%20learning%20y%20transformers%2F1%20Primeros%20modelos%20pre-entrenados%20con%20pipeline%2Fcoche.png)

Nosotros ya sabemos que se trata de un `coche`, pero veamos que tan fácil es para un modelo de `Hugging Face` determinar esto:

```python
from transformers import pipeline

if __name__ == '__main__':

    obj_classification = pipeline(task="image-classification")
    ans = obj_classification("coche.png")
    print(ans)
```
> Nota:
> 
> Para este ejemplo de código hemos puesto `lo mínimo` indispensable para correr un modelo a través de pipeline, lo cual
> es especificar que `task` estamos buscando resolver. En este ejemplo ha sido `image-classification`, sin embargo, no hemos puesto
> ningún otro parámetro, como por ejemplo: `model`. Esto obliga a `Hugging Face` a inferir cuál sería el mejor modelo que se
> adapte a nuestra necesidad, automáticamente propone un modelo y lo descarga por nosotros
>   >No model was supplied, defaulted to google/vit-base-patch16-224 and revision 5dca96d (https://huggingface.co/google/vit-base-patch16-224).
Using a pipeline without specifying a model name and revision in production is not recommended.
Downloading (…)lve/main/config.json: 100%|██████████| 69.7k/69.7k [00:00<00:00, 369kB/s]
Downloading tf_model.h5: 100%|██████████| 347M/347M [00:36<00:00, 9.44MB/s]

Respuesta esperada:
```commandline
[{'score': 0.6160857677459717, 'label': 'minivan'}, {'score': 0.2290043979883194, 'label': 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon'}, {'score': 0.03376132249832153, 'label': 'car wheel'}, {'score': 0.027296774089336395, 'label': 'jeep, landrover'}, {'score': 0.020214203745126724, 'label': 'grille, radiator grille'}]
```

Excelente, el modelo ha clasificado exitosamente a nuestra imagen no solo como un coche, sino que especifica mente un `minivan`
Con esto hemos logrado resolver un problema de clasificación de imágenes con simplemente 3 líneas de código. Ha sido sumamente sencillo
usar `pipeline` y dejar que `Hugging Face` se encargase de todo lo demás por nosotros.

Sin embargo, cabe destacar que la clase `pipeline` tiene varios parámetros configurables, entre ellos podemos encontrar los siguientes:

> `task (str)` — The task defining which pipeline will be returned. Currently accepted tasks are:
> - "audio-classification": will return a AudioClassificationPipeline.
> - "automatic-speech-recognition": will return a AutomaticSpeechRecognitionPipeline.
> - "conversational": will return a ConversationalPipeline.
> - "depth-estimation": will return a DepthEstimationPipeline.
> - "document-question-answering": will return a DocumentQuestionAnsweringPipeline.
> - "feature-extraction": will return a FeatureExtractionPipeline.
> - "fill-mask": will return a FillMaskPipeline:.
> -"image-classification": will return a ImageClassificationPipeline.
>- "image-segmentation": will return a ImageSegmentationPipeline.
>- "image-to-text": will return a ImageToTextPipeline.
>- "object-detection": will return a ObjectDetectionPipeline.
>- "question-answering": will return a QuestionAnsweringPipeline.
>- "summarization": will return a SummarizationPipeline.
>- "table-question-answering": will return a TableQuestionAnsweringPipeline.
>- "text2text-generation": will return a Text2TextGenerationPipeline.
>- "text-classification" (alias "sentiment-analysis" available): will return a TextClassificationPipeline.
>- "text-generation": will return a TextGenerationPipeline:.
>- "token-classification" (alias "ner" available): will return a TokenClassificationPipeline.
> - "translation": will return a TranslationPipeline.
>- "translation_xx_to_yy": will return a TranslationPipeline.
>- "video-classification": will return a VideoClassificationPipeline.
>- "visual-question-answering": will return a VisualQuestionAnsweringPipeline.
>- "zero-shot-classification": will return a ZeroShotClassificationPipeline.
>- "zero-shot-image-classification": will return a ZeroShotImageClassificationPipeline.
> - "zero-shot-audio-classification": will return a ZeroShotAudioClassificationPipeline.
>- "zero-shot-object-detection": will return a ZeroShotObjectDetectionPipeline.
>
> `model (str or PreTrainedModel or TFPreTrainedModel, optional)` — The model that will be used by the pipeline to make predictions. This can be a model identifier or an actual instance of a pretrained model inheriting from PreTrainedModel (for PyTorch) or TFPreTrainedModel (for TensorFlow).
> If not provided, the default for the task will be loaded.
>
> `config (str or PretrainedConfig, optional)` — The configuration that will be used by the pipeline to instantiate the model. This can be a model identifier or an actual pretrained model configuration inheriting from PretrainedConfig.
> If not provided, the default configuration file for the requested model will be used. That means that if model is given, its default configuration will be used. However, if model is not supplied, this task’s default model’s config is used instead.

Puedes leer la documentación completa en: [The pipeline abstraction](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline)


## 2.2 Explorando los datasets

Hugging Face tiene una gran variedad de `datasets` públicos y disponibles, puedes acceder a ellos en: https://huggingface.co/datasets
De igual forma que los modelos, los datasets cuentan con la barra de búsqueda de la izquierda, en donde podremos filtrar 
por `task`, por idioma, licencia entre otros. 

![1.png](imgs%2F2%20Primeros%20pasos%2F1.png)

Cuando entras a alguno de ellos, por ejemplo: [imdb](https://huggingface.co/datasets/imdb)
Puedes observar: un preview del dataset, una descripción de qué es y para qué sirve, e incluso modelos que lo utilicen, 
ya sea que hayan sido entrenados con dicha base o `fine-tuned`.

![2.png](imgs%2F2%20Primeros%20pasos%2F2.png)

## 2.3 Introducción a los Transformers

El mundo de los `transformers` en Deep Learning nace en 2017 de la mano del siguiente artículo: [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

![3.png](imgs%2F2%20Primeros%20pasos%2F3.png)

A continuación un breve resumen del paper:

"Attention is All You Need" es un artículo presentado en la conferencia de 2017 sobre Aprendizaje Profundo y Aprendizaje Representacional de la Asociación para el Avance de la Inteligencia Artificial (AAAI). El artículo propone una nueva arquitectura de red neuronal llamada Transformer, que utiliza exclusivamente mecanismos de atención para procesar secuencias de texto.

La arquitectura Transformer fue diseñada para superar las limitaciones de las arquitecturas de redes neuronales recurrentes (RNN) y de las arquitecturas basadas en convoluciones (CNN), que habían sido las más utilizadas para procesar secuencias de texto hasta ese momento. La principal ventaja de la arquitectura Transformer es que elimina la necesidad de utilizar una red neuronal recurrente o una red neuronal convolucional para procesar las secuencias de entrada, lo que permite una mayor paralelización del procesamiento y un entrenamiento más rápido de la red.

![4.png](imgs%2F2%20Primeros%20pasos%2F4.png)

La arquitectura Transformer utiliza capas de auto-atención, en las que cada elemento de la secuencia de entrada se relaciona con todos los demás elementos de la secuencia para producir una representación contextualizada de cada elemento. Esto se logra mediante la combinación de tres subcapas: una capa de auto-atención multi-cabeza, una capa de normalización de capa y una capa de red neuronal completamente conectada. El uso de múltiples cabezas de atención permite a la red aprender diferentes relaciones entre los elementos de la secuencia, lo que mejora su capacidad para procesar secuencias de texto.

El artículo demuestra la eficacia de la arquitectura Transformer en varias tareas de procesamiento de lenguaje natural, incluyendo la traducción automática y el modelado del lenguaje. La arquitectura Transformer ha demostrado ser muy exitosa y ha sido utilizada en varias aplicaciones de procesamiento de lenguaje natural desde entonces.

A continuación, explicaremos con más detalle cómo funcionan los transformers:

1. `Representación de la entrada:` La primera capa de la red transforma la entrada (por ejemplo, una secuencia de palabras) en una representación vectorial de alta dimensión. Cada elemento de la secuencia se representa mediante un vector de palabras (word embedding), que es una representación densa de baja dimensión de la palabra.

2. `Módulo de atención:` La siguiente capa de la red utiliza un mecanismo de atención para calcular una puntuación de atención para cada elemento de la secuencia. Esto permite que la red se centre en partes específicas de la entrada mientras procesa la secuencia completa. Los valores de atención se calculan como un producto escalar entre vectores de consulta (query) y vectores de clave (key) asociados a cada elemento de la secuencia. El resultado se normaliza y se utiliza para ponderar los vectores de valor (value) asociados a cada elemento de la secuencia.

3. `Capas de transformación:` Después del módulo de atención, la red utiliza varias capas de transformación para procesar la entrada. Cada capa de transformación se compone de varias subcapas, incluyendo una capa de normalización y dos capas completamente conectadas (feedforward). Estas capas ayudan a la red a aprender representaciones más complejas de la entrada y permiten que la red se adapte a una variedad de tareas diferentes.

4. `Capa de salida:` Finalmente, la última capa de la red transforma la representación de salida en una salida específica para la tarea. Por ejemplo, si la tarea es clasificar el sentimiento de una reseña de película, la capa de salida podría ser una capa de clasificación que produce una etiqueta de sentimiento (por ejemplo, positivo o negativo) a partir de la representación de la entrada.

Para una información más simple y amena de entender: [Las REDES NEURONALES ahora prestan ATENCIÓN](https://www.youtube.com/watch?v=aL-EmKuB078&ab_channel=DotCSV)

![6.png](imgs%2F2%20Primeros%20pasos%2F6.png)

Ampliamente Recomendable leer [Cómo funcionan los transformers](https://www.aprendemachinelearning.com/como-funcionan-los-transformers-espanol-nlp-gpt-bert/#:~:text=Los%20Transformers%20aparecieron%20como%20una,seq%20LSTM%20de%20aquel%20entonces)

![7.png](imgs%2F2%20Primeros%20pasos%2F7.png)


## 2.4 Aplicando transfer learning

Te recomiendo ampliamente leer el capítulo 5 [Introducción al aprendizaje por transferencia](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow#5-fundamentos-de-aprendizaje-por-transferencia) de mi curso anterior: [Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow#curso-profesional-de-redes-neuronales-con-tensorflow)
en dónde encontrarás información detallada del proceso de `transfer learning`, adicionalmente encontrarás códigos ejemplo 
de la implementación base en `TensorFlow` aquí el proceso se verá de forma más superficial, sin embargo, te comparto un breve
resumen y recordatorio de como funciona `transfer learning` y `fine tuning`

![8.png](imgs%2F2%20Primeros%20pasos%2F8.png)

Transfer learning y fine-tuning son técnicas populares en el campo del aprendizaje profundo que se utilizan para mejorar la capacidad de generalización de las redes neuronales y reducir el tiempo y el costo de entrenamiento.

- `Transfer Learning:` Esta técnica se basa en el uso de un modelo pre-entrenado en una tarea relacionada para inicializar la red neuronal y luego ajustarla a una tarea específica. El modelo pre-entrenado se entrena en una gran cantidad de datos y se ha demostrado que es eficaz en la extracción de características generales de la entrada. En lugar de entrenar la red desde cero en una tarea específica, se utiliza el modelo pre-entrenado como punto de partida y se ajusta para la tarea específica utilizando datos adicionales. La idea es que la red neuronal pre-entrenada ya haya aprendido algunas características generales útiles que se pueden reutilizar para la tarea específica. Esto ahorra tiempo y costo en el entrenamiento de la red neuronal.

- `Fine-tuning:` Es una técnica de ajuste fino que se utiliza después de la transferencia de aprendizaje. La red neuronal pre-entrenada se ajusta (fine-tune) en la tarea específica mediante el entrenamiento en datos adicionales. En lugar de ajustar la red neuronal completa, se ajustan sólo las últimas capas de la red neuronal para que se ajusten mejor a la tarea específica. Esto se hace porque las últimas capas de la red neuronal son las más especializadas en la tarea específica, mientras que las capas anteriores pueden ser más generales.

## 2.5 Quiz: Primeros pasos con transfer learning

![9.png](imgs%2F2%20Primeros%20pasos%2F9.png)


# 3 Computer Vision

Para este capítulo estaremos trabajando con una tarea de clasificación de imágenes. Nuestro problema será resolver el problema
del dataset [beans](https://huggingface.co/datasets/beans) el cual contiene imágenes de hojas de frijol y queremos clasificar
entre 3 tipos de clases:

```commandline
{
  "angular_leaf_spot": 0,
  "bean_rust": 1,
  "healthy": 2,
}
```

El dataset contiene 3 particiones: Train 1034 imágenes, Validation 133 imágenes, Test 128 imágenes.

![1.png](imgs%2F3%20Computer%20Vision%2F1.png)

Para ello usaremos `transfer learning`, usaremos como modelo base [google / vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)


>Vision Transformer (base-sized model)
Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224. It was introduced in the paper An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al. and first released in this repository. However, the weights were converted from the timm repository by Ross Wightman, who already converted the weights from JAX to PyTorch. Credits go to him.
>Disclaimer: The team releasing ViT did not write a model card for this model so this model card has been written by the Hugging Face team.

Para poder trabajar éxitosamente con `transfer learning` y `PyTorch` y `Transformers` vamos a necesitar definir los siguientes puntos:

- `model`=model,
- `args`=training_args,
- `data_collator`=collate_fn,
- `compute_metrics`=compute_metrics,
- `train_dataset`=prepared_ds.train,
- `eval_dataset`=prepared_ds.validation,
- `tokenizer`=feature_extractor

Cada uno de estos puntos los estaremos definiendo en las próximas clases, y cada clase tendrá su código especializado, adicionalmente
la clase 2 tendrá todo el código de la clase 1, pero solo explicación de la clase más reciente y así sucesivamente.

## 3.1 Carga de dataset para computer vision

El primer paso de nuestra tarea será descargar el dataset de `beans`, para ello necesitamos instalar `datasets` y `transformers` bibliotecas de 
`hugging face` que estaremos usando a lo largo de este mini proyecto.

> ## Nota:
> El código de esta sección lo puedes encontrar [Aquí](3%20Computer%20vision%2F1%20Carga%20de%20datasets%2Fcarga_datasets.py)

```bash
pip install datasets transformers
```

El primer paso de nuestro `journey` será descargar el dataset `beans` con ayuda de `datasets`

```python
from datasets import load_dataset

ds = load_dataset("beans")

print(ds)
```

La primera vez que ejecutes el código descargara todos los datos necesarios de las particiones de `train`, `validation`, `test`
```commandline
Downloading builder script: 100%|██████████| 3.61k/3.61k [00:00<00:00, 5.43MB/s]
Downloading metadata: 100%|██████████| 2.24k/2.24k [00:00<00:00, 1.44MB/s]
Downloading readme: 100%|██████████| 4.75k/4.75k [00:00<00:00, 7.66MB/s]
Downloading data: 100%|██████████| 144M/144M [00:16<00:00, 8.68MB/s]
Downloading data: 100%|██████████| 18.5M/18.5M [00:02<00:00, 6.33MB/s]
Downloading data: 100%|██████████| 17.7M/17.7M [00:02<00:00, 5.91MB/s]
Downloading data files: 100%|██████████| 3/3 [00:30<00:00, 10.14s/it]
Extracting data files: 100%|██████████| 3/3 [00:00<00:00,  7.72it/s]
100%|██████████| 3/3 [00:00<00:00, 1621.30it/s]
```
Respuesta esperada:
```commandline
DatasetDict({
    train: Dataset({
        features: ['image_file_path', 'image', 'labels'],
        num_rows: 1034
    })
    validation: Dataset({
        features: ['image_file_path', 'image', 'labels'],
        num_rows: 133
    })
    test: Dataset({
        features: ['image_file_path', 'image', 'labels'],
        num_rows: 128
    })
})
```

Conozcamos un poco las imágenes del `train set`, como ya sabemos nuestra variable `ds`es un diccionario, entonces podemos 
acceder a sus elementos de la misma forma que lo hacemos en `python` en este caso si queremos acceder al conjunto de `train`
basta con pedir el valor de la llave `train` -> `ds["train"]` ahora veamos un ejemplo en concreto:

```python
import matplotlib.pyplot as plt
# Mostremos un ejemplo:
ex = ds["train"][400]
print(ex)
test_image = ex["image"]
plt.imshow(test_image)
plt.savefig("ejemplo.png")
plt.close()
```
Respuesta esperada:
```commandline
{'image_file_path': '/home/ichcanziho/.cache/huggingface/datasets/downloads/extracted/e2e91becfe5d52af03524711c3b7bb9eb49cc8c6672e84d1f38bda746b2df8ef/train/bean_rust/bean_rust_train.148.jpg', 
'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F882B754F70>, 
'labels': 1}
```
Podemos observar como cada uno de los elementos del `train set` tiene 3 valores: 

- `image_file_path:` La dirección en memoria de donde se encuentra la imagen
- `image:` La imagen en formato `PIL`
- `labels:` La etiqueta de clasificación de la imagen

![ejemplo.png](3%20Computer%20vision%2F1%20Carga%20de%20datasets%2Fejemplo.png)

Sin embargo, a nosotros como seres humanos nos parece mucho más como y entendible leer a qué clase corresponde dicha hoja
y no únicamente el elemento numérico que lo representa, entonces primero, conozcamos cuales son las clases disponibles:

```python
labels = ds["train"].features["labels"]
print(labels)
```
Respuesta esperada:
```commandline
ClassLabel(names=['angular_leaf_spot', 'bean_rust', 'healthy'], id=None)
```
La variable `labels` cuanta con una función muy útil llamada `int2str` que a partir de un número regresa la etiqueta correspondiente:
```python
label_name = labels.int2str(ex["labels"])
print(label_name)
```
Respuesta esperada:
```commandline
bean_rust
```

## 3.2 Procesamiento de dataset para computer vision

Recordemos que estamos usando un modelo `pre-entrenado` es por ello que necesitamos hacer que nuestras imágenes para el 
`fine tuning` correspondan con las mismas caracterísitcas con las que fue implementado el modelo base, para ello necesitamos
implementar un `Feature Extractor` que nos ayude a hacer que las imágenes de nuestro problema correspondan con las necesidades
del modelo base. Para nuestra suerte, el modelo `google/vit-base-patch16-224-in21k` ya cuanta con un `Feature Extractor`
que podremos usar de la siguiente manera:

> ## Nota:
> El código de esta clase lo puedes encontrar [Aquí](3%20Computer%20vision%2F2%20Preprocesamiento%2FFeature%20Extractor.py)

En esta clase necesitamos tener instalado `torch` estás son las instrucciones: 

> Nota: recordemos que este repositorio fue creado con `linux` como sistema operativo.

```bash
conda install mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```
```bash
pip install torch
```

Hasta este momento, el código que aprendimos en la clase pasada fue el siguiente:

```python
from datasets import load_dataset

ds = load_dataset("beans")
ex = ds["train"][400]
test_image = ex["image"]

labels = ds["train"].features["labels"]
label_name = labels.int2str(ex["labels"])
```
Tener en cuenta esto, porque más adelante estaremos usando `ds` y `test_image`

Primero debemos obtener nuestro `Feature Extractor` para problemas de `Vision` es por ello que usaremos la clase
`ViFeatureExtractor`, después le diremos que lo tome desde un modelo `pre-entrenado` y vamos a observar sus características:

```python
from transformers import ViTFeatureExtractor

base_model_url = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(base_model_url)
print(feature_extractor)
```
Respuesta esperada:
```commandline
ViTFeatureExtractor {
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "ViTFeatureExtractor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 224,
    "width": 224
  }
}
```
Observemos que el proceso de `Feature Extractor` de este modelo se basa principalmente en normalizar las imágenes y re-escalarlas
a unas dimensiones de `224x224`.

Veamos cómo podríamos `limpiar` una sola imagen de nuestro set de `train`. El objetivo será pasarle una imagen de prueba y 
regresar la imagen ya normalizada con el `feature_extractor` en formato de `PyTorch - Tensor`

```python
feature_test = feature_extractor(test_image, return_tensors="pt")
print(feature_test)
print(feature_test.keys())
print(feature_test["pixel_values"].shape)
```
Respuesta esperada
```commandline
{'pixel_values': tensor([[[[ 0.7882,  0.6706,  0.7098,  ..., -0.1922, -0.1294, -0.1765],
          [ 0.7098,  0.6000,  0.6784,  ..., -0.2863, -0.1608, -0.1608],
          [ 0.4902,  0.3882,  0.4667,  ..., -0.1922, -0.0196,  0.0275],
          ...,
          [ 0.3804,  0.5294,  0.4824,  ..., -0.8275, -0.8196, -0.8039],
          [ 0.0902,  0.3725,  0.3804,  ..., -0.8667, -0.8431, -0.8510],
          [-0.0510,  0.2784,  0.3176,  ..., -0.8588, -0.8275, -0.8353]],

         [[ 0.4902,  0.3490,  0.3804,  ..., -0.6078, -0.5373, -0.5843],
          [ 0.3569,  0.2000,  0.3176,  ..., -0.7255, -0.6000, -0.5922],
          [ 0.0431, -0.0902,  0.0588,  ..., -0.6392, -0.4745, -0.4275],
          ...,
          [-0.2235, -0.0510, -0.0902,  ..., -0.9686, -0.9529, -0.9294],
          [-0.5059, -0.2078, -0.1922,  ..., -0.9922, -0.9922, -1.0000],
          [-0.6471, -0.2941, -0.2471,  ..., -0.9843, -0.9765, -0.9843]],

         [[ 0.4196,  0.2706,  0.3020,  ..., -0.7098, -0.6392, -0.6863],
          [ 0.2314,  0.0824,  0.2078,  ..., -0.8039, -0.6627, -0.6627],
          [-0.1137, -0.2314, -0.0824,  ..., -0.7020, -0.5373, -0.4980],
          ...,
          [-0.2784, -0.1373, -0.2000,  ..., -0.9529, -0.9529, -0.9451],
          [-0.6000, -0.3098, -0.3176,  ..., -0.9765, -0.9843, -0.9922],
          [-0.7569, -0.4118, -0.3804,  ..., -0.9765, -0.9686, -0.9686]]]])}
dict_keys(['pixel_values'])
torch.Size([1, 3, 224, 224])
```
Ahora podemos comprobar que nuestra `test_image` se ha transformado en un diccionario que tiene una sola key `pixel_values`
y que este contiene un `tensor` con unas dimensiones de `[1, 3, 224, 224]` lo cual significa que es una sola imagen de 224x224 en formato `RGB`

Sin embargo, nosotros sabemos que para entrenar a un modelo de Deep Learning sin importar cuál sea la biblioteca que estemos
usando, necesitamos al menos 2 ingredientes, imágenes y etiquetas. En este momento ya tenemos una forma de transformar él `dataset` a un formato
entendible por el modelo de nuestra `red neuronal` sin embargo, debemos agregar a cuál clase pertenece cada imagen, esto es
sumamente sencillo si tenemos en cuenta que las imágenes pre-procesadas con `feature_extractor` en realidad son diccionarios, esto
nos permite simplemente añadirles una nueva `key` llamada `label`y ponerle su valor correspondiente, esto se hace de la siguiente manera:

La siguiente función recibe un `example` de alguna de nuestras particiones que recordemos que a su vez son diccionarios que contienen:
[image_file_path, image, labels], sin embargo, ahora solo nos interesa enfocarnos en las últimas dos claves. Esta funcion se
encarga de recibir esta muestra y transformarla con `feature_extractor` y adjuntarle su `example["labels"]`
```python
def process_example(example):
    inputs = feature_extractor(example["image"], return_tensors="pt")
    inputs["labels"] = example["labels"]
    return inputs
```

Veamos un ejemplo, primero seleccionemos una muestra, digamos el `train` número 10:

```python
preprocess_test = ds["train"][10]
print(preprocess_test)
```
Respuesta esperada
```commandline
{'image_file_path': '/home/ichcanziho/.cache/huggingface/datasets/downloads/extracted/e2e91becfe5d52af03524711c3b7bb9eb49cc8c6672e84d1f38bda746b2df8ef/train/angular_leaf_spot/angular_leaf_spot_train.107.jpg', 'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x7F035C2589A0>, 'labels': 0}
{'pixel_values': tensor([[[[ 0.0275, -0.0196,  0.0196,  ...,  0.0275,  0.2157, -0.0039],
          [-0.0353,  0.0902, -0.0275,  ...,  0.2000,  0.3569,  0.1922],
          [-0.0902,  0.1216,  0.0353,  ...,  0.1686,  0.3333,  0.3098],
          ...,
          [-0.0353, -0.2784, -0.4353,  ..., -0.2863, -0.1843, -0.3882],
          [ 0.2157, -0.1765, -0.3804,  ..., -0.1294, -0.1608, -0.3647],
          [ 0.3020, -0.0275, -0.3255,  ..., -0.2471, -0.1686, -0.3333]],

         [[-0.3412, -0.3569, -0.2627,  ..., -0.3176, -0.1373, -0.3412],
          [-0.4039, -0.2549, -0.3255,  ..., -0.1373, -0.0118, -0.1843],
          [-0.4745, -0.2314, -0.2863,  ..., -0.1922, -0.0275, -0.0510],
          ...,
          [ 0.2000,  0.0431, -0.0196,  ..., -0.5373, -0.3569, -0.5216],
          [ 0.3412,  0.0745,  0.0118,  ..., -0.4039, -0.3490, -0.4980],
          [ 0.3725,  0.1922,  0.0588,  ..., -0.5216, -0.3647, -0.4745]],

         [[-0.5059, -0.5294, -0.4824,  ..., -0.4431, -0.2784, -0.4745],
          [-0.5686, -0.4431, -0.5294,  ..., -0.3333, -0.1922, -0.3490],
          [-0.6314, -0.4196, -0.4588,  ..., -0.3647, -0.2157, -0.2314],
          ...,
          [-0.5686, -0.7176, -0.7333,  ..., -0.6941, -0.5137, -0.6235],
          [-0.3020, -0.6314, -0.6863,  ..., -0.5765, -0.5137, -0.6078],
          [-0.1843, -0.4667, -0.6235,  ..., -0.7176, -0.5373, -0.5922]]]]), 
          'labels': 0}
```
Excelente, nosotros ya esperábamos ver un formato de 3 `keys`, pero ahora vamos a transformar esta imagen en un formato útil:

```python
preprocess_test = process_example(preprocess_test)
print(preprocess_test)
print(preprocess_test["pixel_values"].shape)
```
```commandline
{'pixel_values': tensor([[[[ 0.0275, -0.0196,  0.0196,  ...,  0.0275,  0.2157, -0.0039],
          [-0.0353,  0.0902, -0.0275,  ...,  0.2000,  0.3569,  0.1922],
          [-0.0902,  0.1216,  0.0353,  ...,  0.1686,  0.3333,  0.3098],
          ...,
          [-0.0353, -0.2784, -0.4353,  ..., -0.2863, -0.1843, -0.3882],
          [ 0.2157, -0.1765, -0.3804,  ..., -0.1294, -0.1608, -0.3647],
          [ 0.3020, -0.0275, -0.3255,  ..., -0.2471, -0.1686, -0.3333]],

         [[-0.3412, -0.3569, -0.2627,  ..., -0.3176, -0.1373, -0.3412],
          [-0.4039, -0.2549, -0.3255,  ..., -0.1373, -0.0118, -0.1843],
          [-0.4745, -0.2314, -0.2863,  ..., -0.1922, -0.0275, -0.0510],
          ...,
          [ 0.2000,  0.0431, -0.0196,  ..., -0.5373, -0.3569, -0.5216],
          [ 0.3412,  0.0745,  0.0118,  ..., -0.4039, -0.3490, -0.4980],
          [ 0.3725,  0.1922,  0.0588,  ..., -0.5216, -0.3647, -0.4745]],

         [[-0.5059, -0.5294, -0.4824,  ..., -0.4431, -0.2784, -0.4745],
          [-0.5686, -0.4431, -0.5294,  ..., -0.3333, -0.1922, -0.3490],
          [-0.6314, -0.4196, -0.4588,  ..., -0.3647, -0.2157, -0.2314],
          ...,
          [-0.5686, -0.7176, -0.7333,  ..., -0.6941, -0.5137, -0.6235],
          [-0.3020, -0.6314, -0.6863,  ..., -0.5765, -0.5137, -0.6078],
          [-0.1843, -0.4667, -0.6235,  ..., -0.7176, -0.5373, -0.5922]]]]), 'labels': 0}
torch.Size([1, 3, 224, 224])
```
Excelente ahora podemos ver como nuestro ejemplo ya es un dato útil, un diccionario que contiene dos llaves: `pixel_values` y `labels`
y no solo eso, la imagen ya está en formato de `tensor` y tiene un `shape`de `[1, 3, 224, 224]`, hasta este punto hemos logrado
exitosamente transformar un ejemplo del `train` en un dato listo para ser entrenado por una red neuronal.

Sin embargo, nos gustaría tener una función que NO solo funcione para una unica imagen sino para un `lote` de imágenes, eso es
sumamente sencillo, simplemente se trata de repetir el proceso para una lista de imágenes, creemos la función `transform` 
que recibirá un `batch` de ejemplos en lugar de uno solo.

```python
def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs
```
Y ahora podemos usar un método sumamente útil de los `datasets` de `hugging face` llamado: `wit_transform(function)`que recibe
como parámetro de entrada una función de transformación, en este caso la que acabamos de crear.
```python
print("batch")
clean = ds.with_transform(transform)
print(clean["train"][0:2])
print(clean["train"][0:2]["pixel_values"].shape)

```
Respuesta esperada:
```commandline
batch
{'pixel_values': tensor([[[[-0.5686, -0.5686, -0.5608,  ..., -0.0275,  0.1843, -0.2471],
          [-0.6078, -0.6000, -0.5765,  ..., -0.0353, -0.0196, -0.2627],
          [-0.6314, -0.6314, -0.6078,  ..., -0.2314, -0.3647, -0.2235],
          ...,
          [-0.5373, -0.5529, -0.5843,  ..., -0.0824, -0.0431, -0.0902],
          [-0.5608, -0.5765, -0.5843,  ...,  0.3098,  0.1843,  0.1294],
          [-0.5843, -0.5922, -0.6078,  ...,  0.2627,  0.1608,  0.2000]],

         [[-0.7098, -0.7098, -0.7490,  ..., -0.3725, -0.1608, -0.6000],
          [-0.7333, -0.7333, -0.7569,  ..., -0.3647, -0.3255, -0.5686],
          [-0.7490, -0.7490, -0.7725,  ..., -0.5373, -0.6549, -0.5373],
          ...,
          [-0.7725, -0.7804, -0.8196,  ..., -0.2235, -0.0353,  0.0824],
          [-0.7961, -0.8118, -0.8118,  ...,  0.1922,  0.3098,  0.3725],
          [-0.8196, -0.8196, -0.8275,  ...,  0.0824,  0.2784,  0.3961]],

         [[-0.9922, -0.9922, -1.0000,  ..., -0.5451, -0.3569, -0.7255],
          [-0.9922, -0.9922, -1.0000,  ..., -0.5529, -0.5216, -0.7176],
          [-0.9843, -0.9922, -1.0000,  ..., -0.6549, -0.7569, -0.6392],
          ...,
          [-0.8431, -0.8588, -0.8980,  ..., -0.5765, -0.5529, -0.5451],
          [-0.8588, -0.8902, -0.9059,  ..., -0.2000, -0.2392, -0.2627],
          [-0.8824, -0.9059, -0.9216,  ..., -0.2549, -0.2000, -0.1216]]],


        [[[-0.5137, -0.4902, -0.4196,  ..., -0.0275, -0.0039, -0.2157],
          [-0.4902, -0.4667, -0.4196,  ..., -0.0588, -0.0118, -0.2392],
          [-0.4588, -0.4902, -0.5137,  ..., -0.0745,  0.0039, -0.3020],
          ...,
          [-0.4980, -0.4980, -0.5294,  ..., -0.2000, -0.2157, -0.3882],
          [-0.5451, -0.5294, -0.5216,  ..., -0.1922, -0.1922, -0.3882],
          [-0.5216, -0.5373, -0.5451,  ..., -0.1294, -0.1529, -0.2627]],

         [[-0.1843, -0.2000, -0.1529,  ...,  0.2157,  0.2078, -0.0902],
          [-0.1686, -0.1686, -0.1529,  ...,  0.1922,  0.2235, -0.0902],
          [-0.1529, -0.2000, -0.2392,  ...,  0.1686,  0.2549, -0.1294],
          ...,
          [-0.7725, -0.7569, -0.7569,  ..., -0.4196, -0.4588, -0.6471],
          [-0.7961, -0.7804, -0.7647,  ..., -0.4196, -0.4510, -0.6627],
          [-0.7725, -0.7961, -0.8039,  ..., -0.3725, -0.4196, -0.5451]],

         [[-0.7569, -0.8510, -0.8353,  ..., -0.3255, -0.2706, -0.5608],
          [-0.7804, -0.8431, -0.8118,  ..., -0.3490, -0.2706, -0.5608],
          [-0.8118, -0.8588, -0.8510,  ..., -0.3647, -0.2314, -0.5373],
          ...,
          [-0.5216, -0.5137, -0.5294,  ..., -0.2471, -0.2627, -0.4431],
          [-0.5529, -0.5373, -0.5216,  ..., -0.2235, -0.2235, -0.4431],
          [-0.5294, -0.5529, -0.5608,  ..., -0.1686, -0.1922, -0.3333]]]]), 'labels': [0, 0]}
torch.Size([2, 3, 224, 224])
```
Excelente, hemos logrado convertir nuestras imágenes del dataset de forma éxitosa en un formato listo para usar en Deep Learning
pero como último paso y cereza del pastel vamos a crear una función que reciba justamente este diccionario y cree un único `tensor`
uno después de otro.

Los data collators, o recopiladores de datos, son objetos que forman batches utilizando una lista de ejemplos de nuestros datasets. Para poder generar los batches, los data collators pueden aplicar algún procesamiento (como padding en los ejemplos con texto).

Definimos una función, collate_fn, que fungirá como nuestro data collator. Devolverá un diccionario por cada batch. Recibirá un batch de datos que luego serán procesadas.

Los batches llegan como listas de dicts. Cada dict tiene los label y pixel_values de sus respectivos ejemplos, por lo que puedes simplemente desempaquetarlos y apilarlos en tensores de batches. torch.stack nos permite concatenar (pegar) tensores.


```python
import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }
```

Excelente, hemos creado perfectamente nuestra función `collate` que usaremos más adelante para entrenar a nuestro modelo.

## 3.3 Configurando un Trainer para computer vision

Si has llegado hasta aquí con un entendimiento de cada parte del código entonces estás listo para continuar con la creación
del entrenador del modelo.

> ## Nota:
> El código de esta clase lo puedes encontrar [Aquí](3%20Computer%20vision%2F3%20Configurando%20Trainer%2Fconfigurando_trainer.py)

Hasta este momento nuestro código limpio y funcional, es el siguiente:

```python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from transformers import ViTFeatureExtractor
import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs


if __name__ == '__main__':
    ds = load_dataset("beans")

    labels = ds["train"].features["labels"].names
    print(labels)

    base_model_url = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(base_model_url)
    clean = ds.with_transform(transform)

```

Asi que continuemos con los demás puntos de esta clase:

En esta clase nuestro objetivo es definir el resto de los argumentos necesarios para Trainer.

Y debemos comenzar con definir la métrica de evaluación que usaremos para nuestro modelo:

```python
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(prediction):
    return metric.compute(predictions=np.argmax(prediction.predictions, axis=1), references=prediction.label_ids)
```

Esto parece un poco más engorroso de lo que realmente es, pero si prestamos atención sigue siendo como se computa cualquier otra
métrica de ML como en scikit-learn, básicamente `load_metric` tiene varias métricas disponibles entre ellas se encuentra:
```commandline
['accuracy',
 'bertscore',
 'bleu',
 'bleurt',
 'brier_score',
 'cer',
 'character',
 ...
 ition_math',
 ypchang/sklearn_proxy',
 'yulong-me/yl_metric',
 'yzha/ctc_eval',
 'zbeloki/m2']
```

Para conocer toda la lista de métricas puedes ejecutar:

```python
import datasets
datasets.list_metrics()
```

Puedes conocer más información en: https://huggingface.co/docs/datasets/how_to_metrics

La función `metric.compute(predictions, references)` recibe 2 parámetros, el primero es el valor de las predicciones, que en nuestro caso
las predicciones serán una lista de probabilidades, por eso usamos `np.argmax` para obtener el resultado más probable, en este caso nuestra función:
`compute_metrics` recibe una `prediction` que cuenta con 2 valores `prediction.prediction` y `prediction.label_ids` 

### Configurando `Trainer`

Carguemos el modelo pre-entrenado. Agregaremos `num_labels` para que el modelo cree un encabezado de clasificación con el número correcto de etiquetas. También incluiremos las asignaciones `id2label` y `label2id` para tener etiquetas legibles por humanos en el widget del Hub.

```python
from transformers import ViTForImageClassification

labels = ds["train"].features["labels"].names

model = ViTForImageClassification.from_pretrained(
    base_model_url,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i,c in enumerate(labels)}
)
```
Hasta este momento, solo habíamos usado él `base_model_url` para obtener su `feature extractor` pero NO habiamos descargado
el modelo en sí mismo, esto lo logramos con: `ViForImageClassification.from_pretrained`

Respuesta esperada:
```commandline
Downloading (…)lve/main/config.json: 100%|██████████| 502/502 [00:00<00:00, 441kB/s]
Downloading pytorch_model.bin: 100%|██████████| 346M/346M [00:56<00:00, 6.09MB/s]
Some weights of the model checkpoint at google/vit-base-patch16-224-in21k were not used when initializing ViTForImageClassification: ['pooler.dense.bias', 'pooler.dense.weight']
- This IS expected if you are initializing ViTForImageClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing ViTForImageClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```
Excelente, hemos logrado descargar exitosamente el modelo y ya hemos cambiado la última capa del mismo por una capa de 3 salidas
correspondientes a nuestro número de `labels` y el propio modelo nos dice que se ha añadido una nueva última capa y que esta 
debería ser `TRAIN` para ser capaz de ser usada en predictions e inferencias.

Lo último que se necesita antes de eso es establecer la configuración de entrenamiento definiendo TrainingArguments.

Pero antes de eso vamos a hacer `login` en `hugging face`

```bash
huggingface-cli login
```
Respuesta esperada:
```commandline
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    
    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token: 
```

Nos pedirá un token de autentificación, debemos entrar al link que nos ofrece: https://huggingface.co/settings/tokens y copiar el
token que nos generó automáticamente.

![token.png](3%20Computer%20vision%2F3%20Configurando%20Trainer%2Ftoken.png)

La mayoría de estos se explican por sí mismos, pero uno que es bastante importante aquí es remove_unused_columns=False. Este eliminará cualquier función que no utilice la función de llamada del modelo. De forma predeterminada es True porque, por lo general, es ideal eliminar las columnas de funciones no utilizadas, lo que facilita el desempaquetado de las entradas en la función de llamada del modelo. Pero, en nuestro caso, necesitamos las funciones no utilizadas ('imagen' en particular) para crear 'pixel_values'.

```python
from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./platzi-vit-model-gabriel-ichcanziho",
        evaluation_strategy="steps",
        num_train_epochs=4,
        push_to_hub_organization="platzi",
        learning_rate=2e-4,
        remove_unused_columns=False,
        push_to_hub=True,
        load_best_model_at_end=True,
    )
```

Finalmente, definamos nuestro `Trainer`

```python
from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=clean["train"],
        eval_dataset=clean["validation"],
        tokenizer=feature_extractor
    )
```

Excelente, hasta este momento ya hemos definido TODO lo necesario para empezar a hacer `fine-tuning` y entrenar a nuestro modelo
con base en el modelo de Google pero adaptándolo a nuestras necesidades. Ha sido un proceso un poco engorroso, puesto que en 
`TensorFlow` estos pasos al ser en una capa de abstracción más baja se tiene mayor control sobre lo que se hace y se siente más
cómo definir las cosas en lugar de estar usando funciones que ya existen para todo; sin embargo, el proceso ha sido bastante similar hasta este momento.


## 3.4 Entrenamiento y evaluación de modelo de computer vision

Estamos a punto de acabar con el tutorial, ya solo es necesario poner a entrenar nuestro modelo, guardar el mejor resultado
y finalmente evaluar los resultados con el dataset de `test`.

> ## Nota:
> El código completo de esta sección lo puedes encontrar [Aquí](3%20Computer%20vision%2F4%20Entrenamiento%20y%20Evaluaci%C3%B3n%2Fentrenamiento.py)

Antes de continuar con la última parte de este tutorial, veamos todo el código limpio que hemos creado hasta el momento:

```python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from datasets import load_metric
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import torch


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs


def compute_metrics(prediction):
    return metric.compute(predictions=np.argmax(prediction.predictions, axis=1), references=prediction.label_ids)


if __name__ == '__main__':
    ds = load_dataset("beans")

    labels = ds["train"].features["labels"].names

    base_model_url = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTFeatureExtractor.from_pretrained(base_model_url)
    clean = ds.with_transform(transform)

    metric = load_metric("accuracy")

    model = ViTForImageClassification.from_pretrained(
        base_model_url,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    training_args = TrainingArguments(
        output_dir="./platzi-vit-model-gabriel-ichcanziho",
        evaluation_strategy="steps",
        num_train_epochs=4,
        push_to_hub_organization="platzi",
        learning_rate=2e-4,
        remove_unused_columns=False,
        push_to_hub=True,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=clean["train"],
        eval_dataset=clean["validation"],
        tokenizer=feature_extractor
    )

```

Podemos observar una secuencia muy lógica entre todas las funciones que hemos creado, y vemos como todas las bibliotecas que hemos
importado tienen una función muy en concreto dentro de nuestro código, ahora para terminar este proceso solo falta empezar a entrenar
el modelo y evaluar los resultados finales:

```python
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
```
Resultado esperado:
```commandline
{'loss': 0.1373, 'learning_rate': 7.692307692307694e-06, 'epoch': 3.85}
{'eval_loss': 0.014603369869291782, 'eval_accuracy': 0.9924812030075187, 'eval_runtime': 1.5139, 'eval_samples_per_second': 87.852, 'eval_steps_per_second': 11.229, 'epoch': 3.85}
{'train_runtime': 130.2439, 'train_samples_per_second': 31.756, 'train_steps_per_second': 3.993, 'train_loss': 0.13224817431316926, 'epoch': 4.0}
***** train metrics *****
  epoch                    =        4.0
  train_loss               =     0.1322
  train_runtime            = 0:02:10.24
  train_samples_per_second =     31.756
  train_steps_per_second   =      3.993
```

Evaluamos en nuestro `split` de `test`:

```python
metrics = trainer.evaluate(clean["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```
Resultado esperado:
```commandline
***** eval metrics *****
  epoch                   =        4.0
  eval_accuracy           =     0.9609
  eval_loss               =     0.1104
  eval_runtime            = 0:00:01.46
  eval_samples_per_second =      87.12
  eval_steps_per_second   =      10.89
```

Felicidades, nuestro modelo ha logrado un 96% De Accuracy en predecir entre las 3 categorías de hojas de frijol.

Puedes encontrar nuestro modelo en el Hub de Hugging Face en:

https://huggingface.co/platzi/platzi-vit-model-gabriel-ichcanziho

![2.png](imgs%2F3%20Computer%20Vision%2F2.png)


## 3.5 Quiz: Computer vision con Hugging Face

![quizz.png](imgs%2F3%20Computer%20Vision%2Fquizz.png)

# 4 Natural Language Processing

En lineamientos generales este proyecto es sumamente similar al proyecto pasado, pero enfocado a trabajar con texto. Este
mini proyecto estará dividido en las mismas 4 secciones que el proyecto anterior.

Vamos a trabajar con el dataset [glue](https://huggingface.co/datasets/glue) Expecificamente con el subset `MRPC`.

![1.png](imgs%2F4%20Natural%20Language%20Processing%20%2F1.png)

El tema de `MRPC` es dada dos oraciones `sentence1, sentence2` etiquetar si ambos textos son equivalentes o no lo son.
Cómo `Feature_Extractor` y `Pretrained Model` vamos a usar [distilroberta-base](https://huggingface.co/distilroberta-base) el cual es un modelo
más pequeño y rápido que un modelo `BERT` convencional.



## 4.1 Carga de datasets para NLP

Como primer paso debemos descargar el dataset que vamos a utilizar para este proyecto:

> ## Nota:
> El códido completo de esta clase lo puedes encontrar [Aquí](3%20Computer%20vision%2F1%20Carga%20de%20datasets%2Fcarga_datasets.py)

```python
from datasets import load_dataset

ds = load_dataset("glue", "mrpc")
```
La primera vez que corramos el código anterior va a descargar el dataset:
```commandline
Downloading builder script: 100%|██████████| 28.8k/28.8k [00:00<00:00, 319kB/s]
Downloading metadata: 100%|██████████| 28.7k/28.7k [00:00<00:00, 312kB/s]
Downloading readme: 100%|██████████| 27.9k/27.9k [00:00<00:00, 280kB/s]
Downloading and preparing dataset glue/mrpc to /home/ichcanziho/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...
...
Downloading data files: 100%|██████████| 3/3 [00:02<00:00,  1.24it/s]
100%|██████████| 3/3 [00:00<00:00, 1703.85it/s]
Dataset glue downloaded and prepared to /home/ichcanziho/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.
```
Una vez descargado el dataset podemos observar que su estructura es la siguiente:
```python
print(ds)
```
Respuesta esperada:
```commandline
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```

Nuestro dataset contiene `3668` ejemplos de entrenamiento, `408`ejemplos de validación y `1725` ejemplos de prueba.
Cada uno de ellos contiene 2 oraciones una etiqueta y un índice.

Mostremos un ejemplo del subset de `train`

```python
# Mostremos un ejemplo:
ejemplo = ds["train"][400]
print(ejemplo)
```
Respuesta esperada:
```commandline
{'sentence1': 'U.S. Agriculture Secretary Ann Veneman , who announced Tuesdays ban , also said Washington would send a technical team to Canada to help .', 
'sentence2': "U.S. Agriculture Secretary Ann Veneman , who announced yesterday 's ban , also said Washington would send a technical team to Canada to assist in the Canadian situation .", 
'label': 1, 
'idx': 446}
```

Recordemos que a nosotros como humanos nos gusta leer etiquetas con un nombre y no con un numero, entonces conozcamos cuáles son los nombres de las `labels`:

```python
labels = ds["train"].features["label"]
print(labels)
```
Respuesta esperada:
```commandline
ClassLabel(names=['not_equivalent', 'equivalent'], id=None)
```
Excelente, ya sabemos que los datos pueden ser `not_equivalent` o `equivalent`, afortunadamente, el objeto `labels` cuenta con un
simple método `int2str` que convierte un número a su representación de texto, entonces nuestro ejemplo que tiene un label `1` podemos acceder
al nombre de la etiqueta de la siguiente manera:

```python
# Nombre de la etiqueta
label_name = labels.int2str(ejemplo["label"])
print(label_name)
```
Respuesta esperada:
```commandline
equivalent
```

Perfecto, hasta este momento ya estamos familiarizados con el dataset que estaremos utilizando y vemos como hasta este punto
el proyecto es sumamente parecido al proyecto pasado, continuemos en la siguiente clase.

## 4.2 Procesamiento de dataset para NLP

Al igual que en el ejemplo de `Computer Vision`, se debe hacer un preprocesamiento para trabajar con el texto de entrada de modo
que tenga la forma correcta que está buscando nuestro modelo `pre-entrenado` específicamente en `NLP` El texto NO puede ser
clasificado directamente, debe ser convertido en una representación numerica para poder trabajar con ellos. A este proceso de 
transformación se le conoce como: `Word Embeddings` Para conocer más acerca de este tema te recomiendo ver el siguiente video:
[Intro al NLP ¿Qué es un EMBEDDING?](https://www.youtube.com/watch?v=RkYuH_K7Fx4&ab_channel=DotCSV). Ya hemos trabajado con
este concepto en otros cursos anteriores de esta ruta de `Deep Learning` si quieres leer un poco más al respecto te recomiendo ver:
[Variables Estadísticas](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales#2-normalizado-de-datos)

> ## Nota:
> El código de esta clase lo puedes encontrar [Aquí](4%20NLP%2F2%20Preprocesamiento%2FFeature%20Extractor%20NLP.py)

En general el objetivo de este paso es pre-procesar los datos de texto de nuestro dataset en una representación vectorial que sea compatible con
el modelo que vas a usar. Para ello vamos a usar el propio `Tokenizer` del modelo que vamos a utilizar en este proyecto:
Primero, tengamos en cuenta que vamos a usar el modelo `distilroberta-base`, vamos a utilizar `AutoTokenizer` para obtener el
tokenizador correspondiente a nuestro modelo:

```python
from transformers import AutoTokenizer

base_model_url = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_url)

ejemplo = ds["train"][400]["sentence1"]
print(ejemplo)
print(tokenizer(ds["train"][400]["sentence1"]))
```
Respuesta esperada:
```commandline
U.S. Agriculture Secretary Ann Veneman , who announced Tuesdays ban , also said Washington would send a technical team to Canada to help .
{'input_ids': [0, 791, 4, 104, 4, 8004, 1863, 3921, 10336, 5649, 2156, 54, 585, 20423, 7033, 2020, 2156, 67, 26, 663, 74, 2142, 10, 3165, 165, 7, 896, 7, 244, 479, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

**¿Qué significa cada uno de los valores que nos retorna el tokenizador?**

- `input_ids` es la traducción de palabras a números.
- `attention_mask` es un tensor con la misma forma que input_ids, pero lleno de 0 y 1: los 1 indican que se debe atender a los tokens correspondientes y los 0 indican que no se deben atender. Es decir, deben ser ignorados por el modelo.
- `token_type_ids` dice al modelo qué parte de la entrada es la primera oración y cuál es la segunda oración.

Existen otros modelos más complejos como: `"bert-base-uncased"` cuyo tokenizador contiene un método especializado en transformar
de regreso los `inputs_ids` en el texto original: `tokenizer.convert_ids_to_tokens(inputs["input_ids"])`En este caso, nuestro modelo
ligero NO contiene dicho método, pero es bueno tener en cuenta que otros modelos sí tienen esta capacidad.

**Creemos una función Tokenizadora:**

```python
def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
```
Afortunadamente, nuestro objeto `tokenizer` permite recibir multiples `parámetros` separados por `,` esto me permite transformar
nuestras 2 oraciones de ejemplo al mismo tiempo. El parámetro `truncatoion = True` significa que oraciones sumamente largas serán
recortadas a unas más pequeñas de un tamaño predefinido por el modelo.

Finalmente, vamos a `limpiar` todo nuestro dataset, a diferencia del dataset de imágenes que ocupábamos el método `ds.with_transform(transform)`
y utilizábamos una función `transform` para modificar el dataset, como específicamente podemos transformar nuestros datos de entrada a su versión
tokenizada con `tokenize_fn` podemos usar el método `map` para mapear los parámetros de `sentence1` y `sentence2` de cada muestra del dataset

Veamos un ejemplo del dataset ya limpio
```python
clean_ds = ds.map(tokenize_fn, batched=True)
print(clean_ds["train"][400])
```
Respuesta esperada:
```commandline
{
'sentence1': 'U.S. Agriculture Secretary Ann Veneman , who announced Tuesdays ban , also said Washington would send a technical team to Canada to help .', 
'sentence2': "U.S. Agriculture Secretary Ann Veneman , who announced yesterday 's ban , also said Washington would send a technical team to Canada to assist in the Canadian situation .", 
'label': 1, 
'idx': 446, 
'input_ids': [0, 791, 4, 104, 4, 8004, 1863, 3921, 10336, 5649, 2156, 54, 585, 20423, 7033, 2020, 2156, 67, 26, 663, 74, 2142, 10, 3165, 165, 7, 896, 7, 244, 479, 2, 2, 791, 4, 104, 4, 8004, 1863, 3921, 10336, 5649, 2156, 54, 585, 2350, 128, 29, 2020, 2156, 67, 26, 663, 74, 2142, 10, 3165, 165, 7, 896, 7, 3991, 11, 5, 1563, 1068, 479, 2], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```
Excelente, ahora nuestro diccionario de una muestra de entrenamiento contiene `6 keys` las 4 originales y 2 adicionales, `input_ids` y `attention_mask  

Finalmente, a pesar de que él `dataset` ya está `limpio` al igual que en el ejemplo de `Computer Vision` necesitamos definir un `Data Collactor`

### Definiendo el data collator: Dynamic padding

Necesitamos que nuestros tensores tengan una forma rectangular. Es decir que tengan el mismo tamaño cada uno de los ejemplos. Sin embargo, los textos no necesariamente tienen el mismo tamaño.

Para ello usamos el relleno o padding. El padding se asegura de que todas nuestras oraciones tengan la misma longitud al agregar una palabra especial llamada padding token a las oraciones con menos valores. Por ejemplo, si tenemos 10 oraciones con 10 palabras y 1 oración con 20 palabras, el relleno garantizará que todas las oraciones tengan 20 palabras.

Dejamos el argumento de padding del tokenizer vacío en nuestra función de tokenización por ahora. Esto se debe a que rellenar (hacer padding) todas las muestras hasta la longitud máxima del dataset no es eficiente, es mejor rellenar las muestras cuando estamos construyendo un batch, ya que entonces solo necesitamos rellenar hasta la longitud máxima en ese batch, y no la longitud máxima en todo el dataset. ¡Esto puede ahorrar mucho tiempo y potencia de procesamiento cuando las entradas tienen longitudes muy variables!

Usaremos un DataCollator para esto. Rellenemos (hagamos padding) todos los ejemplos con la longitud del elemento más largo del batch. A esta técnica se le conoce como relleno dinámico o dynamic padding.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(data_collator)
```
Respuesta esperada:

```commandline
DataCollatorWithPadding(tokenizer=RobertaTokenizerFast(name_or_path='distilroberta-base', vocab_size=50265, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)}), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
```
Perfecto, ya hemos definido exitosamente a nuestro `DataCollator` esto nos permite definir a nuestro `Trainer` pero esto lo veremos en la siguiente clase.


## 4.3 Configurando un Trainer para NLP

Hasta este momento el código completo y limpio es el siguiente:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


if __name__ == '__main__':
    # Descarga de dataset
    ds = load_dataset("glue", "mrpc")
    labels = ds["train"].features["label"].names
    # Descarga de tokenizador de DistilRoberta
    base_model_url = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_url)
    clean_ds = ds.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
En esta clase vamos a definir las funciones que nos faltan para poder crear a nuestro `Trainer`

> ## Nota:
> El código completo de esta sección lo puedes encontrar [Aquí](4%20NLP%2F3%20Configurando%20Trainer%2Fconfigurando%20trainer%20nlp.py)

Vamos a empezar por definir nuestra métrica de éxito para ello vamos a instalar una nueva biblioteca:

```bash
pip install evaluate
```

Esta biblioteca nos va a ayudar con el mismo objetivo que lo hizo `from datasets import load_metric` pero en este caso, la métrica que vamos
a utilizar va a estar definida por el propio dataset, en el curso no queda muy claro porque usar está metodología en lugar de la habitual.

En general observamos que la estructura de la función `compute_metrics` es sumamente similar a la que ya conociamos, solo que en lugar de usar `load_metric`
estamos usando `evaluate.load()`, en general me pareció más entendible la forma pasada.

```python
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("glue", "mrpc")
    logits, labels_ = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels_)
```
A continuación vamos a crear a nuestro modelo desde un `pretrained` como el modelo es una clasificación de texto, lo debemos
importar usando `AutoModelForSquencueClassification`, es importante conocer `transformers` para conocer todas las clases que
podemos utilizar para cargar modelos pre-entrenados, pero en este curso ya vimos las 2 más populares.

> ## NOTA:
> Más adelante cuando hayamos terminado el entrenado del modelo, veremos cuáles son las métricas, y observaremos que el método de evaluate.load() cargo
> dos métricas ACC y F1 score, algo sumamente interesante y curioso que vale la pena poner como nota.

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    base_model_url,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
```

Ahora podemos definir nuestros argumentos de entrenamiento:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./platzi-distilroberta-base-mrpc-glue-gabriel-ichcanziho",
    evaluation_strategy="steps",
    num_train_epochs=4,
    push_to_hub_organization="platzi",
    push_to_hub=True,
    load_best_model_at_end=True
)
```
Esto fue sumamente parecido al problema de clasificación de imágenes, pero noto que aquí NO pusimos `learning_rate=2e-4,
        remove_unused_columns=False,` en la clase no se explica el porqué de este cambio.

Y para terminar con esta clase, vamos a definir a nuestro `trainer`:

```python
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=clean_ds["train"],
    eval_dataset=clean_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

Felicidades, ya tenemos TODO listo para empezar a entrenar a nuestro modelo, pero esto lo haremos en la siguiente clase.


## 4.4 Entrenamiento y evaluación de modelo de NLP

Es importante que antes de proceder a evaluar nuestro modelo, hagamos logging a `hugging face`:

```bash
huggingface-cli login
```
Respuesta esperada:
```commandline
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    
    A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
    Setting a new token will erase the existing one.
    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token: 
Add token as git credential? (Y/n) y
Token is valid.
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/ichcanziho/.cache/huggingface/token
Login successful

```

> ## Nota:
> El código completo de esta sección lo puedes encontrar [Aquí](4%20NLP%2F4%20Entrenamiento%20y%20Evaluaci%C3%B3n%2Fentrenamiento%20nlp.py)

Hasta este momento nuestro código completo y limpio es el siguiente:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import numpy as np


def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_pred):
    metric = evaluate.load("glue", "mrpc")
    logits, labels_ = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels_)


if __name__ == '__main__':

    # 1. Descarga de dataset
    ds = load_dataset("glue", "mrpc")
    labels = ds["train"].features["label"].names

    # 2. Descarga de tokenizador de DistilRoberta
    base_model_url = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_url)

    # 3. Limpiamos la base de datos original
    clean_ds = ds.map(tokenize_fn, batched=True)

    # 4. Creamos nuestro DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Definimos el modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_url,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
    )

    # 6. Definimos los argumentos de entrenamiento del modelo
    training_args = TrainingArguments(
        output_dir="./platzi-distilroberta-base-mrpc-glue-gabriel-ichcanziho",
        evaluation_strategy="steps",
        num_train_epochs=4,
        push_to_hub_organization="platzi",
        push_to_hub=True,
        load_best_model_at_end=True
    )

    # 7. Definimos al propio entrenador
    trainer = Trainer(
        model,
        training_args,
        train_dataset=clean_ds["train"],
        eval_dataset=clean_ds["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

```
Vamos a agregar las últimas líneas de código para entrenar al modelo y subirlo al HUB de Hugging Face y observar sus resultados de entrenamiento, evaluación y test.

Agregamos el código para entrenar el modelo:

```python
# 8. Entrenamos al modelo
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
```
Respuesta esperada
```commandline
***** train metrics *****
  epoch                    =        4.0
  train_loss               =     0.3213
  train_runtime            = 0:01:57.42
  train_samples_per_second =    124.948
  train_steps_per_second   =     15.636
{'eval_loss': 0.8058812022209167, 'eval_accuracy': 0.8455882352941176, 'eval_f1': 0.8868940754039497, 'eval_runtime': 1.7026, 'eval_samples_per_second': 239.639, 'eval_steps_per_second': 29.955, 'epoch': 3.27}

```
Finalmente evaluamos el modelo en el set de `test`:
```python
# 9. Evaluando en el conjunto de test
metrics = trainer.evaluate(clean_ds["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```
Respuesta esperada:
```commandline
***** eval metrics *****
  epoch                   =        4.0
  eval_accuracy           =      0.793
  eval_f1                 =     0.8415
  eval_loss               =     0.6225
  eval_runtime            = 0:00:03.62
  eval_samples_per_second =    476.381
  eval_steps_per_second   =     59.651
```

Felicidades, ya hemos terminado nuestro proyecto de NLP. Y hemos logrado obtener un `accuracy` de casi 80% en el set de pruebas y un 84% En F1 score.
Nuestro modelo ya se encuentra en linea en el siguiente link: https://huggingface.co/platzi/platzi-distilroberta-base-mrpc-glue-gabriel-ichcanziho

![results.png](4%20NLP%2F4%20Entrenamiento%20y%20Evaluaci%C3%B3n%2Fresults.png)


## 4.5 Quiz: NLP con hugging face

![2.png](imgs%2F4%20Natural%20Language%20Processing%20%2F2.png)


# 5 Comparte en el HUB

## 5.1 El Hub como tu curriculum para machine learning

El resumen de esta clase es muy sencillo, podemos crear un buen perfil en el HUB de Hugging Face para poder compartirlo como nuestro curriculum
de machine learning y compartir con el público en general como han sido nuestros modelos, espacios, datasets entre otros.

![1.png](imgs%2F5%20HUB%2F1.png)

![2.png](imgs%2F5%20HUB%2F2.png)

El HUB nos permite observar la cantidad de descargas de nuestras contribuciones.

## 5.2 Compartir tu modelo en el HUB de Hugging Face

En realidad en los puntos anteriores se subía automáticamente al HUB, sin embargo esto es otra opción para agregar más información 
al momento de hacer PUSH al HUB.

```python
kwargs = {
    "finetuned_from": model.config._name_or_path,
    "tasks": "text-classification",
    "dataset": "datasetX",
    "tags": ["text-classification"]
}

trainer.push_to_hub(commit_message="Lo logramos de nuevo equipo Platzi! 🤗", **kwargs)
```

