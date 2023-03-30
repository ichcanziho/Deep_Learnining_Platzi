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

plt.imshow(ex["image"])
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

## 3.3 Configurando un Trainer para computer vision

## 3.4 Entrenamiento y evaluación de modelo de computer vision

## 3.5 Quiz: Computer vision con Hugging Face

# 4 Natural Language Processing

## 4.1 Carga de datasets para NLP

## 4.2 Procesamiento de dataset para NLP

## 4.3 Configurando un Trainer para NLP

## 4.4 Entrenamiento y evaluación de modelo de NLP

## 4.5 Quiz: NLP con hugging face

# 5 Comparte en el HUB

## 5.1 El Hub como tu curriculum para machine learning

## 5.2 Compartir tu modelo en el HUB de Hugging Face