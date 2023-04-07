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

## 1.2 Etiquetado rápido en Python: español e inglés

## 1.3 Etiquetado rápido en Python: Stanza (Stanford NLP)

# 2 Modelos Markovianos Latentes (HMM)

## 2.1 Cadenas de Markov

## 2.2 Modelos Markovianos latentes (HMM)

## 2.3 Entrenando un HMM

## 2.4 Fases de entrenamiento de un HMM

## 2.5 Entrenando un HMM en Python

# 3 Algoritmo de Viterbi

## 3.1 El algoritmo de Viterbi

## 3.2 Cálculo de las probabilidades de Viterbi

## 3.3 Carga del modelo HMM y distribución inicial

## 3.4 Implementación de algoritmo de Viterbi en Python

## 3.5 Entrenamiento directo de HMM con NLTK

# 4 Modelos Markovianos de máxima entropía (MEMM)

## 4.1 Modelos Markovianos de máxima entropia (MEMM)

## 4.2 Algoritmo de Viterbi para MEMM

## 4.3 Reto: construye un MEMM en Python

# 5 Clasificación de texto con NLTK

## 5.1 El problema general de la clasificación de texto

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

