# Curso de Detección y Segmentación de Objetos con TensorFlow

Eleva tu nivel en inteligencia artificial aprendiendo tareas avanzadas de deep learning para detectar y segmentar objetos en imágenes y videos. 
Entrena modelos pre-entrenados de computer vision de acuerdo a las necesidades de tus proyectos.

- Evalúa desempeño de modelos para visión computarizada.
- Utiliza modelos pre-entrenados para visión computarizada.
- Segmenta objetos dentro de imágenes.
- Aplica detección de objetos con Python y TensorFlow.


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> - [5: Curso de Experimentación en Machine Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/5%20Curso%20de%20introducci%C3%B3n%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
> 
> Este Curso es el Número 6 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice
- [1 Introducción a Computer Vision](#1-introducción-a-computer-vision)
  - [1.1 ¿Qué es la visión computarizada y cuáles son sus tipos?](#11-qué-es-la-visión-computarizada-y-cuáles-son-sus-tipos)
  - [2 Detección de objetos](#2-detección-de-objetos)
  - [2.1 Introducción a object detection: sliding window y bounding box](#21-introducción-a-object-detection-sliding-window-y-bounding-box)
  - [2.2 Generando video de sliding window](#22-generando-video-de-sliding-window)
  - [2.3 Introducción a object detection: backbone, non-max suppression y métricas](#23-introducción-a-object-detection-backbone-non-max-suppression-y-métricas)
  - [2.4 Visualización de IoU en object detection](#24-visualización-de-iou-en-object-detection)
  - [2.5 Tipos de arquitecturas en detección de objetos](#25-tipos-de-arquitecturas-en-detección-de-objetos)
  - [2.6 Arquitecturas relevantes en object detection](#26-arquitecturas-relevantes-en-object-detection)
  - [2.7 Utilizando un dataset de object detection](#27-utilizando-un-dataset-de-object-detection)
  - [2.8 Carga de dataset de object detection](#28-carga-de-dataset-de-object-detection)
  - [2.9 Exploración del dataset de object detection](#29-exploración-del-dataset-de-object-detection)
  - [2.10 Visualización de bounding boxes en el dataset de object detection](#210-visualización-de-bounding-boxes-en-el-dataset-de-object-detection)
  - [2.11 Aumentando de datos con Albumentation](#211-aumentando-de-datos-con-albumentation)
  - [2.12 Implementando Albumentation en object detection](#212-implementando-albumentation-en-object-detection)
  - [2.13 Visualizando imágenes con aumentado de datos](#213-visualizando-imágenes-con-aumentado-de-datos)
  - [2.14 Utilizando un modelo de object detection pre-entrenado](#214-utilizando-un-modelo-de-object-detection-pre-entrenado)
  - [2.15 Fine-tuning en detección de objetos](#215-fine-tuning-en-detección-de-objetos)
  - [2.16 Fine-tuning en detección de objetos: carga de datos](#216-fine-tuning-en-detección-de-objetos-carga-de-datos)
  - [2.17 Fine-tuning en detección de objetos: data augmentation](#217-fine-tuning-en-detección-de-objetos-data-augmentation)
  - [2.18 Fine-tuning en detección de objetos: entrenamiento](#218-fine-tuning-en-detección-de-objetos-entrenamiento)
  - [2.19 Fine tuning en detección de objetos: visualización de objetos](#219-fine-tuning-en-detección-de-objetos-visualización-de-objetos)
  - [2.20 Quiz módulo object detection](#220-quiz-módulo-object-detection)
- [3 Segmentación de objetos](#3-segmentación-de-objetos)
  - [3.1 Introduciendo la segmentación de objetos](#31-introduciendo-la-segmentación-de-objetos)
  - [3.2 Tipos de segmentación de objetos](#32-tipos-de-segmentación-de-objetos)
  - [3.3 Tipos de segmentación y sus arquitecturas relevantes](#33-tipos-de-segmentación-y-sus-arquitecturas-relevantes)
  - [3.4 ¿Cómo es un dataset de segmentación?](#34-cómo-es-un-dataset-de-segmentación)
  - [3.5 Utilizando un dataset de segmentación de objetos](#35-utilizando-un-dataset-de-segmentación-de-objetos)
  - [3.6 Visualización de nuestro dataset de segmentación](#36-visualización-de-nuestro-dataset-de-segmentación)
  - [3.7 Creando red neuronal U-Net para segmentación](#37-creando-red-neuronal-u-net-para-segmentación)
  - [3.8 Entrenando y estudiando una red de segmentación](#38-entrenando-y-estudiando-una-red-de-segmentación)
  - [3.9 Generando predicciones con modelo de object segmentation](#39-generando-predicciones-con-modelo-de-object-segmentation)
  - [3.10 Quiz módulo segmentación](#310-quiz-módulo-segmentación)
- [4 Un paso más allá](#4-un-paso-más-allá)
  - [4.1 El estado de la cuestión en computer vision](#41-el-estado-de-la-cuestión-en-computer-vision)


# 1 Introducción a Computer Vision

## 1.1 ¿Qué es la visión computarizada y cuáles son sus tipos?

# 2 Detección de objetos

## 2.1 Introducción a object detection: sliding window y bounding box

## 2.2 Generando video de sliding window

## 2.3 Introducción a object detection: backbone, non-max suppression y métricas

## 2.4 Visualización de IoU en object detection

## 2.5 Tipos de arquitecturas en detección de objetos

## 2.6 Arquitecturas relevantes en object detection

## 2.7 Utilizando un dataset de object detection

## 2.8 Carga de dataset de object detection

## 2.9 Exploración del dataset de object detection

## 2.10 Visualización de bounding boxes en el dataset de object detection

## 2.11 Aumentando de datos con Albumentation

## 2.12 Implementando Albumentation en object detection

## 2.13 Visualizando imágenes con aumentado de datos

## 2.14 Utilizando un modelo de object detection pre-entrenado

## 2.15 Fine-tuning en detección de objetos

## 2.16 Fine-tuning en detección de objetos: carga de datos

## 2.17 Fine-tuning en detección de objetos: data augmentation

## 2.18 Fine-tuning en detección de objetos: entrenamiento

## 2.19 Fine tuning en detección de objetos: visualización de objetos

## 2.20 Quiz módulo object detection

# 3 Segmentación de objetos

## 3.1 Introduciendo la segmentación de objetos

## 3.2 Tipos de segmentación de objetos

## 3.3 Tipos de segmentación y sus arquitecturas relevantes

## 3.4 ¿Cómo es un dataset de segmentación?

## 3.5 Utilizando un dataset de segmentación de objetos

## 3.6 Visualización de nuestro dataset de segmentación

## 3.7 Creando red neuronal U-Net para segmentación

## 3.8 Entrenando y estudiando una red de segmentación

## 3.9 Generando predicciones con modelo de object segmentation

## 3.10 Quiz módulo segmentación

# 4 Un paso más allá

## 4.1 El estado de la cuestión en computer vision
