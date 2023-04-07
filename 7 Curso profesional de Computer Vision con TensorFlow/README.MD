# Curso Profesional de Computer Vision con TensorFlow

Utiliza el poder de todo lo que has aprendido de redes neuronales y TensorFlow para crear productos de visión artificial. 
Aplica detección y clasificación de imágenes, seguimiento de objetos y más.

- Entrena y optimiza modelos de visión computarizada con TensorFlow.
- Pon en producción tu modelo.
- Dimensiona un proyecto de visión computarizada.
- Obtén y procesa datos de imágenes a TFRecord.


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> - [5: Curso de Experimentación en Machine Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/5%20Curso%20de%20introducci%C3%B3n%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
> - [6: Curso de detección y segmentación de objetos con TensorFlow]()
> 
> Este Curso es el Número 7 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice

- [1 Comprender la visión computarizada](#1-comprender-la-visión-computarizada)
  - [1.1 ¿Por qué aprender computer vision?](#11-por-qué-aprender-computer-vision)
  - [1.2 ¿Qué es la visión computarizada?](#12-qué-es-la-visión-computarizada)
  - [1.3 Tipos de visión computarizada](#13-tipos-de-visión-computarizada)
  - [1.4 Introducción a object detection: clasificación + localización](#14-introducción-a-object-detection-clasificación--localización)
  - [1.5 Aprende a identificar problemas](#15-aprende-a-identificar-problemas)
- [2 Dimensionamiento de proyecto de visión computarizada](#2-dimensionamiento-de-proyecto-de-visión-computarizada)
  - [2.1 Cómo definir los tiempos de tu proyecto](#21-cómo-definir-los-tiempos-de-tu-proyecto)
  - [2.2 Cómo costear tu proyecto](#22-cómo-costear-tu-proyecto)
  - [2.3 Cómo identificar los roles necesarios en el proyecto](#23-cómo-identificar-los-roles-necesarios-en-el-proyecto)
  - [2.4 Producto mínimo viable en computer vision](#24-producto-mínimo-viable-en-computer-vision)
- [3 Obtención y procesamiento de los datos](#3-obtención-y-procesamiento-de-los-datos)
  - [3.1 Obtención de datos para tu proyecto](#31-obtención-de-datos-para-tu-proyecto)
  - [3.2 Limpieza de la base de datos](#32-limpieza-de-la-base-de-datos)
  - [3.3 Distribución de datos en entrenamiento y testeo](#33-distribución-de-datos-en-entrenamiento-y-testeo)
  - [3.4 Etiquetado de los datos de test](#34-etiquetado-de-los-datos-de-test)
  - [3.5 Etiquetado de los datos de train](#35-etiquetado-de-los-datos-de-train)
  - [3.6 Transforma tu base de datos a TFRecord](#36-transforma-tu-base-de-datos-a-tfrecord)
  - [3.7 Transformar CSV a TFRecord](#37-transformar-csv-a-tfrecord)
- [4 Entrena, testea y optimiza tus modelos](#4-entrena-testea-y-optimiza-tus-modelos)
  - [4.1 Librerías a importar durante fase de entrenamiento](#41-librerías-a-importar-durante-fase-de-entrenamiento)
  - [4.2 Fase de entrenamiento del modelo](#42-fase-de-entrenamiento-del-modelo)
  - [4.3 Balanceo de imágenes y data augmentation](#43-balanceo-de-imágenes-y-data-augmentation)
  - [4.4 Entrena, evalúa y optimiza con TensorBoard](#44-entrena-evalúa-y-optimiza-con-tensorboard)
  - [4.5 Validación de modelo en un entorno de ejecución](#45-validación-de-modelo-en-un-entorno-de-ejecución)
  - [4.6 Re-entrenamiento del modelo para obtener mejores resultados](#46-re-entrenamiento-del-modelo-para-obtener-mejores-resultados)
  - [4.7 Seguimiento de centroides con OpenCV](#47-seguimiento-de-centroides-con-opencv)
  - [4.8 Configuración de los centroides con OpenCV](#48-configuración-de-los-centroides-con-opencv)
  - [4.9 Algoritmo de dirección y conteo con OpenCV](#49-algoritmo-de-dirección-y-conteo-con-opencv)
  - [4.10 Crea un ciclo de entrenamiento de tu modelo: MLOps](#410-crea-un-ciclo-de-entrenamiento-de-tu-modelo-mlops)
- [5 Producto con visión computarizada en producción](#5-producto-con-visión-computarizada-en-producción)
  - [5.1 Prepara tu entorno en Google Cloud Platform](#51-prepara-tu-entorno-en-google-cloud-platform)
  - [5.2 Carga y preprocesamiento de modelos](#52-carga-y-preprocesamiento-de-modelos)
  - [5.3 Postprocesamiento de modelos](#53-postprocesamiento-de-modelos)
  - [5.4 Despliega y consume tu modelo en producción](#54-despliega-y-consume-tu-modelo-en-producción)
  - [5.5 Bonus: aprende a apagar las máquinas de GCP para evitar sobrecostos](#55-bonus-aprende-a-apagar-las-máquinas-de-gcp-para-evitar-sobrecostos)
- [6 Siguientes pasos en inteligencia artificial](#6-siguientes-pasos-en-inteligencia-artificial)
  - [6.1 Siguientes pasos en inteligencia artificial](#61-siguientes-pasos-en-inteligencia-artificial)


# 1 Comprender la visión computarizada

## 1.1 ¿Por qué aprender computer vision?

## 1.2 ¿Qué es la visión computarizada?

## 1.3 Tipos de visión computarizada

## 1.4 Introducción a object detection: clasificación + localización

## 1.5 Aprende a identificar problemas

# 2 Dimensionamiento de proyecto de visión computarizada

## 2.1 Cómo definir los tiempos de tu proyecto

## 2.2 Cómo costear tu proyecto

## 2.3 Cómo identificar los roles necesarios en el proyecto

## 2.4 Producto mínimo viable en computer vision

# 3 Obtención y procesamiento de los datos

## 3.1 Obtención de datos para tu proyecto

## 3.2 Limpieza de la base de datos

## 3.3 Distribución de datos en entrenamiento y testeo

## 3.4 Etiquetado de los datos de test

## 3.5 Etiquetado de los datos de train

## 3.6 Transforma tu base de datos a TFRecord

## 3.7 Transformar CSV a TFRecord

# 4 Entrena, testea y optimiza tus modelos

## 4.1 Librerías a importar durante fase de entrenamiento

## 4.2 Fase de entrenamiento del modelo

## 4.3 Balanceo de imágenes y data augmentation

## 4.4 Entrena, evalúa y optimiza con TensorBoard

## 4.5 Validación de modelo en un entorno de ejecución

## 4.6 Re-entrenamiento del modelo para obtener mejores resultados

## 4.7 Seguimiento de centroides con OpenCV

## 4.8 Configuración de los centroides con OpenCV

## 4.9 Algoritmo de dirección y conteo con OpenCV

## 4.10 Crea un ciclo de entrenamiento de tu modelo: MLOps

# 5 Producto con visión computarizada en producción

## 5.1 Prepara tu entorno en Google Cloud Platform

## 5.2 Carga y preprocesamiento de modelos

## 5.3 Postprocesamiento de modelos

## 5.4 Despliega y consume tu modelo en producción

## 5.5 Bonus: aprende a apagar las máquinas de GCP para evitar sobrecostos

# 6 Siguientes pasos en inteligencia artificial

## 6.1 Siguientes pasos en inteligencia artificial


