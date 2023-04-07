# Curso de Generación de Imágenes con IA: Dall-E, Midjourney y Stable Diffusion

Explora cómo utilizar modelos de inteligencia artificial para generar imágenes sorprendentes y fotorrealistas con solo 
describirlas en texto. Utilizarás Dall-E 2, Midjourney y Stable Diffusion experimentando con diferentes técnicas y estilos. 
Lleva el poder del deep learning a otro nivel con Carlos Alarcón como tu profesor.

- Estiliza imágenes con prompt tuning.
- Genera imágenes con textos sencillos o detallados.
- Edita imágenes y genera diferentes conceptos con técnicas avanzadas.
- Conoce las implicaciones éticas y económicas de estas tecnologías.


> ## NOTA:
> Antes de continuar te invito a que revises los cursos anteriores:
> - [1: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/1%20Curso%20de%20fundamentos%20de%20redes%20neuronales)
> - [2: Curso de Redes Neuronales Convolucionales con Python y keras](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
> - [3: Curso profesional de Redes Neuronales con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
> - [4: Curso de Transfer Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
> - [5: Curso de Experimentación en Machine Learning con Hugging Face](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/5%20Curso%20de%20introducci%C3%B3n%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
> - [6: Curso de detección y segmentación de objetos con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/6%20Curso%20de%20detecci%C3%B3n%20y%20segmentaci%C3%B3n%20de%20objetos%20con%20Tensorflow)
> - [7: Curso profesional de Computer Vision con TensorFlow](https://github.com/ichcanziho/Deep_Learnining_Platzi/tree/master/7%20Curso%20profesional%20de%20Computer%20Vision%20con%20TensorFlow)
> 
> Este Curso es el Número 8 de una ruta de Deep Learning, quizá algunos conceptos no vuelvan a ser definidos en este repositorio,
> por eso es indispensable que antes de empezar a leer esta guía hayas comprendido los temas vistos anteriormente.
> 
> Sin más por agregar disfruta de este curso

# Índice

- [1 Fundamentos de generación de imágenes con texto](#1-fundamentos-de-generación-de-imágenes-con-texto)
  - [1.1 ¿Qué es text-to-Image en inteligencia artificial?](#11-qué-es-text-to-image-en-inteligencia-artificial)
  - [1.2 Cómo funciona text-to-Image: difussion](#12-cómo-funciona-text-to-image-difussion)
  - [1.3 Exponentes de IA generativa: DALL·E 2, Midjourney y Stable Difussion](#13-exponentes-de-ia-generativa-dalle-2-midjourney-y-stable-difussion)
- [2 DALL·E 2](#2-dalle-2)
  - [2.1 Generación de imágenes con DALL·E 2](#21-generación-de-imágenes-con-dalle-2)
  - [2.2 Restricciones y limitantes de DALL·E 2](#22-restricciones-y-limitantes-de-dalle-2)
  - [2.3 Prompt tuning: luces, detalles, sombras y perspectivas](#23-prompt-tuning-luces-detalles-sombras-y-perspectivas)
  - [2.4 Prompt tuning en DALL·E 2: práctica](#24-prompt-tuning-en-dalle-2-práctica)
  - [2.5 Prompt tuning: estilos y artistas](#25-prompt-tuning-estilos-y-artistas)
  - [2.6 Prompt tuning en DALL·E 2: práctica de estilos y artistas](#26-prompt-tuning-en-dalle-2-práctica-de-estilos-y-artistas)
  - [2.7 Uso de API de DALL·E 2 con Python](#27-uso-de-api-de-dalle-2-con-python)
- [3 Midjourney](#3-midjourney)
  - [3.1 Generación de imágenes con Midjourney](#31-generación-de-imágenes-con-midjourney)
  - [3.2 Prompt tuning con Midjourney](#32-prompt-tuning-con-midjourney)
  - [3.3 Prompt tuning con Midjourney: práctica](#33-prompt-tuning-con-midjourney-práctica)
  - [3.4 Image-to-Image y Blend con Midjourney](#34-image-to-image-y-blend-con-midjourney)
- [4 Stable Diffusion](#4-stable-diffusion)
  - [4.1 Stable Diffusion con Google Colab y DreamStudio](#41-stable-diffusion-con-google-colab-y-dreamstudio)
  - [4.2 Instalación de AUTOMATIC 1111](#42-instalación-de-automatic-1111)
  - [4.3 AUTOMATIC 1111: interfaz web de Stable Diffusion](#43-automatic-1111-interfaz-web-de-stable-diffusion)
  - [4.4 Prompt tuning con Stable Diffusion](#44-prompt-tuning-con-stable-diffusion)
  - [4.5 Prompt tuning con Stable Diffusion: práctica](#45-prompt-tuning-con-stable-diffusion-práctica)
- [5 Inpainting y outpainting](#5-inpainting-y-outpainting)
  - [5.1 DALL·E 2: Inpainting](#51-dalle-2-inpainting)
  - [5.2 Stable Diffusion: Inpainting](#52-stable-diffusion-inpainting)
  - [5.3 DALL·E 2: Outpainting](#53-dalle-2-outpainting)
  - [5.4 Instalación de PaintHua](#54-instalación-de-painthua)
  - [5.5 Stable Diffusion: Outpainting con PaintHua](#55-stable-diffusion-outpainting-con-painthua)
- [6 Stable Diffusion features](#6-stable-diffusion-features)
  - [6.1 Upscale: escalado de imágenes](#61-upscale-escalado-de-imágenes)
  - [6.2 Image-to-Image](#62-image-to-image)
  - [6.3 Depth-to-Image](#63-depth-to-image)
  - [6.4 ¿Cómo funciona DreamBooth?](#64-cómo-funciona-dreambooth)
  - [6.5 DreamBooth: fine-tuning](#65-dreambooth-fine-tuning)
  - [6.6 Merge models: combinar modelos de diffusion](#66-merge-models-combinar-modelos-de-diffusion)
- [7 Implicaciones éticas y económicas de imágenes con inteligencia artificial](#7-implicaciones-éticas-y-económicas-de-imágenes-con-inteligencia-artificial)
  - [7.1 Sesgos y bias en modelos de generación de imágenes](#71-sesgos-y-bias-en-modelos-de-generación-de-imágenes)
  - [7.2 ¿Cómo afecta la generación de imágenes con IA a otras profesiones?](#72-cómo-afecta-la-generación-de-imágenes-con-ia-a-otras-profesiones)
  - [7.3 Impacto de arte y diseño con IA: entrevista con Daniel Torres Buriel](#73-impacto-de-arte-y-diseño-con-ia-entrevista-con-daniel-torres-buriel)
  - [7.4 Perspectiva de artistas y diseñadoras: entrevista con Juan Dávila](#74-perspectiva-de-artistas-y-diseñadoras-entrevista-con-juan-dávila)
  - [7.5 Perspectiva de artistas y diseñadoras: entrevista con Amelia Amórtegui](#75-perspectiva-de-artistas-y-diseñadoras-entrevista-con-amelia-amórtegui)
- [8 Conclusión](#8-conclusión)
  - [8.1 ¿Qué nos deparará el futuro de la IA generativa?](#81-qué-nos-deparará-el-futuro-de-la-ia-generativa)


# 1 Fundamentos de generación de imágenes con texto

## 1.1 ¿Qué es text-to-Image en inteligencia artificial?

## 1.2 Cómo funciona text-to-Image: difussion

## 1.3 Exponentes de IA generativa: DALL·E 2, Midjourney y Stable Difussion

# 2 DALL·E 2

## 2.1 Generación de imágenes con DALL·E 2

## 2.2 Restricciones y limitantes de DALL·E 2

## 2.3 Prompt tuning: luces, detalles, sombras y perspectivas

## 2.4 Prompt tuning en DALL·E 2: práctica

## 2.5 Prompt tuning: estilos y artistas

## 2.6 Prompt tuning en DALL·E 2: práctica de estilos y artistas

## 2.7 Uso de API de DALL·E 2 con Python

# 3 Midjourney

## 3.1 Generación de imágenes con Midjourney

## 3.2 Prompt tuning con Midjourney

## 3.3 Prompt tuning con Midjourney: práctica

## 3.4 Image-to-Image y Blend con Midjourney

# 4 Stable Diffusion

## 4.1 Stable Diffusion con Google Colab y DreamStudio

## 4.2 Instalación de AUTOMATIC 1111

## 4.3 AUTOMATIC 1111: interfaz web de Stable Diffusion

## 4.4 Prompt tuning con Stable Diffusion

## 4.5 Prompt tuning con Stable Diffusion: práctica

# 5 Inpainting y outpainting

## 5.1 DALL·E 2: Inpainting

## 5.2 Stable Diffusion: Inpainting

## 5.3 DALL·E 2: Outpainting

## 5.4 Instalación de PaintHua

## 5.5 Stable Diffusion: Outpainting con PaintHua

# 6 Stable Diffusion features

## 6.1 Upscale: escalado de imágenes

## 6.2 Image-to-Image

## 6.3 Depth-to-Image

## 6.4 ¿Cómo funciona DreamBooth?

## 6.5 DreamBooth: fine-tuning

## 6.6 Merge models: combinar modelos de diffusion

# 7 Implicaciones éticas y económicas de imágenes con inteligencia artificial

## 7.1 Sesgos y bias en modelos de generación de imágenes

## 7.2 ¿Cómo afecta la generación de imágenes con IA a otras profesiones?

## 7.3 Impacto de arte y diseño con IA: entrevista con Daniel Torres Buriel

## 7.4 Perspectiva de artistas y diseñadoras: entrevista con Juan Dávila

## 7.5 Perspectiva de artistas y diseñadoras: entrevista con Amelia Amórtegui

# 8 Conclusión

## 8.1 ¿Qué nos deparará el futuro de la IA generativa?
