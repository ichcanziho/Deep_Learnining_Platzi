# Ruta Completa de Deep Learning con Python

## Cursos integrados en esta ruta:

- ![](https://geps.dev/progress/100) [1. Curso de fundamentos de redes neuronales](1%20Curso%20de%20fundamentos%20de%20redes%20neuronales) 
- ![](https://geps.dev/progress/100) [2. Curso de Redes Neuronales Convolucionales](2%20Curso%20de%20Redes%20Neuronales%20Convolucionales)
- ![](https://geps.dev/progress/100) [3. Curso profesional de Redes Neuronales con TensorFlow](3%20Curso%20profesional%20de%20Redes%20Neuronales%20con%20TensorFlow)
- ![](https://geps.dev/progress/100) [4. Curso de Transfer Learning con Hugging Face](4%20Curso%20de%20Transfer%20Learning%20con%20Hugging%20Face)
- ![](https://geps.dev/progress/100) [5. Curso de introducción a Demos de Machine Learning con Hugging Face](5%20Curso%20de%20introducción%20a%20Demos%20de%20Machine%20Learning%20con%20Hugging%20Face)
- ![](https://geps.dev/progress/100) [6. Curso profesional de Computer Vision con TensorFlow](6%20Curso%20de%20detección%20y%20segmentación%20de%20objetos%20con%20Tensorflow)
- ![](https://geps.dev/progress/85) [7. Curso de detección y segmentación de objetos con TensorFlow](7%20Curso%20profesional%20de%20Computer%20Vision%20con%20TensorFlow)
- ![](https://geps.dev/progress/0) [8. Curso de generación de imágenes](8%20Curso%20de%20generación%20de%20imágenes)
- ![](https://geps.dev/progress/100) [9. Curso de Fundamentos de NLP](9%20Curso%20de%20Fundamentos%20de%20NLP)
- ![](https://geps.dev/progress/69) [10. Curso de Algoritmos de Clasificación de Texto](10%20Curso%20de%20Algoritmos%20de%20Clasificación%20de%20Texto)
- ![](https://geps.dev/progress/95) [11. Curso de Redes Neuronales con PyTorch](11%20Curso%20de%20Redes%20Neuronales%20con%20PyTorch%20)
- ![](https://geps.dev/progress/100) [12. Curso de Desarrollo de Chatbots con OpenAI](12%20Desarrollo%20ChatBot)
- ![](https://geps.dev/progress/33) [13. Curso de Desarrollo de Aplicaciones de IA con LangChain: Chatbots](13%20Curso%20de%20LangChain)


## Características de mi entorno de trabajo:

Este repositorio ha sido creado utilizando Ubuntu 22.04.2 LTS

La fecha inicial de creación de este repositorio es: Marzo 2023

Las características de mi computadora de trabajo son las siguientes:

- **Modelo del hardware:** Acer Nitro AN515-58
- **Memoria:** 16 Gb
- **Procesador:** 12th Gen Intel® Core™ i7-12700H × 20
- **Gráficos:** NVIDIA Corporation GA106M [GeForce RTX 3060 Mobile / Max-Q] 
- **Disco:** 1.5 TB
- **SO:** Ubuntu 22.04.2 LTS
- **Tipo de SO:** 64-bits
- **Versión de GNOME:** 42.5

Drivers de NVIDIA para Deep Learning:

```commandline
Thu Mar 16 14:25:50 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.86.01    Driver Version: 515.86.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   42C    P8    16W /  N/A |     62MiB /  6144MiB |     14%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2626      G   /usr/lib/xorg/Xorg                 61MiB |
+-----------------------------------------------------------------------------+
```

## Configuración de este repositorio

> ## Nota:
> Se ha utilizado python 3.10 como base de este repositorio.

1. Instalar venv:
    ```bash
    pip3 install virtualenv
    ```

2. Crear entorno virtual:
    ```bash
    python3 -m venv model_env
    ```
3. Activar el entorno:
    ```bash
    source model_env/bin/activate
    ```
4. Actualizar pip:
   ```bash
   pip install --upgrade pip
   ```
5. Instalar dependencias:
    ```bash
    pip install -r requirements.txt
    ```
6. Corroborar que se está corriendo el python adecuado:
   ```bash
   which python
   ```
   Respuesta esperada:
   ```commandline
   Deep Learning/model_env/bin/python
   ```
7. Correr python:
    ```bash
    python main.py
    ```

8. Desactivar entorno:
   ```bash
   deactivate
   ```

