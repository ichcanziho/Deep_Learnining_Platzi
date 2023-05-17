# --------------------------------------------------------------
# ------------------ PARÁMETROS EDITABLES ----------------------
# --------------------------------------------------------------

# cuál es el nombre de la carpeta que se utilizará para crear mi modelo custom
CUSTOM_MODEL_NAME = 'vehicle_detection'
# de qué dirección voy a descargar el modelo pre-entrenado
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
# ¿Cuántas clases tiene mi problema de detección de objetos?
N_CLASSES = 2
# Cuando empiece a entrenar, de cuánto será el batch-size?
BATCHSIZE = 4
# De qué clase es el tipo de problema que estoy resolviendo?
PTYPE = "detection"
# Una vez que el modelo haya sido entrenado en qué carpeta guardare los resultados para hacer inferencia?
OUTPUT_DIRECTORY = "vehicle_fine_tuned"
# Número de pasos para el entrenamiento
NUM_STEPS = 5000
# --------------------------------------------------------------

# Automáticamente, obtiene el nombre del modelo pre-entrenado con base en su URL de descarga
PRETRAINED_MODEL_NAME = PRETRAINED_MODEL_URL.split("/")[-1].split(".")[0]
# Ruta por defecto en dónde se va a encontrar el archivo de configuración de entrenamiento del modelo
PIPELINE_CONFIG = CUSTOM_MODEL_NAME+'/pipeline.config'
# Los modelos pre-entrenados vienen con diferentes puntos de control, nosotros podemos decidir desde que punto de
# control podemos reentrenar el modelo
CHECKPOINT = PRETRAINED_MODEL_NAME+"/checkpoint/ckpt-0"
# Cuál es la dirección del archivo que contiene las clases del modelo (está en formato json porque es bastante fácil
# de crear a mano)
JSON_LABEL_MAP_NAME = 'label_map.json'
# Ruta del `label_map` pero en su formato final listo para ser utilizado por `tensorflow`
LABEL_MAP_NAME = CUSTOM_MODEL_NAME + "/label_map.pbtxt"
# Ruta donde se encuentra la carpeta que contiene las imágenes para crear el tf-record de entrenamiento
TRAIN_IMAGES_FOLDER = "dataset/train"
# Archivo que contiene las etiquetas de las imágenes de entrenamiento en formato csv
TRAIN_IMAGES_DATASET_FILE = "dataset/train_labels.csv"
# Ruta donde se encuentra la carpeta que contiene las imágenes para crear el tf-record de test
TEST_IMAGES_FODLER = "dataset/test"
# Archivo que contiene las etiquetas de las imágenes de test en formato csv
TEST_IMAGES_DATASET_FILE = "dataset/test_labels.csv"
# Ruta de dónde se guardará el archivo de train en formato tf-record
TRAIN_RECORD_FILE = f"{CUSTOM_MODEL_NAME}/train.record"
# Ruta de dónde se guardará el archivo de test en formato tf-record
TEST_RECORD_FILE = f"{CUSTOM_MODEL_NAME}/test.record"
