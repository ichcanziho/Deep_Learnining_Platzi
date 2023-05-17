import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import argparse


parser = argparse.ArgumentParser(description="Pipeline config updater")

parser.add_argument("--input_config",
                    help="pipeline configuration file",
                    type=str)
parser.add_argument("--n_classes",
                    help="number of classes",
                    type=int)
parser.add_argument("--bs",
                    help="batch size for training",
                    type=int)
parser.add_argument("--checkpoint",
                    help="checkpoint file to start training",
                    type=str)
parser.add_argument("--ptype",
                    help="problem type {object detection, classification, segmentation}",
                    type=str)
parser.add_argument("--label_map",
                    help="label map file",
                    type=str)

parser.add_argument("--train_record",
                    help="train record file in tf_record format",
                    type=str)

parser.add_argument("--test_record",
                    help="test record file in tf_record format",
                    type=str)

args = parser.parse_args()


def update_config(input_config, n_classes, batch_size, checkpoint, checkpoint_type, label_map, train_record, test_record):
    # Obtenemos la configuración del archivo pipeline
    print("start")
    config = config_util.get_configs_from_pipeline_file(input_config)
    # Creamos una variable proto_str para poder modificar las variables del archivo pbtxt
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(input_config, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    # Cantidad de clases del modelo
    pipeline_config.model.ssd.num_classes = n_classes
    # El tamaño del batch size, entre más grande más costo computacional va a necesitar en el proceso de entrenamiento,
    # pero a su vez entrenará más rapido.
    pipeline_config.train_config.batch_size = batch_size
    # Donde almacenaremos los resultados del entrenamiento
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint
    # Qué tipo de detección aplicaremos (Object detection)
    pipeline_config.train_config.fine_tune_checkpoint_type = checkpoint_type
    # Dirección del label map
    pipeline_config.train_input_reader.label_map_path = label_map
    # Dirección del train TFRecord
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = train_record
    # Dirección del label map
    pipeline_config.eval_input_reader[0].label_map_path = label_map
    # Dirección del test TFRecord
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = test_record
    # Almacenamos nuestro archivo final
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(input_config, "wb") as f:
        f.write(config_text)
    print("done")


if __name__ == '__main__':

    c_file = args.input_config
    n_c = args.n_classes
    bs = args.bs
    ckp = args.checkpoint
    ckp_t = args.ptype
    lb_map = args.label_map
    tra_re = args.train_record
    tst_re = args.test_record
    update_config(c_file, n_c, bs, ckp, ckp_t, lb_map, tra_re, tst_re)

