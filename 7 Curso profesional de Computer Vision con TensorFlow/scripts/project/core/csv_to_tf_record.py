import os
import pandas as pd
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
import argparse

parser = argparse.ArgumentParser(description="CSV To TFRECORD file converter")

parser.add_argument("-c",
                    "--csv_file",
                    help="csv file directory. Format '.csv'",
                    type=str)
parser.add_argument("-i",
                    "--images_dir",
                    help="path where the images are stored",
                    type=str)

parser.add_argument("-o",
                    "--output_file",
                    help="tf record file directory. Format '.record'",
                    type=str)

args = parser.parse_args()


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(row["class_id"])  # Esta es la única alteración al código original de TensorFlow, en mi caso el
                                         # "row" ya tenía el "class_id" que contiene el número de la clase "class"

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(csv_file, images_dir, output_file):

    writer = tf.io.TFRecordWriter(output_file)
    path = os.path.join(images_dir)
    examples = pd.read_csv(csv_file)
    print(examples.head())
    print("Starting :)")
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file at: {}'.format(output_file))


if __name__ == '__main__':
    c_file = args.csv_file
    i_dir = args.images_dir
    o_file = args.output_file
    main(csv_file=c_file, images_dir=i_dir, output_file=o_file)
