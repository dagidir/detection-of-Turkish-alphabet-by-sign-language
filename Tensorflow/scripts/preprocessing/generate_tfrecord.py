"""
Usage:
  # From tensorflow/models/
  # Create train data:
  D:\AnacondaProjects\tez\Tensorflow\scripts\preprocessing>python generate_tfrecord.py 
                                                            --csv_input=D:\AnacondaProjects\tez\Tensorflow\workspace\training\data\train_labels.csv 
                                                            --output_path=D:\AnacondaProjects\tez\Tensorflow\workspace\training\exported-models\train.record 
                                                            --image_dir=D:\AnacondaProjects\tez\Tensorflow\workspace\training\images\train
  # Create test data:
  D:\AnacondaProjects\tez\Tensorflow\scripts\preprocessing>python generate_tfrecord.py 
                                                            --csv_input=D:\AnacondaProjects\tez\Tensorflow\workspace\training\data\test_labels.csv 
                                                            --output_path=D:\AnacondaProjects\tez\Tensorflow\workspace\training\exported-models\test.record 
                                                            --image_dir=D:\AnacondaProjects\tez\Tensorflow\workspace\training\images\test
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from os import path
import io
import pandas as pd
import tensorflow._api.v2.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.compat.v1.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

labels_path = "D:\\AnacondaProjects\\tez\\Tensorflow\\workspace\\training\\annotations"

#TFRecord (.record) file i olusturup buraya koyucam
output_path = "D:\\AnacondaProjects\\tez\\Tensorflow\\workspace\\training\\exported-models\\"

#cektigim resimlerin oldugu klasor. icinde train ve test diye 2 tane daha klasor var
image_dir = "D:\\AnacondaProjects\\tez\\Tensorflow\\workspace\\training\\images\\"

#resimleri csv ye cevrilmis hallerinin oldugu path
csv_input = "D:\\AnacondaProjects\\tez\\Tensorflow\\workspace\\training\\data\\"

labels = [{"name":"A", "id":1},
          {"name":"B", "id":2},
          {"name":"C", "id":3},
          {"name":"D", "id":4},
          {"name":"E", "id":5},
          {"name":"F", "id":6},
          {"name":"G", "id":7},
          {"name":"H", "id":8},
          {"name":"I", "id":9},
          {"name":"J", "id":10},
          {"name":"K", "id":11},
          {"name":"L", "id":12},
          {"name":"M", "id":13},
          {"name":"N", "id":14},
          {"name":"O", "id":15},
          {"name":"P", "id":16},
          {"name":"R", "id":17},
          {"name":"S", "id":18},
          {"name":"T", "id":19},
          {"name":"U", "id":20},
          {"name":"V", "id":21},
          {"name":"Y", "id":22},
          {"name":"Z", "id":23},
         ]

with open(labels_path + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

def class_text_to_int(row_label):
    for i in labels:
        for value in i.get('name'):
            if row_label==value:
                return 1
            else:
                None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf-8') #group.filename.encode('utf-8')
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
        classes.append(class_text_to_int(row['class']))

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

def main(_):
    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)

    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()