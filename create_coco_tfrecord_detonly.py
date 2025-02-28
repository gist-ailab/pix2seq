# coding=utf-8
# Copyright 2022 The Pix2Seq Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert COCO dataset to tfrecords."""

import collections
import json
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import vocab
from data.scripts import tfrecord_lib
import tensorflow as tf



flags.DEFINE_string('coco_image_dir', '', 'Directory containing images.')
flags.DEFINE_string('ins_ann_file', '', 'Instance annotation file.')
flags.DEFINE_string('output_dir', None, 'Output directory')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS


def load_instance_annotations(annotation_path):
  """Load instance annotations.

  Args:
    annotation_path: str. Path to the annotation file.

  Returns:
    image_info: a list of dicts, with information such as file name, image id,
        height, width, etc.
    category_id_to_name_map: dict of category ids to category names.
    img_to_ann: a dict of image_id to annotation.
  """
  logging.info('Building instance index.')
  with tf.io.gfile.GFile(annotation_path, 'r') as f:
    annotations = json.load(f)

  image_info = annotations['images']
  category_id_to_name_map = dict(
      (element['id'], element['name']) for element in annotations['categories'])

  img_to_ann = collections.defaultdict(list)
  for ann in annotations['annotations']:
    image_id = ann['image_id']
    img_to_ann[image_id].append(ann)

  return image_info, category_id_to_name_map, img_to_ann


def coco_annotations_to_lists(obj_annotations, id_to_name_map):
  """Converts COCO annotations to feature lists.

  Args:
    obj_annotations: a list of object annotations.
    id_to_name_map: category id to category name map.

  Returns:
    a dict of list features.
  """

  data = dict((k, list()) for k in [
      'xmin', 'xmax', 'ymin', 'ymax', 'is_crowd', 'category_id',
      'category_names', 'area'])

  for ann in obj_annotations:
    (x, y, width, height) = tuple(ann['bbox'])
    data['xmin'].append(float(x))
    data['xmax'].append(float(x + width))
    data['ymin'].append(float(y))
    data['ymax'].append(float(y + height))
    data['is_crowd'].append(ann['iscrowd'])
    category_id = int(ann['category_id'])
    data['category_id'].append(category_id)
    data['category_names'].append(id_to_name_map[category_id].encode('utf8'))
    data['area'].append(ann['area'])

  return data


def obj_annotations_to_feature_dict(obj_annotations, id_to_name_map):
  """Convert COCO annotations to an encoded feature dict.

  Args:
    obj_annotations: a list of object annotations.
    id_to_name_map: category id to category name map.

  Returns:
    a dict of tf features.
  """

  data = coco_annotations_to_lists(obj_annotations, id_to_name_map)
  feature_dict = {
      'image/object/bbox/xmin':
          tfrecord_lib.convert_to_feature(data['xmin']),
      'image/object/bbox/xmax':
          tfrecord_lib.convert_to_feature(data['xmax']),
      'image/object/bbox/ymin':
          tfrecord_lib.convert_to_feature(data['ymin']),
      'image/object/bbox/ymax':
          tfrecord_lib.convert_to_feature(data['ymax']),
      'image/object/class/text':
          tfrecord_lib.convert_to_feature(data['category_names']),
      'image/object/class/label':
          tfrecord_lib.convert_to_feature(data['category_id']),
      'image/object/is_crowd':
          tfrecord_lib.convert_to_feature(data['is_crowd']),
      'image/object/area':
          tfrecord_lib.convert_to_feature(data['area']),
  }
  return feature_dict


def generate_annotations(images, image_dir, category_id_to_name_map, img_to_obj_ann):
  """Generator for COCO annotations."""
  for image in images:
    object_ann = img_to_obj_ann.get(image['id'], {})

    yield (image, image_dir, category_id_to_name_map, object_ann)


def create_tf_example(image, image_dir, category_id_to_name_map, object_ann):
  """Converts image and annotations to a tf.Example proto."""
  # Add image features.
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  with tf.io.gfile.GFile(os.path.join(image_dir, filename), 'rb') as fid:
    encoded_jpg = fid.read()
  feature_dict = tfrecord_lib.image_info_to_feature_dict(
      image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

  # Add annotation features.
  if object_ann:
    # Bbox, area, etc.
    obj_feature_dict = obj_annotations_to_feature_dict(object_ann,
                                                       category_id_to_name_map)
    feature_dict.update(obj_feature_dict)

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example, 0


def main(_):
  image_info, category_id_to_name_map, img_to_obj_ann = (
      load_instance_annotations(FLAGS.ins_ann_file))

  directory = os.path.dirname(FLAGS.output_dir)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  coco_annotations_iter = generate_annotations(
      images=image_info,
      image_dir=FLAGS.coco_image_dir,
      category_id_to_name_map=category_id_to_name_map,
      img_to_obj_ann=img_to_obj_ann)

  tfrecord_lib.write_tf_record_dataset(
      output_path=FLAGS.output_dir,
      annotation_iterator=coco_annotations_iter,
      process_func=create_tf_example,
      num_shards=FLAGS.num_shards,
      multiple_processes=8)


# Note: internal version of the code overrides this function.
def run_main():
  app.run(main)




if __name__ == '__main__':
  run_main()
