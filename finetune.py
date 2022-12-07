# Imports.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import requests
import json
import abc

import ml_collections
import utils
from data.dataset import Dataset
from models import model as model_lib
from models import ar_model
from tasks import task as task_lib
from tasks import object_detection
import os

# Define a Dataset class to use for finetuning.
class VocDataset(Dataset):

  def extract(self, example, training):
    """Extracts needed features & annotations into a flat dictionary.

    Note: be consisous about 0 in label, which should probably reserved for
       special use (such as padding).

    Args:
      example: `dict` of raw features.
      training: `bool` of training vs eval mode.

    Returns:
      example: `dict` of relevant features and labels
    """
    # These features are needed by the object detection task.
    features = {
        'image': tf.image.convert_image_dtype(example['image'], tf.float32),
        'image/id': 0, # dummy int.
    }

    # The following labels are needed by the object detection task.
    label = example['objects']['label'] + 1  # 0 is reserved for padding.
    bbox = example['objects']['bbox']

    # Use tf.numpy_function to get features not easily computed in tf.
    def get_area(bboxes):
      return np.asarray([
          (b[2] - b[0]) * (b[3] - b[1]) for b in bboxes], dtype=np.int32)

    areas = tf.numpy_function(get_area, (bbox,), (tf.int32,))
    areas = tf.reshape(areas, [tf.shape(label)[0]])

    labels = {
        'label': label,
        'bbox': bbox,
        'area': areas,
        'is_crowd': tf.zeros_like(label, tf.bool),
    }
    return features, labels


if __name__=='__main__':
    # Load config for the pretrained model.
    pretrained_model_dir = 'gs://pix2seq/obj365_pretrain/resnet_640x640_b256_s400k/'
    with tf.io.gfile.GFile(os.path.join(pretrained_model_dir, 'config.json'), 'r') as f:
        config = ml_collections.ConfigDict(json.loads(f.read()))

    # Update config for finetuning (some configs were missing at initial pretraining time).
    config.dataset.tfds_name = 'voc'
    config.dataset.batch_duplicates = 1
    config.dataset.coco_annotations_dir = None
    config.task.name == 'object_classification'
    config.task.vocab_id = 10  # object_detection task vocab id.
    config.task.weight = 1.
    config.task.max_instances_per_image_test = 10
    config.task.eval_outputs_json_path = None
    config.tasks = [config.task]
    config.train.batch_size = 8
    config.model.name = 'encoder_ar_decoder'  # name of model and trainer in registries.
    config.model.pretrained_ckpt = pretrained_model_dir
    config.optimization.learning_rate = 1e-4
    config.optimization.warmup_steps = 10

    # Use a smaller image_size to speed up finetuning here.
    # You can use any image_size of choice.
    config.model.image_size = 320
    config.task.image_size = 320


    # Perform training for 1000 steps. This takes about ~20 minutes on a regular Colab GPU.
    train_steps = 1000
    use_tpu = False  # Set this accordingly.
    steps_per_loop = 10
    tf.config.run_functions_eagerly(False)

    strategy = utils.build_strategy(use_tpu=use_tpu, master='')

    # The following snippets are mostly copied and simplified from run.py.
    with strategy.scope():
        # Get dataset.
        dataset = VocDataset(config)

        # Get task.
        task = task_lib.TaskRegistry.lookup(config.task.name)(config)
        tasks = [task]

        # Create tf.data.Dataset.
        ds = dataset.pipeline(
            process_single_example=task.preprocess_single,
            global_batch_size=config.train.batch_size,
            training=True)
        datasets = [ds]
        
        # Setup training elements.
        trainer = model_lib.TrainerRegistry.lookup(config.model.name)(
            config, model_dir='model_dir',
            num_train_examples=dataset.num_train_examples, train_steps=train_steps)
        data_iterators = [iter(dataset) for dataset in datasets]

        @tf.function
        def train_multiple_steps(data_iterators, tasks):
            train_step = lambda xs, ts=tasks: trainer.train_step(xs, ts, strategy)
            for _ in tf.range(steps_per_loop):  # using tf.range prevents unroll.
                with tf.name_scope(''):  # prevent `while_` prefix for variable names.
                    strategy.run(train_step, ([next(it) for it in data_iterators],))

        global_step = trainer.optimizer.iterations
        cur_step = global_step.numpy()
        while cur_step < train_steps:
            train_multiple_steps(data_iterators, tasks)
            cur_step = global_step.numpy()
            print(f"Done training {cur_step} steps.")

