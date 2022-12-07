#!/bin/bash

# Train (det)
model_dir=./checkpoint_det
python3 run.py --mode=train --model_dir=$model_dir --config=configs/config_det_finetune.py \
                                        --config.train.batch_size=32 --config.train.epochs=60 --config.optimization.learning_rate=3e-5 \
                                        --config.dataset.train_file_pattern="/data/sung/dataset/coco/tfrecord/train*" \
                                        --config.dataset.val_file_pattern="/data/sung/dataset/coco/tfrecord/val*"

# Train (cls)
model_dir=./checkpoint_cls
CUDA_VISIBLE_DEVICES=0,1 python3 run.py --mode=train --model_dir=$model_dir --config=configs/config_cls_finetune.py \
                                        --config.train.batch_size=16 --config.train.epochs=60 --config.optimization.learning_rate=3e-5 \
                                        --config.dataset.train_file_pattern="/data/sung/dataset/coco/tfrecord/train*" \
                                        --config.dataset.val_file_pattern="/data/sung/dataset/coco/tfrecord/val*"


# Train (cls+det)
model_dir=./checkpoint_cls_det
CUDA_VISIBLE_DEVICES=2,3 python3 run.py --mode=train --model_dir=$model_dir --config=configs/config_det_cls.py \
                                        --config.train.batch_size=16 --config.train.epochs=20 --config.optimization.learning_rate=3e-5 \
                                        --config.dataset.train_file_pattern="/data/sung/dataset/coco/tfrecord/train*" \
                                        --config.dataset.val_file_pattern="/data/sung/dataset/coco/tfrecord/val*"


# Evaluation (det)
model_dir=checkpoint
boxes_json_path=$model_dir/boxes.json
python3 run.py --mode=eval --model_dir=$model_dir --config=configs/config_det_finetune.py --config.dataset.val_file_pattern="/data/sung/dataset/coco/tfrecord/val*" --config.eval.batch_size=40