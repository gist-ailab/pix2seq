#!/bin/bash

# # COCO
# python create_coco_tfrecord_detonly.py --coco_image_dir /data/sung/dataset/coco/train2017 \
#                                        --ins_ann_file /data/sung/dataset/coco/annotations/instances_train2017.json \
#                                        --output_dir /data/sung/dataset/coco/tfrecord/train



# python create_coco_tfrecord_detonly.py --coco_image_dir /data/sung/dataset/coco/val2017 \
#                                        --ins_ann_file /data/sung/dataset/coco/annotations/instances_val2017.json \
#                                        --output_dir /data/sung/dataset/coco/tfrecord/val


# VOC
python create_coco_tfrecord_detonly.py --coco_image_dir /home/sung/dataset/VOC2012/JPEGImages \
                                       --ins_ann_file /home/sung/dataset/VOC2012/split0/train.json \
                                       --output_dir /home/sung/dataset/VOC2012/split0/tfrecord/train


python create_coco_tfrecord_detonly.py --coco_image_dir /home/sung/dataset/VOC2012/JPEGImages \
                                       --ins_ann_file /home/sung/dataset/VOC2012/split0/val.json \
                                       --output_dir /home/sung/dataset/VOC2012/split0/tfrecord/val