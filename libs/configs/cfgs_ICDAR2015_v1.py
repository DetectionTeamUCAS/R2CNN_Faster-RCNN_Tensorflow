# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf


# ------------------------------------------------
VERSION = 'R2CNN_20181011_ICDAR2015_v1'
NET_NAME = 'resnet_v1_101'
ADD_BOX_IN_TENSORBOARD = True
# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "2"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 2000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith('resnet'):
    weights_name = NET_NAME
elif NET_NAME.startswith('MobilenetV2'):
    weights_name = 'mobilenet/mobilenet_v2_1.0_224'
else:
    raise NotImplementedError

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')

EVALUATE_H_DIR = ROOT_PATH + '/output' + '/evaluate_h_result_pickle/' + VERSION
EVALUATE_R_DIR = ROOT_PATH + '/output' + '/evaluate_r_result_pickle/' + VERSION
TEST_ANNOTATION_PATH = '/mnt/USBB/gx/DOTA/DOTA_clip/val/labeltxt'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
ROTATE_NMS_USE_GPU = True
FIXED_BLOCKS = 2  # allow 0~3

RPN_LOCATION_LOSS_WEIGHT = 1 / 7
RPN_CLASSIFICATION_LOSS_WEIGHT = 2.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 4.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 2.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0


MUTILPY_BIAS_GRADIENT = None  # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.0003  # 0.0003
DECAY_STEP = [30000, 60000]  # 90000, 120000
MAX_ITERATION = 100000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'ICDAR2015'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 720
IMG_MAX_LENGTH = 2000
CLASS_NUM = 1

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.0001


# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [16]  # can not be modified in most situations
ANCHOR_SCALES = [0.0625, 0.125, 0.25, 0.5, 1., 2.0]  # [4, 8, 16, 32]
ANCHOR_RATIOS = [1, 1 / 2, 2., 1 / 3., 3., 5., 1 / 4., 4., 1 / 5., 6., 1 / 6., 7., 1 / 7.]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None


# --------------------------------------------RPN config
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 10000  # 5000
RPN_MAXIMUM_PROPOSAL_TEST = 300  # 300


# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.5  # only show in tensorboard

FAST_RCNN_NMS_IOU_THRESHOLD = 0.1  # 0.6
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 150
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.4
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.35

ADD_GTBOXES_TO_TRAIN = False



