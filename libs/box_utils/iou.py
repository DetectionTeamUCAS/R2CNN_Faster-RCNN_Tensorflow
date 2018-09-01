# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def iou_calculate(boxes_1, boxes_2):
    '''

    :param boxes_1: [N, 4] [ymin, xmin, ymax, xmax]
    :param boxes_2: [M, 4] [ymin, xmin. ymax, xmax]
    :return:
    '''
    with tf.name_scope('iou_caculate'):

        ymin_1, xmin_1, ymax_1, xmax_1 = tf.split(boxes_1, 4, axis=1)  # ymin_1 shape is [N, 1]..

        ymin_2, xmin_2, ymax_2, xmax_2 = tf.unstack(boxes_2, axis=1)  # ymin_2 shape is [M, ]..

        max_xmin = tf.maximum(xmin_1, xmin_2)
        min_xmax = tf.minimum(xmax_1, xmax_2)

        max_ymin = tf.maximum(ymin_1, ymin_2)
        min_ymax = tf.minimum(ymax_1, ymax_2)

        overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
        overlap_w = tf.maximum(0., min_xmax - max_xmin)

        overlaps = overlap_h * overlap_w

        area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
        area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

        iou = overlaps / (area_1 + area_2 - overlaps)

        return iou
