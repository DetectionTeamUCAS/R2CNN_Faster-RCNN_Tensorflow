# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math



# def encode_boxes(ex_rois, gt_rois, scale_factor=None):
#     ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
#     ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
#     ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
#     ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
#
#     gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
#     gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
#     gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
#     gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
#
#     targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
#     targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
#     targets_dw = np.log(gt_widths / ex_widths)
#     targets_dh = np.log(gt_heights / ex_heights)
#
#     if scale_factor:
#         targets_dx = targets_dx * scale_factor[0]
#         targets_dy = targets_dy * scale_factor[1]
#         targets_dw = targets_dw * scale_factor[2]
#         targets_dh = targets_dh * scale_factor[3]
#
#     targets = np.vstack(
#         (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
#     return targets
#
#
# def _concat_new_axis(t1, t2, t3, t4, axis):
#     return tf.concat(
#         [tf.expand_dims(t1, -1), tf.expand_dims(t2, -1),
#          tf.expand_dims(t3, -1), tf.expand_dims(t4, -1)], axis=axis)
#
#
# def decode_boxes(boxes, deltas, scale_factor=None):
#     widths = boxes[:, 2] - boxes[:, 0] + 1.0
#     heights = boxes[:, 3] - boxes[:, 1] + 1.0
#     ctr_x = tf.expand_dims(boxes[:, 0] + 0.5 * widths, -1)
#     ctr_y = tf.expand_dims(boxes[:, 1] + 0.5 * heights, -1)
#
#     dx = deltas[:, 0::4]
#     dy = deltas[:, 1::4]
#     dw = deltas[:, 2::4]
#     dh = deltas[:, 3::4]
#
#     if scale_factor:
#         dx /= scale_factor[0]
#         dy /= scale_factor[1]
#         dw /= scale_factor[2]
#         dh /= scale_factor[3]
#
#     widths = tf.expand_dims(widths, -1)
#     heights = tf.expand_dims(heights, -1)
#
#     pred_ctr_x = dx * widths + ctr_x
#     pred_ctr_y = dy * heights + ctr_y
#     pred_w = tf.exp(dw) * widths
#     pred_h = tf.exp(dh) * heights
#
#     # x1
#     # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
#     pred_x1 = pred_ctr_x - 0.5 * pred_w
#     # y1
#     # pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
#     pred_y1 = pred_ctr_y - 0.5 * pred_h
#     # x2
#     # pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
#     pred_x2 = pred_ctr_x + 0.5 * pred_w
#     # y2
#     # pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
#     pred_y2 = pred_ctr_y + 0.5 * pred_h
#
#     pred_boxes = _concat_new_axis(pred_x1, pred_y1, pred_x2, pred_y2, 2)
#     pred_boxes = tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], -1))
#     return pred_boxes


def decode_boxes(encode_boxes, reference_boxes, scale_factors=None):
    '''

    :param encoded_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale.

    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by first stage
    :return:decode boxes [N, 4]
    '''

    t_xcenter, t_ycenter, t_w, t_h = tf.unstack(encode_boxes, axis=1)
    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
    # reference boxes are anchors in the first stage

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    predict_xcenter = t_xcenter * reference_w + reference_xcenter
    predict_ycenter = t_ycenter * reference_h + reference_ycenter
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h

    predict_xmin = predict_xcenter - predict_w / 2.
    predict_xmax = predict_xcenter + predict_w / 2.
    predict_ymin = predict_ycenter - predict_h / 2.
    predict_ymax = predict_ycenter + predict_h / 2.

    return tf.transpose(tf.stack([predict_xmin, predict_ymin,
                                  predict_xmax, predict_ymax]))


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None):
    '''

    :param unencode_boxes: [-1, 4]
    :param reference_boxes: [-1, 4]
    :return: encode_boxes [-1, 4]
    '''

    xmin, ymin, xmax, ymax = unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = reference_boxes[:, 0], reference_boxes[:, 1], \
                                                                     reference_boxes[:, 2], reference_boxes[:, 3]

    x_center = (xmin + xmax) / 2.
    y_center = (ymin + ymax) / 2.
    w = xmax - xmin + 1e-8
    h = ymax - ymin + 1e-8

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin + 1e-8
    reference_h = reference_ymax - reference_ymin + 1e-8

    # w + 1e-8 to avoid NaN in division and log below

    t_xcenter = (x_center - reference_xcenter) / reference_w
    t_ycenter = (y_center - reference_ycenter) / reference_h
    t_w = np.log(w/reference_w)
    t_h = np.log(h/reference_h)

    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]

    return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h], axis=0))


def decode_boxes_rotate(encode_boxes, reference_boxes, scale_factors=None):
    '''

    :param encode_boxes:[N, 5]
    :param reference_boxes: [N, 5] .
    :param scale_factors: use for scale
    in the rpn stage, reference_boxes are anchors
    in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 5]
    '''

    t_xcenter, t_ycenter, t_w, t_h, t_theta = tf.unstack(encode_boxes, axis=1)
    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]
        t_theta /= scale_factors[4]
    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
    reference_x_center = (reference_xmin + reference_xmax) / 2.
    reference_y_center = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin
    reference_theta = tf.ones(tf.shape(reference_xmin)) * -90
    predict_x_center = t_xcenter * reference_w + reference_x_center
    predict_y_center = t_ycenter * reference_h + reference_y_center
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h
    predict_theta = t_theta * 180 / math.pi + reference_theta
    # mask1 = tf.less(predict_theta, -90)
    # mask2 = tf.greater_equal(predict_theta, -180)
    # mask7 = tf.less(predict_theta, -180)
    # mask8 = tf.greater_equal(predict_theta, -270)
    #
    # mask3 = tf.greater_equal(predict_theta, 0)
    # mask4 = tf.less(predict_theta, 90)
    # mask5 = tf.greater_equal(predict_theta, 90)
    # mask6 = tf.less(predict_theta, 180)
    #
    # # to keep range in [-90, 0)
    # # [-180, -90)
    # convert_mask = tf.logical_and(mask1, mask2)
    # remain_mask = tf.logical_not(convert_mask)
    # predict_theta += tf.cast(convert_mask, tf.float32) * 90.
    #
    # remain_h = tf.cast(remain_mask, tf.float32) * predict_h
    # remain_w = tf.cast(remain_mask, tf.float32) * predict_w
    # convert_h = tf.cast(convert_mask, tf.float32) * predict_h
    # convert_w = tf.cast(convert_mask, tf.float32) * predict_w
    #
    # predict_h = remain_h + convert_w
    # predict_w = remain_w + convert_h
    #
    # # [-270, -180)
    # cond4 = tf.cast(tf.logical_and(mask7, mask8), tf.float32) * 180.
    # predict_theta += cond4
    #
    # # [0, 90)
    # # cond2 = tf.cast(tf.logical_and(mask3, mask4), tf.float32) * 90.
    # # predict_theta -= cond2
    #
    # convert_mask1 = tf.logical_and(mask3, mask4)
    # remain_mask1 = tf.logical_not(convert_mask1)
    # predict_theta -= tf.cast(convert_mask1, tf.float32) * 90.
    #
    # remain_h = tf.cast(remain_mask1, tf.float32) * predict_h
    # remain_w = tf.cast(remain_mask1, tf.float32) * predict_w
    # convert_h = tf.cast(convert_mask1, tf.float32) * predict_h
    # convert_w = tf.cast(convert_mask1, tf.float32) * predict_w
    #
    # predict_h = remain_h + convert_w
    # predict_w = remain_w + convert_h
    #
    # # [90, 180)
    # cond3 = tf.cast(tf.logical_and(mask5, mask6), tf.float32) * 180.
    # predict_theta -= cond3
    decode_boxes = tf.transpose(tf.stack([predict_x_center, predict_y_center,
                                          predict_w, predict_h, predict_theta]))
    return decode_boxes


def encode_boxes_rotate(unencode_boxes, reference_boxes, scale_factors=None):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
    :param reference_boxes: [H*W*num_anchors_per_location, 5]
    :return: encode_boxes [-1, 5]
    '''
    x_center, y_center, w, h, theta = \
        unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3], unencode_boxes[:, 4]
    reference_xmin, reference_ymin, reference_xmax, reference_ymax = \
        reference_boxes[:, 0], reference_boxes[:, 1], reference_boxes[:, 2], reference_boxes[:, 3]
    reference_x_center = (reference_xmin + reference_xmax) / 2.
    reference_y_center = (reference_ymin + reference_ymax) / 2.
    # here maybe have logical error, reference_w and reference_h should exchange,
    # but it doesn't seem to affect the result.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin
    reference_theta = np.ones(reference_xmin.shape) * -90
    reference_w += 1e-8
    reference_h += 1e-8
    w += 1e-8
    h += 1e-8  # to avoid NaN in division and log below
    t_xcenter = (x_center - reference_x_center) / reference_w
    t_ycenter = (y_center - reference_y_center) / reference_h
    t_w = np.log(w / reference_w)
    t_h = np.log(h / reference_h)
    t_theta = (theta - reference_theta) * math.pi / 180
    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]
        t_theta *= scale_factors[4]
    return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h, t_theta]))