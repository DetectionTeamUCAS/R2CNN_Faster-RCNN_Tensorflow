# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os, sys
sys.path.append("../")
import numpy as np
import time

from libs.networks import build_whole_network
from data.io.read_tfrecord import next_batch
from help_utils import tools
from libs.box_utils.coordinate_convert import back_forward_convert
from libs.box_utils.boxes_utils import get_horizen_minAreaRectangle
from libs.box_utils.show_box_in_tensor import *

os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP


def train():

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=True)

    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                       batch_size=cfgs.BATCH_SIZE,
                       shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                       is_training=True)

        gtboxes_and_label = tf.py_func(back_forward_convert,
                                       inp=[tf.squeeze(gtboxes_and_label_batch, 0)],
                                       Tout=tf.float32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 6])

        gtboxes_and_label_AreaRectangle = get_horizen_minAreaRectangle(gtboxes_and_label)
        gtboxes_and_label_AreaRectangle = tf.reshape(gtboxes_and_label_AreaRectangle, [-1, 5])

    with tf.name_scope('draw_gtboxes'):
        gtboxes_in_img = draw_box_with_color(img_batch, tf.reshape(gtboxes_and_label_AreaRectangle, [-1, 5])[:, :-1],
                                             text=tf.shape(gtboxes_and_label_AreaRectangle)[0])

        gtboxes_rotate_in_img = draw_box_with_color_rotate(img_batch, tf.reshape(gtboxes_and_label, [-1, 6])[:, :-1],
                                                           text=tf.shape(gtboxes_and_label)[0])

    biases_regularizer = tf.no_regularizer
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)

    # list as many types of layers as possible, even if they are not used now
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
        final_boxes_h, final_scores_h, final_category_h, \
        final_boxes_r, final_scores_r, final_category_r, loss_dict = faster_rcnn.build_whole_detection_network(
            input_img_batch=img_batch,
            gtboxes_r_batch=gtboxes_and_label,
            gtboxes_h_batch=gtboxes_and_label_AreaRectangle)

    dets_in_img = draw_boxes_with_categories_and_scores(img_batch=img_batch,
                                                        boxes=final_boxes_h,
                                                        labels=final_category_h,
                                                        scores=final_scores_h)

    dets_rotate_in_img = draw_boxes_with_categories_and_scores_rotate(img_batch=img_batch,
                                                                      boxes=final_boxes_r,
                                                                      labels=final_category_r,
                                                                      scores=final_scores_r)

    # ----------------------------------------------------------------------------------------------------build loss
    weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())
    rpn_location_loss = loss_dict['rpn_loc_loss']
    rpn_cls_loss = loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss

    fastrcnn_cls_loss_h = loss_dict['fastrcnn_cls_loss_h']
    fastrcnn_loc_loss_h = loss_dict['fastrcnn_loc_loss_h']
    fastrcnn_cls_loss_r = loss_dict['fastrcnn_cls_loss_r']
    fastrcnn_loc_loss_r = loss_dict['fastrcnn_loc_loss_r']
    fastrcnn_total_loss = fastrcnn_cls_loss_h + fastrcnn_loc_loss_h + fastrcnn_cls_loss_r + fastrcnn_loc_loss_r

    total_loss = rpn_total_loss + fastrcnn_total_loss + weight_decay_loss
    # ---------------------------------------------------------------------------------------------build loss

    # ---------------------------------------------------------------------------------------------------add summary
    tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
    tf.summary.scalar('RPN_LOSS/location_loss', rpn_location_loss)
    tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)

    tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss_h', fastrcnn_cls_loss_h)
    tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss_h', fastrcnn_loc_loss_h)
    tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss_r', fastrcnn_cls_loss_r)
    tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss_r', fastrcnn_loc_loss_r)
    tf.summary.scalar('FAST_LOSS/fastrcnn_total_loss', fastrcnn_total_loss)

    tf.summary.scalar('LOSS/total_loss', total_loss)
    tf.summary.scalar('LOSS/regular_weights', weight_decay_loss)

    tf.summary.image('img/gtboxes', gtboxes_in_img)
    tf.summary.image('img/gtboxes_rotate', gtboxes_rotate_in_img)
    tf.summary.image('img/dets', dets_in_img)
    tf.summary.image('img/dets_rotate', dets_rotate_in_img)

    # ---------------------------------------------------------------------------------------------add summary

    global_step = slim.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)

    # ---------------------------------------------------------------------------------------------compute gradients
    gradients = faster_rcnn.get_gradients(optimizer, total_loss)

    # enlarge_gradients for bias
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = faster_rcnn.enlarge_gradients_for_bias(gradients)

    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients'):
            gradients = slim.learning.clip_gradient_norms(gradients,
                                                          cfgs.GRADIENT_CLIPPING_BY_NORM)
    # ---------------------------------------------------------------------------------------------compute gradients

    # train_op
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = faster_rcnn.get_restorer()
    saver = tf.train.Saver(max_to_keep=10)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
        tools.mkdir(summary_path)
        summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

        for step in range(cfgs.MAX_ITERATION):
            training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                _, global_stepnp = sess.run([train_op, global_step])

            else:
                if step % cfgs.SHOW_TRAIN_INFO_INTE == 0 and step % cfgs.SMRY_ITER != 0:
                    start = time.time()

                    _global_step, _img_name_batch, _rpn_location_loss, _rpn_classification_loss, \
                    _rpn_total_loss, _fast_rcnn_location_loss, _fast_rcnn_classification_loss, \
                    _fast_rcnn_location_rotate_loss, _fast_rcnn_classification_rotate_loss, \
                    _fast_rcnn_total_loss, _total_loss, _ = \
                        sess.run([global_step, img_name_batch, rpn_location_loss, rpn_cls_loss,
                                  rpn_total_loss, fastrcnn_loc_loss_h, fastrcnn_cls_loss_h,
                                  fastrcnn_loc_loss_r, fastrcnn_cls_loss_r,
                                  fastrcnn_total_loss, total_loss, train_op])

                    end = time.time()
                    print(""" {}: step{}    image_name:{} |\t
                                                    rpn_loc_loss:{} |\t rpn_cla_loss:{} |\t
                                                    rpn_total_loss:{} |
                                                    fast_rcnn_loc_loss:{} |\t fast_rcnn_cla_loss:{} |\t
                                                    fast_rcnn_loc_rotate_loss:{} |\t fast_rcnn_cla_rotate_loss:{} |\t
                                                    fast_rcnn_total_loss:{} |\t
                                                    total_loss:{} |\t pre_cost_time:{}s""" \
                          .format(training_time, _global_step, str(_img_name_batch[0]), _rpn_location_loss,
                                  _rpn_classification_loss, _rpn_total_loss, _fast_rcnn_location_loss,
                                  _fast_rcnn_classification_loss, _fast_rcnn_location_rotate_loss,
                                  _fast_rcnn_classification_rotate_loss, _fast_rcnn_total_loss, _total_loss,
                                  (end - start)))
                else:
                    if step % cfgs.SMRY_ITER == 0:
                        _, global_stepnp, summary_str = sess.run([train_op, global_step, summary_op])
                        summary_writer.add_summary(summary_str, global_stepnp)
                        summary_writer.flush()

            if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or (step == cfgs.MAX_ITERATION - 1):

                save_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_ckpt = os.path.join(save_dir, 'voc_' + str(global_stepnp) + 'model.ckpt')
                saver.save(sess, save_ckpt)
                print(' weights had been saved')

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

    train()

#
















