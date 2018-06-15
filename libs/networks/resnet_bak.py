# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
# from slim.nets import resnet_v1, resnet_utils
# from slim.nets.resnet_v1 import resnet_v1_block
from libs.networks.slim_nets import resnet_v1, resnet_utils
from libs.networks.slim_nets.resnet_v1 import resnet_v1_block


def resnet_arg_scope(
        is_training=True, weight_decay=cfgs.WEIGHT_DECAY, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def modified_resnet_v1_block(scope, bottleneck, parameter_tuple_list):
  """
  Different from The Impletation of Slim. We changed it to same as KaiMing's Paper.
  Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  parameter_list = []
  for a_tuple in parameter_tuple_list:
      parameter_list.append(
          {
              'depth': a_tuple[0],
              'depth_bottleneck': a_tuple[1],
              'stride': a_tuple[2]
          }
      )
  return resnet_utils.Block(scope, bottleneck, parameter_list)


def resnet_50_base(img_batch, is_training=True):
    '''
    this code is from light-head rcnn.

    It is convenient to freeze blocks.
    @author: jemmy li
    @contact: zengarden2009@gmail.com
    '''

    bottleneck = resnet_v1.bottleneck
    #
    blocks = [
        modified_resnet_v1_block('block1', bottleneck,
                           [(256, 64, 1, 1)] * 2 + [(256, 64, 1, 1)]),
        modified_resnet_v1_block('block2', bottleneck,
                           [(512, 128, 2, 1)] + [(512, 128, 1, 1)] * 3),
        modified_resnet_v1_block('block3', bottleneck,
                           [(1024, 256, 2, 1)] + [(1024, 256, 1, 1)] * 5)
    ]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):  # freeze the base conv_net
        with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='SAME', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4-cfgs.FIXED_BLOCKS)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, _ = resnet_v1.resnet_v1(
            net, blocks[0:1], global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, _ = resnet_v1.resnet_v1(
            C2, blocks[1:2], global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, _ = resnet_v1.resnet_v1(
            C3, blocks[2:3], global_pool=False,
            include_root_block=False, scope='resnet_v1_50')

    # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
    return C4


def resnet_101_base(img_batch, is_training):
    '''
        this code is from light-head rcnn.

        It is convenient to freeze blocks.
        @author: jemmy li
        @contact: zengarden2009@gmail.com
    '''

    bottleneck = resnet_v1.bottleneck
    blocks = [
        modified_resnet_v1_block('block1', bottleneck,
                                 [(256, 64, 1, 1)] * 2 + [(256, 64, 1, 1)]),
        modified_resnet_v1_block('block2', bottleneck,
                                 [(512, 128, 2, 1)] + [(512, 128, 1, 1)] * 3),
        modified_resnet_v1_block('block3', bottleneck,
                                 [(1024, 256, 2, 1)] + [(1024, 256, 1, 1)] * 22)
    ]
    with slim.arg_scope(resnet_arg_scope(is_training=False)):  # freeze the base conv_net
        with tf.variable_scope('resnet_v1_101', 'resnet_v1_101'):
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='SAME', scope='pool1')

    not_freezed = [False] * cfgs.FIXED_BLOCKS + (4 - cfgs.FIXED_BLOCKS) * [True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=is_training and not_freezed[0])):
        C2, _ = resnet_v1.resnet_v1(
            net, blocks[0:1], global_pool=False, include_root_block=False,
            scope='resnet_v1_101')

    with slim.arg_scope(resnet_arg_scope(is_training=is_training and not_freezed[1])):
        C3, _ = resnet_v1.resnet_v1(
            C2, blocks[1:2], global_pool=False, include_root_block=False,
            scope='resnet_v1_101')

    with slim.arg_scope(resnet_arg_scope(is_training=is_training and not_freezed[2])):
        C4, _ = resnet_v1.resnet_v1(
            C3, blocks[2:3], global_pool=False,
            include_root_block=False, scope='resnet_v1_101')

    return C4


def restnet_head(input, is_training, scope):

    block4 = [modified_resnet_v1_block('block4', resnet_v1.bottleneck,
                           [(2048, 512, 1, 2)] + [(2048, 512, 1, 2)] * 2)]

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        C5_flatten, _ = resnet_v1.resnet_v1(
            input, block4, global_pool=True,
            spatial_squeeze=True,
            include_root_block=False, scope=scope)
    # global average pooling C5 to obtain fc layers
    return C5_flatten


def build_feature_pyramid(feature_maps_dict):

    '''
    reference: https://github.com/CharlesShang/FastMaskRCNN
    build P2, P3, P4, P5
    :return: multi-scale feature map
    '''

    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid'):
        feature_pyramid['P4'] = slim.conv2d(feature_maps_dict['C4'],
                                            num_outputs=512,
                                            kernel_size=[1, 1],
                                            stride=1,
                                            scope='build_P4')

        feature_pyramid['P5'] = slim.max_pool2d(feature_pyramid['P4'],
                                                kernel_size=[2, 2], stride=2, scope='build_P5')
        # P6 is down sample of P5

        for layer in range(3, 1, -1):
            p, c = feature_pyramid['P' + str(layer + 1)], feature_maps_dict['C' + str(layer)]
            up_sample_shape = tf.shape(c)
            up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                         name='build_P%d/up_sample_nearest_neighbor' % layer)

            c = slim.conv2d(c, num_outputs=512, kernel_size=[1, 1], stride=1,
                            scope='build_P%d/reduce_dimension' % layer)
            p = up_sample + c
            p = slim.conv2d(p, 512, kernel_size=[3, 3], stride=1,
                            padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
            feature_pyramid['P' + str(layer)] = p

    return feature_pyramid































