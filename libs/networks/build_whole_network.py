# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.box_utils import nms_rotate


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_fastrcnn_h(self, rois, bbox_ppred, scores, img_shape):

        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_h'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encode_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def postprocess_fastrcnn_r(self, rois, bbox_ppred, scores, img_shape):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 5]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn_r'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 5])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM+1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes_rotate(encode_boxes=tmp_encoded_box,
                                                                          reference_boxes=rois,
                                                                          scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                # tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                #                                                              img_shape=img_shape)

                # 3. NMS
                keep = nms_rotate.nms_rotate(decode_boxes=tmp_decoded_boxes,
                                             scores=tmp_score,
                                             iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             use_angle_condition=False,
                                             angle_threshold=15,
                                             use_gpu=cfgs.ROTATE_NMS_USE_GPU)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            # if self.is_training:
            '''
            in training. We should show the detecitons in the tensorboard. So we add this.
            '''
            kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])
            final_boxes = tf.gather(final_boxes, kept_indices)
            final_scores = tf.gather(final_scores, kept_indices)
            final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape):
        '''
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('ROI_Warping'):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ],
                                                                             dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )
            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)

        return roi_features

    def build_fastrcnn(self, feature_to_cropped, rois, img_shape):

        with tf.variable_scope('Fast-RCNN'):
            # 5. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                pooled_features = self.roi_pooling(feature_maps=feature_to_cropped, rois=rois, img_shape=img_shape)

            # 6. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):
                fc_flatten = resnet.restnet_head(input=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name)
            elif self.base_network_name.startswith('MobilenetV2'):
                fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                           is_training=self.is_training)
            else:
                raise NotImplementedError('only support resnet and mobilenet')

            # 7. cls and reg in Fast-RCNN
            with tf.variable_scope('horizen_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                    cls_score_h = slim.fully_connected(fc_flatten,
                                                       num_outputs=cfgs.CLASS_NUM+1,
                                                       weights_initializer=cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='cls_fc_h')

                    bbox_pred_h = slim.fully_connected(fc_flatten,
                                                       num_outputs=(cfgs.CLASS_NUM+1) * 4,
                                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='reg_fc_h')
                    # for convient. It also produce (cls_num +1) bboxes

                    cls_score_h = tf.reshape(cls_score_h, [-1, cfgs.CLASS_NUM+1])
                    bbox_pred_h = tf.reshape(bbox_pred_h, [-1, 4*(cfgs.CLASS_NUM+1)])

            with tf.variable_scope('rotation_branch'):
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
                    cls_score_r = slim.fully_connected(fc_flatten,
                                                       num_outputs=cfgs.CLASS_NUM + 1,
                                                       weights_initializer=cfgs.INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='cls_fc_r')

                    bbox_pred_r = slim.fully_connected(fc_flatten,
                                                       num_outputs=(cfgs.CLASS_NUM + 1) * 5,
                                                       weights_initializer=cfgs.BBOX_INITIALIZER,
                                                       activation_fn=None, trainable=self.is_training,
                                                       scope='reg_fc_r')
                    # for convient. It also produce (cls_num +1) bboxes
                    cls_score_r = tf.reshape(cls_score_r, [-1, cfgs.CLASS_NUM + 1])
                    bbox_pred_r = tf.reshape(bbox_pred_r, [-1, 5 * (cfgs.CLASS_NUM + 1)])

            return bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r

    def build_whole_detection_network(self, input_img_batch, gtboxes_r_batch, gtboxes_h_batch):

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        feature_to_cropped = self.build_base_network(input_img_batch)

        # 2. build rpn
        with tf.variable_scope('build_rpn',
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

            rpn_conv3x3 = slim.conv2d(
                feature_to_cropped, 512, [3, 3],
                trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                activation_fn=tf.nn.relu,
                scope='rpn_conv/3x3')
            rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*2, [1, 1], stride=1,
                                        trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                                        activation_fn=None,
                                        scope='rpn_cls_score')
            rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location*4, [1, 1], stride=1,
                                       trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                       activation_fn=None,
                                       scope='rpn_bbox_pred')
            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

        # 3. generate_anchors
        featuremap_height, featuremap_width = tf.shape(feature_to_cropped)[1], tf.shape(feature_to_cropped)[2]
        featuremap_height = tf.cast(featuremap_height, tf.float32)
        featuremap_width = tf.cast(featuremap_width, tf.float32)

        anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                            anchor_scales=cfgs.ANCHOR_SCALES, anchor_ratios=cfgs.ANCHOR_RATIOS,
                                            featuremap_height=featuremap_height,
                                            featuremap_width=featuremap_width,
                                            stride=cfgs.ANCHOR_STRIDE,
                                            name="make_anchors_forRPN")

        # with tf.variable_scope('make_anchors'):
        #     anchors = anchor_utils.make_anchors(height=featuremap_height,
        #                                         width=featuremap_width,
        #                                         feat_stride=cfgs.ANCHOR_STRIDE[0],
        #                                         anchor_scales=cfgs.ANCHOR_SCALES,
        #                                         anchor_ratios=cfgs.ANCHOR_RATIOS, base_size=16
        #                                         )

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_RPN'):
            # rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
            # rpn_cls_prob = slim.softmax(rpn_cls_prob, scope='rpn_cls_prob')
            # rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                         rpn_cls_prob=rpn_cls_prob,
                                                         img_shape=img_shape,
                                                         anchors=anchors,
                                                         is_training=self.is_training)
            # rois shape [-1, 4]

        # -------------------------------------------------------------------------------------------------------------#
        #                                            Fast-RCNN                                                         #
        # -------------------------------------------------------------------------------------------------------------#

        # 5. build Fast-RCNN
        # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
        bbox_pred_h, cls_score_h, bbox_pred_r, cls_score_r = self.build_fastrcnn(feature_to_cropped=feature_to_cropped,
                                                                                 rois=rois,
                                                                                 img_shape=img_shape)
        # bbox_pred shape: [-1, 4*(cls_num+1)].
        # cls_score shape： [-1, cls_num+1]

        cls_prob_h = slim.softmax(cls_score_h, 'cls_prob_h')
        cls_prob_r = slim.softmax(cls_score_r, 'cls_prob_r')

        # ----------------------------------------------add smry-------------------------------------------------------

        #  6. postprocess_fastrcnn
        final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                      bbox_ppred=bbox_pred_h,
                                                                                      scores=cls_prob_h,
                                                                                      img_shape=img_shape)
        final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                      bbox_ppred=bbox_pred_r,
                                                                                      scores=cls_prob_r,
                                                                                      img_shape=img_shape)
        return final_boxes_h, final_scores_h, final_category_h, final_boxes_r, final_scores_r, final_category_r

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # print(model_variables)

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name+'/block4'):
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"---")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path




















