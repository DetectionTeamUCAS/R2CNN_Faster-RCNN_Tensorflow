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
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer import proposal_target_layer
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

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.draw_box_with_color(img, positive_anchor, tf.shape(positive_anchor)[0])
        neg_in_img = show_box_in_tensor.draw_box_with_color(img, negative_anchor, tf.shape(positive_anchor)[0])

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = show_box_in_tensor.draw_box_with_color(img, pos_roi, tf.shape(pos_roi)[0])
        neg_in_img = show_box_in_tensor.draw_box_with_color(img, neg_roi, tf.shape(neg_roi)[0])

        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred_h, bbox_targets_h, cls_score_h, bbox_pred_r, bbox_targets_r, cls_score_r, labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred_h: [-1, 4*(cls_num+1)]
        :param bbox_targets_h: [-1, 4*(cls_num+1)]
        :param cls_score_h: [-1, cls_num+1]
        :param bbox_pred_r: [-1, 5*(cls_num+1)]
        :param bbox_targets_r: [-1, 5*(cls_num+1)]
        :param cls_score_r: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss') as sc:
            with tf.variable_scope('rpn_loss'):

                rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                          bbox_targets=rpn_bbox_targets,
                                                          label=rpn_labels,
                                                          sigma=cfgs.RPN_SIGMA)
                # rpn_cls_loss:
                # rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
                # rpn_labels = tf.reshape(rpn_labels, [-1])
                # ensure rpn_labels shape is [-1]
                rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
                rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                             labels=rpn_labels))

                rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss_h = losses.smooth_l1_loss_rcnn_h(bbox_pred=bbox_pred_h,
                                                               bbox_targets=bbox_targets_h,
                                                               label=labels,
                                                               num_classes=cfgs.CLASS_NUM + 1,
                                                               sigma=cfgs.FASTRCNN_SIGMA)

                    # cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                    # labels = tf.reshape(labels, [-1])
                    cls_loss_h = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score_h,
                        labels=labels))  # beacause already sample before

                    bbox_loss_r = losses.smooth_l1_loss_rcnn_r(bbox_pred=bbox_pred_r,
                                                               bbox_targets=bbox_targets_r,
                                                               label=labels,
                                                               num_classes=cfgs.CLASS_NUM + 1,
                                                               sigma=cfgs.FASTRCNN_SIGMA)

                    cls_loss_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score_r,
                        labels=labels))
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss = bbox_loss = losses.sum_ohem_loss(
                        cls_score=cls_score_h,
                        label=labels,
                        bbox_targets=bbox_targets_h,
                        nr_ohem_sampling=128,
                        nr_classes=cfgs.CLASS_NUM + 1)

                cls_loss_h = cls_loss_h * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss_h = bbox_loss_h * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
                cls_loss_r = cls_loss_r * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                bbox_loss_r = bbox_loss_r * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss_h': cls_loss_h,
                'fastrcnn_loc_loss_h': bbox_loss_h,
                'fastrcnn_cls_loss_r': cls_loss_r,
                'fastrcnn_loc_loss_r': bbox_loss_r,
            }
        return loss_dict

    def build_whole_detection_network(self, input_img_batch, gtboxes_r_batch, gtboxes_h_batch):

        if self.is_training:
            # ensure shape is [M, 5] and [M, 6]
            gtboxes_r_batch = tf.reshape(gtboxes_r_batch, [-1, 6])
            gtboxes_h_batch = tf.reshape(gtboxes_h_batch, [-1, 5])
            gtboxes_r_batch = tf.cast(gtboxes_r_batch, tf.float32)
            gtboxes_h_batch = tf.cast(gtboxes_h_batch, tf.float32)

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
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.is_training:
                rois_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=input_img_batch,
                                                                            boxes=rois,
                                                                            scores=roi_scores)
                tf.summary.image('all_rpn_rois', rois_in_img)

                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=input_img_batch,
                                                                                    boxes=score_gre_05_rois,
                                                                                    scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                rpn_labels, rpn_bbox_targets = \
                    tf.py_func(
                        anchor_target_layer,
                        [gtboxes_h_batch, img_shape, anchors],
                        [tf.float32, tf.float32])
                rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
                rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
                rpn_labels = tf.reshape(rpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, anchors, rpn_labels)

            # --------------------------------------add smry-----------------------------------------------------------

            rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
            rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/rpn_accuracy', acc)

            with tf.control_dependencies([rpn_labels]):
                with tf.variable_scope('sample_RCNN_minibatch'):
                    rois, labels, bbox_targets_h, bbox_targets_r = \
                    tf.py_func(proposal_target_layer,
                               [rois, gtboxes_h_batch, gtboxes_r_batch],
                               [tf.float32, tf.float32, tf.float32, tf.float32])

                    rois = tf.reshape(rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets_h = tf.reshape(bbox_targets_h, [-1, 4*(cfgs.CLASS_NUM+1)])
                    bbox_targets_r = tf.reshape(bbox_targets_r, [-1, 5*(cfgs.CLASS_NUM+1)])
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels)

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
        if self.is_training:
            cls_category_h = tf.argmax(cls_prob_h, axis=1)
            fast_acc_h = tf.reduce_mean(tf.to_float(tf.equal(cls_category_h, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc_h', fast_acc_h)

            cls_category_r = tf.argmax(cls_prob_r, axis=1)
            fast_acc_r = tf.reduce_mean(tf.to_float(tf.equal(cls_category_r, tf.to_int64(labels))))
            tf.summary.scalar('ACC/fast_acc_r', fast_acc_r)

        #  6. postprocess_fastrcnn
        if not self.is_training:
            final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                          bbox_ppred=bbox_pred_h,
                                                                                          scores=cls_prob_h,
                                                                                          img_shape=img_shape)
            final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                          bbox_ppred=bbox_pred_r,
                                                                                          scores=cls_prob_r,
                                                                                          img_shape=img_shape)
            return final_boxes_h, final_scores_h, final_category_h, final_boxes_r, final_scores_r, final_category_r
        else:
            '''
            when trian. We need build Loss
            '''
            loss_dict = self.build_loss(rpn_box_pred=rpn_box_pred,
                                        rpn_bbox_targets=rpn_bbox_targets,
                                        rpn_cls_score=rpn_cls_score,
                                        rpn_labels=rpn_labels,
                                        bbox_pred_h=bbox_pred_h,
                                        bbox_targets_h=bbox_targets_h,
                                        cls_score_h=cls_score_h,
                                        bbox_pred_r=bbox_pred_r,
                                        bbox_targets_r=bbox_targets_r,
                                        cls_score_r=cls_score_r,
                                        labels=labels)

            final_boxes_h, final_scores_h, final_category_h = self.postprocess_fastrcnn_h(rois=rois,
                                                                                          bbox_ppred=bbox_pred_h,
                                                                                          scores=cls_prob_h,
                                                                                          img_shape=img_shape)
            final_boxes_r, final_scores_r, final_category_r = self.postprocess_fastrcnn_r(rois=rois,
                                                                                          bbox_ppred=bbox_pred_r,
                                                                                          scores=cls_prob_r,
                                                                                          img_shape=img_shape)

            return final_boxes_h, final_scores_h, final_category_h, \
                   final_boxes_r, final_scores_r, final_category_r, loss_dict

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

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients




















