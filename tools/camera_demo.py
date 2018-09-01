# -*- coding: utf-8 -*- 

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
import argparse

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils import coordinate_convert


def testCamera():
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def inference(det_net, cap):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     is_resize=False)

    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_h_batch=None,
                                                                                      gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')
        fourcc = cv2.VideoWriter_fourcc(*'xvid')
        out = cv2.VideoWriter('./camera_demo.avi', fourcc, 10, (640, 480))
        ret, frame = cap.read()
        while ret:
            ret, frame = cap.read()
            start = time.time()
            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
            det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, det_boxes_h, det_scores_h, det_category_h, det_boxes_r, det_scores_r, det_category_r],
                    feed_dict={img_plac: frame}
                )
            end = time.time()
            # det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
            #                                                boxes=det_boxes_h_,
            #                                                labels=det_category_h_,
            #                                                scores=det_scores_h_)
            det_detections_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                  boxes=det_boxes_r_,
                                                                  labels=det_category_r_,
                                                                  scores=det_scores_r_)
            # det_detections_h = cv2.resize(det_detections_h,
            #                               (det_detections_h.shape[0] // 2, det_detections_h.shape[1] // 2))
            # cv2.putText(det_detections_h,
            #             text="horizon bbox",
            #             org=(0, 0),
            #             fontFace=3,
            #             fontScale=1,
            #             color=(255, 0, 0))
            # det_detections_r = cv2.resize(det_detections_r,
            #                               (det_detections_r.shape[0] // 2, det_detections_r.shape[1] // 2))
            cv2.putText(det_detections_r,
                        text="rotated bbox--%3.2f"%(1/(end-start)),
                        org=(0, 10),
                        fontFace=1,
                        fontScale=1,
                        color=(0, 255, 0))
            out.write(det_detections_r)
            # hmerge = np.hstack((det_detections_h, det_detections_r))  # 水平拼接
            cv2.imshow("faceDetection", det_detections_r)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)
    cap = cv2.VideoCapture(0)
    inference(det_net, cap)

















