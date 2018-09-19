# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

sys.path.append("../")
import cv2
import numpy as np
from timeit import default_timer as timer
import argparse
import tensorflow as tf

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from libs.box_utils import coordinate_convert
from libs.label_name_dict.label_dict import *
from help_utils import tools
from libs.box_utils import nms_rotate
from libs.box_utils import nms
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms


def get_file_paths_recursive(folder=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list

    for dir_path, dir_names, file_names in os.walk(folder):
        for file_name in file_names:
            if file_ext is None:
                file_list.append(os.path.join(dir_path, file_name))
                continue
            if file_name.endswith(file_ext):
                file_list.append(os.path.join(dir_path, file_name))
    return file_list


def inference(det_net, file_paths, des_folder, h_len, w_len, h_overlap, w_overlap, save_res=False):

    if save_res:
        assert cfgs.SHOW_SCORE_THRSHOLD >= 0.5, \
            'please set score threshold (example: SHOW_SCORE_THRSHOLD = 0.5) in cfgs.py'

    else:
        assert cfgs.SHOW_SCORE_THRSHOLD < 0.005, \
            'please set score threshold (example: SHOW_SCORE_THRSHOLD = 0.00) in cfgs.py'

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

        for count, img_path in enumerate(file_paths):
            start = timer()
            img = cv2.imread(img_path)

            box_res = []
            label_res = []
            score_res = []
            box_res_rotate = []
            label_res_rotate = []

            score_res_rotate = []

            imgH = img.shape[0]
            imgW = img.shape[1]

            if imgH < h_len:
                temp = np.zeros([h_len, imgW, 3], np.float32)
                temp[0:imgH, :, :] = img
                img = temp
                imgH = h_len

            if imgW < w_len:
                temp = np.zeros([imgH, w_len, 3], np.float32)
                temp[:, 0:imgW, :] = img
                img = temp
                imgW = w_len

            for hh in range(0, imgH, h_len - h_overlap):
                if imgH - hh - 1 < h_len:
                    hh_ = imgH - h_len
                else:
                    hh_ = hh
                for ww in range(0, imgW, w_len - w_overlap):
                    if imgW - ww - 1 < w_len:
                        ww_ = imgW - w_len
                    else:
                        ww_ = ww
                    src_img = img[hh_:(hh_ + h_len), ww_:(ww_ + w_len), :]

                    det_boxes_h_, det_scores_h_, det_category_h_, \
                    det_boxes_r_, det_scores_r_, det_category_r_ = \
                        sess.run(
                            [det_boxes_h, det_scores_h, det_category_h,
                             det_boxes_r, det_scores_r, det_category_r],
                            feed_dict={img_plac: src_img}
                        )

                    if len(det_boxes_h_) > 0:
                        for ii in range(len(det_boxes_h_)):
                            box = det_boxes_h_[ii]
                            box[0] = box[0] + ww_
                            box[1] = box[1] + hh_
                            box[2] = box[2] + ww_
                            box[3] = box[3] + hh_
                            box_res.append(box)
                            label_res.append(det_category_h_[ii])
                            score_res.append(det_scores_h_[ii])
                    if len(det_boxes_r_) > 0:
                        for ii in range(len(det_boxes_r_)):
                            box_rotate = det_boxes_r_[ii]
                            box_rotate[0] = box_rotate[0] + ww_
                            box_rotate[1] = box_rotate[1] + hh_
                            box_res_rotate.append(box_rotate)
                            label_res_rotate.append(det_category_r_[ii])
                            score_res_rotate.append(det_scores_r_[ii])

            box_res = np.array(box_res)
            label_res = np.array(label_res)
            score_res = np.array(score_res)

            box_res_rotate = np.array(box_res_rotate)
            label_res_rotate = np.array(label_res_rotate)
            score_res_rotate = np.array(score_res_rotate)

            box_res_rotate_, label_res_rotate_, score_res_rotate_ = [], [], []
            box_res_, label_res_, score_res_ = [], [], []

            r_threshold = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
                           'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.05, 'plane': 0.3,
                           'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
                           'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3}

            h_threshold = {'roundabout': 0.35, 'tennis-court': 0.35, 'swimming-pool': 0.4, 'storage-tank': 0.3,
                           'soccer-ball-field': 0.3, 'small-vehicle': 0.4, 'ship': 0.35, 'plane': 0.35,
                           'large-vehicle': 0.4, 'helicopter': 0.4, 'harbor': 0.3, 'ground-track-field': 0.4,
                           'bridge': 0.3, 'basketball-court': 0.4, 'baseball-diamond': 0.3}

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(label_res_rotate == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_r = box_res_rotate[index]
                tmp_label_r = label_res_rotate[index]
                tmp_score_r = score_res_rotate[index]

                tmp_boxes_r = np.array(tmp_boxes_r)
                tmp = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                tmp[:, 0:-1] = tmp_boxes_r
                tmp[:, -1] = np.array(tmp_score_r)

                try:
                    inx = nms_rotate.nms_rotate_cpu(boxes=np.array(tmp_boxes_r),
                                                    scores=np.array(tmp_score_r),
                                                    iou_threshold=r_threshold[LABEl_NAME_MAP[sub_class]],
                                                    max_output_size=500)
                except:
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                    inx = rotate_gpu_nms(np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                         float(r_threshold[LABEl_NAME_MAP[sub_class]]), 0)

                box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                label_res_rotate_.extend(np.array(tmp_label_r)[inx])

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(label_res == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_h = box_res[index]
                tmp_label_h = label_res[index]
                tmp_score_h = score_res[index]

                tmp_boxes_h = np.array(tmp_boxes_h)
                tmp = np.zeros([tmp_boxes_h.shape[0], tmp_boxes_h.shape[1] + 1])
                tmp[:, 0:-1] = tmp_boxes_h
                tmp[:, -1] = np.array(tmp_score_h)

                inx = nms.py_cpu_nms(dets=np.array(tmp, np.float32),
                                     thresh=h_threshold[LABEl_NAME_MAP[sub_class]],
                                     max_output_size=500)

                box_res_.extend(np.array(tmp_boxes_h)[inx])
                score_res_.extend(np.array(tmp_score_h)[inx])
                label_res_.extend(np.array(tmp_label_h)[inx])

            time_elapsed = timer() - start

            if save_res:
                det_detections_h = draw_box_in_img.draw_box_cv(np.array(img, np.float32) - np.array(cfgs.PIXEL_MEAN),
                                                               boxes=np.array(box_res_),
                                                               labels=np.array(label_res_),
                                                               scores=np.array(score_res_))
                det_detections_r = draw_box_in_img.draw_rotate_box_cv(
                    np.array(img, np.float32) - np.array(cfgs.PIXEL_MEAN),
                    boxes=np.array(box_res_rotate_),
                    labels=np.array(label_res_rotate_),
                    scores=np.array(score_res_rotate_))
                save_dir = os.path.join(des_folder, cfgs.VERSION)
                tools.mkdir(save_dir)
                cv2.imwrite(save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_h.jpg',
                            det_detections_h)
                cv2.imwrite(save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_r.jpg',
                            det_detections_r)

                view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                              time_elapsed), count + 1, len(file_paths))

            else:
                # eval txt
                CLASS_DOTA = NAME_LABEL_MAP.keys()
                # Task1
                write_handle_r = {}
                write_handle_h_ = {}
                txt_dir_r = os.path.join('txt_output', cfgs.VERSION + '_r')
                txt_dir_h_minAreaRect = os.path.join('txt_output', cfgs.VERSION + '_h_minAreaRect')
                tools.mkdir(txt_dir_r)
                tools.mkdir(txt_dir_h_minAreaRect)
                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_r[sub_class] = open(os.path.join(txt_dir_r, 'Task1_%s.txt' % sub_class), 'a+')
                    write_handle_h_[sub_class] = open(os.path.join(txt_dir_h_minAreaRect, 'Task2_%s.txt' % sub_class), 'a+')

                rboxes = coordinate_convert.forward_convert(box_res_rotate_, with_label=False)

                for i, rbox in enumerate(rboxes):
                    command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (img_path.split('/')[-1].split('.')[0],
                                                                                     score_res_rotate_[i],
                                                                                     rbox[0], rbox[1], rbox[2], rbox[3],
                                                                                     rbox[4], rbox[5], rbox[6], rbox[7],)
                    command_ = '%s %.3f %.1f %.1f %.1f %.1f\n' % (img_path.split('/')[-1].split('.')[0],
                                                                  score_res_rotate_[i],
                                                                  min(rbox[::2]), min(rbox[1::2]),
                                                                  max(rbox[::2]), max(rbox[1::2]))
                    write_handle_r[LABEl_NAME_MAP[label_res_rotate_[i]]].write(command)
                    write_handle_h_[LABEl_NAME_MAP[label_res_rotate_[i]]].write(command_)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_r[sub_class].close()

                # Task2
                write_handle_h = {}
                txt_dir_h = os.path.join('txt_output', cfgs.VERSION + '_h')
                tools.mkdir(txt_dir_h)
                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_h[sub_class] = open(os.path.join(txt_dir_h, 'Task2_%s.txt' % sub_class), 'a+')

                for i, hbox in enumerate(box_res_):
                    command = '%s %.3f %.1f %.1f %.1f %.1f\n' % (img_path.split('/')[-1].split('.')[0],
                                                                 score_res_[i],
                                                                 hbox[0], hbox[1], hbox[2], hbox[3])
                    write_handle_h[LABEl_NAME_MAP[label_res_[i]]].write(command)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_h[sub_class].close()

                view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                              time_elapsed), count + 1, len(file_paths))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='images path',
                        default=None, type=str)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='output path',
                        default=None, type=str)
    parser.add_argument('--h_len', dest='h_len',
                        help='image height',
                        default=800, type=int)
    parser.add_argument('--w_len', dest='w_len',
                        help='image width',
                        default=800, type=int)
    parser.add_argument('--h_overlap', dest='h_overlap',
                        help='height overlap',
                        default=200, type=int)
    parser.add_argument('--w_overlap', dest='w_overlap',
                        help='width overlap',
                        default=200, type=int)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.png', type=str)
    parser.add_argument('--save_res', dest='save_res',
                        help='save results',
                        default=True, type=bool)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    file_paths = get_file_paths_recursive(args.src_folder, args.image_ext)

    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)

    inference(det_net, file_paths, args.des_folder, args.h_len, args.w_len,
               args.h_overlap, args.w_overlap,  args.save_res)

    # file_paths = get_file_paths_recursive('/root/userfolder/DOTA/test/', '.png')
    # det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
    #                                                is_training=False)
    # inference(det_net, file_paths, '/root/userfolder/yx/R2CNN_Attention/tools/demo/', 800, 800,
    #           200, 200, False)

















