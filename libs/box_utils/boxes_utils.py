# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from libs.box_utils.coordinate_convert import forward_convert
import math
from PIL import  Image
pi = tf.constant(math.pi)

def ious_calu(boxes_1, boxes_2):
    '''

    :param boxes_1: [N, 4] [xmin, ymin, xmax, ymax]
    :param boxes_2: [M, 4] [xmin, ymin. xmax, ymax]
    :return:
    '''
    boxes_1 = tf.cast(boxes_1, tf.float32)
    boxes_2 = tf.cast(boxes_2, tf.float32)
    xmin_1, ymin_1, xmax_1, ymax_1 = tf.split(boxes_1, 4, axis=1)  # xmin_1 shape is [N, 1]..
    xmin_2, ymin_2, xmax_2, ymax_2 = tf.unstack(boxes_2, axis=1)  # xmin_2 shape is [M, ]..

    max_xmin = tf.maximum(xmin_1, xmin_2)
    min_xmax = tf.minimum(xmax_1, xmax_2)

    max_ymin = tf.maximum(ymin_1, ymin_2)
    min_ymax = tf.minimum(ymax_1, ymax_2)

    overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = tf.maximum(0., min_xmax - max_xmin)

    overlaps = overlap_h * overlap_w

    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    ious = overlaps / (area_1 + area_2 - overlaps)

    return ious


def clip_boxes_to_img_boundaries(decode_boxes, img_shape):
    '''

    :param decode_boxes:
    :return: decode boxes, and already clip to boundaries
    '''

    with tf.name_scope('clip_boxes_to_img_boundaries'):

        # xmin, ymin, xmax, ymax = tf.unstack(decode_boxes, axis=1)
        xmin = decode_boxes[:, 0]
        ymin = decode_boxes[:, 1]
        xmax = decode_boxes[:, 2]
        ymax = decode_boxes[:, 3]
        img_h, img_w = img_shape[1], img_shape[2]

        img_h, img_w = tf.cast(img_h, tf.float32), tf.cast(img_w, tf.float32)

        xmin = tf.maximum(tf.minimum(xmin, img_w-1.), 0.)
        ymin = tf.maximum(tf.minimum(ymin, img_h-1.), 0.)

        xmax = tf.maximum(tf.minimum(xmax, img_w-1.), 0.)
        ymax = tf.maximum(tf.minimum(ymax, img_h-1.), 0.)

        return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))


def filter_outside_boxes(boxes, img_h, img_w):
    '''
    :param anchors:boxes with format [xmin, ymin, xmax, ymax]
    :param img_h: height of image
    :param img_w: width of image
    :return: indices of anchors that inside the image boundary
    '''

    with tf.name_scope('filter_outside_boxes'):
        xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)

        xmin_index = tf.greater_equal(xmin, 0)
        ymin_index = tf.greater_equal(ymin, 0)
        xmax_index = tf.less_equal(xmax, tf.cast(img_w, tf.float32))
        ymax_index = tf.less_equal(ymax, tf.cast(img_h, tf.float32))

        indices = tf.transpose(tf.stack([xmin_index, ymin_index, xmax_index, ymax_index]))
        indices = tf.cast(indices, dtype=tf.int32)
        indices = tf.reduce_sum(indices, axis=1)
        indices = tf.where(tf.equal(indices, 4))
        # indices = tf.equal(indices, 4)
        return tf.reshape(indices, [-1])


def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):

    '''
    num of boxes less than max num of boxes, so it need to pad with zeros[0, 0, 0, 0]
    :param boxes:
    :param scores: [-1]
    :param max_num_of_boxes:
    :return:
    '''

    pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

    zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
    zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)

    final_boxes = tf.concat([boxes, zero_boxes], axis=0)

    final_scores = tf.concat([scores, zero_scores], axis=0)

    return final_boxes, final_scores


def get_horizen_minAreaRectangle(boxs, with_label=True):

    rpn_proposals_boxes_convert = tf.py_func(forward_convert,
                                             inp=[boxs, with_label],
                                             Tout=tf.float32)
    if with_label:
        rpn_proposals_boxes_convert = tf.reshape(rpn_proposals_boxes_convert, [-1, 9])

        boxes_shape = tf.shape(rpn_proposals_boxes_convert)
        x_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])
        y_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])

        label = tf.unstack(rpn_proposals_boxes_convert, axis=1)[-1]

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)
        return tf.transpose(tf.stack([x_min, y_min, x_max, y_max, label], axis=0))
    else:
        rpn_proposals_boxes_convert = tf.reshape(rpn_proposals_boxes_convert, [-1, 8])

        boxes_shape = tf.shape(rpn_proposals_boxes_convert)
        x_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])
        y_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)

    return tf.transpose(tf.stack([x_min, y_min, x_max, y_max], axis=0))
def get_horizen_minAreaRectangle1(boxs, with_label=True):
    '''

    :param boxs: [-1,5] or [-1, 6]
    :param with_label:
    :return:
    '''
    if with_label:
        x, y, w_r, h_r, theta, label = tf.unstack(boxs, axis=1)
        # w_r = tf.abs(w_r)
        # h_r = tf.abs(h_r)
        theta = tf.where(tf.equal(theta, -90.), tf.zeros_like(theta), tf.abs(theta))

        theta = theta / 180 * pi # tf.multiply(tf.divide(theta, tf.constant(180.)), pi)
        abstheta = tf.abs(theta)
        #theta = tf.cast(theta, tf.float32)



        w = tf.add(tf.multiply(w_r, tf.cos(abstheta)), tf.multiply(h_r, tf.sin(abstheta)))
        h = tf.add(tf.multiply(h_r, tf.cos(abstheta)), tf.multiply(w_r, tf.sin(abstheta)))
        xmin = tf.subtract(x, tf.divide(w, 2))
        ymin = tf.subtract(y, tf.divide(h, 2))
        xmax = tf.add(x, tf.divide(w, 2))
        ymax = tf.add(y, tf.divide(h, 2))

        return tf.transpose(tf.stack([xmin, ymin, xmax, ymax, label], axis=0))
    else:
        x, y, w_r, h_r, theta = tf.unstack(boxs, axis=1)
        # w_r = tf.abs(w_r)
        # h_r = tf.abs(h_r)

        theta = theta / 180 * pi #tf.where(tf.equal(theta, -90.), tf.zeros_like(theta), tf.abs(theta))

        theta = tf.multiply(tf.divide(theta, tf.constant(180.)), pi)
        abstheta = tf.abs(theta)

        w = tf.add(tf.multiply(w_r, tf.cos(abstheta)), tf.multiply(h_r, tf.sin(abstheta)))
        h = tf.add(tf.multiply(h_r, tf.cos(abstheta)), tf.multiply(w_r, tf.sin(abstheta)))
        xmin = tf.subtract(x, tf.divide(w, 2))
        ymin = tf.subtract(y, tf.divide(h, 2))
        xmax = tf.add(x, tf.divide(w, 2))
        ymax = tf.add(y, tf.divide(h, 2))

        return tf.transpose(tf.stack([xmin, ymin, xmax, ymax], axis=0))
import cv2
def RROI_(feature_maps, rois, rois_h):

    #rois_h = get_horizen_minAreaRectangle(rois)
    rois = np.array(rois)
    rois_h = np.array(rois_h)
    feature_maps = np.array(feature_maps)
    feature_maps = np.squeeze(feature_maps, axis=0)
    print(feature_maps.shape)
    roi_feature = None
    for i in range(rois.shape[0]):
        x, y, w_r, h_r, theta = rois[i][0],rois[i][1],   rois[i][2],   rois[i][3], rois[i][4]      #  np.split(rois[i], [1,2,3,4])
        print(x, y, w_r, h_r, theta)
        if theta < -90.0:
            theta = -90.0
        xmin, ymin, xmax, ymax = rois_h[i][0],rois_h[i][1],   rois_h[i][2],   rois_h[i][3]
        print(xmin, xmax, ymin, ymax)#np.split(rois_h[i], [1,2,3])
        crop_fature = feature_maps[np.int(ymin/16):np.int(ymax/16) + 1, np.int(xmin/16): np.int(xmax/16), 0:]
        cols = crop_fature.shape[1]
        rows = crop_fature.shape[0]
        print(cols)
        print(rows)
        print(crop_fature.shape)
        #im = Image.fromarray(crop_fature)
        #im_rotate = crop_fature.rotate(theta)
        M = cv2.getRotationMatrix2D((rows/2, cols/2), theta, 1)
        print(M.shape)
        img1 = cv2.warpAffine(np.zeros((60,60,512)), M, (rows, cols))
        img2 = cv2.warpAffine(np.zeros((60,60,512)), M, (rows, cols))
        img = np.concatenate((img1, img2), axis=2)
        hh, ww = img.shape[:2]
        roi = img[np.int(hh-h_r/16)//2:np.int(hh+h_r/16)//2+1, np.int(ww-w_r/16)//2 : np.int(ww+ w_r/15)//2]
        print(roi.shape)
    #     roi = np.expand_dims(roi, axis=0)
    # #     roi_feature.append(roi)
    # # roi_feature = np.array(roi_feature, dtype=np.float32)
    # roi_feature = np.stack((roi_feature, roi), axis=0)
    # #     roi = np.array(roi, dtype=np.float32)
    # #     roi = np.stack(roi)
    # print(roi_feature.shape)
    return roi
def get_mask_tf(rotate_rects, featuremap_size):
    mask_tensor = tf.py_func(get_mask,
                            inp=[rotate_rects, featuremap_size],
                            Tout=tf.float32)
    mask_tensor = tf.reshape(mask_tensor, [tf.shape(rotate_rects)[0], featuremap_size, featuremap_size]) # [300, 14, 14]

    return mask_tensor


def get_mask(rotate_rects, featuremap_size):

    all_mask = []
    for a_rect in rotate_rects:
        rect = ((a_rect[1], a_rect[0]), (a_rect[3], a_rect[2]), a_rect[-1])  # in tf. [x, y, w, h, theta]
        rect_eight = cv2.boxPoints(rect)
        x_list = rect_eight[:, 0:1]
        y_list = rect_eight[:, 1:2]
        min_x, max_x = np.min(x_list), np.max(x_list)
        min_y, max_y = np.min(y_list), np.max(y_list)
        x_list = x_list - min_x
        y_list = y_list - min_y

        new_rect = np.hstack([x_list*featuremap_size*1.0/(max_x-min_x+1),
                             y_list * featuremap_size * 1.0 / (max_y - min_y + 1)])
        mask_array = np.zeros([featuremap_size, featuremap_size], dtype=np.float32)
        for x in range(featuremap_size):
            for y in range(featuremap_size):
                inner_rect = cv2.pointPolygonTest(contour=new_rect, pt=(x, y), measureDist=False)
                mask_array[y, x] = np.float32(0) if inner_rect == -1 else np.float32(1)
        all_mask.append(mask_array)
    return np.array(all_mask)