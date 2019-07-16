# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

from libs.configs import cfgs
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
import tensorflow as tf
import numpy as np


def postprocess_rpn_proposals(rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
    '''

    :param rpn_bbox_pred: [-1, 4]
    :param rpn_cls_prob: [-1, 2]
    :param img_shape:
    :param anchors:[-1, 4]
    :param is_training:
    :return:
    '''

    if is_training:
        pre_nms_topN = cfgs.RPN_TOP_K_NMS_TRAIN
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TARIN
        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD
    else:
        pre_nms_topN = cfgs.RPN_TOP_K_NMS_TEST
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TEST
        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD

    cls_prob = rpn_cls_prob[:, 1]

    # 1. decode boxes
    decode_boxes = encode_and_decode.decode_boxes(encode_boxes=rpn_bbox_pred,
                                                  reference_boxes=anchors,
                                                  scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # decode_boxes = encode_and_decode.decode_boxes(boxes=anchors,
    #                                               deltas=rpn_bbox_pred,
    #                                               scale_factor=None)

    # 2. clip to img boundaries  超过边界的去掉
    decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=decode_boxes,
                                                            img_shape=img_shape)

    # 3. get top N to NMS
    if pre_nms_topN > 0:
        pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='avoid_unenough_boxes')
        cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
        decode_boxes = tf.gather(decode_boxes, top_k_indices)  # 取索引


    # 4. NMS
    keep = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=cls_prob,
        max_output_size=post_nms_topN,
        iou_threshold=nms_thresh)

    final_boxes = tf.gather(decode_boxes, keep)
    final_probs = tf.gather(cls_prob, keep)

    return final_boxes, final_probs  # xmin,ymin, xmax, ymax

def postprocess_fastrcnn_proposals(bbox_ppred, scores, img_shape, rois, is_training):
    '''

    :param rpn_bbox_pred: [-1, 4]
    :param rpn_cls_prob: [-1, 2]
    :param img_shape:
    :param anchors:[-1, 4]
    :param is_training:
    :return:
    '''

    if is_training:
        pre_nms_topN = 2000#cfgs.RPN_TOP_K_NMS_TRAIN
        post_nms_topN = 500#cfgs.RPN_MAXIMUM_PROPOSAL_TARIN
        nms_thresh = 0.8#cfgs.RPN_NMS_IOU_THRESHOLD
    else:
        pre_nms_topN = 1500#cfgs.RPN_TOP_K_NMS_TEST
        post_nms_topN = 300#cfgs.RPN_MAXIMUM_PROPOSAL_TEST
        nms_thresh = 0.7#cfgs.RPN_NMS_IOU_THRESHOLD




    #rois = tf.stop_gradient(rois)
    #scores = tf.stop_gradient(scores)
    bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
    #bbox_ppred = tf.stop_gradient(bbox_ppred)

    bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
    score_list = tf.unstack(scores, axis=1)

    allclasses_boxes = []
    allclasses_scores = []
    categories = []
    for i in range(1, cfgs.CLASS_NUM + 1):
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

        pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(tmp_decoded_boxes)[0], name='avoid_unenough_boxes')
        cls_prob, top_k_indices = tf.nn.top_k(tmp_score, k=pre_nms_topN)
        decode_boxes = tf.gather(tmp_decoded_boxes, top_k_indices)  # 取索引

        # 4. NMS
        keep = tf.image.non_max_suppression(
            boxes=decode_boxes,
            scores=cls_prob,
            max_output_size=post_nms_topN,
            iou_threshold=nms_thresh)

        perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
        perclass_scores = tf.gather(tmp_score, keep)

        allclasses_boxes.append(perclass_boxes)
        allclasses_scores.append(perclass_scores)
        categories.append(tf.ones_like(perclass_scores) * i)

    final_boxes = tf.concat(allclasses_boxes, axis=0)
    final_scores = tf.concat(allclasses_scores, axis=0)
    final_category = tf.concat(categories, axis=0)



    return final_boxes, final_scores