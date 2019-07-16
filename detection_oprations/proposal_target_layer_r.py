# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from libs.configs import cfgs
import numpy as np
import numpy.random as npr

from libs.box_utils import encode_and_decode
from libs.box_utils import iou_rotate


def proposal_target_layer_r(rpn_rois, gt_boxes, overlaps, fast_iou_positive_threshld):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    rois_per_image = np.inf if cfgs.FAST_RCNN_MINIBATCH_SIZE == -1 else cfgs.FAST_RCNN_MINIBATCH_SIZE

    fg_rois_per_image = np.round(cfgs.FAST_RCNN_POSITIVE_RATE * rois_per_image)
    # Sample rois with classification labels and bounding box regression
    labels, rois, bbox_targets = _sample_rois(rpn_rois, gt_boxes, overlaps,
                                              fg_rois_per_image, rois_per_image, cfgs.CLASS_NUM+1, fast_iou_positive_threshld)
    rois = rois.reshape(-1, 5)
    labels = labels.reshape(-1)
    bbox_targets = bbox_targets.reshape(-1, (cfgs.CLASS_NUM+1) * 5)

    return rois, labels, bbox_targets


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th, ttheta)

    This function expands those targets into the 5-of-5*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 5K blob of regression targets
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 5 * num_classes), dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(5 * cls)
        end = start + 5
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]

    return bbox_targets


def _compute_targets(ex_rois, gt_rois_r, labels):
    """Compute bounding-box regression targets for an image.
    that is : [label, tx, ty, tw, th]
    """

    assert ex_rois.shape[0] == gt_rois_r.shape[0]
    assert ex_rois.shape[1] == 5
    assert gt_rois_r.shape[1] == 5

    targets_r = encode_and_decode.encode_boxes_rotate(unencode_boxes=gt_rois_r,
                                                      reference_boxes=ex_rois,
                                                      scale_factors=cfgs.ROI_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=cfgs.ROI_SCALE_FACTORS)
    return np.hstack((labels[:, np.newaxis], targets_r)).astype(np.float32, copy=False)


def _sample_rois(all_rois,  gt_boxes, overlaps, fg_rois_per_image,
                 rois_per_image, num_classes, fast_iou_positive_threshld):
    """Generate a random sample of RoIs comprising foreground and background
    examples.

    all_rois shape is [-1, 5]
    gt_boxes shape is [-1, 6]. that is [x_c, y_c, w, h, theta, label]
    """
    # overlaps: (rois x gt_boxes)

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, -1]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= fast_iou_positive_threshld)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < fast_iou_positive_threshld) &
                       (max_overlaps >= cfgs.FAST_RCNN_IOU_NEGATIVE_THRESHOLD))[0]
    # print("first fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)

    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # print("second fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]
    bbox_target_data = _compute_targets(rois, gt_boxes[gt_assignment[keep_inds], :-1], labels)
    bbox_targets = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets
