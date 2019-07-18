import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
sys.path.append('/media/ys/000021A3000B80CF')
sys.path.append('/media/ys/000021A3000B80CF/R2CNN_FPN_Tensorflow-master')

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from lxml import etree
from libs.box_utils.rotate_polygon_nms import rotate_gpu_nms
from libs.box_utils.nms_rotate import nms_rotate_cpu

# This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
# from object_detection.utils import dataset_util
# from object_detection.utils import np_box_ops

PATH_TO_CKPT='/media/iguang/23a6d0dc-b99b-49a8-9ea9-c857f767ba3a/ship/R2CNN_Faster-RCNN_Tensorflow-master/tools/pretrained/OPship.pb'
PATH_TO_LABELS = '/media/ys/000021A3000B80CF/R2CNN_FPN_Tensorflow-master/output/trained_pb/OPship.pbtxt' #Label map (.pbtxt)
NUM_CLASSES = 37
patch_size = 1024
Threshold = 0.5
num_ships = 0
num_true_detect = 0
num_false_alarm = 0
Ori_img_path = '/media/iguang/23a6d0dc-b99b-49a8-9ea9-c857f767ba3a/ship/VOC2007/JPEGImages'
result_img_path = '/media/ys/000021A3000B80CF/R2CNN_FPN_Tensorflow-master/GF_data/train_test_img_result'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        #det_tensor = tf.get_collection('det_tensor')
        #keep = tf.py_func(nms_rotate_cpu,
        #                  inp=[det_tensor, 0.15, 0],
        #                  Tout=tf.int64)

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
ori_img_listdir = os.listdir(Ori_img_path)
num_ori_img = len(ori_img_listdir)



for k in range(num_ori_img):

    img_name = ori_img_listdir[k]
    Ori_img = cv2.imread(os.path.join(Ori_img_path,ori_img_listdir[k]))
    Ori_img = np.array(Ori_img)

    test_img = Image.open(os.path.join(Ori_img_path,ori_img_listdir[k]))
    test_img = np.array(test_img)

    height, width, _ = np.shape(Ori_img)
    num_hstep = int(np.ceil(height/patch_size))
    num_wstep = int(np.ceil(width/patch_size))

    result_img = np.zeros([height,width,3])

    for i in range(num_hstep+1):
        for j in range(num_wstep+1):
            # cut into patches
            if i != num_hstep and j != num_wstep:
                img_patch = Ori_img[patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1), :]
            elif i == num_hstep and j != num_wstep:
                img_patch[0:(height-i*patch_size),:, :] = Ori_img[i*patch_size:height, patch_size*j:patch_size*(j+1), :]
            elif i != num_hstep and j == num_wstep:
                img_patch[:, 0:(width-patch_size*j), :] = Ori_img[patch_size*i:patch_size*(i+1), patch_size*j:width, :]
            elif i == num_hstep and j == num_wstep:
                img_patch[0:(height-patch_size*i), 0:(width-patch_size*j), :] = Ori_img[patch_size*i:height, patch_size*j:width, :]

            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:
            
                    image_np_expanded = np.expand_dims(img_patch, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
                    fast_rcnn_decode_boxes_rotate_beforeNMS = detection_graph.get_tensor_by_name('fast_rcnn_decode_boxes_rotate_beforeNMS:0')
                    fast_rcnn_scores_rotate_beforeNMS = detection_graph.get_tensor_by_name('fast_rcnn_softmax_scores_rotate:0')
                    #fast_rcnn_decode_boxes_rotate_beforeNMS = detection_graph.get_tensor_by_name('fast_rcnn_decode_boxes:0')
                    #fast_rcnn_scores_rotate_beforeNMS = detection_graph.get_tensor_by_name('fast_rcnn_softmax_scores:0')

                    (_fast_rcnn_decode_boxes_rotate, _fast_rcnn_score_rotate) = sess.run(
                        [fast_rcnn_decode_boxes_rotate_beforeNMS, fast_rcnn_scores_rotate_beforeNMS],
                        feed_dict={image_tensor: image_np_expanded})
                    decode_boxes = np.array(_fast_rcnn_decode_boxes_rotate).astype(np.float) 

                    # NMS start from here
                    category = np.argmax(_fast_rcnn_score_rotate, axis=1)
                    object_mask = (np.not_equal(category,0)).astype('int')
                    decode_boxes = decode_boxes * np.expand_dims(object_mask, axis=1)
                    scores = _fast_rcnn_score_rotate * np.expand_dims(object_mask, axis=1)
                    decode_boxes = np.reshape(decode_boxes, [-1, NUM_CLASSES, 5])
                    decode_boxes_list = []
                    for t in range(NUM_CLASSES):
                        temp_decode_boxe = decode_boxes[:,t,:]
                        decode_boxes_list.append(temp_decode_boxe)
                    #decode_boxes_list = tf.unstack(decode_boxes, axis=1)
                    score_list = []
                    for t in range(NUM_CLASSES):
                        score_list.append(scores[:,t+1])
                    #score_list = tf.unstack(scores[:, 1:], axis=1)
                    after_nms_boxes = []
                    after_nms_scores = []
                    category_list = []

                    for per_class_decode_boxes, per_class_scores in zip(decode_boxes_list, score_list):
                        
                        boxes = per_class_decode_boxes
                        scores = per_class_scores
                        keep = []
                        order = np.argsort(scores)[::-1]
                        overlap = 0.3
                        num = boxes.shape[0]
                        suppressed = np.zeros((num), dtype=np.int)
                    

                        for _i in range(num):
                            if len(keep) >= 20:
                                break

                            temp_i = order[_i]  # an index
                            if suppressed[temp_i] == 1:
                                continue
                            keep.append(temp_i)
                            r1 = ((boxes[temp_i, 1], boxes[temp_i, 0]), (boxes[temp_i, 3], boxes[temp_i, 2]), boxes[temp_i, 4])
                            area_r1 = boxes[temp_i, 2] * boxes[temp_i, 3]
                            for _j in range(_i + 1, num):
                                temp_j = order[_j]
                                if suppressed[temp_i] == 1:
                                    continue
                                r2 = ((boxes[temp_j, 1], boxes[temp_j, 0]), (boxes[temp_j, 3], boxes[temp_j, 2]), boxes[temp_j, 4])
                                area_r2 = boxes[temp_j, 2] * boxes[temp_j, 3]
                                inter = 0.0

                                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                                if int_pts is not None:
                                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                                    if order_pts is not None:
                                        int_area = cv2.contourArea(order_pts)

                                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

                                if inter >= overlap:
                                    suppressed[temp_j] = 1
                        valid_indices = np.array(keep, np.int64)

                        after_nms_boxes.append(per_class_decode_boxes[valid_indices])
                        after_nms_scores.append(per_class_scores[valid_indices])
                        tmp_category = category[valid_indices]
                        category_list.append(tmp_category)

                    all_nms_boxes = []
                    all_nms_scores = []
                    all_category = []
                    all_nms_boxes = np.concatenate(after_nms_boxes, axis=0)  #tf.concat(after_nms_boxes, axis=0)
                    all_nms_scores = np.concatenate(after_nms_scores, axis=0) # tf.concat(after_nms_scores, axis=0)
                    all_category = np.concatenate(category_list, axis=0) #tf.concat(category_list, axis=0)
                    scores_large_than_threshold_indices = np.greater(all_nms_scores,0.5)

                    all_nms_boxes = all_nms_boxes[scores_large_than_threshold_indices]
                    all_nms_scores = all_nms_scores[scores_large_than_threshold_indices]
                    all_category = all_category[scores_large_than_threshold_indices]

                    out_boxes = all_nms_boxes
                    out_scores = all_nms_scores
                    out_num_detections = tf.shape(all_nms_boxes)[0]
                    query = img_name
                    out_category = all_category
                    
                    # NMS end here
                    ratio = 512.0/1024.0
                    for t in range(len(out_boxes)):
                        box = out_boxes[t]
                        y_c, x_c, h, w, theta = box[0]/ratio, box[1]/ratio, box[2]/ratio, box[3]/ratio, (box[4])

                        rect = ((x_c, y_c), (w, h), theta)
                        rect = cv2.boxPoints(rect)
                        rect = np.int0(rect)
                        color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                        cv2.drawContours(img_patch, [rect], -1, color, 3)

            if i != num_hstep and j != num_wstep:
                result_img[patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1),:] = img_patch
            elif i == num_hstep and j != num_wstep:
                result_img[patch_size*i:height, patch_size*j:patch_size*(j+1),:] = img_patch[0:(height-patch_size*i),:,:]
            elif i != num_hstep and j == num_wstep:
                result_img[patch_size*i:patch_size*(i+1), patch_size*j:width,:] = img_patch[:,0:(width-patch_size*j),:]
            elif i == num_hstep and j == num_wstep:
                result_img[patch_size*i:height,patch_size*j:width,:] = img_patch[0:(height-patch_size*i),0:(width-patch_size*j),:]
            
    #result_img = Image.fromarray(result_img.astype(np.uint8))
    save_path = os.path.join(result_img_path, img_name)
    #result_img.save(save_path)
    cv2.imwrite(save_path, result_img)
    print(img_name)