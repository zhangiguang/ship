# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from libs.label_name_dict.label_dict import LABEl_NAME_MAP

from libs.configs import cfgs



label_map = {


    'Background': 0,
    'Aircraft Carrier': 1,
    'Cruiser': 2,
    'Amphibious Ship:Wasp': 3,
    'Amphibious Ship:Austen': 4,
    'Amphibious Ship:Big Corner': 5,
    'Amphibious Ship:Tag': 6,
    'Amphibious Ship:San Antonio': 7,
    'Destroyer:BoKe': 8,
    'Destroyer:ChuXue': 9,
    'Destroyer:ZhenMing': 10,
    'Destroyer:TaiDao': 11,
    'Destroyer:CunYu': 12,
    'Destroyer:ZhaoWu': 13,
    'Destroyer:AiDang': 14,
     'Submarine': 15,
    'Frigate': 16,
    'Other:ShiHeTian Supply Ship': 17,
    'T-ake': 18,
    'Other': 19



}
'''
        'background': 0,
        'aircraft Carrier (USA)': 1,
        'aircraft carrier (China and Russia)': 2,
        'aircraft carrier (Japan)': 3,
        'aircraft carrier (other)': 4,
        'frigate (USA PERRY Class)': 5,
        'cruiser (New Tikang)': 6,
        'destroyer(Japan small-scale)': 7,
        'destroyer(Japan medium-scale)': 8,
        'destroyer(Japan large-scale)': 9,
        'destroyer(USA BoKe)': 10,
        'destroyer(Russia Modern class)': 11,
        'destroyer(Europe)': 12,
        'destroyer(Korea)': 13,
        'destroyer(India)': 14,
        'destroyer(China)': 15,
        'frigate(USA Independence Class)': 16,
        'frigate(USA Freedom Class)': 17,
        'frigate(Russia)': 18,
        'frigate(Europe)': 19,
        'frigate(China)': 20,
        'frigate(Asia-Pacific)': 21,
        'amphibious assault ship(USA)': 22,
        'amphibious assault ship(Europe)': 23,
        'amphibious assault ship(Asia-Pacific)': 24,
        'light duty frigate(Asia-Pacific)': 25,
        'light duty frigate(Europe)': 26,
        'tanker': 27,
        'container ship': 28,
        'grocery ship': 29,
        'amphibious transport ship': 30,
        'small military warship': 31,
        'supply ship': 32,
        'submarine': 33,
        'other': 34,
        'cruiser(Russia)': 35,
        'destroyer(Russia)': 36
'''
'''
        'background':0,
        'aircraft Carrier (USA)':1,
        'frigate (USA PERRY Class)':2,
        'cruiser (New Tikang)':3,
        'destroyer(USA BoKe)':4,
        'frigate (USA Independence Class)':5,
        'frigate (USA Freedom Class)':6,
        'amphibious assault ship(USA)':7,
        'tanker':8,
        'container ship':9,
        'grocery ship':10,
        'amphibious transport ship':11,
        'small military warship':12,
        'supply ship':13,
        'submarine':14,
        'other':15
'''
'''
        'back_ground',
        'airport',
        'baseball-diamond',
        'basketball-court',
        'bridge',
        'container-crane',
        'ground-track-field',
        'harbor',
        'helicopter',
        'helipad',
        'large-vehicle',
        'plane',
        'roundabout',
        'ship',
        'small-vehicle',
        'soccer-ball-field',
        'storage-tank',
        'swimming-pool',
        'tennis-court'
'''


test_label_map = [

    'Background',
    'Aircraft Carrier',
    'Cruiser',
    'Amphibious Ship:Wasp',
    'Amphibious Ship:Austen',
    'Amphibious Ship:Big Corner',
    'Amphibious Ship:Tag',
    'Amphibious Ship:San Antonio',
    'Destroyer:BoKe',
     'Destroyer:ChuXue',
    'Destroyer:ZhenMing',
    'Destroyer:TaiDao',
    'Destroyer:CunYu',
    'Destroyer:ZhaoWu',
    'Destroyer:AiDang',
    'Submarine',
    'Frigate',
    'Other:ShiHeTian Supply Ship',
    'T-ake',
    'Other'
    '''
        'background',
        'aircraft Carrier (USA)',
        'aircraft carrier (China and Russia)',
        'aircraft carrier (Japan)',
        'aircraft carrier (other)',
        'frigate (USA PERRY Class)',
        'cruiser (New Tikang)',
        'destroyer(Japan small-scale)',
        'destroyer(Japan medium-scale)',
        'destroyer(Japan large-scale)',
        'destroyer(USA BoKe)',
        'destroyer(Russia Modern class)',
        'destroyer(Europe)',
        'destroyer(Korea)',
        'destroyer(India)',
        'destroyer(China)',
        'frigate (USA Independence Class)',
        'frigate (USA Freedom Class)',
        'frigate(Russia)',
        'frigate(Europe)',
        'frigate(China)',
        'frigate(Asia-Pacific)',
        'amphibious assault ship(USA)',
        'amphibious assault ship(Europe)',
        'amphibious assault ship(Asia-Pacific)',
        'light duty frigate(Asia-Pacific)',
        'light duty frigate(Europe)',
        'tanker',
        'container ship',
        'grocery ship',
        'amphibious transport ship',
        'small military warship',
        'supply ship',
        'submarine',
        'other',
        'cruiser(Russia)',
        'destroyer(Russia)'
        '''
    '''
        'background',
        'aircraft Carrier (USA)',
        'frigate (USA PERRY Class)',
        'cruiser (New Tikang)',
        'destroyer(USA BoKe)',
        'frigate (USA Independence Class)',
        'frigate (USA Freedom Class)',
        'amphibious assault ship(USA)',
        'tanker',
        'container ship',
        'grocery ship',
        'amphibious transport ship',
        'small military warship',
        'supply ship',
        'submarine',
        'other'
        '''
    '''
        'back_ground',
        'airport',
        'baseball-diamond',
        'basketball-court',
        'bridge',
        'container-crane',
        'ground-track-field',
        'harbor',
        'helicopter',
        'helipad',
        'large-vehicle',
        'plane',
        'roundabout',
        'ship',
        'small-vehicle',
        'soccer-ball-field',
        'storage-tank',
        'swimming-pool',
        'tennis-court'
        '''
        ]
from libs.box_utils import draw_box_in_img

def only_draw_boxes(img_batch, boxes):

    boxes = tf.stop_gradient(boxes)
    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0], ), dtype=tf.int32) * draw_box_in_img.ONLY_DRAW_BOXES
    scores = tf.zeros_like(labels, dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=tf.uint8)
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))  # [batch_size, h, w, c]

    return img_tensor_with_boxes
def draw_boxes_with_scores(img_batch, boxes, scores):

    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.int32) * draw_box_in_img.ONLY_DRAW_BOXES_WITH_SCORES
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

def draw_boxes_with_categories(img_batch, boxes, labels):
    boxes = tf.stop_gradient(boxes)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    scores = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores):
    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes
def draw_box_with_color(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        # img = np.transpose(img, [2, 1, 0])
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    # color = tf.constant([0, 0, 255])
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes


def draw_box_with_color_rotate(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.drawContours(img, [rect], -1, color, 3)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes


# def draw_boxes_with_categories(img_batch, boxes, scores):
#
#     def draw_box_cv(img, boxes, scores):
#         img = img + np.array(cfgs.PIXEL_MEAN)
#         boxes = boxes.astype(np.int64)
#         img = np.array(img*255/np.max(img), np.uint8)
#
#         num_of_object = 0
#         for i, box in enumerate(boxes):
#             xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
#
#             score = scores[i]
#
#             num_of_object += 1
#             color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
#             cv2.rectangle(img,
#                           pt1=(xmin, ymin),
#                           pt2=(xmax, ymax),
#                           color=color,
#                           thickness=2)
#             cv2.rectangle(img,
#                           pt1=(xmin, ymin),
#                           pt2=(xmin+120, ymin+15),
#                           color=color,
#                           thickness=-1)
#             cv2.putText(img,
#                         text=str(score),
#                         org=(xmin, ymin+10),
#                         fontFace=1,
#                         fontScale=1,
#                         thickness=2,
#                         color=(color[1], color[2], color[0]))
#         cv2.putText(img,
#                     text=str(num_of_object),
#                     org=((img.shape[1]) // 2, (img.shape[0]) // 2),
#                     fontFace=3,
#                     fontScale=1,
#                     color=(255, 0, 0))
#         img = img[:, :, ::-1]
#         return img
#
#     img_tensor = tf.squeeze(img_batch, 0)
#     img_tensor_with_boxes = tf.py_func(draw_box_cv,
#                                        inp=[img_tensor, boxes, scores],
#                                        Tout=[tf.uint8])
#     img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
#     return img_tensor_with_boxes
#
#
# def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores):
#
#     def draw_box_cv(img, boxes, labels, scores):
#         img = img + np.array(cfgs.PIXEL_MEAN)
#         boxes = boxes.astype(np.int64)
#         labels = labels.astype(np.int32)
#         img = np.array(img*255/np.max(img), np.uint8)
#
#         num_of_object = 0
#         for i, box in enumerate(boxes):
#             xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
#
#             label = labels[i]
#             score = scores[i]
#             if label != 0:
#                 num_of_object += 1
#                 color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
#                 cv2.rectangle(img,
#                               pt1=(xmin, ymin),
#                               pt2=(xmax, ymax),
#                               color=color,
#                               thickness=2)
#                 cv2.rectangle(img,
#                               pt1=(xmin, ymin),
#                               pt2=(xmin+120, ymin+15),
#                               color=color,
#                               thickness=-1)
#                 category = test_label_map[label]#LABEl_NAME_MAP[label]
#                 cv2.putText(img,
#                             text=category+": "+str(score),
#                             org=(xmin, ymin+10),
#                             fontFace=1,
#                             fontScale=1,
#                             thickness=2,
#                             color=(color[1], color[2], color[0]))
#         cv2.putText(img,
#                     text=str(num_of_object),
#                     org=((img.shape[1]) // 2, (img.shape[0]) // 2),
#                     fontFace=3,
#                     fontScale=1,
#                     color=(255, 0, 0))
#         img = img[:, :, ::-1]
#         return img
#
#     img_tensor = tf.squeeze(img_batch, 0)
#     img_tensor_with_boxes = tf.py_func(draw_box_cv,
#                                        inp=[img_tensor, boxes, labels, scores],
#                                        Tout=[tf.uint8])
#     img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
#     return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores_rotate(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):

            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1

                rect = ((x_c, y_c), (w, h), theta)
                rect = cv2.boxPoints(rect)
                rect = np.int0(rect)
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                cv2.drawContours(img, [rect], -1, color, 3)

                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c+120, y_c+15),
                              color=color,
                              thickness=-1)
                category = test_label_map[label] #LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


if __name__ == "__main__":
    print (1)

