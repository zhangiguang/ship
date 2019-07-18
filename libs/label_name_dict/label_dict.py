# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import sys
reload(sys)
sys.setdefaultencoding('utf8')


from libs.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'FDDB':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'face': 1
    }
elif cfgs.DATASET_NAME == 'ICDAR2015':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'text': 1
    }
elif cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'airport': 1,
        'baseball-diamond': 2,
        'basketball-court': 3,
        'bridge': 4,
        'container-crane':5,
        'ground-track-field': 6,
        'harbor': 7,
        'helicopter': 8,
        'helipad':9,
        'large-vehicle': 10,
        'plane': 11,
        'roundabout': 12,
        'ship': 13,
        'small-vehicle': 14,
        'soccer-ball-field': 15,
        'storage-tank': 16,
        'swimming-pool': 17,
        'tennis-court': 18


    }
elif cfgs.DATASET_NAME == 'OPship':
    NAME_LABEL_MAP = {
        u'背景': 0,
        u'航母（美）': 1,
        u'航母（中俄）': 2,
        u'航母（日本）': 3,
        u'航母（其他）': 4,
        u'护卫舰（美佩里级）': 5,
        u'巡洋舰（新提康）': 6,
        u'驱逐舰（日小型）': 7,
        u'驱逐舰（日中型）': 8,
        u'驱逐舰（日大型）': 9,
        u'驱逐舰（美伯克级）': 10,
        u'驱逐舰（俄现代级）': 11,
        u'驱逐舰（欧）': 12,
        u'驱逐舰（韩）': 13,
        u'驱逐舰（印）': 14,
        u'驱逐舰（中）': 15,
        u'护卫舰（美独立级）': 16,
        u'护卫舰（美自由级）': 17,
        u'护卫舰（俄）': 18,
        u'护卫舰（欧）': 19,
        u'护卫舰（中）': 20,
        u'护卫舰（亚太）': 21,
        u'两栖攻击舰（美）': 22,
        u'两栖攻击舰（欧）': 23,
        u'两栖攻击舰（亚太）': 24,
        u'轻护舰（亚太）': 25,
        u'轻护舰（欧）': 26,
        u'民船（油轮）': 27,
        u'民船（集装箱轮）': 28,
        u'民船（杂货轮）': 29,
        u'两栖运输舰': 30,
        u'小型军用舰艇': 31,
        u'补给舰': 32,
        u'潜艇': 33,
        u'其它': 34,
        u'巡洋舰（俄）': 35,
        u'驱逐舰（俄）': 36
    }
elif cfgs.DATASET_NAME == 'USship':
    NAME_LABEL_MAP = {
        u'背景': 0,
        u'航母（美）': 1,
        u'护卫舰（美佩里级）': 2,
        u'巡洋舰（新提康）': 3,
        u'驱逐舰（美伯克级）': 4,
        u'护卫舰（美独立级）': 5,
        u'护卫舰（美自由级）': 6,
        u'两栖攻击舰（美）': 7,
        u'民船（油轮）': 8,
        u'民船（集装箱轮）': 9,
        u'民船（杂货轮）': 10,
        u'两栖运输舰': 11,
        u'小型军用舰艇': 12,
        u'补给舰': 13,
        u'潜艇': 14,
        u'其它': 15,
    }
elif cfgs.DATASET_NAME == 'USship':
    NAME_LABEL_MAP = {
        u'背景': 0,
        u'航空母舰': 1,
        u'巡洋舰': 2,
        u'两栖舰：黄蜂': 3,
        u'两栖舰：奥斯汀': 4,
        u'两栖舰：大隅': 5,
        u'两栖舰：塔瓦拉': 6,
        u'两栖舰：圣安东尼奥': 7,
        u'驱逐舰：伯克': 8,
        u'驱逐舰：初雪': 9,
        u'驱逐舰：榛名': 10,
        u'驱逐舰：太刀': 11,
        u'驱逐舰：村雨': 12,
        u'驱逐舰：朝雾': 13,
        u'驱逐舰：爱宕': 14,
        u'潜艇': 15,
        u'护卫舰': 16,
        u'其它：十和田补给舰': 17,
        u'其它：T-ake补给舰': 18,
        u'其它': 19
    }
    '''
    ['航空母舰', '巡洋舰', '两栖舰：黄蜂', '两栖舰：奥斯汀', '两栖舰：大隅', '两栖舰：塔瓦拉',
            '两栖舰：圣安东尼奥', '驱逐舰：伯克', '驱逐舰：初雪', '驱逐舰：榛名', '驱逐舰：太刀',
            '驱逐舰：村雨', '驱逐舰：朝雾', '驱逐舰：爱宕', '潜艇', '护卫舰', '其他：十和田补给舰',
            '其他：T-ake补给', '其他']'''
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()