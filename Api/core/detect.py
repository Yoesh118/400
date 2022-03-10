import json

import tensorflow as tf

from .weights import Model
from .mrcnn.visualize import display_instances
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
class_names = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z"
]

graph = tf.get_default_graph()
rcnn = Model()


def detect(img):
    with tf.device('/device:XLA_GPU:0'):
        with graph.as_default():
            results = rcnn.model.detect([img], verbose=1)
    r = results[0]
    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return r

    # print(xyz)
    # sortedXYZ = [list(i) for i in sorted(xyz, key=lambda item: item[1][1])]
    # print(sortedXYZ)
    # if 11 in r['class_ids']:
    #     CN_COD = r['rois'][list(r['class_ids']).index(11)]
    #     CN_COD[0] -= 20
    #     CN_COD[1] = 10
    #     CN_COD[2] += 15
    #     CN_COD[3] = width - 10
    #     res = filter(lambda x: True if CN_COD[0] < x[1][0] < CN_COD[2] else False, sortedXYZ)
    #     return getIndexOf(res, CN_COD)
    # else:
    #     return None

#
# def getIndexOf(sortedList, CN):
#     valid = [str(i) for i in range(0, 11)]
#     scores = []
#     coordinates = []
#     cardNumber = ''
#     prevDigit = [11, CN, 0]
#     for item in sortedList:
#         if str(item[0]) not in valid:
#             continue
#         if CN[0] < item[1][0] and item[1][2] < CN[2] and item[1][3] - item[1][1] < 500:
#             overlap = lambda x, y: False if (y[1][1] - x[1][1]) / (x[1][3] - x[1][1]) > 0.7 or x[0] == 11 or y[
#                 0] == 11 else True
#             if prevDigit is not None:
#                 if overlap(prevDigit, item):
#                     resolve = lambda digit1, digit2: digit1 if digit1[-1] > digit2[-1] else digit2
#                     candidate = resolve(prevDigit, item)
#                     if prevDigit[0] == 5 and item[0] == 6 or prevDigit[0] == 6 and item[0] == 5:
#                         candidate = prevDigit if prevDigit[0] == 5 else item
#                     if prevDigit[0] == 3 and item[0] == 8 or prevDigit[0] == 8 and item[0] == 3:
#                         candidate = prevDigit if prevDigit[0] == 3 else item
#                     class_id = candidate[0]
#                     cardNumber = cardNumber[:len(cardNumber) - 1]
#                     prevDigit = candidate
#                     scores = scores[:-1]
#                     coordinates = coordinates[:-1]
#                     scores.append(candidate[-1])
#                     coordinates.append(candidate[1])
#                 else:
#                     class_id = item[0]
#                     prevDigit = item
#                     scores.append(item[-1])
#                     coordinates.append(item[1])
#                 if class_id == 10:
#                     class_id = 0
#                 cardNumber += str(class_id)
#     return cardNumber, [float(i) for i in scores]


# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# data = detect(img_to_array(load_img(file)))
