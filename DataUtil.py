# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:52:52 2020

@author: rodge
"""


import os
import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from numpy import loadtxt

# from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random;
import cv2;
import shutil


path = './dataset/'
def load_data():
    images=[]
    fileNames = []
    destPath = ''
    files = os.listdir(path)
    # random.shuffle(files)
    for f in files:
        img_path = path + f
        fileNames.append(f[0]) # 因為中文只有一個字
        strs = f.split('_',1)
        num = int(strs[1].replace('.jpg',''))
        if num <= 40:
            destPath = './trainData/' + f
        else:
            destPath = './testData/' + f
        shutil.copyfile(img_path, destPath)
    print('load_data has Done')
    return images, fileNames


# def getEncodeList(fileNames):
#     labels = []
#     for twName in fileNames:
#         labels.append(ord(twName)) # 中文字轉換成Unicode
#     print('getEncodeList has done')
#     return labels

# def normalizeLabel(labels):
#     encodeList = []
#     newLabels = []
#     # 先統計
#     for label in labels:
#         if not label in encodeList:
#             encodeList.append(label)
# #     print(encodeList);
#     for label in labels:
#         newLabels.append(encodeList.index(label))
#     return encodeList, newLabels

images, fileNames = load_data()
# labels = getEncodeList(fileNames)

# encodeList, labels = normalizeLabel(labels)
# labels = np_utils.to_categorical(labels)
# images = np.array(images);
# labels = np.array(labels);
# images = abs(1 - (images / 255));
# images = images.reshape(images.shape[0], area).astype('float32');
# print(images.shape)
# print(labels.shape)

# for f in images:
#     path = './new/' + fileNames
#     cv2.imwrite()