# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:41:10 2020

@author: rodger
"""

import os
import numpy as np
from keras.preprocessing import image as Image
from keras.utils import np_utils, plot_model
import random as Random

class DataLoader:

    def __init__(self):
        
        self._w = 50; self._h = 50
        self._area = self._w * self._h
        self._currIdx = 0

        # Train DataSet
        self._trainImages=[]
        self._trainFileNames = []
        self._trainLabelOneHots = []

        trainFiles = os.listdir('./trainData')
        Random.shuffle(trainFiles)
        for f in trainFiles:
            img_path = './trainData/' + f
            self._trainFileNames.append(f[0]) # 紀錄第一個字
            img = Image.load_img(img_path, grayscale=True)
            img_array = Image.img_to_array(img)
            self._trainImages.append(img_array)
        
        self._trainLabels = [] # input idx of image to find the Unicode of trainImage
        for twName in self._trainFileNames:
            self._trainLabels.append(ord(twName))
        
        # Test DataSet
        self._testImages=[]
        self._testFileNames = []
        self._testLabelOneHots = []

        testFiles = os.listdir('./testData')
        Random.shuffle(testFiles)
        for f in testFiles:
            img_path = './testData/' + f
            self._testFileNames.append(f[0]) # 紀錄第一個字
            img = Image.load_img(img_path, grayscale=True)
            img_array = Image.img_to_array(img)
            self._testImages.append(img_array)

        self._testLabels = [] # input idx of image to find the Unicode of trainImage
        for twName in self._testFileNames:
            self._testLabels.append(ord(twName))

        # EncodeMap        
        self._encodeMap = [] # 
        newLabels = []
        # 先統計
        for label in self._trainLabels:
            if not label in self._encodeMap:
                self._encodeMap.append(int(label))
        self._encodeMap.sort()

        # 
        self._trainImages = np.array(self._trainImages)
        self._trainImages = abs(1 - (self._trainImages / 255))
        self._trainImages = self._trainImages.reshape(self._trainImages.shape[0], self._area).astype('float32')

        self._testImages = np.array(self._testImages)
        self._testImages = abs(1 - (self._testImages / 255))
        self._testImages = self._testImages.reshape(self._testImages.shape[0], self._area).astype('float32')
        
        trainLabels = []
        for label in self._trainLabels:
            trainLabels.append(self._encodeMap.index(label))
        
        testLabels = []
        for label in self._testLabels:
            testLabels.append(self._encodeMap.index(label))

        self._trainLabelOneHots = np.array(np_utils.to_categorical(trainLabels))
        self._testLabelOneHots = np.array(np_utils.to_categorical(testLabels))

    def getBatch(self, size):
        
        total = self._trainImages.shape[0]
        if total < self._currIdx + size:
            diff = (self._currIdx + size) - total
            last = size - diff
            trainImg = np.concatenate((self._trainImages[self._currIdx: self._currIdx + last + 1], self._trainImages[0: diff]))
            trainLabel = np.concatenate((self._trainLabelOneHots[self._currIdx: self._currIdx + last + 1], self._trainLabelOneHots[0: diff]))
            self._currIdx = diff
        else:
            trainImg = self._trainImages[self._currIdx: self._currIdx + size]
            trainLabel = self._trainLabelOneHots[self._currIdx: self._currIdx + size]
            self._currIdx = self._currIdx + size
            if self._currIdx == total:
                self._currIdx = 0
        return trainImg, trainLabel


    def getTrain(self):
        return self._trainImages, self._trainLabelOneHots

    def getTest(self):
        return self._testImages, self._testLabelOneHots

    # ex : loader.showLabel(trainLabels[1])
    # idx : trainLabel
    def showLabel(self, idx):
        print(self._encodeMap[idx], ',' + chr(self._encodeMap[idx]))

