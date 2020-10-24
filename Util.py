# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:52:18 2020

@author: rodger
"""

import matplotlib.pyplot as plt
import cv2

class Util:

    def show(self, img):
        plt.imshow(img)
        plt.show()

    def saveImg(self, img, path):
        # cv2.imwrite(path, img)
        cv2.imencode('.jpg', img)[1].tofile(path) 
