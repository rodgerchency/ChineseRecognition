# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:38:56 2020

@author: rodger
"""

import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from numpy import loadtxt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random;
import cv2;

from DataLoader import DataLoader
from Util import Util

import matplotlib.pyplot as plt

tf.reset_default_graph()

w = 50; h = 50
area = w * h

loader = DataLoader()
x_train, y_train = loader.getTrain()
x_test, y_test = loader.getTest()
# util = Util()
# util.show(x_train[10].reshape(50,50))
# loader.showLabel(getIndex(y_train[10]))
len_label = y_train.shape[1]

print(len_label)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def answer(v_xs):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    ans = tf.argmax(y_pre, 1)
    result = sess.run(ans, feed_dict={xs: v_xs})
    return result

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def getIndex(arr):
    idx = np.where(arr == 1.0)
    return idx[0][0]

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, w*h])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, len_label])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, h, w, 1])

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32], 'W_conv1') # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64], 'W_conv2') # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([13*13*64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 13*13*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, len_label], 'W_fc2')
b_fc2 = bias_variable([len_label], 'b_fc2')
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()


model_path = "./temp/model.ckpt"
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    
    for i in range(2000):
        batch_xs, batch_ys = loader.getBatch(100)
        # print(batch_xs[0])
        # print(getIndex(batch_ys[0]))
        # batch_xs, batch_ys = getRandom(x_train, y_train, 100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i != 0 and i % 50 == 0:
            print(compute_accuracy(x_test[:1000], y_test[:1000]))
    save_path = saver.save(sess, model_path)
    print(compute_accuracy(x_test[:1000], y_test[:1000]))
    print('Finish')
    
    
    
print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
#    print("Model restored from file: %s" % save_path)
    print(compute_accuracy(x_test[:1000], y_test[:1000]))
#    print(answer(mnist.test.images[0:1]))
    for i in range(10):
        ans = answer(x_test[i:i+1])
        print(ans, getIndex(y_test[i]))
        loader.showLabel(ans[0])
        loader.showLabel(getIndex(y_test[i]))
          # loader.showLabel(getIndex(y_test[6]))
          # plt.imshow(x_test[5].reshape(50,50))
#        plt.show();
#    print("Answer:", sess.run(ans, feed_dict={xs: mnist.test.images[0:1]}))
   