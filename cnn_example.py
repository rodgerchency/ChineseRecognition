#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:07:40 2020

@author: rodger_chen
"""

# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
#from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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
   return tf.Variable(initial, name=name)

def bias_variable(shape, name):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
# W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1), name='W_conv1') # patch 5x5, in size 1, out size 32
# b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
W_conv1 = weight_variable([5,5, 1,32], 'W_conv1') # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
# W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1), name='W_conv2') # patch 5x5, in size 32, out size 64
# b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')
W_conv2 = weight_variable([5,5, 32, 64], 'W_conv2') # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64], 'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
# W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1), name='W_fc1')
# b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
W_fc1 = weight_variable([7*7*64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
# W_fc2 =tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1), name='W_fc2')
# b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc2')
W_fc2 = weight_variable([1024, 10], 'W_fc2')
b_fc2 = bias_variable([10], 'b_fc2')
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluate model
# argmax returns the index with the largest value across axes of a tensor
#ans = tf.argmax(prediction, 1)

#sess = tf.Session()
## important step
## tf.initialize_all_variables() no long valid from
## 2017-03-02 if using tensorflow >= 0.12
#if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#    init = tf.initialize_all_variables()
#else:
#    init = tf.global_variables_initializer()
#sess.run(init)

#for i in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
#    if i % 50 == 0:
#        print(compute_accuracy(
#            mnist.test.images[:1000], mnist.test.labels[:1000]))
        
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()


# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

model_path = "./temp/model.ckpt"
with tf.Session() as sess:

  # Run the initializer
  sess.run(init)
  for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
      if i % 50 == 0:
          print(compute_accuracy(
              mnist.test.images[:1000], mnist.test.labels[:1000]))
  # Save model weights to disk
  save_path = saver.save(sess, model_path)
  print("Model saved in file: %s" % save_path)

# Running a new session
#print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
#    print("Model restored from file: %s" % save_path)
    print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
#    print(answer(mnist.test.images[0:1]))
    for i in range(10):
         print(answer(mnist.test.images[i:i+1]), mnist.test.labels[i])
#        plt.imshow(mnist.test.images[i].reshape(28,28));
#        plt.show();
#    print("Answer:", sess.run(ans, feed_dict={xs: mnist.test.images[0:1]}))
   