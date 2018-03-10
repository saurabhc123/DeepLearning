import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import ConvHelper

DATA_DIR = 'MNIST'
STEPS = 5000
MINIBATCH_SIZE = 100

imgDim = 28
imgChannels = 1
nClasses = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = ConvHelper.conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = ConvHelper.max_pool_2x2(conv1)
conv2 = ConvHelper.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = ConvHelper.max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
full_1 = tf.nn.relu(ConvHelper.full_layer(conv2_flat, 1024))
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
y_conv = ConvHelper.full_layer(full1_drop, 10)

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels= y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEPS):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print ("step {}, training accuracy {}".format(i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            X = mnist.test.images.reshape(10, 1000, 784)
            Y = mnist.test.labels.reshape(10, 1000, 10)
            test_accuracy = np.mean([sess.run(accuracy,feed_dict = {x: X[i], y_: Y[i],keep_prob: 1.0})
                                      for i in range(10)])
    print ("test accuracy: {}".format(test_accuracy))
