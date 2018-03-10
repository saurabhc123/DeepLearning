import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import ConvHelper


DATA_DIR = 'MNIST'
STEPS = 1000
MINIBATCH_SIZE = 100

imgDim = 28
imgChannels = 1
nClasses = 10




def GetInputData():
    return input_data.read_data_sets(DATA_DIR, one_hot=True)

def BuildNetwork():
    x_image = tf.reshape(x, [-1, imgDim, imgDim, 1])
    conv1 = ConvHelper.conv_layer(x_image, shape=[3, 3, 1, 32])
    conv1_pool = ConvHelper.max_pool_2x2(conv1)
    # The above results in a [[13 x 13] x 32 filter maps
    conv2 = ConvHelper.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = ConvHelper.max_pool_2x2(conv2)
    # The above results in a [[5 x 5] x 64 filter maps
    conv2_flat = tf.reshape(conv2_pool, [-1, 5 * 5 * 64])
    full = tf.nn.relu(ConvHelper.full_layer(conv2_flat, 1024))
    y_conv = ConvHelper.full_layer(full, nClasses)
    return  y_conv

def runClassification(y_labels , y_conv, data):
    mnist = data
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_labels, logits = y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(STEPS):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y: batch[1]})
            print ("step {}, training accuracy {}".format(i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
            X_ = mnist.test.images.reshape(10, 1000, 784)
            Y_ = mnist.test.labels.reshape(10, 1000, 10)
            test_accuracy = np.mean([sess.run(accuracy,
                                 feed_dict={x:X_[i], y:Y_[i]})
                                 for i in range(10)])
        print ("test accuracy: {}".format(test_accuracy))

data = GetInputData()
x = tf.placeholder(tf.float32, shape=[None, imgDim * imgDim])
y = tf.placeholder(tf.float32, shape=[None, nClasses])
y_conv_network = BuildNetwork()
runClassification(y, y_conv_network,data)
