import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import csv
import random


input_filename = 'text_file.txt'


def get_words_from_file(input_filename):
    f = open(input_filename,"r")
    buffer = f.read()
    words = buffer.split(' ')
    word_dict = {}
    word_list = []
    i = 0
    for word in words:
        if(not word_dict.__contains__(word)):
            word_dict[word] = i
            word_list.append(word)
            i += 1


    return word_dict, word_list

def get_n_words_random(word_list, n):
    random_index = random.randrange(1, len(word_list) -  n - 1)
    return np.array(range(random_index - 1,random_index + n - 1)), np.array(range(random_index + n - 1,random_index + n))

def get_n_words(word_list, start_index, n):
    random_index = start_index
    return np.array(range(random_index - 1,random_index + n - 1)), np.array(range(random_index + n - 1,random_index + n))

def get_training_data(word_list):
    train_x = []
    train_y = []
    for i in range(len(word_list)/batch_size):
        x, y = get_n_words(word_list, i*batch_size, n_steps)
        train_x.append(x)
        train_y.append(y)
    return train_x, train_y


words_dict, words_list = get_words_from_file(input_filename)
#fifty_words = get_n_words(words_list,50)
#print(fifty_words)

n_inputs = 1
n_steps = 5
n_neurons = 15
n_outputs = 1
learning_rate = 0.001

X = tf.placeholder(tf.int32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
#cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.tanh)
#cell = tf.contrib.rnn.BasicLSTMCell(n_neurons, forget_bias=1.0)
#output_cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.tanh), output_size = n_outputs)
#outputs , states = tf.nn.dynamic_rnn(output_cell, X, dtype= tf.float32)
#logits = tf.layers.dense(states, n_outputs)
#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y , logits= logits)
with tf.variable_scope("lstm"):
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_neurons, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(tf.contrib.rnn.BasicLSTMCell(n_neurons, forget_bias=1.0),X, dtype=tf.int32)

weights = {
        'linear_layer': tf.Variable(tf.truncated_normal([n_neurons,
                                                         n_outputs],
                                                         mean=0,stddev=.01))
}


# Extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[1],
                         weights["linear_layer"])
loss = tf.reduce_mean(tf.square(final_output - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.RMSPropOptimizer(0.1, 0.9, 0.01)
training_op = optimizer.minimize(loss)
#correct = tf.nn.in_top_k(logits, y, 1)
#accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

n_iterations = 5
batch_size = 5
batches = 5


init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_iterations):
        print "Epoch:", epoch
        for j in range(batches):  #((len(words_list) - n_steps)/batch_size):
            batch_x , batch_y = get_training_data(words_list)
            sess.run(training_op, feed_dict={X: batch_x,
                                            y: batch_y})
        if epoch % 1 == 0:
            mse = loss.eval(feed_dict={X: batch_x,y: batch_y})
            print ("Iter " + str(epoch) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) )