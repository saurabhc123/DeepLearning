import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import csv


import numpy as np

DATA_DIR = 'MNIST'
sentiment_data = 'sentiment-data'


def getWordVectorDict():
    reader = csv.reader(open(sentiment_data +'/word-vectors-refine.txt'))

    word_vector_dict = {}
    for row in reader:
        key = row[0]
        if key in word_vector_dict:
            # implement your duplicate row handling here
            pass
        word_vector_dict[key] = np.array(row[1:])
    return word_vector_dict

def getPaddedSentenceMatrix(sentenceMatrix):
    wordCount = 100
    return np.vstack((sentenceMatrix, np.zeros((wordCount - np.shape(sentenceMatrix)[0],np.shape(sentenceMatrix)[1]), dtype=np.float32)))

def getVectorForSentence(sentence, word_vec_dict):
    sentence_matrix = []
    for word in sentence.split(' '):
        word_vec = word_vec_dict[word]
        if(len(sentence_matrix) == 0):
            sentence_matrix = word_vec
        else:
            sentence_matrix = np.vstack((sentence_matrix,word_vec))
    return getPaddedSentenceMatrix(sentence_matrix)

DATA_DIR = 'MNIST'
vocabulary_size = 28
n_inputs = 50
n_steps = 100
n_neurons = 150
n_outputs = 2
learning_rate = 0.001

init = tf.global_variables_initializer()


word_vector_size = 50;
time_steps = 100;
num_classes = 2
batch_size = 1000;
n_iterations = 10;
hidden_layer_size = 50

def getData(fileName):
    reader = csv.reader(open(sentiment_data +'/' + fileName))
    trainingData = []
    for row in reader:
        data = {}
        data['label'] =  1 if row[0] == 'postive' else 0
        data['sentence'] = row[1:]
        trainingData.append(data)
    return trainingData

word_vec_dict = getWordVectorDict()


def transform(row):
    return row['label'], getVectorForSentence(row['sentence'][0], word_vec_dict)

training_data = getData('train.csv')
training_rows  = map(lambda row: transform(row), training_data)
training_data = map(lambda row: row[1], training_rows)
training_labels = map(lambda row: row[0], training_rows)
#test_data = map(lambda row: transform(row), getData('test.csv'))
test_data = getData('test.csv')
test_rows  = map(lambda row: transform(row), test_data)
test_data = map(lambda row: row[1], test_rows)
test_labels = map(lambda row: row[0], test_rows)
print training_data[0]

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
outputs , states = tf.nn.dynamic_rnn(basic_cell, X, dtype= tf.float32)
logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y , logits= logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

#mnist = input_data.read_data_sets(DATA_DIR)
#X_test = mnist.test.images.reshape(-1,n_steps,n_inputs)
#y_test = mnist.test.labels

n_epochs = 10
batch_size = 1000

with tf.Session() as sess:
    init.run()
    for i in range(n_epochs):
        for j in range(len(training_data)/batch_size):
            print "j:", j
            startIndex = j*batch_size
            endIndex = startIndex + batch_size
            batch_x = np.array(training_data[startIndex : endIndex]).reshape((-1,time_steps, word_vector_size))
            batch_y = np.array(training_labels[startIndex : endIndex]).reshape(batch_size)
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            #batch_x = batch_x.reshape((batch_size, time_steps, word_vector_size))
            sess.run(training_op, feed_dict={X: batch_x,
                                            y: batch_y})
            if i % 1 == 0:
                acc = sess.run(accuracy, feed_dict={X: batch_x, y: batch_y})
                loss = sess.run(xentropy, feed_dict={X: batch_x, y: batch_y})
                print ("Iter " + str(i) + ", Minibatch Loss= " + \
                        ", Training Accuracy= " + \
                       "{:.5f}".format(acc))


    print ("Testing Accuracy:",
    sess.run(accuracy, feed_dict={X: test_data, y: test_labels}))



