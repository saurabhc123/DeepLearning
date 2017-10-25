import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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


#sm = np.ones((70,50), dtype=np.float32);
#psm = getPaddedSentenceMatrix(sm)
#print psm
#word_vec_d = getWordVectorDict()
#sm = getVectorForSentence("as said that in for on", word_vec_d)
#print sm

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


word_vector_size = 50;
time_steps = 100;
num_classes = 2
batch_size = 1000;
n_iterations = 10;
hidden_layer_size = 50

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

#mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)



#Setting up the input and labels placeholders
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps,
                                            word_vector_size],
                         name='inputs')
y = tf.placeholder(tf.int32, shape=[None],name='inputs')
y_one_hot = tf.one_hot( y , num_classes )

# TensorFlow built-in functions
# Creating the RNN cell and creating the outputs
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size, activation= tf.tanh)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

#Create the network
final_output = tf.layers.dense(states, num_classes)
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=y_one_hot)

#Cross entropy and the optimizer + minimizer
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.1, 0.9, 0.01).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

#Metrics
correct_prediction = tf.equal(tf.argmax(y_one_hot,1), tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

#Initialize session
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#Initialize test data
#test_data = mnist.test.images[:batch_size].reshape((-1,time_steps, word_vector_size))
#test_label = mnist.test.labels[:batch_size]

for epoch in range(n_iterations):
    print "Epoch:", epoch
    for j in range(len(training_data)/batch_size):
        #print "j:", j
        startIndex = j*batch_size
        endIndex = startIndex + batch_size
        batch_x = np.array(training_data[startIndex : endIndex]).reshape((-1,time_steps, word_vector_size))
        batch_y = np.array(training_labels[startIndex : endIndex]).reshape(batch_size)
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        #batch_x = batch_x.reshape((batch_size, time_steps, word_vector_size))
        sess.run(train_step, feed_dict={_inputs: batch_x,
                                        y: batch_y})
        if epoch % 1 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})
            loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x, y: batch_y})
            print ("Iter " + str(epoch) + ", Minibatch Loss= " + \
                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
                   "{:.5f}".format(acc))


print ("Testing Accuracy:",
    sess.run(accuracy, feed_dict={_inputs: test_data, y: test_labels}))


