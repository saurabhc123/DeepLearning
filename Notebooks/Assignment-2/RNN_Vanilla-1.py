import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv

DATA_DIR = 'MNIST'
sentiment_data = 'sentiment-data'


reader = csv.reader(open(sentiment_data +'/word-vectors.txt'))

word_vector_dict = {}
for row in reader:
    key = row[0]
    if key in word_vector_dict:
        # implement your duplicate row handling here
        pass
    word_vector_dict[key] = row[1:]
print word_vector_dict['agassi']


sent1 = "agassi comes"
sent2 = "agassi"
sent1vec = []
sent2vec = []

for word in sent1.split(' '):
    sent1vec.append(word_vector_dict[word])

for word in sent2.split(' '):
    sent2vec.append(word_vector_dict[word])

batched_data = tf.train.batch(
    tensors=[sent1vec],
    batch_size=55,
    dynamic_pad=True,
    name="y_batch"
)

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

element_size = 28;
time_steps = 28;
num_classes = 10
batch_size = 128;
n_iterations = 3000;
hidden_layer_size = 128

#Setting up the input and labels placeholders
_inputs = tf.placeholder(tf.float32,shape=[None, time_steps,
                                           element_size],
                                           name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],name='inputs')

# TensorFlow built-in functions
# Creating the RNN cell and creating the outputs
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                     mean=0,stddev=.01))
bl = tf.Variable(tf.truncated_normal([num_classes],mean=0,stddev=.01))

def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl

last_rnn_output = outputs[:,-1,:]
#final_output = get_linear_layer(last_rnn_output)

#Create the network
final_output = tf.layers.dense(states, num_classes)
softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=y)

#Cross entropy and the optimizer + minimizer
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#Metrics
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

#Initialize session
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#Initialize test data
test_data = mnist.test.images[:batch_size].reshape((-1,
                                            time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(n_iterations):

       batch_x, batch_y = mnist.train.next_batch(batch_size)
       batch_x = batch_x.reshape((batch_size, time_steps, element_size))
       sess.run(train_step,feed_dict={_inputs:batch_x,
                                      y:batch_y})
       if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})
            loss = sess.run(cross_entropy,feed_dict={_inputs:batch_x,y:batch_y})
            print ("Iter " + str(i) + ", Minibatch Loss= " + \
					  "{:.6f}".format(loss) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc))

print ("Testing Accuracy:",
    sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label}))


