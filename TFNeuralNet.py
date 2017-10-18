import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np

# import tensorflow_gpu as tfg

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
device_name = "/cpu:0"


def initialize():
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    #We can use the neuron_layer method too. We are just using the library function for now.
    with tf.device(device_name):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

    #Define the cross-entropy loss
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                   labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    #Setup the optimizer
    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    #Measure the accuracy
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")

    n_epochs = 400
    batch_size = 50
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
                save_path = saver.save(sess, "./model/my_model_final.ckpt")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(np.zeros([n_neurons]), name="biases")
        z = tf.matmul(W, X) + b
        return activation(z)

