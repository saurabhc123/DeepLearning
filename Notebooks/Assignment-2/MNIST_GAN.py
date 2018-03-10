import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

MNIST_Input_Dim = 784
layer1_size = 128

GENERATOR_Input_Dim = 100


#Create the DISCRIMINATOR first
X = tf.placeholder(tf.float32, shape = [None,MNIST_Input_Dim])
D_W1 = tf.Variable(xavier_init([MNIST_Input_Dim, layer1_size]))
D_b1 = tf.Variable(tf.zeros([layer1_size]))

D_W2 = tf.Variable(xavier_init([layer1_size, 1]))
D_b2 = tf.Variable(tf.zeros([1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

#Create the GENERATOR next
Z = tf.placeholder(tf.float32, shape=[None,GENERATOR_Input_Dim])
G_W1 = tf.Variable(xavier_init([GENERATOR_Input_Dim, layer1_size]))
G_b1 = tf.Variable(tf.zeros([layer1_size]))

G_W2 = tf.Variable(xavier_init([layer1_size,MNIST_Input_Dim]))
G_b2 = tf.Variable(tf.zeros([MNIST_Input_Dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    h2 = tf.nn.sigmoid(tf.matmul(h1, G_W2) + G_b2)
    return h2

def discriminator(x):
    h1 = tf.nn.relu(tf.matmul(x,D_W1) + D_b1)
    logits = tf.matmul(h1, D_W2) + D_b2
    outputs = tf.nn.sigmoid(logits)
    return outputs, logits

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


#Now, create the computational graph
g_samples = generator(Z)
D_real, D_real_logits = discriminator(X)
D_fake, D_fake_logits = discriminator(g_samples)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels = tf.ones_like(D_real_logits)))
#The discriminator is trying to assert that the fake sample has the label 0. It will optimize based on this assertion,
#to label the fake as fake
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels = tf.zeros_like(D_fake_logits)))
D_loss = D_loss_fake + D_loss_real

#The generator is trying to assert that the fake example has the label 1. It will optimize based on this assertion, to
#set the generator networks weights so that the discriminator is fooled to believe that this sample is indeed 1.
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logits, labels = tf.ones_like(D_fake_logits)))

D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list= theta_D)
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list= theta_G)

mini_batch_size = 128



mnist = input_data.read_data_sets('MNIST', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(g_samples, feed_dict={Z: sample_Z(16, GENERATOR_Input_Dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mini_batch_size)
    D_loss_curr, _ = sess.run([D_loss,D_optimizer],feed_dict={X:X_mb,Z:sample_Z(mini_batch_size,GENERATOR_Input_Dim)})
    G_loss_curr, _ = sess.run([G_loss, G_optimizer],feed_dict={Z:sample_Z(mini_batch_size,GENERATOR_Input_Dim)})


    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {0:.2f}'. format(D_loss_curr))
        print('G_loss: {0:.2f}'.format(G_loss_curr))
        print()






#print(sample_Z(5,10))






