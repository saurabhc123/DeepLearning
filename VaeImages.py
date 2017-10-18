import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from scipy.misc import imresize

n_channels = 3
n_inputs = 100 * 100 * n_channels # for MNIST
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
n_digits = 100
n_epochs = 100
batch_size = 150

learning_rate = 0.001

def initialize():
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            activation_fn=tf.nn.elu,
            weights_initializer=tf.contrib.layers.variance_scaling_initializer()):
        X = tf.placeholder(tf.float32, [None, n_inputs])
        hidden1 = fully_connected(X, n_hidden1)
        hidden2 = fully_connected(hidden1, n_hidden2)
        hidden3_mean = fully_connected(hidden2, n_hidden3, activation_fn=None)
        hidden3_gamma = fully_connected(hidden2, n_hidden3, activation_fn=None)
        hidden3_sigma = tf.exp(0.5 * hidden3_gamma)
        noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
        hidden3 = hidden3_mean + hidden3_sigma * noise
        hidden4 = fully_connected(hidden3, n_hidden4)
        hidden5 = fully_connected(hidden4, n_hidden5)
        logits = fully_connected(hidden5, n_outputs, activation_fn=None)
        outputs = tf.sigmoid(logits)

    reconstruction_loss = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))
    latent_loss = 0.5 * tf.reduce_sum(
        tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
    cost = reconstruction_loss + latent_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(cost)

    init = tf.global_variables_initializer()
    #X_train, X_test = [...] # load the dataset

    n_iterations = 100

    # See how this works w/ Celeb Images or try your own dataset instead:
    dirname = '/Users/saur6410/Documents/DL/Images'

    # Load every image file in the provided directory
    filenames = [os.path.join(dirname, fname)
                 for fname in os.listdir(dirname) if '.jpg' in fname]

    # Read every filename as an RGB image - Change the below to :3 for RGB
    imgs = [plt.imread(fname)[...,:n_channels] for fname in filenames]

    # Crop every image to a square
    #imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs]


    # Then resize the square image to 100 x 100 pixels
    imgs = [resize(img_i, (100, 100)) for img_i in imgs]
    # Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
    X_train = np.array(imgs).astype(np.float32)
    plt.interactive(True)
    #plt.imshow(montage(X_train.reshape([100,100,100]), saveto='dataset.png'))
    #plt.imshow(imgs[0])
    #plt.show()
    x = X_train.reshape([100,n_inputs])
    print(x.shape)
    #print (X_train.shape)

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            training_op.run(feed_dict={X: x})  # no labels (unsupervised)

        codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
        outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

    for iteration in range(n_digits):
        plt.subplot(n_digits, 10, iteration + 1)
        plt.figure(figsize=(10, 10))
        print(outputs_val.shape)
        plt.imshow(montage(outputs_val.reshape([100,100,100,n_channels]), saveto='dataset.png'))
        plt.show()



def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.

    Also saves the file to the destination specified by `saveto`.

    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    #imsave(arr=np.squeeze(m), name=saveto)
    return m