import numpy as np
import tensorflow as tf
import time
from sklearn.utils import shuffle
import os
import sys
import tarfile
from six.moves import urllib
import argparse
import pickle
import matplotlib.pyplot as plt
from PIL import Image


# ##############################################################################
# ################################## SETTINGS ##################################
# ##############################################################################


class Settings(object):
    batch_size = 128
    max_n_epochs = 20
    nb_filters_conv1 = 64
    nb_filters_conv2 = 64
    image_size = 30
    learning_rate = 0.001

    nb_quickval_batches = 10

    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path_data = './cifar10_data'
    path_training_summaries = './cifar10_training_summaries'
    nb_run = 1

# ##############################################################################
# ################################## NETWORK ###################################
# ##############################################################################


class Network(object):
    def __init__(self):
        self.images = tf.placeholder(tf.float32,
                                     [Settings.batch_size, Settings.image_size * Settings.image_size],
                                     name='images')
        self.depth_major = tf.reshape(self.images,
                                      [Settings.batch_size, 3, Settings.image_size, Settings.image_size])
        self.images_reshaped = tf.transpose(self.depth_major, [0, 2, 3, 1])
        tf.summary.image('inputImage', self.images_reshaped)
        self.labels = tf.placeholder(tf.int64, [Settings.batch_size], name='labels')
        # conv1
        with tf.variable_scope('conv1') as scope:
            self.kernels = tf.get_variable(
                name='kernels',
                shape=[5, 5, 3, Settings.nb_filters_conv1],
                initializer=tf.truncated_normal_initializer(stddev=5e-2))
            self.conv = tf.nn.conv2d(
                self.images_reshaped,
                filter=self.kernels,
                strides=[1, 1, 1, 1],
                padding='SAME')
            self.biases = tf.get_variable(
                name='biases',
                shape=[Settings.nb_filters_conv1],
                initializer=tf.constant_initializer(0.0))
            self.pre_activation = tf.nn.bias_add(self.conv, self.biases)
            self.conv1 = tf.nn.relu(self.pre_activation, name=scope.name)
            tf.summary.histogram('conv1', self.conv1)

        # pool
        self.pool1 = tf.nn.max_pool(
            self.conv1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool1')

        # normalization over all samples in batch for each channel/pixel independently
        self.norm1 = tf.nn.local_response_normalization(
            self.pool1,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm')

        # conv2
        with tf.variable_scope('conv2') as scope:
            self.kernels = tf.get_variable(
                name='kernels',
                shape=[5, 5, Settings.nb_filters_conv1, Settings.nb_filters_conv2],
                initializer=tf.truncated_normal_initializer(stddev=5e-2))
            self.conv = tf.nn.conv2d(
                input=self.norm1,
                filter=self.kernels,
                strides=[1, 1, 1, 1],
                padding='SAME')
            self.biases = tf.get_variable(
                name='biases',
                shape=[Settings.nb_filters_conv2],
                initializer=tf.constant_initializer(0.1))
            self.pre_activation = tf.nn.bias_add(self.conv, self.biases)
            self.conv2 = tf.nn.relu(self.pre_activation, name=scope.name)
            tf.summary.histogram('conv2', self.conv2)

        # norm2
        self.norm2 = tf.nn.local_response_normalization(
            self.conv2,
            depth_radius=4,
            bias=1.0,
            alpha=0.0001 / 9.0,
            beta=0.75,
            name='norm2')

        # pool2
        self.pool2 = tf.nn.max_pool(
            self.norm2,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool2')

        # flatten pool2 for fully connected layer
        self.pool2_flat = tf.reshape(self.pool2, [Settings.batch_size, -1])
        self.dim = self.pool2_flat.get_shape()[1].value
        # dense1
        with tf.variable_scope("dense1") as scope:
            self.weights = tf.get_variable(
                name='weights',
                shape=[self.dim, 384],
                initializer=tf.truncated_normal_initializer(stddev=0.04))
            self.biases = tf.get_variable(
                name='biases',
                shape=[384],
                initializer=tf.constant_initializer(0.1))
            self.pre_activation = tf.add(
                tf.matmul(self.pool2_flat, self.weights),
                self.biases,
                name='pre_activation')
            self.dense1 = tf.nn.relu(self.pre_activation, name='dense1')
            tf.summary.histogram('dense1', self.dense1)

        # dense2
        with tf.variable_scope('dense2') as scope:
            self.weights = tf.get_variable(
                name='weights',
                shape=[384, 192],
                initializer=tf.truncated_normal_initializer(stddev=0.04))
            self.biases = tf.get_variable(
                name='biases',
                shape=[192],
                initializer=tf.constant_initializer(0.1))
            self.pre_activation = tf.add(
                tf.matmul(self.dense1, self.weights),
                self.biases,
                name='pre_activation')
            self.dense2 = tf.nn.relu(self.pre_activation, name='dense2')
            tf.summary.histogram('dense2', self.dense2)

        # softmax_linear (without softmax)
        with tf.variable_scope('softmax_linear') as scope:
            self.weights = tf.get_variable(
                name='weights',
                shape=[192, 10],
                initializer=tf.truncated_normal_initializer(stddev=1 / 192.0))
            self.biases = tf.get_variable(
                name='biases',
                shape=[10],
                initializer=tf.constant_initializer(0.0))
            self.softmax_linear = tf.add(
                tf.matmul(self.dense2, self.weights),
                self.biases,
                name=scope.name)
            # No activation/softmax yet since
            # tf.nn.sparse_softmax_cross_entropy_with_logits accepts
            # unscaled logits
            tf.summary.histogram('softmax_linear', self.softmax_linear)

        # objective
        with tf.variable_scope('objective') as scope:
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.softmax_linear,
                    labels=self.labels))
            self.top1 = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self.softmax_linear, self.labels, 1),
                tf.float32))
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('top1', self.top1)

        # optimizer
        with tf.variable_scope('optimizer') as scope:
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=Settings.learning_rate)
            self.varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.gradients = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(self.gradients)

        self.merged_summary_op = tf.summary.merge_all()


def train(net, batch_maker):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(
            Settings.path_training_summaries + '/run_' +
            str(Settings.nb_run), sess.graph)
        Settings.nb_run += 1
        current_epoch = 1
        while current_epoch <= Settings.max_n_epochs:
            time_minibatch_start = time.time()
            images, labels = batch_maker.next_train_batch()
            _, loss, top1, summary = sess.run(
                [net.train_op, net.loss, net.top1, net.merged_summary_op],
                feed_dict={net.images: images, net.labels: labels})
            summary_writer.add_summary(summary)
            print('Epoch ' + str(current_epoch).zfill(3) +
                  '| Loss: %.3f' % (loss) +
                  '| Top1: %.3f' % (top1) +
                  '| Time passed: %.f' % (time.time() - time_minibatch_start))
            if current_epoch % 10 == 0:
                quickval(sess, net, batch_maker, summary_writer)
            current_epoch += 1


def quickval(sess, net, batch_maker, summary_writer):
    loss_store = []
    top1_store = []
    for i in range(Settings.nb_quickval_batches):
        images, labels = batch_maker.next_test_batch(Settings.batch_size)
        loss, top1, summary = sess.run([net.loss, net.top1, net.merged_summary_op],
                                       feed_dict={net.images: images, net.labels: labels})
        loss_store.append(loss)
        top1_store.append(top1)

        summary_writer.add_summary(summary)

    # Compute mean over batches
    mean_loss = np.mean(loss_store)
    mean_top1 = np.mean(top1_store)

    # Print results
    print('Quickval: Loss: %.3f ' % (mean_loss) +
          '| Top1: %.3f ' % (mean_top1))


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website.
	Copied from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py"""
    if not os.path.exists(Settings.path_data):
        os.makedirs(Settings.path_data)
    filename = Settings.data_url.split('/')[-1]
    filepath = os.path.join(Settings.path_data, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(Settings.data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(Settings.path_data, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(Settings.path_data)


class Batch_maker(object):
    def __init__(self):
        self.current_batch = 1
        self.images = []
        self.labels = []
        self.leftover_images = []
        self.leftover_labels = []
        self.index_list = []

    def next_train_batch(self, batch_size=Settings.batch_size):
        # Fill up images and labels if necessary
        self.images = self.leftover_images
        self.labels = self.leftover_labels
        if len(self.images) < batch_size:
            with open('./cifar10_data/cifar-10-batches-py/data_batch_' + str(self.current_batch), 'rb') as file:
                data_dict = pickle.load(file, encoding='bytes')
            self.images.extend(data_dict[b'data'])
            self.labels.extend(data_dict[b'labels'])
            # Increment current_batch so another file will be used next time
            self.current_batch += 1
            if self.current_batch == 5:
                self.current_batch = 1
        # Shuffle all images and labels (in unison)
        self.images, self.labels = shuffle(self.images, self.labels)
        # Save leftover images and labels
        self.leftover_images = self.images[batch_size:]
        self.leftover_labels = self.labels[batch_size:]
        self.images = self.images[:batch_size]
        self.labels = self.labels[:batch_size]
        self.distort_images()
        return self.images, self.labels

    def next_test_batch(self, batch_size=Settings.batch_size):
        with open('./cifar10_data/cifar-10-batches-py/test_batch', 'rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images, labels = shuffle(images, labels, n_samples=batch_size)

        return images, labels

    def distort_images(self):
        for image in self.images:
            reshaped_image = tf.reshape(image, [32, 32, 3])
            show(reshaped_image)
            distorted_image = tf.random_crop(reshaped_image,
                                             [Settings.image_size, Settings.image_size, 3])
            distorted_image = tf.image.random_flip_left_right(distorted_image)
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.8)
            normalized_image = tf.image.per_image_standardization(
                distorted_image)


def show(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()


# ##############################################################################
# ################################### MAIN #####################################
# ##############################################################################

if __name__ == '__main__':
    # Read basic model parameters
    """
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=128, 
						help='number of images to process in batch')

	FLAGS = parser.parse_args()
	"""
    maybe_download_and_extract()

    bm = Batch_maker()
    images, labels = bm.next_train_batch()
    show(images[1])
# net = Network()
# train(net, bm)
