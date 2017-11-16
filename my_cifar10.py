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


# ##############################################################################
# ################################## SETTINGS ##################################
# ##############################################################################

class Settings(object):
    nb_filters_conv1 = 64
    nb_filters_conv2 = 64
    learning_rate = 0.001
    image_size = 30

    nb_quickval_batches = 10

    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path_data = './cifar10_data'
    path_training_summaries = './cifar10_training_summaries'
    nb_run = 1


# Read basic model parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='number of images to process in batch.')

parser.add_argument('--nb_steps', type=int, default=100000,
                    help='number of steps/batches to train on.')

parser.add_argument('--learn_act_params', type=bool, default=False,
                    help='Whether to learn activation parameters.')

FLAGS = parser.parse_args()


# ##############################################################################
# ################################## NETWORK ###################################
# ##############################################################################

class Network(object):
    def __init__(self, batch_maker):
        with tf.variable_scope('queue') as scope:
            self.images, self.labels = input_pipeline.

        self.keep_prob = tf.placeholder(tf.float32)
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

        # normalization over all samples in batch for each channel/pixel
        # independently
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

        self.pool2_flat = tf.reshape(self.pool2, [FLAGS.batch_size, -1])
        self.dim = self.pool2_flat.get_shape()[1].value
        # dense1
        with tf.variable_scope('dense1') as scope:
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
            self.dropout1 = tf.nn.dropout(self.dense1, self.keep_prob,
                                          name='dropout1')
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
                tf.matmul(self.dropout1, self.weights),
                self.biases,
                name='pre_activation')
            self.dense2 = tf.nn.relu(self.pre_activation, name='dense2')
            self.dropout2 = tf.nn.dropout(self.dense2, self.keep_prob,
                                          name='dropout2')
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
                tf.matmul(self.dropout2, self.weights),
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
    timer = Timer(8)
    qrunner = tf.train.QueueRunner(queue, [batch_maker.enqueue_preprocessed] * 4)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(
            Settings.path_training_summaries + '/run_' +
            str(Settings.nb_run), sess.graph)
        coord = tf.train.Coordinator()
        enqueue_threads = qrunner.create_threads(sess, coord=coord, start=True)
        for step in xrange(FLAGS.nb_steps):
            # Print training summary every 10 steps
            if coord.should_stop():
                break
            if step % 10 == 0:
                timer.start(0)
                _, loss, top1, summary = sess.run(
                    [net.train_op, net.loss, net.top1, net.merged_summary_op],
                    feed_dict={net.keep_prob: 0.5})
                timer.end(0)
                summary_writer.add_summary(summary)
                print('Step ' + str(current_step).zfill(3) +
                      '| Loss: %.3f' % loss +
                      '| Top1: %.3f' % top1)
                timer.print_performance()
                print("=======================================================")
                """
				# Quickval every 100 steps
				if step % 100 == 0:
				quickval(sess, net, batch_maker, summary_writer)
				print("===================================================")
				"""
            else:
                timer.start(0)
                sess.run(net.train_op, feed_dict={net.keep_prob: 0.5})
                timer.end(0)
            coord.request_stop()
            coord.join(enqueue_threads)


def quickval(sess, net, batch_maker, summary_writer):
    loss_store = []
    top1_store = []
    for i in range(Settings.nb_quickval_batches):
        images, labels = batch_maker.next_test_batch(FLAGS.batch_size)
        loss, top1, summary = sess.run([net.loss, net.top1, net.merged_summary_op],
                                       feed_dict={net.images: images, net.labels: labels, net.keep_prob: 1})
        loss_store.append(loss)
        top1_store.append(top1)

        summary_writer.add_summary(summary)

    # Compute mean over batches
    mean_loss = np.mean(loss_store)
    mean_top1 = np.mean(top1_store)

    # Print results
    print('Quickval: Loss: %.3f ' % mean_loss +
          '| Top1: %.3f ' % mean_top1)


class InputPipeline(object):
    def __init__(self, batch_maker):
        self.unprocessed_queue = tf.FIFOQueue(capacity=50000, dtypes=tf.float32,
                                         shapes=[[3 * 32 * 32], [1]])
        self.enqueue_unprocessed = self.unprocessed_queue.enqueue_many(batch_maker.train_images)

        image, label = self.unprocessed_queue.dequeue()
        # Preprocess one example
        self.preprocessed_example = (self.preprocess(image), label)

        # Queue holding preprocessed examples
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * FLAGS.batch_size
        self.preprocessed_queue = tf.RandomShuffleQueue(capacity=capacity,
                                                   min_after_dequeue=min_after_dequeue,
                                                   dtypes=tf.float32,
                                                   shapes=[[Settings.image_size, Settings.image_size, 3]])
        self.enqueue_preprocessed = self.preprocessed_queue.enqueue(preprocessed_example)

class BatchMaker(object):
    def __init__(self):
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.load_train_data()
        self.load_test_data()

    def input_pipeline(self):
        # Queue holding all unprocessed examples
        unprocessed_queue = tf.FIFOQueue(capacity=50000, dtypes=tf.float32,
                                         shapes=[[3 * 32 * 32], [1]])
        enqueue_unprocessed = unprocessed_queue.enqueue_many(self.train_images)

        image, label = unprocessed_queue.dequeue()
        # Preprocess one example
        preprocessed_example = (self.preprocess(image), label)

        # Queue holding preprocessed examples
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * FLAGS.batch_size
        preprocessed_queue = tf.RandomShuffleQueue(capacity=capacity,
                                                   min_after_dequeue=min_after_dequeue,
                                                   dtypes=tf.float32,
                                                   shapes=[[Settings.image_size, Settings.image_size, 3]])
        enqueue_preprocessed = preprocessed_queue.enqueue(preprocessed_example)
        return preprocessed_queue.dequeue_many(FLAGS.batch_size)

    def load_train_data(self):
        # Load training data
        maybe_download_and_extract()
        print("Loading training data. ")
        for current_batch in range(5):
            with open('./cifar10_data/cifar-10-batches-py/data_batch_' + str(current_batch + 1), 'rb') as file:
                data_dict = pickle.load(file, encoding='bytes')
                self.train_images.extend(data_dict[b'data'])
                self.train_labels.extend(data_dict[b'labels'])
        return self.train_images, self.train_labels

    def load_test_data(self):
        # Load test data
        print("Loading test data.")
        with open('./cifar10_data/cifar-10-batches-py/test_batch', 'rb') as file:
            data_dict = pickle.load(file, encoding='bytes')
            self.test_images = data_dict[b'data']
            self.test_labels = data_dict[b'labels']

    def preprocess(self, image):
        depth_major = tf.reshape(image, [3, 32, 32])
        reshaped_image = tf.transpose(depth_major, [0, 2, 3, 1])

        distorted_image = tf.random_crop(reshaped_image,
                                         [Settings.image_size, Settings.image_size, 3])

        distorted_image = tf.image.random_flip_left_right(distorted_image)

        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)

        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        normalized_image = tf.image.per_image_standardization(distorted_image)

        tf.summary.image('inputImage', self.images_reshaped)

        return normalized_image

    @staticmethod
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

            filepath, _ = urllib.request.urlretrieve(Settings.data_url, filepath,
                                                     _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(Settings.path_data,
                                          'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(Settings.path_data)


class Timer(object):
    def __init__(self, nb_tasks):
        self.timetable = [[] for _ in range(nb_tasks)]
        self.start_times = np.zeros(nb_tasks)

    def start(self, task):
        self.start_times[task] = time.time()

    def end(self, task):
        self.timetable[task].append(time.time() - self.start_times[task])

    def print_performance(self):
        print('Avg. training time: ' + str(np.mean(self.timetable[0])))
        print('Avg. time for getting next_train_batch: ' + str(np.mean(self.timetable[1])))
        print('Avg. load_train_data from drive time: ' + str(np.mean(self.timetable[2])))
        print('Avg. shuffling time: ' + str(np.mean(self.timetable[3])))
        print('Avg. distortion time: ' + str(np.mean(self.timetable[4])))
        """print('Avg. bright time: ' + str(np.mean(self.timetable[5])))
		print('Avg. contrast time: ' + str(np.mean(self.timetable[6])))
		print('Avg. norm time: ' + str(np.mean(self.timetable[7])))
		"""


# ##############################################################################
# ################################### MAIN #####################################
# ##############################################################################

if __name__ == '__main__':
    maybe_download_and_extract()
    bm = Batch_maker()
    net = Network(bm)
    train(net, bm)
