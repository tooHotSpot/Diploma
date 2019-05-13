"""
Contains models definitions for base and modified versions of the Siamese Neural Network for One-shot Image Recognition.
"""

import tensorflow as tf
from tensorflow.keras.initializers import he_normal, he_uniform
from tensorflow import glorot_uniform_initializer, glorot_normal_initializer

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
WEIGHT_DECAY = 0.001


class SiameseNetwork:

    def __init__(self, debug, verbose):
        self.debug = debug
        self.verbose = verbose

    def layer_print(self, name, w, b, output):
        if self.verbose:
            print('********************** ', name, ' **********************')
            print('-->Weights: ', w)
            print('-->Bias: ', b)
            print('-->Layer ', output)

    def convolution_block(self, x, name, kernel, channels_out, is_training, stride=1, max_pool=False, padding='VALID'):
        with tf.variable_scope(name):
            weights = tf.get_variable(name='W', shape=[kernel, kernel, x.shape[-1], channels_out], dtype=tf.float32, initializer=he_normal())
            b = tf.get_variable(name='b', shape=[channels_out], dtype=tf.float32, initializer=he_uniform())
            x = tf.nn.conv2d(input=x, filter=weights, strides=[1, stride, stride, 1], padding=padding)
            x = tf.layers.batch_normalization(inputs=x, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                              center=True, scale=True, training=is_training, fused=True)
            x = tf.nn.relu(features=x + b)
            if max_pool:
                x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
            self.layer_print(name=name, w=weights, b=b, output=x)
            return x

    def fully_connected_block(self, x, name, units_out, is_training):
        with tf.variable_scope(name):
            weights = tf.get_variable(name="W", shape=[x.shape[1], units_out], dtype=tf.float32, initializer=glorot_normal_initializer())
            b = tf.get_variable(name="b", shape=[units_out], dtype=tf.float32, initializer=glorot_uniform_initializer())
            x = tf.matmul(x, weights) + b
            x = tf.layers.batch_normalization(inputs=x, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                              center=True, scale=True, training=is_training, fused=True)
            x = tf.sigmoid(x)
            self.layer_print(name=name, w=weights, b=b, output=x)
            return x

    def twin_base(self, x, is_training):
        channels_units_arr = [4, 4, 4, 4, 16] if self.debug else [64, 128, 128, 256, 4096]

        # Input: [?, 105, 105, 1], output: [?, 48, 48, 64]
        x = self.convolution_block(x, name='CONV_0', kernel=10, channels_out=channels_units_arr[0], is_training=is_training, max_pool=True)
        # Input: [?, 48, 48, 64], output: [?, 21, 21, 128]
        x = self.convolution_block(x, name='CONV_1', kernel=7, channels_out=channels_units_arr[1], is_training=is_training, max_pool=True)
        # Input: [?, 21, 21, 128], output: [?, 9, 9, 128]
        x = self.convolution_block(x, name='CONV_2', kernel=4, channels_out=channels_units_arr[2], is_training=is_training, max_pool=True)
        # Input: [?, 9, 9, 128], output: [?, 6, 6, 256],
        x = self.convolution_block(x, name='CONV_3', kernel=4, channels_out=channels_units_arr[3], is_training=is_training)
        # Flatten into single vector, direct sizes diminishes mistake probability,
        x = tf.reshape(x, [-1, 6 * 6 * 256], name='CONV_3-Flattened')
        self.layer_print(name='CONV_3-Flattened', w=None, b=None, output=x)
        # region FC, input: [?, 256 * 6 * 6] = [?, 9216] , output: [?, 4096]
        x = self.fully_connected_block(x, name='FC', units_out=channels_units_arr[4], is_training=is_training)
        return x

    def twin_SE(self, x, is_training):
        return x

    def model(self, x1, x2, is_training, implementation='base'):
        assert implementation in ('base', 'SE'), 'Invalid implementation mode'
        twin = self.twin_base
        if implementation == 'SE':
            twin = self.twin_SE

        with tf.variable_scope("Twin") as scope:
            fc1 = twin(x1, is_training)
        with tf.variable_scope(scope, reuse=True):
            fc2 = twin(x2, is_training)

        alpha = tf.get_variable('FCAlpha', shape=[fc1.shape[-1], 1], dtype=tf.float32, initializer=glorot_normal_initializer())
        fc_main = tf.matmul(tf.abs(tf.subtract(x=fc1, y=fc2, name='Subtract'), name='AbsoluteValue'), b=alpha, name='FC_Final')
        self.layer_print(name=fc_main.name, w=alpha, b=None, output=fc_main)
        return fc_main
