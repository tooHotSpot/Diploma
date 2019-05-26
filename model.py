"""
Contains models definitions for base and modified versions of the Siamese Neural Network for One-shot Image Recognition.
"""

import tensorflow as tf
from tensorflow.keras.initializers import he_normal, he_uniform
from tensorflow import glorot_uniform_initializer, glorot_normal_initializer

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


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

    def convolution_block(self, x, name, kernel, channels_out, is_training, stride=1, max_pool=False, activate_with_relu=True, padding='VALID'):
        with tf.variable_scope(name):
            weights = tf.get_variable(name='W', shape=[kernel, kernel, x.shape[-1], channels_out], dtype=tf.float32, initializer=he_normal())
            b = tf.get_variable(name='b', shape=[channels_out], dtype=tf.float32, initializer=he_uniform())
            x = tf.nn.conv2d(input=x, filter=weights, strides=[1, stride, stride, 1], padding=padding)
            x = tf.layers.batch_normalization(inputs=x, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                              center=True, scale=True, training=is_training, fused=True)
            x += b
            if activate_with_relu:
                x = tf.nn.relu(x + b)
            if max_pool:
                x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
            self.layer_print(name=name, w=weights, b=b, output=x)
            return x

    def fully_connected_block(self, x, name, units_out, is_training, activation='sigmoid'):
        assert activation in ('sigmoid', 'relu')
        with tf.variable_scope(name):
            weights = tf.get_variable(name="W", shape=[x.shape[1], units_out], dtype=tf.float32, initializer=glorot_normal_initializer())
            b = tf.get_variable(name="b", shape=[units_out], dtype=tf.float32, initializer=glorot_uniform_initializer())
            x = tf.matmul(x, weights) + b
            x = tf.layers.batch_normalization(inputs=x, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                              center=True, scale=True, training=is_training, fused=True)
            x = tf.nn.relu(x) if activation == 'relu' else tf.sigmoid(x)
            self.layer_print(name=name, w=weights, b=b, output=x)
            return x

    def squeeze_excitation_block(self, x, name, is_training):
        with tf.variable_scope(name):
            channels = x.shape[-1]
            x = tf.nn.avg_pool(x, ksize=[1, x.shape[1], x.shape[2], 1], strides=[1, 1, 1, 1], padding="VALID", name='GPA')
            x = tf.reshape(x, [-1, x.shape[-1]], name='GPAFlattened')
            x = self.fully_connected_block(x, name='FC1', units_out=channels // 16, is_training=is_training, activation='relu')
            x = self.fully_connected_block(x, name='FC2', units_out=channels, is_training=is_training, activation='sigmoid')
            x = tf.reshape(x, shape=[-1, 1, 1, channels])
            return x

    def twin_base(self, x, is_training):
        channels_units_arr = [4, 4, 4, 4, 16] if self.debug else [64, 128, 128, 256, 4096]

        # Input: [?, 105, 105, 1], output: [?, 48, 48, 64]
        # Params: [10 * 10 * 1] * [64 filters] + [64 bias] = 6464
        x = self.convolution_block(x, name='CONV_0', kernel=10, channels_out=channels_units_arr[0], is_training=is_training, max_pool=True)
        # Input: [?, 48, 48, 64], output: [?, 21, 21, 128]
        # Params: [7 * 7 * 64] * [128 filters] + [128 bias] = 401.536
        x = self.convolution_block(x, name='CONV_1', kernel=7, channels_out=channels_units_arr[1], is_training=is_training, max_pool=True)
        # Input: [?, 21, 21, 128], output: [?, 9, 9, 128]
        # Params: [4 * 4 * 128] * [128 filters] + [128 bias] = 262.272
        x = self.convolution_block(x, name='CONV_2', kernel=4, channels_out=channels_units_arr[2], is_training=is_training, max_pool=True)
        # Input: [?, 9, 9, 128], output: [?, 6, 6, 256]
        # Params: [4 * 4 * 128] * [256 filters] + [128 bias] = 524.288 + 128 = 524.416
        x = self.convolution_block(x, name='CONV_3', kernel=4, channels_out=channels_units_arr[3], is_training=is_training)
        # Flatten into single vector each image tensor
        _, w, h, c = x.shape
        # Better -1 because instead [b, w * h * c] is converted to list
        x = tf.reshape(x, [-1, w * h * c], name='CONV_3-Flattened')
        self.layer_print(name='CONV_3-Flattened', w=None, b=None, output=x)
        # region FC, input: [?, 256 * 6 * 6] = [?, 9216] , output: [?, 4096]
        # Params: [256 * 6 * 6] * [4096] = 37.748.736
        x = self.fully_connected_block(x, name='FC', units_out=channels_units_arr[4], is_training=is_training)
        # Total amount = 6.464 + 401.536 + 262.272 + 524.416 + 37748.736 = 38.943.424 ~ 40 million
        return x

    def twin_SE(self, x, is_training):
        x = tf.pad(x, paddings=tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]]), constant_values=1)
        # Input: [?, 105, 105, 1], output: [?, 56, 56, 64]
        # [3 * 3 * 1] * [64 filters] + [64 bias] = 10 * 64 = 640
        x = self.convolution_block(x, name='CONV0-Initial', kernel=3, channels_out=64, is_training=is_training, stride=2, padding='VALID')
        # First residual block with 64 channels
        group_channels = 16 if self.debug else 64
        # Params: [3 * 3 * 64] * [64 filters] + 64 = 36928
        y = self.convolution_block(x, name='CONV1_1_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # Params: [3 * 3 * 64] * [64 filters] + 64 = 36928
        y = self.convolution_block(y, name='CONV1_1_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # Params: 2 * [64 / 16] * [64] = 2 * 64 * 4 = 512
        z = self.squeeze_excitation_block(y, name="SE1_1", is_training=is_training)
        y = tf.multiply(y, z, name='Scaled')
        x = tf.nn.relu(tf.add(x, y))

         # Params:
        # y = self.convolution_block(x, name='CONV1_2_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV1_2_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE1_2", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV1_3_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV1_3_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE1_3", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # Input: [?, 56, 56, 64], output: [?, 28, 28, 128]
        group_channels = 16 if self.debug else 128
        # Params: [3 * 3 * 64] * [128 filters] + 128 = 73728 + 128 = 73856
        y = self.convolution_block(x, name='CONV2_1_1', kernel=3, channels_out=group_channels, is_training=is_training, stride=2, padding='SAME')
        # Params: [3 * 3 * 128] * [128 filters] + 128 = 147456 + 128 = 147584
        y = self.convolution_block(y, name='CONV2_1_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # Params: 2 * [128 / 16] * [128] = 128 * 128 * 2 / 16 = 128 * 16 = 2048
        z = self.squeeze_excitation_block(y, name="SE2_1", is_training=is_training)
        y = tf.multiply(y, z, name='Scaled')
        # Reduction needed
        # [1 * 1 * 128] * [128 filters] = 16384
        x = self.convolution_block(x, name='CONV1_Reduction', kernel=1, channels_out=group_channels, is_training=is_training, stride=2, activate_with_relu=False)
        x = tf.nn.relu(tf.add(x, y), name='AddWithReduction')

        # y = self.convolution_block(x, name='CONV2_2_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV2_2_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE2_2", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV2_3_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV2_3_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE2_3", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV2_4_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV2_4_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE2_4", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # Input: [?, 28, 28, 128], output: [?, 14, 14, 256]
        group_channels = 16 if self.debug else 256
        # [3 * 3 * 128]
        y = self.convolution_block(x, name='CONV3_1_1', kernel=3, channels_out=group_channels, is_training=is_training, stride=2, padding='SAME')
        y = self.convolution_block(y, name='CONV3_1_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        z = self.squeeze_excitation_block(y, name="SE3_1", is_training=is_training)
        y = tf.multiply(y, z, name='Scaled')
        # Reduction needed
        x = self.convolution_block(x, name='CONV2_Reduction', kernel=1, channels_out=group_channels, is_training=is_training, stride=2, activate_with_relu=False)
        x = tf.nn.relu(tf.add(x, y), name='AddWithReduction')

        # y = self.convolution_block(x, name='CONV3_2_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV3_2_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE3_2", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV3_3_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV3_3_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE3_3", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV3_4_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV3_4_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE3_4", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV3_5_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV3_5_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE3_5", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV3_6_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV3_6_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE3_6", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # Input: [?, 14, 14, 256], output: [?, 7, 7, 256],
        group_channels = 16 if self.debug else 512
        y = self.convolution_block(x, name='CONV4_1_1', kernel=3, channels_out=group_channels, is_training=is_training, stride=2, padding='SAME')
        y = self.convolution_block(y, name='CONV4_1_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        z = self.squeeze_excitation_block(y, name="SE4_1", is_training=is_training)
        y = tf.multiply(y, z, name='Scaled')
        # Reduction needed
        x = self.convolution_block(x, name='CONV3_Reduction', kernel=1, channels_out=group_channels, is_training=is_training, stride=2, activate_with_relu=False)
        x = tf.nn.relu(tf.add(x, y), name='AddWithReduction')

        # y = self.convolution_block(x, name='CONV4_2_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV4_2_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE4_2", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        # y = self.convolution_block(x, name='CONV4_3_1', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME')
        # y = self.convolution_block(y, name='CONV4_3_2', kernel=3, channels_out=group_channels, is_training=is_training, padding='SAME', activate_with_relu=False)
        # z = self.squeeze_excitation_block(y, name="SE4_3", is_training=is_training)
        # y = tf.multiply(y, z, name='Scaled')
        # x = tf.nn.relu(tf.add(x, y))

        _, w, h, c = x.shape
        # Better -1 because instead [b, w * h * c] is converted to list
        x = tf.reshape(x, [-1, w * h * c], name='CONV4-Flattened')
        self.layer_print(name='CONV4-Flattened', w=None, b=None, output=x)
        # region FC, input: [?, 7 * 7 * 512] = [?, 25088] , output: [?, 1000]
        x = self.fully_connected_block(x, name='FC', units_out=1024, is_training=is_training)
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
