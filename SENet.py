"""
This implementation uses validation on same data as training, 10% of training data is used.
"""

import os
import cv2
import data
import numpy as np
from tqdm import tqdm
from datetime import datetime

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.client import device_lib

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

w = h = 105

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
sess = tf.Session(config=config)


class OmniglotDataset(Dataset):
    def __init__(self, chosen_pairs_np_array, chosen_pairs_answers_np_array, background_images_dict, pairs_amount, augment, transform=None):
        assert pairs_amount == len(chosen_pairs_answers_np_array) == len(chosen_pairs_np_array), 'Incorrect pairs sizes'
        self.pairs = chosen_pairs_np_array
        self.pairs_answers = chosen_pairs_answers_np_array
        self.pairs_amount = pairs_amount
        self.background_images_dict = background_images_dict
        self.transform = transform
        self.augment = augment
        self.total_pairs_amount = pairs_amount
        if augment:
            self.total_pairs_amount = 9 * pairs_amount

    def __len__(self):
        return self.total_pairs_amount

    def __getitem__(self, item):
        real_item = item % self.pairs_amount

        entry1, entry2 = self.pairs[real_item]
        alphabet, character1, image_name1 = entry1
        image1 = self.background_images_dict[alphabet][character1][image_name1]
        alphabet, character2, image_name2 = entry2
        image2 = self.background_images_dict[alphabet][character2][image_name2]

        # One time real image needed, others - augmented if it is enabled
        if item > self.pairs_amount:
            image1 = data.augment_int_image(image1, borderValue=255)
            image2 = data.augment_int_image(image2, borderValue=255)

        pair = (np.array([image1, image2], dtype=np.float32) - 127.5) / 127.5
        sample = {
            'pair': pair.reshape(2, 105, 105, 1),
            'pairType': self.pairs_answers[real_item]
        }
        return sample


class SiameseNetwork:
    DEBUG = True
    ROOT = None
    models_folder = None
    logs_path = None
    AUGMENT = False
    RESTORE = False
    include_residual = False

    def __init__(self, DEBUG, AUGMENT, back_pairs_amount, restore_models_folder=None, ROOT='/ssd480/amal/Siamese'):
        self.DEBUG = DEBUG
        self.AUGMENT = AUGMENT
        self.ROOT = ROOT
        self.back_pairs_amount = back_pairs_amount

        if not os.path.exists(self.ROOT):
            print(self.ROOT, ' path not exists as root path')
            self.ROOT = ''

        if restore_models_folder is not None:
            self.RESTORE = True
            print('Trying to restore from the ', restore_models_folder)
            restore_models_folder = os.path.join(self.ROOT, 'models', restore_models_folder)
            assert os.path.exists(restore_models_folder) and len(os.listdir(restore_models_folder)) != 0, \
                print(restore_models_folder, ' does not exist or empty')
            self.models_folder = restore_models_folder
        else:
            b = datetime.now()
            tmp = '{:02}-{:02} {:02}-{:02}-{:02} {}{}'.format(b.month, b.day, b.hour, b.minute, b.second, back_pairs_amount // 1000, ('Kx9' if self.AUGMENT else 'K'))
            self.models_folder = os.path.join(self.ROOT, 'models', tmp)
            if not os.path.exists(self.models_folder):
                os.makedirs(self.models_folder)

        print('Models folder: ', self.models_folder)
        self.logs_path = os.path.join(self.models_folder, 'logs')

        print('Listing available devices')
        local_device_protos = device_lib.list_local_devices()
        # print([x.name for x in local_device_protos if x.device_type == 'GPU'])
        print([x.name for x in local_device_protos])

    def layer_print(self, name, inputs, W, b, output):
        return
        if type(name) == list:
            print(name, end=': ')
        for attr in inputs, W, output:
            try:
                print(attr.name, attr.shape, end=' - ')
            except AttributeError:
                print(attr, end=' ')
        print(name, inputs.name, inputs.shape, W.name, W.shape, b.name, b.shape, output.name, sep=' - ')
        print('********************** ', name, ' **********************')
        # print('-->Inputs: ', inputs, '-->Weights: ', W, '-->Bias: ', b, '-->Layer ', output)

    def twin(self, x, is_training):
        def SEConvolutionalBlock(name, InitialX, r=16):
            with tf.variable_scope(name+'SE'):
                channels = InitialX.shape[-1]
                # Squeeze block
                x = tf.nn.avg_pool(InitialX, ksize=[1, InitialX.shape[1], InitialX.shape[2], 1], strides=[1, 1, 1, 1], padding="VALID", name="GlobalAveragePooling")
                x = tf.reshape(x, [-1, channels], name="GPAFlattened")
                WSqueeze = tf.get_variable("W1", [x.shape[1], channels // r], tf.float32, tf.glorot_normal_initializer())
                x = tf.matmul(x, WSqueeze, name="FC1")
                # biasSqueeze = tf.get_variable("biasSqueeze1", [channels], tf.float32, tf.glorot_uniform_initializer())
                x = tf.nn.relu(x)
                x = tf.layers.batch_normalization(inputs=x, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                                  scale=True, training=is_training, fused=True)
                WSqueeze = tf.get_variable("W2", [channels // r, channels], tf.float32, tf.glorot_normal_initializer())
                # biasSqueeze = tf.get_variable("biasSqueeze2", [channels], tf.float32, tf.glorot_uniform_initializer())
                x = tf.matmul(x, WSqueeze, name="FC2")
                x = tf.layers.batch_normalization(inputs=x, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                                  scale=True, training=is_training, fused=True)
                x = tf.nn.sigmoid(x)
                # Excitation
                Excitation = tf.reshape(x, shape=[-1, 1, 1, channels])
                Scaled = tf.multiply(Excitation, InitialX, name="Scaled")
                return Scaled

        def CONVBlock(name, conv_in, kernel, channels_out, max_pool=False, padding="VALID", residual=False):
            with tf.variable_scope(name):
                W = tf.get_variable('W', [kernel, kernel, x.shape[-1], channels_out], tf.float32, tf.keras.initializers.he_normal())
                b = tf.get_variable('b', [channels_out], tf.float32, tf.keras.initializers.he_uniform())
                if not residual:
                    conv_in = tf.nn.conv2d(conv_in, W, strides=[1, 1, 1, 1], padding=padding)
                conv_in = tf.layers.batch_normalization(inputs=conv_in, axis=-1, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                        center=True, scale=True, training=is_training, fused=True)
                conv_in = tf.nn.relu(conv_in + b)
                # if residual:
                #     conv_in = tf.nn.conv2d(conv_in, W, strides=[1, 1, 1, 1], padding=padding)
                if max_pool:
                    conv_in = tf.nn.max_pool(conv_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                self.layer_print(name, x, W, b, conv_in)
                return conv_in

        # Input: [?, 105, 105, 1], output: [?, 48, 48, 64]
        group_channels = 4 if self.DEBUG else 64
        x = CONVBlock("CONV1", x, 10, group_channels, max_pool=True)
        y = CONVBlock("CONV1_1_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV1_1_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV1_1_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV1_2_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV1_2_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV1_2_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV1_2_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV1_3_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV1_3_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV1_3_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV1_3_3", y)
        x = tf.add(x, scaled_y)

        # Input: [?, 48, 48, 64], output: [?, 21, 21, 128]
        group_channels = 4 if self.DEBUG else 64
        x = CONVBlock("CONV2_1_1", x, 7, group_channels, max_pool=True)
        y = CONVBlock("CONV2_1_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV2_1_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV2_1_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV2_2_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV2_2_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV2_2_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV2_2_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV2_3_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV2_3_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV2_3_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV2_3_3", y)
        x = tf.add(x, scaled_y)

        # Input: [?, 21, 21, 128], output: [?, 9, 9, 128]
        group_channels = 4 if self.DEBUG else 64
        x = CONVBlock("CONV3_1_1", x, 4, group_channels, max_pool=True)
        y = CONVBlock("CONV3_1_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV3_1_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV3_1_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV3_2_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV3_2_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV3_2_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV3_2_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV3_3_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV3_3_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV3_3_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV3_3_3", y)
        x = tf.add(x, scaled_y)

        # Input: [?, 9, 9, 128], output: [?, 6, 6, 256],
        group_channels = 4 if self.DEBUG else 128
        x = CONVBlock("CONV4_1_1", x, 4, group_channels, max_pool=False)
        y = CONVBlock("CONV4_1_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV4_1_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV4_1_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV4_2_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV4_2_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV4_2_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV4_2_3", y)
        x = tf.add(x, scaled_y)

        x = CONVBlock("CONV4_3_1", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV4_3_2", x, 3, group_channels, padding="SAME")
        y = CONVBlock("CONV4_3_3", y, 3, group_channels, padding="SAME")
        scaled_y = SEConvolutionalBlock("CONV4_3_3", y)
        x = tf.add(x, scaled_y)

        CONV4_Flattened = tf.reshape(x, [-1, x.shape[-1] * 6 * 6])
        name = 'Fully-connected'
        with tf.variable_scope(name):
            # region FC, input: [?, 256 * 6 * 6] = [?, 9216] , output: [?, 4096]
            FCSHAPE = 4096
            if self.DEBUG:
                FCSHAPE = 64
            W = tf.get_variable("FC_W", [CONV4_Flattened.shape[1], FCSHAPE], tf.float32, tf.glorot_normal_initializer())
            b = tf.get_variable("FC_b", [FCSHAPE], tf.float32, tf.glorot_uniform_initializer())
            FC = tf.matmul(CONV4_Flattened, W)
            FC = tf.layers.batch_normalization(inputs=FC, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                               center=True, scale=True, training=is_training, fused=True)
            # print('Batch-norm exists here: ', FC)
            FC = tf.sigmoid(FC + b)
            self.layer_print(name, CONV4_Flattened, W, b, FC)

        return FC

    def model(self, x1, x2, is_training):
        with tf.variable_scope("Twin") as scope:
            FC1 = self.twin(x1, is_training)
        with tf.variable_scope(scope, reuse=True):
            FC2 = self.twin(x2, is_training)

        name = 'FC_Final'
        FCAlpha = tf.get_variable('FCAlpha', [FC1.shape[1], 1], tf.float32, tf.glorot_normal_initializer())
        FC = tf.matmul(tf.abs(tf.subtract(FC1, FC2)), FCAlpha)
        # self.layer_print(name, [FC1, FC2], FCAlpha, None, FC)
        return FC

    def check_verification(self, sess_elements, subset_x, subset_y, subset_num_batches, subset_len, batch_size):
        print('')
        sess, x1, x2, y, tf_is_training, predictions, accuracy, loss = sess_elements
        subset_accuracy = 0
        subset_loss = 0
        for i in range(subset_num_batches):
            left, right = i * batch_size, min((i + 1) * batch_size, subset_len)
            feed_dict = {
                x1: subset_x[left:right, 0, :, :],
                x2: subset_x[left:right, 1, :, :],
                y: subset_y[left:right],
                tf_is_training: False
            }
            acc, batch_loss = sess.run([accuracy, loss], feed_dict)
            subset_loss += batch_loss
            subset_accuracy += acc * (right - left)
        return np.around(subset_accuracy / subset_len, decimals=4), subset_loss

    def new_check_one_shot_learning(self, sess_elements, trials, trials_len, trials_num_batches, batch_size):
        sess, x1, x2, y, tf_is_training, predictions, accuracy, loss = sess_elements
        confusion_matrix = np.zeros((20, 20))
        pred = []
        for i in range(trials_num_batches):
            left, right = i * batch_size, min((i + 1) * batch_size, trials_len)
            feed_dict = {
                x1: trials[left:right, 0, :, :],
                x2: trials[left:right, 1, :, :],
                tf_is_training: False
            }
            pred.extend(sess.run(tf.sigmoid(sess.run(predictions, feed_dict))))

        pred = np.array(pred).reshape(-1, 20)
        pred = np.argmax(pred, axis=-1).reshape(-1).astype(int)
        for i in range(400):
            confusion_matrix[i % 20, pred[i]] += 1

        # np.sum(confusion_matrix) will be equal to 400 in case of 400 one-shot learning trials
        total_acc = np.sum(np.diagonal(confusion_matrix)) / 400
        return np.around(total_acc, decimals=4)

    def fit(self, starter_learning_rate=0.01, num_epochs=100, batch_size=128, early_stopping=True, optimization_algorithm='Adagrad', WEIGHT_DECAY=0.00005):

        x1 = tf.placeholder(tf.float32, shape=[None, w, h, 1], name='X1')
        x2 = tf.placeholder(tf.float32, shape=[None, w, h, 1], name='X2')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
        tf_is_training = tf.placeholder(tf.bool)

        predictions = self.model(x1, x2, tf_is_training)
        predictions_labels = tf.round(tf.sigmoid(predictions))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_labels, y), tf.float32))

        back_pairs_amount = self.back_pairs_amount
        print('Background pairs amount: {}, augmentation: {}'.format(back_pairs_amount, self.AUGMENT))
        print('Num epochs: {}, early stopping: {}, weight decay value: {}'.format(num_epochs, early_stopping, WEIGHT_DECAY))
        print('Starter learning rate {}, batch_size {}, optimizer {}'.format(starter_learning_rate, batch_size, optimization_algorithm))
        back_images_dict = data.load_image_dict(self.ROOT, 'images/images_background', 'images_background')

        # region Data
        if self.RESTORE:
            train_pairs_images, train_pairs_types = data.unpickle_it(os.path.join(self.models_folder, 'train_pairs.pkl'))
            val_trials = data.unpickle_it(os.path.join(self.models_folder, 'val_trials.pkl'))
            params = data.unpickle_it(os.path.join(self.models_folder, 'params.json'))
        else:
            train_pairs_images, train_pairs_types = data.generate_image_pairs(images_dict=back_images_dict, pairs_amount=back_pairs_amount, ceil=12, path_only=True)
            data.pickle_it([train_pairs_images, train_pairs_types], os.path.join(self.models_folder, 'train_pairs.pkl'))
            params = {'starter_learning_rate': starter_learning_rate,
                      'learning_rate': starter_learning_rate,
                      'batch_size': batch_size,
                      'optimization_algorithm': optimization_algorithm,
                      'early_stopping': early_stopping,
                      'WEIGHT_DECAY': WEIGHT_DECAY,
                      'last_epoch': -1,
                      'last_subepoch': 0,
                      'num_epochs': num_epochs,
                      'augment': self.AUGMENT,
                      'back_pairs_amount': back_pairs_amount,
                      'top_val_acc': 0,
                      'eval_acc': 0,
                      'hour': datetime.now().hour,
                      'minute': datetime.now().minute,
                      'early_stopping_limit': (90 if self.AUGMENT else 20),
                      'residuals': self.include_residual
                      }
            data.pickle_it(params, os.path.join(self.models_folder, 'params.json'))
            params = data.unpickle_it(os.path.join(self.models_folder, 'params.json'))
            print(params)

            validation_images_dict = data.load_image_dict(self.ROOT, 'images/images_evaluation', 'images_evaluation', clear=True)
            val_trials = data.generate_one_shot_trials(images_dict=validation_images_dict)
            data.pickle_it(val_trials, os.path.join(self.models_folder, 'val_trials.pkl'))
            del validation_images_dict

        omniglot_dataset = OmniglotDataset(train_pairs_images, train_pairs_types, back_images_dict, back_pairs_amount, self.AUGMENT)
        print('Omniglot dataset length: (must be equal to image pairs or x9 if augmentation provided)', len(omniglot_dataset))

        num_train_batches = (len(omniglot_dataset) + batch_size - 1) // batch_size
        num_subepoch_batches = (back_pairs_amount + batch_size - 1) // batch_size
        print('Amount of training batches {}, amount subepoch batches: {}'.format(num_train_batches, num_subepoch_batches))
        del back_images_dict, train_pairs_images, train_pairs_types

        val_trials_len = len(val_trials)
        val_trials = (val_trials.astype(np.float32) - 127.5) / 127.5

        eval_trials = data.get_eval_trials(self.ROOT)
        eval_trials_len = len(eval_trials)
        eval_trials = (eval_trials.astype(np.float32) - 127.5) / 127.5
        num_val_trials_batches, num_eval_trials_batches = (np.array([val_trials_len, eval_trials_len]) + batch_size - 1) // batch_size
        print('Amount of validation [{}] & test [{}] one-shot classification batches'.format(num_val_trials_batches, num_eval_trials_batches))
        # endregion
        # region TFUtils

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predictions))
        reg_loss = WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        total_loss = loss + reg_loss

        tf.summary.scalar('Accuracy/Train', accuracy)
        tf.summary.scalar('Train/Loss', loss)
        tf.summary.scalar('Train/RegularizationLoss', reg_loss)
        tf.summary.scalar('Train/TotalLoss', total_loss)

        saver = tf.train.Saver()
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, num_train_batches, 0.5)

        optimizer = None
        if optimization_algorithm == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optimization_algorithm == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5)
        elif optimization_algorithm == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)
        merged_summary_op = tf.summary.merge_all()

        sess_elements = [sess, x1, x2, y, tf_is_training, predictions, accuracy, loss]

        sess.run(tf.global_variables_initializer())
        if self.DEBUG:
            return
        print('Started the session')

        if self.RESTORE:
            saver.restore(sess, self.models_folder + '/model.ckpt')

        summary_writer = tf.summary.FileWriter(self.logs_path)
        # endregion

        num_subepochs = 1
        if self.AUGMENT:
            num_subepochs = 9

        top_val_acc = 0
        top_val_acc_step = -1
        val_acc = 0
        eval_acc = 0
        step = 0

        for epoch in range(params['last_epoch'], num_epochs):
            # One extra subepoch for code check
            print('Epoch ', epoch)
            subepoch = 0
            subepoch_loss = subepoch_acc = 0
            dataloader = DataLoader(omniglot_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
            if epoch == -1:
                continue
                num_train_batches = 0
                subepoch = num_subepochs - 1
            else:
                num_train_batches = len(dataloader)

            for i, batch in enumerate(dataloader, 0):
                batch_images = batch['pair'].numpy()
                batch_types = batch['pairType'].numpy().astype(np.float32)
                feed_dict = {
                    x1: batch_images[:, 0, :, :],
                    x2: batch_images[:, 1, :, :],
                    y: batch_types,
                    tf_is_training: True
                }
                _, batch_cost, acc, batch_predictions_labels, summary = sess.run([train_op, loss, accuracy, predictions_labels, merged_summary_op], feed_dict)
                summary_writer.add_summary(summary, num_train_batches * epoch + i)
                subepoch_acc += acc / num_subepoch_batches
                subepoch_loss += batch_cost / num_subepoch_batches

                if i == num_train_batches - 1 or (i % num_subepoch_batches == 0 and i > 0) or i == num_train_batches - 1 or epoch == -1:
                    print('Validating')
                    time_now = datetime.now()
                    step = epoch * num_subepochs + subepoch
                    print('Time: {:02}:{:02}, subepoch {}.{}(step {}), loss:{:5.3f}, acc.:{:4.3f};'.format(time_now.hour, time_now.minute, epoch, subepoch,
                                                                                                           step, subepoch_loss, subepoch_acc), end='')
                    if i == num_train_batches - 1:
                        val_acc = self.new_check_one_shot_learning(sess_elements, val_trials, val_trials_len, num_val_trials_batches, batch_size)
                    print(' validation acc.: {:4.3f}/{:4.3f} (best on step {:2})'.format(val_acc, top_val_acc, top_val_acc_step), end='')

                    if val_acc > top_val_acc + 0.001:
                        top_val_acc, top_val_acc_step = val_acc, step
                        # eval_acc = self.new_check_one_shot_learning(sess_elements, eval_trials, eval_trials_len, num_eval_trials_batches, batch_size)
                        print(' Highest! EvalAcc.: {:4.3f}'.format(eval_acc))
                        saver.save(sess, os.path.join(self.models_folder, 'model.ckpt'))
                        params['learning_rate'] = float(sess.run(learning_rate))
                        params['last_epoch'] = epoch
                        params['last_subepoch'] = subepoch
                        params['eval_acc'] = eval_acc
                        params['top_val_acc'] = top_val_acc
                        params['hour'] = datetime.now().hour
                        params['minute'] = datetime.now().minute
                        data.pickle_it(params, os.path.join(self.models_folder, 'params.json'))
                        print('Repickled params.json at {:02}:{:02}'.format(params['hour'], params['minute']))
                    else:
                        print('')

                    summary = tf.Summary()
                    summary.value.add(tag='OneShotLearning/Validation', simple_value=val_acc)
                    summary.value.add(tag='OneShotLearning/Evaluation', simple_value=eval_acc)
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()

                    subepoch_loss = subepoch_acc = 0
                    subepoch += 1

                    if early_stopping and step - top_val_acc_step > params['early_stopping_limit']:
                        print('Finished to train because of early stopping.')
                        return

                if epoch == -1:
                    print('Code checked.')
                    break
        print('Finished to fit.')


if __name__ == '__main__':
    mySiameseNetwork = SiameseNetwork(DEBUG=False, back_pairs_amount=90000, AUGMENT=True, restore_models_folder='05-10 09-28-48 90Kx9')
    # mySiameseNetwork.include_residual = True
    mySiameseNetwork.fit(starter_learning_rate=0.01, num_epochs=100, batch_size=128, optimization_algorithm='Adagrad', WEIGHT_DECAY=0.00005)
    del mySiameseNetwork
