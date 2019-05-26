"""
This implementation uses one-shot learning validation.
"""

import os
import data
from tqdm import tqdm
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.client import device_lib

from torch.utils.data import DataLoader

from model import SiameseNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
ops.reset_default_graph()
sess = tf.Session(config=config)


class Network(SiameseNetwork):
    root_folder = None
    models_folder = None
    logs_path = None
    augment = False
    restore = False

    def __init__(self, debug, verbose, back_pairs_amount, augment, restore_models_folder=None, root_folder='/ssd480/amal/Siamese',
                 starter_learning_rate=0.01, batch_size=128, early_stopping=True, optimization_algorithm='Adagrad', weight_decay=0.00005,
                 lr_decay=0.05, implementation='base', ceil=12):
        super(Network, self).__init__(debug=debug, verbose=verbose)

        self.augment = augment
        self.root_folder = root_folder
        self.ceil = ceil
        self.back_pairs_amount = back_pairs_amount
        self.implementation = implementation
        folder = ('se_' if implementation == 'SE' else '') + 'models'

        if not os.path.exists(self.root_folder):
            print(self.root_folder, ' path not exists as root path')
            self.root_folder = ''

        if restore_models_folder is not None:
            self.restore = True
            print('Trying to restore from the ', restore_models_folder, '...')
            restore_models_folder = os.path.join(self.root_folder, folder, restore_models_folder)
            assert os.path.exists(restore_models_folder) and len(os.listdir(restore_models_folder)) != 0, \
                print(restore_models_folder, ' does not exist or empty')
            self.models_folder = restore_models_folder
            # Add params reading on restore
        else:
            b = datetime.now()
            tmp = '{:02}-{:02} {:02}-{:02}-{:02} {}{}'.format(b.month, b.day, b.hour, b.minute, b.second, back_pairs_amount // 1000, ('Kx9' if self.augment else 'K'))
            tmp += '_{}_{}_B{}_{}_{:.2f}_{}'.format(starter_learning_rate, optimization_algorithm, batch_size, weight_decay, lr_decay, 'ES' if early_stopping else '')
            if implementation == 'SE':
                tmp += '_SE'
            if ceil != 12:
                tmp += '_C{}'.format(ceil)

            self.starter_learning_rate = starter_learning_rate
            self.back_pairs_amount = back_pairs_amount
            self.batch_size = batch_size
            self.early_stopping = early_stopping
            self.optimization_algorithm = optimization_algorithm
            self.weight_decay = weight_decay
            self.lr_decay = lr_decay

            self.models_folder = os.path.join(self.root_folder, folder, tmp)
            if not os.path.exists(self.models_folder):
                os.makedirs(self.models_folder)

        print('Models folder: ', self.models_folder)
        self.logs_path = os.path.join(self.models_folder, 'logs')

        # local_device_protos = device_lib.list_local_devices()
        # print('Listing available devices: ', [x.name for x in local_device_protos])

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
            # pred.extend(sess.run(tf.sigmoid(sess.run(predictions, feed_dict))))
            pred.extend(sess.run(self.sigmoid_operation, feed_dict))

        pred = np.array(pred).reshape(-1, 20)
        pred = np.argmax(pred, axis=-1).reshape(-1).astype(int)
        for i in range(400):
            confusion_matrix[i % 20, pred[i]] += 1

        # np.sum(confusion_matrix) will be equal to 400 in case of 400 one-shot learning trials
        total_acc = np.sum(np.diagonal(confusion_matrix)) / 400
        return np.around(total_acc, decimals=4)

    def fit(self, num_epochs=100):

        x1 = tf.placeholder(tf.float32, shape=[None, 105, 105, 1], name='X1')
        x2 = tf.placeholder(tf.float32, shape=[None, 105, 105, 1], name='X2')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
        tf_is_training = tf.placeholder(tf.bool)

        predictions = self.model(x1, x2, tf_is_training, self.implementation)
        self.sigmoid_operation = tf.sigmoid(predictions)
        predictions_labels = tf.round(self.sigmoid_operation)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions_labels, y), tf.float32))

        back_pairs_amount = self.back_pairs_amount
        back_images_dict = data.load_image_dict(self.root_folder, 'images/images_background', 'images_background')

        # region Data
        if self.restore:
            train_pairs_images, train_pairs_types = data.unpickle_it(os.path.join(self.models_folder, 'train_pairs.pkl'))
            val_trials = data.unpickle_it(os.path.join(self.models_folder, 'val_trials.pkl'))
            params = data.unpickle_it(os.path.join(self.models_folder, 'params.json'))
            self.batch_size = params['batch_size']
            self.weight_decay = params['weight_decay']
            self.starter_learning_rate = params['starter_learning_rate']
            self.lr_decay = params['lr_decay']
            self.optimization_algorithm = params['optimization_algorithm']
            self.early_stopping = params['early_stopping']
        else:
            train_pairs_images, train_pairs_types = data.generate_image_pairs(images_dict=back_images_dict, pairs_amount=back_pairs_amount, ceil=self.ceil, path_only=True)
            data.pickle_it([train_pairs_images, train_pairs_types], os.path.join(self.models_folder, 'train_pairs.pkl'))
            params = {'starter_learning_rate': self.starter_learning_rate, 'learning_rate': self.starter_learning_rate, 'batch_size': self.batch_size,
                      'optimization_algorithm': self.optimization_algorithm, 'early_stopping': self.early_stopping, 'early_stopping_limit': (45 if self.augment else 20),
                      'weight_decay': self.weight_decay, 'last_epoch': -1, 'last_subepoch': 0, 'num_epochs': num_epochs, 'augment': self.augment,
                      'back_pairs_amount': back_pairs_amount, 'val_acc': 0, 'top_val_acc': 0, 'top_val_acc_step': -1, 'eval_acc': 0, 'lr_decay': self.lr_decay,
                      'timestamp': '{:02}:{:02}'.format(datetime.now().hour, datetime.now().minute),
                      'implementation': self.implementation}
            data.pickle_it(params, os.path.join(self.models_folder, 'params.json'))

            validation_images_dict = data.load_image_dict(self.root_folder, 'images/images_evaluation', 'images_evaluation', clear=True)
            val_trials = data.generate_one_shot_trials(images_dict=validation_images_dict)
            data.pickle_it(val_trials, os.path.join(self.models_folder, 'val_trials.pkl'))
            del validation_images_dict

        omniglot_dataset = data.OmniglotDataset(train_pairs_images, train_pairs_types, back_images_dict, back_pairs_amount, self.augment, )
        # print('Omniglot dataset length: (must be equal to image pairs or x9 if augmentation provided)', len(omniglot_dataset))

        batch_size = self.batch_size
        num_train_batches = (len(omniglot_dataset) + batch_size - 1) // batch_size
        num_subepoch_batches = (back_pairs_amount + batch_size - 1) // batch_size
        del back_images_dict, train_pairs_images, train_pairs_types

        val_trials_len = len(val_trials)
        val_trials = (val_trials.astype(np.float32) - 127.5) / 127.5

        eval_trials = data.get_eval_trials(self.root_folder)
        eval_trials_len = len(eval_trials)
        eval_trials = (eval_trials.astype(np.float32) - 127.5) / 127.5
        num_val_trials_batches, num_eval_trials_batches = (np.array([val_trials_len, eval_trials_len]) + batch_size - 1) // batch_size
        print('Amount of batches: {} training, {} subepoch, {} validation '.format(num_train_batches, num_subepoch_batches, num_val_trials_batches))
        # endregion
        # region TFUtils

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predictions))
        reg_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        total_loss = loss + reg_loss

        tf.summary.scalar('Accuracy/Train', accuracy)
        tf.summary.scalar('Train/Loss', loss)
        tf.summary.scalar('Train/RegularizationLoss', reg_loss)
        tf.summary.scalar('Train/TotalLoss', total_loss)

        saver = tf.train.Saver()
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step, num_train_batches, self.lr_decay)
        optimization_algorithm_dict = {
            'Adagrad': tf.train.AdagradOptimizer(learning_rate=learning_rate),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5),
            'Adadelta': tf.train.AdadeltaOptimizer(learning_rate=learning_rate),
            # Have not checked yet
            'Adam': tf.train.AdamOptimizer(learning_rate=learning_rate),
            'AdagradDA': tf.train.AdagradDAOptimizer(learning_rate=learning_rate, global_step=global_step)
        }
        optimizer = optimization_algorithm_dict[self.optimization_algorithm]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step=global_step)
        merged_summary_op = tf.summary.merge_all()

        sess_elements = [sess, x1, x2, y, tf_is_training, predictions, accuracy, loss]

        sess.run(tf.global_variables_initializer())
        print('Started the session')
        if self.restore:
            saver.restore(sess, os.path.join(self.models_folder, 'LatestModel.ckpt'))

        summary_writer = tf.summary.FileWriter(self.logs_path)
        # endregion

        num_subepochs = 1 if not self.augment else 9
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
                # Needed for right train summary log
                num_train_batches = 1
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

                if (i % num_subepoch_batches == 0 and i > 0) or i == num_train_batches - 1 or epoch == -1:
                    time_now = datetime.now()
                    step = epoch * num_subepochs + subepoch
                    print('time: {:02}:{:02}, subepoch {}.{}(step {}), loss:{:5.3f}, acc.:{:4.3f};'.format(time_now.hour, time_now.minute, epoch, subepoch,
                                                                                                           step, subepoch_loss, subepoch_acc), end='')

                    val_acc = self.new_check_one_shot_learning(sess_elements, val_trials, val_trials_len, num_val_trials_batches, batch_size)
                    print(' validation acc.: {:4.3f}/{:4.3f} (best on step {:2})'.format(val_acc, top_val_acc, top_val_acc_step), end='')

                    if val_acc > params['top_val_acc'] + 0.001:
                        top_val_acc, top_val_acc_step = val_acc, step
                        params['top_val_acc'], params['top_val_acc_step'] = val_acc, step
                        eval_acc = self.new_check_one_shot_learning(sess_elements, eval_trials, eval_trials_len, num_eval_trials_batches, batch_size)
                        print(' Highest! EvalAcc.: {:4.3f}'.format(eval_acc))
                        params['eval_acc'] = eval_acc
                        saver.save(sess, os.path.join(self.models_folder, 'BestModel.ckpt'))
                    else:
                        print('')

                    # Better tags were Validation/OSL and Evaluation/OSL
                    summary = tf.Summary()
                    summary.value.add(tag='OneShotLearning/Validation', simple_value=val_acc)
                    summary_writer.add_summary(summary, step)

                    summary = tf.Summary()
                    summary.value.add(tag='OneShotLearning/Evaluation', simple_value=eval_acc)
                    # Add value for test with no step
                    summary_writer.add_summary(summary)
                    summary_writer.flush()

                    subepoch_loss = subepoch_acc = 0
                    subepoch += 1

                    params['learning_rate'] = float(sess.run(learning_rate))
                    params['val_acc'] = val_acc
                    params['last_epoch'] = epoch
                    params['last_subepoch'] = subepoch
                    params['timestamp'] = '{:02}-{:02}'.format(datetime.now().hour, datetime.now().minute)
                    data.pickle_it(params, os.path.join(self.models_folder, 'params.json'))
                    saver.save(sess, os.path.join(self.models_folder, 'LatestModel.ckpt'))
                    # print('Repickled params.json at {:02}:{:02}'.format(params['hour'], params['minute']))

                    if self.early_stopping and step - top_val_acc_step > params['early_stopping_limit']:
                        print('Finished to train because of early stopping.')
                        return

                if epoch == -1:
                    print('Code checked.')
                    break
        print('Finished to fit.')


if __name__ == '__main__':
    # No batch_norm graph
    # Adam
    # AdagradDA
    # Different alphabets pairs
    # Chi-square metric

    net = Network(debug=False, verbose=False, back_pairs_amount=150000, augment=True, batch_size=256, starter_learning_rate=0.05,
                  weight_decay=0.00001, lr_decay=0.99, optimization_algorithm='Adagrad')
    net.fit(num_epochs=100)
    del net

    ops.reset_default_graph()
    sess = tf.Session(config=config)

