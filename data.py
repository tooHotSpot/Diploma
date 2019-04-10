import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
import _pickle as cPickle
import json


def pickle_it(data, path):
    """
    Сохранить данные в pickle файл

    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'wb') as f:
        cPickle.dump(data, f, protocol=4)


def unpickle_it(path):
    """
    Достать данные из pickle файла

    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'rb') as f:
        return cPickle.load(f)


def json_it(data, path):
    """
    Сохранить данные data в json файл

    :param data: данные, класс, массив объектов
    :param path: путь до итогового файла
    :return:
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def unjson_it(path):
    """
    Достать данные из json файла

    :param path: путь до файла с данными
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def show_image(image):
    plt.imshow(image.reshape(105, 105), cmap='gray')
    plt.show()


def load_image_dict(ROOT, subdataset_path, subdataset_name, clear=False):
    """
    Method creates a pickle file with Omniglot images in the dictionary structure.

    :param subdataset_name: just the end of the folder, for safe but implicit code
    :param subdataset_path: path to the subdataset folder
    :param clear: clear from restrictedd alphabets
    :return: images_dict: dictionary of next structure:
            {
                'alphabet1':
                {
                    'character1': {
                            'image1': np.array(cv2.imread(image1_path, 0), dtype=np.uint8)
                            'image20': np.array(cv2.imread(image20_path, 0), dtype=np.uint8)
                    }
                    'character2':{
                            'image1': np.array(cv2.imread(image1_path, 0), dtype=np.uint8)
                            'image20': np.array(cv2.imread(image20_path, 0), dtype=np.uint8)
                    }
                }
            }
    """
    resticted_alpahbets = ('Atlantean', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Malayalam',
                           'Manipuri', 'Old_Church_Slavonic_(Cyrillic)', 'Tengwar', 'Tibetan')

    pkl_file = os.path.join(ROOT, 'OmniglotData', subdataset_name + '.pkl')
    if os.path.exists(pkl_file):
        print(pkl_file, ' already exists, loading to memory.')
        images_dict = unpickle_it(pkl_file)
    else:
        subdataset_path = os.path.join(ROOT, subdataset_path)
        assert os.path.exists(subdataset_path), subdataset_name + ' is not found in the ' + subdataset_path
        print(pkl_file, ' is absent, reading images and gathering them to dict structure.')
        images_dict = {}
        for alphabet in os.listdir(subdataset_path):
            images_dict[alphabet] = {}
            alphabet_path = os.path.join(subdataset_path, alphabet)
            for character in os.listdir(alphabet_path):
                images_dict[alphabet][character] = {}
                character_path = os.path.join(alphabet_path, character)
                for image in os.listdir(character_path):
                    image_path = os.path.join(character_path, image)
                    images_dict[alphabet][character][image] = np.array(cv2.imread(image_path, 0), dtype=np.uint8)
            print('Passed through ', alphabet)
        pickle_it(images_dict, pkl_file)

    if clear:
        print('Deleting alphabets from ', subdataset_name)
        for alphabet in sorted(list(images_dict)):
            if alphabet in resticted_alpahbets:
                print('-->Minus alphabet', alphabet)
                images_dict.pop(alphabet, None)
    else:
        print('No excluded alphabets for ', subdataset_name)
    return images_dict


def generate_image_pairs(images_dict, pairs_amount, ceil, path_only=False, exclude=True):
    '''

    :param pairs_amount:
    :param images_dict:
    :param ceil: due to selection in article
    :param seed: random seed fixed for experiments handling
    :return:
    '''
    # np.random.seed(seed)

    alphabets_amount = len(images_dict.keys())
    alphabet_all_pairs = pairs_amount // alphabets_amount
    alphabet_equal_pairs_needed_amount = alphabet_distinct_pairs_needed_amount = alphabet_all_pairs // 2

    if exclude:
        for alphabet in images_dict:
            characters = images_dict[alphabet].keys()
            for character in characters:
                for image in np.random.choice(list(images_dict[alphabet][character]), size=(20 - ceil)):
                    images_dict[alphabet][character].pop(image, None)
        print('Excluded {} images for every character'.format(20 - ceil))

    equal_pairs = []
    distinct_pairs = []
    # Selecting equal pairs
    for alphabet in images_dict:
        current_alphabet_equal_pairs = []
        characters = images_dict[alphabet].keys()
        for character in characters:
            current_character_images = []
            for image in np.random.choice(list(images_dict[alphabet][character]), size=ceil):
                current_character_images.append([alphabet, character, image])

            # Select combinations('ABCD', 2): AB AC AD BC BD CD
            # List all combinations of the characters
            current_character_equal_pairs = list(itertools.combinations(current_character_images, 2))
            current_alphabet_equal_pairs.extend(current_character_equal_pairs)

        # Equal pairs
        replace = len(current_alphabet_equal_pairs) < alphabet_equal_pairs_needed_amount
        randomly_chosen_equal_pairs_numbers = np.random.choice(len(current_alphabet_equal_pairs),
                                                               size=alphabet_equal_pairs_needed_amount,
                                                               replace=replace)
        for i in randomly_chosen_equal_pairs_numbers:
            entry1, entry2 = current_alphabet_equal_pairs[i]

            if path_only:
                # Here made it explicitly, but could pass just current_alphabet_equal_pairs[i]
                equal_pairs.append([entry1, entry2])
            else:
                alphabet, character1, image1 = entry1
                image1 = images_dict[alphabet][character1][image1]

                alphabet, character2, image2 = entry2
                image2 = images_dict[alphabet][character2][image2]

                equal_pairs.append([image1, image2])

        # Select distinct pairs of characters
        distinct_pairs_combinations = list(itertools.combinations(characters, 2))
        randomly_chosen_distinct_pairs_numbers = np.random.choice(len(distinct_pairs_combinations),
                                                                  size=alphabet_distinct_pairs_needed_amount,
                                                                  replace=True)
        for i in randomly_chosen_distinct_pairs_numbers:
            character1, character2 = distinct_pairs_combinations[i]
            # Choose random images of this characters
            key1 = list(images_dict[alphabet][character1].keys())[np.random.randint(ceil)]
            key2 = list(images_dict[alphabet][character2].keys())[np.random.randint(ceil)]
            if path_only:
                entry1 = [alphabet, character1, key1]
                entry2 = [alphabet, character2, key2]
                distinct_pairs.append([entry1, entry2])
            else:
                image1 = images_dict[alphabet][character1][key1]
                image2 = images_dict[alphabet][character2][key2]
                # image1 = list(images_dict[alphabet][character1].values())[np.random.randint(ceil)]
                # image2 = list(images_dict[alphabet][character2].values())[np.random.randint(ceil)]
                distinct_pairs.append([image1, image2])

    amount_equal_pairs = len(equal_pairs)
    amount_distinct_pairs = len(distinct_pairs)
    print('Amount of equal_pairs:', amount_equal_pairs, ' and different_pairs:', amount_distinct_pairs)
    all_pairs_types = np.concatenate((np.ones(amount_equal_pairs), np.zeros(amount_distinct_pairs))).reshape(-1, 1).astype(np.float32)

    # old way
    # all_pairs_images = np.array(equal_pairs + distinct_pairs).reshape(amount_equal_pairs + amount_distinct_pairs, 2, 105, 105, 1)

    equal_pairs.extend(distinct_pairs)
    all_pairs_images = equal_pairs
    del equal_pairs, distinct_pairs

    if not path_only:
        all_pairs_images = np.expand_dims(np.array(all_pairs_images), axis=-1)
        print('Loading images array with shape: ', all_pairs_images.shape)

    return all_pairs_images, all_pairs_types


def generate_one_shot_trials(images_dict):
    """
    Method generates one-shot trials for classification by verification in dictionary structure.

    :param images_dict: None or dictionary of next structure:
    :param subdataset: path to the subdataset folder
    :param seed: random seed to repeat the experiment
    :return: 400 one-shot trial dictionary of next structure:
    """
    # np.random.seed(seed)

    all_comparisons_list = []
    # all_comparisons_answers_list = []
    for alphabet in images_dict:
        # 20 cause 20 drawers for all characters, but only 2 drawers needed to perform the task
        sorted_chars = sorted(list(images_dict[alphabet].keys()))

        chosen_chars = np.random.choice(sorted_chars, 20, replace=False)
        drawers = np.random.choice(20, 4, replace=False)

        for i in range(2):
            drawer_1, drawer_2 = drawers[i * 2], drawers[i * 2 + 1]
            image_true_tuples_pairs = []
            for char in chosen_chars:
                char_images = list(images_dict[alphabet][char])
                image1, image2 = images_dict[alphabet][char][char_images[drawer_1]], images_dict[alphabet][char][char_images[drawer_2]]
                image_true_tuples_pairs.append([image1, image2])

            for i in range(20):
                for j in range(20):
                    all_comparisons_list.append([image_true_tuples_pairs[i][0], image_true_tuples_pairs[j][1]])
                    # all_comparisons_answers_list.append(i == j)

    all_comparisons_array = np.array(all_comparisons_list).reshape((8000, 2, 105, 105, 1))
    print('Validation one-shot learning array shape: ', all_comparisons_array.shape)
    return all_comparisons_array  # , all_comparisons_answers_list


def augment_image(image):
    image = image.reshape(105, 105)
    a = np.random.randint(low=0, high=1, size=7)
    theta = a[0] * np.random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((105 / 2, 105 / 2), theta, 1)
    image = cv2.warpAffine(image, M, (105, 105), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
    M = np.float32([[1, 0, 0],
                    [0, 1, 0]])
    # Shearing: ρx, ρy ∈ [−0.3, 0.3] in rad -> [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi] in angles
    rad = a[1] * np.random.randint(-5, 5) / 10
    M[0, 1] = -np.sin(rad)
    rad = a[2] * np.random.randint(-5, 5) / 10
    M[1, 1] = np.cos(rad)
    # Translation: tx, ty ∈ [−2, 2]
    M[0, 2] = a[5] * np.random.randint(-5, 5)
    M[1, 2] = a[6] * np.random.randint(-5, 5)
    # Scaling: sx, sy ∈ [0.8, 1.2]
    M[0, 0] = a[3] * np.random.randint(8, 12) / 10 + (1 - a[3])
    M[1, 1] = a[4] * np.random.randint(8, 12) / 10 + (1 - a[4])
    image = cv2.warpAffine(image, M, (105, 105), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)  # > 0
    return image


def augment_train_images_float(train_pairs_images, train_pairs_types, DEBUG_AUGMENTING=False):
    initial_train_len = len(train_pairs_types)
    # (θ, ρx, ρy, sx, sy, tx, tx)
    train_pairs_images = np.squeeze(train_pairs_images)
    new_train_pairs_images = []
    new_train_pairs_types = []
    for _ in range(7):
        for i in range(initial_train_len):
            new_pair = []
            for image in train_pairs_images[i]:
                a = np.random.choice([1], size=7)
                image = image.copy()
                # Rotation: θ ∈ [−10.0, 10.0]
                theta = a[0] * np.random.randint(-10, 10)
                M = cv2.getRotationMatrix2D((105 / 2, 105 / 2), theta, 1)
                image = cv2.warpAffine(image, M, (105, 105), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
                M = np.float32([[1, 0, 0],
                                [0, 1, 0]])
                # Shearing: ρx, ρy ∈ [−0.3, 0.3] in rad -> [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi] in angles
                rad = a[1] * np.random.randint(-2, 2) / 10
                M[0, 1] = -np.sin(rad)
                rad = a[2] * np.random.randint(-2, 2) / 10
                M[1, 1] = np.cos(rad)
                # Translation: tx, ty ∈ [−2, 2]
                M[0, 2] = a[5] * np.random.randint(-2, 2)
                M[1, 2] = a[6] * np.random.randint(-2, 2)
                # Scaling: sx, sy ∈ [0.8, 1.2]
                M[0, 0] = a[3] * np.random.randint(8, 10) / 10 + (1 - a[3])
                M[1, 1] = a[4] * np.random.randint(8, 10) / 10 + (1 - a[4])
                image = cv2.warpAffine(image, M, (105, 105), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)  # > 0
                new_pair.append(image)

            if DEBUG_AUGMENTING:
                fig, ax = plt.subplots(2, 2, figsize=(5, 5))
                # image0 = np.array(255 * (train_pairs_images[i][0] + 1) / 2).astype(np.uint8).reshape(105, 105)
                # image1 = np.array(255 * (train_pairs_images[i][1] + 1) / 2).astype(np.uint8).reshape(105, 105)
                # image2 = np.array(255 * (new_pair[0] + 1) / 2).astype(np.uint8).reshape(105, 105)
                # image3 = np.array(255 * (new_pair[1] + 1) / 2).astype(np.uint8).reshape(105, 105)

                image0 = train_pairs_images[i][0]
                image1 = train_pairs_images[i][1]
                image2 = new_pair[0]
                image3 = new_pair[1]

                ax[0, 0].imshow(image0, cmap='gray')
                ax[0, 1].imshow(image1, cmap='gray')
                ax[1, 0].imshow(image2, cmap='gray')
                ax[1, 1].imshow(image3, cmap='gray')
                for k in range(2):
                    for l in range(2):
                        ax[k, l].set_xticks([])
                        ax[k, l].set_yticks([])
                plt.show()
            new_train_pairs_images.append(new_pair)
            new_train_pairs_types.append(train_pairs_types[i])

    # print('Shape before: ', train_pairs_images.shape)
    train_pairs_images = np.expand_dims(np.vstack((train_pairs_images, new_train_pairs_images)), axis=-1)
    # print('Shape after: ', train_pairs_images.shape)
    train_pairs_types = np.concatenate((train_pairs_types, new_train_pairs_types))
    # print('Pairs amount: ', len(train_pairs_images), ' and ', len(train_pairs_types))
    return train_pairs_images, train_pairs_types


# back_images_dict = load_image_dict('C:/Users/Art/PycharmProjects/Diploma', 'images/images_background', 'images_background')
# train_pairs_images, train_pairs_types = generate_image_pairs(images_dict=back_images_dict, pairs_amount=300, ceil=12, seed=0)
# train_pairs_images = (train_pairs_images.astype(np.float32) - 127.5) / 127.5
# train_pairs_images, train_pairs_types = augment_train_images_float(train_pairs_images, train_pairs_types, DEBUG_AUGMENTING=True)
#

def show_pairs(pairs_images, pairs_types, limit=10):
    fig, a = plt.subplots(limit, 2, figsize=(10, 5))

    for i in range(limit):
        j = np.random.randint(len(pairs_images))
        image1, image2 = pairs_images[j]
        a[i, 0].imshow(image1.reshape(105, 105), cmap='gray')
        a[i, 1].imshow(image2.reshape(105, 105), cmap='gray')
        a[i, 0].set_title('Equal' if pairs_types[j] else 'Distinct')

    plt.tight_layout()
    plt.show()


def get_eval_trials(root_path):
    all_comparisons, all_comparisons_answers = unpickle_it(os.path.join(root_path, 'Data/all_runs_trials.pkl'))
    return all_comparisons

# def show_dataset():
#     for i in range(len(omniglot_dataset)):
#         sample = omniglot_dataset[i]
#         pair = sample['pair']
#         image1, image2 = pair[0].reshape(105, 105), pair[1].reshape(105, 105)
#         # image1, image2 = pair[0][0].reshape(105, 105), pair[0][1].reshape(105, 105)
#         pairType = int(sample['pairType'])
#         print(i, image1.shape, image2.shape, pairType)
#         fig, ax = plt.subplots(1, 2, figsize=(5, 5))
#         ax[0].imshow(image1, cmap='gray')
#         ax[1].imshow(image2, cmap='gray')
#         plt.suptitle('Sample #{}'.format(pairType))
#         plt.show()
#
# def show_dataloader(show=False):
#     dataloader_length = 0
#
#     sess = tf.Session()
#     writer = tf.summary.FileWriter('./logs', sess.graph)
#
#     for i_batch, sample_batched in enumerate(dataloader, 0):
#         batch_images = sample_batched['pair'].numpy()
#         batch_types = sample_batched['pairType'].numpy().astype(np.float32)
#
#         dataloader_length += len(batch_images)
#         print(i_batch, len(batch_images), dataloader_length, sep=' ')
#
#         if show:
#             for j in range(dataloader.batch_size):
#                 pair = batch_images[j]
#                 pairType = int(batch_types[j])
#                 image1, image2 = pair[0].reshape(105, 105), pair[1].reshape(105, 105)
#                 # print(j, image1.shape, image2.shape, pairType)
#                 # fig, ax = plt.subplots(1, 2, figsize=(5, 5))
#                 # ax[0].imshow(image1, cmap='gray')
#                 # ax[1].imshow(image2, cmap='gray')
#                 # plt.suptitle('Batch {}, num {}, sample #{}'.format(i_batch, j, pairType))
#                 # # plt.savefig('Image.png')
#                 # plt.show()
#                 image = np.array((np.concatenate([image1, image2], axis=1) * 127.5) + 127.5, dtype=np.uint8).reshape(1, 105, 210, 1)
#                 import cv2
#                 image = cv2.putText(image, str(bool(pairType)), (5, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=0,
#                                     thickness=2, lineType=cv2.LINE_AA)
#                 summary_op = tf.summary.image("Plots/Images", image)
#                 summary = sess.run(summary_op)
#                 writer.add_summary(summary, j)
#
#                 if j == 10:
#                     break
#             break
#     writer.close()
#     print(dataloader_length)


# different = np.nonzero(batch_predictions_labels != batch_types)[0]
#                     if len(different) != 0:
#                         r = np.random.choice(different)
#                         pair = batch_images[r]
#                         pairType = bool(batch_types[r])
#                         image1, image2 = pair[0].reshape(105, 105), pair[1].reshape(105, 105)
#                         image = np.array((np.concatenate([image1, image2], axis=1) * 127.5) + 127.5, dtype=np.uint8).reshape(105, 210)
#                         text = '{:02}/{:02}: {}'.format(epoch, i, 'Equal' if bool(pairType) else 'Different')
#                         # print('Added image, text:', text)
#                         image = cv2.resize(image, (100, 50))
#                         image = cv2.putText(image, text, (5, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=0, thickness=1, lineType=cv2.LINE_AA)
#                         image = image.reshape(1, 50, 100, 1)
#                         image_summary = sess.run(tf.summary.image('{}/{}'.format(epoch, i), image))
#                         summary_writer.add_summary(image_summary, epoch)  # summary_writer.add_summary(image_summary, i + num_train_batches * epoch)
