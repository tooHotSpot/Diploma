import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
import _pickle
import json

from torch.utils.data import Dataset, DataLoader


def pickle_it(data, path):
    """
    Save data to pickle file

    :param data: data, class, array of objects
    :param path: full path to new file
    :return:
    """
    with open(path, 'wb') as f:
        _pickle.dump(data, f, protocol=4)


def unpickle_it(path):
    """
    Get data from pickle file

    :param path: full path to file with data
    :return:
    """
    with open(path, 'rb') as f:
        return _pickle.load(f)


def json_it(data, path):
    """
    Save data to json file

    :param data: data, class, array of objects
    :param path: full path to new file
    :return:
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def unjson_it(path):
    """
    Get data from json file

    :param path: full path to file with data
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

    pkl_file = os.path.join(ROOT, 'Data', subdataset_name + '.pkl')
    # print('Trying to load ', pkl_file)
    if os.path.exists(pkl_file):
        # print(pkl_file, ' already exists, loading to memory.')
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
        # print('Deleting alphabets from ', subdataset_name)
        for alphabet in sorted(list(images_dict)):
            if alphabet in resticted_alpahbets:
                # print('-->Minus alphabet', alphabet)
                images_dict.pop(alphabet, None)
    else:
        pass
        # print('No excluded alphabets for ', subdataset_name)
    return images_dict


def generate_image_pairs(images_dict, pairs_amount, ceil, path_only=False, exclude=True):
    """
    Method generates array of image pairs basically for verification train/validation.

    :param pairs_amount:
    :param images_dict:
    :param ceil: due to selection in article
    :param seed: random seed fixed for experiments handling
    :param path_only
    :param exclude
    :return:
    """

    alphabets_amount = len(images_dict.keys())
    alphabet_all_pairs = pairs_amount // alphabets_amount
    alphabet_equal_pairs_needed_amount = alphabet_distinct_pairs_needed_amount = alphabet_all_pairs // 2

    if exclude:
        for alphabet in images_dict:
            characters = images_dict[alphabet].keys()
            for character in characters:
                for image in np.random.choice(list(images_dict[alphabet][character]), size=(20 - ceil), replace=False):
                    images_dict[alphabet][character].pop(image, None)
        # print('Excluded {} images for every character'.format(20 - ceil))

    equal_pairs = []
    distinct_pairs = []
    # Selecting equal pairs
    for alphabet in images_dict:
        current_alphabet_equal_pairs = []
        characters = images_dict[alphabet].keys()
        for character in characters:
            character_images = []
            for image in np.random.choice(list(images_dict[alphabet][character]), size=ceil):
                character_images.append([alphabet, character, image])

            # Select combinations('ABCD', 2): AB AC AD BC BD CD
            # List all combinations of the characters
            current_character_equal_pairs = list(itertools.combinations(character_images, 2))
            current_alphabet_equal_pairs.extend(current_character_equal_pairs)

        # Equal pairs
        replace = len(current_alphabet_equal_pairs) < alphabet_equal_pairs_needed_amount
        equal_pairs_numbers = np.random.choice(len(current_alphabet_equal_pairs),
                                               size=alphabet_equal_pairs_needed_amount,
                                               replace=replace)
        for i in equal_pairs_numbers:
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
                distinct_pairs.append([image1, image2])

    amount_equal_pairs = len(equal_pairs)
    amount_distinct_pairs = len(distinct_pairs)
    # print('Amount of equal_pairs:', amount_equal_pairs, ' and different_pairs:', amount_distinct_pairs)
    all_pairs_types = np.concatenate((np.ones(amount_equal_pairs), np.zeros(amount_distinct_pairs))).reshape(-1, 1).astype(np.float32)

    equal_pairs.extend(distinct_pairs)
    all_pairs_images = equal_pairs
    del equal_pairs, distinct_pairs

    if not path_only:
        all_pairs_images = np.expand_dims(np.array(all_pairs_images), axis=-1)
        # print('Loading images array with shape: ', all_pairs_images.shape)

    return all_pairs_images, all_pairs_types


def generate_one_shot_trials(images_dict):
    """
    Method generates one-shot trials single array to iterate through and calculate accuracy

    :param images_dict: None or dictionary of next structure:
    :param subdataset: path to the subdataset folder
    :param seed: random seed to repeat the experiment
    :return: 400 one-shot trials result in 8000 pairs
    """

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
    # print('Validation one-shot learning array shape: ', all_comparisons_array.shape)
    return all_comparisons_array  # , all_comparisons_answers_list


def augment_int_image(oldimage, borderValue):
    image = oldimage.copy()
    image = image.reshape(105, 105)
    # Choose transformations
    transforms = np.random.choice([0, 1], size=7)
    # Rotation, theta in [-10, 10]
    if transforms[0]:
        M = cv2.getRotationMatrix2D(center=(105 / 2, 105 / 2), angle=np.random.randint(-15, 16), scale=1)
        image = cv2.warpAffine(image, M, (105, 105), borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)

    # Create matrix in order to fill separate values
    M = np.float32([[1, 0, 0],
                    [0, 1, 0]])

    # Shearing: ρx, ρy ∈ [−0.3, 0.3] in rad -> [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi] in angles
    M[0, 1] = -np.sin(np.random.randint(-3, 4) / 10) if transforms[1] else M[0, 1]
    M[1, 1] = np.cos(np.random.randint(-3, 4) / 10) if transforms[2] else M[1, 1]

    # Translation: tx, ty ∈ [−2, 2]
    M[0, 2] = np.random.randint(-4, 5) if transforms[3] else M[0, 2]
    M[1, 2] = np.random.randint(-4, 5) if transforms[4] else M[1, 2]

    # # Scaling: sx, sy ∈ [0.8, 1.2]
    M[0, 0] = np.random.randint(8, 13) / 10 if transforms[5] else M[0, 0]
    M[1, 1] = np.random.randint(8, 13) / 10 if transforms[6] else M[1, 1]

    # Perform all the transforms in one
    image = cv2.warpAffine(image, M, (105, 105), borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return image


class OmniglotDataset(Dataset):
    def __init__(self, chosen_pairs_np_array, chosen_pairs_answers_np_array, background_images_dict, pairs_amount, augment):
        assert pairs_amount == len(chosen_pairs_answers_np_array) == len(chosen_pairs_np_array), 'Incorrect pairs sizes'
        self.pairs = chosen_pairs_np_array
        self.pairs_answers = chosen_pairs_answers_np_array
        self.pairs_amount = pairs_amount
        self.background_images_dict = background_images_dict
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

        # One time real images, others - augmented if it is enabled
        if item > self.pairs_amount:
            image1 = augment_int_image(image1, borderValue=255)
            image2 = augment_int_image(image2, borderValue=255)

        pair = (np.array([image1, image2], dtype=np.float32) - 127.5) / 127.5
        sample = {
            'pair': pair.reshape(2, 105, 105, 1),
            'pairType': self.pairs_answers[real_item]
        }
        return sample


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


def show_augmentation(dataloader):
    for i, batch in enumerate(dataloader, 0):
        batch_images = batch['pair'].numpy()
        for j in range(32):
            # print('Showing ', i, ' ', j)
            a = np.reshape(batch_images[j][0], newshape=(105, 105))
            b = augment_int_image(a, borderValue=1)
            a[:, -2:] = -1
            image = np.hstack((a, b))
            plt.imshow(image, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()


def get_eval_trials(root_path):
    all_comparisons, all_comparisons_answers = unpickle_it(os.path.join(root_path, 'Data/all_runs_trials.pkl'))
    return all_comparisons


def show_dataset(omniglot_dataset):
    for i in range(len(omniglot_dataset)):
        sample = omniglot_dataset[i]
        pair = sample['pair']
        image1, image2 = pair[0].reshape(105, 105), pair[1].reshape(105, 105)
        # image1, image2 = pair[0][0].reshape(105, 105), pair[0][1].reshape(105, 105)
        pairType = int(sample['pairType'])
        print(i, image1.shape, image2.shape, pairType)
        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        ax[0].imshow(image1, cmap='gray')
        ax[1].imshow(image2, cmap='gray')
        plt.suptitle('Sample #{}'.format(pairType))
        plt.show()


def utilize_tensorboard_image(dataloader, show=False):
    dataloader_length = 0
    import tensorflow as tf

    sess = tf.Session()
    writer = tf.summary.FileWriter('Images/TensorboardImages/logs', sess.graph)

    for i_batch, sample_batched in enumerate(dataloader, 0):
        batch_images = sample_batched['pair'].numpy()
        batch_types = sample_batched['pairType'].numpy().astype(np.float32)

        dataloader_length += len(batch_images)
        print(i_batch, len(batch_images), dataloader_length, sep=' ')

        if show:
            for j in range(dataloader.batch_size):
                pair = batch_images[j]
                pairType = int(batch_types[j])
                image1, image2 = pair[0].reshape(105, 105), pair[1].reshape(105, 105)
                # Stack horizontally and put meta info text
                image = np.array((np.concatenate([image1, image2], axis=1) * 127.5) + 127.5, dtype=np.uint8).reshape(1, 105, 210, 1)
                image = cv2.putText(image, str(bool(pairType)), (5, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=0,
                                    thickness=2, lineType=cv2.LINE_AA)
                summary_op = tf.summary.image("Plots/Images", image)
                summary = sess.run(summary_op)
                writer.add_summary(summary, j)

                if j == 10:
                    break
            break
    writer.close()
    print(dataloader_length)


if __name__ == '__main__':
    back_pairs_amount = 3000
    back_images_dict = load_image_dict('', 'images/images_background', 'images_background')
    train_pairs_images, train_pairs_types = generate_image_pairs(images_dict=back_images_dict, pairs_amount=back_pairs_amount, ceil=12, path_only=True)
    omniglot_dataset = OmniglotDataset(train_pairs_images, train_pairs_types, back_images_dict, back_pairs_amount, augment=False)

    print('Omniglot dataset length: (must be equal to image pairs or x9 if augmentation provided)', len(omniglot_dataset))
    show_dataset(omniglot_dataset)

    dataloader = DataLoader(omniglot_dataset, batch_size=32, shuffle=True, num_workers=1)
    utilize_tensorboard_image(dataloader)
