from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import data
import numpy as np
import matplotlib.pyplot as plt
import os
import Net


if __name__ == '__main__':
    back_pairs_amount = 3000
    back_images_dict = data.load_image_dict('', 'images/images_background', 'images_background')
    train_pairs_images, train_pairs_types = data.generate_image_pairs(images_dict=back_images_dict, pairs_amount=back_pairs_amount, ceil=12, path_only=True)
    omniglot_dataset = Net.OmniglotDataset(train_pairs_images, train_pairs_types, back_images_dict, back_pairs_amount, augment=True)
    print('Omniglot dataset length: (must be equal to image pairs or x9 if augmentation provided)', len(omniglot_dataset))

    for epoch in range(10):
        dataloader = DataLoader(omniglot_dataset, batch_size=32, shuffle=True, num_workers=1)
        for i, batch in enumerate(dataloader, 0):
            batch_images = batch['pair'].numpy()
            batch_types = batch['pairType'].numpy().astype(np.float32)
            print(batch_images.shape)
            for j in range(32):
                print('Showing ', i, ' ', j)
                a = np.reshape(batch_images[j][0], newshape=(105, 105))
                b = np.reshape(batch_images[j][1], newshape=(105, 105))
                a[:, -2:] = -1
                image = np.hstack((a, b))
                plt.imshow(image, cmap='gray')
                plt.title('Similar' if bool(batch_types[j]) else 'Different')
                plt.show()
