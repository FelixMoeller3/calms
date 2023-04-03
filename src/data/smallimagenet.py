import os
import pickle

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class SmallImagenet(VisionDataset):
    #batch_nums_to_use = [1]
    #train_list = ['train_data_batch_{}'.format(i) for i in batch_nums_to_use]
    val_list = ['val_data']

    def __init__(self, root="data", batch_num:int=1,size=32, train=True, transform=None, classes=None):
        super().__init__(root, transform=transform)
        if batch_num < 1 or batch_num > 10:
            raise ValueError("Batch number must be between 1 and 10")
        batch_nums_to_use = [batch_num]
        train_list = ['train_data_batch_{}'.format(i) for i in batch_nums_to_use]
        file_list = train_list if train else self.val_list
        self.data = []
        self.targets = []
        self.root = os.path.join(root,"SmallImageNet_32x32")
        for filename in file_list:
            filename = os.path.join(self.root, filename)
            with open(filename, 'rb') as f:
                entry = pickle.load(f)
            self.data.append(entry['data'].reshape(-1, 3, size, size))
            self.targets.append(entry['labels'])

        self.data = np.vstack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.concatenate(self.targets).astype(int) - 1

        if classes is not None:
            classes = np.array(classes)
            filtered_data = []
            filtered_targets = []

            for l in classes:
                idxs = self.targets == l
                filtered_data.append(self.data[idxs])
                filtered_targets.append(self.targets[idxs])

            self.data = np.vstack(filtered_data)
            self.targets = np.concatenate(filtered_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
