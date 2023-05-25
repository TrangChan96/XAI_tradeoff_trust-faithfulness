import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# get train and test set and return list (label, path)
def get_img_path(root, train):
    img_path_list = []
    if train:
        f = open(os.path.join(root, 'meta', 'train.txt'), 'r').readlines()
    else:
        f = open(os.path.join(root, 'meta', 'test.txt'), 'r').readlines()
    for i, line in enumerate(f):
        label = line.split('/')[0]
        path = line.strip('\n')
        img_path_list.append((i, label, path))
    return img_path_list

# map label to index (for classification)
def label2idx(root):
    classes = {}
    path = os.path.join(root, 'meta', 'classes.txt')
    file = open(path, 'r')
    for i, line in enumerate(file.readlines()):
        classes[line.strip('\n')] = i
    return classes

def loader(path, transform):
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor

class Data_Loader(Dataset):
    def __init__(
            self,
            root,
            train=True,
            transform=None,
    ) -> None:

        super().__init__()

        self.root = root # "../dataset/food-101/"
        self.train = train # train or test
        self.transform = transform

        self.class_dict = label2idx(self.root)

        self.dataset_path = os.path.join(self.root, 'images')
        self.img_path_list = get_img_path(self.root, self.train)
        # print(len(self.img_path_list))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        i, label, img_path = self.img_path_list[index]
        target = self.class_dict[label]
        img = loader(os.path.join(self.dataset_path, '%s.jpg' % (img_path,)), self.transform)

        return i, img, target

# ---------------------------------------

class DataLoaderExplanation(Dataset):
    def __init__(self,
            root,
            data_length,
            train=True,
            explanation=True, # if False -> Predictions
    ) -> None:

        super().__init__()

        self.root = root # root of explanation or prediction
        self.train = train
        self.explanation = explanation # get explanation or prediction
        self.data_length = data_length

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        if self.explanation:
            if self.train:
                img = np.load(os.path.join(self.root, 'explanation', 'train', '%s.npy' % (index,)))
            else:
                img = np.load(os.path.join(self.root, 'explanation', 'test', '%s.npy' % (index,)))
        else:
            if self.train:
                img = np.load(os.path.join(self.root, 'prediction', 'train', '%s.npy' % (index,)))
            else:
                img = np.load(os.path.join(self.root, 'prediction', 'test', '%s.npy' % (index,)))

        return img