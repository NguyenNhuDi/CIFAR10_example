import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import yaml
import albumentations as A
from torch.utils.data import DataLoader

"""
Change your paths here
"""
csv_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\TRAIN_VAL\0.csv'
yml_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\TRAIN_VAL\train_labels.yml'
train_image_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\cifar10\train'
test_image_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\cifar10\test'
batch_size = 32


class CIFARDataset(Dataset):
    def __init__(self, csv_dir, yml_labels, image_dir, transform: A.Compose = None, train=True):
        data = pd.read_csv(csv_dir)
        if train:
            self.images = data['train'].values.tolist()
        else:
            self.images = data['val'].values.tolist()

            out = None
            for index, item in enumerate(self.images):
                try:
                    int(item)
                except ValueError:
                    out = index
                    break

            self.images = self.images[: out]
        self.yml_labels = yml_labels
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = int(self.images[index])
        out_image = np.array(Image.open(os.path.join(self.image_dir, f'{image_name}.png')))
        if self.transform is not None:
            augmenter = self.transform(image=out_image)
            out_image = augmenter['image']

        out_label = self.yml_labels[image_name]
        print(image_name)
        return out_image, out_label


with open(yml_path, 'r') as f:
    yml_label = yaml.safe_load(f)

if __name__ == '__main__':
    transform = A.Compose(
        transforms=[
            A.Flip(p=0.5),
            A.RandomRotate90(p=1)
        ],
        p=1.0
    )

    train_set = CIFARDataset(yml_labels=yml_label,
                             image_dir=train_image_path,
                             csv_dir=csv_path,
                             transform=transform,
                             train=False)

    val_set = CIFARDataset(yml_labels=yml_label,
                           image_dir=train_image_path,
                           csv_dir=csv_path,
                           transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # for i, data in enumerate(train_loader):
    #     image, label = data
    #     plt.imshow(image[0])
    #     plt.title(label[0])
    #     plt.show()

    for i in range(len(train_set)):
        image, label = train_set.__getitem__(i)
        print(label)
        # plt.imshow(image)
        # plt.title(label)
        # plt.show()

    # for j in range(2):
    #     for i in range(3):
    #         image, label = test_val_set.__getitem__(i)
    #         plt.imshow(image)
    #         plt.title(label)
    #         plt.show()
