import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import yaml
import albumentations as A

csv_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\CSV\0.csv'
yml_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\YAMLS\train_labels.yml'
image_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\cifar-10\train'

if __name__ == '__main__':
    class CIFARDataset(Dataset):
        def __init__(self, csv_dir, yml_labels, image_dir, transform: A.Compose = None, train=True):
            data = pd.read_csv(csv_dir)
            if train:
                self.images = data['train'].values.tolist()
            else:
                self.images = data['val'].values.tolist()

            self.yml_labels = yml_labels
            self.transform = transform
            self.image_dir = image_dir

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            image_name = self.images[index]
            print(image_name)
            out_image = np.array(Image.open(os.path.join(self.image_dir, f'{image_name}.png')))
            if self.transform is not None:
                augmenter = self.transform(image=out_image)
                out_image = augmenter['image']

            out_label = self.yml_labels[image_name]

            return out_image, out_label


    with open(yml_path, 'r') as f:
        yml_label = yaml.safe_load(f)

    testSet = CIFARDataset(yml_labels=yml_label, image_dir=image_path, csv_dir=csv_path)

    for i in range(3):
        image, label = testSet.__getitem__(i)

        plt.imshow(image)
        plt.title(label)
        plt.show()