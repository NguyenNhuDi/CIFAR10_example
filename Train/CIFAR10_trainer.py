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
from torchvision import models
import torch.nn as nn
import time
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

"""
Change your paths here
"""
csv_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\TRAIN_VAL\0.csv'
yml_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\TRAIN_VAL\train_labels.yml'
train_image_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\cifar10\train'
test_image_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\cifar10\test'
save_path = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\Models\baseline.pth'
batch_size = 32
epochs = 25
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()
momentum = 0.9
learning_rate = 0.1
weight_decay = 0.75
num_workers = 8


def evaluate(val_batches, model):
    model.eval()

    total_correct = 0
    total_loss = 0
    total = 0

    for data in val_batches:
        image, label = data
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(image)
            loss = criterion(outputs, label)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)
            _, prediction = outputs.max(1)

            total_correct += (label == prediction).sum()

        loss = total_loss / total
        accuracy = total_correct / total
        # print(f'Evaluate --- Epoch: {epoch}, Loss: {loss:6.8f}, Accuracy: {accuracy:6.8f}')

        return loss, accuracy


def train_model(model, val_batches, train_batches):
    model = model.double()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    total = 0
    total_correct = 0
    total_loss = 0

    best_loss = 10000
    best_accuracy = -1
    best_epoch = 0

    for epoch in range(epochs):
        start = time.time()

        for data in tqdm(train_batches):
            image, label = data
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()

            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total += image.size(0)
            _, predictions = outputs.max(1)
            total_correct += (predictions == label).sum()

            total_loss += loss.item() * image.size(0)

        time_per_epoch = time.time() - start
        eval_loss, eval_accuracy = evaluate(val_batches, model)

        model.train()

        train_total_loss = total_loss / total
        train_accuracy = total_correct / total

        print(f'--- Epoch time: {time_per_epoch / 60.0:6.8f} minutes ---\n'
              f'Train Accuracy: {train_accuracy:6.8f} --- Train Loss: {train_total_loss:6.8f}\n'
              f'Eval Accuracy: {eval_accuracy:6.8f} --- Eval Loss: {eval_loss:6.8f}')

        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_epoch = epoch
            torch.save(model, save_path)
            best_loss = eval_loss if eval_loss <= best_loss else best_loss
        print(f'Best Accuracy: {best_accuracy} --- Best Loss: {best_loss}\n'
              f'Current Epoch: {epoch} --- Best Epoch: {best_epoch}')


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
        if out_label == 'airplane':
            out_label = 1
        elif out_label == 'automobile':
            out_label = 2
        elif out_label == 'bird':
            out_label = 3
        elif out_label == 'cat':
            out_label = 4
        elif out_label == 'deer':
            out_label = 5
        elif out_label == 'dog':
            out_label = 6
        elif out_label == 'frog':
            out_label = 7
        elif out_label == 'horse':
            out_label = 8
        elif out_label == 'ship':
            out_label = 9
        else:
            out_label = 0

        out_image = torch.from_numpy(out_image).permute(2, 0, 1)
        out_image = out_image.double()
        out_label = torch.tensor(out_label).long()

        return out_image, out_label


with open(yml_path, 'r') as f:
    yml_label = yaml.safe_load(f)

if __name__ == '__main__':
    transform = A.Compose(
        transforms=[
            A.Resize(height=224, width=224, p=1),
            A.Flip(p=0.5),
            A.RandomRotate90(p=1)
        ],
        p=1.0
    )

    train_set = CIFARDataset(yml_labels=yml_label,
                             image_dir=train_image_path,
                             csv_dir=csv_path,
                             transform=transform,
                             train=True)

    val_set = CIFARDataset(yml_labels=yml_label,
                           image_dir=train_image_path,
                           csv_dir=csv_path,
                           transform=transform,
                           train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    baseline_model = models.resnet34(pretrained=False)
    baseline_model.to(device)

    train_model(model=baseline_model, train_batches=train_loader, val_batches=val_loader)