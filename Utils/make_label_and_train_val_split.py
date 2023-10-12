import pandas as pd
import yaml
import os
from sklearn.model_selection import KFold
from tqdm import tqdm

csv_labels = r'C:\Users\coanh\Desktop\UNI\AIC\Classification Competetion\cifar10\trainLabels.csv'
label_save_path = r'C:\\Users\\coanh\\Desktop\\UNI\\AIC\\Classification Competetion\\TRAIN_VAL'
output_dir = "C:\\Users\\coanh\\Desktop\\UNI\\AIC\\Classification Competetion\\TRAIN_VAL"
folds = 5
seed = 3011182772811103
val_percent = 0.2
img_per_class = 5000

if __name__ == '__main__':

    data = pd.read_csv(csv_labels)

    iD = data['ID'].values.tolist()
    label = data['label'].values.tolist()

    yaml_labels = {}

    for i, key in enumerate(iD):
        yaml_labels[key] = label[i]

    with open(os.path.join(label_save_path, 'train_labels.yml'), 'w') as f:
        yaml.dump(yaml_labels, f)

    val_size = img_per_class * val_percent

    val_set = {}
    train_set = {}

    classes = [[] for i in range(10)]

    for image_name in tqdm(yaml_labels):
        curr_class = yaml_labels[image_name]
        class_index = 0

        if curr_class == 'airplane':
            class_index = 1
        elif curr_class == 'automobile':
            class_index = 2
        elif curr_class == 'bird':
            class_index = 3
        elif curr_class == 'cat':
            class_index = 4
        elif curr_class == 'deer':
            class_index = 5
        elif curr_class == 'dog':
            class_index = 6
        elif curr_class == 'frog':
            class_index = 7
        elif curr_class == 'horse':
            class_index = 8
        elif curr_class == 'ship':
            class_index = 9

        classes[class_index].append(image_name)

    kf = KFold(n_splits=folds, shuffle=True)

    out_csvs = [{
        'train': [],
        'val': []
    }
        for i in range(folds)]

    for i in range(10):
        for j, (train_index, val_index) in enumerate(kf.split(classes[i])):

            for index in train_index:
                out_csvs[j]['train'].append(classes[i][index])

            for index in val_index:
                out_csvs[j]['val'].append(str(classes[i][index]).split('.')[0])


    for i in range(folds):
        while len(out_csvs[i]['train']) != len(out_csvs[i]['val']):
            out_csvs[i]['val'].append(None)

        df = pd.DataFrame.from_dict(out_csvs[i])
        save_path = os.path.join(output_dir)

        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        df.to_csv(os.path.join(save_path, f'{i}.csv'), index=False)
