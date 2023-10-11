from PIL import Image
import pandas as pd
import yaml
import json
import argparse
import os
from sklearn.model_selection import KFold
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Winter Wheat, Winter Rye Classifier pre process',
        description='This program will create different val and train set for classification of winter wheat and '
                    'winter rye based on a  and hash function',
        epilog='Vision Research Lab')

    parser.add_argument('-c', '--config', required=True,
                        help='The path to the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        args = json.load(f)

    yaml_paths = args['yaml_paths']
    output_dir = args['output_dir']
    folds = args['folds']
    seed = args['seed']
    val_percent = args['val_percent']
    test_dir = args['test_dir']

    img_per_class = 5000

    val_size = img_per_class * val_percent

    val_set = {}
    train_set = {}
    labels = {}

    for yaml_path in yaml_paths:
        with open(yaml_path, 'r') as f:
            labels.update(yaml.safe_load(f))

    classes = [[] for i in range(10)]

    for image_name in tqdm(labels):

        try:
            _ = Image.open(os.path.join(test_dir, f'{image_name}.png'))
            print(f'{image_name}.png')
            continue
        except FileNotFoundError:
            pass

        curr_class = labels[image_name]

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

    for i in range(10):
        for j, (train_index, test_index) in enumerate(kf.split(classes[i])):

            out_csv = {
                'train': [],
                'val': []
            }

            for index in train_index:
                out_csv['train'].append(classes[i][index])

            for index in test_index:
                out_csv['val'].append(str(classes[i][index]))

            while len(out_csv['train']) != len(out_csv['val']):
                out_csv['val'].append(None)

            df = pd.DataFrame.from_dict(out_csv)
            save_path = os.path.join(output_dir)

            try:
                os.makedirs(save_path)
            except FileExistsError:
                pass

            df.to_csv(os.path.join(save_path, f'{j}.csv'), index=False)
