
import numpy as np
from csv import reader
import os


def get_one_hot_rep(id):
    base = np.zeros(25)
    base[id-1] = 1
    return base


def flat_dataset(dataset):
    flatten_dataset = list()

    for sample in dataset:
        flatten_sample = list()

        for elem in sample:
            flatten_sample += list(elem)

        flatten_dataset.append(flatten_sample)

    return flatten_dataset


def load_dataset(split=True):
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'train.csv')
    csv_reader = reader(open(file_path, 'rt'))

    X_train = list()
    y_train = list()

    first_row = True
    for row in csv_reader:
        if first_row:
            first_row = False
            continue

        sample = list()
        for cell_index in range(1, len(row)-3, 7):
            feature_set = row[cell_index+1:cell_index+7]
            if feature_set[0] == 'NaN':
                sample.append(np.zeros(6))
            else:
                sample.append(list(map(float, feature_set)))

        # sample_class = get_one_hot_rep(int(row[len(row)-1]))
        sample_class = int(row[len(row) - 1])

        X_train.append(sample)
        y_train.append(sample_class)

    if split:
        return np.array(X_train[1:int(len(X_train)*0.7)]), np.array(y_train[1:int(len(X_train)*0.7)]), \
                np.array(X_train[int(len(X_train)*0.7):]), np.array(y_train[int(len(X_train)*0.7):])

    else:
        return np.array(X_train[1:]), np.array(y_train[1:])
