import os
import gzip
import numpy as np


def _load_images(file_path):
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, 1, 28, 28)
    data = data.astype(np.float32) / 255.0

    return data


def _load_labels(file_path):
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels


def load_fashion_mnist(data_dir="./data", normalize=True, flatten=False, one_hot_label=False):
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    x_train = _load_images(train_images_path)
    t_train = _load_labels(train_labels_path)
    x_test = _load_images(test_images_path)
    t_test = _load_labels(test_labels_path)

    if not normalize:
        x_train = (x_train * 255).astype(np.uint8)
        x_test = (x_test * 255).astype(np.uint8)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)

    return (x_train, t_train), (x_test, t_test)


def _change_one_hot_label(labels):
    one_hot = np.zeros((labels.size, 10))

    for idx, row in enumerate(one_hot):
        row[labels[idx]] = 1

    return one_hot
