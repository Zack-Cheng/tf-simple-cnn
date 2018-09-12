import os
import requests
import gzip
import pickle
import numpy as np
from urllib.request import urlopen
from tqdm import tqdm

DATASET_PATH = './dataset/'
TRAIN_DATASET_GZIP = 'train-images-idx3-ubyte.gz'
TRAIN_LABEL_GZIP = 'train-labels-idx1-ubyte.gz'
TRAIN_DATASET_PATH = DATASET_PATH + TRAIN_DATASET_GZIP
TRAIN_LABEL_PATH = DATASET_PATH + TRAIN_LABEL_GZIP
DATASET_URL = 'http://yann.lecun.com/exdb/mnist/'
DATASET_PKL = DATASET_PATH + 'train-dataset.pkl'


def download_dataset(filename):
    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    url = DATASET_URL + filename
    file_size = int(urlopen(url).info().get('Content-Length', -1))

    dst_path = DATASET_PATH + filename
    if os.path.exists(dst_path):
        first_byte = os.path.getsize(dst_path)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return

    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
                total=file_size, initial=first_byte,
                unit='B', unit_scale=True, desc='Download {}'.format(filename)
    )
    req = requests.get(url, headers=header, stream=True)
    with(open(dst_path, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

def raw_to_data(raw_data):
    def _int(byte):
        return int.from_bytes(byte, 'big')

    data = []
    magic_num = _int(raw_data[:4])
    if magic_num == 2051:
        # image file
        img_num = _int(raw_data[4:8])
        row_num = _int(raw_data[8:12])
        col_num = _int(raw_data[12:16])

        for i in tqdm(range(img_num), desc='Loading datasets', leave=False):
            img = []
            for r in range(row_num):
                row = []
                for c in range(col_num):
                    row.append(raw_data[16 + i*row_num*col_num + r*col_num + c])
                img.append(row)
            data.append([img])

    elif magic_num == 2049:
        # label file
        img_item = _int(raw_data[4:8])
        for i in tqdm(range(img_item), desc='Loading labels', leave=False):
            data.append(raw_data[8+i])

    return data

def load_train_dataset():
    if os.path.exists(DATASET_PKL):
        with open(DATASET_PKL, 'rb') as f:
            out = pickle.load(f)
        return out

    if not os.path.exists(TRAIN_DATASET_PATH):
        download_dataset(TRAIN_DATASET_GZIP)

    if not os.path.exists(TRAIN_LABEL_PATH):
        download_dataset(TRAIN_LABEL_GZIP)

    with gzip.open(TRAIN_DATASET_PATH) as f:
        raw_dataset = f.read()

    with gzip.open(TRAIN_LABEL_PATH) as f:
        raw_label = f.read()

    dataset = np.array(raw_to_data(raw_dataset)) / 255.
    dataset = dataset.transpose(0, 2, 3, 1)
    label = np.array(raw_to_data(raw_label))

    with open(DATASET_PKL, 'wb+') as f:
        pickle.dump((dataset, label), f)

    return dataset, label

def to_hot_vector(label):
        m = label.size
        t = np.zeros((m, 10))
        t[np.arange(m), label] = 1
        return t

