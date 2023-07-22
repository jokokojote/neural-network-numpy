"""
Utils for data download, loading and batching
"""
import numpy as np
import requests
import gzip
import os

def download_file(url, filename):
    response = requests.get(url)
    open(filename, 'wb').write(response.content)

def load_mnist(kind='train'):
    """Load MNIST data from the web."""
    labels_path = f'{kind}-labels-idx1-ubyte.gz'
    images_path = f'{kind}-images-idx3-ubyte.gz'

    if not os.path.exists(labels_path):
        download_file(f'http://yann.lecun.com/exdb/mnist/{labels_path}', labels_path)

    if not os.path.exists(images_path):
        download_file(f'http://yann.lecun.com/exdb/mnist/{images_path}', images_path)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def get_batches(X, y, batch_size):
    """Generator function that yields batches of data."""
    num_batches = (len(X) + batch_size - 1) // batch_size
    for i in range(num_batches):
        yield X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]