# Python Standard Library
import glob
import os
import pickle
from urllib.request import urlretrieve
import zipfile

# Public Libraries
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Project
import config


LOAD_PNG_PATTERN = 'sign*.png'
SAVE_PNG_PATTERN = 'sign_%05d.png'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def load_data(data_set):
    # Convert pickled data to human readable images
    image_files = os.path.join(config.IMAGES_DIR, data_set, LOAD_PNG_PATTERN)
    if len(glob.glob(image_files)) == 0:
        extract_images()

    # Sort file names in alphabetical order to line up with labels
    files = glob.glob(image_files)
    files.sort()

    # Load images and save in X matrix. Convert to numpy array.
    x = []
    for file in files:
        img = Image.open(file)
        x.append(np.asarray(img.copy()))
        img.close()
    x = np.array(x)

    # Load labels
    labels_file = os.path.join(config.LABELS_DIR, '%s.csv' % data_set)
    y = pd.read_csv(labels_file, header=None).values

    # Return images and labels
    return x, y


def maybe_download_traffic_signs():

    data_file = os.path.join(config.DATA_DIR, config.DATA_FILE)

    if not os.path.exists(data_file):
        if not os.path.exists(config.DATA_DIR):
            os.makedirs(config.DATA_DIR)

        # Download Traffic Sign data
        print('Downloading Traffic Sign data...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                config.DATA_URL,
                data_file,
                pbar.hook)

    # Extract
    print('Extracting Traffic Sign data...')
    zip_ref = zipfile.ZipFile(data_file, 'r')
    zip_ref.extractall(config.DATA_DIR)
    zip_ref.close()


def extract_images():

    # Download data
    maybe_download_traffic_signs()

    for data_set in config.DATA_SETS:
        # Load Data
        with open(os.path.join(config.DATA_DIR, '%s.p' % data_set), 'rb') as f:
            data = pickle.load(f)
        x = data['features']
        y = data['labels']

        # Save to CSV. No label for columns or rows
        y = pd.DataFrame(y)
        labels_dir = config.LABELS_DIR
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)
        labels_file = os.path.join(labels_dir, '%s.csv' % data_set)
        y.to_csv(labels_file, header=False, index=False)

        # Create image directory
        directory = os.path.join(config.IMAGES_DIR, '%s' % data_set)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Load images and save as picture files
        num_images = x.shape[0]
        for i in range(num_images):
            file = os.path.join(directory, SAVE_PNG_PATTERN % i)
            img = x[i]
            img = Image.fromarray(img)
            img.save(file)
