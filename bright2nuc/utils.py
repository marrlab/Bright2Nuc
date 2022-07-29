import sys
import logging
from urllib.request import urlopen
import os, tempfile
from tqdm import tqdm
import shutil
import argparse
import os
import json
from bright2nuc.data_generator.data_generator import *
from bright2nuc.data_generator.auto_evaluation_segmentation_regression import segmentation_regression_evaluation
from bright2nuc.segmentation.UNet_models import UNetBuilder
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import time
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
from keras import backend as K
tf.config.experimental.list_physical_devices('GPU')
import glob
from keras.optimizers import Adam, SGD
from bright2nuc.data_generator.data import write_logbook
logging.basicConfig(level=logging.INFO)

def load_json(file_path):
    with open(file_path, 'r') as stream:
        return json.load(stream)


'''adapted from: https://github.com/MouseLand/cellpose/blob/bb7be7871bbc13793ac32790987111823ad5a48e/cellpose/utils.py#L45 '''
def download_weights(url):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    print(dst)
    print(dst_dir)
    dst_dir_final = os.getcwd() + '/pretrained_model/'
    os.makedirs(dst_dir_final, exist_ok=True)

    print(dst_dir)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst_dir_final + dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)