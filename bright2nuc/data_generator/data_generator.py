'''
Author: Dominik Waibel

This file contains the functions with which the data is improted and exported before training and testing and after testing.
'''

from bright2nuc.data_generator.data import *
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from bright2nuc.data_generator.data_augmentation import data_augentation
import os
import csv as csv
import sys
import copy
import glob
import logging
from keras.utils import to_categorical
from skimage.color import gray2rgb
import warnings
import math
import random

def reshape_to_nuclei_size(data, resize_factor):
    data_shape = np.shape(data)
    return resize(data, (data_shape[0], int(data_shape[1]*resize_factor), int(data_shape[2]*resize_factor)
                         , data_shape[3]), preserve_range=True)

def pad(data_x, data_y):
    padded_x = np.zeros((np.shape(data_x)[0],384,384, np.shape(data_x)[3]))
    padded_x[0:np.shape(data_x)[0], 0:np.shape(data_x)[1], 0:np.shape(data_x)[2], 0:np.shape(data_x)[3]] = data_x
    padded_y = np.zeros((np.shape(data_y)[0],384,384, np.shape(data_y)[3]))
    padded_y[0:np.shape(data_y)[0], 0:np.shape(data_y)[1], 0:np.shape(data_y)[2], 0:np.shape(data_y)[3]] = data_y
    return padded_x, padded_y

def crop_pad(data_x, data_y):
    if np.shape(data_x)[2] > 384:#512:
        x = random.randint(0, data_x.shape[2] - 384)
        data_x = data_x[:, :, x:x + 384, :]
        data_y = data_y[:, :, x:x + 384, :]
    if np.shape(data_x)[1] > 384:#512:
        y = random.randint(0, data_x.shape[1] - 384)
        data_x = data_x[:, y:y + 384, :, :]
        data_y = data_y[:, y:y + 384, :, :]
    if np.shape(data_x)[1] == np.shape(data_x)[2] == 384:#512:
        return data_x, data_y
    else:
        data_x, data_y = pad(data_x, data_y)
        return data_x, data_y

def pad_cut(data, Training_input_shape):
    data = data[0:Training_input_shape[0], 0:Training_input_shape[1], 0:Training_input_shape[2]]
    padded = np.zeros(Training_input_shape)
    padded[0:np.shape(data)[0], 0:np.shape(data)[1], 0:np.shape(data)[2]] = data
    return padded

def un_pad_cut(data, input_shape):
    data = data[0:input_shape[0], 0:input_shape[1], 0:input_shape[2]]
    return data

def get_min_max(data_path, folder_name, image_files):
    '''
    This function gets the minimum and maximum pixel values for the folder
    
    Args:
        data_path (str): path to project folder
        folder_name (str): one of the data folder names, e.g. image or groundtruth
        train_image_files (list): list of image files
    
    return: 
        min_value: the minimum pixel value of the dataset 
        max_value: the maximum pixel value of the dataset
    '''
    num_img = len(image_files)
    Xmin = np.empty(num_img)
    Xmax = np.empty(num_img)
    for i, img_file in enumerate(image_files):
        if img_file.endswith(".npy"):
            Xmin[i] = np.min(np.array(np.load(data_path + folder_name + img_file)))
            Xmax[i] = np.max(np.array(np.load(data_path + folder_name + img_file)))
        else:
            Xmin[i] = np.min(np.array(imread(data_path + folder_name + img_file)))
            Xmax[i] = np.max(np.array(imread(data_path + folder_name + img_file)))
    
    min_value = np.min(Xmin)
    max_value = np.max(Xmax)

    logging.info("min value of %s is %s and the max value is %s" % \
                                            (folder_name, min_value, max_value))
    return min_value, max_value

def import_image(path_name):
    '''
    This function loads the image from the specified path
    NOTE: The alpha channel is removed (if existing) for consistency

    Args:
        path_name (str): path to image file
    
    return: 
        image_data: numpy array containing the image data in at the given path. 
    '''
    if path_name.endswith('.npy'):
        image_data = np.array(np.load(path_name))
    else:
        image_data = imread(path_name)
        # If has an alpha channel, remove it for consistency
    if np.array(np.shape(image_data))[-1] == 4:
        image_data = image_data[:,:,0:3]
    return image_data.astype("float32")


def image_generator(train_image_file, folder_name, data_path, X_min, X_max):
    '''
    This function normalizes the imported images, resizes them and create batches

    Args:
        Training_Input_shape: The dimensions of one image used for training. Can be set in the config.json file
        batchsize: the batchsize used for training
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        train_image_file: the file name
        folder_name: the folder name of the file to be imported
        data_path: the project directory
        X_min: the minimum pixel value of this dataset
        X_max: the maximum pixel value of this dataset
    return: 
        X: image data with dimensions ([z-dim], x-dim, y-dim, channels)
    '''
    img_file = train_image_file
    image_data = import_image(data_path + folder_name + img_file)
    X = (image_data - X_min) / (X_max - X_min)
    if np.shape(X)[-1] not in [1,2,3]:
        X = X[..., np.newaxis]
    return X


def training_data_generator(Training_Input_shapes, batchsize, num_channels,
                            num_channels_label, train_image_files, 
                            data_gen_args,data_path, resize_factor):
    '''
    Generate the data for training and return images and groundtruth 
    for regression and segmentation

    Args
        Training_Input_shape: The dimensions of one image used for training. Can be set in the config.json file
        batchsize: the batchsize used for training
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        num_channels_label: the number of channels of one groundtruth image. Typically 1 (grayscale) or 3 (rgb)
        train_image_files: list of files in the training dataset
        data_gen_args: augmentation arguments
        data_path: path to the project directory

    return: 
        X_train: batched training data
        Y: label or ground truth
    '''
    min_img, max_img = get_min_max(data_path, "/image/", train_image_files)
    min_gt, max_gt = get_min_max(data_path, "/groundtruth/", train_image_files)
    while True:
        for i, train_image_file in enumerate(train_image_files):
            X_import = image_generator(train_image_file,
                                   "/image/", data_path, min_img, max_img)
            if len(np.shape(X_import)) == 3:
                X_import = X_import[np.newaxis,...]
            if resize_factor != 1.:
                X_import = reshape_to_nuclei_size(X_import, resize_factor)

            Y_import = image_generator(train_image_file,
                                   "/groundtruth/", data_path, min_gt, max_gt)
            if len(np.shape(Y_import)) == 3:
                Y_import = Y_import[np.newaxis, ...]
            if resize_factor != 1.:
                Y_import = reshape_to_nuclei_size(Y_import, resize_factor)
            X_import, Y_import = crop_pad(X_import, Y_import)
            X_import = np.concatenate((np.zeros((1,
                                                 np.shape(X_import)[1],
                                                 np.shape(X_import)[2],
                                                 np.shape(X_import)[3])),
                                       X_import,
                                   np.zeros((1,
                                             np.shape(X_import)[1],
                                             np.shape(X_import)[2],
                                             np.shape(X_import)[3]))), axis = 0)

            Y_import = np.concatenate((np.zeros((1,
                                                 np.shape(Y_import)[1],
                                                 np.shape(Y_import)[2],
                                                 np.shape(Y_import)[3])),
                                       Y_import,
                           np.zeros((1, np.shape(Y_import)[1],
                                     np.shape(Y_import)[2],
                                     np.shape(Y_import)[3]))), axis = 0)
            X_train = []
            Y_train = []
            for z in range(1,np.shape(X_import)[0]-1):
                X = X_import[z-1:z+2,...]
                Y = Y_import[z-1:z+2,...]
                X, Y = data_augentation(X, Y, data_gen_args, data_path + str(train_image_file)+str(z))
                X_train.append(X)
                Y_train.append(Y)
                if z % batchsize == 0:
                    X_train = np.array(X_train)
                    Y_train = np.array(Y_train)
                    #print(np.shape(X_train), np.min(X_train), np.max(X_train))
                    #print(np.shape(Y_train), np.min(Y_train), np.max(Y_train))
                    yield (X_train, Y_train)
                    X_train = []
                    Y_train = []

def testGenerator(Input_image_shape, path, num_channels, test_image_files, resize_factor):
    '''
    Generate test images for segmentation, regression and classification

    Args
        Input_image_shape: The dimensions of one image used for training. Can be set in the config.json file
        path: path to the project directory
        num_channels: the number of channels of one image. Typically 1 (grayscale) or 3 (rgb)
        test_image_files: list of filenames in the test dataset

    return: 
        X: one image on which a model prediction is executed
    '''
    test_path = path + "/test/"
    batchsize = 1
    min_img, max_img = get_min_max(test_path, "/image/", test_image_files)
    while True:
        for i, test_image_file in enumerate(test_image_files):
            X_import = image_generator(test_image_file,
                       "/image/", test_path, min_img, max_img)
            if resize_factor != 1.:
                X_import = reshape_to_nuclei_size(X_import, resize_factor)
                Input_image_shape = tuple((Input_image_shape[0], int(Input_image_shape[1]*resize_factor),
                                     int(Input_image_shape[2]*resize_factor), Input_image_shape[3]))

            if np.shape(X_import) != Input_image_shape:
                X_import = pad_cut(X_import, Input_image_shape)
            if len(np.shape(X_import)) == 3:
                X_import = X_import[np.newaxis,...]
            X_import = np.concatenate((np.zeros((1,
                                                 np.shape(X_import)[1],
                                                 np.shape(X_import)[2],
                                                 np.shape(X_import)[3])),
                                       X_import,
                                   np.zeros((1, np.shape(X_import)[1],
                                             np.shape(X_import)[2],
                                             np.shape(X_import)[3]))), axis = 0)
            for z in range(1,np.shape(X_import)[0]-1):
                X = X_import[z-1:z+2,...]
                X = X[np.newaxis, ...]
                yield X


def saveResult(path, test_image_files, results, Input_image_shape, resize_factor):
    '''
    saves the predicted segmentation or image to the Results 
                            folder in the project directory
    
    Args:
        path: path to the project directory
        test_image_files: list of filenames in the test dataset
        results: the predicted segmentation or image
        Input_image_shape: The dimensions of one image used for 
                            training. Can be set in the config.json file
    
    return: None
    '''
    #results = un_pad_cut(results, Input_image_shape)
    results_path = path + "/results/"
    results = results * 255
    os.makedirs(path, exist_ok=True)
    z_stepsize = int(np.shape(results)[0] / len(test_image_files))
    z_steps = []
    shapes = []
    for test_file in test_image_files:
        shape = np.shape(imread(path + "/test/image/" + test_file))
        z_steps.append(shape[0])
        shapes.append(shape)
    done_z = 0
    for i, z in enumerate(z_steps):
        titlenpy = (test_image_files[i] + "_predict")
        results_out = results[done_z:done_z+z, 1, ...]
        done_z = done_z + z
        if resize_factor != 1.:
            results_out = np.squeeze(reshape_to_nuclei_size(results_out, 1./resize_factor))
        results_out = np.squeeze(un_pad_cut(results_out, shapes[i]))
        plottestimage_npy(results_out.astype("uint8"), results_path, titlenpy)


def training_validation_data_split(data_path):
    '''
    Import filenames and split them into train and validation set according to the variable -validation_split = 20%
    Splits files in the train folder into a training and validation dataset and returns both lists containing the filenames
    
    Args
        
        data_path: path to the project directory
    
    return: 
        train_image_files: list of filenames of training data files
        val_image_files: list of filenames of validaton data files
    '''
    if os.path.isdir(data_path + "/image"):
        image_files = os.listdir(os.path.join(data_path + "/image"))
        lenval = int(len(image_files) * 0.2)
        validation_spilt_id = np.array(list(range(0, len(image_files), int(len(image_files) / lenval))))
        logging.info(validation_spilt_id)
        train_image_files = []
        val_image_files = []
        for i in range(0, len(image_files)):
            if i in validation_spilt_id:
                val_image_files.append(image_files[i])
            if i not in validation_spilt_id:
                train_image_files.append(image_files[i])
        train_image_files = np.random.permutation(train_image_files)
    else:
        train_image_files = []
        val_image_files = []
    logging.info("Found: %s images in training set" % len(train_image_files))
    logging.info("Found: %s images in validation set" % len(val_image_files))
    return train_image_files, val_image_files


def get_input_image_sizes(iterations_over_dataset, path):
    '''
    Get the size of the input images and check dimensions

    Args:
        path: path to project directory


    return: 
        Training_Input_shape: the shape of the training data
        num_channels: number of channels
        Input_image_shape: the shape of the input image
    '''
    Training_Input_shape_all = []
    if iterations_over_dataset != 0:
        data_path = path + '/train'
    else:
        data_path = path + '/test/'
    for img_file in os.listdir(data_path + "/image/"):
        Input_image_shape = np.array(np.shape(np.array(import_image(data_path + "/image/" + img_file))))
        logging.info("Input shape Input_image_shape %s" % Input_image_shape)
        Training_Input_shape = copy.deepcopy(Input_image_shape)
        logging.info("Input_image_shape %s" % Input_image_shape)
        if len(Input_image_shape) == 3:
            if any([int(Input_image_shape[1]) % 16 != 0, int(Input_image_shape[2]) % 16 != 0]):
                print("The Input data needs to have a multiple of 16 as pixel dimension")
                Training_Input_shape[1] = math.ceil(Training_Input_shape[1] / 16) * 16
                Training_Input_shape[2] = math.ceil(Training_Input_shape[2] / 16) * 16
        elif len(Input_image_shape) == 2:
            if any([int(Input_image_shape[0]) % 16 != 0, int(Input_image_shape[1]) % 16 != 0]):
                print("The Input data needs to have a multiple of 16 as pixel dimension")
                Training_Input_shape[0] = math.ceil(Training_Input_shape[0] / 16)*16
                Training_Input_shape[1] = math.ceil(Training_Input_shape[1] / 16)*16

        # If has an alpha channel, remove it for consistency

        if Training_Input_shape[-1] == 4:
            logging.info("Removing alpha channel")
            Training_Input_shape[-1] = 3

        if all([Input_image_shape[-1] != 1, Input_image_shape[-1] != 3]):
            logging.info("Adding an empty channel dimension to the image dimensions")
            Training_Input_shape = np.array(tuple(Training_Input_shape) + (1,))

        num_channels = Training_Input_shape[-1]
        input_size = tuple(Training_Input_shape)
        logging.info("Input size is: %s" % (input_size,))
        Training_Input_shape_all.append(tuple(Training_Input_shape))
    return Training_Input_shape_all, num_channels
