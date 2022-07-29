"""
NucleiPredict
Utils for data evaluation
Written by Dominik Waibel
"""

from bright2nuc.data_generator.data_generator import *
import os
import shutil

def test_write_logbook():
    os.makedirs(os.getcwd()+"/tests/data_generator/testimages/", exist_ok=True)
    write_logbook(os.getcwd()+"/tests/data_generator/testimages/", 2, "categorical_crossentropy",
        {"save_augmented_images": False,
         "resample_images": False,
         "std_normalization": True,
         "feature_scaling": False,
         "horizontal_flip": True})
    file = open(os.getcwd()+"/tests/data_generator/testimages/Logbook.txt", "r")
    file.readline()
    file.readline()
    assert str(file.readline()) == 'With lossfunction: categorical_crossentropy for : 2 epochs\n'
    assert file.readline() == "The augmentations are: {'save_augmented_images': False, 'resample_images': False, 'std_normalization': True, 'feature_scaling': False, 'horizontal_flip': True}\n"
    os.remove(os.getcwd()+"/tests/data_generator/testimages/Logbook.txt")


def test_plottestimage_npy():
    plottestimage_npy(np.ones((10,10,3)), os.getcwd()+"/tests/data_generator/testimages/", "plottestimage_test")
    assert os.path.isfile(os.getcwd()+"/tests/data_generator/testimages/plottestimage_test.tif") == True
    os.remove(os.getcwd()+"/tests/data_generator/testimages/plottestimage_test.tif")

def test_plot2images():
    plot2images(np.ones((10,10,3)), np.ones((10,10,3)), os.getcwd()+"/tests/data_generator/testimages/", "plot2images_test")
    assert os.path.isfile(os.getcwd()+"/tests/data_generator/testimages/plot2images_test.png") == True
    os.remove(os.getcwd()+"/tests/data_generator/testimages/plot2images_test.png")

    # Delete created test data
    if os.path.exists(os.getcwd()+"/tests/data_generator/testimages") and os.path.isdir(os.getcwd()+"/tests/data_generator/testimages"):
        shutil.rmtree(os.getcwd()+"/tests/data_generator/testimages")
    if os.path.exists(os.getcwd()+"/tests/data_generator/testimages_classification") and os.path.isdir(os.getcwd()+"/tests/data_generator/testimages_classification"):
        shutil.rmtree(os.getcwd()+"/tests/data_generator/testimages_classification")
