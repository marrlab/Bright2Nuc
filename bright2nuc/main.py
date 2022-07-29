'''
NucleiPredict
Written by Dominik Waibel and Ali Boushehri

In this file the functions are started to train and test the networks
'''

import os
import argparse
from bright2nuc.utils import load_json, download_weights
from bright2nuc import GetPipeLine
import logging
from keras import backend as K
import wget

def start_learning(configs):
    logging.info("Start learning")

    pipeline = GetPipeLine(configs)

    pipeline.run()
    K.clear_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        default="config.json", \
                        help='config json file address', \
                        type=str)

    args = vars(parser.parse_args())

    configs = load_json(args['config'])

    for k in configs:
        logging.info("%s : %s \n" % (k,configs[k]))

   
    '''
    Sanity checks in order to ensure all settings in config
    have been set so the programm is able to run
    '''

    if "batchsize" in configs:
        if not isinstance(configs["batchsize"], int):
            logging.warning("Batchsize has not been set. Setting batchsize = 1")
            batchsize = 1
    else:
        logging.warning("Batchsize has not been set. Setting batchsize = 1")
        configs["batchsize"] = 1

    if "iterations_over_dataset" not in configs:
        logging.warning("Epochs has not been set. Setting epochs = 500 and using early stopping")
        configs.update({'iterations_over_dataset': 0})

    if "pretrained_weights" in configs:
        if configs["pretrained_weights"] == "random" or configs["pretrained_weights"] == False:
            configs["pretrained_weights"] = None
        elif not isinstance(configs["pretrained_weights"], str):
            if not os.path.isfile((configs["pretrained_weights"])):
                if os.path.isfile(os.getcwd() + "/pretrained_model/pretrained_weights_.hdf5") == False:
                    logging.warning("Downloading pretrained model to" + str(os.getcwd()) + "/pretrained_model/")
                    url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/PtMpZy6oSagWFpX/pretrained_weights_.hdf5'
                    download_weights(url)
                configs["pretrained_weights"] = os.getcwd() + "/pretrained_model/pretrained_weights_.hdf5"
    if "pretrained_weights" not in configs:
        logging.warning("Downloading pretrained model to" + str(os.getcwd()) + "/pretrained_model/")
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/PtMpZy6oSagWFpX/'
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/MdW8rkn3W9kPHWF/'
        wget.download(url, os.getcwd() + '/pretrained_model/pretrained_weights_.hdf5')
        configs["pretrained_weights"] = os.getcwd() + "/pretrained_model/pretrained_weights_.hdf5"

    if "evaluation" not in configs:
        configs.update({'evaluation': False})

    if "nuclei_size" not in configs:
        configs.update({'nuclei_size': 30})

    start_learning(configs)