from bright2nuc.utils import *
from bright2nuc.segmentation.Regression import Regression

def GetPipeLine(configs):
    if "seeds" in configs:
        if configs["seeds"] == True:
            import numpy as np
            np.random.seed(123)
            import random as python_random
            python_random.seed(123)
            import tensorflow as tf
            tf.random.set_random_seed(123)
            sess = tf.Session(graph=tf.get_default_graph())
            K.set_session(sess)
    configs.update({"use_algorithm": "Regression"})
    pipeline = Regression(**configs)
    return pipeline