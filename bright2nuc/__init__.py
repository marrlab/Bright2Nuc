from bright2nuc.utils import *
from bright2nuc.segmentation.Regression import Regression

def GetPipeLine(configs, random_state=123):
    if "seeds" in configs:
        if configs["seeds"] == True:
            import numpy as np
            np.random.seed(random_state)
            import random as python_random
            python_random.seed(random_state)
            import tensorflow as tf
            tf.random.set_random_seed(random_state)
            sess = tf.Session(graph=tf.get_default_graph())
            K.set_session(sess)
    configs.update({"use_algorithm": "Regression"})
    pipeline = Regression(**configs)
    return pipeline