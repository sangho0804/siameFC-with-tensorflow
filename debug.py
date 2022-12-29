import tensorflow as tf
import keras.backend as k

from keras import Model
from keras import Input
import backbone.convolutional_alexnet as em

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    input = np.random.randint(0, 100, size=(227, 227, 3))
    input_tensor = tf.convert_to_tensor(input)
    
    # input_tensor = k.reshape(-1, 227, 227, 3)
    model = em.convolutional_alexnet(input_tensor)
    print("done")

    
    
