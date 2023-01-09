from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np

# loss function
def loss_of_scoreMap(y_true, y_pred):
    # product = -y_true * y_pred
    # probs = 1 + K.clip(K.exp(product), 0, 1e6)
    # loss = K.log(probs + 1e-7)

    
    product = tf.matmul(-y_true, y_pred)
    probs = 1 + K.clip(tf.exp(product), 0, 1e6)
    loss = K.log(probs)
    test1 = tf.reduce_mean(loss)


    return test1


