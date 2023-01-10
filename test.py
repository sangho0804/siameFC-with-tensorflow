from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from keras import losses

if __name__=='__main__':
    # score = tf.constant(1, shape=(742,17,17), dtype=tf.float32)

    # corr = tf.constant(0.1, shape=(742,17,17), dtype=tf.float32)




    # for i in range(1,742+1):
        
    #     test = tt.loss_of_scoreMap(score, corr)

    #     if i % 8 == 0:
    #         print(test)
    test1 = tf.constant(2, shape=(3,3), dtype=tf.float32)
    print(test1)
    test2 = tf.constant([[-1, 0 ,-1],[-1,5,0],[-1,-1,-1]],  dtype=tf.float32)
    print(test2)
    print(tf.reduce_max(test2))
    test = tf.where(test2 == tf.reduce_max(test2))
    test4 = tf.constant([[3, 6]], dtype=tf.float32)

    mse = tf.keras.losses.MeanSquaredError()
    print(mse(test, test4).numpy())

    # zeros = array_ops.zeros_like(test2)
    # print(zeros)

    # cond = (test2 >= zeros)
    # print(cond)
    # neg_abs_logits = array_ops.where(cond, -test1, test1)
    # print(neg_abs_logits)
    # test = math_ops.log1p(math_ops.exp(neg_abs_logits))
    # print(test)

    # loss = tf.reduce_mean(test)
    # print(loss)
