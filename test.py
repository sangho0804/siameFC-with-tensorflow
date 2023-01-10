import tensorflow as tf
import siameFC.loss_of_scoreMap as tt
import os
os.environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__=='__main__':
    score = tf.constant(1, shape=(742,17,17), dtype=tf.float32)

    corr = tf.constant(0.1, shape=(742,17,17), dtype=tf.float32)


    for i in range(1,742+1):
        
        test = tt.loss_of_scoreMap(score, corr)

        if i % 8 == 0:
            print(test)
