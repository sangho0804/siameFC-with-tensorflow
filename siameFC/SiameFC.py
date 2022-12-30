import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Lambda,
)
import backbone.convolutional_alexnet as alexnet

def x_corr_map(x):
    exemplar = x[1]
    search = x[0]

    output = tf.nn.conv2d(search, exemplar, strides=1, padding='VALID')
    
    return output

def x_corr_layer():
    return Lambda(x_corr_map, output_shape=(17, 17))
    

def siameFc_model(x_shape, z_shape):

    exemplar = Input(shape=(z_shape))
    search = Input(shape=(x_shape))
    score_map = Input(shape=(17,17))

    embedding_exemplar = alexnet.convolutional_alexnet(exemplar)
    embedding_search = alexnet.convolutional_alexnet(search)

    score_map = x_corr_layer()([embedding_search, embedding_exemplar])
    
    outputs = [score_map]
    inputs = [search, exemplar]


    model = Model(inputs=inputs, outputs=outputs)

    return model


    










